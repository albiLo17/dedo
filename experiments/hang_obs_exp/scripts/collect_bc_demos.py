"""
Collect scripted-controller demos with state + RGB + point-cloud obs
simultaneously, for cross-modality behavior-cloning experiments.

Each rollout uses the same hole-aware waypoint controller as
train_privileged.py's BC dataset, but here we record ALL three modalities
per timestep so a single dataset can train any of:

  state  — privileged 18-dim hole_centroid (also includes the larger
           privileged modes: hole_centroid_corners, hole_vertices,
           full_mesh — pick any at training time)
  rgb    — (H, W, 3) uint8 image rendered from a fixed camera viewpoint
  pcd    — (N, 3) float32 point cloud captured by back-projecting the
           camera depth buffer into world coordinates

Saves to <demos_dir>/demo_NNN.pkl. Pkl schema (one episode per file):
  {
    'obs': {
       'hole_centroid':         (T, 18)  float32,
       'hole_centroid_corners': (T, 30)  float32,
       'hole_vertices':         (T, 132) float32,
       'full_mesh':             (T, 762) float32,
       'rgb':                   (T, H, W, 3) uint8,
       'pcd':                   (T, N, 3) float32   # WORLD coordinates (m)
    },
    'acts':       (T, 6) float32  # in [-1, 1] (already normalized by MAX_ACT_VEL)
    'rewards':    (T,)   float32
    'reward':     float,            # episode sum
    'success_hanging' / 'success_topological' / 'success_legacy': int 0/1
    'success':    int 0/1           # = success_hanging (the default metric)
    'success_factor': float,        # criterion used (saved for downstream filtering)
    'success_metric': 'hanging',
    'recorded_in': 'scripted',
    'len':        int,
    'cam_resolution': int,
    'pcd_n_points':   int,
    'cam_viewmat':    list[float] (6),
    'hole_radius':    float,
    'max_act_vel':    float,
    'ctrl_freq':      float,   # actual achieved Hz (sim_freq / steps_per_action)
    'sim_freq':       int,
    'sim_steps_per_action': int,
  }

Compatible reader: experiments/hang_obs_exp/scripts/train_diffusion_bc.py.

Usage (from repo root):
  python experiments/hang_obs_exp/scripts/collect_bc_demos.py \
      --demos_dir logs/hang_obs_exp/bc_demos_v1 \
      --n_demos 100 --only_success \
      --cam_resolution 96 --pcd_n_points 512
"""
import argparse
import os
import pickle
import sys
import time
from copy import deepcopy
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT))

import gym
import numpy as np
import pybullet

import dedo  # noqa: F401  (registers gym envs)
from dedo.envs.deform_env import DeformEnv
from dedo.utils.args import get_args_parser, args_postprocess
from dedo.utils.mesh_utils import get_mesh_data
from dedo.demo_preset import build_traj, merge_traj

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _helpers import RetryResetEnv, build_hole_aware_waypoints  # noqa: E402
from _bc_obs_helpers import (  # noqa: E402
    capture_rgb_depth, depth_to_pcd, cloth_only_pcd,
    get_hole_indices, get_hole_loops, measure_hole_radius,
    check_hanging_on_peg, check_threaded_topological, check_legacy,
    resolve_deform)
from _debug_viz import (  # noqa: E402
    hole_centroid_world, overlay_pcd_on_rgb, pcd_camera_view_image,
    render_sim_with_centroid, build_video_frame, write_video_mp4,
    save_grid_png, save_actions_plot, summarize_action_stream)

from experiments.hang_obs_exp.envs.privileged_env import (  # noqa: E402
    build_privileged_obs, identify_cloth_corners)


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
PRIV_MODES = ('hole_centroid', 'hole_centroid_corners',
              'hole_vertices', 'full_mesh')


# ---------------------------------------------------------------------------
# Argparse
# ---------------------------------------------------------------------------
parser = argparse.ArgumentParser()
parser.add_argument('--demos_dir', type=str, required=True,
                    help='Output directory. Created if missing. '
                         'Existing demo_NNN.pkl files are kept; new demos '
                         'continue the numbering.')
parser.add_argument('--n_demos', type=int, default=100,
                    help='Target NUMBER of demos to KEEP. Build_traj '
                         'failures and (with --only_success) failed '
                         'rollouts do NOT count toward the target — '
                         'they trigger a retry — so dataset size is '
                         'deterministic regardless of scripted controller '
                         'success rate.')
parser.add_argument('--only_success', action='store_true', default=True,
                    help='Default. Keep only demos that pass the chosen '
                         'success metric. Strongly recommended; the '
                         'hole-aware scripted controller misses ~20-30%% '
                         'of the time and those failed demos make BC '
                         'datasets dirty. Pass --no_only_success to keep '
                         'everything.')
parser.add_argument('--no_only_success', dest='only_success',
                    action='store_false',
                    help='Keep all rollouts including failures.')
parser.add_argument('--success_metric', type=str, default='hanging',
                    choices=['hanging', 'topological', 'legacy'],
                    help='Which criterion --only_success filters by. '
                         'All three are computed and saved in every pkl '
                         'regardless, so a downstream filter can use any.')
parser.add_argument('--success_factor', type=float, default=1.2,
                    help='Adaptive success threshold = success_factor * '
                         'hole_radius. 1.2 is the train_privileged.py '
                         'default. Saved into the pkl so the training '
                         'script can flag mismatches.')
parser.add_argument('--seed', type=int, default=2026)
parser.add_argument('--max_episode_len', type=int, default=200)
parser.add_argument('--cam_resolution', type=int, default=96,
                    help='RGB + depth image height/width (square). 96 '
                         'matches the pusht diffusion-policy demo and is '
                         'cheap to store (~28 KB/frame); 64 cuts storage '
                         'further but loses cloth detail; 128 doubles '
                         'storage. Per-demo size scales linearly with H*W.')
parser.add_argument('--pcd_n_points', type=int, default=512,
                    help='Number of points sampled from the back-projected '
                         'depth buffer. PointNet++ default is 512.')
parser.add_argument('--max_act_vel', type=float, default=10.0,
                    help='DeformEnv.MAX_ACT_VEL. Demo actions are stored '
                         'as clip(traj / MAX_ACT_VEL, -1, 1). Default 10 '
                         'matches dedo. Lower values give better BC signal '
                         '(action range better utilized) but will saturate '
                         'and break demos if set below the trajectory peak '
                         '(~1.5-3 m/s in the lift phase).')
parser.add_argument('--cam_viewmat', type=float, nargs=6,
                    default=[14.0, -5.0, 45.0, 0.0, 0.0, 5.5],
                    help='dedo cam_viewmat: dist pitch yaw tx ty tz. '
                         'Default zoomed-out, low-pitch diagonal so cloth '
                         'stays in frame across all trajectory phases AND '
                         'PCD pixel density stays roughly constant (~1000 '
                         'valid px per frame at cam_resolution=128) — '
                         'validated against _diag_pcd_framing.py. The '
                         'lower pitch keeps the cloth oriented more '
                         'face-on to the camera, so the hole stays '
                         'visible at most timesteps. Same view is used '
                         'for both RGB and PCD obs, so the two '
                         'modalities receive equivalent info.')
parser.add_argument('--ctrl_freq', type=float, default=15.0,
                    help='Control frequency in Hz (one env.step every '
                         '1/ctrl_freq seconds of sim time). Implemented '
                         'by setting sim_steps_per_action = round(sim_freq '
                         '/ ctrl_freq). 15 Hz is the default — slower than '
                         'dedo\'s 62.5 Hz default so each recorded action '
                         'spans more sim time and the resulting trajectory '
                         'has fewer / coarser steps (better fit for '
                         'diffusion policy receding-horizon control). The '
                         'actual achieved freq is saved into each demo pkl '
                         'as `ctrl_freq`, so a downstream training script '
                         'can verify parity. Pass --sim_freq to change the '
                         'physics step rate (default 500 Hz); 500/15=33.33, '
                         'so the rounded sim_steps_per_action=33 yields an '
                         'actual ctrl_freq of ~15.15 Hz.')
parser.add_argument('--sim_freq', type=int, default=500,
                    help='PyBullet physics frequency. Default 500 matches '
                         'dedo. Adjust only if ctrl_freq doesn\'t round '
                         'cleanly — e.g. sim_freq=450 with ctrl_freq=15 '
                         'gives sim_steps_per_action=30 (exact 15 Hz). '
                         'Lower sim_freq risks soft-body instability.')
parser.add_argument('--debug_viz_first_n', type=int, default=3,
                    help='Generate debug visualization artifacts for the '
                         'first N KEPT demos (PNG grid + MP4 video + action '
                         'stats plot). Saves under <demos_dir>/debug_viz/. '
                         '0 disables. Cost: ~5-10 s overhead per debug '
                         'demo (high-res sim render + matplotlib).')
parser.add_argument('--debug_viz_every', type=int, default=0,
                    help='Also generate debug viz for every Nth kept demo '
                         '(in addition to --debug_viz_first_n). 0 disables. '
                         'Useful with large n_demos to spot-check '
                         'mid-collection: --debug_viz_every 25 -> ~6 viz '
                         'demos for a 150-demo run.')
parser.add_argument('--debug_render_size', type=int, default=300,
                    help='Per-panel size (H=W) for the video frames\' '
                         'high-res sim panel. 300 keeps the mp4 small and '
                         'each panel readable; bump to 480 for slides.')
parser.add_argument('--debug_fps', type=int, default=15,
                    help='Video frame rate. Matches the default ctrl_freq '
                         'so playback runs at sim wall-clock speed. Bump '
                         'to 30 for smoother scrubbing.')
parser.add_argument('--debug_n_grid_samples', type=int, default=5,
                    help='Number of in-trajectory timesteps sampled for '
                         'the PNG grid (post-settle frame is added as an '
                         'extra row). 5 + 1 = 6 rows is a comfortable '
                         'PNG height.')
parser.add_argument('--episode_tail_frames', type=int, default=5,
                    help='Number of zero-velocity hold frames appended '
                         'after the planned trajectory ends, before '
                         '`done` fires and the gravity settle runs. '
                         'Each frame gives the PD controller a chance '
                         'to brake the anchor against the cloth\'s '
                         'momentum, so make_final_steps starts from a '
                         'near-stationary pose. 5 frames is ~0.33 s at '
                         '15 Hz / ~0.08 s at 62.5 Hz. 0 disables (drops '
                         'directly into gravity settle). The dedo '
                         'make_final_steps phase always runs after, '
                         'regardless. With this knob set, demo length '
                         '= len(traj) + episode_tail_frames (capped at '
                         '--max_episode_len for safety), so episode '
                         'duration scales naturally with ctrl_freq '
                         'instead of being a fixed step count.')
extra = parser.parse_args()


# Derive sim_steps_per_action from ctrl_freq. Round to nearest int and
# report the actual achieved freq so the user knows what gets saved.
_steps_per_action = max(1, int(round(extra.sim_freq / extra.ctrl_freq)))
_actual_ctrl_freq = extra.sim_freq / _steps_per_action
if abs(_actual_ctrl_freq - extra.ctrl_freq) / extra.ctrl_freq > 0.05:
    print(f'[init] WARN: requested ctrl_freq={extra.ctrl_freq} Hz rounds '
          f'to sim_steps_per_action={_steps_per_action} -> actual '
          f'ctrl_freq={_actual_ctrl_freq:.3f} Hz (>5% deviation). Pick a '
          f'--sim_freq that divides ctrl_freq more cleanly to fix.')
else:
    print(f'[init] ctrl_freq={extra.ctrl_freq} Hz -> '
          f'sim_steps_per_action={_steps_per_action} '
          f'(actual {_actual_ctrl_freq:.3f} Hz, sim_freq={extra.sim_freq})')

os.makedirs(extra.demos_dir, exist_ok=True)

# ---------------------------------------------------------------------------
# Build dedo args + env
# ---------------------------------------------------------------------------
sys.argv = [
    'collect_bc_demos',
    '--env=HangProcCloth-v1',
    f'--cam_resolution={extra.cam_resolution}',  # enables camera at all
    '--num_envs=0',
    '--total_env_steps=0',
    '--seed', str(extra.seed),
    '--max_episode_len', str(extra.max_episode_len),
    f'--sim_freq={extra.sim_freq}',
    f'--sim_steps_per_action={_steps_per_action}',
    '--cam_viewmat',
    *[str(x) for x in extra.cam_viewmat],
]
args, _ = get_args_parser()
args_postprocess(args)
args.debug = False
args.viz = False
args.uint8_pixels = True  # we manage RGB ourselves; this is a defensive default

# Apply MAX_ACT_VEL globally (action normalization).
_orig_mav = DeformEnv.MAX_ACT_VEL
DeformEnv.MAX_ACT_VEL = float(extra.max_act_vel)
if extra.max_act_vel != _orig_mav:
    print(f'[init] DeformEnv.MAX_ACT_VEL: {_orig_mav} -> '
          f'{DeformEnv.MAX_ACT_VEL}')

env = gym.make(args.env, args=args)
env = RetryResetEnv(env)
env.seed(extra.seed)
deform = resolve_deform(env)

np.random.seed(extra.seed)

ctrl_freq = args.sim_freq / args.sim_steps_per_action


# ---------------------------------------------------------------------------
# Demo numbering: continue past existing demos in the dir.
# ---------------------------------------------------------------------------
def _next_demo_id():
    existing = [p for p in os.listdir(extra.demos_dir)
                if p.startswith('demo_') and p.endswith('.pkl')]
    nums = []
    for p in existing:
        try:
            nums.append(int(p[len('demo_'):-len('.pkl')]))
        except ValueError:
            pass
    return max(nums) + 1 if nums else 0


# ---------------------------------------------------------------------------
# Debug visualization scheduling. We pre-compute which kept demo indices
# (0-based, counting from this run) get full viz: PNG grid + MP4 video +
# action stats plot. The collector buffers high-res renders + depth +
# centroid coords only for these demos so non-debug demos stay cheap.
# ---------------------------------------------------------------------------
_debug_enabled = (extra.debug_viz_first_n > 0 or extra.debug_viz_every > 0)
_debug_dir = os.path.join(extra.demos_dir, 'debug_viz') if _debug_enabled else None
if _debug_dir is not None:
    os.makedirs(_debug_dir, exist_ok=True)

# Monkey-patch deform.render() so the dedo-side settle-frame captures
# inside make_final_steps use the SAME projection matrix as our obs
# RGB camera (fov=60, near=0.1, far=30). Without this, settle frames
# look zoomed out (~90° fov from DEFAULT_CAM_PROJECTION) relative to
# the policy-phase sim panel in the debug video. View matrix is
# already identical (both use deform._cam_viewmat), so this only
# affects the projection — no camera-pose shift.
if _debug_enabled:
    import types as _types
    from _bc_obs_helpers import proj_matrix as _bc_proj_matrix
    _MATCHED_PROJ = _bc_proj_matrix()

    def _matched_render(self, mode='rgb_array', width=300, height=300):
        assert mode == 'rgb_array'
        _, _, rgba, _, _ = self.sim.getCameraImage(
            width=width, height=height,
            renderer=pybullet.ER_BULLET_HARDWARE_OPENGL,
            viewMatrix=self._cam_viewmat,
            projectionMatrix=_MATCHED_PROJ)
        return np.asarray(rgba)[:, :, :3]

    deform.render = _types.MethodType(_matched_render, deform)
    print('[init] patched deform.render() to use obs-camera projection '
          '(fov=60) so debug-video settle frames match policy-phase frames')


def _should_debug(kept_index_zero_based: int) -> bool:
    """`kept_index_zero_based` is "this is the Nth kept demo of this run"."""
    if kept_index_zero_based < extra.debug_viz_first_n:
        return True
    if extra.debug_viz_every > 0 and \
       (kept_index_zero_based + 1) % extra.debug_viz_every == 0:
        return True
    return False


# ---------------------------------------------------------------------------
# Main collection loop.
# ---------------------------------------------------------------------------
print(f'\n=== BC demo collection ===')
print(f'  demos_dir:       {extra.demos_dir}')
print(f'  target n_demos:  {extra.n_demos}')
print(f'  only_success:    {extra.only_success}  ({extra.success_metric})')
print(f'  cam_resolution:  {extra.cam_resolution}')
print(f'  pcd_n_points:    {extra.pcd_n_points}')
print(f'  MAX_ACT_VEL:     {DeformEnv.MAX_ACT_VEL}')
print(f'  ctrl_freq:       {_actual_ctrl_freq:.3f} Hz '
      f'(sim_freq={extra.sim_freq}, steps/action={_steps_per_action})')
if _debug_enabled:
    print(f'  debug_viz:       first {extra.debug_viz_first_n} kept'
          f'{f" + every {extra.debug_viz_every}th" if extra.debug_viz_every > 0 else ""}'
          f' -> {_debug_dir}')
else:
    print(f'  debug_viz:       disabled')
print(f'  starting demo_id at: {_next_demo_id()}\n')

# Action-stat aggregation across all kept demos (printed at end of run).
_action_stats_kept: list = []

n_kept = 0
n_dropped_failed = 0
attempts = 0
max_attempts = max(extra.n_demos * 5, 30)
start_time = time.time()

while n_kept < extra.n_demos and attempts < max_attempts:
    attempts += 1
    env.reset()
    hole_idx = get_hole_indices(deform)
    if not hole_idx:
        print(f'[demo] attempt {attempts}: no hole loop on cloth, retrying')
        continue
    hole_loops = get_hole_loops(deform)
    _, verts0 = get_mesh_data(deform.sim, deform.deform_id)
    corner_idx = identify_cloth_corners(verts0)
    hole_radius = measure_hole_radius(deform, hole_idx)

    wp = build_hole_aware_waypoints(deform)
    if wp is None:
        print(f'[demo] attempt {attempts}: build waypoints failed, retrying')
        continue
    try:
        _, va = build_traj(deform, wp, 'a', anchor_idx=0,
                           ctrl_freq=ctrl_freq, robot=None)
        _, vb = build_traj(deform, wp, 'b', anchor_idx=1,
                           ctrl_freq=ctrl_freq, robot=None)
        traj = merge_traj(va, vb)
    except Exception as e:
        print(f'[demo] attempt {attempts}: build_traj failed ({e!r}), retrying')
        continue

    # Defensive check: trajectory peak must not exceed MAX_ACT_VEL or
    # `clip(traj / MAX_ACT_VEL, -1, 1)` silently saturates and the
    # scripted controller can't keep up with its own plan.
    if attempts == 1:
        peak = float(np.abs(traj).max())
        flag = ' <- TOO LOW, demos will saturate' \
            if peak > DeformEnv.MAX_ACT_VEL else ''
        print(f'[demo] traj peak |vel| = {peak:.3f} m/s; '
              f'MAX_ACT_VEL = {DeformEnv.MAX_ACT_VEL:.3f} m/s{flag}')

    # Bound the per-episode max_episode_len to the trajectory length plus
    # a small zero-velocity tail. This lets us run at low ctrl_freq
    # without recording 75+ frames of "spare time" filler. The user's
    # --max_episode_len is the SAFETY UPPER BOUND; we don't exceed it.
    # The dedo env reads stepnum >= max_episode_len each env.step(), so
    # mutating after every reset() is safe.
    _eff_max_ep_len = min(int(len(traj)) + int(extra.episode_tail_frames),
                          int(extra.max_episode_len))
    deform.max_episode_len = _eff_max_ep_len
    if attempts == 1:
        print(f'[demo] per-episode max_episode_len = {_eff_max_ep_len} '
              f'(traj_len={len(traj)} + tail={extra.episode_tail_frames}, '
              f'safety cap={extra.max_episode_len})')

    # Per-episode buffers.
    ep_state = {m: [] for m in PRIV_MODES}
    ep_rgb = []
    ep_pcd = []
    ep_grip = []  # 12-dim gripper_pos_vel, /WBOX-normalized (matches PixelObsWrapper / PointCloudObsWrapper convention so RGB & PCD BC see the same proprioception the PPO baselines see).
    ep_goal = []  # 3-dim hanger goal position, /WBOX-normalized (matches privileged obs). Saved per-step so RGB and PCD encoders can take it as auxiliary conditioning — gives all 3 modalities equivalent goal info, isolating the privileged advantage to the hole-centroid extraction only.
    ep_act = []
    ep_rwd = []
    # Decide upfront whether this attempt is a debug-instrumented demo.
    # If this attempt becomes the next kept demo, it would be kept-index
    # `n_kept` (zero-based). Build the extra buffers eagerly so we don't
    # need to re-run the episode after the success check.
    _is_debug_attempt = _debug_enabled and _should_debug(n_kept)
    ep_debug_depth = [] if _is_debug_attempt else None
    ep_debug_centroid = [] if _is_debug_attempt else None
    ep_debug_video_frames = [] if _is_debug_attempt else None
    # Tell the dedo env to capture RGB frames inside make_final_steps
    # so the debug video shows the gravity settle (where the cloth
    # actually drapes onto the peg). We reuse the dedo-side hook the
    # diffusion eval script also uses, then re-stitch the captured
    # settle frames into 3-panel video frames after env.step() returns.
    if _is_debug_attempt:
        deform._record_settle_frames = True
        deform._settle_render_kwargs = dict(
            width=extra.debug_render_size,
            height=extra.debug_render_size)
        # 1 sub-step = sim_steps_per_action sim ticks. stride=2 keeps the
        # settle clip short (~7-8 frames at 15 Hz, ~15 at 62.5 Hz) without
        # losing the drape dynamics.
        deform._settle_frame_stride = 2
    else:
        # Defensive — make sure stale state from a prior debug demo
        # doesn't bleed in.
        deform._record_settle_frames = False
    last_action = np.zeros_like(traj[0])
    step = 0
    done = False
    info = {}
    traj_len_for_demo = int(len(traj))  # snapshot before stepping
    # Panels from the final policy-phase step, reused as the static
    # right-side panels under the settle-phase sim frames so the viewer
    # sees the cloth drop in the LEFT panel while the obs-PCD context
    # stays frozen at the moment make_final_steps started.
    _last_obs_overlay = None
    _last_pcd_camera = None

    while not done:
        # 1) Capture obs at current state (BEFORE step).
        for m in PRIV_MODES:
            ep_state[m].append(build_privileged_obs(
                deform, m, hole_idx, corner_indices=corner_idx))
        # Grip captured separately so RGB and PCD obs modes can use it as
        # proprioception (the privileged state modes already include it
        # inline, but RGB/PCD need it concatenated at the encoder).
        grip = np.asarray(deform.get_grip_obs(), dtype=np.float32)
        grip = np.clip(grip / 20.0, -2.0, 2.0)  # 20.0 = DeformEnv.WORKSPACE_BOX_SIZE
        ep_grip.append(grip)
        # Hanger goal world-pos, /WBOX-normalized. Matches the privileged
        # obs's last 3 dims so RGB/PCD encoders that concat this to their
        # grip projection see the same goal vector the privileged state
        # vector embeds — fair-comparison invariant.
        goal_w = np.asarray(deform.goal_pos[0], dtype=np.float32) / 20.0
        ep_goal.append(goal_w)
        rgb, depth, seg, view, proj = capture_rgb_depth(
            deform, extra.cam_resolution, extra.cam_resolution)
        # Cloth-only PCD: filter pixels by pybullet segmentation mask so
        # the encoder spends 100% of its 2048-point budget on the cloth
        # surface instead of ~30-40% on the static peg/pole/flag/base.
        # The peg's geometry is constant across episodes, so removing it
        # only loses redundant info while concentrating point density on
        # the deformable surface the policy needs to reason about.
        pcd_world = cloth_only_pcd(
            depth, seg, view, proj, deform.deform_id, extra.pcd_n_points)
        ep_rgb.append(rgb)
        ep_pcd.append(pcd_world)

        # Debug-only: depth (for PNG grid), centroid world pos, and one
        # composed video frame: sim_high_res | obs+PCD overlay | PCD
        # projected through the SAME camera as the obs (= what the
        # policy's PCD encoder sees, in screen space).
        if _is_debug_attempt:
            ep_debug_depth.append(depth.copy())
            centroid_w = hole_centroid_world(deform, hole_idx)
            ep_debug_centroid.append(centroid_w)
            sim_panel = render_sim_with_centroid(
                deform, view, proj, centroid_w,
                size=extra.debug_render_size)
            obs_overlay = overlay_pcd_on_rgb(rgb, pcd_world, view, proj)
            pcd_cam = pcd_camera_view_image(
                pcd_world, view, proj,
                size=extra.debug_render_size,
                colormap_by='depth')
            ep_debug_video_frames.append(build_video_frame(
                sim_panel, obs_overlay, pcd_cam,
                size=extra.debug_render_size))
            _last_obs_overlay = obs_overlay
            _last_pcd_camera = pcd_cam

        # 2) Step with normalized waypoint velocity. After the planned
        # trajectory runs out, command ZERO velocity for the remaining
        # hold frames. The original `last_action` (final trajectory
        # velocity) is preserved at high ctrl_freq by an artifact of
        # `build_traj`'s chunking (the truncated last chunk gives a
        # near-zero velocity at e.g. 62.5 Hz), but at low ctrl_freq the
        # truncated chunk is nearly a full second of displacement and
        # the trailing velocity is large (~2 m/s). That converts the
        # hold into an aggressive PD pull-through, which works for
        # in-sim success metrics but is a sim2real anti-pattern: a real
        # robot dragging the cloth taut against the peg for 5 seconds
        # is the opposite of what we want a deployable policy to learn.
        # Zero-action hold matches the natural 62.5 Hz behavior and
        # leaves the gravity-settle phase to drape the cloth.
        if step < len(traj):
            act_unscaled = traj[step]
        else:
            act_unscaled = np.zeros_like(traj[0])
        act = np.clip(act_unscaled / DeformEnv.MAX_ACT_VEL,
                      -1.0, 1.0).astype(np.float32)
        ep_act.append(act)
        _, rwd, done, info = env.step(act)
        ep_rwd.append(float(rwd))
        last_action = act_unscaled
        step += 1

    # Debug-only: stitch the dedo-captured settle frames into the video,
    # then take one final RGB+depth+PCD capture for the PNG grid's
    # post-settle row.
    if _is_debug_attempt:
        # 1) Settle frames live in info['settle_frames'] as a list of
        # high-res RGB arrays from inside make_final_steps. Pair each
        # with the LAST policy-phase obs+PCD/PCD-camera panels so the
        # 3-panel layout stays consistent.
        settle_rgb_frames = (info.get('settle_frames', [])
                             if isinstance(info, dict) else [])
        for sf in settle_rgb_frames:
            ep_debug_video_frames.append(build_video_frame(
                sf, _last_obs_overlay, _last_pcd_camera,
                size=extra.debug_render_size))
        # 2) Post-settle still-frame for the PNG grid (uses obs camera).
        rgb_ps, depth_ps, seg_ps, view_ps, proj_ps = capture_rgb_depth(
            deform, extra.cam_resolution, extra.cam_resolution)
        pcd_ps = cloth_only_pcd(
            depth_ps, seg_ps, view_ps, proj_ps,
            deform.deform_id, extra.pcd_n_points)
        centroid_ps = hole_centroid_world(deform, hole_idx)
        _post_settle_panel = {
            'rgb': rgb_ps, 'depth': depth_ps, 'pcd': pcd_ps,
            'view': view_ps, 'proj': proj_ps,
            'centroid': centroid_ps,
        }
        # 3) Final video frame on the fully-settled pose.
        sim_panel_ps = render_sim_with_centroid(
            deform, view_ps, proj_ps, centroid_ps,
            size=extra.debug_render_size)
        obs_overlay_ps = overlay_pcd_on_rgb(rgb_ps, pcd_ps, view_ps, proj_ps)
        pcd_cam_ps = pcd_camera_view_image(
            pcd_ps, view_ps, proj_ps,
            size=extra.debug_render_size,
            colormap_by='depth')
        ep_debug_video_frames.append(build_video_frame(
            sim_panel_ps, obs_overlay_ps, pcd_cam_ps,
            size=extra.debug_render_size))
        # Reset the dedo-side flag so a subsequent non-debug attempt
        # doesn't pay the settle-frame capture cost.
        deform._record_settle_frames = False
    else:
        _post_settle_panel = None

    # 3) At terminal step, evaluate ALL THREE success metrics.
    success_hanging = check_hanging_on_peg(
        deform, hole_idx, hole_radius, extra.success_factor)
    success_topological, max_winding = check_threaded_topological(
        deform, hole_loops)
    success_legacy = check_legacy(
        deform, hole_idx, hole_radius, extra.success_factor)

    success_by_metric = {'hanging': int(success_hanging),
                         'topological': int(success_topological),
                         'legacy': int(success_legacy)}
    keep_success = success_by_metric[extra.success_metric]
    ep_reward_total = float(np.sum(ep_rwd))

    if extra.only_success and not keep_success:
        n_dropped_failed += 1
        print(f'[demo] attempt {attempts} (kept {n_kept}/{extra.n_demos})  '
              f'len={len(ep_act)}  rwd={ep_reward_total:.1f}  '
              f'h={success_hanging} t={success_topological} '
              f'l={success_legacy}  (dropped)')
        continue

    # 4) Save pkl.
    demo_id = _next_demo_id()
    out_path = os.path.join(extra.demos_dir, f'demo_{demo_id:03d}.pkl')
    payload = {
        'obs': {
            **{m: np.asarray(ep_state[m], dtype=np.float32)
               for m in PRIV_MODES},
            'rgb': np.asarray(ep_rgb, dtype=np.uint8),
            'pcd': np.asarray(ep_pcd, dtype=np.float32),
            # 12-dim gripper proprioception. Used by RGB and PCD modes as
            # an auxiliary input; the privileged state modes already have
            # it embedded in their first 12 dims so they don't need it.
            'grip': np.asarray(ep_grip, dtype=np.float32),
            # 3-dim hanger goal pose (/WBOX-normalized). Same value the
            # privileged obs embeds in its last 3 dims; saving it
            # separately so RGB/PCD encoders can take it as auxiliary
            # conditioning. Constant within an episode for HangProcCloth
            # (peg is at a fixed world pose), so for memory we still
            # save the per-step value to keep the schema parallel with
            # grip — downstream encoders just read one entry per step.
            'goal': np.asarray(ep_goal, dtype=np.float32),
        },
        'acts': np.asarray(ep_act, dtype=np.float32),
        'rewards': np.asarray(ep_rwd, dtype=np.float32),
        'reward': ep_reward_total,
        'success_hanging': int(success_hanging),
        'success_topological': int(success_topological),
        'success_legacy': int(success_legacy),
        'success': int(success_by_metric[extra.success_metric]),
        'max_winding': float(max_winding),
        'success_factor': extra.success_factor,
        'success_metric': extra.success_metric,
        'recorded_in': 'scripted',
        'len': len(ep_act),
        'cam_resolution': int(extra.cam_resolution),
        'pcd_n_points': int(extra.pcd_n_points),
        'cam_viewmat': list(extra.cam_viewmat),
        'hole_radius': float(hole_radius),
        'max_act_vel': float(DeformEnv.MAX_ACT_VEL),
        # Control-frequency parity. The trajectory was built at this
        # ctrl_freq and each row of `acts` advances 1/ctrl_freq seconds
        # of sim time. A training/eval env at a different ctrl_freq
        # would interpret the same action stream at a different speed.
        # train_diffusion_bc.py reads these to patch its eval env.
        'ctrl_freq': float(_actual_ctrl_freq),
        'sim_freq': int(extra.sim_freq),
        'sim_steps_per_action': int(_steps_per_action),
    }
    with open(out_path, 'wb') as f:
        pickle.dump(payload, f)

    # Action-distribution stats (active = scripted-trajectory frames,
    # hold = post-trajectory `last_action` frames). Tracked for every
    # kept demo so the run summary can report aggregate hold ratios.
    act_stats = summarize_action_stream(
        np.asarray(ep_act, dtype=np.float32), traj_len_for_demo)
    _action_stats_kept.append(act_stats)

    n_kept += 1
    pkl_mb = os.path.getsize(out_path) / 1e6
    print(f'[demo] attempt {attempts} (kept {n_kept}/{extra.n_demos})  '
          f'len={act_stats["ep_len"]}  '
          f'active={act_stats["n_active"]} hold={act_stats["n_hold"]} '
          f'({act_stats["hold_frac"]*100:.0f}% hold)  '
          f'rwd={ep_reward_total:.1f}  '
          f'h={success_hanging} t={success_topological} '
          f'l={success_legacy}  saved {os.path.basename(out_path)} '
          f'({pkl_mb:.1f} MB)')

    # Emit debug viz artifacts. Done after the pkl is on disk so the
    # filenames line up: demo_NNN.pkl <-> debug_viz/demo_NNN_*.{png,mp4}.
    if _is_debug_attempt and _post_settle_panel is not None:
        # PNG grid: sample N evenly-spaced timesteps from the captured
        # rgb/depth buffer, plus the post-settle panel as a final row.
        n_steps = len(ep_rgb)
        sample_ix = np.linspace(0, n_steps - 1, extra.debug_n_grid_samples,
                                dtype=np.int64).tolist()
        rows = []
        for ix in sample_ix:
            n_valid = int((ep_debug_depth[ix] < 0.999).sum())
            rows.append({
                'label': f'step {ix}/{n_steps - 1}',
                'rgb': ep_rgb[ix],
                'depth': ep_debug_depth[ix],
                'pcd': ep_pcd[ix],
                'view': deform._cam_viewmat,
                'proj': None,  # filled below — same proj for all rows
                'n_valid': n_valid,
                'in_frame': None,  # cloth-in-frame fraction not
                                   # cheaply available post-hoc; skip
            })
        # proj_matrix is identical across rows (capture_rgb_depth reuses
        # the same fov + aspect), so grab one from the helper.
        from _bc_obs_helpers import proj_matrix as _proj_matrix
        _shared_proj = _proj_matrix()
        for r in rows:
            r['proj'] = _shared_proj
        rows.append({
            'label': 'post-settle',
            'rgb': _post_settle_panel['rgb'],
            'depth': _post_settle_panel['depth'],
            'pcd': _post_settle_panel['pcd'],
            'view': _post_settle_panel['view'],
            'proj': _post_settle_panel['proj'],
            'n_valid': int((_post_settle_panel['depth'] < 0.999).sum()),
            'in_frame': None,
        })
        grid_path = os.path.join(
            _debug_dir, f'demo_{demo_id:03d}_grid.png')
        cam_str = (f'dist={extra.cam_viewmat[0]}, '
                   f'pitch={extra.cam_viewmat[1]}, '
                   f'yaw={extra.cam_viewmat[2]}, '
                   f'target=({extra.cam_viewmat[3]}, '
                   f'{extra.cam_viewmat[4]}, {extra.cam_viewmat[5]})')
        succ_str = f'h={int(success_hanging)}/t={int(success_topological)}/l={int(success_legacy)}'
        save_grid_png(
            rows, grid_path,
            title=(f'demo_{demo_id:03d}  |  cam: {cam_str}  |  '
                   f'ctrl_freq={_actual_ctrl_freq:.2f} Hz  |  '
                   f'success {succ_str}  |  '
                   f'len={act_stats["ep_len"]} active={act_stats["n_active"]} '
                   f'hold={act_stats["n_hold"]} '
                   f'({act_stats["hold_frac"]*100:.0f}%)'))

        # MP4 of the per-step combined frames.
        video_path = os.path.join(
            _debug_dir, f'demo_{demo_id:03d}_video.mp4')
        try:
            write_video_mp4(ep_debug_video_frames, video_path,
                            fps=extra.debug_fps)
        except Exception as _e:
            print(f'  [debug_viz] WARN: video write failed: {_e!r}')

        # Per-demo action stats plot (||a|| + per-component traces).
        acts_path = os.path.join(
            _debug_dir, f'demo_{demo_id:03d}_actions.png')
        save_actions_plot(
            np.asarray(ep_act, dtype=np.float32),
            traj_len_for_demo, acts_path,
            title=(f'demo_{demo_id:03d}  actions  |  '
                   f'len={act_stats["ep_len"]} '
                   f'(active={act_stats["n_active"]}, '
                   f'hold={act_stats["n_hold"]})  '
                   f'peak|a|={act_stats["peak_abs_a"]:.2f}  '
                   f'mean||a||={act_stats["mean_norm_a"]:.2f}'))
        print(f'  [debug_viz] wrote {os.path.basename(grid_path)}, '
              f'{os.path.basename(video_path)}, '
              f'{os.path.basename(acts_path)}')


env.close()
elapsed = time.time() - start_time
print(f'\nDone. kept={n_kept}, dropped={n_dropped_failed}, '
      f'attempts={attempts}, elapsed={elapsed:.0f}s '
      f'({elapsed/max(n_kept,1):.1f}s/demo)')

# Aggregate action stats across kept demos. Helpful for "is this dataset
# 50% padding or 75%?" at a glance.
if _action_stats_kept:
    _lens = np.array([s['ep_len'] for s in _action_stats_kept])
    _actives = np.array([s['n_active'] for s in _action_stats_kept])
    _holds = np.array([s['n_hold'] for s in _action_stats_kept])
    _hold_fracs = np.array([s['hold_frac'] for s in _action_stats_kept])
    _peaks = np.array([s['peak_abs_a'] for s in _action_stats_kept])
    _stationary_fracs = np.array([s['stationary_frac']
                                  for s in _action_stats_kept])
    print(f'\n=== Action-stat summary (across {len(_action_stats_kept)} '
          f'kept demos) ===')
    print(f'  episode length     : mean={_lens.mean():.1f}  '
          f'min={_lens.min()}  max={_lens.max()}')
    print(f'  active steps       : mean={_actives.mean():.1f}  '
          f'min={_actives.min()}  max={_actives.max()}')
    print(f'  hold steps         : mean={_holds.mean():.1f}  '
          f'min={_holds.min()}  max={_holds.max()}')
    print(f'  hold fraction      : mean={_hold_fracs.mean()*100:.1f}%  '
          f'min={_hold_fracs.min()*100:.1f}%  '
          f'max={_hold_fracs.max()*100:.1f}%')
    print(f'  stationary frac    : mean={_stationary_fracs.mean()*100:.1f}%  '
          f'(||a|| < 0.05 — true near-zero actions, not "hold")')
    print(f'  peak |a| / demo    : mean={_peaks.mean():.2f}  '
          f'min={_peaks.min():.2f}  max={_peaks.max():.2f}  '
          f'(1.0 = saturated)')
    if _peaks.max() >= 0.99:
        n_sat = int((_peaks >= 0.99).sum())
        print(f'  NOTE: {n_sat}/{len(_peaks)} demo(s) have peak |a| ≥ 0.99 — '
              f'actions saturated. Bump --max_act_vel to avoid.')

if n_kept < extra.n_demos:
    print(f'WARNING: target {extra.n_demos} not reached. Either increase '
          f'--n_demos, raise the attempt cap, or relax --only_success / '
          f'--success_factor.')
