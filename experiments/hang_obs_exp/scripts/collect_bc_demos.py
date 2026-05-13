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
    capture_rgb_depth, depth_to_pcd,
    get_hole_indices, get_hole_loops, measure_hole_radius,
    check_hanging_on_peg, check_threaded_topological, check_legacy,
    resolve_deform)

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
extra = parser.parse_args()

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
# Main collection loop.
# ---------------------------------------------------------------------------
print(f'\n=== BC demo collection ===')
print(f'  demos_dir:       {extra.demos_dir}')
print(f'  target n_demos:  {extra.n_demos}')
print(f'  only_success:    {extra.only_success}  ({extra.success_metric})')
print(f'  cam_resolution:  {extra.cam_resolution}')
print(f'  pcd_n_points:    {extra.pcd_n_points}')
print(f'  MAX_ACT_VEL:     {DeformEnv.MAX_ACT_VEL}')
print(f'  starting demo_id at: {_next_demo_id()}\n')

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

    # Per-episode buffers.
    ep_state = {m: [] for m in PRIV_MODES}
    ep_rgb = []
    ep_pcd = []
    ep_grip = []  # 12-dim gripper_pos_vel, /WBOX-normalized (matches PixelObsWrapper / PointCloudObsWrapper convention so RGB & PCD BC see the same proprioception the PPO baselines see).
    ep_act = []
    ep_rwd = []
    last_action = np.zeros_like(traj[0])
    step = 0
    done = False
    info = {}

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
        rgb, depth, view, proj = capture_rgb_depth(
            deform, extra.cam_resolution, extra.cam_resolution)
        ep_rgb.append(rgb)
        ep_pcd.append(depth_to_pcd(depth, view, proj, extra.pcd_n_points))

        # 2) Step with normalized waypoint velocity.
        act_unscaled = traj[step] if step < len(traj) else last_action
        act = np.clip(act_unscaled / DeformEnv.MAX_ACT_VEL,
                      -1.0, 1.0).astype(np.float32)
        ep_act.append(act)
        _, rwd, done, info = env.step(act)
        ep_rwd.append(float(rwd))
        last_action = act_unscaled
        step += 1

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
    }
    with open(out_path, 'wb') as f:
        pickle.dump(payload, f)

    n_kept += 1
    pkl_mb = os.path.getsize(out_path) / 1e6
    print(f'[demo] attempt {attempts} (kept {n_kept}/{extra.n_demos})  '
          f'len={len(ep_act)}  rwd={ep_reward_total:.1f}  '
          f'h={success_hanging} t={success_topological} '
          f'l={success_legacy}  saved {os.path.basename(out_path)} '
          f'({pkl_mb:.1f} MB)')


env.close()
elapsed = time.time() - start_time
print(f'\nDone. kept={n_kept}, dropped={n_dropped_failed}, '
      f'attempts={attempts}, elapsed={elapsed:.0f}s '
      f'({elapsed/max(n_kept,1):.1f}s/demo)')
if n_kept < extra.n_demos:
    print(f'WARNING: target {extra.n_demos} not reached. Either increase '
          f'--n_demos, raise the attempt cap, or relax --only_success / '
          f'--success_factor.')
