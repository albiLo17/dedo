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
import trimesh

import dedo  # noqa: F401  (registers gym envs)
from dedo.envs.deform_env import DeformEnv
from dedo.utils.args import get_args_parser, args_postprocess
from dedo.utils.mesh_utils import get_mesh_data
from dedo.demo_preset import build_traj, merge_traj

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _helpers import (  # noqa: E402
    SlackPhase, cloth_min_z,
    RetryResetEnv, build_hole_aware_waypoints, HoleServo, make_in_view_fn,
    anchor_positions, anchor_velocities,
    REAL_SPEED_P50, REAL_SPEED_P90, REAL_SPEED_MAX)
from _bc_obs_helpers import (  # noqa: E402
    capture_rgb_depth, depth_to_pcd, cloth_only_pcd,
    get_hole_indices, get_hole_loops, measure_hole_radius,
    hole_frame, mesh_is_sane, mesh_extent,
    check_hanging_on_peg, check_threaded_topological, check_legacy,
    resolve_deform, patch_deform_render_to_obs_camera)
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
parser.add_argument('--randomize_goal_radius', type=float, default=0.0,
                    help='Half-extent (meters) of a uniform xy box around the '
                         'nominal hanger pose. When >0, every reset() samples '
                         '(dx, dy) ~ Uniform[-r, +r]^2 and shifts the hanger + '
                         'tallrod + goal_pos by the same delta, so the policy '
                         'has to use the goal-conditioning input rather than '
                         'memorizing a fixed peg location. Default 0 keeps '
                         'the v3 fixed-goal behavior. The chosen value is '
                         'saved per pkl as `randomize_goal_radius` and the '
                         'training script enforces strict parity across the '
                         'demo dir.')
parser.add_argument('--cam_viewmat', type=float, nargs=6,
                    default=[12.9307, -23.9624, 316.4960,
                             -2.9135, 1.8052, 5.1593],
                    help='dedo cam_viewmat: dist pitch yaw tx ty tz. '
                         'DEFAULT IS THE MEASURED ZED POSE, solved from '
                         'sim2real/cam_calibration.json by '
                         'sim2real/solve_sim_camera.py (eye residual 0.04 mm, '
                         'forward 0.07 deg, up 0.00 deg). Re-run that script '
                         'and paste its output if the camera is recalibrated '
                         '-- do not hand-edit these numbers. The previous '
                         'default put the sim camera 158 mm from the real '
                         'one: the ANGLES were right but dist and target were '
                         'left at the old scene centre, and dedo orbits the '
                         'target, so position was free to be wrong. '
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
parser.add_argument('--demo_speed', type=float, default=1.0,
                    help='Multiplicative factor on the scripted-controller '
                         'commanded velocities. <1.0 records slower demos '
                         '(more realistic for a real robot); >1.0 records '
                         'faster ones. The trajectory is stretched by '
                         'round(1/demo_speed) — each waypoint repeated N '
                         'times — and the velocity magnitudes scaled by '
                         '1/N, so total displacement (and therefore task '
                         'completion) is preserved, only the commanded '
                         'magnitude changes. --max_episode_len is '
                         'automatically scaled by the same N so the longer '
                         'rollouts fit. Use 1/N values (0.5, 0.333, 0.25) '
                         'for exact slowdowns; other values are rounded '
                         'to the nearest 1/N with a warning. Saved per '
                         'pkl as `demo_speed`.')
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
parser.add_argument('--debug_viz_first_n_failed', type=int, default=3,
                    help='Write an MP4 for the first N attempts that fail '
                         'the success check (when --only_success is on, '
                         'these are dropped from the dataset). Saved as '
                         '<demos_dir>/debug_viz/failed_attempt_NNN_video.mp4. '
                         '0 disables. Useful for diagnosing why the '
                         'scripted controller misses (cloth orientation, '
                         'hole geometry, peg overshoot, etc.). Independent '
                         'of --debug_viz_first_n / --debug_viz_every which '
                         'only cover KEPT demos.')
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

# --- v5: goal-chained ("chain") episodes -----------------------------------
# A `thread` episode is v4's behaviour verbatim: one fixed peg goal, one
# open-loop 3-waypoint trajectory. A `chain` episode instead samples a
# hole-centroid goal, servos the hole to it closed-loop while varying the
# inter-anchor geometry, and resamples on arrival. That is what makes the
# recorded `goal` channel vary WITHIN an episode (so goal-conditioning has to
# be used) and what actually deforms the cloth (so the policy sees more than
# rigid translation).
parser.add_argument('--randomize_goal_dz', type=float, default=0.0,
                    help='Per-episode vertical jitter of the peg and its goal, '
                         'sim units. --randomize_goal_radius only jitters '
                         '(dx, dy), so without this every threading goal sits '
                         'at ONE height — measured: the real hanger tip was '
                         '91.7 mm from the nearest training goal, 0%% within '
                         '90 mm, purely because of height.')
parser.add_argument('--chain_fraction', type=float, default=0.5,
                    help='Fraction of episodes collected as `chain` rather '
                         'than `thread`. 0.0 reproduces v4 exactly.')
parser.add_argument('--chain_n_goals', type=int, default=4,
                    help='Goals attempted per chain episode.')
parser.add_argument('--chain_min_goals', type=int, default=2,
                    help='Goals that must actually be reached to keep the '
                         'demo. Below --chain_n_goals on purpose: dedo ends '
                         'an episode as soon as a measured anchor velocity '
                         'exceeds MAX_OBS_VEL (20 units/s), and a 1 kg cloth '
                         'swinging on 0.1 kg anchors reaches that on its own '
                         '— measured, with commands capped at 0.7. That is '
                         'an early END, not corrupt data: the frames before '
                         'it are a valid goal-reaching demo, so they are kept '
                         '(minus a short tail, see --chain_abort_trim).')
parser.add_argument('--chain_abort_trim', type=int, default=5,
                    help='Frames dropped from the end of a chain episode that '
                         'ended on an env abort. Those frames contain the '
                         'velocity spike itself — physically implausible '
                         'motion we do not want to clone.')
parser.add_argument('--chain_r_min', type=float, default=1.5,
                    help='Min goal distance from the current hole centroid, '
                         'sim units (1 unit = 45 mm; hole radius ~0.53).')
parser.add_argument('--chain_r_max', type=float, default=4.0,
                    help='Max goal distance from the current hole centroid.')
parser.add_argument('--chain_v_max', type=float, default=1.1,
                    help='Common-mode anchor speed cap, sim units/s. Sized '
                         'from the REAL teleop demos (0807_demos, 8 demos, '
                         'both arms): p50 0.60, p90 1.02, max 1.89. Default '
                         'sits just above the real p90 so collected demos '
                         'move at teleop speed. MAX_ACT_VEL (4.0) is ~4x '
                         'this and must not become the operative limit.')
parser.add_argument('--chain_v_diff_max', type=float, default=0.5,
                    help='Differential (inter-anchor) speed cap, sim units/s. '
                         'Deliberately below --chain_v_max so deformation '
                         'never dominates goal tracking.')
parser.add_argument('--chain_slew_max', type=float, default=0.30,
                    help='Max change in commanded anchor velocity per control '
                         'step. dedo drives anchors with a force-limited '
                         'velocity PD, so a step change saturates the force '
                         'and the cloth flings the anchor.')
parser.add_argument('--chain_brake_speed', type=float, default=2.5,
                    help='Measured anchor speed above which the servo '
                         'commands a decelerating target instead of chasing '
                         'the goal.')
parser.add_argument('--chain_max_episode_len', type=int, default=400,
                    help='Safety cap for chain episodes only; thread '
                         'episodes keep --max_episode_len.')
parser.add_argument('--chain_timeout_steps', type=int, default=90,
                    help='Per-goal timeout in control steps (~6 s at 15 Hz). '
                         'On timeout the goal is recorded unreached and the '
                         'next one is sampled.')
parser.add_argument('--mesh_max_growth', type=float, default=2.5,
                    help='Reject an episode whose cloth AABB diagonal exceeds '
                         'this multiple of its rest value, or that contains '
                         'non-finite / escaped vertices. The differential '
                         'anchor term can over-stretch the spring mesh, and '
                         'an exploded cloth still renders a plausible-looking '
                         'point cloud — so this is a hard drop, not a warning.')

# --- v5: per-episode camera randomization ----------------------------------
# The pcd obs is in WORLD frame, so moving the camera changes WHICH surface
# points exist (self-occlusion) and their density — it does not move the
# points. That makes this a real but second-order transfer lever; the
# first-order one is the rigid SE(3) augmentation in train_diffusion_bc.py.
# --- v5: per-episode start-pose randomization ------------------------------
# Every real demo starts from the SAME pose (sd 0.1 mm across all 8), so all
# start diversity has to come from sim. Without this the sim start is a single
# point too, and the real start sits outside the training distribution.
parser.add_argument('--start_jitter_xy', type=float, default=2.0,
                    help='Per-episode jitter of deform_init_pos in x and y, '
                         'sim units (2.0 = 90 mm).')
parser.add_argument('--start_jitter_z', type=float, default=1.0,
                    help='Per-episode jitter of deform_init_pos in z.')
parser.add_argument('--deform_init_pos', type=float, nargs=3, default=None,
                    help='NOMINAL cloth start, centre of the jitter box. '
                         'Default None keeps the scene preset. To match the '
                         'real rig use the measured anchor midpoint, '
                         '-1.70 3.20 9.62, minus the anchor-to-cloth-centre '
                         'offset for the scene.')
parser.add_argument('--cam_jitter_yaw_deg', type=float, default=8.0)
parser.add_argument('--cam_jitter_pitch_deg', type=float, default=5.0)
parser.add_argument('--cam_jitter_dist_frac', type=float, default=0.15)
parser.add_argument('--cam_jitter_target', type=float, default=0.5,
                    help='Per-axis jitter of the camera target, sim units.')
parser.add_argument('--cam_roll_deg', type=float, default=5.8429,
                    help='NOMINAL camera roll, solved alongside --cam_viewmat. '
                         'Note pybullet COUPLES roll into the orbit position '
                         '(0 -> 7 deg moves the eye 29 mm), so roll and the '
                         'other five parameters are only valid as the set '
                         'solve_sim_camera.py emits together.')
parser.add_argument('--cam_jitter_roll_deg', type=float, default=0.0,
                    help='Per-episode roll jitter. Near-no-op for the pcd '
                         'obs mode (roll rotates the sampling lattice, not '
                         'the visible surface set); set it only when '
                         'collecting for the rgb obs mode.')
# --- deformation: slack + floor contact ------------------------------------
# A cloth held taut between two anchors is a flat sheet. These put folds into
# it before the goal-chasing starts, using motions a real arm can execute.
parser.add_argument('--slack_fraction', type=float, default=0.0,
                    help='Fraction of CHAIN episodes that begin with a slack '
                         'phase: converge the grippers and lower the cloth '
                         'into floor contact, then settle. 0 disables.')
parser.add_argument('--slack_squeeze', type=float, nargs=2, default=(0.45, 0.80),
                    help='Target anchor separation as a fraction of the '
                         'cloth rest width. Both values must be <= 1.0: '
                         'driving the anchors APART past the rest width is '
                         'elastic stretch, not folding.')
parser.add_argument('--slack_floor_clear', type=float, default=0.4,
                    help='Descend until the lowest cloth vertex is this far '
                         'above the ground, in sim units.')
parser.add_argument('--slack_hold_steps', type=int, default=8)
parser.add_argument('--slack_max_steps', type=int, default=120)
parser.add_argument('--no_post', action='store_true',
                    help='Delete the tallrod support post, leaving the hanger '
                         'floating (it loads with mass 0, so it stays put). '
                         'The post is a collision body the cloth can snag on '
                         'AND it occupies a large fraction of the rgb/pcd '
                         'observation, where the real rig has no equivalent.')
# --- mesh variation --------------------------------------------------------
# node_density was pinned at 15 in deform_env for every cloth ever generated,
# so the entire dataset shared one triangulation density. Hole extents are a
# fraction of it, so randomizing the two together is what produces genuinely
# different hole shapes rather than one hole rendered at several resolutions.
parser.add_argument('--proc_cloth_wh', type=float, nargs=2, default=None,
                    help='Pin the cloth to WIDTH HEIGHT in obj units (before '
                         'deform_scale), instead of sampling. HEIGHT is the '
                         'gripper separation -- see gen_procedural_hang_cloth.')
parser.add_argument('--deform_init_ori', type=float, nargs=3, default=None,
                    help='Cloth initial orientation, xyz euler radians. Needed '
                         'to put the grasp edge on the real rig baseline; the '
                         'preset value orients it for the old scene.')
parser.add_argument('--node_density_range', type=int, nargs=2, default=None,
                    help='Per-episode mesh resolution range, e.g. 10 22. '
                         'Unset keeps the historical fixed 15.')
parser.add_argument('--proc_cloth_size_range', type=float, nargs=2, default=None,
                    help='Per-episode cloth side-length range, drawn '
                         'independently for width and height so aspect ratio '
                         'varies too. Default (0.5, 2.8); real cloth is 2.55 x '
                         '2.16 in these units.')
parser.add_argument('--proc_hole_frac_range', type=float, nargs=2, default=None,
                    help='Hole extent as a fraction of node_density. Default '
                         '(0.06, 0.30).')
parser.add_argument('--proc_cloth_shapes', type=str, nargs='+', default=None,
                    help='Outline shapes to sample from, e.g. rect taper flare '
                         'corner_cut notch round_corners. Until now EVERY '
                         'procedural cloth was an axis-aligned rectangle, so '
                         'the only silhouette variation in the dataset was its '
                         'width and height. Unset keeps rectangles only.')
parser.add_argument('--peg_nominal', type=float, nargs=3, default=None,
                    help='Peg TIP position (= goal_pos), sim units. Moves the '
                         'whole peg assembly: hanger, tallrod and goal_pos '
                         'together. Default None keeps whatever is in '
                         'task_info.SCENE_INFO. Comparing geometries used to '
                         'mean editing task_info.py between runs, which is '
                         'how the v5/v6 scenes silently diverged.')

extra = parser.parse_args()

# Apply the peg override before the env is built. SCENE_INFO is re-read on
# every reset, so mutating the module dict is what makes it stick — and it
# keeps the tip->hanger-base and tip->tallrod offsets of the preset rather
# than re-deriving them, so only the position changes.
if extra.peg_nominal is not None:
    from dedo.utils.task_info import SCENE_INFO as _SCENE
    _sc = _SCENE['hangcloth']
    _tip = np.asarray(extra.peg_nominal, dtype=float)
    # Placed ABSOLUTELY from the tip, not as a delta from whatever is in
    # SCENE_INFO. A delta silently inherits an already-wrong preset, and this
    # one is wrong: v6 lowered the hanger to the measured tip but left the
    # support rod at z=0. tallrod.urdf is 0.8 m long at globalScaling=10, so
    # its top is always base + 8.0 -- which is exactly why the original preset
    # paired hanger z=8.0 with rod z=0, putting the peg ON the post. Leaving
    # the rod grounded under a lowered hanger stands a bare post 1.79 units
    # ABOVE the goal, and the expert then threads the cloth onto a spike.
    # A lower peg is a SHORTER post; sinking the rod into the floor is how you
    # get one. On the v5 numbers these formulas reproduce the preset exactly.
    _HANGER_BELOW_TIP = 0.2   # goal_pos was hanger basePosition + 0.2
    _ROD_LEN = 8.0            # 0.8 m urdf cylinder * globalScaling 10
    _place = {}
    for _name in _sc['entities']:
        if 'hanger' in _name:
            _place[_name] = _tip - np.array([0.0, 0.0, _HANGER_BELOW_TIP])
        elif 'tallrod' in _name:
            _place[_name] = _tip - np.array(
                [0.0, 0.0, _HANGER_BELOW_TIP + _ROD_LEN])
        else:
            raise ValueError(
                f'--peg_nominal does not know where to put scene entity '
                f'{_name!r}; add a rule rather than leaving it behind.')
    for _name, _p in _place.items():
        _sc['entities'][_name]['basePosition'] = list(_p)

if extra.no_post:
    from dedo.utils.task_info import SCENE_INFO as _SCENE2
    _ents = _SCENE2['hangcloth']['entities']
    _gone = [k for k in _ents if 'tallrod' in k]
    if not _gone:
        raise ValueError('--no_post: no tallrod entity in the hangcloth scene '
                         'to remove; the scene has already been changed.')
    for _k in _gone:
        del _ents[_k]
    print(f'[init] --no_post: removed {_gone} — the hanger loads with mass 0 '
          f'so it stays where it is placed')
    _old_tip = list(_sc['goal_pos'][0])
    _sc['goal_pos'] = [list(_tip)]
    print(f'[init] --peg_nominal: peg tip {_old_tip} -> {list(_tip)}; '
          f'entities placed at ' +
          ', '.join(f'{k.split("/")[-1]}={np.round(v, 3).tolist()}'
                    for k, v in _place.items()))

if max(extra.slack_squeeze) > 1.0:
    raise ValueError(
        f'--slack_squeeze must be <= 1.0 in both entries, got '
        f'{extra.slack_squeeze}. Above 1.0 the anchors are driven APART past '
        f'the cloth rest width, which stretches the spring mesh instead of '
        f'folding it.')
if not 0.0 <= extra.slack_fraction <= 1.0:
    raise ValueError(f'--slack_fraction must be in [0, 1], '
                     f'got {extra.slack_fraction}')
if not 0.0 <= extra.chain_fraction <= 1.0:
    raise ValueError(f'--chain_fraction must be in [0, 1], '
                     f'got {extra.chain_fraction}')
if extra.chain_r_min >= extra.chain_r_max:
    raise ValueError(f'--chain_r_min ({extra.chain_r_min}) must be < '
                     f'--chain_r_max ({extra.chain_r_max})')
if extra.chain_v_max > 0.5 * extra.max_act_vel:
    print(f'[init] WARN: --chain_v_max={extra.chain_v_max} is more than half '
          f'of MAX_ACT_VEL={extra.max_act_vel}; demos will sit close to the '
          f'action-normalization ceiling.')


# Derive sim_steps_per_action from ctrl_freq. Round to nearest int and
# report the actual achieved freq so the user knows what gets saved.
_steps_per_action = max(1, int(round(extra.sim_freq / extra.ctrl_freq)))
_actual_ctrl_freq = extra.sim_freq / _steps_per_action

# Derive demo-speed trajectory stretch. We slow the demo by repeating each
# velocity waypoint N times and scaling magnitudes by 1/N — preserves total
# displacement, lowers commanded velocity by 1/N. The dedo env's
# max_episode_len safety cap is scaled by the same N so the longer rollout
# isn't truncated.
if extra.demo_speed <= 0:
    raise ValueError(f'--demo_speed must be > 0, got {extra.demo_speed}')
_demo_stretch = max(1, int(round(1.0 / extra.demo_speed)))
_actual_demo_speed = 1.0 / _demo_stretch
_effective_safety_cap = int(extra.max_episode_len * _demo_stretch)
if extra.demo_speed != 1.0:
    if abs(_actual_demo_speed - extra.demo_speed) > 1e-6:
        print(f'[init] WARN: --demo_speed={extra.demo_speed} rounds to '
              f'stretch={_demo_stretch}x -> actual speed '
              f'{_actual_demo_speed:.4f}. Use a 1/N value for exact match.')
    else:
        print(f'[init] --demo_speed={extra.demo_speed} -> trajectory '
              f'stretched {_demo_stretch}x, velocity magnitudes scaled '
              f'by {_actual_demo_speed} (safety cap raised '
              f'{extra.max_episode_len} -> {_effective_safety_cap})')
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
    '--max_episode_len', str(_effective_safety_cap),
    f'--sim_freq={extra.sim_freq}',
    f'--sim_steps_per_action={_steps_per_action}',
    '--cam_viewmat',
    *[str(x) for x in extra.cam_viewmat],
    f'--cam_roll_deg={extra.cam_roll_deg}',
    f'--randomize_goal_radius={extra.randomize_goal_radius}',
    f'--randomize_goal_dz={extra.randomize_goal_dz}',
]
# preset_override_util re-applies DEFORM_INFO on EVERY reset and only skips an
# arg whose name appears in sys.argv — so deform_init_pos must be listed here
# or the per-episode jitter below is silently overwritten at each reset.
if extra.deform_init_pos is not None or extra.start_jitter_xy > 0 \
        or extra.start_jitter_z > 0:
    sys.argv += ['--deform_init_pos', '0', '0', '0']  # values set per episode
# Same reason: preset_override_util reverts anything not named in sys.argv.
if extra.deform_init_ori is not None:
    sys.argv += ['--deform_init_ori', *[str(v) for v in extra.deform_init_ori]]
args, _ = get_args_parser()
args_postprocess(args)
args.debug = False
args.viz = False
# Mesh-variation ranges live on the dedo args namespace because that is what
# deform_env and procedural_utils are handed at reset time. None means "keep
# the historical fixed value", which both sites check for explicitly.
args.node_density_range = extra.node_density_range
args.proc_cloth_wh = extra.proc_cloth_wh
args.proc_cloth_shapes = extra.proc_cloth_shapes
args.proc_cloth_size_range = extra.proc_cloth_size_range
args.proc_hole_frac_range = extra.proc_hole_frac_range
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
# Patch deform.render() unconditionally so debug-video + settle frames
# render through the obs-camera projection (fov=60) instead of dedo's
# fov≈90 default. The view matrix already comes from args.cam_viewmat.
# Done here so anything below that calls deform.render() (debug viz,
# future hooks) is correct by default, not gated on _debug_enabled.
patch_deform_render_to_obs_camera(deform)
print('[init] patched deform.render() to use obs-camera projection '
      '(fov=60) — eval-video + settle frames match obs-camera renders')

np.random.seed(extra.seed)

ctrl_freq = args.sim_freq / args.sim_steps_per_action

# ---------------------------------------------------------------------------
# v5: nominal camera + per-episode jitter.
#
# CAM_VIEWMAT_NOMINAL is what eval must use. The per-episode sampled viewmat
# is recorded too, but purely as data: train_diffusion_bc.py does NOT raise on
# mixed cam_viewmats, it warns and picks sorted(...)[0], which would silently
# put eval at an arbitrary sampled viewpoint. The nominal field is what breaks
# that tie correctly.
# ---------------------------------------------------------------------------
CAM_VIEWMAT_NOMINAL = tuple(float(x) for x in extra.cam_viewmat)
CAM_ROLL_NOMINAL = float(extra.cam_roll_deg)
CAM_RANDOMIZATION = {
    'yaw_deg': float(extra.cam_jitter_yaw_deg),
    'pitch_deg': float(extra.cam_jitter_pitch_deg),
    'dist_frac': float(extra.cam_jitter_dist_frac),
    'target': float(extra.cam_jitter_target),
    'roll_deg': float(extra.cam_jitter_roll_deg),
}
_cam_rng = np.random.default_rng(extra.seed + 7717)
_start_rng = np.random.default_rng(extra.seed + 9091)
START_NOMINAL = (np.asarray(extra.deform_init_pos, dtype=np.float64)
                 if extra.deform_init_pos is not None
                 else np.asarray(args.deform_init_pos, dtype=np.float64))


def sample_start_pos():
    j = np.array([extra.start_jitter_xy, extra.start_jitter_xy,
                  extra.start_jitter_z])
    return (START_NOMINAL + _start_rng.uniform(-j, j)).tolist()

_chain_rng = np.random.default_rng(extra.seed + 4231)


def sample_camera():
    """(viewmat, roll_deg) jittered around the nominal, once per episode."""
    dist, pitch, yaw, tx, ty, tz = CAM_VIEWMAT_NOMINAL
    j = CAM_RANDOMIZATION
    vm = (
        dist * (1.0 + _cam_rng.uniform(-j['dist_frac'], j['dist_frac'])),
        pitch + _cam_rng.uniform(-j['pitch_deg'], j['pitch_deg']),
        yaw + _cam_rng.uniform(-j['yaw_deg'], j['yaw_deg']),
        tx + _cam_rng.uniform(-j['target'], j['target']),
        ty + _cam_rng.uniform(-j['target'], j['target']),
        tz + _cam_rng.uniform(-j['target'], j['target']),
    )
    roll = CAM_ROLL_NOMINAL + _cam_rng.uniform(-j['roll_deg'], j['roll_deg'])
    return tuple(float(x) for x in vm), float(roll)


print(f'[init] camera nominal viewmat={CAM_VIEWMAT_NOMINAL} '
      f'roll={CAM_ROLL_NOMINAL} deg; per-episode jitter={CAM_RANDOMIZATION}')
print(f'[init] chain_fraction={extra.chain_fraction} '
      f'n_goals={extra.chain_n_goals} '
      f'r=[{extra.chain_r_min}, {extra.chain_r_max}] sim units '
      f'v_max={extra.chain_v_max} (real p50/p90/max = '
      f'{REAL_SPEED_P50}/{REAL_SPEED_P90}/{REAL_SPEED_MAX} sim units/s)')


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
_debug_enabled = (extra.debug_viz_first_n > 0 or extra.debug_viz_every > 0
                  or extra.debug_viz_first_n_failed > 0)
_debug_dir = os.path.join(extra.demos_dir, 'debug_viz') if _debug_enabled else None
if _debug_dir is not None:
    os.makedirs(_debug_dir, exist_ok=True)

# Render-projection patch is applied unconditionally at env construction
# above (see patch_deform_render_to_obs_camera). Debug-video + settle
# frames in make_final_steps therefore always use the obs-camera fov=60,
# regardless of --debug_viz_first_n / --debug_viz_every settings.


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
print(f'  demo_speed:      {_actual_demo_speed}'
      f'{" (stretch=" + str(_demo_stretch) + "x)" if _demo_stretch > 1 else ""}')
print(f'  randomize_goal:  radius={extra.randomize_goal_radius} m'
      f'{" (off — fixed goal)" if extra.randomize_goal_radius <= 0 else ""}')
if _debug_enabled:
    print(f'  debug_viz:       first {extra.debug_viz_first_n} kept'
          f'{f" + every {extra.debug_viz_every}th" if extra.debug_viz_every > 0 else ""}'
          f'{f" + first {extra.debug_viz_first_n_failed} failed" if extra.debug_viz_first_n_failed > 0 else ""}'
          f' -> {_debug_dir}')
else:
    print(f'  debug_viz:       disabled')
print(f'  starting demo_id at: {_next_demo_id()}\n')

# Action-stat aggregation across all kept demos (printed at end of run).
_action_stats_kept: list = []

n_kept = 0
n_dropped_failed = 0
n_dropped_exploded = 0
n_failed_videos_written = 0
n_kept_by_kind = {'thread': 0, 'chain': 0}
_speed_samples: list = []
attempts = 0
max_attempts = max(extra.n_demos * 5, 30)
start_time = time.time()

while n_kept < extra.n_demos and attempts < max_attempts:
    attempts += 1

    # Episode kind and camera are drawn BEFORE reset so the sampled viewmat is
    # already on args when the scene loads and the first obs is captured.
    episode_kind = ('chain' if _chain_rng.random() < extra.chain_fraction
                    else 'thread')
    ep_cam_viewmat, ep_cam_roll = sample_camera()
    args.cam_viewmat = list(ep_cam_viewmat)
    args.cam_roll_deg = ep_cam_roll
    ep_start_pos = sample_start_pos()
    args.deform_init_pos = list(ep_start_pos)

    env.reset()
    hole_idx = get_hole_indices(deform)
    if not hole_idx:
        print(f'[demo] attempt {attempts}: no hole loop on cloth, retrying')
        continue
    hole_loops = get_hole_loops(deform)
    _, verts0 = get_mesh_data(deform.sim, deform.deform_id)
    corner_idx = identify_cloth_corners(verts0)
    hole_radius = measure_hole_radius(deform, hole_idx)

    # Rest size of THIS episode's procedural cloth. Reference for the
    # mesh-explosion gate and for the servo's inter-anchor targets.
    rest_extent = mesh_extent(deform)
    _a0, _b0 = anchor_positions(deform)
    cloth_width = float(np.linalg.norm(_a0 - _b0))

    servo = None
    traj = None
    if episode_kind == 'thread':
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

        # Apply demo-speed slowdown: repeat each waypoint N times and scale
        # velocity magnitudes by 1/N. Same total displacement, lower commanded
        # velocity at each step.
        if _demo_stretch > 1:
            traj = np.repeat(traj, _demo_stretch, axis=0) * _actual_demo_speed

        # Defensive check: trajectory peak must not exceed MAX_ACT_VEL or
        # `clip(traj / MAX_ACT_VEL, -1, 1)` silently saturates and the
        # scripted controller can't keep up with its own plan.
        if attempts == 1:
            peak = float(np.abs(traj).max())
            flag = ' <- TOO LOW, demos will saturate' \
                if peak > DeformEnv.MAX_ACT_VEL else ''
            print(f'[demo] traj peak |vel| = {peak:.3f} m/s; '
                  f'MAX_ACT_VEL = {DeformEnv.MAX_ACT_VEL:.3f} m/s{flag}')
            # Separately from saturation: is this replayable on the real rig?
            # The chain servo is capped at --chain_v_max by construction, but
            # thread episodes inherit whatever build_traj produces, which is
            # ~3.3 units/s — 1.7x the fastest motion ever recorded in teleop.
            if peak > REAL_SPEED_MAX:
                print(f'[demo] WARN: thread trajectories peak at {peak:.2f} '
                      f'sim units/s, above the real teleop max of '
                      f'{REAL_SPEED_MAX} (p90 {REAL_SPEED_P90}). Pass '
                      f'--demo_speed {max(0.125, 1.0 / round(peak / REAL_SPEED_P90)):.3f}'
                      f' to bring them into the real range.')

        # Bound the per-episode max_episode_len to the trajectory length plus
        # a small zero-velocity tail. This lets us run at low ctrl_freq
        # without recording 75+ frames of "spare time" filler. The user's
        # --max_episode_len is the SAFETY UPPER BOUND; we don't exceed it.
        # The dedo env reads stepnum >= max_episode_len each env.step(), so
        # mutating after every reset() is safe.
        _eff_max_ep_len = min(int(len(traj)) + int(extra.episode_tail_frames),
                              int(_effective_safety_cap))
    else:
        # Chain: no precomputed trajectory. The frustum test needs this
        # episode's actual view/proj, which come from the same capture path
        # the obs uses, so the goal envelope matches what the policy sees.
        _, _, _, _view0, _proj0 = capture_rgb_depth(
            deform, extra.cam_resolution, extra.cam_resolution)
        servo = HoleServo(
            _chain_rng, cloth_width,
            r_min=extra.chain_r_min, r_max=extra.chain_r_max,
            v_max=extra.chain_v_max, v_diff_max=extra.chain_v_diff_max,
            timeout_steps=extra.chain_timeout_steps,
            slew_max=extra.chain_slew_max,
            brake_speed=extra.chain_brake_speed,
            box=DeformEnv.WORKSPACE_BOX_SIZE,
            in_view_fn=make_in_view_fn(_view0, _proj0))
        # Slack phase runs BEFORE the first goal is sampled: it moves the
        # cloth a long way, so a goal chosen beforehand would be relative to a
        # centroid that no longer exists by the time the servo takes over.
        slack = None
        if extra.slack_fraction > 0 and \
                _chain_rng.random() < extra.slack_fraction:
            slack = SlackPhase(
                _chain_rng, cloth_width,
                squeeze_range=tuple(extra.slack_squeeze),
                floor_clear=extra.slack_floor_clear,
                v_max=extra.chain_v_max,
                hold_steps=extra.slack_hold_steps,
                max_steps=extra.slack_max_steps,
                slew_max=extra.chain_slew_max,
                brake_speed=extra.chain_brake_speed)
        c0, n0, _, _, _ = hole_frame(deform, hole_idx)
        if c0 is None:
            print(f'[demo] attempt {attempts}: no usable hole frame, retrying')
            continue
        a0, b0 = anchor_positions(deform)
        if not servo.new_goal(c0, a0, b0, n0):
            print(f'[demo] attempt {attempts}: could not sample a first goal '
                  f'(cloth near the frustum edge?), retrying')
            continue
        _eff_max_ep_len = int(extra.chain_max_episode_len)

    deform.max_episode_len = _eff_max_ep_len
    if attempts == 1:
        print(f'[demo] per-episode max_episode_len = {_eff_max_ep_len} '
              f'(kind={episode_kind}, safety cap={_effective_safety_cap})')

    # Slack-phase record, defined for every episode so the thread branch (which
    # never runs one) still writes the field.
    _slack_used, _slack_squeeze, _slack_min_z = False, None, None

    # Per-episode buffers.
    ep_state = {m: [] for m in PRIV_MODES}
    ep_rgb = []
    ep_pcd = []
    ep_grip = []  # 12-dim gripper_pos_vel, /WBOX-normalized (matches PixelObsWrapper / PointCloudObsWrapper convention so RGB & PCD BC see the same proprioception the PPO baselines see).
    ep_goal = []  # 3-dim hanger goal position, /WBOX-normalized (matches privileged obs). Saved per-step so RGB and PCD encoders can take it as auxiliary conditioning — gives all 3 modalities equivalent goal info, isolating the privileged advantage to the hole-centroid extraction only.
    ep_act = []
    ep_rwd = []
    ep_exploded = None      # set to a reason string if the mesh blows up
    # Decide upfront whether this attempt is a debug-instrumented demo.
    # If this attempt becomes the next kept demo, it would be kept-index
    # `n_kept` (zero-based). Build the extra buffers eagerly so we don't
    # need to re-run the episode after the success check. We also keep
    # recording while we still need failure videos — most attempts will
    # succeed so we may record a few attempts that ultimately don't end
    # up needing a failure video; those buffers are just discarded.
    _need_failure_video = (
        extra.debug_viz_first_n_failed > 0
        and n_failed_videos_written < extra.debug_viz_first_n_failed)
    _is_debug_attempt = _debug_enabled and (
        _should_debug(n_kept) or _need_failure_video)
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
    last_action = np.zeros(6, dtype=np.float32)
    ep_aborted = False      # env ended the episode (anchor velocity limit)
    step = 0
    done = False
    info = {}
    # Snapshot before stepping. `summarize_action_stream` uses it to split
    # "active" from post-trajectory "hold" frames; a chain episode is servoed
    # end to end and has no hold phase, so the whole stream is active and the
    # boundary is set past the safety cap.
    traj_len_for_demo = (int(len(traj)) if traj is not None
                         else int(_eff_max_ep_len))
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
        # thread: the (possibly randomized) peg pose, constant all episode.
        # chain: the servo's CURRENT goal, which changes mid-episode — that
        # variation is the whole point, since it is what forces the policy to
        # actually read the goal input instead of memorizing one target.
        if episode_kind == 'chain':
            goal_w = np.asarray(servo.goal, dtype=np.float32) / 20.0
        else:
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
        if episode_kind == 'thread':
            if step < len(traj):
                act_unscaled = traj[step]
            else:
                act_unscaled = np.zeros_like(traj[0])
        else:
            # Chain: closed-loop. Recompute the hole frame every step — the
            # open-loop planner's frozen `delta = grip - hole` goes stale as
            # the cloth deforms, which is exactly the error this avoids.
            c, n_hole, r_hole, _planarity, reliable = hole_frame(
                deform, hole_idx)
            if c is None:
                ep_exploded = 'hole frame vanished mid-episode'
                break
            a_pos, b_pos = anchor_positions(deform)
            v_a, v_b = anchor_velocities(deform)
            slack_act = None
            if slack is not None:
                # Only SUPPLIES an action — it must fall through to the shared
                # step below, or it skips the episode-abort and mesh-explosion
                # checks and the episode ends silently.
                slack_act = slack.action(a_pos, b_pos, cloth_min_z(deform),
                                         v_a, v_b)
                if slack_act is None:
                    # Finished: re-sample the goal against the cloth as it NOW
                    # is. The pre-slack goal was chosen against a centroid the
                    # cloth has since left, so drop it from the record rather
                    # than count it as a goal the episode abandoned.
                    _slack_used = True
                    _slack_squeeze = slack.squeeze
                    _slack_min_z = slack.min_z_seen
                    slack = None
                    if servo.goal_positions:
                        servo.goal_positions.pop()
                    servo.goals_attempted = max(0, servo.goals_attempted - 1)
                    if not servo.new_goal(c, a_pos, b_pos, n_hole):
                        break
            if slack_act is not None:
                act_unscaled = slack_act
            else:
                if servo.reached(c, n_hole, r_hole, reliable):
                    servo.close_segment(True)
                    if (servo.goals_reached >= extra.chain_n_goals
                            or not servo.new_goal(c, a_pos, b_pos, n_hole)):
                        break
                elif servo.timed_out():
                    servo.close_segment(False)
                    if not servo.new_goal(c, a_pos, b_pos, n_hole):
                        break
                act_unscaled = servo.action(c, a_pos, b_pos, v_a, v_b)

        act = np.clip(act_unscaled / DeformEnv.MAX_ACT_VEL,
                      -1.0, 1.0).astype(np.float32)
        ep_act.append(act)
        _, rwd, done, info = env.step(act)
        ep_rwd.append(float(rwd))
        last_action = act_unscaled
        step += 1
        if done and episode_kind == 'chain' and step < _eff_max_ep_len:
            ep_aborted = True

        # Mesh-explosion gate. The differential anchor term pulls the anchors
        # apart, so an over-stretched spring mesh is a live failure mode. An
        # exploded cloth still renders a plausible-looking point cloud, so
        # this has to be caught here rather than trusted to look wrong later.
        if step % 10 == 0 or done:
            sane, why = mesh_is_sane(deform, rest_extent, extra.mesh_max_growth)
            if not sane:
                ep_exploded = why
                break

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

    # 2a) Obs are captured BEFORE the step, so any path that leaves the loop
    # between the capture and the env.step() (goal reached, segment timeout,
    # hole lost, mesh explosion) leaves exactly one more obs frame than
    # actions. train_diffusion_bc.py only WARNS on that mismatch and skips the
    # demo, so it silently costs most of the dataset. Trim to the action count.
    _n = len(ep_act)
    for _buf in (ep_rgb, ep_pcd, ep_grip, ep_goal, ep_rwd):
        del _buf[_n:]
    for _m in PRIV_MODES:
        del ep_state[_m][_n:]

    # 2b) Hard drop on an exploded mesh, before any success bookkeeping.
    if ep_exploded is not None:
        n_dropped_exploded += 1
        print(f'[demo] attempt {attempts} (kept {n_kept}/{extra.n_demos})  '
              f'kind={episode_kind}  MESH EXPLODED: {ep_exploded}  (dropped)')
        continue

    # 3) At terminal step, evaluate success.
    #
    # `thread` uses the three peg metrics, unchanged. `chain` has no peg to
    # thread, so it is scored on whether the servo actually reached every
    # goal it was given. Because train_diffusion_bc.py filters on ONE
    # success_* key (whichever --success_metric selects), the chain result is
    # written into all three, so a chain demo survives any metric choice.
    # NOTE this redefines those keys for chain demos as "achieved what it was
    # asked to do" rather than "threaded the peg" — `episode_kind` is the
    # field to branch on if you need the strict meaning back.
    if episode_kind == 'chain':
        max_winding = 0.0
        if ep_aborted and extra.chain_abort_trim > 0:
            k = min(int(extra.chain_abort_trim), max(len(ep_act) - 1, 0))
            if k > 0:
                for _buf in (ep_rgb, ep_pcd, ep_grip, ep_goal, ep_act, ep_rwd):
                    del _buf[-k:]
                for _m in PRIV_MODES:
                    del ep_state[_m][-k:]
        chain_ok = int(servo.goals_reached >= extra.chain_min_goals
                       and len(ep_act) > 0)
        success_hanging = success_topological = success_legacy = chain_ok
        success_by_metric = {'hanging': chain_ok, 'topological': chain_ok,
                             'legacy': chain_ok}
    else:
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
        # Write a failure-attempt MP4 if requested and we have buffered
        # debug frames for this attempt. ep_debug_video_frames already
        # includes the post-step settle frames stitched above, so the
        # video shows where the cloth ended up.
        failed_video_path = None
        if (_is_debug_attempt
                and ep_debug_video_frames
                and n_failed_videos_written
                    < extra.debug_viz_first_n_failed):
            failed_video_path = os.path.join(
                _debug_dir,
                f'failed_attempt_{n_failed_videos_written:03d}_video.mp4')
            try:
                write_video_mp4(ep_debug_video_frames, failed_video_path,
                                fps=extra.debug_fps)
                n_failed_videos_written += 1
            except Exception as _e:
                print(f'  [debug_viz] WARN: failed-attempt video write '
                      f'failed: {_e!r}')
                failed_video_path = None
        suffix = (f'  -> wrote {os.path.basename(failed_video_path)}'
                  if failed_video_path is not None else '')
        print(f'[demo] attempt {attempts} (kept {n_kept}/{extra.n_demos})  '
              f'len={len(ep_act)}  rwd={ep_reward_total:.1f}  '
              f'h={success_hanging} t={success_topological} '
              f'l={success_legacy}  (dropped){suffix}')
        continue

    # 4) Save pkl.
    # Cloth triangulation for THIS episode's procedural mesh. The cloth is
    # re-randomized (size + hole placement) every reset and saved to a unique
    # /tmp path, so faces must be stored per-demo. They let a geometry-only
    # replay (pick_camera_angle.py) reconstruct the exact recorded cloth
    # surface without re-simulating — re-simulation can't reproduce a demo
    # because the env builds a structurally different cloth each reset.
    try:
        cloth_faces = np.asarray(
            trimesh.load(deform.args.deform_obj, process=False,
                         force='mesh').faces, dtype=np.int32)
    except Exception as _e:
        print(f'  [warn] could not load cloth faces: {_e!r}')
        cloth_faces = np.zeros((0, 3), dtype=np.int32)
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
        # The viewmat this episode was actually RENDERED with (jittered).
        'cam_viewmat': list(ep_cam_viewmat),
        'cam_roll_deg': float(ep_cam_roll),
        # The viewmat EVAL must use. Constant across the whole dataset and
        # parity-checked downstream. Without it, train_diffusion_bc.py sees
        # mixed `cam_viewmat` values, warns, and silently picks sorted()[0] —
        # i.e. eval would run at an arbitrary sampled viewpoint.
        'cam_viewmat_nominal': list(CAM_VIEWMAT_NOMINAL),
        'cam_roll_deg_nominal': float(CAM_ROLL_NOMINAL),
        'cam_randomization': dict(CAM_RANDOMIZATION),
        'hole_radius': float(hole_radius),
        # v5 episode-type bookkeeping. `episode_kind` is the field to branch
        # on when the strict meaning of success_* matters (see the success
        # block above).
        'episode_kind': episode_kind,
        'chain_n_goals': (int(extra.chain_n_goals)
                          if episode_kind == 'chain' else 0),
        'chain_goals_total': (int(servo.goals_attempted)
                              if servo is not None else 0),
        'chain_goals_reached': (int(servo.goals_reached)
                                if servo is not None else 0),
        'chain_goal_positions': (np.asarray(servo.goal_positions,
                                            dtype=np.float32)
                                 if servo is not None
                                 else np.zeros((0, 3), np.float32)),
        'chain_segment_steps': (np.asarray(servo.segment_steps, dtype=np.int32)
                                if servo is not None
                                else np.zeros((0,), np.int32)),
        'chain_unreliable_hole_frames': (int(servo.n_unreliable_frames)
                                         if servo is not None else 0),
        'chain_min_goals': (int(extra.chain_min_goals)
                            if episode_kind == 'chain' else 0),
        'chain_ended_early': bool(ep_aborted),
        'chain_brake_steps': (int(servo.n_brake_steps)
                              if servo is not None else 0),
        'chain_v_max': float(extra.chain_v_max),
        'chain_v_diff_max': float(extra.chain_v_diff_max),
        'chain_r_range': [float(extra.chain_r_min), float(extra.chain_r_max)],
        'cloth_width': float(cloth_width),
        'deform_init_pos': [float(x) for x in ep_start_pos],
        'deform_init_pos_nominal': [float(x) for x in START_NOMINAL],
        'rest_extent': float(rest_extent),
        # Per-episode cloth triangulation (F, 3). Enables physics-free
        # geometry replay of obs['full_mesh'] in pick_camera_angle.py.
        'cloth_faces': cloth_faces,
        'max_act_vel': float(DeformEnv.MAX_ACT_VEL),
        # Control-frequency parity. The trajectory was built at this
        # ctrl_freq and each row of `acts` advances 1/ctrl_freq seconds
        # of sim time. A training/eval env at a different ctrl_freq
        # would interpret the same action stream at a different speed.
        # train_diffusion_bc.py reads these to patch its eval env.
        'ctrl_freq': float(_actual_ctrl_freq),
        'sim_freq': int(extra.sim_freq),
        'sim_steps_per_action': int(_steps_per_action),
        # Demo-speed slowdown factor applied to the scripted-controller
        # velocities (1.0 = unmodified, 0.5 = half-speed commanded
        # velocities with 2x trajectory length). Demos with non-1.0
        # demo_speed teach the policy slower commanded velocities, which
        # are more realistic to replay on a real robot.
        'demo_speed': float(_actual_demo_speed),
        # Per-episode hanger goal randomization. Recorded so the training
        # script can enforce parity across the demo dir, and so eval-time
        # env construction matches the distribution the policy trained on.
        # The actual sampled (dx, dy) for THIS episode is recoverable from
        # obs['goal'][0] vs the nominal (0, 0, 8.2)/WBOX; we store the
        # radius (the distribution parameter) here, not the per-episode
        # draw.
        # Nominal peg tip for THIS run. The v5 and v6 datasets differ by this
        # value alone and nothing in the pkls said so, which is what made the
        # 48% -> 11% expert regression hard to attribute.
        'peg_nominal': (None if extra.peg_nominal is None
                        else [float(v) for v in extra.peg_nominal]),
        'no_post': bool(extra.no_post),
        'slack_used': bool(_slack_used),
        'slack_squeeze': (float(_slack_squeeze) if _slack_used else None),
        'slack_min_z': (float(_slack_min_z) if _slack_used else None),
        # Mesh-variation ranges for the run, plus THIS episode's realized mesh
        # resolution — node_density drives vertex count and hole granularity,
        # so it is the one per-episode mesh parameter worth recovering later.
        'node_density': int(getattr(args, 'node_density', 15)),
        'node_density_range': extra.node_density_range,
        # The outline shape actually drawn for THIS cloth; procedural_utils
        # writes it back onto args after sampling.
        'cloth_shape': getattr(args, 'proc_cloth_shape', 'rect'),
        'cloth_shapes_enabled': extra.proc_cloth_shapes,
        'proc_cloth_size_range': extra.proc_cloth_size_range,
        'proc_hole_frac_range': extra.proc_hole_frac_range,
        'randomize_goal_radius': float(extra.randomize_goal_radius),
        'randomize_goal_dz': float(extra.randomize_goal_dz),
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
    n_kept_by_kind[episode_kind] += 1
    # Realized per-step anchor speed in sim units/s, for the real-vs-sim
    # comparison printed at the end of the run. acts are normalized by
    # MAX_ACT_VEL, so undo that and take the per-anchor magnitude.
    _a = np.asarray(ep_act, dtype=np.float32) * float(DeformEnv.MAX_ACT_VEL)
    _speed_samples.append(
        np.linalg.norm(_a.reshape(len(_a), 2, 3), axis=2).reshape(-1))
    pkl_mb = os.path.getsize(out_path) / 1e6
    # Unnormalized peak commanded velocity (m/s). peak_abs_a is in
    # normalized [-1, 1] units (the policy's space); multiplying by
    # MAX_ACT_VEL recovers the raw velocity that was sent to the env.
    # Use this to pick a tighter --max_act_vel: pick a value slightly
    # above the run's max peak|v| so the [-1, 1] range stays well-
    # utilized without saturating.
    _peak_v_mps = act_stats["peak_abs_a"] * float(DeformEnv.MAX_ACT_VEL)
    print(f'[demo] attempt {attempts} (kept {n_kept}/{extra.n_demos})  '
          f'len={act_stats["ep_len"]}  '
          f'active={act_stats["n_active"]} hold={act_stats["n_hold"]} '
          f'({act_stats["hold_frac"]*100:.0f}% hold)  '
          f'peak|a|={act_stats["peak_abs_a"]:.2f} '
          f'(peak|v|={_peak_v_mps:.2f} m/s) '
          f'mean||a||={act_stats["mean_norm_a"]:.2f}  '
          f'rwd={ep_reward_total:.1f}  '
          f'{episode_kind}'
          f'{f"[{servo.goals_reached}/{servo.goals_attempted} goals]" if servo is not None else ""}  '
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
if extra.debug_viz_first_n_failed > 0:
    print(f'  failed-attempt videos written: {n_failed_videos_written}'
          f'/{extra.debug_viz_first_n_failed} (target)')

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

# ---------------------------------------------------------------------------
# v5 summary: episode mix, mesh-explosion drops, and the speed comparison
# against the real teleop demos. The speed check is the one that decides
# whether these demos are replayable on the rig at all — a dataset that
# commands 3x the fastest human motion is not a sim2real dataset.
# ---------------------------------------------------------------------------
print(f'\n=== v5 episode mix ===')
print(f'  kept thread : {n_kept_by_kind["thread"]}')
print(f'  kept chain  : {n_kept_by_kind["chain"]}')
print(f'  dropped (mesh exploded): {n_dropped_exploded}')
if n_dropped_exploded > 0.2 * max(attempts, 1):
    print(f'  WARNING: {n_dropped_exploded}/{attempts} attempts exploded. '
          f'Lower --chain_v_diff_max or tighten the separation range in '
          f'HoleServo before scaling up.')

if _speed_samples:
    sp = np.concatenate(_speed_samples)
    sp = sp[sp > 1e-6]  # ignore the zero-velocity hold frames
    if len(sp):
        print(f'\n=== Commanded anchor speed vs the real rig (sim units/s) ===')
        print(f'  {"":10s} {"p50":>8s} {"p90":>8s} {"max":>8s}')
        print(f'  {"collected":10s} {np.percentile(sp, 50):8.2f} '
              f'{np.percentile(sp, 90):8.2f} {sp.max():8.2f}')
        print(f'  {"real":10s} {REAL_SPEED_P50:8.2f} {REAL_SPEED_P90:8.2f} '
              f'{REAL_SPEED_MAX:8.2f}   (0807_demos, 8 demos, both arms)')
        ratio = float(np.percentile(sp, 90)) / REAL_SPEED_P90
        verdict = ('OK' if ratio <= 1.5 else
                   'TOO FAST — lower --chain_v_max / --demo_speed')
        print(f'  p90 ratio collected/real = {ratio:.2f}x  -> {verdict}')
        sat = float((sp >= 0.99 * DeformEnv.MAX_ACT_VEL).mean())
        print(f'  fraction of steps at the MAX_ACT_VEL ceiling: {sat*100:.2f}% '
              f'(must be ~0; the ceiling is {DeformEnv.MAX_ACT_VEL})')
