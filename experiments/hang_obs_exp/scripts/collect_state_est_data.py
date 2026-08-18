"""
Collect a UniClothDiff-format state-estimation dataset from dedo HangProcCloth.

Each rollout re-randomizes the procedural cloth (size + hole placement), so every
episode is its own *cloth* with its own topology. We record, per simulation step:

  - the full GT cloth mesh vertex positions   (V, 3)  WORLD meters
  - a cloth-only partial point cloud           (P, 3)  WORLD meters

and, once per cloth, its topology (faces/edges), a rest-pose template, and the
hole-loop vertex indices. A trained UniClothDiff GPSStateEstModel reconstructs the
full mesh from the partial point cloud; downstream we index the hole-loop vertices
in the predicted mesh and take their centroid to recover the privileged
hole-location the diffusion BC policy consumes.

Two passes (run both for the "diverse + on-trajectory" mix):
  scripted : the same hole-aware waypoint expert used by collect_bc_demos.py, so
             the estimator sees the exact states the policy will encounter.
  random   : smooth random anchor motions that drape the cloth into a broad
             variety of configurations, for state-estimation coverage.

Output is a single HDF5 matching ClothStateEstVariableDataset's schema:

  <out>.h5
  ├── training/
  │   └── cloth_NNNNN/                         (attrs: hole_vertex_indices, source)
  │       ├── rest_positions  (V, 3)           step-0 world pose (template)
  │       ├── faces           (F, 3)
  │       ├── edges           (E, 2)           undirected unique
  │       └── trajectory_0/
  │           └── step_NNNN/
  │               ├── positions       (V, 3)
  │               └── pointclouds/cam_0 (P, 3)
  └── validation/   [same layout]

Rest-pose note: we use the step-0 getMeshData pose (not the object-local .obj
verts) as `rest_positions` so the template lives in the SAME world frame/scale as
the deformed `positions` — the .obj verts are in object-local units (~±1) while
the sim mesh is in world meters, and the model's [noisy_pos || rest_pos] input
channel needs both in one frame.

Usage (from repo root):
  python experiments/hang_obs_exp/scripts/collect_state_est_data.py \
      --out experiments/hang_obs_exp/data/state_est/dedo_hang.h5 \
      --n_scripted 150 --n_random 150 \
      --randomize_yaw --randomize_pitch --randomize_dist --randomize_physics \
      --depth_noise_std 1e-3 --depth_dropout 0.1

  For anything intended to transfer to the real robot, add
  `--env HangProcClothReal-v1` (world metres) and `--demo_speed 0.1` — at 15 Hz
  a real-scale expert demo is only ~6 control steps long, so it must be
  stretched to yield a useful number of frames.
"""
import argparse
import json
import os
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT))

import gym
import h5py
import numpy as np
import trimesh

import dedo  # noqa: F401  (registers gym envs)
from dedo.envs.deform_env import DeformEnv
from dedo.utils.args import get_args_parser, args_postprocess
from dedo.utils.mesh_utils import get_mesh_data
from dedo.demo_preset import build_traj, merge_traj

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _helpers import RetryResetEnv, build_hole_aware_waypoints  # noqa: E402
from _bc_obs_helpers import (  # noqa: E402
    capture_rgb_depth, cloth_only_pcd, get_hole_indices,
    resolve_deform, patch_deform_render_to_obs_camera)
from _debug_viz import (  # noqa: E402
    hole_centroid_world, overlay_pcd_on_rgb, pcd_camera_view_image,
    render_sim_with_centroid, build_video_frame, write_video_mp4)


# ---------------------------------------------------------------------------
# Argparse
# ---------------------------------------------------------------------------
parser = argparse.ArgumentParser()
parser.add_argument('--out', type=str,
                    default='experiments/hang_obs_exp/data/state_est/dedo_hang.h5',
                    help='Output HDF5 path. Written into the repo workspace by '
                         'default (visible in VSCode), not /tmp.')
parser.add_argument('--env', type=str, default='HangProcCloth-v1',
                    choices=['HangProcCloth-v1', 'HangProcClothReal-v1'],
                    help='Use HangProcClothReal-v1 for anything intended to '
                         'transfer: it is in world METRES, whereas '
                         'HangProcCloth-v1 is ~22x real scale '
                         '(frame_transforms.md: scale=0.045). The estimator '
                         'normalizes rest positions per cloth, but the action '
                         'scale, camera distances and physics ranges are all '
                         'scene-unit dependent, so mixing the two in one '
                         'dataset is a bug, not augmentation.')
parser.add_argument('--stats_json', type=str, default='',
                    help='Optional path for a per-episode randomization + data '
                         'statistics dump (JSON). Default: <out>_stats.json.')
parser.add_argument('--n_scripted', type=int, default=150,
                    help='Number of scripted-expert episodes (on-trajectory '
                         'states the policy will actually see).')
parser.add_argument('--n_random', type=int, default=150,
                    help='Number of random-motion episodes (diverse drapes for '
                         'state-estimation coverage).')
parser.add_argument('--val_ratio', type=float, default=0.1,
                    help='Fraction of cloths routed to the validation split.')
parser.add_argument('--max_steps', type=int, default=120,
                    help='Max sim steps recorded per episode.')
parser.add_argument('--step_stride', type=int, default=1,
                    help='Record every Nth step (decorrelates frames / shrinks '
                         'the dataset). 1 = every step.')
parser.add_argument('--seed', type=int, default=2026)
parser.add_argument('--cam_resolution', type=int, default=None,
                    help='RGB+depth image height/width. Default is per env '
                         '(128 sim-scale / 384 real): the real cloth covers '
                         'only a few %% of the image, so 128 supplies far fewer '
                         'than 2048 unique points. None = use the preset.')
parser.add_argument('--pcd_n_points', type=int, default=2048,
                    help='Points sampled from the back-projected depth buffer. '
                         '2048 matches the model num_sample_points; collect '
                         'dense here regardless of the policy PCD budget.')
parser.add_argument('--cam_viewmat', type=float, nargs=6, default=None,
                    help='dedo cam_viewmat: dist pitch yaw tx ty tz. Default is '
                         'PER ENV so the estimator sees the same viewpoint as '
                         'the policy: [14, -5, 45, 0, 0, 5.5] for '
                         'HangProcCloth-v1 (matches collect_bc_demos.py), '
                         '[0.7, -25, 45, 0.4, 0, 0.42] for HangProcClothReal-v1 '
                         '(MEASURED by coverage sweep, not rescaled: the naive '
                         'rescale aims at the peg and misses the cloth, which '
                         'spawns 0.34 m higher, yielding ZERO cloth pixels). When '
                         '--randomize_{yaw,pitch,dist} are set the '
                         'corresponding entries are overwritten per episode.')
parser.add_argument('--randomize_yaw', action='store_true',
                    help='Sample the camera YAW uniformly in --yaw_range each '
                         'episode (other cam params fixed). Gives the state '
                         'estimator varied visible cloth subsets / occlusion '
                         'patterns in ONE dataset — viewpoint augmentation, not '
                         'conflicting data, since the model works on a centered '
                         '3-D point cloud rather than a 2-D image.')
parser.add_argument('--yaw_range', type=float, nargs=2, default=None,
                    help='[min, max] degrees for --randomize_yaw. Default is '
                         'per env (see _RAND_PRESETS): 0..90 sim-scale, 15..60 '
                         'real (measured — the real cloth leaves frame outside '
                         'that band).')
# Pitch and distance complete the camera axis. Same argument as --randomize_yaw:
# the model consumes a centered 3-D cloud, so moving the camera changes WHICH
# surface is visible (the occlusion pattern) without changing the frame the
# points live in. Grazing pitches are what make the estimator robust to the
# viewpoints the policy will be scored under (--eval_cam_yaw) and to a real
# camera that will never sit exactly where the sim one did.
parser.add_argument('--randomize_pitch', action='store_true',
                    help='Sample camera PITCH uniformly in --pitch_range each '
                         'episode.')
parser.add_argument('--pitch_range', type=float, nargs=2, default=None,
                    help='[min, max] degrees for --randomize_pitch. Negative '
                         'looks down; -5 is near-horizontal (grazing). Default '
                         'is per env (see _RAND_PRESETS).')
parser.add_argument('--randomize_dist', action='store_true',
                    help='Sample camera DISTANCE uniformly in --dist_range each '
                         'episode (changes point density, like a real camera at '
                         'an unknown standoff).')
parser.add_argument('--dist_range', type=float, nargs=2, default=None,
                    help='[min, max] for --randomize_dist, in scene units. '
                         'Default is per env (see _RAND_PRESETS), falling back '
                         'to 0.8x..1.3x the --cam_viewmat distance.')
parser.add_argument('--randomize_goal_radius', type=float, default=0.0,
                    help='Match the demo dir setting for env parity.')
parser.add_argument('--randomize_goal_dz', type=float, default=0.0,
                    help='Peg-height randomization half-extent (scene units). '
                         'Match the demo dir setting for env parity.')
# Cloth physics. The env reads these from args when it loads the deformable on
# every reset (dedo/utils/init_utils.py::load_deform_object), so overwriting
# deform.args.* before env.reset() takes effect for that episode. Ranges should
# BRACKET the real cloth rather than match it — see cloth_sweep/ for the
# bending x damping grid these defaults were read off.
parser.add_argument('--randomize_physics', action='store_true',
                    help='Jitter cloth physics per episode MULTIPLICATIVELY '
                         'around the values the scene already tuned (see '
                         '--physics_jitter). Absolute ranges would be wrong: '
                         'procedural_hang_cloth_real is tuned to mass 0.1 / '
                         'elastic 120 / bending 20 / damping 0.1, whereas the '
                         'sim-scale scene uses mass 1 / elastic 50 / bending 1, '
                         'and task_info.py warns that elastic must stay inside '
                         'the proven-stable 50-150 band (k=50 collapses the '
                         'small cloth, k=1100 explodes the solver at dt=1/500).')
parser.add_argument('--physics_jitter', type=float, default=1.6,
                    help='Multiplicative half-range f: each parameter is scaled '
                         'by exp(U(-ln f, +ln f)), i.e. within [x/f, x*f] of the '
                         "scene's own value, log-uniform. 1.6 gives a ~2.5x "
                         'spread. Set 1.0 to disable while keeping the flag.')
parser.add_argument('--elastic_clamp', type=float, nargs=2, default=[50.0, 150.0],
                    help='Hard clamp on deform_elastic_stiffness after jitter — '
                         'the stability band from task_info.py. Outside it the '
                         'cloth either collapses or the explicit spring solver '
                         'blows up, and BOTH failures silently produce frames '
                         'with the cloth out of view.')
# Depth-sensor realism. Applied to the depth buffer BEFORE back-projection, so
# the noise lands where a real depth camera's does (range error along the ray,
# plus dropped pixels) rather than as isotropic jitter in world space.
parser.add_argument('--depth_noise_std', type=float, default=0.0,
                    help='Gaussian sigma added to the NON-LINEAR depth buffer '
                         'in [0,1) units, per pixel, before back-projection. '
                         '0 = off. Try 1e-3.')
parser.add_argument('--min_valid_px', type=int, default=256,
                    help='Truncate the episode when unique cloth pixels drop '
                         'below this. Late in a hang episode the cloth often '
                         'rotates edge-on and coverage collapses (observed: '
                         '3262 -> 1906 -> 40 px over one rollout). Such a frame '
                         'is not merely occluded, it is unusable supervision: '
                         'depth_to_pcd up-samples WITH REPLACEMENT, so 40 real '
                         'points become 2048 duplicated ones that look like a '
                         'full observation. NB the dataset-side `min_pcd_points` '
                         'filter cannot catch this — stored clouds are always '
                         'exactly pcd_n_points long by construction — so the '
                         'floor has to be enforced here. 0 = only drop empty '
                         'frames.')
parser.add_argument('--max_edge_stretch', type=float, default=3.0,
                    help='Truncate the episode at the first frame whose max '
                         'edge length exceeds this multiple of its rest '
                         'length — i.e. when the explicit spring solver has '
                         'gone unstable and the mesh is inflating. Such frames '
                         'are physically invalid GT but structurally perfect '
                         '(right shape, no NaNs), so nothing downstream '
                         'complains: the 2026-07-30 collection shipped with '
                         '100% of random-action and 26% of scripted episodes '
                         'affected, peaking at 22.7x. Default 3.0 sits above '
                         'the 2.94x worst case across the 854 clean BC demos. '
                         '0 = off.')
parser.add_argument('--guard_episodes', type=int, default=3,
                    help='Check cloth-pixel coverage over the first N kept '
                         'episodes and abort if the camera is framing empty '
                         'space. 0 disables the guard.')
parser.add_argument('--guard_max_empty', type=float, default=0.25,
                    help='Abort if more than this fraction of guarded frames '
                         'have zero valid cloth pixels.')
parser.add_argument('--depth_dropout', type=float, default=0.0,
                    help='Fraction of valid cloth pixels dropped at random '
                         'before back-projection, mimicking a depth sensor '
                         'failing on thin/oblique/dark surfaces. 0 = off.')
parser.add_argument('--sim_freq', type=int, default=None,
                    help='PyBullet step rate. Default per env: 500 sim-scale, '
                         '1000 real — task_info.py notes the real scene needs '
                         'the higher rate because its stiffer/lighter springs '
                         'are unavoidably faster at dt=1/500.')
parser.add_argument('--ctrl_freq', type=float, default=15.0)
# Random-motion controls.
parser.add_argument('--rand_amp', type=float, default=None,
                    help='Random-pass action amplitude (normalized [-1,1] units '
                         'before MAX_ACT_VEL scaling). Default per env: 0.35 '
                         'sim-scale, 0.02 real. At MAX_ACT_VEL=10 and 15 Hz, '
                         '0.35 commands 0.23 m/step, which is the ENTIRE real '
                         'workspace in one step - the cloth is yanked out of '
                         'bounds and the episode terminates after ~5 frames.')
parser.add_argument('--rand_hold', type=int, default=8,
                    help='Random-pass: hold each sampled action this many steps '
                         '(low-frequency motion drapes the cloth smoothly).')
parser.add_argument('--max_act_vel', type=float, default=10.0,
                    help='DeformEnv.MAX_ACT_VEL (matches collect_bc_demos.py).')
parser.add_argument('--demo_speed', type=float, default=1.0,
                    help='Slowdown factor on the SCRIPTED expert velocities '
                         '(mirrors collect_bc_demos.py). <1 stretches the '
                         'trajectory by round(1/demo_speed) and scales the '
                         'commanded velocities by demo_speed, so total '
                         'displacement is preserved but motion is slower / more '
                         'realistic. Use e.g. 0.3 with --max_act_vel 2.0. Only '
                         'affects the scripted pass (random already gentle).')
# Sim-RGB debug videos: re-render the actual PyBullet scene during collection
# and write a 3-panel MP4 (sim render w/ hole centroid | point cloud overlaid
# on the obs RGB | point cloud from the camera view) for the first N kept
# episodes of EACH source. Confirms the back-projected cloud lines up with the
# real cloth/peg from a human-readable angle. 0 = off (the lightweight
# viz_state_est_data.py --mp4 path animates the stored data without re-sim).
parser.add_argument('--debug_video_first_n', type=int, default=0,
                    help='Write a 3-panel sim-RGB MP4 for the first N kept '
                         'scripted AND N kept random episodes. 0 = off.')
parser.add_argument('--debug_render_size', type=int, default=300,
                    help='Per-panel size for the sim-RGB debug video.')
parser.add_argument('--debug_fps', type=int, default=10,
                    help='Frame rate for the sim-RGB debug video.')
extra = parser.parse_args()


# Per-env camera preset (see --cam_viewmat). Getting this wrong is silent: the
# capture succeeds, the stored cloud is the all-zeros fallback from
# depth_to_pcd, and only `valid_cloth_px` reveals it — which is why the guard
# below aborts rather than warns.
#
# The real-env value was MEASURED, not rescaled: a coverage sweep over
# (dist, pitch, yaw, tz) scored mean cloth-pixel count across a scripted
# trajectory. The naive rescale of the sim camera ([0.63, -5, 45, 0.5, 0, 0.21])
# aims at the PEG, but the cloth spawns at (0.275, 0, 0.55) — 0.34 m higher —
# so it saw the cloth in <10% of frames. Raising the target to z=0.42 and
# pulling x back to 0.4 gives 100% usable frames at ~4% image coverage.
# Counter-intuitively yaw=90 (face-on to the XZ-plane cloth) scored WORST
# (0% usable), so keep 45.
_CAM_PRESETS = {
    'HangProcCloth-v1': [14.0, -5.0, 45.0, 0.0, 0.0, 5.5],
    'HangProcClothReal-v1': [0.7, -25.0, 45.0, 0.4, 0.0, 0.42],
}
# The real cloth covers only a few percent of the image, so resolution decides
# how many UNIQUE points a cloud can have (depth_to_pcd up-samples with
# replacement below --pcd_n_points). Measured over 329 frames of randomized
# cameras, median unique cloth pixels:
#     res 256, yaw 0-90 :  894    res 384, yaw 0-90 : 2022
#     res 256, yaw 15-60: 1295    res 384, yaw 15-60: 2930  <- shipped
_RES_PRESETS = {'HangProcCloth-v1': 128, 'HangProcClothReal-v1': 384}
# Camera randomization bands, per env. These are NOT free parameters: a yaw
# sweep showed the real cloth (XZ plane) drops out of view past ~75 deg and at
# ~0 deg, so widening yaw to 0-90 cuts median coverage from 2930 to 2022 px for
# no extra diversity that the estimator can use. Randomization ranges have to be
# validated JOINTLY with the nominal pose — a per-axis check misses this.
_RAND_PRESETS = {
    'HangProcCloth-v1':     {'yaw': [0.0, 90.0], 'pitch': [-60.0, -5.0],
                             'dist': None, 'rand_amp': 0.35, 'sim_freq': 500},
    'HangProcClothReal-v1': {'yaw': [15.0, 60.0], 'pitch': [-50.0, -5.0],
                             'dist': [0.6, 0.9], 'rand_amp': 0.02,
                             'sim_freq': 1000},
}
if extra.cam_viewmat is None:
    extra.cam_viewmat = list(_CAM_PRESETS[extra.env])
    print(f'[cam] using {extra.env} preset cam_viewmat={extra.cam_viewmat}')
if extra.cam_resolution is None:
    extra.cam_resolution = _RES_PRESETS[extra.env]
    print(f'[cam] using {extra.env} preset cam_resolution={extra.cam_resolution}')
_rp = _RAND_PRESETS[extra.env]
if extra.sim_freq is None:
    extra.sim_freq = _rp['sim_freq']
if extra.rand_amp is None:
    extra.rand_amp = _rp['rand_amp']
if extra.yaw_range is None:
    extra.yaw_range = list(_rp['yaw'])
if extra.pitch_range is None:
    extra.pitch_range = list(_rp['pitch'])
if extra.dist_range is None and _rp['dist'] is not None:
    extra.dist_range = list(_rp['dist'])
print(f'[cam] randomization bands: yaw={extra.yaw_range} '
      f'pitch={extra.pitch_range} dist={extra.dist_range}')

_steps_per_action = max(1, int(round(extra.sim_freq / extra.ctrl_freq)))
_ctrl_freq = extra.sim_freq / _steps_per_action

# Demo-speed stretch for the scripted pass (same scheme as collect_bc_demos.py):
# repeat each waypoint N=round(1/demo_speed) times and scale velocities by 1/N.
if extra.demo_speed <= 0:
    raise ValueError(f'--demo_speed must be > 0, got {extra.demo_speed}')
_demo_stretch = max(1, int(round(1.0 / extra.demo_speed)))
_actual_demo_speed = 1.0 / _demo_stretch

os.makedirs(os.path.dirname(os.path.abspath(extra.out)), exist_ok=True)


# ---------------------------------------------------------------------------
# Build dedo args + env (mirrors collect_bc_demos.py).
# ---------------------------------------------------------------------------
sys.argv = [
    'collect_state_est_data',
    f'--env={extra.env}',
    f'--cam_resolution={extra.cam_resolution}',
    '--num_envs=0',
    '--total_env_steps=0',
    '--seed', str(extra.seed),
    '--max_episode_len', str(max(extra.max_steps * 2, 400)),
    f'--sim_freq={extra.sim_freq}',
    f'--sim_steps_per_action={_steps_per_action}',
    '--cam_viewmat', *[str(x) for x in extra.cam_viewmat],
    f'--randomize_goal_radius={extra.randomize_goal_radius}',
    f'--randomize_goal_dz={extra.randomize_goal_dz}',
]
args, _ = get_args_parser()
args_postprocess(args)
args.debug = False
args.viz = False
args.uint8_pixels = True

DeformEnv.MAX_ACT_VEL = float(extra.max_act_vel)

env = gym.make(args.env, args=args)
env = RetryResetEnv(env)
env.seed(extra.seed)
deform = resolve_deform(env)
patch_deform_render_to_obs_camera(deform)
np.random.seed(extra.seed)

# The scene's own tuned physics, captured BEFORE any episode randomization
# overwrites deform.args. --randomize_physics jitters multiplicatively around
# these, so the same jitter factor is correct for both the sim-scale and the
# real-metre scene (whose tuned values differ by ~10-20x).
_BASE_PHYS = {k: float(getattr(deform.args, k))
              for k in ('deform_bending_stiffness', 'deform_damping_stiffness',
                        'deform_elastic_stiffness', 'deform_mass',
                        'deform_friction_coeff')}
print('[phys] scene baseline: ' + '  '.join(
    f'{k.replace("deform_", "")}={v:g}' for k, v in _BASE_PHYS.items()))


# ---------------------------------------------------------------------------
# Topology helpers.
# ---------------------------------------------------------------------------
def _faces_to_unique_edges(faces):
    """Triangle faces -> unique undirected edge list (E, 2) int64.
    The loader makes these bidirectional; we store undirected for compactness."""
    edges = set()
    for f in faces:
        for k in range(len(f)):
            a, b = int(f[k]), int(f[(k + 1) % len(f)])
            edges.add((a, b) if a < b else (b, a))
    return np.array(sorted(edges), dtype=np.int64) if edges else np.zeros((0, 2), np.int64)


def _load_cloth_topology(num_sim_verts):
    """Load faces from the current procedural .obj and derive edges.

    getMeshData returns the *simulation* mesh; faces come from the .obj. For
    dedo's procedural cloths these share vertex indexing, but we assert the
    face indices fit the sim vertex count and bail on mismatch rather than
    silently writing corrupt topology.
    """
    faces = np.asarray(
        trimesh.load(deform.args.deform_obj, process=False, force='mesh').faces,
        dtype=np.int64)
    if faces.size == 0 or int(faces.max()) >= num_sim_verts:
        return None, None
    edges = _faces_to_unique_edges(faces)
    return faces, edges


def _log_uniform(lo, hi):
    return float(np.exp(np.random.uniform(np.log(lo), np.log(hi))))


def _sample_episode_randomization():
    """Sample per-episode camera + physics and write them onto deform.args.

    MUST be called BEFORE env.reset(): the deformable's physical parameters are
    read out of args when the env loads it on reset
    (dedo/utils/init_utils.py::load_deform_object), so setting them afterwards
    has no effect on that episode. The camera is different — deform._cam_viewmat
    is a property recomputed from args on every access, so it can be set either
    side of the reset; we do it here to keep one code path.

    Returns the dict of sampled values (for the stats dump / H5 attrs).
    """
    rec = {}
    vm = list(deform.args.cam_viewmat)
    if extra.randomize_dist:
        lo, hi = (extra.dist_range if extra.dist_range is not None
                  else (0.8 * extra.cam_viewmat[0], 1.3 * extra.cam_viewmat[0]))
        vm[0] = float(np.random.uniform(lo, hi))
    if extra.randomize_pitch:
        vm[1] = float(np.random.uniform(*extra.pitch_range))
    if extra.randomize_yaw:
        vm[2] = float(np.random.uniform(*extra.yaw_range))
    deform.args.cam_viewmat = vm
    rec['cam_dist'], rec['cam_pitch'], rec['cam_yaw'] = vm[0], vm[1], vm[2]

    if extra.randomize_physics and extra.physics_jitter > 1.0:
        f = float(extra.physics_jitter)
        jitter = ('deform_bending_stiffness', 'deform_damping_stiffness',
                  'deform_elastic_stiffness', 'deform_mass',
                  'deform_friction_coeff')
        p = {k: _BASE_PHYS[k] * _log_uniform(1.0 / f, f) for k in jitter}
        p['deform_elastic_stiffness'] = float(np.clip(
            p['deform_elastic_stiffness'], *extra.elastic_clamp))
        for k, v in p.items():
            setattr(deform.args, k, v)
        rec.update(p)
    else:
        for k in ('deform_bending_stiffness', 'deform_damping_stiffness',
                  'deform_elastic_stiffness', 'deform_mass',
                  'deform_friction_coeff'):
            rec[k] = float(getattr(deform.args, k, float('nan')))
    return rec


def _corrupt_depth(depth, seg):
    """Add range noise + dropout to the depth buffer before back-projection.

    Noise goes on the non-linear buffer (where a real sensor's range error
    lives, i.e. along the camera ray), not on world-space points. Dropout is
    applied by pushing pixels past the depth_to_pcd background cutoff (0.999)
    so it flows through the existing valid mask with no signature change.
    Only cloth pixels are touched; the peg/background are already filtered.
    """
    if extra.depth_noise_std <= 0 and extra.depth_dropout <= 0:
        return depth
    depth = depth.copy()
    cloth = (seg == int(deform.deform_id)) & (depth < 0.999)
    if extra.depth_noise_std > 0:
        noise = np.random.normal(0.0, extra.depth_noise_std, size=depth.shape)
        depth[cloth] = np.clip(depth[cloth] + noise[cloth], 0.0, 0.9989)
    if extra.depth_dropout > 0:
        drop = cloth & (np.random.random(depth.shape) < extra.depth_dropout)
        depth[drop] = 1.0
    return depth


def _capture_step(build_debug=False, hole_idx=None):
    """Return (positions (V,3), pcd (P,3), debug_frame) at the current sim
    state. positions/pcd are None if the mesh has NaN verts. debug_frame is a
    3-panel uint8 image (sim render | pcd-on-RGB | pcd camera view) when
    build_debug, else None."""
    _, verts = get_mesh_data(deform.sim, deform.deform_id)
    positions = np.asarray(verts, dtype=np.float32)
    if not np.isfinite(positions).all():
        return None, None, None, 0
    rgb, depth, seg, view, proj = capture_rgb_depth(
        deform, extra.cam_resolution, extra.cam_resolution)
    depth = _corrupt_depth(depth, seg)
    pcd = cloth_only_pcd(depth, seg, view, proj, deform.deform_id,
                         extra.pcd_n_points)
    # True valid cloth-pixel count BEFORE the fixed-size resample. depth_to_pcd
    # up-samples with replacement when there are fewer valid pixels than
    # pcd_n_points, so the stored cloud is always n_points long and this is the
    # only place the real coverage is visible. It doubles as the per-frame
    # occlusion proxy for the dataset statistics.
    valid_px = int(((seg == int(deform.deform_id)) & (depth < 0.999)).sum())
    frame = None
    if build_debug:
        sz = extra.debug_render_size
        centroid_w = (hole_centroid_world(deform, hole_idx)
                      if hole_idx else None)
        sim_panel = render_sim_with_centroid(deform, view, proj, centroid_w, size=sz)
        obs_overlay = overlay_pcd_on_rgb(rgb, pcd, view, proj)
        pcd_cam = pcd_camera_view_image(pcd, view, proj, size=sz,
                                        colormap_by='depth')
        frame = build_video_frame(sim_panel, obs_overlay, pcd_cam, size=sz)
    return positions, pcd, frame, valid_px


# ---------------------------------------------------------------------------
# Action generators.
# ---------------------------------------------------------------------------
def _scripted_actions():
    """Build the hole-aware expert trajectory (normalized actions), or None."""
    wp = build_hole_aware_waypoints(deform)
    if wp is None:
        return None
    try:
        _, va = build_traj(deform, wp, 'a', anchor_idx=0,
                           ctrl_freq=_ctrl_freq, robot=None)
        _, vb = build_traj(deform, wp, 'b', anchor_idx=1,
                           ctrl_freq=_ctrl_freq, robot=None)
        traj = merge_traj(va, vb)
    except Exception as e:
        print(f'  [scripted] build_traj failed ({e!r})')
        return None
    # Slow the expert: repeat each waypoint and scale velocity magnitudes so the
    # cloth makes the same motion over more, gentler steps (matches the
    # collect_bc_demos.py --demo_speed behavior the policy was trained on).
    if _demo_stretch > 1:
        traj = np.repeat(traj, _demo_stretch, axis=0) * _actual_demo_speed
    return np.clip(traj / DeformEnv.MAX_ACT_VEL, -1.0, 1.0).astype(np.float32)


def _random_action_stream(n_steps):
    """Piecewise-constant smooth random actions in [-amp, amp]^6."""
    acts = np.zeros((n_steps, 6), dtype=np.float32)
    cur = np.zeros(6, dtype=np.float32)
    for t in range(n_steps):
        if t % extra.rand_hold == 0:
            cur = np.random.uniform(-extra.rand_amp, extra.rand_amp,
                                    size=6).astype(np.float32)
        acts[t] = cur
    return acts


# ---------------------------------------------------------------------------
# One episode: reset, capture topology, roll out, collect per-step frames.
# ---------------------------------------------------------------------------
def run_episode(source, debug=False):
    """source in {'scripted','random'}. Returns a dict ready to write, or None
    if the episode was unusable (no hole, bad topology, expert failed, ...).
    When debug, also builds a 3-panel sim-RGB video frame per recorded step."""
    # Sample camera + physics BEFORE reset so the deformable is loaded with this
    # episode's physical parameters (see _sample_episode_randomization).
    rand_rec = _sample_episode_randomization()
    env.reset()
    cam_yaw = rand_rec['cam_yaw']
    # Peg pose actually used this episode (goal randomization happens inside
    # reset()). Recorded so the stats can show the realized goal distribution
    # rather than just the requested range.
    rand_rec['goal_pos'] = np.asarray(deform.goal_pos[0], dtype=np.float32).tolist()
    rand_rec['goal_delta'] = np.asarray(
        getattr(deform, '_last_goal_delta', np.zeros(3)), dtype=np.float32).tolist()
    hole_idx = get_hole_indices(deform)
    if not hole_idx:
        return None

    _, verts0 = get_mesh_data(deform.sim, deform.deform_id)
    rest_pos = np.asarray(verts0, dtype=np.float32)
    num_verts = rest_pos.shape[0]
    if not np.isfinite(rest_pos).all():
        return None
    if max(hole_idx) >= num_verts:
        return None

    faces, edges = _load_cloth_topology(num_verts)
    if faces is None:
        print('  [warn] cloth faces/sim-mesh mismatch, skipping episode')
        return None

    # Actuated (grasped/anchored) vertices — the nodes the GNS dynamics model
    # treats as externally driven. deform.anchors maps anchor_id -> {'vertices'}
    # in anchor-creation order (anchor 0, anchor 1, ...). We store the flat
    # union for the GNS dataset (binary node type) plus the per-anchor groups
    # so the particle filter can drive dedo's TWO anchors with independent
    # velocities at rollout time.
    anchor_groups = [sorted(int(v) for v in deform.anchors[aid]['vertices']
                            if int(v) < num_verts)
                     for aid in deform.anchors]
    actuated = sorted({v for grp in anchor_groups for v in grp})

    if source == 'scripted':
        acts = _scripted_actions()
        if acts is None:
            return None
        # The stretched scripted demo is ~_demo_stretch x longer, so raise its
        # cap by the same factor (random keeps the plain --max_steps budget).
        cap = extra.max_steps * _demo_stretch
    else:
        acts = _random_action_stream(extra.max_steps)
        cap = extra.max_steps

    n_steps = min(len(acts), cap)
    deform.max_episode_len = n_steps + 5

    positions_seq, pcd_seq, debug_frames, valid_px_seq = [], [], [], []
    done = False
    truncated_at = None
    for t in range(n_steps):
        if done:
            break
        if t % extra.step_stride == 0:
            pos, pcd, frame, valid_px = _capture_step(
                build_debug=debug, hole_idx=hole_idx)
            # Stop at the first frame with no visible cloth. depth_to_pcd
            # returns ALL-ZEROS there, which downstream cannot distinguish from
            # a real observation. Truncating (rather than skipping) keeps the
            # stored frames consecutive, which `velocities` below depends on —
            # a skipped frame would make a 2-step displacement look like a
            # 1-step one and silently corrupt the dynamics targets.
            # Genuinely-occluded frames belong in the eval-time particle-filter
            # path (`step(point_cloud=None)`), not in the training tensors.
            if valid_px is not None and valid_px < extra.min_valid_px:
                truncated_at = len(positions_seq)
                break
            # Stop at the first frame where the SOLVER has gone unstable.
            # dedo's explicit springs diverge when the cloth is driven hard;
            # the mesh then inflates into a ball and never recovers, while the
            # arrays stay perfectly well-formed (right shape, no NaNs), so
            # nothing downstream can tell. Measured on the 2026-07-30
            # collection: 100% of random-action and 26% of scripted episodes
            # contained such frames, peaking at 22.7x rest edge length.
            # Truncate rather than skip, for the same reason as above.
            if (extra.max_edge_stretch > 0 and pos is not None
                    and pos.shape[0] == num_verts and len(edges)):
                _e = np.asarray(edges, dtype=np.int64)
                _rl = np.linalg.norm(rest_pos[_e[:, 0]] - rest_pos[_e[:, 1]],
                                     axis=-1)
                _ok = _rl > 1e-9
                _cur = np.linalg.norm(pos[_e[_ok, 0]] - pos[_e[_ok, 1]],
                                      axis=-1)
                _stretch = float((_cur / _rl[_ok]).max()) if _ok.any() else 0.0
                if not np.isfinite(pos).all() or _stretch > extra.max_edge_stretch:
                    print(f'    [guard] solver blow-up at frame '
                          f'{len(positions_seq)} (max edge stretch '
                          f'{_stretch:.1f}x) — truncating episode')
                    truncated_at = len(positions_seq)
                    break
            if pos is not None and pos.shape[0] == num_verts:
                positions_seq.append(pos)
                pcd_seq.append(pcd)
                valid_px_seq.append(valid_px)
                if frame is not None:
                    debug_frames.append(frame)
        _, _, done, _ = env.step(acts[t])

    if len(positions_seq) < 2:
        return None

    # Per-step velocities for GNS as the backward difference of the stored
    # frames (per-step displacement; step 0 = zeros). The GNS rollout advances
    # pos += vel * dt and overwrites grasped-node velocities directly, so with
    # the dedo GNS config dt=1.0 these displacement-units are self-consistent
    # between training data and rollout. Keep --step_stride=1 for dynamics so
    # consecutive frames are one env-step apart.
    pos_arr = np.stack(positions_seq).astype(np.float32)
    vel_arr = np.zeros_like(pos_arr)
    vel_arr[1:] = pos_arr[1:] - pos_arr[:-1]

    # Gripper position per step: the on-cloth centroid of the actuated vertices.
    # ClothDynamicsVariableDataset reads a (3,) `gripper_pos` per step group and
    # uses it as the action channel, so without this the dynamics model cannot
    # be trained on this dataset at all. Deriving it from the already-recorded
    # mesh (rather than querying the sim) keeps it exactly consistent with
    # `positions` and costs nothing. Same convention as the DexGarmentLab
    # pipeline: one 3-DoF gripper = the centroid of the grasped vertices.
    act_idx = np.asarray(actuated, dtype=np.int64)
    if act_idx.size == 0:
        return None
    gripper_seq = pos_arr[:, act_idx, :].mean(axis=1)  # (T, 3)
    # Per-anchor gripper positions, for the particle filter driving dedo's TWO
    # anchors with independent velocities (mirrors actuated_anchor_{i}).
    anchor_gripper_seq = [pos_arr[:, np.asarray(g, dtype=np.int64), :].mean(axis=1)
                          if len(g) else np.zeros((len(pos_arr), 3), np.float32)
                          for g in anchor_groups]

    return {
        'source': source,
        'rest_positions': rest_pos,
        'faces': faces,
        'edges': edges,
        'hole_vertex_indices': np.asarray(hole_idx, dtype=np.int64),
        'actuated_vertices': np.asarray(actuated, dtype=np.int64),
        'anchor_groups': [np.asarray(g, dtype=np.int64) for g in anchor_groups],
        'cam_yaw': cam_yaw,
        'randomization': rand_rec,
        'num_verts': num_verts,
        'positions': positions_seq,
        'velocities': [vel_arr[i] for i in range(len(vel_arr))],
        'gripper_pos': [gripper_seq[i] for i in range(len(gripper_seq))],
        'anchor_gripper_pos': anchor_gripper_seq,
        'pcds': pcd_seq,
        'valid_px': valid_px_seq,
        'truncated_at': truncated_at,
        'debug_frames': debug_frames,
    }


# ---------------------------------------------------------------------------
# Write one cloth group into the HDF5.
# ---------------------------------------------------------------------------
def write_cloth(split_group, cloth_idx, ep):
    cg = split_group.create_group(f'cloth_{cloth_idx:05d}')
    cg.attrs['hole_vertex_indices'] = ep['hole_vertex_indices']
    cg.attrs['source'] = ep['source']
    cg.attrs['num_verts'] = ep['num_verts']
    cg.attrs['cam_yaw'] = ep['cam_yaw']
    # Per-episode randomization, stored alongside the data so any downstream
    # analysis can condition on it without re-reading the stats JSON.
    for k, v in ep['randomization'].items():
        cg.attrs[f'rand_{k}'] = v
    cg.create_dataset('rest_positions', data=ep['rest_positions'],
                      compression='gzip')
    cg.create_dataset('faces', data=ep['faces'], compression='gzip')
    cg.create_dataset('edges', data=ep['edges'], compression='gzip')
    tg = cg.create_group('trajectory_0')
    # Flat actuated-vertex union for the GNS dataset (binary node type), under
    # the trajectory group where ClothDynamicsGraphDataset looks for it.
    tg.create_dataset('actuated_vertices', data=ep['actuated_vertices'],
                      compression='gzip')
    # Per-anchor groups (for the particle filter's 2-gripper rollout). Variable
    # length, so store one dataset per anchor + a count attr.
    tg.attrs['num_anchors'] = len(ep['anchor_groups'])
    for ai, grp in enumerate(ep['anchor_groups']):
        tg.create_dataset(f'actuated_anchor_{ai}', data=grp, compression='gzip')
    for s, (pos, pcd, vel, grip) in enumerate(
            zip(ep['positions'], ep['pcds'], ep['velocities'],
                ep['gripper_pos'])):
        sg = tg.create_group(f'step_{s:04d}')
        sg.create_dataset('positions', data=pos, compression='gzip')
        sg.create_dataset('velocities', data=vel, compression='gzip')
        # (3,) action channel read by ClothDynamicsVariableDataset.
        sg.create_dataset('gripper_pos', data=np.asarray(grip, dtype=np.float32))
        for ai, ag in enumerate(ep['anchor_gripper_pos']):
            sg.create_dataset(f'gripper_pos_anchor_{ai}',
                              data=np.asarray(ag[s], dtype=np.float32))
        sg.attrs['valid_cloth_px'] = int(ep['valid_px'][s])
        pg = sg.create_group('pointclouds')
        pg.create_dataset('cam_0', data=pcd, compression='gzip')


# ---------------------------------------------------------------------------
# Main collection loop.
# ---------------------------------------------------------------------------
print(f'\n=== state-est data collection ===')
print(f'  out:            {extra.out}')
print(f'  scripted/random:{extra.n_scripted}/{extra.n_random}')
print(f'  cam_resolution: {extra.cam_resolution}  pcd_n_points: {extra.pcd_n_points}')
print(f'  ctrl_freq:      {_ctrl_freq:.2f} Hz  max_steps: {extra.max_steps}'
      f'  stride: {extra.step_stride}')
print(f'  demo_speed:     {_actual_demo_speed:.3f}'
      f'{f" (scripted stretched {_demo_stretch}x)" if _demo_stretch > 1 else ""}'
      f'  max_act_vel: {extra.max_act_vel}\n')

plan = (['scripted'] * extra.n_scripted) + (['random'] * extra.n_random)
val_every = max(2, int(round(1.0 / extra.val_ratio))) if extra.val_ratio > 0 else 0

n_train = n_val = n_attempts = 0
max_v_seen = max_e_seen = 0
val_idx = train_idx = 0
start_time = time.time()

with h5py.File(extra.out, 'w') as h5:
    train_grp = h5.create_group('training')
    val_grp = h5.create_group('validation')
    h5.attrs['cam_viewmat'] = np.asarray(extra.cam_viewmat, dtype=np.float32)
    h5.attrs['pcd_n_points'] = int(extra.pcd_n_points)

    kept = 0
    dbg_written = {'scripted': 0, 'random': 0}
    ep_stats = []
    guard_px = []
    for want in plan:
        n_attempts += 1
        debug_this = dbg_written[want] < extra.debug_video_first_n
        ep = run_episode(want, debug=debug_this)
        if ep is None:
            continue
        # One row per kept episode: what was randomized, and what the data
        # actually looks like under it. Feeds the randomization/coverage plots.
        _npx = np.asarray(ep['valid_px'], dtype=np.float32)
        ep_stats.append({
            'source': ep['source'],
            'split': 'val' if (val_every and kept % val_every == 0) else 'train',
            'num_verts': int(ep['num_verts']),
            'num_edges_bidir': int(2 * ep['edges'].shape[0]),
            'num_steps': len(ep['positions']),
            'hole_verts': int(ep['hole_vertex_indices'].size),
            'actuated_verts': int(ep['actuated_vertices'].size),
            'truncated_at': ep['truncated_at'],
            'valid_px_mean': float(_npx.mean()),
            'valid_px_min': float(_npx.min()),
            'valid_px_max': float(_npx.max()),
            'pcd_coverage_mean': float(min(1.0, _npx.mean() / extra.pcd_n_points)),
            'gripper_path_len': float(np.abs(np.diff(
                np.stack(ep['gripper_pos']), axis=0)).sum()),
            **ep['randomization'],
        })
        # Fail fast on degenerate observations. depth_to_pcd returns all-zeros
        # when no pixel survives the mask, so an empty cloud is indistinguishable
        # from a valid one downstream (still (n_points, 3) float32) — a 90-minute
        # collection would finish "successfully" with unusable data. Same class
        # of silent failure as the PhysTwin z-flip: every other metric looks fine.
        if kept < extra.guard_episodes:
            _px = np.asarray(ep['valid_px'], dtype=np.float32)
            guard_px.append(_px)
            if kept == extra.guard_episodes - 1:
                _all = np.concatenate(guard_px)
                _frac_zero = float((_all == 0).mean())
                _med = float(np.median(_all))
                print(f'[guard] over first {extra.guard_episodes} kept episodes '
                      f'({_all.size} frames): median {_med:.0f} cloth px, '
                      f'{_frac_zero * 100:.0f}% of frames empty')
                if _frac_zero > extra.guard_max_empty:
                    raise SystemExit(
                        f'ABORT: {_frac_zero * 100:.0f}% of frames have ZERO valid '
                        f'cloth pixels (limit {extra.guard_max_empty * 100:.0f}%).\n'
                        f'  env={extra.env}  cam_viewmat={extra.cam_viewmat}  '
                        f'res={extra.cam_resolution}\n'
                        f'  depth_to_pcd returns ALL-ZEROS for an empty frame, and '
                        f'that is indistinguishable from real data downstream — the '
                        f'whole run would look successful and be unusable.\n'
                        f'  The camera is probably framing empty space. Drop '
                        f'--cam_viewmat for the per-env preset, or narrow '
                        f'--pitch_range / --dist_range.')
                if _med < extra.pcd_n_points:
                    print(f'[guard] WARNING: median {_med:.0f} cloth px < '
                          f'--pcd_n_points {extra.pcd_n_points}, so stored clouds '
                          f'are mostly DUPLICATED points (depth_to_pcd samples '
                          f'with replacement). Raise --cam_resolution or lower '
                          f'--pcd_n_points.')
        max_v_seen = max(max_v_seen, ep['num_verts'])
        max_e_seen = max(max_e_seen, 2 * ep['edges'].shape[0])  # bidirectional
        to_val = val_every and (kept % val_every == 0)
        if to_val:
            write_cloth(val_grp, val_idx, ep)
            val_idx += 1
            n_val += 1
        else:
            write_cloth(train_grp, train_idx, ep)
            train_idx += 1
            n_train += 1
        kept += 1
        if debug_this and ep['debug_frames']:
            vid = (os.path.splitext(extra.out)[0]
                   + f'_simrgb_{want}_{dbg_written[want]:02d}.mp4')
            write_video_mp4(ep['debug_frames'], vid, fps=extra.debug_fps)
            dbg_written[want] += 1
            print(f'  [debug] wrote {vid} ({len(ep["debug_frames"])} frames)')
        if kept % 10 == 0:
            print(f'  [{kept}/{len(plan)}] kept (train={n_train} val={n_val}) '
                  f'last={want} steps={len(ep["positions"])} V={ep["num_verts"]}')

print(f'\nDone. attempts={n_attempts} kept={n_train + n_val} '
      f'(train={n_train}, val={n_val})')
print(f'  max vertices seen : {max_v_seen}  -> set config max_num_nodes >= '
      f'{max_v_seen}')
print(f'  max bidir edges   : {max_e_seen}  -> set config max_num_edges >= '
      f'{max_e_seen}')
_kept = n_train + n_val
_elapsed = time.time() - start_time
_mins, _secs = divmod(int(_elapsed), 60)
print(f'  elapsed           : {_mins}m{_secs:02d}s '
      f'({_elapsed / max(_kept, 1):.1f}s/episode over {_kept} kept)')
print(f'  wrote {extra.out}')

# Statistics dump: per-episode randomization + data-shape rows, plus the run
# config. plot_dataset_stats.py turns this into the figures that go in Notion.
_stats_path = (extra.stats_json if extra.stats_json
               else os.path.splitext(extra.out)[0] + '_stats.json')
with open(_stats_path, 'w') as f:
    json.dump({
        'config': {
            'env': extra.env,
            'out': extra.out,
            'n_scripted': extra.n_scripted,
            'n_random': extra.n_random,
            'attempts': n_attempts,
            'kept': _kept,
            'n_train': n_train,
            'n_val': n_val,
            'max_verts_seen': int(max_v_seen),
            'max_bidir_edges_seen': int(max_e_seen),
            'cam_resolution': extra.cam_resolution,
            'pcd_n_points': extra.pcd_n_points,
            'cam_viewmat_nominal': list(extra.cam_viewmat),
            'ctrl_freq': _ctrl_freq,
            'max_act_vel': extra.max_act_vel,
            'demo_speed': _actual_demo_speed,
            'step_stride': extra.step_stride,
            'randomize': {
                'yaw': [extra.randomize_yaw, extra.yaw_range],
                'pitch': [extra.randomize_pitch, extra.pitch_range],
                'dist': [extra.randomize_dist, extra.dist_range],
                'physics': [extra.randomize_physics, {
                    'jitter': extra.physics_jitter,
                    'elastic_clamp': extra.elastic_clamp,
                    'scene_baseline': _BASE_PHYS}],
                'goal_radius': extra.randomize_goal_radius,
                'goal_dz': extra.randomize_goal_dz,
                'min_valid_px': extra.min_valid_px,
                'depth_noise_std': extra.depth_noise_std,
                'depth_dropout': extra.depth_dropout,
            },
            'elapsed_s': _elapsed,
            'seed': extra.seed,
        },
        'episodes': ep_stats,
    }, f, indent=1)
print(f'  wrote {_stats_path} ({len(ep_stats)} episode rows)')
env.close()
