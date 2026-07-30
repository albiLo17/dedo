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
      --cam_resolution 128 --pcd_n_points 2048
"""
import argparse
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
parser.add_argument('--cam_resolution', type=int, default=128,
                    help='RGB+depth image height/width. 128 gives denser PCD '
                         'than the 96 used for BC demos — more points to '
                         'reconstruct the mesh from.')
parser.add_argument('--pcd_n_points', type=int, default=2048,
                    help='Points sampled from the back-projected depth buffer. '
                         '2048 matches the model num_sample_points; collect '
                         'dense here regardless of the policy PCD budget.')
parser.add_argument('--cam_viewmat', type=float, nargs=6,
                    default=[14.0, -5.0, 45.0, 0.0, 0.0, 5.5],
                    help='dedo cam_viewmat: dist pitch yaw tx ty tz. Default '
                         'matches collect_bc_demos.py so the estimator sees '
                         'the same viewpoint as the policy. When --randomize_yaw '
                         'is set, the yaw entry is overwritten per episode.')
parser.add_argument('--randomize_yaw', action='store_true',
                    help='Sample the camera YAW uniformly in --yaw_range each '
                         'episode (other cam params fixed). Gives the state '
                         'estimator varied visible cloth subsets / occlusion '
                         'patterns in ONE dataset — viewpoint augmentation, not '
                         'conflicting data, since the model works on a centered '
                         '3-D point cloud rather than a 2-D image.')
parser.add_argument('--yaw_range', type=float, nargs=2, default=[0.0, 90.0],
                    help='[min, max] degrees for --randomize_yaw. Default 0..90.')
parser.add_argument('--randomize_goal_radius', type=float, default=0.0,
                    help='Match the demo dir setting for env parity.')
parser.add_argument('--sim_freq', type=int, default=500)
parser.add_argument('--ctrl_freq', type=float, default=15.0)
# Random-motion controls.
parser.add_argument('--rand_amp', type=float, default=0.35,
                    help='Random-pass action amplitude (in normalized [-1,1] '
                         'action units before MAX_ACT_VEL scaling).')
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
    '--env=HangProcCloth-v1',
    f'--cam_resolution={extra.cam_resolution}',
    '--num_envs=0',
    '--total_env_steps=0',
    '--seed', str(extra.seed),
    '--max_episode_len', str(max(extra.max_steps * 2, 400)),
    f'--sim_freq={extra.sim_freq}',
    f'--sim_steps_per_action={_steps_per_action}',
    '--cam_viewmat', *[str(x) for x in extra.cam_viewmat],
    f'--randomize_goal_radius={extra.randomize_goal_radius}',
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


def _capture_step(build_debug=False, hole_idx=None):
    """Return (positions (V,3), pcd (P,3), debug_frame) at the current sim
    state. positions/pcd are None if the mesh has NaN verts. debug_frame is a
    3-panel uint8 image (sim render | pcd-on-RGB | pcd camera view) when
    build_debug, else None."""
    _, verts = get_mesh_data(deform.sim, deform.deform_id)
    positions = np.asarray(verts, dtype=np.float32)
    if not np.isfinite(positions).all():
        return None, None, None
    rgb, depth, seg, view, proj = capture_rgb_depth(
        deform, extra.cam_resolution, extra.cam_resolution)
    pcd = cloth_only_pcd(depth, seg, view, proj, deform.deform_id,
                         extra.pcd_n_points)
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
    return positions, pcd, frame


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
    env.reset()
    # Per-episode camera yaw randomization. deform._cam_viewmat is a property
    # recomputed from deform.args.cam_viewmat on every access, so overwriting
    # the yaw entry here takes effect for all captures/renders this episode.
    cam_yaw = float(deform.args.cam_viewmat[2])
    if extra.randomize_yaw:
        cam_yaw = float(np.random.uniform(*extra.yaw_range))
        vm = list(deform.args.cam_viewmat)
        vm[2] = cam_yaw
        deform.args.cam_viewmat = vm
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

    positions_seq, pcd_seq, debug_frames = [], [], []
    done = False
    for t in range(n_steps):
        if done:
            break
        if t % extra.step_stride == 0:
            pos, pcd, frame = _capture_step(build_debug=debug, hole_idx=hole_idx)
            if pos is not None and pos.shape[0] == num_verts:
                positions_seq.append(pos)
                pcd_seq.append(pcd)
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

    return {
        'source': source,
        'rest_positions': rest_pos,
        'faces': faces,
        'edges': edges,
        'hole_vertex_indices': np.asarray(hole_idx, dtype=np.int64),
        'actuated_vertices': np.asarray(actuated, dtype=np.int64),
        'anchor_groups': [np.asarray(g, dtype=np.int64) for g in anchor_groups],
        'cam_yaw': cam_yaw,
        'num_verts': num_verts,
        'positions': positions_seq,
        'velocities': [vel_arr[i] for i in range(len(vel_arr))],
        'pcds': pcd_seq,
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
    for s, (pos, pcd, vel) in enumerate(
            zip(ep['positions'], ep['pcds'], ep['velocities'])):
        sg = tg.create_group(f'step_{s:04d}')
        sg.create_dataset('positions', data=pos, compression='gzip')
        sg.create_dataset('velocities', data=vel, compression='gzip')
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
    for want in plan:
        n_attempts += 1
        debug_this = dbg_written[want] < extra.debug_video_first_n
        ep = run_episode(want, debug=debug_this)
        if ep is None:
            continue
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
env.close()
