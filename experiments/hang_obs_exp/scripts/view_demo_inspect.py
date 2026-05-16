"""
view_demo_inspect.py — Viser-based interactive inspector for the
HangProcCloth scripted demo.

Built for sim-to-real workspace calibration (Path B in the sim-to-real
plan): inspect where gripper anchors actually go during a scripted
demo and what velocities are commanded, so you can size the deploy-
time normalization wrapper that maps your real robot's <1m workspace
to dedo's coordinate frame.

Architecture
------------
  - PyBullet runs HEADLESS (DIRECT mode) as the physics engine.
  - Viser serves the 3D viz + GUI at http://localhost:8080.
  - Open that URL in your browser after starting the script.

What you see in the browser
---------------------------
  - World coordinate frame at the origin (Viser's R=+x, G=+y, B=+z).
  - Ground grid in the xy plane (1 m cells, 5 m sections).
  - Cloth surface, updated live from PyBullet's softbody state.
  - Per-anchor coordinate frames at each gripper's live position.
  - Velocity arrows (orange = anchor a, purple = anchor b) from each
    gripper, length = commanded velocity * (lookahead seconds).
  - A yellow sphere at the hanger goal position.
  - GUI panel (right edge) with Pause/Play/Step/Reset + sliders + live
    anchor pose / velocity readouts.

Reproducibility
---------------
Every reset re-seeds `np.random` to `--seed` so the procedural cloth
(dimensions, hole position) and resulting scripted trajectory are
byte-identical across runs and across Reset clicks. Pick a `--seed`
that produces a nice demo and stick with it — that's exactly what
will play back on your real robot.

Usage
-----
  pip install viser trimesh                      # one-time
  python experiments/hang_obs_exp/scripts/view_demo_inspect.py
  python experiments/hang_obs_exp/scripts/view_demo_inspect.py --seed 42
"""
import sys, os, time, argparse, threading, io, pickle, copy, socket
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT))

import numpy as np
import gym

import viser
from viser.extras import ViserUrdf
import trimesh

import dedo  # noqa: F401
from dedo.utils.args import get_args_parser, args_postprocess
from dedo.envs.deform_env import DeformEnv
from dedo.demo_preset import build_traj, merge_traj
from dedo.utils.mesh_utils import get_mesh_data
from dedo.utils.task_info import SCENE_INFO

from experiments.hang_obs_exp.envs.privileged_env import PrivilegedObsWrapper
sys.path.insert(0, str(Path(__file__).resolve().parent))
from _helpers import RetryResetEnv, build_hole_aware_waypoints  # noqa: E402


# --------------------------------------------------------------------------
# args
# --------------------------------------------------------------------------
parser = argparse.ArgumentParser()
parser.add_argument('--seed', type=int, default=42)
parser.add_argument('--port', type=int, default=8080,
                    help='Viser HTTP port (default 8080)')
parser.add_argument('--max_episode_len', type=int, default=2000,
                    help='Override env max_episode_len so you can pause '
                         'without the episode auto-ending.')
parser.add_argument('--obs_mode', type=str, default='hole_centroid',
                    choices=['hole_centroid', 'hole_centroid_corners',
                             'hole_vertices', 'full_mesh'])
parser.add_argument('--success_factor', type=float, default=1.2)
parser.add_argument(
    '--real_traj', type=str, default=None,
    # default=str(REPO_ROOT / 'experiments' / 'hang_obs_exp'
                # / 'real_world_gripper_trajectories.npz'),
    help='Path to an npz with keys "ee_pos_left", "ee_pos_right" (each '
         '(T, 3)) — real-world gripper poses in the Franka world frame. '
         'Visualized under /real_workspace/. Set to empty string to skip.')
parser.add_argument(
    '--export_traj', type=str, default='',
    help='Path to a pickled exported trajectory list-of-dicts to visualize '
         'in the world frame. If set, left/right EE paths are rendered.')
parser.add_argument('--sim_scale', type=float, default=0.045,
                    help='Visualization-only uniform scale factor applied '
                         'to the entire sim scene (cloth, hanger, rod, '
                         'anchors, world axes). 1.0 = sim native scale. '
                         '~0.05-0.10 = roughly matched to a sub-meter '
                         'Franka workspace. Changeable live via the GUI '
                         'slider; this flag just sets the initial value.')
parser.add_argument('--sim_offset_x', type=float, default=0.5,
                    help='Initial x offset of the sim scene in viz '
                         '(meters). Live-editable via GUI slider.')
parser.add_argument('--sim_offset_y', type=float, default=0.0,
                    help='Initial y offset of the sim scene in viz (m).')
parser.add_argument('--sim_offset_z', type=float, default=-0.038999999999999924,
                    help='Initial z offset of the sim scene in viz (m).')
parser.add_argument('--sim_yaw_deg', type=float, default=90,
                    help='Initial rotational offset of the sim scene '
                         'about the world +z axis, in degrees. -90 maps '
                         "sim's +y (forward) to franka's +x (forward).")
parser.add_argument('--real_goal', type=float, nargs=3,
                    default=[0.50, 0.00, 0.33],
                    help='Goal position in the real-world (Franka world) '
                         'frame, plotted as a magenta sphere with the '
                         'label "goal position in real world".')
parser.add_argument('--export_pcd', dest='export_pcd', action='store_true',
                    default=True,
                    help='Include a pointcloud (transformed into the real-'
                         'world frame) in each step of the downloaded '
                         'trajectory. Use --no_export_pcd to skip.')
parser.add_argument('--no_export_pcd', dest='export_pcd',
                    action='store_false')
parser.add_argument('--export_pcd_width', type=int, default=100,
                    help='Depth-image resolution (W=H pixels) for the '
                         'pointcloud renderer. Higher = denser pcd, slower.')
parser.add_argument('--export_pcd_max_points', type=int, default=2048,
                    help='Random subsample each step pcd down to this many '
                         'points before adding to the trajectory. 0 = keep all.')
# Live ZED real-camera pointcloud (from scripts/zed_pcd_publisher.py
# running on the ZED host). Rendered under /real_workspace and posed
# interactively via GUI sliders to align with the sim cloud.
parser.add_argument('--zed_stream', type=str, default='',
                    help='HOST:PORT of a zed_pcd_publisher TCP stream '
                         '(e.g. 192.168.1.50:5556). Empty = disabled.')
parser.add_argument('--zed_pcd_file', type=str, default='',
                    help='Path to a .npy the zed_pcd_publisher rewrites '
                         'atomically (--mode file). Tailed by mtime. '
                         'Used only if --zed_stream is empty.')
parser.add_argument('--zed_max_points', type=int, default=20000,
                    help='Viewer-side safety subsample of each ZED frame.')
parser.add_argument('--zed_cam_x', type=float, default=0.0,
                    help='Initial ZED cloud pose in the Franka world '
                         'frame (m). Live-editable via GUI slider.')
parser.add_argument('--zed_cam_y', type=float, default=0.0)
parser.add_argument('--zed_cam_z', type=float, default=0.0)
parser.add_argument('--zed_cam_yaw', type=float, default=0.0,
                    help='Initial ZED cloud yaw/pitch/roll (deg), applied '
                         'Rz@Ry@Rx, before the xyz offset.')
parser.add_argument('--zed_cam_pitch', type=float, default=0.0)
parser.add_argument('--zed_cam_roll', type=float, default=0.0)
# Camera-frame trajectory overlay. Same sim-format pkl schema as
# --export_traj but with obs in the sim CAMERA frame. Rendered under a
# /sim_scene/cam frame at cam_pos*sim_scale (so the camera origin scales
# + moves with the sliders, as expected) with rotation cam_wxyz; this
# composes to the SAME world transform as the sim-world --export_traj,
# so a correct camframe conversion overlays it exactly at any sim_scale.
parser.add_argument('--cam_traj', type=str, default='',
                    help='Path to a camera-frame rollout pkl to overlay '
                         '(verifies the sim-world -> camera-frame '
                         'transform while you calibrate to real).')
parser.add_argument('--cam_pos', type=float, nargs=3,
                    default=[9.8618, -9.8618, 6.7202],
                    help='Sim camera origin in sim-world coords.')
parser.add_argument('--cam_wxyz', type=float, nargs=4,
                    default=[-0.6242, 0.6812, 0.2821, -0.2585],
                    help='Sim camera orientation, viser wxyz quaternion '
                         '(= R_cam_to_world).')
extra = parser.parse_args()

# Build dedo args. cam_resolution=0 forces low-dim grip obs (the wrapper
# expects this); --viz is OFF so PyBullet runs headless (we render via
# Viser, not PyBullet's native window).
sys.argv = [
    'view_demo_inspect',
    '--env=HangProcCloth-v1',
    '--cam_resolution', '0',
    '--num_envs=0',
    '--total_env_steps=0',
    '--seed', str(extra.seed),
]
args, _ = get_args_parser()
args_postprocess(args)
args.viz = False
args.debug = False
if extra.max_episode_len is not None:
    args.max_episode_len = extra.max_episode_len

sf = None if extra.success_factor < 0 else float(extra.success_factor)


# --------------------------------------------------------------------------
# env construction + deterministic reset
# --------------------------------------------------------------------------
env = gym.make(args.env, args=args)
env = RetryResetEnv(env)
env = PrivilegedObsWrapper(env, obs_mode=extra.obs_mode, success_factor=sf)


def fixed_reset():
    """Reseed numpy + env to FIXED_SEED before every reset, so the
    procedural cloth and resulting scripted trajectory are byte-
    identical across runs and across user-triggered Reset clicks."""
    np.random.seed(extra.seed)
    env.seed(extra.seed)
    return env.reset()


fixed_reset()
print(f'[inspect] FIXED DEMO seed={extra.seed} — every reset reproduces '
      f'the same cloth + trajectory.')

underlying = env
while hasattr(underlying, 'env'):
    underlying = underlying.env
    if isinstance(underlying, DeformEnv):
        break

sim = underlying.sim
ctrl_freq = args.sim_freq / args.sim_steps_per_action


def _build_traj():
    """Build the scripted velocity trajectory for the current cloth.
    Returns (T, 6) ndarray, or None if cloth has no usable hole."""
    wp = build_hole_aware_waypoints(underlying)
    if wp is None:
        return None
    _, va = build_traj(underlying, wp, 'a', anchor_idx=0,
                       ctrl_freq=ctrl_freq, robot=None)
    _, vb = build_traj(underlying, wp, 'b', anchor_idx=1,
                       ctrl_freq=ctrl_freq, robot=None)
    return merge_traj(va, vb)


def _load_cloth_faces():
    """Load procedural cloth OBJ to get triangle face indices. PyBullet's
    getMeshData returns vertex positions but not triangulation, so we
    cache faces from the OBJ that dedo procedurally wrote.

    Assumes (verified true for dedo's procedural_utils.create_cloth_obj)
    that the OBJ vertex order matches PyBullet's softbody vertex order."""
    obj_path = underlying.args.deform_obj
    mesh = trimesh.load(obj_path, process=False, force='mesh')
    return np.asarray(mesh.faces, dtype=np.int32)


traj = _build_traj()
assert traj is not None, (
    f'seed={extra.seed} produced a cloth with no hole loop; pick a '
    f'different --seed.')
T = len(traj)
cloth_faces = _load_cloth_faces()


def _rigid_body_aabb(body_id):
    """Union AABB across all links of a multi-link PyBullet rigid body.
    Returns (min_xyz, max_xyz) as float32 ndarrays."""
    lo, hi = sim.getAABB(body_id, -1)
    lo = np.array(lo, dtype=np.float32)
    hi = np.array(hi, dtype=np.float32)
    for link_idx in range(sim.getNumJoints(body_id)):
        clo, chi = sim.getAABB(body_id, link_idx)
        lo = np.minimum(lo, clo)
        hi = np.maximum(hi, chi)
    return lo, hi


def _fmt_extents(ext):
    """Format an (x, y, z) extent triple. Switches between meters and cm
    based on the largest dim, so 0.05 m shows as "5.0 cm" and 1.3 m
    shows as "1.30 m"."""
    mx = float(max(ext))
    if mx >= 0.5:
        return f'{ext[0]:.2f} × {ext[1]:.2f} × {ext[2]:.2f} m'
    return f'{ext[0]*100:.1f} × {ext[1]*100:.1f} × {ext[2]*100:.1f} cm'

print(f'\n[inspect] sim constants:')
print(f'    MAX_ACT_VEL        = {DeformEnv.MAX_ACT_VEL:.1f} m/s')
print(f'    MAX_OBS_VEL        = {DeformEnv.MAX_OBS_VEL:.1f} m/s')
print(f'    WORKSPACE_BOX_SIZE = {DeformEnv.WORKSPACE_BOX_SIZE:.1f} m')
print(f'    ctrl_freq          = {ctrl_freq:.1f} Hz')
print(f'[inspect] trajectory: T={T} actions, duration={T/ctrl_freq:.2f} s, '
      f'peak |vel|={float(np.abs(traj).max()):.3f} m/s')

# Sim "where the cloth starts" reference — useful for picking offset
# + yaw values to align with the real robot. All in sim-native meters
# (multiply by --sim_scale to compare against franka coords).
_anc0_init = sim.getBasePositionAndOrientation(underlying.anchor_ids[0])[0]
_anc1_init = sim.getBasePositionAndOrientation(underlying.anchor_ids[1])[0]
_goal = underlying.goal_pos[0]
print(f'\n[inspect] sim "where things start" (sim-native meters):')
print(f'    hanger goal (peg)  = [{_goal[0]:+.2f}, {_goal[1]:+.2f}, {_goal[2]:+.2f}]')
print(f'    anchor a0 init     = [{_anc0_init[0]:+.2f}, {_anc0_init[1]:+.2f}, {_anc0_init[2]:+.2f}]')
print(f'    anchor a1 init     = [{_anc1_init[0]:+.2f}, {_anc1_init[1]:+.2f}, {_anc1_init[2]:+.2f}]')
print(f'    sim "forward" is +y (cloth starts at +y=5, moves toward y=0 '
      f'to thread the peg). franka "forward" is +x. Try --sim_yaw_deg=-90.')


# --------------------------------------------------------------------------
# Viser server + scene
# --------------------------------------------------------------------------
server = viser.ViserServer(host='0.0.0.0', port=extra.port)
print(f'\n[inspect] Viser running — open http://localhost:{extra.port}\n')


def anchor_pos(idx):
    """Live anchor body position from PyBullet."""
    aid = list(underlying.anchors.keys())[idx]
    pos, _ = sim.getBasePositionAndOrientation(aid)
    return np.array(pos, dtype=np.float32)


def cloth_verts():
    _, v = get_mesh_data(sim, underlying.deform_id)
    return np.asarray(v, dtype=np.float32)


def anchor_pose(idx, env_obj=None):
    """Return (pos, orn) for an anchor body in sim coords."""
    if env_obj is None:
        env_obj = underlying
    aid = list(env_obj.anchors.keys())[idx]
    pos, orn = env_obj.sim.getBasePositionAndOrientation(aid)
    return np.asarray(pos, dtype=np.float64), np.asarray(orn, dtype=np.float64)


def _sim_to_world_transform():
    s = float(sld_sim_scale.value)
    yaw = np.radians(float(sld_sim_yaw.value))
    c, sn = np.cos(yaw), np.sin(yaw)
    R = np.array([[c, -sn, 0.0], [sn, c, 0.0], [0.0, 0.0, 1.0]], dtype=np.float64)
    offset = np.asarray((float(sld_sim_x.value),
                         float(sld_sim_y.value),
                         float(sld_sim_z.value)), dtype=np.float64)
    return offset, R, float(s)


def _sim_point_to_world(p, offset=None, R=None, s=None):
    if offset is None or R is None or s is None:
        offset, R, s = _sim_to_world_transform()
    return offset + R.dot(np.asarray(p, dtype=np.float64) * s)


def _sim_vec_to_world(v, R=None, s=None):
    if R is None or s is None:
        _, R, s = _sim_to_world_transform()
    return R.dot(np.asarray(v, dtype=np.float64) * s)


def _quat_mul(q1, q2):
    w1, x1, y1, z1 = q1
    w2, x2, y2, z2 = q2
    return np.asarray([
        w1*w2 - x1*x2 - y1*y2 - z1*z2,
        w1*x2 + x1*w2 + y1*z2 - z1*y2,
        w1*y2 - x1*z2 + y1*w2 + z1*x2,
        w1*z2 + x1*y2 - y1*x2 + z1*w2,
    ], dtype=np.float64)


def _build_export_step_dict(step_idx, left_pos_sim, left_orn_sim,
                             left_vel_sim, right_pos_sim,
                             right_orn_sim, right_vel_sim,
                             pcd_sim=None, pcd_ids=None):
    offset, R, s = _sim_to_world_transform()
    yaw = np.radians(float(sld_sim_yaw.value))
    yaw_quat = np.asarray((np.cos(yaw / 2.0), 0.0, 0.0,
                           np.sin(yaw / 2.0)), dtype=np.float64)

    left_pos_world = _sim_point_to_world(left_pos_sim, offset, R, s)
    right_pos_world = _sim_point_to_world(right_pos_sim, offset, R, s)
    left_vel_world = _sim_vec_to_world(left_vel_sim, R, s)
    right_vel_world = _sim_vec_to_world(right_vel_sim, R, s)
    left_action_pos = left_pos_world + left_vel_world / float(ctrl_freq)
    right_action_pos = right_pos_world + right_vel_world / float(ctrl_freq)

    left_orn_world = _quat_mul(yaw_quat, left_orn_sim)
    right_orn_world = _quat_mul(yaw_quat, right_orn_sim)

    step = {
        't': float(step_idx) / float(ctrl_freq),
        'action_gripper_left': 0.0,
        'action_gripper_right': 0.0,
        'action_ori_left': left_orn_world,
        'action_ori_right': right_orn_world,
        'action_pos_left': left_action_pos,
        'action_pos_right': right_action_pos,
        'action_qpos_left': np.zeros((7,), dtype=np.float64),
        'action_qpos_right': np.zeros((7,), dtype=np.float64),
        'ee_ori_left': left_orn_world,
        'ee_ori_right': right_orn_world,
        'ee_pos_left': left_pos_world,
        'ee_pos_right': right_pos_world,
        'gripper_pos_left': [3.0],
        'gripper_pos_right': [3.0],
        'qpos_left': np.zeros((7,), dtype=np.float64),
        'qpos_right': np.zeros((7,), dtype=np.float64),
        'robot_ft_left': np.zeros((6,), dtype=np.float64),
        'robot_ft_right': np.zeros((6,), dtype=np.float64),
    }

    if pcd_sim is not None and len(pcd_sim) > 0:
        # Transform the pcd from sim world into the real-world frame
        # using the same offset/R/sim_scale as everything else.
        pcd_world = (np.asarray(pcd_sim, dtype=np.float64) * s) @ R.T + offset
        step['pcd'] = pcd_world.astype(np.float32)
        if pcd_ids is not None:
            step['pcd_ids'] = np.asarray(pcd_ids, dtype=np.int32).reshape(-1)
    return step


def _build_export_trajectory():
    """Replay the current sim trajectory on a fresh env and return a
    world-frame list-of-dicts matching the sample trajectory file format.
    Optionally captures a transformed pointcloud per step (--export_pcd)."""
    from copy import deepcopy
    replay_args = copy.deepcopy(args)
    replay_args.seed = extra.seed
    # Enable pcd mode on the replay env so DeformEnv.__init__ builds
    # self.camera_config + self.object_ids from --cam_config_path.
    if extra.export_pcd:
        replay_args.pcd = True
    replay_env = gym.make(replay_args.env, args=replay_args)
    replay_env = PrivilegedObsWrapper(replay_env, obs_mode=extra.obs_mode,
                                     success_factor=sf)

    np.random.seed(extra.seed)
    replay_env.seed(extra.seed)
    replay_env.reset()

    underlying_env = replay_env
    while hasattr(underlying_env, 'env'):
        underlying_env = underlying_env.env
        if isinstance(underlying_env, DeformEnv):
            break

    pcd_enabled = extra.export_pcd and hasattr(underlying_env, 'camera_config')
    if extra.export_pcd and not pcd_enabled:
        print('[inspect] WARN: --export_pcd requested but replay env has no '
              'camera_config; skipping pointcloud capture.')

    # Import unconditionally (dedo is already a top-level import; this
    # also makes the conditional ProcessCamera.render() call safe for
    # static type checkers).
    from dedo.utils.process_camera import ProcessCamera
    pcd_W = int(extra.export_pcd_width)
    pcd_max = int(extra.export_pcd_max_points)
    # dedo mutates rigid_ids when populating object_ids — use a fresh
    # list so we don't depend on that quirk.
    pcd_obj_ids: list = (
        list(underlying_env.rigid_ids) + [underlying_env.deform_id]
        if pcd_enabled else []
    )

    traj_steps = []
    for i in range(len(traj)):
        # PCD captured BEFORE stepping, so it matches the same env state
        # that produces the recorded pose for this step.
        pcd_sim = pcd_ids = None
        if pcd_enabled:
            try:
                pcd_sim, pcd_ids = ProcessCamera.render(
                    underlying_env.sim, underlying_env.camera_config,
                    width=pcd_W, height=pcd_W,
                    object_ids=list(pcd_obj_ids),
                )
                if pcd_max > 0 and len(pcd_sim) > pcd_max:
                    sub = np.random.permutation(len(pcd_sim))[:pcd_max]
                    pcd_sim = pcd_sim[sub]
                    pcd_ids = pcd_ids[sub]
            except Exception as e:
                if i == 0:
                    print(f'[inspect] WARN: pcd capture failed: {e!r}')
                pcd_sim = pcd_ids = None

        # Anchor 0 → right EE, anchor 1 → left EE (matches sample format).
        right_pos_sim, right_orn_sim = anchor_pose(0, env_obj=underlying_env)
        left_pos_sim, left_orn_sim = anchor_pose(1, env_obj=underlying_env)
        right_vel_sim = np.asarray(traj[i, :3], dtype=np.float64)
        left_vel_sim = np.asarray(traj[i, 3:], dtype=np.float64)
        traj_steps.append(_build_export_step_dict(
            i, left_pos_sim, left_orn_sim, left_vel_sim,
            right_pos_sim, right_orn_sim, right_vel_sim,
            pcd_sim=pcd_sim, pcd_ids=pcd_ids))
        normalized = np.clip(traj[i] / DeformEnv.MAX_ACT_VEL, -1.0, 1.0)
        replay_env.step(normalized.astype(np.float32))

    try:
        replay_env.close()
    except Exception:
        pass
    return traj_steps


def goal_position():
    return np.asarray(underlying.goal_pos[0], dtype=np.float32)


# Parent frame for grouping all sim-side scene nodes under /sim_scene/...
# (Viser 1.0.27's frame-level `scale` parameter is broken — it doesn't
# propagate to children — so we scale coordinates manually in Python
# instead. See `_S()` and `_pos()` below.)
sim_scene_root = server.scene.add_frame(
    '/sim_scene', show_axes=False,
    position=(0.0, 0.0, 0.0),
)


def _S():
    """Current sim-to-viz scale factor (read from the GUI slider)."""
    return float(sld_sim_scale.value)


def _pos(p):
    """Scale a position vector from sim coords to viz coords."""
    s = _S()
    return tuple(float(x) * s for x in p)


def _scl(v):
    """Scale a scalar length/radius from sim coords to viz coords."""
    return float(v) * _S()


# Track the last sim_scale we applied to one-shot scene nodes (grid,
# URDFs), so we know when to rebuild them on slider change.
_last_applied_sim_scale = None


# Hanger + tallrod via ViserUrdf. Both are static rigid bodies — their
# poses come from SCENE_INFO['hangcloth']. We multiply base_pos and the
# URDF's globalScaling by the current sim_scale, then rebuild whenever
# the user moves the slider (URDFs can't have their transform changed
# in-place, so we remove + re-add).
_hangcloth_scene = SCENE_INFO['hangcloth']['entities']
_hanger_info = _hangcloth_scene['urdf/hanger.urdf']
_rod_info    = _hangcloth_scene['urdf/tallrod.urdf']

_static_urdf_handles = {}  # name -> ViserUrdf handle


def _load_static_urdf(name, urdf_relpath, base_pos, base_scale,
                       color, sim_s):
    urdf_abs = Path(underlying.args.data_path) / urdf_relpath
    if not urdf_abs.exists():
        print(f'[inspect] WARN: {urdf_abs} not found; skipping {name}')
        return None
    scaled_pos = tuple(float(x) * sim_s for x in base_pos)
    scaled_scale = float(base_scale) * sim_s
    server.scene.add_frame(
        f'/sim_scene/{name}', position=scaled_pos, show_axes=False,
    )
    try:
        urdf = ViserUrdf(
            server, urdf_abs, scale=scaled_scale,
            root_node_name=f'/sim_scene/{name}',
            mesh_color_override=color,
        )
        urdf.update_cfg({})
        return urdf
    except Exception as e:
        print(f'[inspect] WARN: failed to load {name} URDF: {e!r}')
        return None


def rebuild_static_sim_nodes(sim_s):
    """(Re)build the grid + hanger + tallrod at the given sim scale.
    Called once at startup and whenever the sim_scale slider changes."""
    # Grid — re-add (same name replaces previous node).
    server.scene.add_grid(
        '/sim_scene/grid',
        width=20.0 * sim_s, height=20.0 * sim_s, plane='xy',
        cell_size=1.0 * sim_s, section_size=5.0 * sim_s,
    )
    # URDFs — remove existing then re-add at the new scale.
    for h in list(_static_urdf_handles.values()):
        try:
            h.remove()
        except Exception:
            pass
    _static_urdf_handles.clear()
    _static_urdf_handles['hanger'] = _load_static_urdf(
        'hanger', 'urdf/hanger.urdf',
        _hanger_info['basePosition'], _hanger_info['globalScaling'],
        color=(0.95, 0.95, 0.95), sim_s=sim_s,
    )
    _static_urdf_handles['tallrod'] = _load_static_urdf(
        'tallrod', 'urdf/tallrod.urdf',
        _rod_info['basePosition'], _rod_info['globalScaling'],
        color=(0.72, 0.55, 0.35), sim_s=sim_s,
    )


rebuild_static_sim_nodes(float(extra.sim_scale))


ANCHOR_COLOR_A = (255, 128, 0)    # orange
ANCHOR_COLOR_B = (178, 0, 255)    # purple


# --------------------------------------------------------------------------
# Real-robot reachable workspace overlay (dual Franka).
#
# Per-arm EEF limits (from robots/franka.py lines 64-67):
#   Right arm: x [0.3, 0.8], y [-0.05, 0.25], z [0.2, 0.6]
#   Left  arm: x [0.3, 0.8], y [-0.25, 0.05], z [0.2, 0.6]
#   Both:      radius-from-base ∈ [0.2, 0.9] m
#
# The reachable set for each arm = AABB ∩ outer_sphere(R=0.9) ∖
# inner_sphere(R=0.2), all in the arm-base frame. We build the meshes
# with trimesh's CSG boolean ops (manifold3d backend).
#
# Convention: both arms' bases at the same origin in the parent frame
# (right arm reaches into +y, left arm into -y). The whole parent
# frame is translatable via GUI sliders so you can visually align the
# real-world workspace with the sim scene.
# --------------------------------------------------------------------------
FRANKA_LIMITS = {
    'right': {'x': (0.3, 0.8), 'y': (-0.05, 0.25), 'z': (0.2, 0.6)},
    'left':  {'x': (0.3, 0.8), 'y': (-0.25, 0.05), 'z': (0.2, 0.6)},
}
FRANKA_COLORS = {
    'right': (255, 80, 80),    # red
    'left':  (80, 120, 255),   # blue
}
FRANKA_RADIUS_INNER = 0.2
FRANKA_RADIUS_OUTER = 0.9


def _build_reachable_mesh(lims):
    """Reachable workspace for one Franka arm in its base frame:
    AABB(lims) ∩ outer_sphere(0.9 m) ∖ inner_sphere(0.2 m).

    Returns (vertices, faces) ndarrays."""
    cx = 0.5 * (lims['x'][0] + lims['x'][1])
    cy = 0.5 * (lims['y'][0] + lims['y'][1])
    cz = 0.5 * (lims['z'][0] + lims['z'][1])
    dx = lims['x'][1] - lims['x'][0]
    dy = lims['y'][1] - lims['y'][0]
    dz = lims['z'][1] - lims['z'][0]

    T = np.eye(4)
    T[:3, 3] = [cx, cy, cz]
    box = trimesh.creation.box(extents=(dx, dy, dz), transform=T)
    outer = trimesh.creation.icosphere(subdivisions=4, radius=FRANKA_RADIUS_OUTER)
    inner = trimesh.creation.icosphere(subdivisions=4, radius=FRANKA_RADIUS_INNER)

    reachable = box.intersection(outer).difference(inner)
    return (np.asarray(reachable.vertices, dtype=np.float32),
            np.asarray(reachable.faces, dtype=np.int32))


# Root frame for the real workspace — visible axes so you know which
# way the franka world x/y/z point. Translatable via GUI sliders.
real_ws_root = server.scene.add_frame(
    '/real_workspace', show_axes=True,
    axes_length=0.3, axes_radius=0.015,
    position=(0.0, 0.0, 0.0),
)
server.scene.add_label(
    '/real_workspace/label', text='franka world',
    position=(0.32, 0.0, 0.0),
)


_workspace_mesh_handles = {}    # side -> MeshHandle (for the "Show workspace" toggle)


def _add_arm_reachable(side):
    verts, faces = _build_reachable_mesh(FRANKA_LIMITS[side])
    _workspace_mesh_handles[side] = server.scene.add_mesh_simple(
        f'/real_workspace/{side}_arm',
        vertices=verts, faces=faces,
        color=FRANKA_COLORS[side],
        opacity=0.30,
        side='double',
        flat_shading=False,
    )


print('[inspect] building reachable workspace meshes (CSG)...')
_add_arm_reachable('right')
_add_arm_reachable('left')
print('[inspect] reachable workspace meshes built.')


# --------------------------------------------------------------------------
# Real-world recorded gripper trajectories (in the Franka world frame).
# Loaded once from the npz; rendered as polylines under /real_workspace.
# --------------------------------------------------------------------------
_real_traj_handles = {}    # name -> SceneNodeHandle
_exported_traj_endpoint_names = []
_exported_traj_left_start = 'n/a'
_exported_traj_left_end = 'n/a'
_exported_traj_right_start = 'n/a'
_exported_traj_right_end = 'n/a'


def _polyline_segments(pts):
    """(T, 3) waypoints -> (T-1, 2, 3) line-segment array for add_line_segments."""
    p = np.asarray(pts, dtype=np.float32)
    return np.stack([p[:-1], p[1:]], axis=1)


def _fmt_xyz(pt):
    pt = np.asarray(pt, dtype=np.float32)
    return f'[{pt[0]:.3f}, {pt[1]:.3f}, {pt[2]:.3f}]'


def _add_recorded_traj(name, pts, line_color, marker_color):
    """Render one recorded trajectory as a polyline + start/end markers."""
    if pts is None or len(pts) < 2:
        return
    segs = _polyline_segments(pts)
    line_h = server.scene.add_line_segments(
        f'/real_workspace/{name}', points=segs,
        colors=np.asarray(line_color, dtype=np.uint8),
        line_width=2.5,
    )
    start_h = server.scene.add_icosphere(
        f'/real_workspace/{name}_start', radius=0.008,
        position=tuple(map(float, pts[0])),
        color=marker_color,
    )
    end_h = server.scene.add_icosphere(
        f'/real_workspace/{name}_end', radius=0.008,
        position=tuple(map(float, pts[-1])),
        color=marker_color,
    )
    _real_traj_handles[name] = line_h
    _real_traj_handles[f'{name}_start'] = start_h
    _real_traj_handles[f'{name}_end'] = end_h


# The policy-eval exporter divides gripper proprio + goal by this
# before feeding the policy; we multiply back to recover sim-world
# coords. obs['pcd'] is exported as raw sim-world coords (NOT divided).
_SIM_PROPRIO_SCALE = 20.0


def _load_exported_traj(path):
    """Load an exported trajectory and normalize it to one dict::

        {'kind': 'realworld' | 'sim',
         'ee_left':  (T, 3) float32,    # left  end-effector path
         'ee_right': (T, 3) float32,    # right end-effector path
         'goal':     (3,)  float32 | None,    # 'sim' only, sim-native
         'pcd':      (P, 3) float32 | None,   # 'sim' only, sim-native
         'summary':  str}

    Accepted on-disk formats:
      - npz with flat (T, 3) 'ee_pos_left'/'ee_pos_right' arrays
        (teleop recordings)                       -> kind 'realworld'
      - pickle list-of-dicts, one dict per step with
        'ee_pos_left'/'ee_pos_right' (this script's own "Download
        trajectory" output, already Franka world frame)
                                                  -> kind 'realworld'
      - pickle single dict with obs={'pcd','grip','goal'}, 'acts' and
        eval metadata (policy-eval rollout). Poses come from
        obs['grip'] (= [Lpos3 Lvel3 | Rpos3 Rvel3], divided by
        _SIM_PROPRIO_SCALE on export); obs['goal'] is likewise scaled;
        obs['pcd'] is raw sim-world coords.       -> kind 'sim'

    'realworld' trajectories stay in the Franka world frame and are
    rendered statically under /real_workspace. 'sim' trajectories are
    in the sim-world frame and are rendered under the live /sim_scene
    group so they overlay the procedural cloth/anchors/goal."""
    if not path:
        return None
    traj_path = Path(path).expanduser()
    if not traj_path.exists():
        print(f'[inspect] exported trajectory not found: {traj_path}')
        return None

    suffix = traj_path.suffix.lower()
    if suffix == '.npz':
        try:
            data = np.load(traj_path, allow_pickle=False)
            if 'ee_pos_left' not in data or 'ee_pos_right' not in data:
                print(f'[inspect] WARN: npz {traj_path} missing '
                      f'ee_pos_left/ee_pos_right (keys: {list(data.keys())})')
                return None
            left  = np.asarray(data['ee_pos_left'],  dtype=np.float32)
            right = np.asarray(data['ee_pos_right'], dtype=np.float32)
            n = min(len(left), len(right))
            return {
                'kind': 'realworld',
                'ee_left':  left[:n],
                'ee_right': right[:n],
                'goal': None, 'pcd': None,
                'summary': f'{n} steps (npz, real-world frame)',
            }
        except Exception as e:
            print(f'[inspect] WARN: failed to load npz {traj_path}: {e!r}')
            return None

    try:
        with traj_path.open('rb') as f:
            data = pickle.load(f)
    except Exception as e:
        print(f'[inspect] WARN: failed to load pickle {traj_path}: {e!r}')
        return None

    # --- New policy-eval rollout: single dict with obs/acts/metadata ---
    if (isinstance(data, dict) and isinstance(data.get('obs'), dict)
            and 'grip' in data['obs']):
        obs = data['obs']
        grip = np.asarray(obs['grip'], dtype=np.float32)
        if grip.ndim != 2 or grip.shape[1] < 6:
            print(f'[inspect] WARN: unexpected obs["grip"] shape '
                  f'{grip.shape}; expected (T, >=6)')
            return None
        half = grip.shape[1] // 2     # 12 -> 6: [Lpos3 Lvel3 | Rpos3 Rvel3]
        ee_left  = grip[:, 0:3]            * _SIM_PROPRIO_SCALE
        ee_right = grip[:, half:half + 3] * _SIM_PROPRIO_SCALE
        goal = None
        if len(np.asarray(obs.get('goal', []))):
            goal = (np.asarray(obs['goal'], dtype=np.float32)[0]
                    * _SIM_PROPRIO_SCALE)
        pcd = None
        nsteps = 0
        if 'pcd' in obs:
            p = np.asarray(obs['pcd'], dtype=np.float32)   # (T, N, 3) raw
            if p.ndim == 2:        # single frame -> (1, N, 3)
                p = p[None]
            pcd = p                # keep per-frame for replay
            nsteps = p.shape[0]
        succ = (f"hang={data.get('success_hanging')} "
                f"topo={data.get('success_topological')} "
                f"legacy={data.get('success_legacy')}")
        ckpt = data.get('ckpt', 'n/a')
        n = len(ee_left)
        print(f'[inspect] imported policy-eval rollout: {n} steps, {succ}')
        print(f'[inspect]   ckpt: {ckpt}')
        if pcd is not None:
            print(f'[inspect]   cloth pcd: {pcd.shape[1]} pts/frame '
                  f'x {nsteps} frames')
        return {
            'kind': 'sim',
            'ee_left':  ee_left,
            'ee_right': ee_right,
            'goal': goal, 'pcd': pcd,
            'summary': f'{n} steps (policy-eval, sim frame) | {succ}',
        }

    # --- Legacy sample format: list-of-dicts, real-world frame ---
    if isinstance(data, list) and len(data) > 0:
        try:
            left  = np.asarray([s['ee_pos_left']  for s in data],
                               dtype=np.float32)
            right = np.asarray([s['ee_pos_right'] for s in data],
                               dtype=np.float32)
        except (KeyError, TypeError, IndexError) as e:
            print(f'[inspect] WARN: list-format trajectory missing '
                  f'ee_pos_left/right: {e!r}')
            return None
        return {
            'kind': 'realworld',
            'ee_left':  left,
            'ee_right': right,
            'goal': None, 'pcd': None,
            'summary': f'{len(data)} steps (pickle list, real-world frame)',
        }

    print(f'[inspect] WARN: exported trajectory has unexpected '
          f'format: {type(data).__name__}')
    return None


def _add_exported_traj(name, left_pts, right_pts, line_color, marker_color):
    """Render a real-world-frame imported trajectory statically under
    /real_workspace (left/right polylines + start/end markers)."""
    if left_pts is None or len(left_pts) < 2:
        return
    _add_recorded_traj(f'{name}_left', left_pts, line_color, marker_color)
    _add_recorded_traj(f'{name}_right', right_pts, line_color, marker_color)
    _exported_traj_endpoint_names.extend([
        f'{name}_left_start', f'{name}_left_end',
        f'{name}_right_start', f'{name}_right_end',
    ])


traj_left = traj_right = None
imported_traj = None          # normalized dict from _load_exported_traj
if extra.real_traj:
    traj_path = Path(extra.real_traj)
    if traj_path.exists():
        try:
            data = np.load(traj_path, allow_pickle=False)
            traj_left  = np.asarray(data['ee_pos_left'],  dtype=np.float32)
            traj_right = np.asarray(data['ee_pos_right'], dtype=np.float32)
            print(f'[inspect] loaded real-world trajectories: '
                  f'left ({len(traj_left)} pts), right ({len(traj_right)} pts)')
        except Exception as e:
            print(f'[inspect] WARN: failed to load {traj_path}: {e!r}')
    else:
        print(f'[inspect] real_traj not found: {traj_path}')

if extra.export_traj:
    imported_traj = _load_exported_traj(extra.export_traj)
    if imported_traj is not None:
        L = np.asarray(imported_traj['ee_left'],  dtype=np.float32)
        R = np.asarray(imported_traj['ee_right'], dtype=np.float32)
        _frame_note = ('' if imported_traj['kind'] == 'realworld'
                       else ' (sim-native coords)')
        _exported_traj_left_start  = _fmt_xyz(L[0])  + _frame_note
        _exported_traj_left_end    = _fmt_xyz(L[-1]) + _frame_note
        _exported_traj_right_start = _fmt_xyz(R[0])  + _frame_note
        _exported_traj_right_end   = _fmt_xyz(R[-1]) + _frame_note
        print(f'[inspect] loaded exported trajectory: '
              f"{imported_traj['summary']}")

cam_traj = None
_cam_pose_pos = np.asarray(extra.cam_pos, dtype=np.float64)
_cam_pose_wxyz = np.asarray(extra.cam_wxyz, dtype=np.float64)
if extra.cam_traj:
    cam_traj = _load_exported_traj(extra.cam_traj)
    if cam_traj is not None:
        print(f"[inspect] loaded CAMERA-FRAME trajectory: "
              f"{cam_traj['summary']}")
        print(f'[inspect]   cam pose: pos={_cam_pose_pos.tolist()} '
              f'wxyz={_cam_pose_wxyz.tolist()} (sim-world coords; '
              f'origin scales with sim_scale)')

# Slightly darker than the workspace box colors so the trajectories
# stand out against them.
_add_recorded_traj('traj_right', traj_right,
                    line_color=(200, 30, 30), marker_color=(255, 240, 0))
_add_recorded_traj('traj_left',  traj_left,
                    line_color=(30, 60, 200), marker_color=(255, 240, 0))
# Real-world-frame imports keep the old static /real_workspace render.
# Sim-frame imports (policy-eval rollouts) ride the live /sim_scene
# transform instead — built by rebuild_imported_traj_nodes() once the
# GUI sliders + pcd toggle exist (see below).
if imported_traj is not None and imported_traj['kind'] == 'realworld':
    _add_exported_traj('exported_traj',
                        imported_traj['ee_left'], imported_traj['ee_right'],
                        line_color=(100, 220, 100),
                        marker_color=(0, 255, 100))


# --------------------------------------------------------------------------
# Sim-frame imported trajectory (policy-eval rollouts via --export_traj).
# Rendered under /sim_scene so it rides the live sim_scale/offset/yaw
# transform and overlays the procedural cloth/anchors/goal. /sim_scene
# encodes scale in child COORDS (Viser 1.0.27 frame `scale` is broken),
# so these nodes are rebuilt whenever sim_scale moves — same pattern as
# rebuild_static_sim_nodes().
#
# Rendering is split: rebuild_imported_traj_nodes() lays down the STATIC
# context (faint full EE polylines + goal), while update_imported_replay()
# moves the per-frame nodes (bright EE markers + that frame's cloth pcd)
# so the rollout can be replayed/scrubbed via its own GUI controls,
# independent of the scripted-env playback.
# --------------------------------------------------------------------------
_imp_traj = imported_traj if (imported_traj is not None
                              and imported_traj.get('kind') == 'sim') else None
_imp_enabled = _imp_traj is not None
# Camera-frame overlay (same schema; obs in the sim camera frame).
_cam_traj = cam_traj if (cam_traj is not None
                         and cam_traj.get('kind') == 'sim') else None
_cam_enabled = _cam_traj is not None
cam_n = int(len(_cam_traj['ee_left'])) if _cam_traj is not None else 0
_cam_visible = True                # whole-overlay toggle (/sim_scene/cam)
_cam_root_handle = None
# Shared replay: one frame index drives both overlays in lockstep so a
# correct camframe conversion stays glued to the sim-world import.
_replay_enabled = _imp_enabled or _cam_enabled
_sim_imp_n = int(len(_imp_traj['ee_left'])) if _imp_traj is not None else 0
imp_n = max(_sim_imp_n, cam_n)
imp_frame = 0                      # main-loop-owned current replay frame
_imported_pcd_visible = True       # mirrors the GUI checkbox
_imported_pcd_handle = None
# Visibility of the sim import is split across the two existing
# checkboxes: chk_traj_show -> faint EE polylines + goal ("the path");
# chk_exported_traj_endpoints -> bright moving EE marker spheres ("the
# endpoints"). The nodes get re-added every tick (markers) / on scale
# change (path), so the desired state is held here and re-applied after
# every (re-)add, same trick as _imported_pcd_visible.
_imp_path_visible = True
_imp_markers_visible = True
_imp_path_handles = []             # line + goal + label handles
_imp_marker_handles = []           # ee_left/ee_right handles


def _set_visible(handles, vis):
    for h in handles:
        try:
            h.visible = bool(vis)
        except Exception:
            pass

# Cross-thread replay requests (set by Viser callbacks, consumed in the
# main loop). 'nudge' accumulates Step -/+ clicks; 'scrub' is a one-shot
# absolute target frame from the scrubber slider.
_imp_lock = threading.Lock()
_imp_state = {'playing': False, 'nudge': 0, 'scrub': None}


def consume_imp_flags():
    """Atomically read+clear replay flags -> (playing, nudge, scrub)."""
    with _imp_lock:
        playing = _imp_state['playing']
        nudge = _imp_state['nudge']; _imp_state['nudge'] = 0
        scrub = _imp_state['scrub']; _imp_state['scrub'] = None
    return playing, nudge, scrub


def rebuild_imported_traj_nodes(sim_s):
    """(Re)build the STATIC sim-frame context (faint full EE polylines +
    goal) and refresh the moving per-frame nodes at the current frame.
    No-op unless --export_traj loaded a sim-frame rollout."""
    global _imp_path_handles
    it = _imp_traj
    if it is None:
        return
    L = np.asarray(it['ee_left'],  dtype=np.float32)
    R = np.asarray(it['ee_right'], dtype=np.float32)
    handles = []
    if len(L) >= 2:
        handles.append(server.scene.add_line_segments(
            '/sim_scene/imported_traj/left',
            points=_polyline_segments(L) * sim_s,
            colors=np.asarray((70, 95, 140), dtype=np.uint8),   # faint
            line_width=1.5))
        handles.append(server.scene.add_line_segments(
            '/sim_scene/imported_traj/right',
            points=_polyline_segments(R) * sim_s,
            colors=np.asarray((140, 95, 55), dtype=np.uint8),   # faint
            line_width=1.5))
    g = it.get('goal')
    if g is not None:
        gp = np.asarray(g, dtype=np.float32) * sim_s
        handles.append(server.scene.add_icosphere(
            '/sim_scene/imported_traj/goal',
            radius=0.30 * sim_s,
            position=tuple(float(x) for x in gp),
            color=(255, 0, 200)))
        handles.append(server.scene.add_label(
            '/sim_scene/imported_traj/goal/label',
            text='imported goal', position=(0.0, 0.0, 0.4 * sim_s)))
    _imp_path_handles = handles
    _set_visible(_imp_path_handles, _imp_path_visible)
    update_imported_replay(imp_frame, sim_s)


def update_imported_replay(frame, sim_s):
    """Move the bright per-frame EE markers + that frame's cloth pcd to
    `frame`. Cheap (re-adds same-named nodes); called every loop tick."""
    global _imported_pcd_handle, _imp_marker_handles
    it = _imp_traj
    if it is None:
        return
    L = np.asarray(it['ee_left'],  dtype=np.float32)
    R = np.asarray(it['ee_right'], dtype=np.float32)
    f = int(np.clip(frame, 0, len(L) - 1))
    h_left = server.scene.add_icosphere(
        '/sim_scene/imported_traj/ee_left',
        radius=0.18 * sim_s,
        position=tuple(float(x) for x in L[f] * sim_s),
        color=(60, 150, 255))
    h_right = server.scene.add_icosphere(
        '/sim_scene/imported_traj/ee_right',
        radius=0.18 * sim_s,
        position=tuple(float(x) for x in R[f] * sim_s),
        color=(255, 150, 40))
    _imp_marker_handles = [h_left, h_right]
    _set_visible(_imp_marker_handles, _imp_markers_visible)
    pc = it.get('pcd')
    if pc is not None and len(pc):
        fp = int(np.clip(f, 0, pc.shape[0] - 1))
        pts = np.asarray(pc[fp], dtype=np.float32) * sim_s
        col = np.broadcast_to(np.asarray((0, 210, 210), dtype=np.uint8),
                              pts.shape).copy()
        _imported_pcd_handle = server.scene.add_point_cloud(
            '/sim_scene/imported_traj/pcd',
            points=pts, colors=col,
            point_size=max(1e-4, 0.015 * sim_s))
        try:
            _imported_pcd_handle.visible = bool(_imported_pcd_visible)
        except Exception:
            pass


def rebuild_cam_traj_nodes(sim_s):
    """(Re)build the camera-frame overlay under /sim_scene/cam.

    The cam frame sits at cam_pos*sim_s (so the camera origin scales +
    moves with the sliders, as expected) with rotation cam_wxyz; child
    points are camera-frame coords * sim_s. Composing the /sim_scene
    parent (offset + Rz(yaw)) with this gives exactly
    offset + Rz(yaw) @ (sim_s * (R_cam @ p_cam + cam_pos)) — identical
    to _sim_point_to_world of the sim-world import, so a correct
    camframe conversion overlays it at any sim_scale. Colors differ so
    any residual misalignment is visible."""
    global _cam_root_handle
    it = _cam_traj
    if it is None:
        return
    _cam_root_handle = server.scene.add_frame(
        '/sim_scene/cam',
        position=tuple(float(x) for x in _cam_pose_pos * sim_s),
        wxyz=tuple(float(x) for x in _cam_pose_wxyz),
        show_axes=True,
        axes_length=0.6 * sim_s, axes_radius=0.02 * sim_s)
    server.scene.add_label('/sim_scene/cam/label', text='sim camera',
                           position=(0.0, 0.0, 0.0))
    L = np.asarray(it['ee_left'],  dtype=np.float32)
    R = np.asarray(it['ee_right'], dtype=np.float32)
    if len(L) >= 2:
        server.scene.add_line_segments(
            '/sim_scene/cam/left', points=_polyline_segments(L) * sim_s,
            colors=np.asarray((170, 90, 170), dtype=np.uint8),
            line_width=1.5)
        server.scene.add_line_segments(
            '/sim_scene/cam/right', points=_polyline_segments(R) * sim_s,
            colors=np.asarray((120, 100, 170), dtype=np.uint8),
            line_width=1.5)
    g = it.get('goal')
    if g is not None:
        server.scene.add_icosphere(
            '/sim_scene/cam/goal', radius=0.30 * sim_s,
            position=tuple(float(x) for x in np.asarray(g, np.float32) * sim_s),
            color=(255, 255, 255))
        server.scene.add_label(
            '/sim_scene/cam/goal/label', text='cam goal',
            position=(0.0, 0.0, 0.4 * sim_s))
    update_cam_replay(imp_frame, sim_s)
    try:
        _cam_root_handle.visible = bool(_cam_visible)
    except Exception:
        pass


def update_cam_replay(frame, sim_s):
    """Per-frame camera-frame EE markers + that frame's pcd, under
    /sim_scene/cam (so they inherit the cam pose). Distinct colors from
    the sim-world import. Whole-overlay visibility is governed by the
    /sim_scene/cam parent frame, so per-tick child re-adds can't
    override a hidden overlay."""
    it = _cam_traj
    if it is None:
        return
    L = np.asarray(it['ee_left'],  dtype=np.float32)
    R = np.asarray(it['ee_right'], dtype=np.float32)
    f = int(np.clip(frame, 0, len(L) - 1))
    server.scene.add_icosphere(
        '/sim_scene/cam/ee_left', radius=0.18 * sim_s,
        position=tuple(float(x) for x in L[f] * sim_s),
        color=(255, 60, 200))
    server.scene.add_icosphere(
        '/sim_scene/cam/ee_right', radius=0.18 * sim_s,
        position=tuple(float(x) for x in R[f] * sim_s),
        color=(255, 230, 0))
    pc = it.get('pcd')
    if pc is not None and len(pc):
        fp = int(np.clip(f, 0, pc.shape[0] - 1))
        pts = np.asarray(pc[fp], dtype=np.float32) * sim_s
        col = np.broadcast_to(np.asarray((255, 150, 40), dtype=np.uint8),
                              pts.shape).copy()
        server.scene.add_point_cloud(
            '/sim_scene/cam/pcd', points=pts, colors=col,
            point_size=max(1e-4, 0.015 * sim_s))


# --------------------------------------------------------------------------
# Real-world goal point. Lives under /real_workspace so it sits in the
# franka world frame (NOT scaled/rotated by --sim_*). Magenta to stand
# out vs the yellow sim peg goal.
# --------------------------------------------------------------------------
_real_goal_pos = tuple(map(float, extra.real_goal))
server.scene.add_icosphere(
    '/real_workspace/real_goal', radius=0.015,
    position=_real_goal_pos,
    color=(255, 0, 200),
)
server.scene.add_label(
    '/real_workspace/real_goal/label',
    text='goal position in real world',
    position=(0.0, 0.0, 0.04),
)
print(f'[inspect] real-world goal plotted at {_real_goal_pos}')


# --------------------------------------------------------------------------
# Live ZED real-camera pointcloud. A background thread pulls frames from
# scripts/zed_pcd_publisher.py (TCP stream or atomically-rewritten .npy)
# and parks the latest cloud in a locked slot; the main loop transforms
# it by the GUI pose sliders and renders it under /real_workspace so it
# can be visually aligned with the sim cloud. Stdlib + numpy only.
# --------------------------------------------------------------------------
_zed_enabled = bool(extra.zed_stream) or bool(extra.zed_pcd_file)
_zed_lock = threading.Lock()
_zed_slot = {'pts': None, 'col': None, 'seq': 0, 'status': 'off'}
_zed_visible = True                # mirrors the GUI checkbox
_zed_handle = None                 # current /real_workspace/zed_pcd node


def _rpy_matrix(yaw_deg, pitch_deg, roll_deg):
    """Rz(yaw) @ Ry(pitch) @ Rx(roll), degrees -> (3,3) float64."""
    y, p, r = np.radians([yaw_deg, pitch_deg, roll_deg])
    cz, sz = np.cos(y), np.sin(y)
    cy, sy = np.cos(p), np.sin(p)
    cx, sx = np.cos(r), np.sin(r)
    Rz = np.array([[cz, -sz, 0], [sz, cz, 0], [0, 0, 1]], dtype=np.float64)
    Ry = np.array([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]], dtype=np.float64)
    Rx = np.array([[1, 0, 0], [0, cx, -sx], [0, sx, cx]], dtype=np.float64)
    return Rz @ Ry @ Rx


def _zed_store(xyzrgb):
    """(N,6) float32 [xyz|rgb 0..1] -> locked slot (subsampled)."""
    a = np.asarray(xyzrgb, dtype=np.float32)
    if a.ndim != 2 or a.shape[1] < 3 or a.shape[0] == 0:
        return
    mx = int(extra.zed_max_points)
    if mx > 0 and a.shape[0] > mx:
        a = a[np.random.choice(a.shape[0], mx, replace=False)]
    pts = a[:, :3].copy()
    if a.shape[1] >= 6:
        col = np.clip(a[:, 3:6] * 255.0, 0, 255).astype(np.uint8)
    else:
        col = np.broadcast_to(np.uint8([0, 200, 255]),
                              pts.shape).copy()
    with _zed_lock:
        _zed_slot['pts'] = pts
        _zed_slot['col'] = col
        _zed_slot['seq'] += 1
        _zed_slot['status'] = f"live ({pts.shape[0]} pts)"


def _zed_set_status(msg):
    with _zed_lock:
        _zed_slot['status'] = msg


def _zed_tcp_loop(host, port):
    while True:
        try:
            _zed_set_status(f'connecting {host}:{port}…')
            sock = socket.create_connection((host, port), timeout=5.0)
            sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
            _zed_set_status(f'connected {host}:{port}')

            def _recvall(n):
                chunks = b''
                while len(chunks) < n:
                    b = sock.recv(n - len(chunks))
                    if not b:
                        raise ConnectionError('stream closed')
                    chunks += b
                return chunks

            while True:
                hdr = _recvall(4)
                ln = int.from_bytes(hdr, 'big')
                payload = _recvall(ln)
                arr = np.load(io.BytesIO(payload), allow_pickle=False)
                _zed_store(arr)
        except Exception as e:
            _zed_set_status(f'reconnecting ({e!r})')
            time.sleep(2.0)


def _zed_file_loop(path):
    p = Path(path).expanduser()
    last_mtime = -1.0
    while True:
        try:
            if p.exists():
                m = p.stat().st_mtime
                if m != last_mtime:
                    last_mtime = m
                    arr = np.load(p, allow_pickle=False)
                    _zed_store(arr)
                    _zed_set_status(f'file ok ({p.name})')
            else:
                _zed_set_status(f'waiting for {p}')
        except Exception as e:
            _zed_set_status(f'file err ({e!r})')
        time.sleep(0.1)


if _zed_enabled:
    if extra.zed_stream:
        _h, _, _p = extra.zed_stream.partition(':')
        threading.Thread(target=_zed_tcp_loop,
                         args=(_h, int(_p or 5556)),
                         daemon=True).start()
        print(f'[inspect] ZED stream thread -> {extra.zed_stream}')
    else:
        threading.Thread(target=_zed_file_loop,
                         args=(extra.zed_pcd_file,), daemon=True).start()
        print(f'[inspect] ZED file-tail thread -> {extra.zed_pcd_file}')


# --------------------------------------------------------------------------
# Viser GUI
# --------------------------------------------------------------------------
gui_status = server.gui.add_text(
    'Status', initial_value='initializing...', disabled=False)
gui_a0 = server.gui.add_text(
    'Anchor a0 (orange)', initial_value='—', disabled=False)
gui_a1 = server.gui.add_text(
    'Anchor a1 (purple)', initial_value='—', disabled=False)
gui_seed_info = server.gui.add_text(
    'Demo', initial_value=f'seed={extra.seed} (fixed)', disabled=False)

gui_download_status = server.gui.add_text(
    'Trajectory export', initial_value='ready', disabled=True)


# Live "how big is the sim after scaling" panel. Useful sanity check
# when calibrating --sim_scale: shows cloth + hanger extents in both
# sim-native units and post-scale (viz) units, alongside the real-
# trajectory bbox for direct comparison. Rendered as a single markdown
# block so the text is high-contrast (disabled gui.add_text fields are
# greyed out and hard to read).
with server.gui.add_folder('Sim dimensions (live)'):
    md_dimensions = server.gui.add_markdown('initializing...')


# Cloth + hole geometry captured at the moment of cloth generation.
# At t=0 the cloth is FLAT, so its world-space bbox along its two
# non-degenerate axes equals the procedurally generated width/height.
# Same trick for each hole: bbox of its loop vertices gives the hole
# size, and the centroid offset gives "where in the cloth the hole sits".
_cloth_info_cache = {'verts': None, 'hole_loops': []}


def _capture_cloth_info():
    """Snapshot cloth verts + hole loop indices RIGHT AFTER reset, while
    the cloth is still flat — so its bbox equals the generated dims."""
    _cloth_info_cache['verts'] = cloth_verts().copy()
    _cloth_info_cache['hole_loops'] = (
        [list(loop) for loop in underlying.args.deform_true_loop_vertices]
        if hasattr(underlying.args, 'deform_true_loop_vertices')
        else []
    )


with server.gui.add_folder('Cloth & holes (generated)'):
    md_cloth_info = server.gui.add_markdown('initializing...')


def update_cloth_info_hud():
    """Refresh the cloth+hole info panel. Dimensions are computed from
    the t=0 bbox snapshot; the "after sim_scale" line updates live as
    you move the sim_scale slider."""
    info = _cloth_info_cache
    cv = info.get('verts')
    if cv is None or len(cv) == 0:
        md_cloth_info.content = '*no cloth captured*'
        return

    cv_min = cv.min(axis=0)
    cv_max = cv.max(axis=0)
    cv_ext = cv_max - cv_min
    # Cloth lies in a plane at t=0 — the two largest bbox extents are
    # its width & height. The smallest (~0) is the cloth's normal axis.
    sort_idx = np.argsort(cv_ext)[::-1]
    cloth_a = float(cv_ext[sort_idx[0]])
    cloth_b = float(cv_ext[sort_idx[1]])

    # World axis label (x/y/z) for each of the two cloth-plane axes.
    _axes_names = ['x', 'y', 'z']
    axis_a_name = _axes_names[int(sort_idx[0])]
    axis_b_name = _axes_names[int(sort_idx[1])]

    s = _S()
    lines = [
        f'*cloth was generated flat in two world axes; the cloth plane '
        f'spans world **{axis_a_name}** (longer side a, "width") and '
        f'world **{axis_b_name}** (shorter side b, "height"). All `(a, b)` '
        f'values below are along these two axes — NOT world x/y.*',
        '',
        f'**Cloth size (sim native):** '
        f'`{cloth_a:.2f} (a) × {cloth_b:.2f} (b) m`',
        f'**Cloth size (after sim_scale):** '
        f'`{cloth_a * s * 100:.1f} (a) × {cloth_b * s * 100:.1f} (b) cm`',
        f'**# holes:** `{len(info["hole_loops"])}`',
    ]

    # Cloth's "top-left" corner, in the cloth's two non-degenerate
    # bbox axes (axis 0 = longer side, axis 1 = shorter side).
    # Convention: top-left = (min along axis 0, min along axis 1).
    tl_cloth = np.array([cv_min[sort_idx[0]], cv_min[sort_idx[1]]],
                         dtype=np.float64)

    for hi, loop_idxs in enumerate(info['hole_loops']):
        if not loop_idxs:
            continue
        hv = cv[loop_idxs]
        hv = hv[~np.isnan(hv).any(axis=1)]
        if len(hv) == 0:
            continue
        hv_min = hv.min(axis=0)
        hv_max = hv.max(axis=0)
        hv_ext = hv_max - hv_min
        hole_a = float(hv_ext[sort_idx[0]])
        hole_b = float(hv_ext[sort_idx[1]])

        # 4 corners of the hole bbox, in the same (axis 0, axis 1) plane.
        hole_corners = {
            'TL': np.array([hv_min[sort_idx[0]], hv_min[sort_idx[1]]]),
            'TR': np.array([hv_max[sort_idx[0]], hv_min[sort_idx[1]]]),
            'BL': np.array([hv_min[sort_idx[0]], hv_max[sort_idx[1]]]),
            'BR': np.array([hv_max[sort_idx[0]], hv_max[sort_idx[1]]]),
        }

        lines.extend([
            '',
            f'**Hole {hi} size (sim):** '
            f'`{hole_a:.3f} (a) × {hole_b:.3f} (b) m`',
            f'**Hole {hi} size (scaled):** '
            f'`{hole_a * s * 100:.2f} (a) × {hole_b * s * 100:.2f} (b) cm`',
            f'**Hole {hi} corners — `(a, b)` from cloth top-left, cm:** '
            f'sim / scaled',
        ])
        for name, corner in hole_corners.items():
            rel = corner - tl_cloth
            # In SIM-native cm, and in viz cm after sim_scale.
            sim_cm = (float(rel[0]) * 100.0, float(rel[1]) * 100.0)
            scaled_cm = (sim_cm[0] * s, sim_cm[1] * s)
            lines.append(
                f'&nbsp;&nbsp;**{name}:** '
                f'`({sim_cm[0]:+6.1f}, {sim_cm[1]:+6.1f}) cm` / '
                f'`({scaled_cm[0]:+6.2f}, {scaled_cm[1]:+6.2f}) cm`'
            )

    md_cloth_info.content = '\n\n'.join(lines)


# Real-trajectory bbox strings — computed once at startup; the markdown
# block below splices them in next to the live cloth/hanger fields.
_real_left_bbox_str  = '—'
_real_right_bbox_str = '—'
if traj_left is not None and len(traj_left) > 0:
    _real_left_bbox_str = _fmt_extents(traj_left.max(0) - traj_left.min(0))
if traj_right is not None and len(traj_right) > 0:
    _real_right_bbox_str = _fmt_extents(traj_right.max(0) - traj_right.min(0))


def update_dimensions_hud():
    """Refresh the Sim dimensions panel: live cloth + hanger bboxes,
    in both sim-native units and after sim_scale, plus the static real-
    trajectory bbox for direct comparison."""
    s = _S()

    verts = cloth_verts()
    if len(verts) > 0:
        cext = verts.max(axis=0) - verts.min(axis=0)
        cloth_sim = _fmt_extents(cext)
        cloth_viz = _fmt_extents(cext * s)
    else:
        cloth_sim = cloth_viz = '—'

    # Union AABB of hanger + tallrod (rigid_ids[0] = coathanger T-shape,
    # rigid_ids[1] = vertical support pole). Reported as a single
    # "hanger assembly" bbox.
    lo, hi = None, None
    for rid in underlying.rigid_ids:
        rlo, rhi = _rigid_body_aabb(rid)
        lo = rlo if lo is None else np.minimum(lo, rlo)
        hi = rhi if hi is None else np.maximum(hi, rhi)
    if lo is not None and hi is not None:
        hext = hi - lo
        hanger_sim = _fmt_extents(hext)
        hanger_viz = _fmt_extents(hext * s)

        # World-z of the hanger assembly's top / peg, after applying
        # sim_scale + sim_offset_z. Rotation about +z doesn't affect z
        # so we can ignore --sim_yaw_deg here.
        off_z = float(sld_sim_z.value)
        hanger_top_world  = off_z + float(hi[2]) * s
        peg_world         = off_z + float(underlying.goal_pos[0][2]) * s

        # Franka workspace z bounds — same min/max across both arms.
        ws_top = FRANKA_LIMITS['right']['z'][1]
        ws_bot = FRANKA_LIMITS['right']['z'][0]
        # Gap from hanger top to workspace BOTTOM: how far above the
        # gripper's lowest reachable z does the peg sit?
        top_above_bot_cm = (hanger_top_world - ws_bot) * 100.0
        peg_above_bot_cm = (peg_world - ws_bot) * 100.0
        peg_inside_z     = (ws_bot <= peg_world <= ws_top)

        rel_str = (
            f'**Hanger top (world z):** `{hanger_top_world:.3f} m`  \n'
            f'**Hanger top vs ws bot ({ws_bot:.2f} m):** '
            f'`{top_above_bot_cm:+.1f} cm`  \n'
            f'**Peg (world z):** `{peg_world:.3f} m`  '
            f'{"✓ inside ws z" if peg_inside_z else "✗ outside ws z"}, '
            f'`{peg_above_bot_cm:+.1f} cm` vs ws bot\n\n'
        )
    else:
        hanger_sim = hanger_viz = '—'
        rel_str = ''

    md_dimensions.content = (
        f'**sim_scale:** `{s:.4f}`\n\n'
        f'*bbox values are `x × y × z`. Hanger assembly = coathanger '
        f'+ tall vertical support rod combined.*\n\n'
        f'**Cloth bbox (sim):** `{cloth_sim}`  \n'
        f'**Cloth bbox (scaled):** `{cloth_viz}`\n\n'
        f'**Hanger assembly (sim):** `{hanger_sim}`  \n'
        f'**Hanger assembly (scaled):** `{hanger_viz}`\n\n'
        + rel_str +
        f'**Real traj bbox (left):** `{_real_left_bbox_str}`  \n'
        f'**Real traj bbox (right):** `{_real_right_bbox_str}`'
    )


with server.gui.add_folder('Playback'):
    btn_play  = server.gui.add_button('Pause / Play')
    btn_step  = server.gui.add_button('Step +1 (when paused)')
    btn_reset = server.gui.add_button('Reset episode (same seed)')
    sld_hz    = server.gui.add_slider('Playback Hz',
                                       min=1, max=120, step=1, initial_value=30)
    sld_spt   = server.gui.add_slider('Steps per tick',
                                       min=1, max=10, step=1, initial_value=1)

# Independent replay controls for a sim-frame --export_traj rollout.
# Only shown when one is loaded; fully decoupled from the scripted-env
# Playback folder above (different length/seed).
btn_imp_play = btn_imp_back = btn_imp_fwd = None
sld_imp_frame = sld_imp_hz = gui_imp_status = None
if _replay_enabled:
    with server.gui.add_folder('Imported replay'):
        btn_imp_play = server.gui.add_button('Pause / Play import')
        btn_imp_back = server.gui.add_button('Step -1')
        btn_imp_fwd  = server.gui.add_button('Step +1')
        sld_imp_frame = server.gui.add_slider(
            'Frame', min=0, max=max(0, imp_n - 1), step=1,
            initial_value=0,
            hint='Scrub the imported rollout. Drag to jump to a frame; '
                 'moves on its own while playing.')
        sld_imp_hz = server.gui.add_slider(
            'Replay Hz', min=1, max=60, step=1, initial_value=15)
        gui_imp_status = server.gui.add_text(
            'Import', initial_value=f'frame 0/{max(0, imp_n - 1)}',
            disabled=True)

    @btn_imp_play.on_click
    def _(_event):
        with _imp_lock:
            _imp_state['playing'] = not _imp_state['playing']

    @btn_imp_back.on_click
    def _(_event):
        with _imp_lock:
            _imp_state['nudge'] -= 1
            _imp_state['playing'] = False

    @btn_imp_fwd.on_click
    def _(_event):
        with _imp_lock:
            _imp_state['nudge'] += 1
            _imp_state['playing'] = False

    @sld_imp_frame.on_update
    def _(event):
        # Client-initiated scrub only (Viser doesn't fire on_update for
        # server-side .value writes, so the main loop's slider sync
        # below won't feed back into this).
        with _imp_lock:
            _imp_state['scrub'] = int(event.target.value)

with server.gui.add_folder('Visualization'):
    sld_wax   = server.gui.add_slider('World axis len (m)',
                                       min=0.5, max=20.0, step=0.5, initial_value=5.0)
    sld_aax   = server.gui.add_slider('Anchor axis len (m)',
                                       min=0.1, max=2.0, step=0.1, initial_value=0.5)
    sld_vel   = server.gui.add_slider('Vel arrow scale (s)',
                                       min=0.01, max=2.0, step=0.01, initial_value=0.3)

with server.gui.add_folder('Sim → Real scale'):
    sld_sim_scale = server.gui.add_slider(
        'Sim scale (vis only)',
        min=0.01, max=2.0, step=0.01,
        initial_value=float(extra.sim_scale),
        hint='Uniform scale on the entire sim scene. ~0.05-0.10 '
             'roughly matches a sub-meter Franka workspace.',
    )
    sld_sim_x = server.gui.add_slider('Sim offset x (m)',
                                       min=-15.0, max=15.0, step=0.05,
                                       initial_value=float(extra.sim_offset_x))
    sld_sim_y = server.gui.add_slider('Sim offset y (m)',
                                       min=-15.0, max=15.0, step=0.05,
                                       initial_value=float(extra.sim_offset_y))
    sld_sim_z = server.gui.add_slider('Sim offset z (m)',
                                       min=-2.0, max=15.0, step=0.05,
                                       initial_value=float(extra.sim_offset_z))
    sld_sim_yaw = server.gui.add_slider(
        'Sim yaw (deg, about +z)',
        min=-180.0, max=180.0, step=1.0,
        initial_value=float(extra.sim_yaw_deg),
        hint="Rotates the entire sim scene about world +z. -90 maps "
             "sim's +y (the direction the cloth starts in) to franka's "
             "+x (the direction the arms reach in).")
    chk_sim_native = server.gui.add_checkbox(
        'Show sim (native) coord frame', initial_value=False,
        hint='Renders a second sim coord frame at the absolute world '
             'origin with sim-native axis size (NOT scaled by '
             'sim_scale). Useful for comparing the original sim frame '
             'to the scaled+translated one inside the franka workspace.')
    btn_align_goals = server.gui.add_button(
        'Align sim goal → real goal (xyz only)',
        hint='Solves for the sim offset x/y/z that lands the sim peg '
             '(goal_pos) exactly on --real_goal, given the current '
             'sim_scale and sim_yaw. Does not change scale or yaw.')
    btn_download_traj = server.gui.add_button(
        'Download trajectory',
        hint='Save the current sim trajectory in world-frame pickle '
             'format matching the sample trajectory format.')


# No on_update callback for sim_scale — Viser 1.0.27's frame `scale`
# is broken, so we do manual coord scaling per-tick in the main loop
# (cheap nodes) and rebuild the grid + URDFs in the main loop only
# when the slider value actually changes.


def _apply_sim_offset(_event=None):
    """Move + rotate the /sim_scene parent frame. Children's local
    coords are unchanged; they ride along. Rotation is about +z at
    /sim_scene's own origin (i.e., at the offset point in world
    coords), so dial in offset first, then yaw, and the pivot stays
    consistent with what you see."""
    global sim_scene_root
    new_pos = (float(sld_sim_x.value),
               float(sld_sim_y.value),
               float(sld_sim_z.value))
    yaw = np.radians(float(sld_sim_yaw.value))
    # Quaternion (w, x, y, z) for rotation about +z by yaw.
    wxyz = (float(np.cos(yaw / 2.0)), 0.0, 0.0,
            float(np.sin(yaw / 2.0)))
    try:
        sim_scene_root.position = new_pos
        sim_scene_root.wxyz = wxyz
    except Exception:
        # Fallback: re-add the parent frame (same name -> replaces node)
        sim_scene_root = server.scene.add_frame(
            '/sim_scene', show_axes=False,
            position=new_pos, wxyz=wxyz,
        )


sld_sim_x.on_update(_apply_sim_offset)
sld_sim_y.on_update(_apply_sim_offset)
sld_sim_z.on_update(_apply_sim_offset)
sld_sim_yaw.on_update(_apply_sim_offset)


@btn_align_goals.on_click
def _(_event):
    """Solve for sim_offset s.t. world(sim_goal) == real_goal.

    The /sim_scene transform sends a local point p to world via:
        world(p) = sim_offset + R_yaw @ (p * sim_scale)

    Setting p = sim's goal_pos and world(p) = real_goal:
        sim_offset = real_goal - R_yaw @ (sim_goal * sim_scale)
    """
    s = float(sld_sim_scale.value)
    yaw = np.radians(float(sld_sim_yaw.value))
    c, sn = np.cos(yaw), np.sin(yaw)
    R = np.array([[ c, -sn, 0.0],
                  [sn,   c, 0.0],
                  [0.0, 0.0, 1.0]], dtype=np.float64)
    g_sim = np.asarray(underlying.goal_pos[0], dtype=np.float64)
    real  = np.asarray(extra.real_goal,        dtype=np.float64)
    new_off = real - R @ (g_sim * s)
    sld_sim_x.value = float(new_off[0])
    sld_sim_y.value = float(new_off[1])
    sld_sim_z.value = float(new_off[2])
    _apply_sim_offset()
    print(f'[inspect] aligned sim goal -> real goal: '
          f'sim_offset=[{new_off[0]:+.4f}, {new_off[1]:+.4f}, '
          f'{new_off[2]:+.4f}]  '
          f'(sim_scale={s:.4f}, sim_yaw={float(sld_sim_yaw.value):.1f}°)')


def _download_trajectory_worker():
    gui_download_status.value = 'exporting trajectory...'
    export_name = (
        f'trajectory_export_seed{extra.seed}_'
        f'scale{float(sld_sim_scale.value):.4f}_'
        f'yaw{float(sld_sim_yaw.value):.1f}.pkl'
    )
    export_path = REPO_ROOT / 'experiments' / 'hang_obs_exp' / export_name
    try:
        traj_data = _build_export_trajectory()
        with export_path.open('wb') as f:
            pickle.dump(traj_data, f, protocol=pickle.HIGHEST_PROTOCOL)
        gui_download_status.value = f'saved {export_path}'
        print(f'[inspect] trajectory exported to {export_path}')
    except Exception as e:
        gui_download_status.value = f'export failed: {e}'
        print(f'[inspect] trajectory export failed: {e!r}')


@btn_download_traj.on_click
def _(_event):
    threading.Thread(target=_download_trajectory_worker, daemon=True).start()


# Apply the CLI-provided initial pose to /sim_scene immediately so
# the sim scene starts at the right place + orientation (sliders alone
# only fire on user input).
_apply_sim_offset()


with server.gui.add_folder('Real-robot workspace (dual Franka)'):
    chk_ws_show = server.gui.add_checkbox('Show workspace', initial_value=True)
    chk_traj_show = server.gui.add_checkbox(
        'Show recorded trajectory', initial_value=True,
        hint='--real_traj npz polylines, and (for a sim-frame '
             '--export_traj) the faint imported EE path + goal.')
    chk_exported_traj_endpoints = server.gui.add_checkbox(
        'Show imported trajectory endpoints',
        initial_value=bool(_imp_enabled),
        hint='Realworld --export_traj start/end markers, and (for a '
             'sim-frame --export_traj) the moving left/right EE markers.')
    gui_exported_left_start = server.gui.add_text(
        'Imported left start', initial_value=_exported_traj_left_start,
        disabled=True)
    gui_exported_left_end = server.gui.add_text(
        'Imported left end', initial_value=_exported_traj_left_end,
        disabled=True)
    gui_exported_right_start = server.gui.add_text(
        'Imported right start', initial_value=_exported_traj_right_start,
        disabled=True)
    gui_exported_right_end = server.gui.add_text(
        'Imported right end', initial_value=_exported_traj_right_end,
        disabled=True)
    chk_imported_pcd = server.gui.add_checkbox(
        'Show imported cloth pcd', initial_value=_imported_pcd_visible,
        hint='Toggle the cloth pointcloud from a sim-frame --export_traj '
             'policy-eval rollout. EE paths + goal stay visible.')


@chk_ws_show.on_update
def _(_event):
    # Only toggle the workspace reachable meshes — don't touch the
    # parent /real_workspace frame, since the trajectories also live
    # under it and have their own checkbox.
    for h in _workspace_mesh_handles.values():
        try:
            h.visible = chk_ws_show.value
        except Exception:
            pass


@chk_traj_show.on_update
def _(_event):
    global _imp_path_visible
    for name, h in _real_traj_handles.items():
        if name in _exported_traj_endpoint_names:
            continue
        try:
            h.visible = chk_traj_show.value
        except Exception:
            pass
    # Sim-frame import: this checkbox owns the faint EE path + goal.
    _imp_path_visible = bool(chk_traj_show.value)
    _set_visible(_imp_path_handles, _imp_path_visible)


@chk_exported_traj_endpoints.on_update
def _(_event):
    global _imp_markers_visible
    for name in _exported_traj_endpoint_names:
        try:
            _real_traj_handles[name].visible = chk_exported_traj_endpoints.value
        except Exception:
            pass
    # Sim-frame import: this checkbox owns the moving EE markers.
    _imp_markers_visible = bool(chk_exported_traj_endpoints.value)
    _set_visible(_imp_marker_handles, _imp_markers_visible)

for name in _exported_traj_endpoint_names:
    try:
        _real_traj_handles[name].visible = chk_exported_traj_endpoints.value
    except Exception:
        pass


@chk_imported_pcd.on_update
def _(_event):
    global _imported_pcd_visible
    _imported_pcd_visible = bool(chk_imported_pcd.value)
    if _imported_pcd_handle is not None:
        try:
            _imported_pcd_handle.visible = _imported_pcd_visible
        except Exception:
            pass


# Camera-frame overlay visibility. Toggling the /sim_scene/cam parent
# frame hides the whole subtree (pcd + EE + goal + axes) regardless of
# the per-tick child re-adds.
chk_cam_show = None
if _cam_enabled:
    chk_cam_show = server.gui.add_checkbox(
        'Show camframe overlay', initial_value=_cam_visible,
        hint='The --cam_traj rollout rendered through the sim camera '
             'pose. Should sit exactly on the sim-world import if the '
             'camera-frame conversion is correct.')

    @chk_cam_show.on_update
    def _(event):
        global _cam_visible
        _cam_visible = bool(event.target.value)
        if _cam_root_handle is not None:
            try:
                _cam_root_handle.visible = _cam_visible
            except Exception:
                pass


# Live ZED real-camera cloud controls. Pose sliders are read per-tick by
# the main loop (no on_update needed), mirroring the sim_scale pattern.
chk_zed_show = sld_zed_size = gui_zed_status = None
sld_zed_x = sld_zed_y = sld_zed_z = None
sld_zed_yaw = sld_zed_pitch = sld_zed_roll = None
if _zed_enabled:
    with server.gui.add_folder('ZED real cloud'):
        chk_zed_show = server.gui.add_checkbox(
            'Show ZED cloud', initial_value=_zed_visible,
            hint='Live real-camera pointcloud under /real_workspace. '
                 'Pose it with the sliders below to align it onto the '
                 'sim cloud (or move the sim with Sim → Real scale).')
        sld_zed_size = server.gui.add_slider(
            'ZED point size (m)', min=0.001, max=0.03, step=0.001,
            initial_value=0.006)
        sld_zed_x = server.gui.add_slider(
            'ZED x (m)', min=-3.0, max=3.0, step=0.005,
            initial_value=float(extra.zed_cam_x))
        sld_zed_y = server.gui.add_slider(
            'ZED y (m)', min=-3.0, max=3.0, step=0.005,
            initial_value=float(extra.zed_cam_y))
        sld_zed_z = server.gui.add_slider(
            'ZED z (m)', min=-3.0, max=3.0, step=0.005,
            initial_value=float(extra.zed_cam_z))
        sld_zed_yaw = server.gui.add_slider(
            'ZED yaw (deg)', min=-180.0, max=180.0, step=0.5,
            initial_value=float(extra.zed_cam_yaw))
        sld_zed_pitch = server.gui.add_slider(
            'ZED pitch (deg)', min=-180.0, max=180.0, step=0.5,
            initial_value=float(extra.zed_cam_pitch))
        sld_zed_roll = server.gui.add_slider(
            'ZED roll (deg)', min=-180.0, max=180.0, step=0.5,
            initial_value=float(extra.zed_cam_roll))
        gui_zed_status = server.gui.add_text(
            'ZED', initial_value='off', disabled=True)

    @chk_zed_show.on_update
    def _(event):
        global _zed_visible
        _zed_visible = bool(event.target.value)
        if _zed_handle is not None:
            try:
                _zed_handle.visible = _zed_visible
            except Exception:
                pass


# Build the sim-frame imported trajectory + camera-frame overlay now
# that the GUI sliders + toggles exist (each a no-op if not loaded).
# Both are rebuilt on sim_scale changes in the main loop.
rebuild_imported_traj_nodes(float(extra.sim_scale))
rebuild_cam_traj_nodes(float(extra.sim_scale))


# Cross-thread state. Viser callbacks fire on its own thread; we just
# set flags here and consume them from the main loop.
_state = {'playing': True, 'step_once': False, 'reset': False}
_state_lock = threading.Lock()


@btn_play.on_click
def _(_event):
    with _state_lock:
        _state['playing'] = not _state['playing']


@btn_step.on_click
def _(_event):
    with _state_lock:
        _state['step_once'] = True


@btn_reset.on_click
def _(_event):
    with _state_lock:
        _state['reset'] = True


def consume_flags():
    """Atomically read+clear one-shot flags. Returns (playing, step_once, reset)."""
    with _state_lock:
        playing = _state['playing']
        s = _state['step_once']; _state['step_once'] = False
        r = _state['reset']; _state['reset'] = False
    return playing, s, r


# --------------------------------------------------------------------------
# Scene update helpers
# --------------------------------------------------------------------------
def update_world_frame():
    """Sim world coordinate frame, two variants:
      1. /sim_scene/world — sits at the scaled+translated sim origin,
         axes themselves shrink with sim_scale (so they're in proportion
         to the scaled cloth/hanger).
      2. /sim_world_native — sits at the absolute world origin (0,0,0)
         with axes at sim-native length (no sim_scale applied). Shows
         the sim's ORIGINAL coordinate frame for comparison. Toggleable
         via the GUI checkbox; hidden by default."""
    # Variant 1: scaled + translated (lives under /sim_scene)
    server.scene.add_frame(
        '/sim_scene/world', show_axes=True,
        axes_length=_scl(sld_wax.value),
        axes_radius=_scl(0.06),
    )
    server.scene.add_label(
        '/sim_scene/world/label', text='sim (scaled)',
        position=(_scl(sld_wax.value) * 1.1, 0.0, 0.0),
    )
    # Variant 2: native-scale, at world origin (outside /sim_scene)
    show_native = bool(chk_sim_native.value)
    server.scene.add_frame(
        '/sim_world_native', show_axes=True,
        axes_length=float(sld_wax.value),
        axes_radius=0.06,
        visible=show_native,
    )
    server.scene.add_label(
        '/sim_world_native/label', text='sim (native)',
        position=(float(sld_wax.value) * 1.1, 0.0, 0.0),
        visible=show_native,
    )


def update_goal_marker(pos):
    server.scene.add_icosphere(
        '/sim_scene/goal', radius=_scl(0.25), position=_pos(pos),
        color=(255, 220, 0),
    )
    # Label position is relative to its parent (/sim_scene/goal), and
    # the parent is already at the scaled goal position. The label
    # offset itself (above the sphere) also needs to be in scaled units.
    server.scene.add_label('/sim_scene/goal/label', text='goal',
                           position=(0.0, 0.0, _scl(0.4)))


def update_cloth():
    verts_sim = cloth_verts()
    server.scene.add_mesh_simple(
        '/sim_scene/cloth',
        vertices=(verts_sim * _S()).astype(np.float32),
        faces=cloth_faces,
        color=(80, 180, 100),
        side='double',
        flat_shading=False,
    )


def update_anchors(pa, pb):
    aax = _scl(sld_aax.value)
    server.scene.add_frame(
        '/sim_scene/anchor_a', position=_pos(pa),
        axes_length=aax, axes_radius=_scl(0.03),
    )
    server.scene.add_frame(
        '/sim_scene/anchor_b', position=_pos(pb),
        axes_length=aax, axes_radius=_scl(0.03),
    )
    # Floating label above each anchor (relative to anchor frame).
    server.scene.add_label(
        '/sim_scene/anchor_a/label', text='a0',
        position=(0.0, 0.0, aax + _scl(0.2)),
    )
    server.scene.add_label(
        '/sim_scene/anchor_b/label', text='a1',
        position=(0.0, 0.0, aax + _scl(0.2)),
    )


def update_native_anchors(pa, pb):
    aax = float(sld_aax.value)
    show_native = bool(chk_sim_native.value)
    server.scene.add_frame(
        '/sim_world_native/anchor_a', position=tuple(pa),
        axes_length=aax * 0.8, axes_radius=0.02,
        visible=show_native,
    )
    server.scene.add_frame(
        '/sim_world_native/anchor_b', position=tuple(pb),
        axes_length=aax * 0.8, axes_radius=0.02,
        visible=show_native,
    )
    server.scene.add_label(
        '/sim_world_native/anchor_a/label', text='a0 native',
        position=(0.0, 0.0, aax * 0.8 + 0.05),
        visible=show_native,
    )
    server.scene.add_label(
        '/sim_world_native/anchor_b/label', text='a1 native',
        position=(0.0, 0.0, aax * 0.8 + 0.05),
        visible=show_native,
    )


def update_velocity_arrows(pa, pb, va_vec, vb_vec):
    vscale = float(sld_vel.value)
    S = _S()
    pts = []
    cols = []
    if np.linalg.norm(va_vec) > 1e-4:
        start = np.asarray(pa) * S
        end   = (np.asarray(pa) + np.asarray(va_vec) * vscale) * S
        pts.append([start, end])
        cols.append(ANCHOR_COLOR_A)
    if np.linalg.norm(vb_vec) > 1e-4:
        start = np.asarray(pb) * S
        end   = (np.asarray(pb) + np.asarray(vb_vec) * vscale) * S
        pts.append([start, end])
        cols.append(ANCHOR_COLOR_B)
    if not pts:
        pts = [[(0, 0, -1000), (0, 0, -1000.001)]]
        cols = [(0, 0, 0)]
    server.scene.add_arrows(
        '/sim_scene/vel',
        points=np.asarray(pts, dtype=np.float32),
        colors=np.asarray(cols, dtype=np.uint8),
        shaft_radius=_scl(0.05),
        head_radius=_scl(0.15),
        head_length=_scl(0.3),
    )


# --------------------------------------------------------------------------
# Main loop
# --------------------------------------------------------------------------
step = 0
done_flag = False
peak_seen = 0.0
ep_idx = 1

# Initial scene paint so the browser shows something before the first
# tick.
update_world_frame()
update_goal_marker(goal_position())
update_cloth()
pa0 = anchor_pos(0)
pb0 = anchor_pos(1)
update_anchors(pa0, pb0)
update_native_anchors(pa0, pb0)
update_velocity_arrows(pa0, pb0,
                       np.zeros(3), np.zeros(3))
_capture_cloth_info()


def do_reset():
    """Re-seed and reset; rebuild trajectory + reload cloth faces."""
    global traj, T, step, done_flag, peak_seen, ep_idx, cloth_faces
    fixed_reset()
    new_traj = _build_traj()
    if new_traj is None:
        print(f'\n[inspect] reset failed: no hole loop on this cloth.')
        return
    traj = new_traj
    T = len(traj)
    cloth_faces = _load_cloth_faces()
    step = 0
    done_flag = False
    peak_seen = 0.0
    ep_idx += 1
    update_goal_marker(goal_position())
    _capture_cloth_info()
    print(f'\n[inspect] reset (ep {ep_idx}, same seed={extra.seed}): '
          f'T={T}, peak |vel|={float(np.abs(traj).max()):.3f} m/s')


def update_zed_cloud():
    """Pull the latest ZED frame, transform by the GUI pose sliders, and
    render it under /real_workspace. Cheap (cloud is pre-subsampled);
    re-adds the same node each tick so it tracks slider edits."""
    global _zed_handle
    # All ZED widgets are created together under `if _zed_enabled:`; the
    # explicit None-checks also narrow the optional handles for static
    # analysis.
    if (not _zed_enabled or gui_zed_status is None
            or sld_zed_size is None
            or sld_zed_x is None or sld_zed_y is None or sld_zed_z is None
            or sld_zed_yaw is None or sld_zed_pitch is None
            or sld_zed_roll is None):
        return
    with _zed_lock:
        pts = _zed_slot['pts']
        col = _zed_slot['col']
        status = _zed_slot['status']
    gui_zed_status.value = status
    if pts is None or len(pts) == 0:
        return
    R = _rpy_matrix(sld_zed_yaw.value, sld_zed_pitch.value,
                    sld_zed_roll.value)
    t = np.array([sld_zed_x.value, sld_zed_y.value, sld_zed_z.value],
                 dtype=np.float64)
    world = (np.asarray(pts, dtype=np.float64) @ R.T + t).astype(np.float32)
    _zed_handle = server.scene.add_point_cloud(
        '/real_workspace/zed_pcd',
        points=world, colors=np.asarray(col, dtype=np.uint8),
        point_size=max(1e-4, float(sld_zed_size.value)))
    try:
        _zed_handle.visible = bool(_zed_visible)
    except Exception:
        pass


_last_applied_sim_scale = float(extra.sim_scale)
_imp_last_t = time.time()        # imported-replay frame-advance pacer
try:
    while True:
        playing, step_once, reset_req = consume_flags()
        if reset_req:
            do_reset()
            playing, step_once, reset_req = consume_flags()

        # Rebuild grid + URDFs + sim-frame imported traj if the
        # sim_scale slider moved (scale is baked into child coords).
        cur_S = float(sld_sim_scale.value)
        if abs(cur_S - _last_applied_sim_scale) > 1e-6:
            rebuild_static_sim_nodes(cur_S)
            rebuild_imported_traj_nodes(cur_S)
            rebuild_cam_traj_nodes(cur_S)
            _last_applied_sim_scale = cur_S

        # Imported-rollout replay: advance/scrub on its own Hz, decoupled
        # from the scripted-env playback below. (Widget None-checks also
        # narrow the optional GUI handles for static analysis.)
        if (_replay_enabled and sld_imp_hz is not None
                and sld_imp_frame is not None
                and gui_imp_status is not None):
            imp_playing, imp_nudge, imp_scrub = consume_imp_flags()
            if imp_scrub is not None:
                imp_frame = int(np.clip(imp_scrub, 0, imp_n - 1))
            if imp_nudge:
                imp_frame = int(np.clip(imp_frame + imp_nudge,
                                        0, imp_n - 1))
            now = time.time()
            if (imp_playing and imp_n > 1
                    and (now - _imp_last_t)
                    >= 1.0 / max(1, int(sld_imp_hz.value))):
                imp_frame = (imp_frame + 1) % imp_n      # loop the rollout
                _imp_last_t = now
            update_imported_replay(imp_frame, cur_S)
            update_cam_replay(imp_frame, cur_S)
            try:
                if int(sld_imp_frame.value) != imp_frame:
                    sld_imp_frame.value = imp_frame
            except Exception:
                pass
            _imp_tag = ('PLAY' if imp_playing else 'PAUSE')
            gui_imp_status.value = (
                f'[{_imp_tag}] frame {imp_frame}/{imp_n - 1}')

        # Live ZED real cloud (re-rendered every tick so it tracks pose
        # slider edits + new frames from the publisher).
        update_zed_cloud()

        # Pending action for this tick.
        if step < T:
            pending_act = np.asarray(traj[step], dtype=np.float32)
        else:
            pending_act = np.zeros(traj.shape[1], dtype=np.float32)

        n_steps = 1 if step_once else int(sld_spt.value)
        if not playing and not step_once:
            n_steps = 0

        applied = pending_act
        for _ in range(n_steps):
            if done_flag:
                break
            if step < T:
                applied = np.asarray(traj[step], dtype=np.float32)
            else:
                applied = np.zeros(traj.shape[1], dtype=np.float32)
            normalized = np.clip(
                applied / DeformEnv.MAX_ACT_VEL, -1.0, 1.0)
            _, _, done, info = env.step(normalized.astype(np.float32))
            peak_seen = max(peak_seen, float(np.abs(applied).max()))
            step += 1
            if done:
                done_flag = True
                with _state_lock:
                    _state['playing'] = False
                succ = info.get('is_success', None)
                print(f'\n[inspect] episode done @ step {step}.  '
                      f'success={succ}.  click Reset to replay.')
                break

        # Update scene every loop tick (whether we stepped or paused).
        pa = anchor_pos(0)
        pb = anchor_pos(1)
        va_vec = applied[:3]
        vb_vec = applied[3:]
        update_world_frame()
        update_cloth()
        update_anchors(pa, pb)
        update_native_anchors(pa, pb)
        update_velocity_arrows(pa, pb, va_vec, vb_vec)
        update_dimensions_hud()
        update_cloth_info_hud()

        # GUI text readouts.
        state_str = 'PLAY' if playing and not done_flag else ('DONE' if done_flag else 'PAUSE')
        gui_status.value = (
            f'[{state_str}] ep{ep_idx} step={step}/{T-1} '
            f't={step/ctrl_freq:.2f}s  peak|v|={peak_seen:.2f} m/s')
        pa_world = _sim_point_to_world(pa)
        pb_world = _sim_point_to_world(pb)
        gui_a0.value = (
            f'p=[{pa_world[0]:+6.2f},{pa_world[1]:+6.2f},{pa_world[2]:+6.2f}] m  '
            f'v=[{float(va_vec[0]):+5.2f},{float(va_vec[1]):+5.2f},'
            f'{float(va_vec[2]):+5.2f}] m/s')
        gui_a1.value = (
            f'p=[{pb_world[0]:+6.2f},{pb_world[1]:+6.2f},{pb_world[2]:+6.2f}] m  '
            f'v=[{float(vb_vec[0]):+5.2f},{float(vb_vec[1]):+5.2f},'
            f'{float(vb_vec[2]):+5.2f}] m/s')

        # Pace the loop.
        time.sleep(1.0 / max(1, int(sld_hz.value)))

except KeyboardInterrupt:
    print('\n[inspect] interrupted by user.')
finally:
    try:
        env.close()
    except Exception:
        pass
    print('[inspect] exit.')
