"""
pick_camera_angle.py — Viser app to choose a render-camera angle for the
HangProcCloth diffusion-BC experiments (train_diffusion_bc.py).

What it gives you
-----------------
  - The full sim scene: procedural cloth, hanger, tallrod, ground grid,
    world frame, the two gripper anchors, and the hanger goal sphere.
  - A real recorded demo trajectory (--demo_pkl) replayed in a loop so
    you can watch the actual cloth motion the policy was trained on.
    Falls back to a scripted trajectory if --demo_pkl is not given.
  - A frustum marker showing exactly where the CURRENT render camera sits
    (cam_viewmat read from the pkl, or from --cam_viewmat).
  - A "candidate" frustum you orbit with a YAW slider. It stays at the
    SAME radius and z-height as the current camera (only yaw changes).
  - The Viser viewport auto-follows the candidate frustum as you drag the
    slider, so you see exactly what the policy would see at each yaw.
    Uncheck "Lock view to candidate" to roam freely.
  - A live readout prints the exact `--cam_viewmat` tuple to copy.

Camera math
-----------
We call `sim.computeViewMatrixFromYawPitchRoll()` with EXACTLY the same
arguments as `DeformEnv._cam_viewmat`, then invert the returned matrix to
recover the camera world position and true up-vector. The snap therefore
matches the real render camera — not an approximation.

Why yaw == "same radius and z height"
--------------------------------------
PyBullet places the camera at target + dist*dir(yaw, pitch). With pitch and
dist fixed the camera rides a horizontal circle: horizontal radius
(dist*cos(pitch)) and height (target_z + dist*sin(pitch)) are both
independent of yaw.

Usage
-----
  python experiments/hang_obs_exp/scripts/pick_camera_angle.py \\
      --demo_pkl logs/hang_obs_exp/bc_demos_randgoal_0.3_full/demo_000.pkl
  # open http://localhost:8080
"""
from __future__ import annotations

import sys, time, argparse, threading
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
parser.add_argument('--port', type=int, default=8080)
parser.add_argument('--max_episode_len', type=int, default=2000)
parser.add_argument('--obs_mode', type=str, default='hole_centroid',
                    choices=['hole_centroid', 'hole_centroid_corners',
                             'hole_vertices', 'full_mesh'])
parser.add_argument('--success_factor', type=float, default=1.2)
parser.add_argument(
    '--demo_pkl', type=str, default=None,
    help='Path to a demo_NNN.pkl from collect_bc_demos.py. When given, '
         'the recorded actions are replayed instead of a scripted '
         'trajectory, and cam_viewmat + sim parameters are read from '
         'the pkl so they match collection exactly.')
parser.add_argument(
    '--cam_viewmat', type=float, nargs=6, default=None,
    metavar=('DIST', 'PITCH', 'YAW', 'TX', 'TY', 'TZ'),
    help='Override the render camera. If not set, read from --demo_pkl '
         'when available, else fall back to collect_bc_demos.py default '
         '(14 -5 45 0 0 5.5).')
parser.add_argument('--cam_fov_deg', type=float, default=60.0,
                    help='Vertical FOV for frustum markers; matches '
                         '_bc_obs_helpers.proj_matrix default fov=60.')
extra = parser.parse_args()


# --------------------------------------------------------------------------
# Load demo pkl (if given) — extract acts, cam params, sim params.
# --------------------------------------------------------------------------
import pickle as _pickle

_demo_acts = None       # (T, 6) float32 already in [-1, 1], or None
_pkl_sim_steps = None   # sim_steps_per_action from pkl
_pkl_max_act_vel = None
_pkl_ctrl_freq = None   # actual recorded ctrl_freq (Hz) from pkl
_demo_mesh_verts = None  # (T, 250, 3) recorded cloth verts (world m), or None
_demo_grip = None        # (T, 12) recorded anchor pos/vel (world m), or None
_demo_goal = None        # (T, 3) recorded hanger-goal pos (world m), or None
_pkl_cloth_faces = None  # (F, 3) recorded cloth triangulation, or None

if extra.demo_pkl is not None:
    _pkl_path = Path(extra.demo_pkl)
    if not _pkl_path.exists():
        raise FileNotFoundError(f'--demo_pkl not found: {_pkl_path}')
    with _pkl_path.open('rb') as _f:
        _demo_data = _pickle.load(_f)
    _demo_acts = np.asarray(_demo_data['acts'], dtype=np.float32)
    # Use pkl's cam_viewmat unless --cam_viewmat was explicitly passed.
    if extra.cam_viewmat is None and 'cam_viewmat' in _demo_data:
        extra.cam_viewmat = list(_demo_data['cam_viewmat'])
    if 'sim_steps_per_action' in _demo_data:
        _pkl_sim_steps = int(_demo_data['sim_steps_per_action'])
    if 'max_act_vel' in _demo_data:
        _pkl_max_act_vel = float(_demo_data['max_act_vel'])
    if 'ctrl_freq' in _demo_data:
        _pkl_ctrl_freq = float(_demo_data['ctrl_freq'])
    # Recorded geometry for direct (physics-free) replay. full_mesh row =
    # grip(12) + cloth verts(250*3), all normalized by WBOX=20.0 — undo that.
    _obs = _demo_data.get('obs', {})
    if 'full_mesh' in _obs:
        _fm = np.asarray(_obs['full_mesh'], dtype=np.float32)
        _demo_mesh_verts = _fm[:, 12:].reshape(len(_fm), 250, 3) * 20.0
    if 'grip' in _obs:
        _demo_grip = np.asarray(_obs['grip'], dtype=np.float32) * 20.0
    elif _demo_mesh_verts is not None:
        _demo_grip = np.asarray(_obs['full_mesh'], dtype=np.float32)[:, :12] * 20.0
    if 'goal' in _obs:
        _demo_goal = np.asarray(_obs['goal'], dtype=np.float32) * 20.0
    if 'cloth_faces' in _demo_data:
        _pkl_cloth_faces = np.asarray(_demo_data['cloth_faces'], dtype=np.int32)
    print(f'[cam] loaded demo: {_pkl_path.name}  '
          f'T={len(_demo_acts)}  max_act_vel={_pkl_max_act_vel}  '
          f'ctrl_freq={_pkl_ctrl_freq}  '
          f'cam_viewmat={extra.cam_viewmat}')

# Fallback cam_viewmat if still not set.
if extra.cam_viewmat is None:
    extra.cam_viewmat = [14.0, -5.0, 45.0, 0.0, 0.0, 5.5]

_dedo_argv = [
    'pick_camera_angle',
    '--env=HangProcCloth-v1',
    '--cam_resolution', '0',
    '--num_envs=0',
    '--total_env_steps=0',
    '--seed', str(extra.seed),
]
if _pkl_sim_steps is not None:
    _dedo_argv += ['--sim_steps_per_action', str(_pkl_sim_steps)]
sys.argv = _dedo_argv
args, _ = get_args_parser()
args_postprocess(args)
args.viz = False
args.debug = False
args.max_episode_len = extra.max_episode_len

if _pkl_max_act_vel is not None:
    DeformEnv.MAX_ACT_VEL = _pkl_max_act_vel

sf = None if extra.success_factor < 0 else float(extra.success_factor)


# --------------------------------------------------------------------------
# env + trajectory
# --------------------------------------------------------------------------
env = gym.make(args.env, args=args)
env = RetryResetEnv(env)
env = PrivilegedObsWrapper(env, obs_mode=extra.obs_mode, success_factor=sf)


def fixed_reset():
    np.random.seed(extra.seed)
    env.seed(extra.seed)
    return env.reset()


fixed_reset()

underlying = env
while hasattr(underlying, 'env'):
    underlying = underlying.env
    if isinstance(underlying, DeformEnv):
        break

sim = underlying.sim
ctrl_freq = args.sim_freq / args.sim_steps_per_action


def _load_cloth_faces():
    mesh = trimesh.load(underlying.args.deform_obj, process=False, force='mesh')
    return np.asarray(mesh.faces, dtype=np.int32)


cloth_faces = _load_cloth_faces()

if _demo_acts is not None:
    # Replay the real recorded demo. Acts are already in [-1, 1].
    traj = _demo_acts
    _traj_normalized = True
    print(f'[cam] replaying demo  T={len(traj)} steps, '
          f'{len(traj)/ctrl_freq:.2f} s @ {ctrl_freq:.1f} Hz')
else:
    # Fall back to scripted trajectory.
    print('\n' + '=' * 72)
    print('[cam] WARNING: no --demo_pkl given (or it failed to load).')
    print('[cam] Replaying a FRESHLY-BUILT SCRIPTED trajectory, NOT a recorded')
    print('[cam] demo. This differs from demo_NNN.pkl in BOTH trajectory and')
    print(f'[cam] speed: scripted runs at the dedo-default ctrl_freq '
          f'({ctrl_freq:.1f} Hz),')
    print('[cam] not the demo cadence. Pass --demo_pkl <path> to replay an')
    print('[cam] actual recorded demo and match its debug_viz mp4.')
    print('=' * 72 + '\n')

    def _build_traj():
        wp = build_hole_aware_waypoints(underlying)
        if wp is None:
            return None
        _, va = build_traj(underlying, wp, 'a', anchor_idx=0,
                           ctrl_freq=ctrl_freq, robot=None)
        _, vb = build_traj(underlying, wp, 'b', anchor_idx=1,
                           ctrl_freq=ctrl_freq, robot=None)
        return merge_traj(va, vb)

    traj = _build_traj()
    assert traj is not None, (
        f'seed={extra.seed} produced a cloth with no hole; try another --seed.')
    _traj_normalized = False
    print(f'[cam] scripted traj  T={len(traj)} steps, '
          f'{len(traj)/ctrl_freq:.2f} s @ {ctrl_freq:.1f} Hz')

T = len(traj)


# --------------------------------------------------------------------------
# Geometry-replay setup.
#
# Open-loop replay of the recorded ANCHOR VELOCITIES in a freshly-reset env
# CANNOT reproduce a demo: HangProcCloth re-randomizes the cloth width/height
# AND hole placement every reset (procedural_utils.py:37-47), saving a unique
# mesh to a random /tmp path. So this env builds a structurally different
# cloth (different vertex count, size, topology) than the demo's, and the
# demo's velocities drive it to diverge — it whips around and tunnels through
# the floor. The pkl stores neither the seed, the .obj, nor (for old demos)
# the faces, so the exact cloth is unrecoverable by re-simulation.
#
# Fix: when the pkl carries the recorded cloth geometry (obs['full_mesh']),
# DRAW the recorded vertices directly each frame — no physics at all — so the
# viewer matches the debug_viz mp4 exactly and nothing falls through. Recorded
# world coords are mapped into THIS env's (un-randomized) goal frame so the
# static hanger / tallrod / goal sphere still line up with the cloth.
# --------------------------------------------------------------------------
_geom_replay = _demo_mesh_verts is not None
_geom_delta = np.zeros(3, dtype=np.float64)
_geom_n_verts = None
_geom_render = None
if _geom_replay:
    _nominal_goal = np.asarray(underlying.goal_pos[0], dtype=np.float64)
    if _demo_goal is not None:
        # Shift recorded coords so the recorded peg lands on this env's
        # (un-randomized) peg. Randomization is xy-only, so dz ≈ 0.
        _geom_delta = _demo_goal[0].astype(np.float64) - _nominal_goal
    if _pkl_cloth_faces is not None:
        _geom_n_verts = int(_pkl_cloth_faces.max()) + 1
        _geom_render = 'mesh'
    else:
        # No faces in this pkl: infer the real (un-padded) vertex count from
        # the recorded verts and render as a point cloud.
        _nz = np.any(np.abs(_demo_mesh_verts) > 1e-6, axis=(0, 2))
        _geom_n_verts = (int(np.where(_nz)[0].max()) + 1
                         if _nz.any() else _demo_mesh_verts.shape[1])
        _geom_render = 'pointcloud'
    print('[cam] GEOMETRY REPLAY: drawing recorded cloth verts directly '
          '(no physics) — matches the demo mp4 exactly.')
    print(f'[cam]   render={_geom_render}  n_verts={_geom_n_verts}  '
          f'goal_shift(xy)={np.round(_geom_delta[:2], 3).tolist()}')
    if _geom_render == 'pointcloud':
        print('[cam]   NOTE: this pkl has no cloth_faces, so the cloth shows '
              'as a POINT CLOUD. Re-collect with the updated '
              'collect_bc_demos.py to store faces and get a solid mesh.')


# --------------------------------------------------------------------------
# camera-pose math
# --------------------------------------------------------------------------
def cam_pose_from_viewmat(dist, pitch_deg, yaw_deg, target):
    """Exact camera world-pose from a PyBullet [dist, pitch, yaw, target] spec.

    Calls sim.computeViewMatrixFromYawPitchRoll with the same arguments as
    DeformEnv._cam_viewmat — no approximation. Then inverts the OpenGL view
    matrix to recover:
      cam_pos   (3,)  world position
      wxyz      (4,)  camera->world orientation, viser frustum convention
                      (OpenCV: +Z forward, +X right, +Y down)
      up_world  (3,)  camera's true up direction in world space
                      (use this for client.camera.up_direction so the snap
                      matches the PyBullet render without any roll error)
    """
    vm = sim.computeViewMatrixFromYawPitchRoll(
        cameraTargetPosition=[float(target[0]), float(target[1]),
                              float(target[2])],
        distance=float(dist), yaw=float(yaw_deg), pitch=float(pitch_deg),
        roll=0.0, upAxisIndex=2)
    # PyBullet returns column-major 4x4 (OpenGL convention).
    # V maps world -> camera: p_cam = V @ p_world.
    V = np.asarray(vm, dtype=np.float64).reshape(4, 4, order='F')
    R_wc = V[:3, :3].T         # camera->world rotation (columns = cam axes in world)
    cam_pos = -R_wc @ V[:3, 3] # = R_wc @ (-t) where t = -R_cam_to_world @ cam_pos
    # OpenGL cam: col0=+X(right), col1=+Y(up), col2=+Z(back from viewer).
    right_w = R_wc[:, 0]
    up_w    = R_wc[:, 1]    # true camera up in world; use for snap up_direction
    fwd_w   = -R_wc[:, 2]  # camera looks down -Z in OpenGL
    # Viser frustum: OpenCV axes X=right, Y=down, Z=forward.
    R_cv = np.column_stack([right_w, -up_w, fwd_w])
    return cam_pos, _rotmat_to_wxyz(R_cv), up_w


def _rotmat_to_wxyz(R):
    """3x3 rotation matrix -> viser-convention (w,x,y,z) unit quaternion."""
    R = np.asarray(R, dtype=np.float64)
    tr = np.trace(R)
    if tr > 0.0:
        s = np.sqrt(tr + 1.0) * 2.0
        w = 0.25 * s
        x = (R[2, 1] - R[1, 2]) / s
        y = (R[0, 2] - R[2, 0]) / s
        z = (R[1, 0] - R[0, 1]) / s
    elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
        s = np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2]) * 2.0
        w = (R[2, 1] - R[1, 2]) / s
        x = 0.25 * s
        y = (R[0, 1] + R[1, 0]) / s
        z = (R[0, 2] + R[2, 0]) / s
    elif R[1, 1] > R[2, 2]:
        s = np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2]) * 2.0
        w = (R[0, 2] - R[2, 0]) / s
        x = (R[0, 1] + R[1, 0]) / s
        y = 0.25 * s
        z = (R[1, 2] + R[2, 1]) / s
    else:
        s = np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1]) * 2.0
        w = (R[1, 0] - R[0, 1]) / s
        x = (R[0, 2] + R[2, 0]) / s
        y = (R[1, 2] + R[2, 1]) / s
        z = 0.25 * s
    q = np.array([w, x, y, z], dtype=np.float64)
    return q / np.linalg.norm(q)


CUR_DIST  = float(extra.cam_viewmat[0])
CUR_PITCH = float(extra.cam_viewmat[1])
CUR_YAW   = float(extra.cam_viewmat[2])
CAM_TARGET = np.asarray(extra.cam_viewmat[3:6], dtype=np.float64)

_radius = CUR_DIST * np.cos(np.radians(CUR_PITCH))
_height = CAM_TARGET[2] + CUR_DIST * np.sin(np.radians(CUR_PITCH))

print(f'[cam] viewmat = dist={CUR_DIST} pitch={CUR_PITCH} yaw={CUR_YAW} '
      f'target={CAM_TARGET.tolist()}')
print(f'[cam] orbit: horiz_radius={_radius:.2f} m, z_height={_height:.2f} m '
      f'(fixed as yaw sweeps)')
print(f'[cam] NOTE: default matches collect_bc_demos.py. If you collected '
      f'demos with a custom --cam_viewmat, pass that same value here so the '
      f'red marker matches what the policy actually sees.')


# --------------------------------------------------------------------------
# Viser server + static scene
# --------------------------------------------------------------------------
server = viser.ViserServer(host='0.0.0.0', port=extra.port)
print(f'\n[cam] Viser running — open http://localhost:{extra.port}\n')

server.scene.add_frame('/world', show_axes=True,
                       axes_length=1.5, axes_radius=0.03)
server.scene.add_grid('/grid', width=20.0, height=20.0, plane='xy',
                      cell_size=1.0, section_size=5.0)

_entities = SCENE_INFO['hangcloth']['entities']


def _load_static_urdf(name, urdf_relpath, base_pos, base_scale, color):
    urdf_abs = Path(underlying.args.data_path) / urdf_relpath
    if not urdf_abs.exists():
        print(f'[cam] WARN: {urdf_abs} not found; skipping {name}')
        return
    server.scene.add_frame(f'/{name}', position=tuple(map(float, base_pos)),
                           show_axes=False)
    try:
        urdf = ViserUrdf(server, urdf_abs, scale=float(base_scale),
                         root_node_name=f'/{name}',
                         mesh_color_override=color)
        urdf.update_cfg({})
    except Exception as e:
        print(f'[cam] WARN: failed to load {name} URDF: {e!r}')


_load_static_urdf('hanger', 'urdf/hanger.urdf',
                  _entities['urdf/hanger.urdf']['basePosition'],
                  _entities['urdf/hanger.urdf']['globalScaling'],
                  color=(0.95, 0.95, 0.95))
_load_static_urdf('tallrod', 'urdf/tallrod.urdf',
                  _entities['urdf/tallrod.urdf']['basePosition'],
                  _entities['urdf/tallrod.urdf']['globalScaling'],
                  color=(0.72, 0.55, 0.35))

_goal = np.asarray(underlying.goal_pos[0], dtype=np.float32)
server.scene.add_icosphere('/goal', radius=0.35,
                           position=tuple(map(float, _goal)),
                           color=(255, 220, 0))
server.scene.add_label('/goal/label', text='goal',
                       position=(float(_goal[0]), float(_goal[1]),
                                 float(_goal[2]) + 0.6))


# --------------------------------------------------------------------------
# live scene helpers
# --------------------------------------------------------------------------
def cloth_verts():
    _, v = get_mesh_data(sim, underlying.deform_id)
    return np.asarray(v, dtype=np.float32)


def anchor_pos(idx):
    aid = list(underlying.anchors.keys())[idx]
    pos, _ = sim.getBasePositionAndOrientation(aid)
    return np.array(pos, dtype=np.float32)


def update_cloth():
    server.scene.add_mesh_simple('/cloth', vertices=cloth_verts(),
                                 faces=cloth_faces, color=(80, 180, 100),
                                 side='double', flat_shading=False)


def update_anchors():
    pa, pb = anchor_pos(0), anchor_pos(1)
    server.scene.add_frame('/anchor_a', position=tuple(map(float, pa)),
                           axes_length=0.8, axes_radius=0.04)
    server.scene.add_frame('/anchor_b', position=tuple(map(float, pb)),
                           axes_length=0.8, axes_radius=0.04)


def update_cloth_geom(t):
    """Draw the recorded cloth at timestep t (no physics). Solid mesh when
    the pkl stored faces, else a point cloud."""
    v = (_demo_mesh_verts[t, :_geom_n_verts] - _geom_delta).astype(np.float32)
    if _geom_render == 'mesh':
        server.scene.add_mesh_simple('/cloth', vertices=v,
                                     faces=_pkl_cloth_faces,
                                     color=(80, 180, 100), side='double',
                                     flat_shading=False)
    else:
        server.scene.add_point_cloud('/cloth', points=v,
                                     colors=(80, 180, 100), point_size=0.04)


def update_anchors_geom(t):
    pa = (_demo_grip[t, 0:3] - _geom_delta).astype(np.float32)
    pb = (_demo_grip[t, 6:9] - _geom_delta).astype(np.float32)
    server.scene.add_frame('/anchor_a', position=tuple(map(float, pa)),
                           axes_length=0.8, axes_radius=0.04)
    server.scene.add_frame('/anchor_b', position=tuple(map(float, pb)),
                           axes_length=0.8, axes_radius=0.04)


# --------------------------------------------------------------------------
# camera markers
# --------------------------------------------------------------------------
_FRUSTUM_SCALE = max(0.6, 0.18 * CUR_DIST)
_aspect = 1.0
_fov = np.radians(float(extra.cam_fov_deg))


def _draw_frustum(name, dist, pitch, yaw, color):
    """Draw a camera frustum + sight line. Returns (pos, wxyz, up_world)."""
    pos, wxyz, up_w = cam_pose_from_viewmat(dist, pitch, yaw, CAM_TARGET)
    server.scene.add_camera_frustum(
        name, fov=_fov, aspect=_aspect, scale=_FRUSTUM_SCALE,
        color=color, position=tuple(map(float, pos)),
        wxyz=tuple(map(float, wxyz)))
    server.scene.add_line_segments(
        f'{name}/sight',
        points=np.stack([pos, CAM_TARGET])[None].astype(np.float32),
        colors=np.asarray(color, dtype=np.uint8), line_width=1.5)
    return pos, wxyz, up_w


# Current camera marker (red, fixed).
_draw_frustum('/cam_current', CUR_DIST, CUR_PITCH, CUR_YAW, (230, 40, 40))
server.scene.add_label('/cam_current/label', text='current camera',
                       position=(0.0, 0.0, 0.0))

# Orbit center (white sphere at the shared look-at target).
server.scene.add_icosphere('/cam_target', radius=0.18,
                           position=tuple(map(float, CAM_TARGET)),
                           color=(255, 255, 255))

# Orbit ring — circle of constant horizontal radius + z-height.
_ring_n = 128
_ring_t = np.linspace(0, 2 * np.pi, _ring_n)
_ring_pts = np.stack([
    CAM_TARGET[0] + _radius * np.cos(_ring_t),
    CAM_TARGET[1] + _radius * np.sin(_ring_t),
    np.full(_ring_n, _height),
], axis=1).astype(np.float32)
server.scene.add_line_segments(
    '/orbit_ring',
    points=np.stack([_ring_pts[:-1], _ring_pts[1:]], axis=1),
    colors=np.asarray((120, 120, 255), dtype=np.uint8), line_width=1.5)


# Mutable candidate state — updated by main loop, read by on_client_connect.
# CPython object reference assignment is atomic, so no lock needed.
_candidate = {
    'pos': np.zeros(3, dtype=np.float64),
    'up':  np.array([0., 0., 1.], dtype=np.float64),
}


def update_candidate(yaw_deg):
    """Redraw the candidate frustum and update _candidate state."""
    pos, wxyz, up_w = _draw_frustum('/cam_candidate', CUR_DIST, CUR_PITCH,
                                    yaw_deg, (40, 200, 90))
    server.scene.add_label('/cam_candidate/label', text='candidate',
                           position=(0.0, 0.0, 0.0))
    _candidate['pos'] = pos
    _candidate['up']  = up_w
    return pos, up_w


def _snap_to_candidate(client):
    """Snap one Viser client to the current candidate camera.

    Uses the true up vector extracted from the PyBullet view matrix so the
    Viser view has the same roll as the actual render — no approximation."""
    pos = _candidate['pos']
    up  = _candidate['up']
    client.camera.position     = tuple(float(x) for x in pos)
    client.camera.look_at      = tuple(float(x) for x in CAM_TARGET)
    client.camera.up_direction = tuple(float(x) for x in up)


@server.on_client_connect
def _(client) -> None:
    """Snap every new client immediately to the current candidate view."""
    _snap_to_candidate(client)


# --------------------------------------------------------------------------
# GUI
# --------------------------------------------------------------------------
server.gui.add_markdown(
    '### Camera angle picker\n'
    'Drag **Candidate yaw** to orbit the green camera at the same radius + '
    'height as the red current camera. The viewer auto-follows.\n\n'
    'Copy the `--cam_viewmat` line below into `collect_bc_demos.py` / '
    '`train_diffusion_bc.py`.')

sld_yaw = server.gui.add_slider('Candidate yaw (deg)', min=-180.0, max=180.0,
                                step=1.0,
                                initial_value=(CUR_YAW + 180.0) % 360.0 - 180.0)
txt_viewmat = server.gui.add_text('--cam_viewmat', initial_value='')

with server.gui.add_folder('View'):
    chk_lock = server.gui.add_checkbox('Lock view to candidate',
                                       initial_value=True)
    btn_snap  = server.gui.add_button('Snap now')
    btn_reset = server.gui.add_button('Reset yaw to current camera')

with server.gui.add_folder('Playback'):
    chk_play = server.gui.add_checkbox('Play trajectory', initial_value=True)
    # Prefer the pkl's recorded ctrl_freq so playback matches the demo's
    # real-time cadence (and the debug_viz mp4). Recomputing from args is
    # only a fallback for the scripted path, where it can silently land on
    # the dedo default (62.5 Hz) and play ~4x too fast.
    _playback_hz = _pkl_ctrl_freq if _pkl_ctrl_freq is not None else ctrl_freq
    sld_hz   = server.gui.add_slider(
        'Playback Hz', min=1, max=120, step=1,
        initial_value=min(120, max(1, int(round(_playback_hz)))))


def _wrap180(a):
    return (a + 180.0) % 360.0 - 180.0


def _viewmat_str(yaw_deg):
    return (f'{CUR_DIST:g} {CUR_PITCH:g} {_wrap180(yaw_deg):g} '
            f'{CAM_TARGET[0]:g} {CAM_TARGET[1]:g} {CAM_TARGET[2]:g}')


@btn_snap.on_click
def _(_) -> None:
    for client in server.get_clients().values():
        _snap_to_candidate(client)


@btn_reset.on_click
def _(_) -> None:
    sld_yaw.value = CUR_YAW


# --------------------------------------------------------------------------
# main loop
# --------------------------------------------------------------------------
if _geom_replay:
    update_cloth_geom(0)
    update_anchors_geom(0)
else:
    update_cloth()
    update_anchors()
update_candidate(sld_yaw.value)
txt_viewmat.value = _viewmat_str(sld_yaw.value)
# Snap before the loop so any clients already connected get the right view.
for client in server.get_clients().values():
    _snap_to_candidate(client)

step_i  = 0
_last_yaw = sld_yaw.value

while True:
    yaw_changed = sld_yaw.value != _last_yaw
    if yaw_changed:
        _last_yaw = sld_yaw.value
        update_candidate(sld_yaw.value)
        txt_viewmat.value = _viewmat_str(sld_yaw.value)

    if chk_lock.value and yaw_changed:
        for client in server.get_clients().values():
            _snap_to_candidate(client)

    if chk_play.value:
        if _geom_replay:
            # Physics-free: just draw the recorded geometry at this step.
            update_cloth_geom(step_i)
            update_anchors_geom(step_i)
            step_i += 1
            if step_i >= T:
                step_i = 0
        else:
            if _traj_normalized:
                act = traj[step_i]                   # pkl: already in [-1, 1]
            else:
                act = np.clip(traj[step_i] / DeformEnv.MAX_ACT_VEL, -1.0, 1.0)
            env.step(act.astype(np.float32))
            update_cloth()
            update_anchors()
            step_i += 1
            if step_i >= T:
                step_i = 0
                fixed_reset()

    time.sleep(1.0 / max(1, int(sld_hz.value)))
