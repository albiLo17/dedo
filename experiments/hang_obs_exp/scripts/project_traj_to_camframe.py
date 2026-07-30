"""
Reproject a trajectory pkl from world frame to sim-camera frame.

Camera-frame convention after projection:
  +Z  points out of the camera (forward / depth)
  +X  points right
  +Y  points down
This is the standard OpenCV/ROS camera convention.

The transformation is recovered directly from the pkl's cam_viewmat via
pybullet.computeViewMatrixFromYawPitchRoll — no running sim required.

What gets transformed:
  obs['pcd']   (T, N, 3) world metres   → full rigid transform
  obs['grip']  (T, 12)  /WBOX           → positions (dims 0:3, 6:9): full rigid;
                                           velocities (dims 3:6, 9:12): rotation only
  obs['goal']  (T, 3)   /WBOX           → full rigid transform
  acts         (T, 6)   /MAX_ACT_VEL    → rotation only (velocity vectors)

All other keys are copied verbatim.  A 'cam_frame = True' flag is added to
the output pkl so downstream code can detect the transformed format.

Metadata added to output pkl
-----------------------------
  cam_extrinsics_sim   — camera pose in the sim world frame
  cam_intrinsics       — FOV, focal lengths in pixels, principal point
  sim_to_real          — transform parameters (optional, via --sim_to_real_*)
  cam_extrinsics_real  — camera pose in the real world frame (only if sim_to_real given)

Sim-to-real transform convention:
  p_real = R_z(rotation_z_deg) @ (scale * p_sim) + offset_xyz
  Orientation: R_cam_to_real = R_z(rotation_z_deg) @ R_cam_to_sim

Usage:
  python experiments/hang_obs_exp/scripts/project_traj_to_camframe.py \\
      --pkl logs/hang_obs_exp/eval_trajs/.../traj_ep001_....pkl

  # with sim-to-real transform to also get real-world camera pose
  python experiments/hang_obs_exp/scripts/project_traj_to_camframe.py \\
      --pkl logs/.../traj_ep001.pkl \\
      --sim_to_real_scale 0.045 \\
      --sim_to_real_offset 0.5 0.0 -0.039 \\
      --sim_to_real_rotation_z_deg 90
"""
from __future__ import annotations

import argparse
import pickle
from pathlib import Path

import numpy as np
import pybullet  # pure-math fns work without a physics server


WBOX   = 20.0   # grip / goal are stored as world_metres / WBOX
_FOV   = 60.0   # matches proj_matrix() in _bc_obs_helpers.py
_NEAR  = 0.1
_FAR   = 30.0


# ---------------------------------------------------------------------------
# Camera math
# ---------------------------------------------------------------------------

def _world_to_cam_transform(cam_viewmat):
    """Return (R, t) mapping world positions to the desired camera frame.

    cam_viewmat: [dist, pitch, yaw, tx, ty, tz]
    Desired frame: +Z forward, +X right, +Y down  (OpenCV convention).

    R: (3,3) rotation matrix   — apply to any world vector (pos or vel)
    t: (3,)  translation (m)   — add ONLY when transforming positions
    """
    dist, pitch, yaw, tx, ty, tz = cam_viewmat
    V_flat = pybullet.computeViewMatrixFromYawPitchRoll(
        cameraTargetPosition=(tx, ty, tz),
        distance=dist,
        yaw=yaw,
        pitch=pitch,
        roll=0,
        upAxisIndex=2,          # Z-up world, matching dedo's convention
    )
    # PyBullet returns the 16 elements in column-major (Fortran) order.
    V = np.array(V_flat, dtype=np.float64).reshape(4, 4, order='F')

    # V transforms world → OpenGL camera frame: -Z forward, +X right, +Y up.
    # Flip Y and Z to reach desired frame: +Z forward, +X right, +Y down.
    R_flip = np.diag([1.0, -1.0, -1.0])
    R = (R_flip @ V[:3, :3]).astype(np.float64)
    t = (R_flip @ V[:3, 3]).astype(np.float64)
    return R, t


def _cam_extrinsics_sim(cam_viewmat):
    """Return dict with camera position and orientation in the sim world frame."""
    R, t = _world_to_cam_transform(cam_viewmat)
    R_cam_to_sim = R.T                        # cam → sim world
    pos_sim      = -R.T @ t                   # camera origin in sim world (metres)
    from scipy.spatial.transform import Rotation as _Rot
    q_xyzw = _Rot.from_matrix(R_cam_to_sim).as_quat()
    return {
        'position_sim':  pos_sim.tolist(),
        'R_cam_to_sim':  R_cam_to_sim.tolist(),
        'quat_xyzw_sim': q_xyzw.tolist(),
        'quat_wxyz_sim': [q_xyzw[3], *q_xyzw[:3]],
    }


def _cam_intrinsics(resolution):
    """Camera intrinsics matching _bc_obs_helpers.proj_matrix()."""
    fov_rad = np.radians(_FOV)
    # principal point at image centre (square image)
    cx = cy = resolution / 2.0
    # focal length in pixels from vertical FOV
    f = cx / np.tan(fov_rad / 2.0)
    return {
        'fov_deg':    _FOV,
        'near':       _NEAR,
        'far':        _FAR,
        'resolution': resolution,
        'fx_px':      round(f, 4),
        'fy_px':      round(f, 4),
        'cx_px':      cx,
        'cy_px':      cy,
        # 3x3 K matrix (row-major)
        'K': [[f, 0, cx],
              [0, f, cy],
              [0, 0, 1]],
    }


def _cam_extrinsics_real(cam_viewmat, scale, offset_xyz, rotation_z_deg):
    """Return dict with camera pose in the real world frame.

    Sim-to-real:  p_real = R_z(rotation_z_deg) @ (scale * p_sim) + offset_xyz
    Orientation:  R_cam_to_real = R_z(rotation_z_deg) @ R_cam_to_sim
    """
    R, t = _world_to_cam_transform(cam_viewmat)
    R_cam_to_sim = R.T
    pos_sim      = -R.T @ t

    angle  = np.radians(rotation_z_deg)
    c, s   = np.cos(angle), np.sin(angle)
    R_z    = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])
    offset = np.asarray(offset_xyz, dtype=np.float64)

    pos_real      = R_z @ (scale * pos_sim) + offset
    R_cam_to_real = R_z @ R_cam_to_sim

    from scipy.spatial.transform import Rotation as _Rot
    q_xyzw = _Rot.from_matrix(R_cam_to_real).as_quat()
    return {
        'position_real':  pos_real.tolist(),
        'R_cam_to_real':  R_cam_to_real.tolist(),
        'quat_xyzw_real': q_xyzw.tolist(),
        'quat_wxyz_real': [q_xyzw[3], *q_xyzw[:3]],
    }


# ---------------------------------------------------------------------------
# Per-field transforms
# ---------------------------------------------------------------------------

def _xf_positions(pts, R, t):
    """(*, 3) world metres → (*, 3) cam metres.  t must be in same units."""
    return (pts @ R.T + t).astype(pts.dtype)


def _xf_velocities(vels, R):
    """(*, 3) world velocity vectors → (*, 3) cam velocity vectors (no t)."""
    return (vels @ R.T).astype(vels.dtype)


def _xf_pcd(pcd, R, t):
    """(T, N, 3) float32 world metres → (T, N, 3) float32 cam metres."""
    T, N, _ = pcd.shape
    return _xf_positions(pcd.reshape(-1, 3), R, t).reshape(T, N, 3)


def _xf_grip(grip, R, t_wbox):
    """(T, 12) /WBOX → (T, 12) /WBOX in cam frame.

    Layout: [anc1_pos(3)  anc1_vel(3)  anc2_pos(3)  anc2_vel(3)]
    """
    g = grip.copy()
    g[:, 0:3] = _xf_positions(grip[:, 0:3], R, t_wbox)   # anchor-1 position
    g[:, 3:6] = _xf_velocities(grip[:, 3:6], R)           # anchor-1 velocity
    g[:, 6:9] = _xf_positions(grip[:, 6:9], R, t_wbox)    # anchor-2 position
    g[:, 9:12] = _xf_velocities(grip[:, 9:12], R)          # anchor-2 velocity
    return g


def _xf_acts(acts, R):
    """(T, 6) /MAX_ACT_VEL → (T, 6) in cam frame.

    Layout: [anc1_vel(3)  anc2_vel(3)] — velocity vectors, rotation only.
    """
    a = acts.copy()
    a[:, 0:3] = _xf_velocities(acts[:, 0:3], R)
    a[:, 3:6] = _xf_velocities(acts[:, 3:6], R)
    return a


# ---------------------------------------------------------------------------
# Top-level transform
# ---------------------------------------------------------------------------

def transform_traj(data, R, t, sim_to_real=None):
    """Return a copy of `data` with all spatial fields in camera frame.

    sim_to_real: optional dict with keys scale, offset_xyz, rotation_z_deg.
                 When provided, cam_extrinsics_real is added to the output.
    """
    t_wbox = (t / WBOX).astype(np.float32)

    obs = dict(data['obs'])
    obs['pcd']  = _xf_pcd(np.asarray(obs['pcd'],  dtype=np.float32), R, t)
    obs['goal'] = _xf_positions(np.asarray(obs['goal'], dtype=np.float32), R, t_wbox)
    obs['grip'] = _xf_grip(np.asarray(obs['grip'], dtype=np.float32), R, t_wbox)

    cam_viewmat = data['cam_viewmat']
    resolution  = data.get('cam_resolution', 128)

    out = dict(data)
    out['obs']                = obs
    out['acts']               = _xf_acts(np.asarray(data['acts'], dtype=np.float32), R)
    out['cam_frame']          = True
    out['cam_extrinsics_sim'] = _cam_extrinsics_sim(cam_viewmat)
    out['cam_intrinsics']     = _cam_intrinsics(resolution)

    if sim_to_real is not None:
        out['sim_to_real'] = sim_to_real
        out['cam_extrinsics_real'] = _cam_extrinsics_real(
            cam_viewmat,
            sim_to_real['scale'],
            sim_to_real['offset_xyz'],
            sim_to_real['rotation_z_deg'],
        )

    return out


# ---------------------------------------------------------------------------
# Sanity checks
# ---------------------------------------------------------------------------

def _sanity_check(original, transformed, R, t, cam_viewmat):
    pcd_orig = np.asarray(original['obs']['pcd'])
    pcd_cam  = np.asarray(transformed['obs']['pcd'])

    z = pcd_cam[..., 2]
    print(f'  PCD z range (should be all positive): [{z.min():.3f}, {z.max():.3f}]  '
          f'all>0={z.min() > 0}')

    # Camera target in world should map to (0, 0, dist) in cam frame
    dist, pitch, yaw, tx, ty, tz = cam_viewmat
    target_world = np.array([tx, ty, tz])
    target_cam   = R @ target_world + t
    print(f'  Camera target {tuple(target_world)} → cam frame {target_cam.round(3)}'
          f'  (expect [0, 0, {dist}])')

    # Round-trip check: inverse transform should recover original PCD
    R_inv = R.T
    t_inv = -R.T @ t
    pcd_rt = _xf_pcd(pcd_cam, R_inv, t_inv)
    err = np.abs(pcd_rt - pcd_orig).max()
    print(f'  Round-trip max error: {err:.6f} m  (should be ~0)')


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--pkl', required=True, help='Input trajectory pkl')
    ap.add_argument('--out_dir', default=None,
                    help='Output directory (default: same dir as --pkl)')
    # Optional sim-to-real transform
    ap.add_argument('--sim_to_real_scale', type=float, default=0.045,
                    help='Uniform scale from sim to real (default: 0.045)')
    ap.add_argument('--sim_to_real_offset', type=float, nargs=3,
                    metavar=('X', 'Y', 'Z'), default=[0.5, 0.0, -0.039],
                    help='Translation offset in real world metres after rotation (default: 0.5 0.0 -0.039)')
    ap.add_argument('--sim_to_real_rotation_z_deg', type=float, default=90.0,
                    help='Rotation about Z axis from sim to real world in degrees (default: 90)')
    args = ap.parse_args()

    pkl_path = Path(args.pkl)
    out_dir  = Path(args.out_dir) if args.out_dir else pkl_path.parent
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / (pkl_path.stem + '_camframe.pkl')

    with open(pkl_path, 'rb') as f:
        data = pickle.load(f)

    cam_viewmat = data['cam_viewmat']
    R, t = _world_to_cam_transform(cam_viewmat)

    print(f'Input : {pkl_path}')
    print(f'cam_viewmat [dist pitch yaw tx ty tz]: {cam_viewmat}')
    print(f'R (world → cam):\n{R.round(6)}')
    print(f't (world → cam, metres): {t.round(4)}')
    print()

    sim_to_real = {
        'scale':          args.sim_to_real_scale,
        'offset_xyz':     args.sim_to_real_offset,
        'rotation_z_deg': args.sim_to_real_rotation_z_deg,
    }

    out = transform_traj(data, R, t, sim_to_real=sim_to_real)

    print('Sanity checks:')
    _sanity_check(data, out, R, t, cam_viewmat)
    print()

    print('Camera extrinsics (sim world):')
    ex_sim = out['cam_extrinsics_sim']
    print(f'  position : {[round(v, 4) for v in ex_sim["position_sim"]]}')
    print(f'  quat wxyz: {[round(v, 4) for v in ex_sim["quat_wxyz_sim"]]}')

    print('Camera intrinsics:')
    intr = out['cam_intrinsics']
    print(f'  fov={intr["fov_deg"]}°  resolution={intr["resolution"]}  '
          f'f={intr["fx_px"]}px  cx=cy={intr["cx_px"]}px')

    if 'cam_extrinsics_real' in out:
        print('Camera extrinsics (real world):')
        ex_real = out['cam_extrinsics_real']
        print(f'  position : {[round(v, 4) for v in ex_real["position_real"]]}')
        print(f'  quat wxyz: {[round(v, 4) for v in ex_real["quat_wxyz_real"]]}')
    print()

    with open(out_path, 'wb') as f:
        pickle.dump(out, f, protocol=4)

    print(f'Saved → {out_path}')


if __name__ == '__main__':
    main()
