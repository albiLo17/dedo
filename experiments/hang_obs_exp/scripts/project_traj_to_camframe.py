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

Usage:
  python experiments/hang_obs_exp/scripts/project_traj_to_camframe.py \\
      --pkl logs/hang_obs_exp/eval_trajs/.../traj_ep001_....pkl

  # custom output directory
  python experiments/hang_obs_exp/scripts/project_traj_to_camframe.py \\
      --pkl logs/.../traj_ep001.pkl --out_dir /tmp/cam_frame_trajs
"""
from __future__ import annotations

import argparse
import pickle
from pathlib import Path

import numpy as np
import pybullet  # pure-math fns work without a physics server


WBOX = 20.0  # grip / goal are stored as world_metres / WBOX


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

def transform_traj(data, R, t):
    """Return a copy of `data` with all spatial fields in camera frame."""
    t_wbox = (t / WBOX).astype(np.float32)

    obs = dict(data['obs'])
    obs['pcd']  = _xf_pcd(np.asarray(obs['pcd'],  dtype=np.float32), R, t)
    obs['goal'] = _xf_positions(np.asarray(obs['goal'], dtype=np.float32), R, t_wbox)
    obs['grip'] = _xf_grip(np.asarray(obs['grip'], dtype=np.float32), R, t_wbox)

    out = dict(data)
    out['obs']       = obs
    out['acts']      = _xf_acts(np.asarray(data['acts'], dtype=np.float32), R)
    out['cam_frame'] = True
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

    out = transform_traj(data, R, t)

    print('Sanity checks:')
    _sanity_check(data, out, R, t, cam_viewmat)
    print()

    with open(out_path, 'wb') as f:
        pickle.dump(out, f, protocol=4)

    print(f'Saved → {out_path}')


if __name__ == '__main__':
    main()
