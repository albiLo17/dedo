"""
Diagnostic: visualize what the camera sees across a full scripted-demo
trajectory, so we can iterate on cam_viewmat without re-collecting demos.

Runs one scripted episode under user-provided cam_viewmat / cam_resolution
/ fov, then saves a multi-row PNG sampling N frames evenly across the
trajectory plus the post-settle frame. Each row shows:

    RGB  |  depth heatmap  |  PCD overlay on RGB  |  valid PCD mask

Use the saved PNG to eyeball:
  * Is the cloth fully in frame across ALL phases (init, hover, thread,
    hold, post-settle)?
  * Is PCD coverage uniform across the cloth surface, or sparse/biased?
  * Are background pixels (peg base / floor) eating the PCD budget?

Usage examples:

  # Current default camera (matches train_privileged.py)
  python experiments/hang_obs_exp/scripts/_diag_pcd_framing.py \
      --out /tmp/framing_default.png \
      --dist 9.0 --pitch -25 --yaw 45 --tx 0 --ty 0.5 --tz 6.5

  # Proposed zoomed-out camera
  python experiments/hang_obs_exp/scripts/_diag_pcd_framing.py \
      --out /tmp/framing_zoomed.png \
      --dist 14 --pitch -15 --yaw 45 --tx 0 --ty 0 --tz 5.5

  # Same but bigger frames + more samples + 1024 PCD points
  python experiments/hang_obs_exp/scripts/_diag_pcd_framing.py \
      --out /tmp/framing_wide.png \
      --dist 14 --pitch -15 --tz 5.5 \
      --cam_resolution 192 --n_samples 6 --n_points 1024
"""
import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT))

import gym
import numpy as np
import pybullet

import matplotlib
matplotlib.use('Agg')  # headless save; no display server needed
import matplotlib.pyplot as plt

import dedo  # noqa: F401
from dedo.envs.deform_env import DeformEnv
from dedo.utils.args import get_args_parser, args_postprocess
from dedo.utils.mesh_utils import get_mesh_data
from dedo.demo_preset import build_traj, merge_traj

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _helpers import RetryResetEnv, build_hole_aware_waypoints  # noqa: E402
from _bc_obs_helpers import (  # noqa: E402
    capture_rgb_depth, depth_to_pcd, resolve_deform,
    get_hole_indices, measure_hole_radius,
    check_hanging_on_peg, check_threaded_topological, check_legacy)


# ---------------------------------------------------------------------------
# Argparse
# ---------------------------------------------------------------------------
parser = argparse.ArgumentParser()
parser.add_argument('--out', type=str, required=True,
                    help='Output PNG path.')
parser.add_argument('--seed', type=int, default=2026)
parser.add_argument('--cam_resolution', type=int, default=128,
                    help='H=W for both RGB and depth render in the '
                         'diagnostic. Pick separately from collection '
                         'cam_resolution.')
parser.add_argument('--n_points', type=int, default=512,
                    help='PCD point budget per frame (only affects the '
                         'overlay panel; cropping decisions don\'t '
                         'depend on this).')
parser.add_argument('--fov', type=float, default=60.0,
                    help='Vertical FOV in degrees. Wider FOV captures '
                         'more scene but bends straight lines and warps '
                         'back-projected PCD.')
parser.add_argument('--dist', type=float, default=9.0)
parser.add_argument('--pitch', type=float, default=-25.0)
parser.add_argument('--yaw', type=float, default=45.0)
parser.add_argument('--tx', type=float, default=0.0)
parser.add_argument('--ty', type=float, default=0.5)
parser.add_argument('--tz', type=float, default=6.5)
parser.add_argument('--n_samples', type=int, default=5,
                    help='Number of trajectory points to capture, evenly '
                         'spaced from step 0 to terminal. The post-settle '
                         'frame is added as an extra row beyond the '
                         'in-traj samples, so the PNG has n_samples + 1 '
                         'rows total.')
parser.add_argument('--max_act_vel', type=float, default=3.5,
                    help='Action normalization cap. Default 3.5 just '
                         'above the scripted controller\'s peak velocity '
                         'so demos don\'t saturate. Lower values give '
                         'better BC SNR if you re-collect under this '
                         'setting.')
parser.add_argument('--max_episode_len', type=int, default=200)
extra = parser.parse_args()


# ---------------------------------------------------------------------------
# Build env with the user-specified camera. cam_resolution must be > 0 so
# the dedo env initializes a renderable camera; for the actual capture we
# call pybullet.getCameraImage directly with our own view/proj matrices
# (so this script controls the camera entirely, not args.cam_resolution).
# ---------------------------------------------------------------------------
sys.argv = [
    '_diag_pcd_framing',
    '--env=HangProcCloth-v1',
    f'--cam_resolution={extra.cam_resolution}',
    '--num_envs=0',
    '--total_env_steps=0',
    '--seed', str(extra.seed),
    '--max_episode_len', str(extra.max_episode_len),
    '--cam_viewmat',
    str(extra.dist), str(extra.pitch), str(extra.yaw),
    str(extra.tx), str(extra.ty), str(extra.tz),
]
args, _ = get_args_parser()
args_postprocess(args)
args.viz = False
args.debug = False
args.uint8_pixels = True

DeformEnv.MAX_ACT_VEL = float(extra.max_act_vel)

env = gym.make(args.env, args=args)
env = RetryResetEnv(env)
env.seed(extra.seed)
deform = resolve_deform(env)
ctrl_freq = args.sim_freq / args.sim_steps_per_action

print(f'\n=== PCD framing diagnostic ===')
print(f'  cam: dist={extra.dist} pitch={extra.pitch} yaw={extra.yaw} '
      f'target=({extra.tx}, {extra.ty}, {extra.tz})')
print(f'  cam_resolution={extra.cam_resolution} fov={extra.fov}')
print(f'  n_points={extra.n_points}  n_samples={extra.n_samples}')


# ---------------------------------------------------------------------------
# Make one custom view matrix using our parameters (don't rely on the
# env's _cam_viewmat — that gets clobbered in some env reset paths).
# ---------------------------------------------------------------------------
def _make_view_matrix():
    return pybullet.computeViewMatrixFromYawPitchRoll(
        cameraTargetPosition=[extra.tx, extra.ty, extra.tz],
        distance=extra.dist,
        yaw=extra.yaw,
        pitch=extra.pitch,
        roll=0,
        upAxisIndex=2)


def _make_proj_matrix():
    return pybullet.computeProjectionMatrixFOV(
        fov=extra.fov, aspect=1.0, nearVal=0.1, farVal=30.0)


VIEW = _make_view_matrix()
PROJ = _make_proj_matrix()


def _capture(deform, width, height):
    """Render RGB + depth via OUR view/proj (not deform_env._cam_viewmat)."""
    _, _, rgb_raw, depth_buf, _ = deform.sim.getCameraImage(
        width=width, height=height,
        viewMatrix=VIEW, projectionMatrix=PROJ,
        renderer=pybullet.ER_BULLET_HARDWARE_OPENGL)
    rgb = np.asarray(rgb_raw, dtype=np.uint8).reshape(height, width, 4)[:, :, :3]
    depth = np.asarray(depth_buf, dtype=np.float64).reshape(height, width)
    return rgb, depth


def _overlay_pcd(rgb, pcd_world, width, height):
    """Project world-coord points back to screen and paint magenta dots."""
    if len(pcd_world) == 0:
        return rgb.copy()
    v = np.asarray(VIEW, dtype=np.float64).reshape(4, 4, order='F')
    p = np.asarray(PROJ, dtype=np.float64).reshape(4, 4, order='F')
    pts_h = np.concatenate(
        [pcd_world, np.ones((len(pcd_world), 1))], axis=1)
    clip = (p @ v @ pts_h.T).T
    ok = clip[:, 3] != 0.0
    clip = clip[ok]
    ndc = clip[:, :3] / clip[:, 3:4]
    u = ((ndc[:, 0] + 1.0) * 0.5 * width).astype(np.int32)
    vv = ((1.0 - (ndc[:, 1] + 1.0) * 0.5) * height).astype(np.int32)
    in_view = ((u >= 0) & (u < width) & (vv >= 0) & (vv < height)
               & (ndc[:, 2] > -1) & (ndc[:, 2] < 1))
    u, vv = u[in_view], vv[in_view]
    out = rgb.copy()
    for du in (-1, 0, 1):
        for dv in (-1, 0, 1):
            u2 = np.clip(u + du, 0, width - 1)
            v2 = np.clip(vv + dv, 0, height - 1)
            out[v2, u2] = [255, 0, 255]
    return out


def _cloth_in_frame_fraction(deform, width, height):
    """Fraction of cloth mesh vertices that project inside the camera frame
    (and in front of the camera). 0.0 means cloth is entirely off-screen,
    1.0 means fully in frame. This is the headline number for picking a
    camera that doesn't crop the cloth."""
    _, verts = get_mesh_data(deform.sim, deform.deform_id)
    verts = np.asarray(verts, dtype=np.float64)
    verts = verts[~np.isnan(verts).any(axis=1)]
    if len(verts) == 0:
        return float('nan')
    v = np.asarray(VIEW, dtype=np.float64).reshape(4, 4, order='F')
    p = np.asarray(PROJ, dtype=np.float64).reshape(4, 4, order='F')
    pts_h = np.concatenate([verts, np.ones((len(verts), 1))], axis=1)
    clip = (p @ v @ pts_h.T).T
    ok_w = clip[:, 3] > 0  # in front of camera
    clip = clip[ok_w]
    if len(clip) == 0:
        return 0.0
    ndc = clip[:, :3] / clip[:, 3:4]
    in_view = ((ndc[:, 0] >= -1) & (ndc[:, 0] <= 1)
               & (ndc[:, 1] >= -1) & (ndc[:, 1] <= 1)
               & (ndc[:, 2] >= -1) & (ndc[:, 2] <= 1))
    return float(in_view.mean()) * (len(clip) / len(verts))


# ---------------------------------------------------------------------------
# Reset + build a scripted trajectory.
# ---------------------------------------------------------------------------
env.reset()
hole_idx = get_hole_indices(deform)
if not hole_idx:
    raise RuntimeError(
        '[_diag_pcd_framing] no hole loop on this cloth — try a different '
        '--seed.')
hole_radius = measure_hole_radius(deform, hole_idx)

wp = build_hole_aware_waypoints(deform)
if wp is None:
    raise RuntimeError('[_diag_pcd_framing] build_hole_aware_waypoints '
                       'failed; try a different --seed.')
_, va = build_traj(deform, wp, 'a', anchor_idx=0,
                   ctrl_freq=ctrl_freq, robot=None)
_, vb = build_traj(deform, wp, 'b', anchor_idx=1,
                   ctrl_freq=ctrl_freq, robot=None)
traj = merge_traj(va, vb)
traj_len = len(traj)
print(f'  traj_len = {traj_len}  '
      f'peak_|vel| = {float(np.abs(traj).max()):.3f} '
      f'(MAX_ACT_VEL = {DeformEnv.MAX_ACT_VEL})')

sample_indices = np.linspace(0, traj_len - 1, extra.n_samples,
                             dtype=np.int64).tolist()
print(f'  sample steps = {sample_indices}  (+ 1 post-settle frame)')

# ---------------------------------------------------------------------------
# Step the scripted controller; capture at each sample step.
# ---------------------------------------------------------------------------
panels = []  # list of (label, rgb, depth, pcd_overlay, cloth_in_frame_frac)
last_action = np.zeros_like(traj[0])
done = False
captured_steps = set(sample_indices)
step = 0
while not done and step < extra.max_episode_len:
    if step in captured_steps:
        rgb, depth = _capture(deform, extra.cam_resolution,
                              extra.cam_resolution)
        pcd = depth_to_pcd(depth, VIEW, PROJ, extra.n_points)
        overlay = _overlay_pcd(rgb, pcd, extra.cam_resolution,
                               extra.cam_resolution)
        in_frame = _cloth_in_frame_fraction(
            deform, extra.cam_resolution, extra.cam_resolution)
        n_valid = int((depth < 0.999).sum())
        panels.append((f'step {step}/{traj_len-1}', rgb, depth, overlay,
                       in_frame, n_valid))
        print(f'  captured step {step:>3d}  '
              f'cloth_in_frame={in_frame*100:5.1f}%  '
              f'n_valid_depth_px={n_valid}')

    act_unscaled = traj[step] if step < traj_len else last_action
    act = np.clip(act_unscaled / DeformEnv.MAX_ACT_VEL,
                  -1.0, 1.0).astype(np.float32)
    _, _, done, info = env.step(act)
    last_action = act_unscaled
    step += 1

# Post-settle frame (after env's make_final_steps gravity-settle ran inside
# the final step()'s done=True path).
rgb, depth = _capture(deform, extra.cam_resolution, extra.cam_resolution)
pcd = depth_to_pcd(depth, VIEW, PROJ, extra.n_points)
overlay = _overlay_pcd(rgb, pcd, extra.cam_resolution, extra.cam_resolution)
in_frame = _cloth_in_frame_fraction(
    deform, extra.cam_resolution, extra.cam_resolution)
n_valid = int((depth < 0.999).sum())
panels.append(('post-settle', rgb, depth, overlay, in_frame, n_valid))
print(f'  captured post-settle  cloth_in_frame={in_frame*100:5.1f}%  '
      f'n_valid_depth_px={n_valid}')


# ---------------------------------------------------------------------------
# Final success metrics so the user knows whether this seed was a successful
# scripted demo (irrelevant for framing per se, but useful context).
# ---------------------------------------------------------------------------
from _bc_obs_helpers import get_hole_loops
s_h = check_hanging_on_peg(deform, hole_idx, hole_radius, 1.2)
s_t, w = check_threaded_topological(deform, get_hole_loops(deform))
s_l = check_legacy(deform, hole_idx, hole_radius, 1.2)
print(f'\n  success: hanging={s_h}  topological={s_t}  legacy={s_l}  '
      f'(seed={extra.seed})')


# ---------------------------------------------------------------------------
# Render a multi-row PNG.
#   Columns: RGB | depth | PCD overlay | valid-depth-mask
#   Rows:    one per captured step + post-settle
# ---------------------------------------------------------------------------
n_rows = len(panels)
fig, axes = plt.subplots(n_rows, 4, figsize=(14, 3.4 * n_rows))
if n_rows == 1:
    axes = axes[None, :]

for row, (label, rgb, depth, overlay, in_frame, n_valid) in enumerate(panels):
    axes[row, 0].imshow(rgb)
    axes[row, 0].set_title(
        f'{label}\nRGB  ({extra.cam_resolution}×{extra.cam_resolution})')
    axes[row, 0].axis('off')

    axes[row, 1].imshow(depth, cmap='viridis', vmin=0.0, vmax=1.0)
    axes[row, 1].set_title(
        f'depth  (1.0 = bg)\n{n_valid} valid px')
    axes[row, 1].axis('off')

    axes[row, 2].imshow(overlay)
    axes[row, 2].set_title(
        f'PCD overlay (N={extra.n_points})\n'
        f'cloth in frame: {in_frame*100:.1f}%')
    axes[row, 2].axis('off')

    valid_mask = (depth < 0.999).astype(np.float32)
    axes[row, 3].imshow(valid_mask, cmap='gray')
    axes[row, 3].set_title('valid depth mask\n(white = used for PCD)')
    axes[row, 3].axis('off')

cam_str = (f'dist={extra.dist}, pitch={extra.pitch}, yaw={extra.yaw}, '
           f'target=({extra.tx}, {extra.ty}, {extra.tz}), fov={extra.fov}°')
fig.suptitle(
    f'PCD framing — {cam_str}  |  seed={extra.seed}  '
    f'success(h/t/l)={int(s_h)}/{int(s_t)}/{int(s_l)}',
    fontsize=11, y=1.0)
plt.tight_layout()
plt.savefig(extra.out, dpi=110, bbox_inches='tight')
plt.close()
print(f'\nSaved → {extra.out}')

env.close()
