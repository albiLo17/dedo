"""
Visualize the index structure of the cloth point cloud across a scripted
HangProcCloth trajectory. Mirrors the reference real-world viser loop
(make_pointcloud + _index_colors + add_point_cloud) but for the sim.

For each captured frame we render three panels side by side:

  1. RGB image — the camera view (context).
  2. Raster-ordered PCD — every cloth pixel back-projected, KEPT in raster
     scan order (no subsample). Colored by index. A clean gradient = points
     are ordered along the depth image's raster scan (top-to-bottom rows,
     left-to-right cols), restricted to the cloth's segmentation mask.
  3. Policy PCD — what the BC collector actually saves and the policy sees:
     cloth_only_pcd → np.random.choice subsample to n_points. Colored by
     index in the (n_points, 3) array. Looks like salt-and-pepper noise
     because the random subsample destroys any spatial ordering.

Reveals: there IS deterministic structure in the raw raster pcd, but the
collector's `np.random.choice` shuffle wipes it before the policy ever
sees it. If you want structure available to the policy, change the
sampling strategy (e.g. raster-strided, FPS, or sort by something
meaningful after sampling).

Output (default inside the repo so it shows in the editor file tree):
  experiments/hang_obs_exp/pcd_structure/pcd_structure.mp4
"""
import argparse
import os
import sys
import io

import numpy as np

_p = argparse.ArgumentParser(description=__doc__,
                             formatter_class=argparse.RawDescriptionHelpFormatter)
_p.add_argument('--seed', type=int, default=42)
_p.add_argument('--sim_freq', type=int, default=500)
_p.add_argument('--steps_per_action', type=int, default=8)
_p.add_argument('--n_points', type=int, default=512,
                help='Policy-pcd subsample size (matches collector default).')
_p.add_argument('--cam_resolution', type=int, default=200,
                help='Depth/RGB capture resolution (HxW).')
_p.add_argument('--frame_stride', type=int, default=2,
                help='Capture every Nth control step (during traj AND settle).')
_p.add_argument('--settle_steps', type=int, default=240,
                help='How many extra sim sub-steps to run after the scripted '
                     'traj ends, so the settle dynamics show up in the video.')
_p.add_argument('--fps', type=int, default=15)
_p.add_argument('--point_size', type=float, default=10.0,
                help='matplotlib scatter point size for the 3D panels.')
_p.add_argument('--cmap', type=str, default='viridis')
_p.add_argument('--out', type=str,
                default=os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                     '..', 'pcd_structure'))
extra = _p.parse_args()
extra.out = os.path.abspath(extra.out)
os.makedirs(extra.out, exist_ok=True)

# --- dedo args (sys.argv hijack pattern, same as sweep_cloth_physics) ----
sys.argv = [
    'viz_pcd_index_structure',
    '--env=HangProcCloth-v1',
    '--cam_resolution=-1',
    '--seed', str(extra.seed),
    '--max_episode_len', '100000',          # never auto-trigger make_final_steps
    f'--sim_freq={extra.sim_freq}',
    f'--sim_steps_per_action={extra.steps_per_action}',
]
from dedo.utils.args import get_args_parser, args_postprocess  # noqa: E402
from dedo.envs.deform_env import DeformEnv                      # noqa: E402
import gym                                                      # noqa: E402
import imageio                                                  # noqa: E402
import matplotlib                                               # noqa: E402
matplotlib.use('Agg')
import matplotlib.pyplot as plt                                 # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _helpers import RetryResetEnv, build_hole_aware_waypoints  # noqa: E402
from _bc_obs_helpers import (                                   # noqa: E402
    capture_rgb_depth, cloth_only_pcd, depth_to_pcd, proj_matrix)
from dedo.demo_preset import build_traj, merge_traj             # noqa: E402

args, _ = get_args_parser()
args_postprocess(args)
args.debug = False
args.viz = False


def _unwrap(env):
    u = env
    while hasattr(u, 'env'):
        u = u.env
        if isinstance(u, DeformEnv):
            return u
    return u


def raster_cloth_pcd(depth, seg, view, proj, deform_id):
    """Mirror of depth_to_pcd but WITHOUT the np.random.choice subsample —
    returns valid cloth-masked points in their original raster scan order
    (top-to-bottom row, left-to-right col). Index in the returned array
    therefore reflects pixel scan position."""
    H, W = depth.shape
    v = np.asarray(view, dtype=np.float64).reshape(4, 4, order='F')
    p = np.asarray(proj, dtype=np.float64).reshape(4, 4, order='F')
    inv_vp = np.linalg.inv(p @ v)
    ys, xs = np.meshgrid(np.arange(H), np.arange(W), indexing='ij')
    u = (xs.astype(np.float64) + 0.5) / W * 2.0 - 1.0
    vv = 1.0 - (ys.astype(np.float64) + 0.5) / H * 2.0
    z = depth * 2.0 - 1.0
    clip = np.stack([u, vv, z, np.ones_like(z)], axis=-1).reshape(-1, 4)
    world_h = clip @ inv_vp.T
    world = world_h[:, :3] / world_h[:, 3:4]
    valid = ((depth.reshape(-1) < 0.999)
             & ~np.isnan(world).any(axis=1)
             & (seg.reshape(-1) == int(deform_id)))
    return world[valid].astype(np.float32)


def render_frame(rgb, raster_pts, policy_pts, lims, t_label):
    """Build one composite frame (RGB | raster pcd | policy pcd)."""
    fig = plt.figure(figsize=(12, 4.2))
    ax_rgb = fig.add_subplot(1, 3, 1)
    ax_rgb.imshow(rgb)
    ax_rgb.set_title('RGB (camera view)')
    ax_rgb.set_xticks([]); ax_rgb.set_yticks([])

    cmap = plt.get_cmap(extra.cmap)

    def _plot_pcd(ax, pts, title):
        ax.set_title(title)
        if len(pts) == 0:
            ax.text2D(0.5, 0.5, 'empty', ha='center', transform=ax.transAxes)
        else:
            colors = cmap(np.linspace(0.0, 1.0, len(pts)))
            ax.scatter(pts[:, 0], pts[:, 1], pts[:, 2],
                       c=colors, s=extra.point_size,
                       marker='.', depthshade=False, linewidths=0)
        ax.set_xlim(lims[0]); ax.set_ylim(lims[1]); ax.set_zlim(lims[2])
        ax.set_xlabel('x'); ax.set_ylabel('y'); ax.set_zlabel('z')
        ax.view_init(elev=15, azim=-60)

    ax_r = fig.add_subplot(1, 3, 2, projection='3d')
    _plot_pcd(ax_r, raster_pts,
              f'Raster-ordered PCD  (N={len(raster_pts)})')
    ax_p = fig.add_subplot(1, 3, 3, projection='3d')
    _plot_pcd(ax_p, policy_pts,
              f'Policy PCD (random subsample)  (N={len(policy_pts)})')

    fig.suptitle(f'Color = index (viridis 0→1)   |   {t_label}', fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.94])

    buf = io.BytesIO()
    fig.savefig(buf, format='png', dpi=110)
    plt.close(fig)
    buf.seek(0)
    return imageio.imread(buf)[..., :3]


def main():
    np.random.seed(extra.seed)
    env = gym.make(args.env, args=args)
    env = RetryResetEnv(env)
    env.seed(extra.seed)
    env.reset()
    deform = _unwrap(env)

    # Build trajectory
    wp = build_hole_aware_waypoints(deform)  # waypoint_scale default 1.0
    if wp is None:
        raise RuntimeError('build_hole_aware_waypoints returned None')
    ctrl_freq = args.sim_freq / args.sim_steps_per_action
    _, va = build_traj(deform, wp, 'a', anchor_idx=0,
                       ctrl_freq=ctrl_freq, robot=None)
    _, vb = build_traj(deform, wp, 'b', anchor_idx=1,
                       ctrl_freq=ctrl_freq, robot=None)
    traj = merge_traj(va, vb)

    # Pre-pass: capture initial frame to compute a stable axis box (so the
    # 3D view doesn't jitter frame-to-frame from autoscale).
    rgb0, depth0, seg0, view0, proj0 = capture_rgb_depth(
        deform, extra.cam_resolution, extra.cam_resolution)
    raster0 = raster_cloth_pcd(depth0, seg0, view0, proj0, deform.deform_id)
    if len(raster0) == 0:
        raise RuntimeError('No cloth pixels visible at frame 0 — '
                           'check seg mask / camera view.')
    pad = 1.5
    mn, mx = raster0.min(0) - pad, raster0.max(0) + pad
    # Widen vertical so anchors moving over the peg stay in frame.
    mn[2] -= 2.0; mx[2] += 2.0
    lims = [(mn[0], mx[0]), (mn[1], mx[1]), (mn[2], mx[2])]

    frames = []

    def _capture(label):
        rgb, depth, seg, view, proj = capture_rgb_depth(
            deform, extra.cam_resolution, extra.cam_resolution)
        raster = raster_cloth_pcd(depth, seg, view, proj, deform.deform_id)
        policy = cloth_only_pcd(depth, seg, view, proj,
                                deform.deform_id, extra.n_points)
        frames.append(render_frame(rgb, raster, policy, lims, label))

    print(f'[viz] trajectory length: {len(traj)} steps', flush=True)
    for step in range(len(traj)):
        act = np.clip(traj[step] / DeformEnv.MAX_ACT_VEL,
                      -1.0, 1.0).astype(np.float32)
        env.step(act)
        if step % extra.frame_stride == 0:
            _capture(f'traj step {step}/{len(traj)}')
    print(f'[viz] traj captured ({len(frames)} frames). Settling ...',
          flush=True)

    # Manual settle: step sim directly so we can keep capturing pcds. The
    # env's auto make_final_steps is bypassed (max_episode_len is huge).
    for s in range(extra.settle_steps):
        deform.sim.stepSimulation()
        if s % extra.frame_stride == 0:
            _capture(f'settle step {s}/{extra.settle_steps}')

    env.close()

    out_path = os.path.join(extra.out, 'pcd_structure.mp4')
    imageio.mimwrite(out_path, frames, fps=extra.fps, codec='libx264',
                     quality=8, macro_block_size=None)
    print(f'[viz] wrote {out_path} ({len(frames)} frames)', flush=True)


if __name__ == '__main__':
    main()
