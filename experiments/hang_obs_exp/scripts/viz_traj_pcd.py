"""
Visualize a trajectory pkl produced by eval_diffusion_bc.py --save_traj_dir.

Outputs (same directory as --pkl by default, override with --out_dir):
  <stem>_keyframes.png   — static 2×3 panel: PCD keyframes + time-series
  <stem>_anim.mp4        — animated 3D PCD with gripper paths

Usage:
  python experiments/hang_obs_exp/scripts/viz_traj_pcd.py \\
      --pkl logs/hang_obs_exp/eval_trajs/.../traj_ep005_....pkl
"""
from __future__ import annotations
import argparse
import pickle
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
from matplotlib.animation import FuncAnimation, FFMpegWriter


WBOX = 20.0  # matches capture_obs() /20 normalisation


def load(pkl_path):
    with open(pkl_path, 'rb') as f:
        d = pickle.load(f)
    pcd  = np.asarray(d['obs']['pcd'],  dtype=np.float32)   # (T, N, 3) world m
    grip = np.asarray(d['obs']['grip'], dtype=np.float32)   # (T, 12)  /WBOX
    goal = np.asarray(d['obs']['goal'], dtype=np.float32)   # (T, 3)   /WBOX
    acts = np.asarray(d['acts'],        dtype=np.float32)   # (T, 6)
    # grip layout: [anc1_pos(3), anc1_vel(3), anc2_pos(3), anc2_vel(3)]
    anc1 = grip[:, 0:3] * WBOX   # world metres
    anc2 = grip[:, 6:9] * WBOX
    goal_world = goal * WBOX      # (T, 3) — constant per episode
    meta = {k: v for k, v in d.items() if k not in ('obs', 'acts')}
    return pcd, anc1, anc2, goal_world, acts, meta


def _scatter3(ax, xyz, c, alpha=0.6, s=1, cmap='viridis', vmin=None, vmax=None):
    return ax.scatter(xyz[:, 0], xyz[:, 1], xyz[:, 2],
                      c=c, s=s, alpha=alpha, cmap=cmap,
                      vmin=vmin, vmax=vmax, depthshade=False)


def _axis_equal_3d(ax):
    """Force equal aspect ratio on a 3-D axis."""
    limits = np.array([ax.get_xlim3d(), ax.get_ylim3d(), ax.get_zlim3d()])
    center = limits.mean(axis=1)
    half   = (limits[:, 1] - limits[:, 0]).max() / 2
    ax.set_xlim3d(center[0] - half, center[0] + half)
    ax.set_ylim3d(center[1] - half, center[1] + half)
    ax.set_zlim3d(center[2] - half, center[2] + half)


def make_keyframes_figure(pcd, anc1, anc2, goal_world, acts, meta, out_path):
    T = len(pcd)
    keyframes = [0, T // 4, T // 2, 3 * T // 4, T - 1]

    fig = plt.figure(figsize=(18, 10))
    fig.suptitle(
        f'Trajectory: {Path(out_path).stem.replace("_keyframes", "")}   '
        f'T={T}  hang={meta.get("success_hanging")} '
        f'leg={meta.get("success_legacy")} '
        f'topo={meta.get("success_topological")}',
        fontsize=10)

    gs = gridspec.GridSpec(2, 5, figure=fig,
                           hspace=0.35, wspace=0.3,
                           left=0.04, right=0.97)

    # ---- Row 0: PCD at each keyframe ----------------------------------------
    colors_per_kf = plt.cm.plasma(np.linspace(0.1, 0.9, len(keyframes)))
    goal_pt = goal_world[0]  # constant per episode

    for col, (ti, kc) in enumerate(zip(keyframes, colors_per_kf)):
        ax = fig.add_subplot(gs[0, col], projection='3d')
        pts = pcd[ti]
        ax.scatter(pts[:, 0], pts[:, 1], pts[:, 2],
                   color=kc, s=1, alpha=0.5, depthshade=False)
        ax.scatter(*anc1[ti], color='red',    s=60, marker='^', zorder=5,
                   label='anc1' if col == 0 else None)
        ax.scatter(*anc2[ti], color='blue',   s=60, marker='^', zorder=5,
                   label='anc2' if col == 0 else None)
        ax.scatter(*goal_pt,  color='gold',   s=80, marker='*', zorder=5,
                   label='goal' if col == 0 else None)
        _axis_equal_3d(ax)
        ax.set_title(f't = {ti}', fontsize=8)
        ax.set_xlabel('x', fontsize=6); ax.set_ylabel('y', fontsize=6)
        ax.set_zlabel('z', fontsize=6)
        ax.tick_params(labelsize=5)
        if col == 0:
            ax.legend(fontsize=5, loc='upper left')

    # ---- Row 1, col 0-1: Gripper trajectories (3D) --------------------------
    ax3d = fig.add_subplot(gs[1, 0:2], projection='3d')
    t_arr = np.arange(T)
    cmap  = plt.cm.viridis
    for i in range(T - 1):
        c = cmap(i / max(T - 1, 1))
        ax3d.plot(anc1[i:i+2, 0], anc1[i:i+2, 1], anc1[i:i+2, 2],
                  color=c, lw=1.2)
        ax3d.plot(anc2[i:i+2, 0], anc2[i:i+2, 1], anc2[i:i+2, 2],
                  color=c, lw=1.2, linestyle='--')
    ax3d.scatter(*goal_pt, color='gold', s=100, marker='*', zorder=5,
                 label='goal')
    # Draw final PCD faintly as spatial context
    _scatter3(ax3d, pcd[-1], c='lightgray', alpha=0.15, s=1)
    ax3d.set_title('Gripper paths (solid=anc1, dashed=anc2)', fontsize=8)
    ax3d.set_xlabel('x', fontsize=6); ax3d.set_ylabel('y', fontsize=6)
    ax3d.set_zlabel('z', fontsize=6)
    ax3d.tick_params(labelsize=5)
    ax3d.legend(fontsize=6)
    _axis_equal_3d(ax3d)

    # ---- Row 1, col 2: Gripper Z + goal Z over time -------------------------
    ax_z = fig.add_subplot(gs[1, 2])
    ax_z.plot(t_arr, anc1[:, 2], color='red',  lw=1.5, label='anc1 z')
    ax_z.plot(t_arr, anc2[:, 2], color='blue', lw=1.5, label='anc2 z')
    ax_z.axhline(goal_pt[2], color='gold', lw=1, ls='--', label='goal z')
    ax_z.set_xlabel('step', fontsize=8); ax_z.set_ylabel('z (m)', fontsize=8)
    ax_z.set_title('Gripper Z height', fontsize=8)
    ax_z.legend(fontsize=7); ax_z.grid(True, alpha=0.3)

    # ---- Row 1, col 3: Action components over time --------------------------
    ax_act = fig.add_subplot(gs[1, 3])
    labels = ['a1x', 'a1y', 'a1z', 'a2x', 'a2y', 'a2z']
    colors = ['#e41a1c', '#377eb8', '#4daf4a', '#984ea3', '#ff7f00', '#a65628']
    for i, (lbl, col) in enumerate(zip(labels, colors)):
        ax_act.plot(t_arr, acts[:, i], color=col, lw=1, alpha=0.8, label=lbl)
    ax_act.set_xlabel('step', fontsize=8); ax_act.set_ylabel('action (norm.)', fontsize=8)
    ax_act.set_title('Action components', fontsize=8)
    ax_act.legend(fontsize=5, ncol=2); ax_act.grid(True, alpha=0.3)

    # ---- Row 1, col 4: PCD centroid XYZ over time ---------------------------
    ax_cen = fig.add_subplot(gs[1, 4])
    centroid = pcd.mean(axis=1)  # (T, 3)
    ax_cen.plot(t_arr, centroid[:, 0], lw=1.2, label='cx', color='tab:red')
    ax_cen.plot(t_arr, centroid[:, 1], lw=1.2, label='cy', color='tab:green')
    ax_cen.plot(t_arr, centroid[:, 2], lw=1.5, label='cz', color='tab:blue')
    ax_cen.axhline(goal_pt[2], color='gold', lw=1, ls='--', label='goal z')
    ax_cen.set_xlabel('step', fontsize=8)
    ax_cen.set_ylabel('centroid (m)', fontsize=8)
    ax_cen.set_title('PCD centroid (cloth centre)', fontsize=8)
    ax_cen.legend(fontsize=7); ax_cen.grid(True, alpha=0.3)

    fig.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'[viz] saved keyframes figure -> {out_path}')


def make_animation(pcd, anc1, anc2, goal_world, meta, out_path, fps=10):
    T = len(pcd)
    goal_pt = goal_world[0]

    fig = plt.figure(figsize=(7, 6))
    ax  = fig.add_subplot(111, projection='3d')
    fig.subplots_adjust(left=0, right=1, bottom=0, top=0.93)

    # Pre-compute global axis limits from all frames
    all_pts = pcd.reshape(-1, 3)
    lim_pts = np.vstack([all_pts, anc1, anc2, goal_pt[None]])
    lo, hi  = lim_pts.min(0), lim_pts.max(0)
    center  = (lo + hi) / 2
    half    = (hi - lo).max() / 2 * 0.6

    def init():
        ax.clear()
        ax.set_xlim3d(center[0]-half, center[0]+half)
        ax.set_ylim3d(center[1]-half, center[1]+half)
        ax.set_zlim3d(center[2]-half, center[2]+half)
        ax.set_xlabel('x'); ax.set_ylabel('y'); ax.set_zlabel('z')
        return []

    def update(ti):
        ax.clear()
        ax.set_xlim3d(center[0]-half, center[0]+half)
        ax.set_ylim3d(center[1]-half, center[1]+half)
        ax.set_zlim3d(center[2]-half, center[2]+half)
        ax.set_xlabel('x', fontsize=7)
        ax.set_ylabel('y', fontsize=7)
        ax.set_zlabel('z', fontsize=7)
        ax.tick_params(labelsize=5)

        # Cloth PCD (current frame)
        pts = pcd[ti]
        ax.scatter(pts[:, 0], pts[:, 1], pts[:, 2],
                   c=pts[:, 2], cmap='Blues', s=1.5, alpha=0.7,
                   vmin=lo[2], vmax=hi[2], depthshade=False)

        # Gripper trail (past steps)
        trail = max(0, ti - 8)
        ax.plot(anc1[trail:ti+1, 0], anc1[trail:ti+1, 1], anc1[trail:ti+1, 2],
                color='red', lw=1.5, alpha=0.8)
        ax.plot(anc2[trail:ti+1, 0], anc2[trail:ti+1, 1], anc2[trail:ti+1, 2],
                color='blue', lw=1.5, alpha=0.8, linestyle='--')
        ax.scatter(*anc1[ti], color='red',  s=60, marker='^', zorder=5)
        ax.scatter(*anc2[ti], color='blue', s=60, marker='^', zorder=5)
        ax.scatter(*goal_pt,  color='gold', s=80, marker='*', zorder=5)

        ax.set_title(
            f't={ti}/{T-1}   hang={meta.get("success_hanging")} '
            f'leg={meta.get("success_legacy")} '
            f'topo={meta.get("success_topological")}',
            fontsize=8)
        return []

    anim = FuncAnimation(fig, update, frames=T,
                         init_func=init, blit=False, interval=1000 // fps)
    writer = FFMpegWriter(fps=fps, bitrate=800,
                          extra_args=['-vcodec', 'libx264',
                                      '-pix_fmt', 'yuv420p',
                                      '-movflags', '+faststart'])
    anim.save(out_path, writer=writer, dpi=120)
    plt.close(fig)
    print(f'[viz] saved animation      -> {out_path}')


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--pkl', required=True,
                   help='Path to traj_ep*.pkl from eval_diffusion_bc.py')
    p.add_argument('--out_dir', default=None,
                   help='Output directory (default: same as --pkl)')
    p.add_argument('--fps', type=int, default=10,
                   help='Animation frame rate (default 10 = 10× slower than '
                        '15 Hz ctrl rate, so motion is easy to follow)')
    p.add_argument('--no_anim', action='store_true',
                   help='Skip animation (faster if you only want the figure)')
    args = p.parse_args()

    pkl_path = Path(args.pkl)
    out_dir  = Path(args.out_dir) if args.out_dir else pkl_path.parent
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = pkl_path.stem

    pcd, anc1, anc2, goal_world, acts, meta = load(pkl_path)
    print(f'[viz] loaded {pkl_path.name}: T={len(pcd)} pcd_pts={pcd.shape[1]}')
    print(f'      PCD z range  : {pcd[...,2].min():.2f} – {pcd[...,2].max():.2f} m')
    print(f'      Anc1 z range : {anc1[:,2].min():.2f} – {anc1[:,2].max():.2f} m')
    print(f'      Goal pos     : {goal_world[0]}')

    make_keyframes_figure(pcd, anc1, anc2, goal_world, acts, meta,
                          out_dir / f'{stem}_keyframes.png')
    if not args.no_anim:
        make_animation(pcd, anc1, anc2, goal_world, meta,
                       out_dir / f'{stem}_anim.mp4', fps=args.fps)


if __name__ == '__main__':
    main()
