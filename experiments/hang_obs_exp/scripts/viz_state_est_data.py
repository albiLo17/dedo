"""
Visual sanity-check for a collected state-estimation HDF5
(collect_state_est_data.py output).

Renders a grid of sampled frames; each panel overlays, in the world frame:
  - the partial point cloud           (gray dots)
  - the GT cloth mesh vertices        (light blue)
  - the hole-loop vertices            (red)
  - the hole centroid                 (green star)

Use this to confirm BEFORE training that (a) the point cloud lines up with the
GT mesh, (b) the stored hole-vertex indices actually pick out the hole, and
(c) the hole centroid sits where the hole is. Output PNG is written into the
workspace (visible in VSCode), not /tmp.

Usage:
  python experiments/hang_obs_exp/scripts/viz_state_est_data.py \
      --h5 experiments/hang_obs_exp/data/state_est/dedo_hang.h5 \
      --out experiments/hang_obs_exp/data/state_est/sanity.png \
      --n_panels 6
"""
import argparse

import h5py
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402


def _iter_steps(cloth_grp):
    tg = cloth_grp['trajectory_0']
    return sorted(k for k in tg if k.startswith('step_'))


def _fig_to_rgb(fig):
    """Render an Agg figure to an (H, W, 3) uint8 array."""
    fig.canvas.draw()
    return np.asarray(fig.canvas.buffer_rgba())[:, :, :3].copy()


def animate_cloth(cg, ck, out_path, fps):
    """Write an MP4 of one cloth's trajectory: per recorded step, a 3-D scatter
    of the partial point cloud + GT mesh + hole verts + centroid. Frames come
    straight from the HDF5, so this shows exactly what the model trains on."""
    import imageio
    hv = np.asarray(cg.attrs['hole_vertex_indices'], dtype=np.int64)
    src = cg.attrs.get('source', '?')
    yaw = float(cg.attrs.get('cam_yaw', float('nan')))
    steps = _iter_steps(cg)

    # Fixed axis limits across the whole trajectory so the view doesn't jiggle.
    allp = np.concatenate(
        [cg[f'trajectory_0/{s}/positions'][:] for s in steps], axis=0)
    lo, hi = allp.min(0), allp.max(0)
    pad = 0.1 * (hi - lo + 1e-6)
    lo, hi = lo - pad, hi + pad

    frames = []
    fig = plt.figure(figsize=(6, 5))
    for i, s in enumerate(steps):
        pos = cg[f'trajectory_0/{s}/positions'][:]
        pcd = cg[f'trajectory_0/{s}/pointclouds/cam_0'][:]
        hole = pos[hv]
        centroid = hole.mean(0)
        ax = fig.add_subplot(111, projection='3d')
        ax.scatter(pcd[:, 0], pcd[:, 1], pcd[:, 2], s=2, c='0.6', alpha=0.4)
        ax.scatter(pos[:, 0], pos[:, 1], pos[:, 2], s=6, c='tab:blue', alpha=0.5)
        ax.scatter(hole[:, 0], hole[:, 1], hole[:, 2], s=25, c='tab:red')
        ax.scatter(*centroid, s=160, marker='*', c='tab:green')
        ax.set_xlim(lo[0], hi[0]); ax.set_ylim(lo[1], hi[1]); ax.set_zlim(lo[2], hi[2])
        ax.set_title(f'{ck}  src={src}  cam_yaw={yaw:.0f}  '
                     f'step {i + 1}/{len(steps)}', fontsize=9)
        ax.tick_params(labelsize=5)
        frames.append(_fig_to_rgb(fig))
        fig.clf()
    plt.close(fig)

    writer = imageio.get_writer(
        out_path, fps=fps, codec='libx264', quality=8,
        macro_block_size=2, pixelformat='yuv420p',
        ffmpeg_params=['-movflags', '+faststart'])
    for fr in frames:
        writer.append_data(np.ascontiguousarray(fr))
    writer.close()
    print(f'wrote {out_path}  ({len(frames)} frames @ {fps} fps)')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--h5', required=True)
    ap.add_argument('--out', default=None,
                    help='Output PNG. Defaults next to the h5.')
    ap.add_argument('--n_panels', type=int, default=6)
    ap.add_argument('--split', default='training',
                    choices=['training', 'validation'])
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--mp4', action='store_true',
                    help='Also write per-cloth trajectory MP4s (one full '
                         'rollout each) so you can watch the generated motion.')
    ap.add_argument('--mp4_cloths', type=int, default=3,
                    help='How many cloths to animate when --mp4 is set.')
    ap.add_argument('--fps', type=int, default=10,
                    help='Frame rate for --mp4 videos.')
    args = ap.parse_args()
    out = args.out or args.h5.rsplit('.', 1)[0] + '_sanity.png'

    rng = np.random.default_rng(args.seed)
    with h5py.File(args.h5, 'r') as f:
        grp = f[args.split]
        cloth_keys = sorted(grp.keys())
        if not cloth_keys:
            raise SystemExit(f'no cloths in split {args.split!r}')

        # Sample (cloth, step) pairs across distinct cloths where possible.
        panels = []
        for _ in range(args.n_panels):
            ck = cloth_keys[rng.integers(len(cloth_keys))]
            cg = grp[ck]
            steps = _iter_steps(cg)
            sk = steps[rng.integers(len(steps))]
            panels.append((ck, sk))

        ncol = 3
        nrow = int(np.ceil(len(panels) / ncol))
        fig = plt.figure(figsize=(5 * ncol, 4.5 * nrow))
        for i, (ck, sk) in enumerate(panels):
            cg = grp[ck]
            hv = np.asarray(cg.attrs['hole_vertex_indices'], dtype=np.int64)
            src = cg.attrs.get('source', '?')
            pos = cg[f'trajectory_0/{sk}/positions'][:]
            pcd = cg[f'trajectory_0/{sk}/pointclouds/cam_0'][:]
            hole = pos[hv]
            centroid = hole.mean(0)

            ax = fig.add_subplot(nrow, ncol, i + 1, projection='3d')
            ax.scatter(pcd[:, 0], pcd[:, 1], pcd[:, 2], s=2, c='0.6',
                       alpha=0.4, label='pcd')
            ax.scatter(pos[:, 0], pos[:, 1], pos[:, 2], s=6,
                       c='tab:blue', alpha=0.5, label='mesh')
            ax.scatter(hole[:, 0], hole[:, 1], hole[:, 2], s=25,
                       c='tab:red', label='hole verts')
            ax.scatter(*centroid, s=140, marker='*', c='tab:green',
                       label='centroid')
            ax.set_title(f'{ck}/{sk}\nsrc={src}  V={pos.shape[0]} '
                         f'hole={len(hv)}', fontsize=8)
            ax.tick_params(labelsize=5)
            if i == 0:
                ax.legend(fontsize=6, loc='upper left')
        fig.suptitle(f'state-est data sanity — {args.h5} [{args.split}]',
                     fontsize=11)
        fig.tight_layout()
        fig.savefig(out, dpi=120)
        print(f'wrote {out}')

        # Per-cloth trajectory MP4s.
        if args.mp4:
            pick = cloth_keys[:args.mp4_cloths]
            for ck in pick:
                vid = (args.h5.rsplit('.', 1)[0]
                       + f'_{args.split}_{ck}.mp4')
                animate_cloth(grp[ck], ck, vid, args.fps)


if __name__ == '__main__':
    main()
