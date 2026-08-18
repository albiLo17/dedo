#!/usr/bin/env python
"""Render a filmstrip of one HDF5 trajectory's mesh, so a blow-up can be seen.

`_diag_h5_explosions.py` says which trajectories are unstable and by how much;
this says what that looks like. Draws the mesh wireframe at evenly spaced
steps, each panel titled with its own max edge stretch, on a FIXED axis range
taken from the rest pose — so a mesh that inflates visibly leaves the box
instead of being silently re-fit to it, which is exactly how these go unnoticed.

    python _viz_h5_explosion.py --h5 <f.h5> --cloth cloth_00056 --out strip.png
"""
import argparse

import h5py
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from mpl_toolkits.mplot3d.art3d import Line3DCollection  # noqa: E402

INK, GRID, SURFACE = '#0b0b0b', '#d8d7d2', '#fcfcfb'
C_OK, C_BAD = '#12907a', '#d1642f'


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--h5', required=True)
    p.add_argument('--cloth', required=True)
    p.add_argument('--split', default='training')
    p.add_argument('--traj', default='trajectory_0')
    p.add_argument('--n_panels', type=int, default=6)
    p.add_argument('--stretch_thresh', type=float, default=3.0)
    p.add_argument('--out', required=True)
    args = p.parse_args()

    with h5py.File(args.h5, 'r') as h:
        c = h[args.split][args.cloth]
        rest = np.asarray(c['rest_positions'], dtype=np.float64)
        edges = np.asarray(c['edges'], dtype=np.int64)
        t = c[args.traj]
        steps = sorted(k for k in t.keys() if k.startswith('step_'))
        pos = np.stack([np.asarray(t[s]['positions'], dtype=np.float64)
                        for s in steps])
        source = c.attrs.get('source', '?')

    rest_len = np.linalg.norm(rest[edges[:, 0]] - rest[edges[:, 1]], axis=-1)
    keep = rest_len > 1e-9
    edges, rest_len = edges[keep], rest_len[keep]
    lens = np.linalg.norm(pos[:, edges[:, 0]] - pos[:, edges[:, 1]], axis=-1)
    stretch = (lens / rest_len[None, :]).max(axis=1)

    idx = np.linspace(0, len(steps) - 1, args.n_panels).astype(int)

    # Fixed box from the REST pose, generous enough to show a 3x inflation.
    c0 = rest.mean(axis=0)
    r0 = max(np.ptp(rest, axis=0).max(), 1e-6) * 1.6

    fig = plt.figure(figsize=(3.1 * args.n_panels, 3.6), dpi=130,
                     facecolor=SURFACE)
    for k, i in enumerate(idx):
        ax = fig.add_subplot(1, args.n_panels, k + 1, projection='3d')
        ax.set_facecolor(SURFACE)
        v = pos[i]
        segs = np.stack([v[edges[:, 0]], v[edges[:, 1]]], axis=1)
        blown = stretch[i] > args.stretch_thresh
        ax.add_collection3d(Line3DCollection(
            segs, colors=(C_BAD if blown else C_OK), linewidths=0.5, alpha=0.85))
        ax.set_xlim(c0[0] - r0, c0[0] + r0)
        ax.set_ylim(c0[1] - r0, c0[1] + r0)
        ax.set_zlim(c0[2] - r0, c0[2] + r0)
        ax.set_axis_off()
        try:
            ax.set_box_aspect((1, 1, 1))
        except Exception:
            pass
        ax.set_title(f'step {i}\nmax stretch {stretch[i]:.1f}x',
                     fontsize=10, color=(C_BAD if blown else INK), pad=2)

    fig.suptitle(f'{args.split}/{args.cloth}/{args.traj}   source={source}   '
                 f'peak stretch {stretch.max():.1f}x at step {stretch.argmax()}',
                 fontsize=12, color=INK, y=1.02)
    fig.tight_layout()
    fig.savefig(args.out, bbox_inches='tight', facecolor=SURFACE)
    print(f'[viz] wrote {args.out}  (peak stretch {stretch.max():.1f}x)')


if __name__ == '__main__':
    main()
