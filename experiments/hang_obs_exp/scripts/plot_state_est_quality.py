"""Qualitative + quantitative figures for the state-estimation module.

Consumes the artefacts written by `eval_state_est_quality.py` (the JSON of
metrics and the optional .npz of meshes), so the pictures and the numbers can
never disagree — they come from the same inference calls.

Three figures:

  `*_curves.png`   error and extent_ratio vs MEASURED occlusion. The x-axis is
                   the observable (fraction of the GT mesh not within tau of a
                   point), not the drop fraction we dialled, so it is
                   comparable with the policy arm and with the GarmentLab
                   report.

  `*_qualitative.png`  per-vertex error painted on the predicted mesh, one row
                   per cloth and one column per occlusion level, with the
                   observed cloud drawn underneath. This is the figure that
                   shows HOW the estimate fails — whether it degrades
                   gracefully, collapses toward the mean cloth, or breaks in
                   the unobserved region specifically.

  `*_shrink.png`   predicted vs GT bounding-box diagonal, one point per sample.
                   A systematic slope below the diagonal is the shrinkage
                   failure mode: a model that hedges toward a smaller mesh
                   keeps its per-vertex error respectable while pulling every
                   derived quantity (the hole centroid the policy consumes)
                   toward the cloth centre.
"""

from __future__ import annotations

import argparse
import json

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt          # noqa: E402
import numpy as np                        # noqa: E402


def load_meshes(npz_path):
    """Regroup the flat `<i>_<key>` npz layout back into per-sample dicts."""
    z = np.load(npz_path, allow_pickle=True)
    out = {}
    for k in z.files:
        i, _, key = k.partition('_')
        out.setdefault(int(i), {})[key] = z[k]
    return [out[i] for i in sorted(out)]


def fig_curves(summary, rows, out):
    fig, ax = plt.subplots(1, 3, figsize=(15, 4.2))
    occ = [s['occlusion_mean'] for s in summary]

    ax[0].plot(occ, [s['vert_mean'] * 1000 for s in summary], 'o-', lw=2,
               color='#2b6cb0', label='mean')
    ax[0].plot(occ, [s['vert_mean_p90'] * 1000 for s in summary], 's--', lw=1.5,
               color='#90cdf4', label='p90 across samples')
    ax[0].set_ylabel('per-vertex error (mm)')
    ax[0].set_title('Mesh accuracy vs occlusion')
    ax[0].legend(frameon=False)

    ax[1].plot(occ, [s['hole_err_mean'] * 1000 for s in summary], 'o-', lw=2,
               color='#c05621', label='mean')
    ax[1].plot(occ, [s['hole_err_median'] * 1000 for s in summary], 's--',
               lw=1.5, color='#f6ad55', label='median')
    ax[1].set_ylabel('hole-centroid error (mm)')
    ax[1].set_title('The quantity the policy consumes')
    ax[1].legend(frameon=False)

    ax[2].plot(occ, [s['extent_ratio_mean'] for s in summary], 'o-', lw=2,
               color='#2f855a')
    ax[2].axhline(1.0, color='k', ls=':', lw=1)
    ax[2].set_ylabel('extent ratio (pred / GT)')
    ax[2].set_title('Size fidelity — 1.0 is correct')
    ax[2].set_ylim(0, 1.2)

    for a in ax:
        a.set_xlabel('measured occlusion')
        a.grid(alpha=0.3)
        a.spines[['top', 'right']].set_visible(False)
    fig.tight_layout()
    fig.savefig(out, dpi=140)
    print(f'wrote {out}')


def fig_qualitative(meshes, out, n_cloths=4, elev=18, azim=-60):
    """Rows = cloths, columns = occlusion level."""
    by_cloth = {}
    for m in meshes:
        by_cloth.setdefault(str(m['cloth']), []).append(m)
    cloths = sorted(by_cloth)[:n_cloths]
    if not cloths:
        print('no meshes to plot')
        return
    fracs = sorted({float(m['drop_frac']) for m in meshes})

    fig = plt.figure(figsize=(3.1 * len(fracs), 3.2 * len(cloths)))
    # Shared colour scale so panels are comparable across the whole figure —
    # a per-panel scale would make a catastrophic estimate look like a good one.
    vmax = np.percentile(
        [np.linalg.norm(m['pred'] - m['gt'], axis=-1).mean() for m in meshes],
        90) * 2.0

    for r, ck in enumerate(cloths):
        # One frame per cloth (the same step across columns) so the row varies
        # ONLY in occlusion.
        step = sorted({int(m['step']) for m in by_cloth[ck]})[len(
            {int(m['step']) for m in by_cloth[ck]}) // 2]
        for c, fr in enumerate(fracs):
            cand = [m for m in by_cloth[ck]
                    if float(m['drop_frac']) == fr and int(m['step']) == step]
            ax = fig.add_subplot(len(cloths), len(fracs),
                                 r * len(fracs) + c + 1, projection='3d')
            if not cand:
                ax.axis('off')
                continue
            m = cand[0]
            gt, pred, pcd = m['gt'], m['pred'], m['pcd']
            err = np.linalg.norm(pred - gt, axis=-1)
            # GT as a faint wireframe-ish cloud, prediction painted by error.
            ax.scatter(gt[:, 0], gt[:, 1], gt[:, 2], s=3, c='#cbd5e0',
                       alpha=0.55, linewidths=0)
            ax.scatter(pcd[:, 0], pcd[:, 1], pcd[:, 2], s=1, c='#3182ce',
                       alpha=0.18, linewidths=0)
            p = ax.scatter(pred[:, 0], pred[:, 1], pred[:, 2], s=9, c=err,
                           cmap='magma_r', vmin=0, vmax=vmax, linewidths=0)
            hi = m['hole_idx']
            if len(hi):
                gc, pc = gt[hi].mean(0), pred[hi].mean(0)
                ax.scatter(*gc, s=70, marker='X', c='#e53e3e',
                           edgecolors='k', linewidths=0.5, depthshade=False)
                ax.scatter(*pc, s=70, marker='X', c='#38a169',
                           edgecolors='k', linewidths=0.5, depthshade=False)
            ax.view_init(elev=elev, azim=azim)
            ax.set_xticks([]); ax.set_yticks([]); ax.set_zticks([])
            ax.set_box_aspect((1, 1, 1))
            if r == 0:
                ax.set_title(f'occl {float(m["occlusion"]):.2f}', fontsize=10)
            if c == 0:
                ax.text2D(-0.05, 0.5, ck, transform=ax.transAxes,
                          rotation=90, va='center', fontsize=9)
            ax.text2D(0.02, 0.02, f'{err.mean() * 1000:.0f} mm',
                      transform=ax.transAxes, fontsize=8)

    fig.suptitle('Predicted mesh coloured by per-vertex error  '
                 '(grey = ground truth, blue = observed cloud, '
                 'X = hole centroid: red GT / green predicted)', fontsize=11)
    cax = fig.add_axes([0.92, 0.15, 0.012, 0.7])
    fig.colorbar(p, cax=cax, label='per-vertex error (m)')
    fig.tight_layout(rect=[0, 0, 0.9, 0.96])
    fig.savefig(out, dpi=130)
    print(f'wrote {out}')


def fig_shrink(rows, out):
    fig, ax = plt.subplots(figsize=(5.2, 5))
    occ = np.array([r['occlusion'] for r in rows])
    gt = np.array([r['gt_diag'] for r in rows])
    ratio = np.array([r['extent_ratio'] for r in rows])
    s = ax.scatter(gt, gt * ratio, c=occ, cmap='viridis', s=14, alpha=0.75,
                   linewidths=0)
    lim = [0, max(gt.max(), (gt * ratio).max()) * 1.05]
    ax.plot(lim, lim, 'k:', lw=1, label='perfect size')
    ax.set_xlim(lim); ax.set_ylim(lim)
    ax.set_xlabel('GT bbox diagonal (m)')
    ax.set_ylabel('predicted bbox diagonal (m)')
    ax.set_title(f'Size fidelity (median ratio {np.median(ratio):.3f})')
    ax.legend(frameon=False)
    ax.grid(alpha=0.3)
    ax.spines[['top', 'right']].set_visible(False)
    fig.colorbar(s, ax=ax, label='measured occlusion')
    fig.tight_layout()
    fig.savefig(out, dpi=140)
    print(f'wrote {out}')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--json', required=True)
    ap.add_argument('--meshes', default='')
    ap.add_argument('--out_prefix', required=True)
    ap.add_argument('--n_cloths', type=int, default=4)
    args = ap.parse_args()

    d = json.load(open(args.json))
    fig_curves(d['summary'], d['rows'], f'{args.out_prefix}_curves.png')
    fig_shrink(d['rows'], f'{args.out_prefix}_shrink.png')
    if args.meshes:
        fig_qualitative(load_meshes(args.meshes),
                        f'{args.out_prefix}_qualitative.png',
                        n_cloths=args.n_cloths)


if __name__ == '__main__':
    main()
