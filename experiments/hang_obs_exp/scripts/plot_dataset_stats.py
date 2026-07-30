"""Turn a collect_state_est_data.py `*_stats.json` into review figures.

Two PNGs, because they answer two different questions:

  <out>_randomization.png  did the randomization actually span what we asked?
                           (realized camera / physics / goal-pose distributions)
  <out>_data.png           what does the resulting dataset look like?
                           (point-cloud coverage, episode length, mesh size, split)

Usage:
  python experiments/hang_obs_exp/scripts/plot_dataset_stats.py \
      --stats experiments/hang_obs_exp/data/state_est/dedo_hang_stats.json

Palette: categorical slots 1-3 of the validated default (blue/orange/aqua),
which is the largest subset that clears the all-pairs CVD and normal-vision
floors in both light and dark. Aqua sits under 3:1 on a light surface, so every
aqua mark carries a visible label.
"""
import argparse
import json
import os

import matplotlib
matplotlib.use('Agg')
# Emit <text> elements instead of glyph outlines: ~5x smaller SVG, which is what
# lets a figure be attached to Notion inline (its 200 KiB text cap).
matplotlib.rcParams['svg.fonttype'] = 'none'
import matplotlib.pyplot as plt
import numpy as np

C_BLUE, C_ORANGE, C_AQUA = '#2a78d6', '#eb6834', '#1baf7a'
INK, INK_2, GRID = '#0b0b0b', '#52514e', '#d8d7d2'


def _style(ax, title, xlabel):
    """Recessive frame: no top/right spine, hairline grid behind the marks."""
    ax.set_title(title, fontsize=9, color=INK, loc='left', pad=6)
    ax.set_xlabel(xlabel, fontsize=8, color=INK_2)
    ax.tick_params(labelsize=7, colors=INK_2, length=3)
    for side in ('top', 'right'):
        ax.spines[side].set_visible(False)
    for side in ('left', 'bottom'):
        ax.spines[side].set_color(GRID)
    ax.grid(True, axis='y', color=GRID, linewidth=0.6, alpha=0.9)
    ax.set_axisbelow(True)


def _hist(ax, vals, title, xlabel, color=C_BLUE, band=None, logx=False):
    """Histogram + median line. `band` draws the REQUESTED range so the figure
    shows requested-vs-realized, which is the whole point of the panel."""
    vals = np.asarray([v for v in vals if v is not None and np.isfinite(v)], float)
    if vals.size == 0:
        _style(ax, title + ' — no data', xlabel)
        return
    # Only go log when the values actually span more than a decade. Below that,
    # matplotlib labels log MINOR ticks and they collide into an unreadable smear
    # (e.g. "7x10^-2 8x10^-2 9x10^-2" on top of each other).
    span_decades = (np.log10(vals.max() / vals.min())
                    if logx and (vals > 0).all() and vals.min() > 0 else 0.0)
    if span_decades > 1.0:
        bins = np.logspace(np.log10(vals.min()), np.log10(vals.max()), 18)
        ax.set_xscale('log')
    else:
        bins = 18
    ax.hist(vals, bins=bins, color=color, edgecolor='white', linewidth=0.6)
    med = float(np.median(vals))
    ax.axvline(med, color=INK, linewidth=1.2, linestyle='--')
    ax.annotate(f'median {med:.3g}', xy=(med, ax.get_ylim()[1] * 0.94),
                xytext=(4, 0), textcoords='offset points',
                fontsize=7, color=INK, va='top')
    if band is not None and all(b is not None for b in band):
        for b in band:
            ax.axvline(float(b), color=INK_2, linewidth=0.8, alpha=0.6)
        ax.annotate('requested', xy=(float(band[0]), ax.get_ylim()[1] * 0.62),
                    xytext=(3, 0), textcoords='offset points',
                    fontsize=6.5, color=INK_2, va='top')
    _style(ax, title, xlabel)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--stats', required=True)
    ap.add_argument('--out_prefix', default='',
                    help='Default: the stats path without _stats.json')
    ap.add_argument('--svg', action='store_true',
                    help='Also write SVG next to each PNG. SVG is UTF-8 text, so '
                         'it can be attached to Notion inline (200 KiB cap) '
                         'without a public URL or the file-upload API.')
    args = ap.parse_args()

    with open(args.stats) as f:
        blob = json.load(f)
    cfg, eps = blob['config'], blob['episodes']
    if not eps:
        raise SystemExit('no episode rows in stats file')
    pre = args.out_prefix or args.stats.replace('_stats.json', '')
    rnd = cfg.get('randomize', {})

    def col(k):
        return [e.get(k) for e in eps]

    # ---------------- figure 1: randomization ----------------
    fig, axes = plt.subplots(3, 3, figsize=(12, 8.6))
    fig.patch.set_facecolor('#fcfcfb')
    _hist(axes[0][0], col('cam_yaw'), 'Camera yaw', 'deg',
          band=rnd.get('yaw', [None, None])[1] if rnd.get('yaw') else None)
    _hist(axes[0][1], col('cam_pitch'), 'Camera pitch', 'deg',
          band=rnd.get('pitch', [None, None])[1] if rnd.get('pitch') else None)
    _hist(axes[0][2], col('cam_dist'), 'Camera distance', 'scene units',
          band=rnd.get('dist', [None, None])[1] if rnd.get('dist') else None)

    base = (rnd.get('physics') or [None, {}])[1].get('scene_baseline', {}) or {}
    for ax, key, label in (
            (axes[1][0], 'deform_bending_stiffness', 'Bending stiffness'),
            (axes[1][1], 'deform_elastic_stiffness', 'Elastic stiffness'),
            (axes[1][2], 'deform_mass', 'Cloth mass')):
        _hist(ax, col(key), label, 'value', color=C_ORANGE, logx=True)
        if key in base:
            ax.axvline(float(base[key]), color=C_AQUA, linewidth=1.6)
            ax.annotate(f'scene baseline {base[key]:g}',
                        xy=(float(base[key]), ax.get_ylim()[1] * 0.78),
                        xytext=(4, 0), textcoords='offset points',
                        fontsize=6.5, color=INK, va='top')

    # Realized peg pose. Scatter, not a histogram: the point is the 2-D box.
    gp = np.array([e['goal_pos'] for e in eps if e.get('goal_pos')], float)
    ax = axes[2][0]
    if gp.size:
        sc = ax.scatter(gp[:, 0], gp[:, 1], c=gp[:, 2], cmap='Blues',
                        s=46, edgecolor='white', linewidth=0.7)
        cb = fig.colorbar(sc, ax=ax, fraction=0.046)
        cb.set_label('peg z', fontsize=7, color=INK_2)
        cb.ax.tick_params(labelsize=6, colors=INK_2)
        r = cfg.get('randomize', {}).get('goal_radius', 0) or 0
        if r:
            c = gp.mean(axis=0)
            ax.add_patch(plt.Rectangle((c[0] - r, c[1] - r), 2 * r, 2 * r,
                                       fill=False, edgecolor=INK_2,
                                       linewidth=0.8, linestyle='--'))
    _style(ax, 'Realized peg pose (dashed = requested xy box)', 'x')
    ax.set_ylabel('y', fontsize=8, color=INK_2)
    ax.grid(True, color=GRID, linewidth=0.6)

    _hist(axes[2][1], col('deform_damping_stiffness'), 'Damping stiffness',
          'value', color=C_ORANGE, logx=True)
    _hist(axes[2][2], col('deform_friction_coeff'), 'Friction coefficient',
          'value', color=C_ORANGE)

    fig.suptitle(f"Randomization actually realized — {os.path.basename(pre)}  "
                 f"({len(eps)} episodes, env {cfg.get('env')})",
                 fontsize=11, color=INK, x=0.01, ha='left')
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    p1 = pre + '_randomization.png'
    fig.savefig(p1, dpi=140, facecolor=fig.get_facecolor())
    if args.svg:
        fig.savefig(pre + '_randomization.svg', facecolor=fig.get_facecolor())
    plt.close(fig)

    # ---------------- figure 2: the data ----------------
    fig, axes = plt.subplots(2, 3, figsize=(12, 6))
    fig.patch.set_facecolor('#fcfcfb')
    npts = cfg.get('pcd_n_points', 2048)

    # Coverage is the headline data statistic: below pcd_n_points the stored
    # cloud is padded with duplicates, so this panel says how real the clouds are.
    ax = axes[0][0]
    _hist(ax, col('valid_px_mean'), 'Unique cloth pixels per frame (episode mean)',
          'pixels')
    ax.axvline(npts, color=C_ORANGE, linewidth=1.6)
    ax.annotate(f'pcd_n_points {npts}\n(below = duplicated points)',
                xy=(npts, ax.get_ylim()[1] * 0.72), xytext=(4, 0),
                textcoords='offset points', fontsize=6.5, color=INK, va='top')

    _hist(axes[0][1], col('num_steps'), 'Frames per episode', 'frames')
    _hist(axes[0][2], col('num_verts'), 'Mesh vertices per cloth', 'vertices')
    _hist(axes[1][0], col('hole_verts'), 'Hole-loop vertices', 'vertices')
    _hist(axes[1][1], col('gripper_path_len'),
          'Gripper path length (L1 over episode)', 'scene units')

    # Composition: split x source. Grouped bars, direct-labelled.
    ax = axes[1][2]
    srcs = sorted({e['source'] for e in eps})
    splits = ['train', 'val']
    w = 0.36
    for i, sp in enumerate(splits):
        vals = [sum(1 for e in eps if e['source'] == s and e.get('split') == sp)
                for s in srcs]
        xs = np.arange(len(srcs)) + (i - 0.5) * (w + 0.03)
        bars = ax.bar(xs, vals, width=w, color=(C_BLUE, C_AQUA)[i],
                      edgecolor='white', linewidth=1.0, label=sp)
        for b, v in zip(bars, vals):
            ax.annotate(str(v), xy=(b.get_x() + b.get_width() / 2, v),
                        xytext=(0, 2), textcoords='offset points',
                        ha='center', fontsize=7, color=INK)
    ax.set_xticks(np.arange(len(srcs)))
    ax.set_xticklabels(srcs, fontsize=7, color=INK_2)
    ax.legend(fontsize=7, frameon=False, labelcolor=INK_2)
    _style(ax, 'Episodes by source and split', '')

    tot_frames = int(np.sum([e['num_steps'] for e in eps]))
    fig.suptitle(f"Dataset shape — {os.path.basename(pre)}  "
                 f"({len(eps)} cloths, {tot_frames} frames, "
                 f"{cfg.get('n_train')}/{cfg.get('n_val')} train/val)",
                 fontsize=11, color=INK, x=0.01, ha='left')
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    p2 = pre + '_data.png'
    fig.savefig(p2, dpi=140, facecolor=fig.get_facecolor())
    if args.svg:
        fig.savefig(pre + '_data.svg', facecolor=fig.get_facecolor())
    plt.close(fig)

    print(f'wrote {p1}\nwrote {p2}')
    # Text summary, so the numbers can be pasted into notes without the figures.
    cov = np.asarray([e['valid_px_mean'] for e in eps], float)
    print(f'\nsummary: {len(eps)} episodes, {tot_frames} frames | '
          f'coverage median {np.median(cov):.0f}px '
          f'({100 * np.mean(cov >= npts):.0f}% of episodes >= pcd_n_points) | '
          f'frames/ep median {np.median(col("num_steps")):.0f} | '
          f'verts median {np.median(col("num_verts")):.0f}')


if __name__ == '__main__':
    main()
