#!/usr/bin/env python
"""Plot the occlusion sweep: success against the MEASURED visibility axis.

Reads the CSV written by `sweep_occlusion.py` and produces

  * one curve per observation mode, success vs measured hole visibility, and
  * the headline scalar per mode — **occlusion tolerance**, the visibility at
    which success falls to half its clear-view value (linear interpolation
    between the bracketing levels; reported as "<lowest measured" when the
    curve never falls that far).

Read the `state` curve first. It is the privileged oracle reading the
simulator, so a visual occluder cannot reach it: if it is not flat, the
occluder is perturbing the task and no other curve on the figure means
anything.

    python plot_occlusion_sweep.py --csv <runs>/occlusion_sweep.csv
"""
import argparse
import csv
import glob
import json
import os
from collections import defaultdict

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

# Fixed hue per mode — colour follows the entity, never its rank, so dropping
# a mode from the figure never repaints the others. Validated for CVD
# separation and contrast against a light surface.
MODE_COLOR = {
    'state':      '#3b6bd6',
    'rgb':        '#d1642f',
    'pcd':        '#12907a',
    'mesh':       '#8a5cc4',
    'gse':        '#b8336a',
    'gse+belief': '#7a7f2e',
}
MODE_LABEL = {
    'state':      'state (privileged oracle)',
    'rgb':        'rgb',
    'pcd':        'pcd (DP3 baseline)',
    'mesh':       'mesh (ours, privileged)',
    # The deployable rows: same policy, same slot, but the hole centroid is
    # ESTIMATED from the point cloud rather than read from the simulator.
    'gse':        'gse (estimated centroid)',
    'gse+belief': 'gse + belief (particle filter)',
}
MODE_ORDER = ['state', 'mesh', 'rgb', 'pcd', 'gse', 'gse+belief']


def load_csv(csv_path, rows):
    with open(csv_path) as f:
        for r in csv.DictReader(f):
            if not r.get('hole_visibility'):
                continue
            rows[r['obs_mode']].append((
                float(r['hole_visibility']),
                float(r['success_hanging']),
                int(r['n_episodes']),
                float(r['occluder_size']),
            ))


def load_runs(runs_root, rows):
    """Read the per-cell `final_eval_metrics.json` files directly.

    Preferred over the CSV: each sweep invocation writes its own CSV to the
    same directory, so concurrent per-mode sweeps clobber each other's copy.
    The per-cell JSONs are one file per (mode, size) and never collide.
    """
    pattern = os.path.join(runs_root, '*', 'occ_*', 'final_eval_metrics.json')
    by_ckpt = defaultdict(list)
    for path in sorted(glob.glob(pattern)):
        with open(path) as f:
            m = json.load(f)
        if 'final_eval/hole_visibility' not in m:
            print(f'[warn] no visibility measured in {path} — skipped')
            continue
        # `row` names the deployable variants apart from the obs mode they run
        # under (gse / gse+belief both use obs_mode=state). Older metrics files
        # predate it, so fall back.
        row = m.get('row') or m['obs_mode']
        by_ckpt[(row, m.get('resume', ''))].append((
            float(m['final_eval/hole_visibility']),
            float(m['final_eval/success_hanging']),
            int(m['final_eval/n_episodes']),
            float(m['final_eval/occluder_size']),
        ))

    # A mode can have cells from more than one checkpoint (e.g. a 200-epoch
    # curve and the 400-epoch curve that supersedes it). Never mix them into
    # one line: keep the newest checkpoint's cells and say what was dropped.
    for mode in {k[0] for k in by_ckpt}:
        ckpts = sorted(k[1] for k in by_ckpt if k[0] == mode)
        keep = ckpts[-1]
        for c in ckpts[:-1]:
            print(f'[warn] {mode}: ignoring {len(by_ckpt[(mode, c)])} cells '
                  f'from superseded ckpt {c}')
        rows[mode].extend(by_ckpt[(mode, keep)])


def finalize(rows):
    # Ascending visibility, so a curve reads left (occluded) to right (clear).
    return {m: sorted(v) for m, v in rows.items() if v}


def tolerance(points):
    """Visibility at which success falls to half its clear-view value.

    `points` is ascending in visibility, so the clear-view value is the last
    one. Walk down from it and linearly interpolate the first crossing.
    """
    if len(points) < 2:
        return None
    clear = points[-1][1]
    half = clear / 2.0
    for (v_lo, s_lo, *_), (v_hi, s_hi, *_) in zip(points, points[1:]):
        if s_lo <= half <= s_hi and s_hi != s_lo:
            return v_lo + (half - s_lo) * (v_hi - v_lo) / (s_hi - s_lo)
    return None


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--runs_root', default=None,
                   help='Sweep logdir_root; reads every '
                        '<mode>/occ_*/final_eval_metrics.json under it. '
                        'Preferred — the per-mode CSVs clobber each other.')
    p.add_argument('--csv', action='append', default=[],
                   help='CSV from sweep_occlusion.py; repeatable.')
    p.add_argument('--out', default=None,
                   help='Output png (default: alongside the inputs).')
    args = p.parse_args()
    if not args.runs_root and not args.csv:
        p.error('pass --runs_root or at least one --csv')

    rows = defaultdict(list)
    if args.runs_root:
        load_runs(args.runs_root, rows)
    for c in args.csv:
        load_csv(c, rows)
    data = finalize(rows)
    if not data:
        raise SystemExit('no scored cells found')

    fig, ax = plt.subplots(figsize=(7.0, 4.6), dpi=160)
    modes = [m for m in MODE_ORDER if m in data] + \
            [m for m in data if m not in MODE_ORDER]

    # Direct labels sit at the clear-view end, where curves often converge
    # (that is the whole point of the experiment), so nudge them apart rather
    # than letting two labels print on top of each other.
    label_y = {}
    _gap = 0.045
    for mode in sorted(modes, key=lambda m: -data[m][-1][1]):
        y = data[mode][-1][1]
        for taken in label_y.values():
            if abs(y - taken) < _gap:
                y = taken - _gap
        label_y[mode] = y

    for mode in modes:
        pts = data[mode]
        vis = np.array([q[0] for q in pts])
        suc = np.array([q[1] for q in pts])
        n = np.array([q[2] for q in pts])
        # Binomial SE on the success rate — 50 episodes is SE ~0.07 at p=0.5,
        # so the band is not decoration, it is the resolution of the row.
        se = np.sqrt(np.clip(suc * (1 - suc), 0, None) / np.maximum(n, 1))
        color = MODE_COLOR.get(mode, '#6b7280')
        ax.fill_between(vis, suc - se, suc + se, color=color, alpha=0.13,
                        linewidth=0)
        ax.plot(vis, suc, color=color, linewidth=2.0, marker='o',
                markersize=6, markeredgecolor='white', markeredgewidth=1.5,
                label=MODE_LABEL.get(mode, mode), zorder=3)
        # Direct label at the clear-view end; the legend carries the rest.
        ax.annotate(MODE_LABEL.get(mode, mode).split(' ')[0],
                    xy=(vis[-1], label_y[mode]), xytext=(6, 0),
                    textcoords='offset points', va='center',
                    fontsize=9, color='#374151')

    ax.set_xlabel('measured hole visibility  (fraction of hole loop seen)')
    ax.set_ylabel('success rate  (hanging)')
    ax.set_title('Cross-modality BC under occlusion', fontsize=12, pad=10)
    ax.set_ylim(-0.03, 1.03)
    ax.grid(True, color='#e5e7eb', linewidth=0.8)
    ax.set_axisbelow(True)
    for side in ('top', 'right'):
        ax.spines[side].set_visible(False)
    for side in ('left', 'bottom'):
        ax.spines[side].set_color('#9ca3af')
    ax.legend(frameon=False, fontsize=9, loc='lower right')
    fig.tight_layout()

    out = args.out or os.path.join(
        args.runs_root or os.path.dirname(args.csv[0]), 'occlusion_sweep.png')
    fig.savefig(out, bbox_inches='tight')
    print(f'[plot] wrote {out}')

    print('\nOcclusion tolerance (visibility at half the clear-view rate):')
    for mode in [m for m in MODE_ORDER if m in data]:
        pts = data[mode]
        t = tolerance(pts)
        clear = pts[-1][1]
        lo = pts[0][0]
        t_str = f'{t:.3f}' if t is not None else f'<{lo:.3f} (never halves)'
        print(f'  {mode:<6} clear-view={clear:.2f}  tolerance={t_str}')


if __name__ == '__main__':
    main()
