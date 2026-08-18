"""Figures for the STATIC-obstacle belief experiment.

Two panels, because the claim needs both:

  left  — the visibility PROFILE within an episode. A static obstacle is only
          worth running if visibility actually varies as the cloth moves past
          it; this panel is the evidence that it does, against the clear-view
          baseline (which already swings, from the cloth's own motion, so the
          occluder has to beat that and not merely be non-flat).

  right — centroid error BINNED by the visibility present when it was produced.
          This is the comparison the pooled episode mean cannot make: a filter
          that holds its estimate through the dark stretches and a per-frame
          model that only works in the clear ones can post identical means and
          completely different curves here. A flat line is the signature of
          memory; a rising line is the signature of per-frame perception.
"""

from __future__ import annotations

import argparse
import json
import os

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt          # noqa: E402
import numpy as np                        # noqa: E402

SIM_TO_MM = 45.0     # x0.045 (sim->real m) x1000 (m->mm)
BINS = [(0, 0.05), (0.05, 0.2), (0.2, 0.5), (0.5, 1.01)]
BIN_X = [0.025, 0.125, 0.35, 0.75]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--runs_root', required=True)
    ap.add_argument('--clear_run', default='',
                    help='a no-occluder run dir, for the baseline profile')
    ap.add_argument('--out', required=True)
    args = ap.parse_args()

    cells = [('gse_static_0.30', 'per-frame estimator, occl 0.30', '#2b6cb0', '-'),
             ('pf_static_0.30', 'belief (PF), occl 0.30', '#c05621', '-'),
             ('gse_static_0.60', 'per-frame estimator, occl 0.60', '#63b3ed', '--'),
             ('pf_static_0.60', 'belief (PF), occl 0.60', '#f6ad55', '--')]

    fig, ax = plt.subplots(1, 2, figsize=(13, 4.6))

    # --- left: visibility profile over the episode ---------------------------
    def profile(run):
        p = os.path.join(args.runs_root, run, 'final_eval_hole_visibility_by_ep.npy')
        return np.load(p) if os.path.exists(p) else None

    if args.clear_run:
        pr = profile(args.clear_run)
        if pr is not None:
            t = np.linspace(0, 1, pr.shape[1])
            ax[0].plot(t, np.nanmean(pr, axis=0), color='#718096', lw=2,
                       ls=':', label='no occluder (baseline)')
    for run, lab, col, ls in cells:
        if run.startswith('pf_'):
            continue                     # same geometry as its gse twin
        pr = profile(run)
        if pr is None:
            continue
        t = np.linspace(0, 1, pr.shape[1])
        m = np.nanmean(pr, axis=0)
        lo = np.nanpercentile(pr, 25, axis=0)
        hi = np.nanpercentile(pr, 75, axis=0)
        ax[0].plot(t, m, color=col, lw=2, ls=ls,
                   label=lab.replace('per-frame estimator, ', 'static box '))
        ax[0].fill_between(t, lo, hi, color=col, alpha=0.15, linewidth=0)
    ax[0].set_xlabel('fraction of episode elapsed')
    ax[0].set_ylabel('hole visibility')
    ax[0].set_title('A fixed obstacle makes visibility vary WITHIN the episode')
    ax[0].legend(frameon=False, fontsize=9)

    # --- right: error binned by instantaneous visibility ----------------------
    for run, lab, col, ls in cells:
        p = os.path.join(args.runs_root, run, 'final_eval_metrics.json')
        if not os.path.exists(p):
            continue
        d = json.load(open(p))
        xs, ys = [], []
        for (lo, hi), x in zip(BINS, BIN_X):
            v = d.get(f'final_eval/centroid_err_vis{lo:g}_{hi:g}')
            n = d.get(f'final_eval/centroid_err_vis{lo:g}_{hi:g}_n', 0)
            if v is not None and n >= 3:
                xs.append(x)
                ys.append(v * SIM_TO_MM)
        if xs:
            ax[1].plot(xs, ys, 'o-', color=col, ls=ls, lw=2, label=lab)
    ax[1].set_xlabel('hole visibility at that frame')
    ax[1].set_ylabel('median hole-centroid error (mm)')
    ax[1].set_title('Flat = memory, rising = per-frame perception')
    ax[1].legend(frameon=False, fontsize=9)
    ax[1].set_ylim(bottom=0)

    for a in ax:
        a.grid(alpha=0.3)
        a.spines[['top', 'right']].set_visible(False)
    fig.tight_layout()
    fig.savefig(args.out, dpi=140)
    print(f'wrote {args.out}')


if __name__ == '__main__':
    main()
