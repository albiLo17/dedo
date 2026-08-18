"""Gather every v5 result into one figure + one machine-readable summary.

Deliberately tolerant of missing pieces: an overnight run where one mode
crashed should still produce a report that says so, rather than failing and
leaving nothing to read in the morning.

    python make_v5_report.py --policy_root logs/hang_obs_exp/v5_policies \
        --est_dir logs/hang_obs_exp/v5_est --out_dir logs/hang_obs_exp/v5_report
"""
import argparse
import glob
import json
import os
import re

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt                                  # noqa: E402

MODES = ['state', 'pcd', 'rgb', 'mesh']
SCALE_MM = 45.0          # 1 sim unit = 45 mm


def find_run_dir(policy_root, mode):
    """train_diffusion_bc.py nests an auto-named run dir under the logdir
    (<logdir>/<mode>/diff<ts>_<mode>_.../), so the metrics sit two levels
    deeper than the path handed to --logdir. Take the newest."""
    cands = sorted(glob.glob(os.path.join(policy_root, mode, '*', 'final_eval_metrics.json')))
    cands += sorted(glob.glob(os.path.join(policy_root, mode, '*', '*', 'final_eval_metrics.json')))
    return os.path.dirname(cands[-1]) if cands else None


def load_json(p):
    if p is None:
        return None
    try:
        with open(p) as f:
            return json.load(f)
    except Exception:
        return None


def row_of(d):
    """The metrics live at the TOP level as 'final_eval/*' keys. The 'row'
    key is a preformatted string label, not a nested dict — treating it as one
    silently iterates its characters and every lookup returns None."""
    if d is None:
        return {}
    r = d.get('row')
    return r if isinstance(r, dict) else d


def pick(r, *names, default=None):
    for n in names:
        for k in r:
            if k.endswith(n):
                return r[k]
    return default


def parse_loss_curve(log_path):
    """Epoch-mean training loss, scraped from the trainer's stdout."""
    if not os.path.exists(log_path):
        return [], []
    ep, loss = [], []
    pat = re.compile(r'epoch\s+(\d+).*?loss[=: ]+([0-9.eE+-]+)', re.I)
    with open(log_path, errors='ignore') as f:
        for line in f:
            m = pat.search(line)
            if m:
                try:
                    ep.append(int(m.group(1)))
                    loss.append(float(m.group(2)))
                except ValueError:
                    pass
    return ep, loss


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--policy_root', required=True)
    ap.add_argument('--est_dir', default='')
    ap.add_argument('--out_dir', required=True)
    a = ap.parse_args()
    os.makedirs(a.out_dir, exist_ok=True)

    summary = {'policies': {}, 'estimator': {}, 'in_the_loop': {}, 'videos': {}}

    # ---- policy rows -------------------------------------------------------
    for m in MODES:
        run_dir = find_run_dir(a.policy_root, m)
        d = load_json(os.path.join(run_dir, 'final_eval_metrics.json')
                      if run_dir else None)
        r = row_of(d)
        summary['policies'][m] = {
            'success_rate': pick(r, 'success_rate'),
            'success_hanging': pick(r, 'success_rate_hanging', 'is_threaded_hanging'),
            'success_topological': pick(r, 'success_rate_topological'),
            'reward': pick(r, 'episode_reward', 'reward'),
            'centroid_err': pick(r, 'centroid_err', 'hole_err'),
            'n_episodes': pick(r, 'n_episodes', 'n_eval_episodes'),
            'trained': d is not None,
        }
        summary['policies'][m]['run_dir'] = run_dir
        summary['videos'][m] = (sorted(glob.glob(os.path.join(run_dir, '*.mp4')))
                                if run_dir else [])

    # ---- estimator accuracy ------------------------------------------------
    est_rows = []
    if a.est_dir:
        q = load_json(os.path.join(a.est_dir, 'state_est_quality.json'))
        if isinstance(q, list):
            est_rows = q
        elif isinstance(q, dict):
            # 'summary' is the per-drop-level aggregate (4 rows); 'rows' is
            # every individual sample (640). Plotting 'rows' would draw a
            # scatter where a 4-point curve belongs.
            est_rows = q.get('summary') or q.get('rows', [])
        summary['estimator']['rows'] = est_rows

        gse_dir = find_run_dir(a.est_dir, 'state_gse') or os.path.join(
            a.est_dir, 'state_gse')
        g = load_json(os.path.join(gse_dir, 'final_eval_metrics.json'))
        rg = row_of(g)
        summary['in_the_loop'] = {
            'success_rate': pick(rg, 'success_rate'),
            'success_hanging': pick(rg, 'success_hanging'),
            'reward': pick(rg, 'mean_reward', 'episode_reward', 'reward'),
            # centroid_err_* only exist on this row: it is the live error of
            # the ESTIMATED centroid against ground truth, in sim units.
            'centroid_err_mean': pick(rg, 'centroid_err_mean'),
            'centroid_err_p50': pick(rg, 'centroid_err_p50'),
            'centroid_err_p90': pick(rg, 'centroid_err_p90'),
            'ran': g is not None,
        }
        summary['videos']['state_gse'] = sorted(
            glob.glob(os.path.join(gse_dir, '*.mp4')))

    # ---- figure ------------------------------------------------------------
    fig, ax = plt.subplots(1, 3, figsize=(19, 5.4))

    # panel 1: BOTH success metrics side by side. Showing only `legacy` hides
    # the finding that the metrics disagree enough to reorder the modes — rgb
    # and mesh score far higher under `hanging` than under `legacy`.
    labels, leg, han, miss = [], [], [], []
    for m in MODES:
        p = summary['policies'][m]
        labels.append(m)
        leg.append(float(p['success_rate']) if p['success_rate'] is not None else 0.0)
        han.append(float(p['success_hanging']) if p['success_hanging'] is not None else 0.0)
        miss.append(p['success_rate'] is None)
    itl = summary['in_the_loop'].get('success_rate')
    if itl is not None:
        labels.append('state\n+estimator')
        leg.append(float(itl))
        han.append(float(summary['in_the_loop'].get('success_hanging') or 0.0))
        miss.append(False)
    x = np.arange(len(labels))
    w = 0.38
    ax[0].bar(x - w / 2, leg, w, label='legacy (centroid distance)', color='#4c78a8')
    ax[0].bar(x + w / 2, han, w, label='hanging (3D on-peg test)', color='#59a14f')
    for i in range(len(labels)):
        if miss[i]:
            ax[0].text(x[i], 0.02, 'n/a', ha='center', fontsize=10, color='#b45309')
        else:
            ax[0].text(x[i] - w / 2, leg[i] + 0.012, f'{leg[i]:.2f}',
                       ha='center', fontsize=9)
            ax[0].text(x[i] + w / 2, han[i] + 0.012, f'{han[i]:.2f}',
                       ha='center', fontsize=9)
    ax[0].set_xticks(x)
    ax[0].set_xticklabels(labels)
    ax[0].set(ylabel='success rate', ylim=(0, 1.0),
              title='BC policy success — the two metrics disagree')
    ax[0].legend(fontsize=8)
    ax[0].grid(alpha=0.25, axis='y')

    # panel 2: estimator accuracy vs occlusion
    if est_rows:
        occ = [r.get('occlusion_mean', r.get('drop_frac', 0)) for r in est_rows]
        he = [r.get('hole_err_mean', np.nan) * 1000 for r in est_rows]
        hm = [r.get('hole_err_median', np.nan) * 1000 for r in est_rows]
        ex = [r.get('extent_ratio_mean', np.nan) for r in est_rows]
        o = np.argsort(occ)
        occ = np.asarray(occ)[o]
        ax[1].plot(occ, np.asarray(he)[o], 'o-', label='hole-centroid err (mean)')
        ax[1].plot(occ, np.asarray(hm)[o], 's--', label='hole-centroid err (median)')
        ax[1].axhline(24.0, color='#d62728', ls=':',
                      label='real hole radius 24 mm')
        ax[1].set(xlabel='mean occlusion fraction', ylabel='error (mm)',
                  title='State-estimator accuracy vs occlusion')
        ax[1].legend(fontsize=8)
        ax2 = ax[1].twinx()
        ax2.plot(occ, np.asarray(ex)[o], '^-', color='#59a14f', alpha=0.7)
        ax2.set_ylabel('extent ratio (pred/gt)', color='#59a14f')
        ax2.axhline(1.0, color='#59a14f', ls=':', alpha=0.5)
    else:
        ax[1].text(0.5, 0.5, 'no estimator eval', ha='center')
    ax[1].grid(alpha=0.25)

    # panel 3: training loss curves
    any_curve = False
    for m in MODES:
        ep, loss = parse_loss_curve(os.path.join(a.policy_root, f'{m}.log'))
        if len(ep) > 3:
            ax[2].plot(ep, loss, label=m)
            any_curve = True
    if any_curve:
        ax[2].set(xlabel='epoch', ylabel='train loss', yscale='log',
                  title='BC training curves')
        ax[2].legend(fontsize=8)
    else:
        ax[2].text(0.5, 0.5, 'no parseable loss curves', ha='center')
    ax[2].grid(alpha=0.25)

    fig.suptitle('v5 — goal-chained demos: policies, state estimator, '
                 'estimator in the loop')
    fig.tight_layout()
    fig_path = os.path.join(a.out_dir, 'v5_results.png')
    fig.savefig(fig_path, dpi=115)

    with open(os.path.join(a.out_dir, 'summary.json'), 'w') as f:
        json.dump(summary, f, indent=2, default=str)

    # ---- console table -----------------------------------------------------
    print('\n=== POLICY SUCCESS (legacy metric) ===')
    print(f'{"obs mode":<16s} {"success":>9s} {"reward":>10s} {"n":>5s}  status')
    for m in MODES:
        p = summary['policies'][m]
        sr = p['success_rate']
        print(f'{m:<16s} {("%.3f" % sr) if sr is not None else "-":>9s} '
              f'{("%.1f" % p["reward"]) if p["reward"] is not None else "-":>10s} '
              f'{str(p["n_episodes"] or "-"):>5s}  '
              f'{"ok" if p["trained"] else "MISSING"}')
    if summary['in_the_loop'].get('ran'):
        s = summary['in_the_loop']['success_rate']
        base = summary['policies']['state']['success_rate']
        print(f'{"state+estimator":<16s} {("%.3f" % s) if s is not None else "-":>9s}')
        if s is not None and base:
            print(f'  -> retains {100*float(s)/float(base):.0f}% of the '
                  f'GT-centroid policy')
    if est_rows:
        print('\n=== STATE ESTIMATOR (own val split, through the serving path) ===')
        print(f'{"drop":>6s} {"occl":>6s} {"hole mm":>9s} {"median":>8s} '
              f'{"extent":>7s}')
        for r in est_rows:
            print(f'{r.get("drop_frac", 0):6.2f} {r.get("occlusion_mean", 0):6.2f} '
                  f'{r.get("hole_err_mean", 0)*1000:9.1f} '
                  f'{r.get("hole_err_median", 0)*1000:8.1f} '
                  f'{r.get("extent_ratio_mean", 0):7.3f}')
    print(f'\nwrote {fig_path}')
    print(f'wrote {os.path.join(a.out_dir, "summary.json")}')


if __name__ == '__main__':
    main()
