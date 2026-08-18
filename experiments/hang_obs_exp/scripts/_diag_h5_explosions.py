#!/usr/bin/env python
"""Scan a collected state-estimation HDF5 for cloth meshes that blow up.

Same three signatures as `_diag_mesh_explosions.py` (edge stretch, bbox growth,
non-finite), applied to the `<split>/cloth_*/trajectory_*/step_*/positions`
layout. This dataset is the more likely place to find instability than the BC
demos: half its episodes are driven by RANDOM actions rather than the scripted
expert, and dedo's explicit spring solver goes unstable when the cloth is
driven hard.

Reports per-trajectory worst values and the dataset distribution, and names the
worst offenders so they can be looked at rather than argued about.

    python _diag_h5_explosions.py --h5 <file.h5> [--max_traj 200]
"""
import argparse

import h5py
import numpy as np


def scan_trajectory(g, rest, edges, rest_len):
    steps = sorted(k for k in g.keys() if k.startswith('step_'))
    if not steps:
        return None
    pos = np.stack([np.asarray(g[s]['positions'], dtype=np.float64)
                    for s in steps])
    finite = np.isfinite(pos).all(axis=(1, 2))
    n_nonfinite = int((~finite).sum())
    v = np.nan_to_num(pos, nan=0.0, posinf=0.0, neginf=0.0)

    lens = np.linalg.norm(v[:, edges[:, 0]] - v[:, edges[:, 1]], axis=-1)
    ratio = lens / rest_len[None, :]
    # Max-over-edges is brittle on a decimated mesh: a quadric-decimated
    # garment carries sliver edges whose rest length is a fraction of the
    # median, and dividing by one of those reports a 30x "explosion" while the
    # edge's ABSOLUTE length is unremarkable. Degenerate edges are dropped by
    # the caller; p99.9 is reported next to the max so one sliver cannot carry
    # the verdict.
    stretch = ratio.max(axis=1)
    stretch_p999 = np.percentile(ratio, 99.9, axis=1)

    extent = np.linalg.norm(v.max(axis=1) - v.min(axis=1), axis=-1)
    rest_extent = np.linalg.norm(rest.max(axis=0) - rest.min(axis=0))
    growth = extent / max(rest_extent, 1e-9)

    return {
        'n_steps': len(steps),
        'max_stretch': float(stretch.max()),
        'max_stretch_p999': float(stretch_p999.max()),
        'max_stretch_step': int(stretch.argmax()),
        'max_growth': float(growth.max()),
        'max_growth_step': int(growth.argmax()),
        'n_nonfinite': n_nonfinite,
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--h5', required=True)
    p.add_argument('--max_traj', type=int, default=0,
                   help='Stop after this many trajectories (0 = all).')
    p.add_argument('--stretch_thresh', type=float, default=3.0)
    p.add_argument('--growth_thresh', type=float, default=3.0)
    p.add_argument('--min_rest_frac', type=float, default=0.25,
                   help='Ignore rest edges shorter than this fraction of the '
                        'mesh median. Decimation slivers otherwise dominate '
                        'the max-stretch statistic.')
    args = p.parse_args()

    rows = []
    with h5py.File(args.h5, 'r') as h:
        for split in h.keys():
            for cloth_name in sorted(h[split].keys()):
                c = h[split][cloth_name]
                if 'rest_positions' not in c:
                    continue
                rest = np.asarray(c['rest_positions'], dtype=np.float64)
                edges = np.asarray(c['edges'], dtype=np.int64)
                rest_len = np.linalg.norm(
                    rest[edges[:, 0]] - rest[edges[:, 1]], axis=-1)
                # Drop degenerate rest edges, not just exactly-zero ones: a
                # decimated garment has slivers an order of magnitude below the
                # median, and a ratio against those measures the decimator, not
                # the physics.
                med = np.median(rest_len[rest_len > 0]) if (rest_len > 0).any() else 0.0
                keep = rest_len > max(args.min_rest_frac * med, 1e-9)
                edges, rest_len = edges[keep], rest_len[keep]

                for tname in sorted(k for k in c.keys()
                                    if k.startswith('trajectory_')):
                    r = scan_trajectory(c[tname], rest, edges, rest_len)
                    if r is None:
                        continue
                    r.update(split=split, cloth=cloth_name, traj=tname)
                    # Carry the per-cloth randomization so an explosion can be
                    # attributed to a physics setting rather than guessed at.
                    for k in ('source', 'rand_deform_elastic_stiffness',
                              'rand_deform_bending_stiffness',
                              'rand_deform_damping_stiffness',
                              'rand_deform_mass', 'num_verts'):
                        if k in c.attrs:
                            v = c.attrs[k]
                            r[k] = v.item() if hasattr(v, 'item') else v
                    rows.append(r)
                    if args.max_traj and len(rows) >= args.max_traj:
                        break
                if args.max_traj and len(rows) >= args.max_traj:
                    break
                if len(rows) % 50 == 0:
                    print(f'  ...{len(rows)} trajectories')
            if args.max_traj and len(rows) >= args.max_traj:
                break

    if not rows:
        raise SystemExit('no trajectories found')

    stretch = np.array([r['max_stretch'] for r in rows])
    growth = np.array([r['max_growth'] for r in rows])
    nonfinite = np.array([r['n_nonfinite'] for r in rows])

    print(f'\nScanned {len(rows)} trajectories')
    p999 = np.array([r['max_stretch_p999'] for r in rows])
    print('\nMax edge stretch (x rest length), max over edges:')
    for q in (50, 90, 99, 100):
        print(f'  p{q:<3} {np.percentile(stretch, q):.2f}')
    print('Same, p99.9 over edges (sliver-robust):')
    for q in (50, 90, 99, 100):
        print(f'  p{q:<3} {np.percentile(p999, q):.2f}')
    print('\nMax bbox growth (x rest diagonal):')
    for q in (50, 90, 99, 100):
        print(f'  p{q:<3} {np.percentile(growth, q):.2f}')
    print(f'\nTrajectories with non-finite positions: {(nonfinite > 0).sum()}')

    bad = [r for r in rows
           if r['max_stretch'] > args.stretch_thresh
           or r['max_growth'] > args.growth_thresh
           or r['n_nonfinite'] > 0]
    print(f'\nOver threshold: {len(bad)}/{len(rows)}')

    # Attribute the blow-ups. dedo's task_info.py warns that elastic stiffness
    # outside 50-150 either collapses the cloth or explodes the explicit spring
    # solver, so that is the first suspect; `source` separates the scripted
    # expert from the random-action half.
    badset = {id(r) for r in bad}
    for key in ('source',):
        vals = sorted({r[key] for r in rows if key in r})
        if len(vals) > 1 or vals:
            print(f'\nExplosion rate by {key}:')
            for v in vals:
                sub = [r for r in rows if r.get(key) == v]
                nb = sum(1 for r in sub if id(r) in badset)
                print(f'  {v:<12} {nb:4d}/{len(sub):<4d} '
                      f'({100.0 * nb / max(len(sub), 1):.0f}%)')

    key = 'rand_deform_elastic_stiffness'
    if any(key in r for r in rows):
        el = np.array([r[key] for r in rows if key in r], dtype=float)
        st = np.array([r['max_stretch'] for r in rows if key in r])
        print(f'\nElastic stiffness: min={el.min():.1f} max={el.max():.1f} '
              f'at-ceiling(150)={int((el >= 149.999).sum())}/{len(el)}')
        edges_ = np.percentile(el, [0, 25, 50, 75, 100])
        print('Explosion rate by elastic-stiffness quartile:')
        for i in range(4):
            lo, hi = edges_[i], edges_[i + 1]
            m = (el >= lo) & (el <= hi if i == 3 else el < hi)
            if m.sum():
                print(f'  [{lo:6.1f}, {hi:6.1f}]  '
                      f'{int((st[m] > args.stretch_thresh).sum()):4d}/{int(m.sum()):<4d} '
                      f'  median stretch {np.median(st[m]):.2f}')
        if len(el) > 2 and el.std() > 0:
            print(f'Pearson r(elastic, max_stretch) = '
                  f'{np.corrcoef(el, st)[0, 1]:.2f}')
    for r in sorted(bad, key=lambda r: -r['max_stretch'])[:25]:
        print(f'  {r["split"]}/{r["cloth"]}/{r["traj"]}  '
              f'stretch={r["max_stretch"]:.1f}@s{r["max_stretch_step"]}  '
              f'growth={r["max_growth"]:.1f}@s{r["max_growth_step"]}  '
              f'nonfinite={r["n_nonfinite"]}')


if __name__ == '__main__':
    main()
