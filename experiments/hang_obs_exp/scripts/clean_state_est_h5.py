#!/usr/bin/env python
"""Write a copy of a state-est HDF5 with solver blow-ups removed.

Some trajectories in `dedo_hang_real.h5` go unstable partway through (see
`_diag_h5_explosions.py`): the cloth is fine for tens of steps, then dedo's
explicit spring solver diverges and the mesh inflates into a ball it never
recovers from. Those frames are physically invalid GT.

**Truncate, do not skip.** Every episode keeps its valid PREFIX, cut at the
first frame whose max edge stretch exceeds the threshold. Dropping interior
frames instead would silently corrupt `velocities`, which are backward
differences of consecutive stored frames — the same trap as the coverage-floor
fix in the collector. A trajectory whose surviving prefix is shorter than
`--min_steps` is dropped entirely rather than kept as a stub.

**Threshold.** Default 3.0x rest edge length. That is not arbitrary: the 854
expert-driven BC demos, which are clean, peak at 2.94x — so 3.0 is above
anything a valid episode in this task has been observed to reach. Use
`--dry_run` to see the survival table at several thresholds before committing.

    python clean_state_est_h5.py --in <src.h5> --out <clean.h5> [--dry_run]
"""
import argparse
import os

import h5py
import numpy as np


def stretch_series(t, edges, rest_len, steps):
    pos = np.stack([np.asarray(t[s]['positions'], dtype=np.float64)
                    for s in steps])
    bad_finite = ~np.isfinite(pos).all(axis=(1, 2))
    v = np.nan_to_num(pos, nan=0.0, posinf=0.0, neginf=0.0)
    lens = np.linalg.norm(v[:, edges[:, 0]] - v[:, edges[:, 1]], axis=-1)
    s = (lens / rest_len[None, :]).max(axis=1)
    # A non-finite frame is a blow-up whatever its finite-part stretch says.
    s[bad_finite] = np.inf
    return s


def first_bad(s, thresh):
    """Index of the first frame over threshold, or len(s) if none."""
    over = np.nonzero(s > thresh)[0]
    return int(over[0]) if len(over) else len(s)


def iter_trajectories(h):
    for split in h.keys():
        for cloth_name in sorted(h[split].keys()):
            c = h[split][cloth_name]
            if 'rest_positions' not in c:
                continue
            rest = np.asarray(c['rest_positions'], dtype=np.float64)
            edges = np.asarray(c['edges'], dtype=np.int64)
            rl = np.linalg.norm(rest[edges[:, 0]] - rest[edges[:, 1]], axis=-1)
            keep = rl > 1e-9
            for tname in sorted(k for k in c.keys()
                                if k.startswith('trajectory_')):
                yield split, cloth_name, c, tname, edges[keep], rl[keep]


def copy_attrs(src, dst):
    for k, v in src.attrs.items():
        dst.attrs[k] = v


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--in', dest='src', required=True)
    p.add_argument('--out', dest='dst', default=None)
    p.add_argument('--thresh', type=float, default=3.0)
    p.add_argument('--min_steps', type=int, default=10)
    p.add_argument('--dry_run', action='store_true')
    p.add_argument('--limit', type=int, default=0)
    args = p.parse_args()
    if not args.dry_run and not args.dst:
        raise SystemExit('--out is required unless --dry_run')

    rows = []
    with h5py.File(args.src, 'r') as h:
        for split, cname, c, tname, edges, rl in iter_trajectories(h):
            t = c[tname]
            steps = sorted(k for k in t.keys() if k.startswith('step_'))
            if not steps:
                continue
            s = stretch_series(t, edges, rl, steps)
            rows.append({
                'split': split, 'cloth': cname, 'traj': tname,
                'n_steps': len(steps),
                'first_bad': first_bad(s, args.thresh),
                'peak': float(np.nanmax(s[np.isfinite(s)])) if np.isfinite(s).any() else np.inf,
                'source': str(c.attrs.get('source', '?')),
            })
            if args.limit and len(rows) >= args.limit:
                break
            if len(rows) % 100 == 0:
                print(f'  ...scanned {len(rows)}')

    if not rows:
        raise SystemExit('no trajectories found')

    total_steps = sum(r['n_steps'] for r in rows)
    print(f'\nScanned {len(rows)} trajectories / {total_steps} frames')

    print(f'\nSurvival at threshold {args.thresh}x, min_steps {args.min_steps}:')
    for src in sorted({r['source'] for r in rows}):
        sub = [r for r in rows if r['source'] == src]
        kept = [r for r in sub if r['first_bad'] >= args.min_steps]
        f_in = sum(r['n_steps'] for r in sub)
        f_out = sum(min(r['first_bad'], r['n_steps']) for r in kept)
        untouched = sum(1 for r in sub if r['first_bad'] >= r['n_steps'])
        print(f'  {src:<10} traj {len(kept):4d}/{len(sub):<4d}   '
              f'frames {f_out:6d}/{f_in:<6d} ({100.0 * f_out / max(f_in, 1):.0f}%)   '
              f'untouched {untouched}/{len(sub)}')

    fb = np.array([r['first_bad'] for r in rows if r['first_bad'] < r['n_steps']])
    if len(fb):
        print(f'\nFirst bad frame (blown-up trajectories only, n={len(fb)}): '
              f'p10={np.percentile(fb, 10):.0f} p50={np.percentile(fb, 50):.0f} '
              f'p90={np.percentile(fb, 90):.0f}')

    if args.dry_run:
        print('\nThreshold sensitivity (kept frames / total):')
        for th in (2.0, 2.5, 3.0, 4.0, 6.0):
            # Recomputing per threshold would need the series again; instead
            # report only what this run's threshold supports, and say so.
            if abs(th - args.thresh) < 1e-9:
                f_out = sum(min(r['first_bad'], r['n_steps']) for r in rows
                            if r['first_bad'] >= args.min_steps)
                print(f'  {th:<4} {f_out}/{total_steps} '
                      f'({100.0 * f_out / total_steps:.0f}%)   <- this run')
            else:
                print(f'  {th:<4} (re-run with --thresh {th})')
        return

    keep = {(r['split'], r['cloth'], r['traj']): min(r['first_bad'], r['n_steps'])
            for r in rows if r['first_bad'] >= args.min_steps}

    with h5py.File(args.src, 'r') as h, h5py.File(args.dst, 'w') as o:
        copy_attrs(h, o)
        o.attrs['cleaned_from'] = os.path.basename(args.src)
        o.attrs['clean_stretch_thresh'] = float(args.thresh)
        o.attrs['clean_min_steps'] = int(args.min_steps)
        n_traj = n_frames = 0
        for split, cname, c, tname, _, _ in iter_trajectories(h):
            n_keep = keep.get((split, cname, tname))
            if not n_keep:
                continue
            g_split = o.require_group(split)
            if cname not in g_split:
                g_cloth = g_split.create_group(cname)
                copy_attrs(c, g_cloth)
                for k in ('rest_positions', 'edges', 'faces'):
                    if k in c:
                        g_cloth.create_dataset(k, data=np.asarray(c[k]))
            else:
                g_cloth = g_split[cname]
            src_t = c[tname]
            g_t = g_cloth.create_group(tname)
            copy_attrs(src_t, g_t)
            for k in src_t.keys():
                if not k.startswith('step_'):
                    g_t.create_dataset(k, data=np.asarray(src_t[k]))
            steps = sorted(k for k in src_t.keys() if k.startswith('step_'))
            for sname in steps[:n_keep]:
                src_s = src_t[sname]
                g_s = g_t.create_group(sname)
                copy_attrs(src_s, g_s)

                def _copy(name, obj):
                    if isinstance(obj, h5py.Dataset):
                        g_s.create_dataset(name, data=np.asarray(obj))

                src_s.visititems(_copy)
            g_t.attrs['n_steps_kept'] = int(n_keep)
            g_t.attrs['n_steps_original'] = len(steps)
            n_traj += 1
            n_frames += n_keep
            if n_traj % 100 == 0:
                print(f'  ...wrote {n_traj} trajectories')

    print(f'\n[clean] wrote {args.dst}')
    print(f'[clean] {n_traj} trajectories, {n_frames} frames '
          f'({100.0 * n_frames / total_steps:.0f}% of input frames)')


if __name__ == '__main__':
    main()
