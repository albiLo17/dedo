#!/usr/bin/env python
"""Scan collected BC demos for cloth meshes that blow up.

A demo can be structurally valid and physically nonsense: dedo's explicit
spring solver goes unstable if the cloth is driven too hard, and the result is
a mesh whose vertices fly apart. Nothing downstream complains — the arrays are
the right shape, the success metric may even pass — so the only way to catch it
is to measure the geometry.

Three independent signatures, because each catches a different failure:

  * **edge stretch** — max edge length / its own rest length. The direct
    signature of spring blow-up, and the one that fires first.
  * **extent growth** — bounding-box diagonal / rest diagonal. Catches a mesh
    that inflates without any single edge looking extreme.
  * **non-finite** — NaN/Inf vertices. The terminal state of a blow-up.

Reports the worst frame per demo and the distribution over the dataset, so
"do meshes explode?" gets a number rather than an impression.

    python _diag_mesh_explosions.py --demos <dir> [--stretch_thresh 3.0]
"""
import argparse
import glob
import os
import pickle

import numpy as np


def cloth_edges(faces):
    """Unique undirected edges of a triangle mesh."""
    e = np.concatenate([faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]])
    e = np.sort(e, axis=1)
    return np.unique(e, axis=0)


def scan_demo(path):
    with open(path, 'rb') as f:
        d = pickle.load(f)
    mesh = np.asarray(d['obs']['full_mesh'], dtype=np.float64)
    T = mesh.shape[0]
    # Layout: 12 grip values, then 250 padded vertex slots of xyz.
    verts = mesh[:, 12:].reshape(T, -1, 3)
    faces = np.asarray(d['cloth_faces'], dtype=np.int64)

    # Real vertices are the ones the collector actually filled; the padding is
    # exact zeros. Key the mask off the REST frame, so a real vertex that
    # happens to pass through the origin mid-episode is not dropped.
    rest = verts[0]
    node_mask = np.abs(rest).sum(-1) > 0
    n_real = int(node_mask.sum())

    edges = cloth_edges(faces)
    edges = edges[(edges[:, 0] < verts.shape[1]) & (edges[:, 1] < verts.shape[1])]
    edges = edges[node_mask[edges[:, 0]] & node_mask[edges[:, 1]]]

    finite = np.isfinite(verts).all(axis=(1, 2))
    n_nonfinite_frames = int((~finite).sum())

    v = np.nan_to_num(verts, nan=0.0, posinf=0.0, neginf=0.0)
    rest_len = np.linalg.norm(rest[edges[:, 0]] - rest[edges[:, 1]], axis=-1)
    ok = rest_len > 1e-9
    edges, rest_len = edges[ok], rest_len[ok]

    lens = np.linalg.norm(v[:, edges[:, 0]] - v[:, edges[:, 1]], axis=-1)
    stretch = lens / rest_len[None, :]
    per_frame_stretch = stretch.max(axis=1)

    real = v[:, node_mask]
    extent = np.linalg.norm(real.max(axis=1) - real.min(axis=1), axis=-1)
    rest_extent = max(extent[0], 1e-9)
    per_frame_growth = extent / rest_extent

    return {
        'demo': os.path.basename(path),
        'T': T,
        'n_real': n_real,
        'n_edges': len(edges),
        'max_stretch': float(per_frame_stretch.max()),
        'max_stretch_frame': int(per_frame_stretch.argmax()),
        'final_stretch': float(per_frame_stretch[-1]),
        'max_growth': float(per_frame_growth.max()),
        'max_growth_frame': int(per_frame_growth.argmax()),
        'n_nonfinite_frames': n_nonfinite_frames,
        'success_hanging': int(d['success_hanging']),
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--demos', required=True)
    p.add_argument('--stretch_thresh', type=float, default=3.0,
                   help='Max edge stretch above which a demo is called blown up.')
    p.add_argument('--growth_thresh', type=float, default=3.0)
    p.add_argument('--limit', type=int, default=0)
    p.add_argument('--csv', default=None)
    args = p.parse_args()

    paths = sorted(glob.glob(os.path.join(args.demos, '*.pkl')))
    if args.limit:
        paths = paths[:args.limit]
    if not paths:
        raise SystemExit(f'no demos under {args.demos}')

    rows = []
    for i, path in enumerate(paths):
        try:
            rows.append(scan_demo(path))
        except Exception as e:
            print(f'[warn] {os.path.basename(path)}: {e!r}')
        if (i + 1) % 100 == 0:
            print(f'  ...{i + 1}/{len(paths)}')

    stretch = np.array([r['max_stretch'] for r in rows])
    growth = np.array([r['max_growth'] for r in rows])
    nonfinite = np.array([r['n_nonfinite_frames'] for r in rows])

    print(f'\nScanned {len(rows)} demos '
          f'({rows[0]["n_real"]}–{max(r["n_real"] for r in rows)} real verts)')
    print('\nMax edge stretch (x rest length) over the whole episode:')
    for q in (50, 90, 99, 100):
        print(f'  p{q:<3} {np.percentile(stretch, q):.2f}')
    print('\nMax bbox growth (x rest diagonal):')
    for q in (50, 90, 99, 100):
        print(f'  p{q:<3} {np.percentile(growth, q):.2f}')
    print(f'\nDemos with non-finite vertices: {(nonfinite > 0).sum()}')

    bad = [r for r in rows
           if r['max_stretch'] > args.stretch_thresh
           or r['max_growth'] > args.growth_thresh
           or r['n_nonfinite_frames'] > 0]
    print(f'\nDemos over threshold (stretch>{args.stretch_thresh} or '
          f'growth>{args.growth_thresh} or non-finite): {len(bad)}/{len(rows)}')
    for r in sorted(bad, key=lambda r: -r['max_stretch'])[:20]:
        print(f'  {r["demo"]}  stretch={r["max_stretch"]:.1f}@f{r["max_stretch_frame"]}'
              f'  growth={r["max_growth"]:.1f}@f{r["max_growth_frame"]}'
              f'  nonfinite={r["n_nonfinite_frames"]}'
              f'  hanging={r["success_hanging"]}')

    if args.csv:
        import csv
        with open(args.csv, 'w', newline='') as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)
        print(f'\n[csv] wrote {args.csv}')


if __name__ == '__main__':
    main()
