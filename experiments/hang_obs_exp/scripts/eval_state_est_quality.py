"""Measure the state estimator ITSELF, offline, on held-out topologies.

Why this exists separately from the policy eval
-----------------------------------------------
`train_diffusion_bc.py --use_state_estimator` reports a hole-centroid error, but
only at the operating points the policy happens to visit, in a DIFFERENT env
than the estimator trained on (HangProcCloth-v1 is ~22.2x real scale;
`dedo_hang_real_clean.h5` was collected in the real-scale env). Mixing "is the
module accurate" with "does the policy tolerate it" and with a scale change is
how you end up unable to attribute a bad number.

This probe removes all three confounds: it runs on the estimator's OWN
validation split (held-out topologies — every DEDO episode re-randomizes the
procedural cloth, so `validation/` shares no cloth with `training/`), in the
units the model trained in, and scores the mesh rather than the policy.

It deliberately goes through the SAME serving path the eval uses
(`cloth_state_estimator.HoleEstimator` -> websocket -> `serve_predictor.py`)
rather than calling the pipeline in-process. A number produced by a different
code path than the one under test would not catch a serving bug, which is
exactly the class of bug suspected here.

Metrics, all in the H5's own (real-scale) metres:
  * `vert_mean` / `vert_p90` — per-vertex Euclidean error to the GT mesh. The
    headline: it scores the whole representation, not one derived point.
  * `hole_err` — error of the hole-loop centroid, i.e. the quantity the policy
    is actually handed. Reported alongside `vert_mean` so a model that gets the
    cloth right but the hole wrong is visible.
  * `extent_ratio` — predicted bbox diagonal / GT bbox diagonal. A diffusion
    estimator that quietly SHRINKS its meshes still posts a respectable
    per-vertex error while biasing the hole centroid toward the cloth centre.
    Watch this sit at 1.0, not just the error go down.
  * `occlusion` — MEASURED, not assumed: 1 - (fraction of GT verts within
    `--tau` of an observed point). Same definition as the GarmentLab report, so
    the two are comparable. The stored clouds are already single-camera partial,
    so occlusion is nonzero even at drop fraction 0.

Occlusion is swept by removing a spatial CHUNK (the points furthest along a
random direction), not by random point dropout. Random dropout thins the cloud
uniformly and the model barely notices; real occluders remove contiguous
regions, which is the failure mode worth measuring.

Usage
-----
    # server (UniClothDiff venv):
    .venv/bin/python scripts/serve_predictor.py --task state_estimation \
        --checkpoint checkpoints/state-estimation/output/dedohang_se_clean --port 8004
    # this probe (dedo conda env):
    python eval_state_est_quality.py --h5 <path> --port 8004 --out results.json
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections import defaultdict

import h5py
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from cloth_state_estimator import HoleEstimator  # noqa: E402


def occlude_chunk(pcd, frac, rng):
    """Remove the `frac` of points furthest along a random direction.

    A contiguous spatial bite, not uniform thinning: an object in the way takes
    a connected region of the cloth, and that is what the estimator has to
    inpaint. Uniform dropout leaves the global shape intact at any rate short of
    extreme, so it measures point-density robustness, not occlusion.
    """
    if frac <= 0:
        return pcd
    n_keep = max(16, int(round(len(pcd) * (1.0 - frac))))
    if n_keep >= len(pcd):
        return pcd
    d = rng.normal(size=3)
    d /= np.linalg.norm(d) + 1e-9
    keep = np.argsort(pcd @ d)[:n_keep]
    return pcd[keep]


def measure_occlusion(gt_verts, pcd, tau):
    """1 - fraction of GT vertices within `tau` of an observed point.

    Measures what the sensor actually delivered about the cloth, so it stays
    comparable across cloths, cameras and occluder geometries — unlike a drop
    fraction, which is a knob setting rather than an observable.
    """
    if len(pcd) == 0:
        return 1.0
    # (V, P) distances: V<=256, P<=2048 here, so the dense form is fine.
    d = np.linalg.norm(gt_verts[:, None, :] - pcd[None, :, :], axis=-1)
    return float(1.0 - (d.min(axis=1) < tau).mean())


def bbox_diag(v):
    return float(np.linalg.norm(v.max(axis=0) - v.min(axis=0)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--h5', required=True)
    ap.add_argument('--split', default='validation')
    ap.add_argument('--host', default='localhost')
    ap.add_argument('--port', type=int, default=8004)
    ap.add_argument('--steps', type=int, default=50,
                    help='diffusion denoising steps (50 = the eval setting)')
    ap.add_argument('--n_episodes', type=int, default=40)
    ap.add_argument('--frames_per_episode', type=int, default=4,
                    help='frames sampled evenly across the trajectory, so the '
                         'lift/thread phases are all represented')
    ap.add_argument('--drop_fracs', type=str, default='0,0.2,0.4,0.6,0.8')
    ap.add_argument('--tau', type=float, default=0.02,
                    help='metres; a GT vertex counts observed within this of a '
                         'point. 2 cm matches the GarmentLab report.')
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--out', required=True)
    ap.add_argument('--dump_meshes', type=str, default='',
                    help='optional .npz of predicted/GT meshes for the '
                         'qualitative renders (keeps this the single source of '
                         'truth for both the numbers and the pictures)')
    args = ap.parse_args()

    fracs = [float(x) for x in args.drop_fracs.split(',')]
    rng = np.random.default_rng(args.seed)
    est = HoleEstimator(backend='remote', host=args.host, port=args.port,
                        num_inference_steps=args.steps)
    print(f'[probe] connected; metadata={est.metadata}', flush=True)

    rows = []
    dumps = []
    with h5py.File(args.h5, 'r') as f:
        g = f[args.split]
        keys = sorted(g.keys())[:args.n_episodes]
        for ei, ck in enumerate(keys):
            ep = g[ck]
            rest = np.asarray(ep['rest_positions'], np.float32)
            edges = np.asarray(ep['edges'], np.int64)
            hole_idx = np.asarray(ep.attrs['hole_vertex_indices'], np.int64)
            traj = ep['trajectory_0']
            steps = sorted(k for k in traj.keys() if k.startswith('step_'))
            if not steps:
                continue
            # Evenly spaced frames: the cloth's observability and shape change a
            # lot between the flat start and the threaded end, so a fixed early
            # frame would report a systematically easy number.
            sel = np.linspace(0, len(steps) - 1,
                              min(args.frames_per_episode, len(steps))).astype(int)
            for si in sel:
                s = traj[steps[si]]
                gt = np.asarray(s['positions'], np.float32)[:len(rest)]
                # `pointclouds` is a GROUP with one dataset per camera (cam_0,
                # ...). Concatenate them the way the training dataset's
                # _merge_pointclouds_from_group does, so the probe sees exactly
                # the cloud the model was trained against.
                pcd_full = np.concatenate(
                    [np.asarray(s['pointclouds'][c], np.float32)
                     for c in sorted(s['pointclouds'].keys())], axis=0)
                if not (np.isfinite(gt).all() and np.isfinite(pcd_full).all()):
                    continue
                for fr in fracs:
                    pcd = occlude_chunk(pcd_full, fr, rng)
                    if len(pcd) < 16:
                        continue
                    try:
                        pred = est.predict_mesh(pcd, rest, edges=edges)
                    except Exception as e:      # a dead server must not look
                        print(f'[probe] infer failed: {e}', flush=True)
                        raise
                    pred = np.asarray(pred, np.float32)[:len(rest)]
                    verr = np.linalg.norm(pred - gt, axis=-1)
                    hi = hole_idx[hole_idx < len(gt)]
                    hole_err = (float(np.linalg.norm(
                        pred[hi].mean(0) - gt[hi].mean(0))) if len(hi) else float('nan'))
                    rows.append({
                        'cloth': ck, 'step': int(si), 'drop_frac': fr,
                        'occlusion': measure_occlusion(gt, pcd, args.tau),
                        'n_points': int(len(pcd)),
                        'vert_mean': float(verr.mean()),
                        'vert_median': float(np.median(verr)),
                        'vert_p90': float(np.percentile(verr, 90)),
                        'hole_err': hole_err,
                        'extent_ratio': bbox_diag(pred) / max(bbox_diag(gt), 1e-9),
                        'rest_scale': float(np.sqrt(
                            ((rest - rest.mean(0)) ** 2).sum(-1).mean())),
                        'gt_diag': bbox_diag(gt),
                    })
                    if args.dump_meshes and len(dumps) < 400:
                        dumps.append({
                            'cloth': ck, 'step': int(si), 'drop_frac': fr,
                            'pred': pred, 'gt': gt, 'pcd': pcd,
                            'hole_idx': hi,
                            'occlusion': rows[-1]['occlusion'],
                            'vert_mean': rows[-1]['vert_mean'],
                        })
            print(f'[probe] {ei + 1}/{len(keys)} {ck}: {len(rows)} rows',
                  flush=True)

    est.close()

    # Aggregate per drop fraction. Report the MEASURED occlusion alongside, so
    # the table's x-axis is an observable rather than the knob we turned.
    by = defaultdict(list)
    for r in rows:
        by[r['drop_frac']].append(r)
    summary = []
    for fr in fracs:
        rs = by.get(fr, [])
        if not rs:
            continue

        def col(k):
            return np.array([r[k] for r in rs if np.isfinite(r[k])])
        summary.append({
            'drop_frac': fr, 'n': len(rs),
            'occlusion_mean': float(col('occlusion').mean()),
            'vert_mean': float(col('vert_mean').mean()),
            'vert_mean_p90': float(np.percentile(col('vert_mean'), 90)),
            'hole_err_mean': float(col('hole_err').mean()),
            'hole_err_median': float(np.median(col('hole_err'))),
            'extent_ratio_mean': float(col('extent_ratio').mean()),
        })

    out = {'args': vars(args), 'summary': summary, 'rows': rows,
           'metadata': {k: (v.tolist() if isinstance(v, np.ndarray) else v)
                        for k, v in est.metadata.items()}}
    with open(args.out, 'w') as fh:
        json.dump(out, fh, indent=2)
    print(json.dumps(summary, indent=2))
    print(f'[probe] wrote {args.out} ({len(rows)} rows)')

    if args.dump_meshes and dumps:
        np.savez_compressed(
            args.dump_meshes,
            **{f'{i}_{k}': v for i, d in enumerate(dumps) for k, v in d.items()})
        print(f'[probe] wrote {args.dump_meshes} ({len(dumps)} meshes)')


if __name__ == '__main__':
    main()
