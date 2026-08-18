#!/usr/bin/env python
"""Convert BC demo pkls into the HDF5 UniClothDiff's state estimator reads.

Why convert rather than re-collect: `collect_bc_demos_GSE.py` writes this same
schema but by running the simulator again, which would cost hours and produce
DIFFERENT episodes. Everything the estimator needs is already in the BC pkls,
so the estimator can be trained on exactly the episodes the policies saw.

Schema written (matches collect_bc_demos_GSE.py, and what
src/datasets/cloth_state_est_variable.py actually reads):

    {training,validation}/cloth_NNNNN/
        rest_positions (V,3) float32
        faces          (F,3) int64
        edges          (E,2) int64          <- derived from faces
        trajectory_0/step_TTTT/
            positions          (V,3) float32
            pointclouds/cam_0  (N,3) float32

UNITS ARE THE TRAP. `obs['pcd']` is stored in raw sim units, but `full_mesh` is
stored pre-divided by WBOX=20. Writing them without rescaling would hand the
estimator a mesh 20x smaller than its own point cloud, and nothing downstream
would complain -- it would just never learn. Verified here per episode: the
mesh and cloud extents must agree to within a few percent, and an episode that
fails that check is dropped loudly rather than written.

`rest_positions` is the first frame, which is what
train_diffusion_bc.py:1468 already uses and what the reference dataset
`dedo_hang_real_clean.h5` contains (its rest_positions equal step_0000 exactly).

    python bc_demos_to_state_est_h5.py --demo_path <dir> --out <file.h5>
"""
import argparse
import glob
import os
import pickle

import h5py
import numpy as np

WBOX = 20.0
# Sim units -> metres. The estimator's configs normalise by metre-scale
# constants (rest_extent_norm 0.1225), and the reference dataset it was
# developed on stores metres: rest extent 0.169 m. Writing sim units instead
# makes every input ~47x too large after normalisation -- training looks fine
# for ~50 validations and then diverges three orders of magnitude and never
# recovers. Cost six hours of GPU to find, so it is asserted below, not
# trusted.
SIM_TO_M = 0.045


def edges_from_faces(faces):
    """Unique undirected edges, both directions, as the dataset expects."""
    e = set()
    for a, b, c in faces:
        for u, v in ((a, b), (b, c), (c, a)):
            e.add((int(u), int(v)))
            e.add((int(v), int(u)))
    return np.asarray(sorted(e), dtype=np.int64)


def episode_arrays(d):
    """(positions (T,V,3), rest (V,3), faces (F,3), pcd (T,N,3)) in SIM UNITS."""
    fm = np.asarray(d['obs']['full_mesh'], dtype=np.float32)
    pos_all = fm[:, 12:].reshape(len(fm), -1, 3) * WBOX      # undo the /WBOX
    # Padded vertex slots are exactly zero in every frame; the real vertex
    # count is per-episode, so it has to be measured, not assumed.
    keep = np.abs(pos_all[0]).sum(-1) > 0
    pos = pos_all[:, keep]
    faces = np.asarray(d['cloth_faces'], dtype=np.int64)
    faces = faces[(faces < pos.shape[1]).all(axis=1)]
    pcd = np.asarray(d['obs']['pcd'], dtype=np.float32)      # sim units
    pos = pos * SIM_TO_M
    pcd = pcd * SIM_TO_M
    return pos, pos[0].copy(), faces, pcd


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--demo_path', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--val_ratio', type=float, default=0.1)
    ap.add_argument('--max_demos', type=int, default=0)
    ap.add_argument('--extent_tol', type=float, default=0.25,
                    help='max relative disagreement between mesh and cloud '
                         'extent before an episode is rejected as unit-broken')
    a = ap.parse_args()

    files = sorted(glob.glob(os.path.join(a.demo_path, 'demo_*.pkl')))
    if a.max_demos:
        files = files[:a.max_demos]
    if not files:
        raise SystemExit(f'no demo_*.pkl under {a.demo_path}')
    n_val = int(round(len(files) * a.val_ratio))
    # Deterministic split by index, so a re-run reproduces it exactly.
    val_idx = set(np.linspace(0, len(files) - 1, n_val).astype(int).tolist())

    if os.path.exists(a.out):
        raise SystemExit(f'{a.out} exists; refusing to append to it '
                         f'(h5py would silently merge two datasets)')

    kept = dropped = 0
    with h5py.File(a.out, 'w') as f:
        for i, fp in enumerate(files):
            with open(fp, 'rb') as fh:
                d = pickle.load(fh)
            pos, rest, faces, pcd = episode_arrays(d)
            if len(faces) == 0 or pos.shape[1] < 8:
                print(f'  drop {os.path.basename(fp)}: degenerate mesh')
                dropped += 1
                continue
            # The unit check described in the docstring.
            me = float(np.linalg.norm(pos[0].ptp(0)))
            ce = float(np.linalg.norm(pcd[0].ptp(0)))
            if ce <= 1e-6 or abs(me - ce) / ce > a.extent_tol:
                print(f'  drop {os.path.basename(fp)}: mesh extent {me:.2f} vs '
                      f'cloud extent {ce:.2f} -- units disagree')
                dropped += 1
                continue

            # Unit guard: a cloth is tens of centimetres. Anything far outside
            # that means the sim-unit conversion was lost somewhere upstream.
            span = float(np.linalg.norm(rest.max(0) - rest.min(0)))
            if not 0.05 <= span <= 1.0:
                raise SystemExit(
                    f'{os.path.basename(fp)}: rest extent {span:.3f} m is not a '
                    f'plausible cloth size -- positions are probably still in '
                    f'sim units (expected ~0.25 m). Refusing to write a '
                    f'dataset that will silently diverge in training.')

            split = 'validation' if i in val_idx else 'training'
            g = f.require_group(f'{split}/cloth_{kept:05d}')
            g.create_dataset('rest_positions', data=rest.astype(np.float32))
            g.create_dataset('faces', data=faces)
            g.create_dataset('edges', data=edges_from_faces(faces))
            tg = g.create_group('trajectory_0')
            for t in range(len(pos)):
                sg = tg.create_group(f'step_{t:04d}')
                sg.create_dataset('positions', data=pos[t].astype(np.float32))
                sg.create_group('pointclouds').create_dataset(
                    'cam_0', data=pcd[t].astype(np.float32))
            kept += 1
            if kept % 50 == 0:
                print(f'  {kept}/{len(files)} written')

        f.attrs['source'] = a.demo_path
        f.attrs['pcd_n_points'] = int(pcd.shape[1])
        f.attrs['converted_from'] = 'bc_demos pkl (no re-simulation)'

    print(f'\nwrote {a.out}: {kept} episodes kept, {dropped} dropped')
    with h5py.File(a.out, 'r') as f:
        for s in ('training', 'validation'):
            if s in f:
                print(f'  {s}: {len(f[s])} clothes')


if __name__ == '__main__':
    main()
