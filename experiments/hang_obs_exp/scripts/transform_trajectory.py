#!/usr/bin/env python3
"""Bake the viewer's sim -> Franka-world transform into a trajectory.

`view_demo_inspect.py` renders a sim-frame rollout under `/sim_scene`,
which the GUI transforms into the Franka world frame by::

    world = offset + R_z(yaw_deg) @ (p_sim * sim_scale)

(see `_sim_to_world_transform` / `_sim_point_to_world` in that file).
This script applies that exact transform OFFLINE so you get world-frame
coordinates without launching the viewer. Defaults match the viewer's
default sliders::

    sim_scale = 0.045
    sim_offset = (0.5, 0.0, -0.039)
    sim_yaw_deg = 90

Input formats (auto-detected, same as the viewer's _load_exported_traj):

  - policy-eval rollout: pickle dict with obs={'pcd','grip','goal'},
    'acts' + metadata. grip/goal were divided by --proprio_scale (20)
    before export, and obs['pcd'] is raw sim-world coords -> this is
    the "sim frame" case the transform is meant for.
  - pickle list-of-dicts with 'ee_pos_left'/'ee_pos_right' per step,
    or an npz with flat (T,3) 'ee_pos_left'/'ee_pos_right'. These are
    USUALLY ALREADY in the Franka world frame (they come from the
    viewer's own export); the transform is still applied because you
    asked for it, but a warning is printed.

Output: a pickle (or .npz if --out ends with .npz) with explicit
world-frame arrays::

    ee_left  (T,3)   ee_right (T,3)        # metres, Franka world frame
    ee_left_vel (T,3) ee_right_vel (T,3)   # if grip carried velocity
    goal (3,) | absent
    pcd  (T,N,3) | absent                  # per-frame, world frame
    acts (T,6) | absent                    # passed through unchanged
    transform = {sim_scale, sim_offset, sim_yaw_deg, proprio_scale}
    plus any source metadata (success_*, ckpt, eval_seed, ...)

Examples
--------
    python transform_trajectory.py hemal_traj_ep001_seed12025_l1_h1_t0.pkl
    python transform_trajectory.py in.pkl --out world.npz --no-pcd
    python transform_trajectory.py in.pkl --sim_scale 0.05 --sim_yaw_deg 90 \
        --sim_offset 0.5 0.0 -0.039
"""
import argparse
import pickle
from pathlib import Path

import numpy as np


def build_transform(sim_scale, sim_offset, sim_yaw_deg):
    """Return (offset (3,), R (3,3), scale) — identical convention to
    view_demo_inspect._sim_to_world_transform (rotation about +z)."""
    yaw = np.radians(float(sim_yaw_deg))
    c, s = np.cos(yaw), np.sin(yaw)
    R = np.array([[c, -s, 0.0],
                  [s,  c, 0.0],
                  [0.0, 0.0, 1.0]], dtype=np.float64)
    offset = np.asarray(sim_offset, dtype=np.float64).reshape(3)
    return offset, R, float(sim_scale)


def to_world_points(p, offset, R, scale):
    """(..., 3) sim points -> (..., 3) world points.
    world = offset + R @ (p * scale), broadcast over leading axes."""
    p = np.asarray(p, dtype=np.float64)
    flat = p.reshape(-1, 3)
    out = (flat * scale) @ R.T + offset
    return out.reshape(p.shape).astype(np.float32)


def to_world_vectors(v, R, scale):
    """(..., 3) sim vectors (velocities) -> world; rotation+scale, no
    translation."""
    v = np.asarray(v, dtype=np.float64)
    flat = v.reshape(-1, 3)
    out = (flat * scale) @ R.T
    return out.reshape(v.shape).astype(np.float32)


def _bbox(a):
    a = np.asarray(a, dtype=np.float64).reshape(-1, 3)
    if a.size == 0:
        return 'empty'
    lo, hi = a.min(0), a.max(0)
    return (f'[{lo[0]:.3f},{lo[1]:.3f},{lo[2]:.3f}] .. '
            f'[{hi[0]:.3f},{hi[1]:.3f},{hi[2]:.3f}]')


def load_trajectory(path, proprio_scale):
    """Detect format and return a dict of SIM-FRAME arrays + metadata:
        {'kind', 'ee_left', 'ee_right', 'ee_left_vel', 'ee_right_vel',
         'goal', 'pcd', 'acts', 'meta'}
    Missing fields are None. Mirrors view_demo_inspect._load_exported_traj.
    """
    p = Path(path).expanduser()
    if not p.exists():
        raise FileNotFoundError(p)

    out = {'kind': None, 'ee_left': None, 'ee_right': None,
            'ee_left_vel': None, 'ee_right_vel': None,
            'goal': None, 'pcd': None, 'acts': None, 'meta': {}}

    if p.suffix.lower() == '.npz':
        d = np.load(p, allow_pickle=False)
        if 'ee_pos_left' not in d or 'ee_pos_right' not in d:
            raise ValueError(f'npz missing ee_pos_left/right; keys={list(d)}')
        n = min(len(d['ee_pos_left']), len(d['ee_pos_right']))
        out.update(kind='realworld',
                   ee_left=np.asarray(d['ee_pos_left'][:n], np.float32),
                   ee_right=np.asarray(d['ee_pos_right'][:n], np.float32))
        return out

    with p.open('rb') as f:
        data = pickle.load(f)

    # Policy-eval rollout (the sim-frame case this script targets).
    if (isinstance(data, dict) and isinstance(data.get('obs'), dict)
            and 'grip' in data['obs']):
        obs = data['obs']
        grip = np.asarray(obs['grip'], dtype=np.float32)
        if grip.ndim != 2 or grip.shape[1] < 6:
            raise ValueError(f'unexpected obs["grip"] shape {grip.shape}')
        half = grip.shape[1] // 2     # 12 -> 6: [Lpos3 Lvel3 | Rpos3 Rvel3]
        out['kind'] = 'sim'
        out['ee_left'] = grip[:, 0:3] * proprio_scale
        out['ee_right'] = grip[:, half:half + 3] * proprio_scale
        if half >= 6:
            out['ee_left_vel'] = grip[:, 3:6] * proprio_scale
            out['ee_right_vel'] = grip[:, half + 3:half + 6] * proprio_scale
        if len(np.asarray(obs.get('goal', []))):
            out['goal'] = (np.asarray(obs['goal'], np.float32)[0]
                           * proprio_scale)
        if 'pcd' in obs:
            pcd = np.asarray(obs['pcd'], np.float32)   # raw sim-world
            out['pcd'] = pcd[None] if pcd.ndim == 2 else pcd
        if 'acts' in data:
            out['acts'] = np.asarray(data['acts'], np.float32)
        for k in ('success_hanging', 'success_topological',
                  'success_legacy', 'success', 'len', 'obs_mode',
                  'ctrl_freq', 'sim_freq', 'sim_steps_per_action',
                  'max_act_vel', 'eval_seed', 'episode_idx', 'ckpt',
                  'cam_viewmat', 'cam_resolution', 'pcd_n_points'):
            if k in data:
                out['meta'][k] = data[k]
        return out

    # Legacy list-of-dicts (already world frame, normally).
    if isinstance(data, list) and len(data) > 0:
        out['kind'] = 'realworld'
        out['ee_left'] = np.asarray([s['ee_pos_left'] for s in data],
                                    np.float32)
        out['ee_right'] = np.asarray([s['ee_pos_right'] for s in data],
                                     np.float32)
        if 'pcd' in data[0]:
            out['pcd'] = np.stack(
                [np.asarray(s['pcd'], np.float32) for s in data])
        return out

    raise ValueError(f'unrecognized trajectory format: {type(data).__name__}')


def main():
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('input', help='Trajectory file (pkl or npz).')
    ap.add_argument('--out', default='',
                    help='Output path. Default: <input>_world.pkl. '
                         'End with .npz to save arrays as npz.')
    ap.add_argument('--sim_scale', type=float, default=0.045)
    ap.add_argument('--sim_offset', type=float, nargs=3,
                    default=[0.5, 0.0, -0.039],
                    metavar=('X', 'Y', 'Z'))
    ap.add_argument('--sim_yaw_deg', type=float, default=90.0)
    ap.add_argument('--proprio_scale', type=float, default=20.0,
                    help="Undo the exporter's grip/goal /N normalization "
                         '(sim-frame inputs only).')
    ap.add_argument('--no-pcd', action='store_true',
                    help='Skip transforming/writing the (large) pcd.')
    args = ap.parse_args()

    offset, R, scale = build_transform(
        args.sim_scale, args.sim_offset, args.sim_yaw_deg)
    print(f'[transform] sim_scale={scale} '
          f'offset={offset.tolist()} yaw_deg={args.sim_yaw_deg}')

    traj = load_trajectory(args.input, args.proprio_scale)
    print(f'[transform] loaded {args.input}  (kind={traj["kind"]})')
    if traj['kind'] == 'realworld':
        print('[transform] WARNING: this format is normally ALREADY in '
              'the Franka world frame; applying the transform anyway '
              'because you asked. Use --sim_scale 1 --sim_yaw_deg 0 '
              '--sim_offset 0 0 0 for an identity passthrough.')

    result = {
        'transform': {
            'sim_scale': scale,
            'sim_offset': offset.tolist(),
            'sim_yaw_deg': float(args.sim_yaw_deg),
            'proprio_scale': float(args.proprio_scale),
        },
        'source': str(Path(args.input).expanduser()),
        'frame': 'franka_world',
        **traj['meta'],
    }

    for key, is_vec in (('ee_left', False), ('ee_right', False),
                        ('ee_left_vel', True), ('ee_right_vel', True),
                        ('goal', False), ('pcd', False)):
        v = traj.get(key)
        if v is None:
            continue
        if key == 'pcd' and args.no_pcd:
            continue
        if is_vec:
            result[key] = to_world_vectors(v, R, scale)
        else:
            before = _bbox(v)
            result[key] = to_world_points(v, offset, R, scale)
            print(f'[transform] {key}: {np.asarray(v).shape}  '
                  f'sim {before}  ->  world {_bbox(result[key])}')
    if traj.get('acts') is not None:
        result['acts'] = traj['acts']           # control inputs, unchanged

    out_path = Path(args.out).expanduser() if args.out else \
        Path(args.input).expanduser().with_name(
            Path(args.input).stem + '_world.pkl')
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if out_path.suffix.lower() == '.npz':
        np.savez_compressed(
            out_path,
            **{k: v for k, v in result.items()
               if isinstance(v, np.ndarray)})
    else:
        with out_path.open('wb') as f:
            pickle.dump(result, f, protocol=pickle.HIGHEST_PROTOCOL)
    print(f'[transform] wrote {out_path}')


if __name__ == '__main__':
    main()
