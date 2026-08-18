#!/usr/bin/env python
"""Occlusion sweep for the cross-modality BC table.

Scores every trained observation mode at a pre-registered ladder of occluder
sizes, on IDENTICAL eval seeds, and reports success against the *measured*
hole visibility rather than against the occluder setting.

Why it is built this way
------------------------
* **The axis is measured, not set.** `--eval_occluder_size` is a knob; the
  x-axis is `final_eval/hole_visibility`, the fraction of the hole loop the
  eval camera can actually see (projection + depth test). Occluder settings
  are not comparable across scenes, cameras or cloths; visibility is, and it
  is the same axis the state-estimation arm sweeps.
* **Paired comparison.** Every (mode, size) cell runs the same
  `--n_episodes` on the same eval seed, so cells differ only in what the
  policy sees. At n=30 unpaired the SE is ~0.09 and resolves nothing.
* **The validity control is the `state` row.** It reads the simulator, so a
  visual occluder cannot reach it. If its success curve bends with occluder
  size, the occluder is perturbing the task and every other curve is
  uninterpretable. Check that first, before reading anything else.
* **Resumable.** A cell whose `final_eval_metrics.json` already exists is
  skipped, so an interrupted sweep picks up where it stopped.

Usage
-----
    python -u sweep_occlusion.py \
        --ckpt state=<path/policy.pt> --ckpt rgb=<path/policy.pt> \
        --ckpt pcd=<path/policy.pt>  --ckpt mesh=<path/policy.pt> \
        --n_episodes 50

Writes one run dir per cell under
`<logdir_root>/<mode>/occ_<mode>_s<size>` and an aggregate CSV at
`<logdir_root>/occlusion_sweep.csv`.
"""
import argparse
import csv
import json
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(HERE, '..', '..', '..'))

# Pre-registered ladder. Registered BEFORE looking at any success number on
# it, per plans/001-policy-usefulness.md.
#
# Re-calibrated 2026-07-31 with the TRACKING occluder, over whole episodes
# rather than one frame (mesh ckpt, n=5, mean visibility per step):
#   half-extent 0.00 -> 0.445 visible   (clear view; the cloth self-occludes
#                                        the rest, which is why the axis is
#                                        measured and not labelled 100%)
#   half-extent 0.15 -> 0.285
#   half-extent 0.30 -> 0.058           (effectively blind)
# So the whole usable range lives below 0.30 and the ladder is dense there.
# Above it the run stops measuring hole occlusion and starts measuring
# blindness.
DEFAULT_SIZES = [0.0, 0.06, 0.12, 0.18, 0.24, 0.30]

# Batch size per mode, mirroring the training runs. Eval does not depend on
# it, but keeping it identical keeps the run configs comparable.
BATCH_SIZE = {'state': 256, 'rgb': 64, 'pcd': 128, 'mesh': 64}


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--ckpt', action='append', required=True, metavar='MODE=PATH',
                   help='Checkpoint per obs mode, repeatable, e.g. '
                        '--ckpt state=runs/state/<run>/policy.pt')
    p.add_argument('--demo_path', type=str,
                   default=os.path.join(
                       REPO_ROOT, 'experiments', 'hang_obs_exp', 'data', 'bc',
                       'demos_v5_merged'))
    p.add_argument('--logdir_root', type=str,
                   default=os.path.join(
                       REPO_ROOT, 'experiments', 'hang_obs_exp', 'data', 'bc',
                       'runs'))
    p.add_argument('--sizes', type=float, nargs='+', default=DEFAULT_SIZES)
    p.add_argument('--n_episodes', type=int, default=50)
    p.add_argument('--seed', type=int, default=2026)
    p.add_argument('--action_horizon', type=int, default=4)
    p.add_argument('--python', type=str, default=sys.executable)
    p.add_argument('--dry_run', action='store_true')
    return p.parse_args()


def ckpt_tag(ckpt):
    """Short identifier for WHICH checkpoint a cell was scored with.

    It is part of the cell directory name so the resume-by-existence check
    can never hand back a cached cell computed from a different checkpoint —
    the exact way a 200-epoch curve would otherwise be silently reused for a
    400-epoch model.
    """
    run_dir = os.path.basename(os.path.dirname(os.path.abspath(ckpt)))
    stem = os.path.splitext(os.path.basename(ckpt))[0]   # policy / policy_best
    tag = run_dir.split('_')[0] or run_dir               # e.g. diff260730-145731
    return f'{tag}-{stem}'


def cell_dir(logdir_root, mode, size, ckpt):
    return os.path.join(logdir_root, mode,
                        f'occ_{mode}_{ckpt_tag(ckpt)}_s{size:g}')


def run_cell(args, mode, ckpt, size):
    """Run one (mode, occluder size) eval. Returns the metrics dict."""
    out_dir = cell_dir(args.logdir_root, mode, size, ckpt)
    metrics_path = os.path.join(out_dir, 'final_eval_metrics.json')
    if os.path.exists(metrics_path):
        print(f'[skip] {mode} size={size:g} — already scored on this ckpt')
        with open(metrics_path) as f:
            return json.load(f)

    cmd = [args.python, '-u',
           os.path.join(HERE, 'train_diffusion_bc.py'),
           '--demo_path', args.demo_path,
           '--obs_mode', mode,
           '--logdir_root', args.logdir_root,
           '--run_name', os.path.basename(out_dir),
           '--seed', str(args.seed),
           '--batch_size', str(BATCH_SIZE.get(mode, 64)),
           '--action_horizon', str(args.action_horizon),
           '--eval_only', '--resume', ckpt,
           '--n_final_eval_episodes', str(args.n_episodes),
           '--eval_occluder_size', str(size),
           '--measure_hole_visibility',
           # Videos are the dominant per-episode cost and a sweep needs
           # numbers, not footage. Record them separately for the levels
           # worth looking at.
           '--no_record_failed_videos', '--video_every_evals', '0']
    if mode == 'rgb':
        cmd.append('--pretrained_rgb')

    os.makedirs(args.logdir_root, exist_ok=True)
    log_path = os.path.join(args.logdir_root,
                            os.path.basename(out_dir) + '.log')
    print(f'[run ] {mode} size={size:g} -> {log_path}')
    if args.dry_run:
        print('       ' + ' '.join(cmd))
        return None
    with open(log_path, 'w') as log:
        rc = subprocess.call(cmd, stdout=log, stderr=subprocess.STDOUT,
                             cwd=REPO_ROOT)
    if rc != 0:
        print(f'[FAIL] {mode} size={size:g} exited {rc}; see {log_path}')
        return None
    with open(metrics_path) as f:
        return json.load(f)


def main():
    args = parse_args()
    ckpts = {}
    for spec in args.ckpt:
        mode, _, path = spec.partition('=')
        if not path:
            sys.exit(f'--ckpt expects MODE=PATH, got {spec!r}')
        if not os.path.exists(path):
            sys.exit(f'checkpoint for {mode} not found: {path}')
        ckpts[mode] = path

    rows = []
    csv_path = os.path.join(args.logdir_root, 'occlusion_sweep.csv')
    # Mode-major: a full curve for one mode lands before the next starts, so
    # a sweep that dies partway still has complete curves to read.
    for mode, ckpt in ckpts.items():
        for size in args.sizes:
            m = run_cell(args, mode, ckpt, size)
            if m is None:
                continue
            rows.append({
                'obs_mode': mode,
                'occluder_size': size,
                'hole_visibility': m.get('final_eval/hole_visibility'),
                'hole_visibility_std': m.get('final_eval/hole_visibility_std'),
                'success_hanging': m.get('final_eval/success_hanging'),
                'success_legacy': m.get('final_eval/success_legacy'),
                'success_topological': m.get('final_eval/success_topological'),
                'n_episodes': m.get('final_eval/n_episodes'),
                'eval_seed': m.get('eval_seed'),
                'ckpt': ckpt,
            })
            with open(csv_path, 'w', newline='') as f:
                w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
                w.writeheader()
                w.writerows(rows)
            print(f'  -> visibility={rows[-1]["hole_visibility"]} '
                  f'hanging={rows[-1]["success_hanging"]}')

    print(f'\n[done] {len(rows)} cells -> {csv_path}')


if __name__ == '__main__':
    main()
