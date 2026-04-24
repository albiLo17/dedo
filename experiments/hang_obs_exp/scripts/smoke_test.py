"""
Smoke test: launch a minimal PPO run for each camera condition to verify the
full training pipeline executes without errors.

Runs 2000 env steps per condition (enough for ~10 PPO updates with 2 envs).
Saves checkpoints and TB logs under logs/hang_obs_exp/<cond>/smoke_test/

Usage (from repo root):
    python experiments/hang_obs_exp/scripts/smoke_test.py
"""

import sys, os, time, subprocess
from pathlib import Path

REPO_ROOT = str(Path(__file__).resolve().parents[3])
PYTHON = os.environ.get('PYTHON', 'python')

CONDITIONS = {
    'A_full': {
        'cam_resolution': 64,
        'cam_viewmat':   '9.0 -25.0 45.0 0.0 0.5 6.5',
        'extra_flags':   '--flat_obs',
        'description':   'Front-angled camera (hole+hanger visible)',
    },
    'B_partial': {
        'cam_resolution': 64,
        'cam_viewmat':   '8.0 -82.0 45.0 0.0 0.5 8.0',
        'extra_flags':   '--flat_obs',
        'description':   'Near-top-down camera (hole-hanger depth lost)',
    },
    'C_lowdim': {
        'cam_resolution': 0,
        'cam_viewmat':   None,
        'extra_flags':   '',
        'description':   'Low-dim gripper-only baseline (no camera)',
    },
}

SMOKE_STEPS = 2000
NUM_ENVS    = 2


def run_condition(name, cfg):
    logdir = os.path.join(REPO_ROOT, 'logs', 'hang_obs_exp', name, 'smoke_test')
    os.makedirs(logdir, exist_ok=True)

    cmd = [
        PYTHON, '-m', 'dedo.run_rl_sb3',
        '--env=HangProcCloth-v1',
        '--rl_algo=PPO',
        f'--logdir={logdir}',
        f'--cam_resolution={cfg["cam_resolution"]}',
        f'--num_envs={NUM_ENVS}',
        f'--total_env_steps={SMOKE_STEPS}',
        '--log_save_interval=5',
        '--lr=3e-4',
        '--seed=0',
    ]

    if cfg['cam_viewmat'] is not None:
        cmd += ['--cam_viewmat'] + cfg['cam_viewmat'].split()

    if cfg['extra_flags']:
        cmd += cfg['extra_flags'].split()

    print(f"\n{'='*60}")
    print(f"Condition {name}: {cfg['description']}")
    print(f"Command: {' '.join(cmd)}")
    print(f"Logdir:  {logdir}")
    print('='*60)

    t0 = time.time()
    proc = subprocess.run(cmd, capture_output=False, cwd=REPO_ROOT)
    elapsed = time.time() - t0

    if proc.returncode == 0:
        print(f"\n[OK] {name} completed in {elapsed:.1f}s")
        ckpt_dirs = [d for d in os.listdir(logdir) if os.path.isdir(os.path.join(logdir, d))]
        if ckpt_dirs:
            print(f"     Checkpoints found: {ckpt_dirs[:3]}")
        return True
    else:
        print(f"\n[FAIL] {name} exited with code {proc.returncode}")
        return False


if __name__ == '__main__':
    results = {}
    for name, cfg in CONDITIONS.items():
        ok = run_condition(name, cfg)
        results[name] = 'PASS' if ok else 'FAIL'

    print(f"\n{'='*60}")
    print("SMOKE TEST SUMMARY")
    print('='*60)
    for name, status in results.items():
        print(f"  {name:20s}: {status}")
    all_ok = all(v == 'PASS' for v in results.values())
    sys.exit(0 if all_ok else 1)
