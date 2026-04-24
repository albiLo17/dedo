"""
Generate RGB screenshots of HangProcCloth-v1 under different camera configurations.

This script creates the two key observability conditions:
  A (full):    Front-angled camera — hole AND hanger hook clearly visible.
  B (partial): Near-top-down camera — cloth blocks hole↔hook relationship.

Usage (from repo root):
  python experiments/hang_obs_exp/scripts/gen_screenshots.py

Output: PNG images in experiments/hang_obs_exp/viz_output/
"""

import sys, os
sys.argv = ['gen_screenshots', '--env=HangProcCloth-v1', '--cam_resolution=200']

from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import gym
import dedo
from dedo.utils.args import get_args

OUTDIR = str(Path(__file__).resolve().parents[1] / 'viz_output')
os.makedirs(OUTDIR, exist_ok=True)


def make_env(cam_viewmat, seed=42, cam_resolution=200):
    sys.argv = [
        'gen_screenshots',
        '--env=HangProcCloth-v1',
        f'--cam_resolution={cam_resolution}',
        '--cam_viewmat', *[str(v) for v in cam_viewmat],
    ]
    args = get_args()
    args.seed = seed
    env = gym.make(args.env, args=args)
    env.seed(seed)
    return env


def run_n_steps(env, n=10):
    obs = env.reset()
    for i in range(n):
        act = np.zeros(env.action_space.shape)
        act[1] = -0.3
        act[4] = -0.3
        obs, _, done, _ = env.step(act)
        if done:
            break
    return obs


CAM_CONFIGS = {
    'A_full': {
        'cam_viewmat': [9.0, -25.0, 45.0, 0.0, 0.5, 6.5],
        'label': 'Full Observability\n(hole + hanger visible)',
        'description': 'Front-angled camera at 45° yaw, -25° pitch. '
                       'Both the cloth hole and the hanger hook are visible.',
    },
    'B_partial': {
        'cam_viewmat': [8.0, -82.0, 45.0, 0.0, 0.5, 8.0],
        'label': 'Partial Observability\n(hole-hanger relation hidden)',
        'description': 'Near-top-down camera at -82° pitch. '
                       'Cloth top surface visible but depth to hanger hook is lost.',
    },
    'preset_default': {
        'cam_viewmat': [8.8, -12.6, 314.0, -0.4, 0.6, 5.3],
        'label': 'Preset Default\n(original repo view)',
        'description': 'The default camera from DEFORM_INFO[procedural_hang_cloth].',
    },
}


def capture_frames(cam_config, n_steps_list=(0, 5, 15, 30), seed=42):
    env = make_env(cam_config['cam_viewmat'], seed=seed)
    obs = env.reset()
    frames = {}
    step = 0

    if 0 in n_steps_list:
        frames[0] = obs.copy() if isinstance(obs, np.ndarray) else obs

    act = np.zeros(env.action_space.shape, dtype=np.float32)
    act[1] = -0.3
    act[4] = -0.3

    while step < max(n_steps_list):
        obs, _, done, _ = env.step(act)
        step += 1
        if step in n_steps_list:
            frames[step] = obs.copy() if isinstance(obs, np.ndarray) else obs
        if done:
            break

    env.close()
    return frames


def save_comparison_figure():
    print("Generating comparison figure...")
    cond_frames = {}

    for cond_name, cfg in CAM_CONFIGS.items():
        print(f"  Rendering {cond_name} ...")
        frames = capture_frames(cfg, n_steps_list=[0, 10, 30])
        cond_frames[cond_name] = frames

    n_conds = len(CAM_CONFIGS)
    steps_to_show = [0, 10, 30]
    fig, axes = plt.subplots(n_conds, len(steps_to_show),
                             figsize=(4 * len(steps_to_show), 3.5 * n_conds))

    for row, (cond_name, cfg) in enumerate(CAM_CONFIGS.items()):
        for col, step_n in enumerate(steps_to_show):
            ax = axes[row][col]
            frame = cond_frames[cond_name].get(step_n)
            if frame is not None:
                if frame.dtype != np.uint8:
                    disp = (frame * 255).clip(0, 255).astype(np.uint8)
                else:
                    disp = frame
                ax.imshow(disp)
            ax.set_xticks([])
            ax.set_yticks([])
            if col == 0:
                ax.set_ylabel(cfg['label'], fontsize=9)
            if row == 0:
                ax.set_title(f'Step {step_n}', fontsize=10)

    fig.suptitle('HangProcCloth-v1: Observability Conditions\n'
                 '(Row = camera config; Col = timestep)', fontsize=11)
    plt.tight_layout()
    out_path = os.path.join(OUTDIR, 'observability_comparison.png')
    plt.savefig(out_path, dpi=120, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {out_path}")
    return out_path


def save_individual_frames():
    saved = []
    for cond_name, cfg in CAM_CONFIGS.items():
        print(f"  Saving individual frames for {cond_name} ...")
        frames = capture_frames(cfg, n_steps_list=[0, 10, 30])
        for step_n, frame in frames.items():
            if frame.dtype != np.uint8:
                disp = (frame * 255).clip(0, 255).astype(np.uint8)
            else:
                disp = frame
            path = os.path.join(OUTDIR, f'{cond_name}_step{step_n:03d}.png')
            plt.imsave(path, disp)
            saved.append(path)
            print(f"    -> {path}")
    return saved


if __name__ == '__main__':
    print("=" * 60)
    print("DEDO HangProcCloth-v1 Observability Visualization")
    print("=" * 60)
    print()
    save_comparison_figure()
    save_individual_frames()
    print()
    print("Done. Output saved to:", OUTDIR)
    print("Files:")
    for f in sorted(os.listdir(OUTDIR)):
        print(" ", os.path.join(OUTDIR, f))
