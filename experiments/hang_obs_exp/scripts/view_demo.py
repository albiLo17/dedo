"""
View the hole-aware scripted demo on HangProcCloth-v1.

This uses the SAME build_hole_aware_waypoints logic as the BC pretrain
in train_privileged.py — so you can visually confirm the demos succeed
before kicking off a long training run.

Usage (from repo root):
  # Live pybullet GUI (recommended for debugging):
  python experiments/hang_obs_exp/scripts/view_demo.py --viz

  # Save mp4(s) to logs/hang_obs_exp/demo_view/:
  python experiments/hang_obs_exp/scripts/view_demo.py --num_episodes 5

  # Both:
  python experiments/hang_obs_exp/scripts/view_demo.py --viz --num_episodes 3
"""
import sys, os, argparse
from copy import deepcopy
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT))

import numpy as np
import gym
import pybullet
import cv2

import dedo
from dedo.utils.args import get_args_parser, args_postprocess
from dedo.envs.deform_env import DeformEnv
from dedo.demo_preset import build_traj, merge_traj

from experiments.hang_obs_exp.envs.privileged_env import PrivilegedObsWrapper
sys.path.insert(0, str(Path(__file__).resolve().parent))
from _helpers import RetryResetEnv, build_hole_aware_waypoints  # noqa: E402


parser = argparse.ArgumentParser()
parser.add_argument('--num_episodes', type=int, default=3)
parser.add_argument('--viz', action='store_true',
                    help='Open the live pybullet GUI window')
parser.add_argument('--cam_resolution', type=int, default=400,
                    help='mp4 frame size (set 0 to disable mp4)')
parser.add_argument('--logdir', type=str,
                    default=str(REPO_ROOT / 'logs' / 'hang_obs_exp' /
                                'demo_view'))
parser.add_argument('--seed', type=int, default=42)
parser.add_argument('--obs_mode', type=str, default='hole_centroid',
                    choices=['hole_centroid', 'hole_vertices', 'full_mesh'])
parser.add_argument('--success_factor', type=float, default=1.2,
                    help='Adaptive success: dist < success_factor*hole_radius. '
                         'Matches the default used by train_privileged.py / '
                         'train_pixels.py. Pass a negative value to disable '
                         'and use dedo\'s fixed 0.125 m criterion instead.')
extra = parser.parse_args()
sf = None if extra.success_factor < 0 else float(extra.success_factor)

# Build dedo args.
sys.argv = [
    'view_demo',
    '--env=HangProcCloth-v1',
    '--cam_resolution', str(extra.cam_resolution),
    '--num_envs=0',
    '--total_env_steps=0',
    '--seed', str(extra.seed),
]
args, _ = get_args_parser()
args_postprocess(args)
args.viz = extra.viz
args.debug = False
args.cam_viewmat = [9.0, -25.0, 45.0, 0.0, 0.5, 6.5]

os.makedirs(extra.logdir, exist_ok=True)

env = gym.make(args.env, args=args)
env = RetryResetEnv(env)
env = PrivilegedObsWrapper(env, obs_mode=extra.obs_mode, success_factor=sf)
env.seed(extra.seed)
print(f'[view_demo] success criterion: '
      f'{"adaptive (sf=" + str(sf) + " * hole_radius)" if sf is not None else "dedo fixed 0.125 m"}')

n_success = 0
for ep in range(extra.num_episodes):
    obs = env.reset()

    # Walk to underlying DeformEnv.
    underlying = env
    while hasattr(underlying, 'env'):
        underlying = underlying.env
        if isinstance(underlying, DeformEnv):
            break

    ctrl_freq = args.sim_freq / args.sim_steps_per_action
    preset_wp = build_hole_aware_waypoints(underlying)
    if preset_wp is None:
        print(f'[ep {ep+1}] no hole loop, skipping')
        continue

    _, vel_a = build_traj(underlying, preset_wp, 'a',
                          anchor_idx=0, ctrl_freq=ctrl_freq, robot=None)
    _, vel_b = build_traj(underlying, preset_wp, 'b',
                          anchor_idx=1, ctrl_freq=ctrl_freq, robot=None)
    traj = merge_traj(vel_a, vel_b)
    last = np.zeros_like(traj[0])

    vidwriter = None
    if extra.cam_resolution > 0:
        vidpath = os.path.join(
            extra.logdir, f'demo_ep{ep:02d}_seed{extra.seed}.mp4')
        vidwriter = cv2.VideoWriter(
            vidpath, cv2.VideoWriter_fourcc(*'mp4v'), 24,
            (extra.cam_resolution, extra.cam_resolution))
        print(f'[ep {ep+1}] writing {vidpath}')

    step, ep_rwd, ep_success = 0, 0.0, 0
    while True:
        act_unscaled = traj[step] if step < len(traj) else last
        normalized = np.clip(act_unscaled / DeformEnv.MAX_ACT_VEL, -1.0, 1.0)
        obs, rwd, done, info = env.step(normalized.astype(np.float32))
        ep_rwd += float(rwd)
        if 'is_success' in info:
            ep_success = max(ep_success, int(info['is_success']))
        if vidwriter is not None and extra.cam_resolution > 0:
            img = underlying.render(mode='rgb_array',
                                    width=extra.cam_resolution,
                                    height=extra.cam_resolution)
            vidwriter.write(img[..., ::-1])
        if done:
            break
        step += 1

    if vidwriter is not None:
        vidwriter.release()
    n_success += ep_success
    extra_info = ''
    if sf is not None and env.hole_radius is not None:
        extra_info = (f'  hole_r={env.hole_radius:.3f}m  '
                      f'thresh<{env.success_threshold_m:.3f}m')
    print(f'[ep {ep+1}] reward={ep_rwd:.2f}  success={ep_success}{extra_info}')

print(f'\nDone — {n_success}/{extra.num_episodes} demos succeeded '
      f'({100.0 * n_success / max(extra.num_episodes, 1):.1f}%).')
env.close()
