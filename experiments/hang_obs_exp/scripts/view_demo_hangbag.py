"""
View the bag-handle-aware scripted demo on HangBag-v1.

HangBag analogue of view_demo.py: per-episode the script reads the
live bag mesh, picks one handle loop (the "primary"), and uses
build_bag_handle_waypoints to plan a hover -> pass -> hold trajectory
that lands that handle on the hook. Both anchors are translated by
the same vector so the bag isn't torn.

Usage (from repo root):
  # Live pybullet GUI (recommended for debugging):
  python experiments/hang_obs_exp/scripts/view_demo_hangbag.py --viz

  # Save mp4(s) to logs/hang_obs_exp/demo_view_hangbag/:
  python experiments/hang_obs_exp/scripts/view_demo_hangbag.py --num_episodes 5

  # Both:
  python experiments/hang_obs_exp/scripts/view_demo_hangbag.py --viz --num_episodes 3
"""
import sys, os, argparse
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT))

import numpy as np
import gym
import cv2

import dedo
from dedo.utils.args import get_args_parser, args_postprocess
from dedo.envs.deform_env import DeformEnv
from dedo.demo_preset import build_traj, merge_traj

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _helpers import RetryResetEnv, build_bag_handle_waypoints  # noqa: E402


parser = argparse.ArgumentParser()
parser.add_argument('--num_episodes', type=int, default=3)
parser.add_argument('--viz', action='store_true',
                    help='Open the live pybullet GUI window')
parser.add_argument('--cam_resolution', type=int, default=400,
                    help='mp4 render size. 0 to skip writing mp4.')
parser.add_argument('--logdir', type=str,
                    default=str(REPO_ROOT / 'logs' / 'hang_obs_exp' /
                                'demo_view_hangbag'))
parser.add_argument('--seed', type=int, default=42)
parser.add_argument('--env', type=str, default='HangBag-v1',
                    help='HangBag-v0 picks a random tote mesh per reset; '
                         'HangBag-v1/v2/v3 pin a specific mesh.')
parser.add_argument('--primary_loop_idx', type=int, default=0,
                    help='Which entry of deform_true_loop_vertices is '
                         'the handle that lands on the hook. HangBag '
                         'totes have two handle loops (indices 0 and 1).')
extra = parser.parse_args()

sys.argv = [
    'view_demo_hangbag',
    f'--env={extra.env}',
    '--cam_resolution', '0',
    '--num_envs=0',
    '--total_env_steps=0',
    '--seed', str(extra.seed),
]
args, _ = get_args_parser()
args_postprocess(args)
args.viz = extra.viz
args.debug = False

os.makedirs(extra.logdir, exist_ok=True)

env = gym.make(args.env, args=args)
env = RetryResetEnv(env)
env.seed(extra.seed)

n_success = 0
peaks = []
for ep in range(extra.num_episodes):
    obs = env.reset()

    underlying = env
    while hasattr(underlying, 'env'):
        underlying = underlying.env
        if isinstance(underlying, DeformEnv):
            break

    ctrl_freq = args.sim_freq / args.sim_steps_per_action
    preset_wp = build_bag_handle_waypoints(
        underlying, primary_loop_idx=extra.primary_loop_idx)
    if preset_wp is None:
        print(f'[ep {ep+1}] no loop info available, skipping')
        continue

    _, vel_a = build_traj(underlying, preset_wp, 'a',
                          anchor_idx=0, ctrl_freq=ctrl_freq, robot=None)
    _, vel_b = build_traj(underlying, preset_wp, 'b',
                          anchor_idx=1, ctrl_freq=ctrl_freq, robot=None)
    traj = merge_traj(vel_a, vel_b)
    last = np.zeros_like(traj[0])

    ep_peak = float(np.abs(traj).max())
    peaks.append(ep_peak)

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
    print(f'[ep {ep+1}] reward={ep_rwd:.2f}  success={ep_success}  '
          f'peak|vel|={ep_peak:.3f} m/s')

print(f'\nDone — {n_success}/{extra.num_episodes} demos succeeded '
      f'({100.0 * n_success / max(extra.num_episodes, 1):.1f}%).')
if peaks:
    _global_peak = max(peaks)
    _suggested = float(np.ceil(_global_peak * 1.2 * 10) / 10)
    print(f'\n[velocity-audit]')
    print(f'  Per-episode peak |vel|: '
          f'{[f"{p:.3f}" for p in peaks]}')
    print(f'  Global peak across {len(peaks)} episodes: '
          f'{_global_peak:.3f} m/s')
    print(f'  Recommended --max_act_vel = {_suggested:.1f}  '
          f'(global peak * 1.2 safety, rounded up to 0.1)')
env.close()
