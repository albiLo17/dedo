"""
Replay a recorded demo (.pkl from record_demo.py or record_demo_pcd.py)
either in the live pybullet GUI or as an mp4.

Usage examples (from repo root):

  # Live pybullet window:
  python experiments/hang_obs_exp/scripts/replay_demo.py \
      --pkl logs/hang_obs_exp/manual_demos/demo_000.pkl --viz

  # Save mp4 (no GUI):
  python experiments/hang_obs_exp/scripts/replay_demo.py \
      --pkl logs/hang_obs_exp/manual_demos/demo_000.pkl \
      --logdir logs/hang_obs_exp/replays

  # Both:
  python experiments/hang_obs_exp/scripts/replay_demo.py \
      --pkl logs/hang_obs_exp/manual_demos/demo_000.pkl --viz \
      --logdir logs/hang_obs_exp/replays

Caveat: cloth dynamics are non-deterministic, and procedural cloth gen
re-rolls per reset, so the replayed trajectory will resemble the
original recording but not match it exactly. The hanger / cloth start
pose is similar; the in-flight cloth shape will diverge subtly. Use
this for a qualitative "what did the demo do" view, not for byte-exact
reproduction.
"""
import sys, os, argparse, pickle
from copy import deepcopy
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT))

import numpy as np
import gym
import cv2

import dedo
from dedo.utils.args import get_args_parser, args_postprocess
from dedo.envs.deform_env import DeformEnv

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _helpers import RetryResetEnv  # noqa: E402


parser = argparse.ArgumentParser()
parser.add_argument('--pkl', type=str, required=True,
                    help='Path to demo_NNN.pkl')
parser.add_argument('--viz', action='store_true',
                    help='Open the live pybullet GUI window')
parser.add_argument('--logdir', type=str, default=None,
                    help='If set, save mp4 to <logdir>/<pkl_stem>.mp4')
parser.add_argument('--cam_resolution', type=int, default=400)
parser.add_argument('--seed', type=int, default=42)
parser.add_argument('--fps_video', type=int, default=24)
extra = parser.parse_args()

if not extra.viz and extra.logdir is None:
    print('ERROR: pick at least one of --viz or --logdir.')
    sys.exit(1)

# Load demo.
with open(extra.pkl, 'rb') as f:
    d = pickle.load(f)
acts = np.asarray(d['acts'], dtype=np.float32)
recorded_success = int(d.get('success', 0))
recorded_reward = float(d.get('reward', 0.0))
print(f'[replay] loaded {extra.pkl}')
print(f'         {len(acts)} steps   recorded reward={recorded_reward:.2f}   '
      f'success={recorded_success}')

# Build dedo args.
sys.argv = [
    'replay_demo',
    '--env=HangProcCloth-v1',
    f'--cam_resolution={extra.cam_resolution}',
    '--num_envs=0',
    '--total_env_steps=0',
    '--seed', str(extra.seed),
] + (['--viz'] if extra.viz else [])
args, _ = get_args_parser()
args_postprocess(args)
args.viz = extra.viz
args.debug = False
args.cam_viewmat = [9.0, -25.0, 45.0, 0.0, 0.5, 6.5]

env = gym.make(args.env, args=args)
env = RetryResetEnv(env)
env.seed(extra.seed)

# Reach the underlying DeformEnv for render().
underlying = env
while hasattr(underlying, 'env'):
    underlying = underlying.env
    if isinstance(underlying, DeformEnv):
        break

vidwriter = None
if extra.logdir is not None:
    os.makedirs(extra.logdir, exist_ok=True)
    stem = Path(extra.pkl).stem
    out_path = os.path.join(extra.logdir, f'{stem}.mp4')
    vidwriter = cv2.VideoWriter(
        out_path, cv2.VideoWriter_fourcc(*'mp4v'),
        extra.fps_video, (extra.cam_resolution, extra.cam_resolution))
    print(f'[replay] writing video to {out_path}')

obs = env.reset()
ep_rwd = 0.0
ep_success = 0
for step, a in enumerate(acts):
    obs, rwd, done, info = env.step(a.astype(np.float32))
    ep_rwd += float(rwd)
    if 'is_success' in info:
        ep_success = max(ep_success, int(info['is_success']))
    if vidwriter is not None:
        img = underlying.render(mode='rgb_array',
                                width=extra.cam_resolution,
                                height=extra.cam_resolution)
        vidwriter.write(img[..., ::-1])
    if done:
        print(f'[replay] env done at step {step+1}/{len(acts)}  '
              f'rwd={ep_rwd:.2f}  success={ep_success}')
        break

if vidwriter is not None:
    vidwriter.release()
    print(f'[replay] saved {out_path}')

env.close()
print('Done.')
