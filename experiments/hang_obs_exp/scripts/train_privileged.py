"""
Train PPO on HangProcCloth-v1 with privileged ground-truth cloth observations.

Three conditions launched from this script (select via --obs_mode):

  hole_centroid  — 18-dim: gripper + hole centroid + hanger goal   [fastest]
  hole_vertices  — 132-dim: gripper + all hole-boundary vertices
  full_mesh      — ~762-dim: gripper + all cloth vertex positions

Usage (from repo root):
  python experiments/hang_obs_exp/scripts/train_privileged.py --obs_mode hole_centroid
  python experiments/hang_obs_exp/scripts/train_privileged.py --obs_mode hole_vertices
  python experiments/hang_obs_exp/scripts/train_privileged.py --obs_mode full_mesh

Outputs saved under logs/hang_obs_exp/<obs_mode>/
"""

import sys, os, argparse, pickle
from copy import deepcopy
from pathlib import Path

# Repo root is three levels up from this script (scripts/ -> hang_obs_exp/ -> experiments/ -> repo)
REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT))

import numpy as np
import torch
import gym
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import DummyVecEnv

import dedo  # registers gym envs
from dedo.utils.args import get_args_parser, args_postprocess
from dedo.utils.train_utils import init_train
from dedo.utils.rl_sb3_utils import CustomCallback

from experiments.hang_obs_exp.envs.privileged_env import PrivilegedObsWrapper


# ---------------------------------------------------------------------------
# Parse our extra arg on top of dedo's args
# ---------------------------------------------------------------------------
parser = argparse.ArgumentParser(add_help=False)
parser.add_argument('--obs_mode', type=str, default='hole_centroid',
                    choices=['hole_centroid', 'hole_vertices', 'full_mesh'])
parser.add_argument('--total_env_steps', type=int, default=2_000_000)
parser.add_argument('--num_envs', type=int, default=4)
parser.add_argument('--lr', type=float, default=3e-4)
parser.add_argument('--seed', type=int, default=42)
parser.add_argument('--logdir_root', type=str,
                    default=str(REPO_ROOT / 'logs' / 'hang_obs_exp'))
extra_args, remaining = parser.parse_known_args()

# Build dedo args with cam_resolution=0 (wrapper takes care of geometry obs)
sys.argv = [
    'train_privileged',
    '--env=HangProcCloth-v1',
    '--cam_resolution=0',
    '--num_envs=0',
    '--total_env_steps=0',
    '--log_save_interval=50',
    '--seed', str(extra_args.seed),
]
dedo_args, _ = get_args_parser()
args_postprocess(dedo_args)
dedo_args.rl_algo = 'PPO'
dedo_args.seed = extra_args.seed
dedo_args.total_env_steps = extra_args.total_env_steps
dedo_args.num_envs = extra_args.num_envs
dedo_args.lr = extra_args.lr
dedo_args.debug = False
dedo_args.viz = False
dedo_args.log_save_interval = 50

obs_mode = extra_args.obs_mode
logdir_base = os.path.join(extra_args.logdir_root, obs_mode)
os.makedirs(logdir_base, exist_ok=True)
dedo_args.logdir = logdir_base

np.random.seed(dedo_args.seed)
torch.manual_seed(dedo_args.seed)


# ---------------------------------------------------------------------------
# Factory: wrapped env (DummyVecEnv — all envs run in-process)
# ---------------------------------------------------------------------------
def make_wrapped_env(args, obs_mode_str):
    def _init():
        _args = deepcopy(args)
        _args.debug = False
        _args.viz = False
        env = gym.make(_args.env, args=_args)
        env = PrivilegedObsWrapper(env, obs_mode=obs_mode_str)
        return env
    return _init


# ---------------------------------------------------------------------------
# Build vec envs
# ---------------------------------------------------------------------------
n_envs = extra_args.num_envs
env_fns = [make_wrapped_env(dedo_args, obs_mode) for _ in range(n_envs)]
vec_env = DummyVecEnv(env_fns)
vec_env.seed(dedo_args.seed)

eval_env = gym.make(dedo_args.env, args=deepcopy(dedo_args))
eval_env = PrivilegedObsWrapper(eval_env, obs_mode=obs_mode)
eval_env.seed(dedo_args.seed)

obs_shape = vec_env.observation_space.shape
print(f'\n{"="*60}')
print(f'Condition: privileged/{obs_mode}')
print(f'Obs shape: {obs_shape}  Action shape: {vec_env.action_space.shape}')
print(f'Steps: {extra_args.total_env_steps:,}  Envs: {n_envs}  LR: {extra_args.lr}')
print(f'Logdir: {logdir_base}')
print(f'{"="*60}\n')

# ---------------------------------------------------------------------------
# Init run dir and train
# ---------------------------------------------------------------------------
dedo_args.logdir, dedo_args.device = init_train('PPO', dedo_args)

rl_kwargs = {
    'learning_rate': dedo_args.lr,
    'device': dedo_args.device,
    'tensorboard_log': dedo_args.logdir,
    'verbose': 1,
}
agent = PPO('MlpPolicy', vec_env, **rl_kwargs)

num_steps_between_save = dedo_args.log_save_interval * 10 * 10
cb = CustomCallback(eval_env, dedo_args.logdir, n_envs, dedo_args,
                    num_steps_between_save=num_steps_between_save,
                    viz=False, debug=False)

print('Start privileged RL training ...')
agent.learn(total_timesteps=extra_args.total_env_steps, callback=cb)

ckpt_path = os.path.join(dedo_args.logdir, 'agent.zip')
agent.save(ckpt_path)
pickle.dump(dedo_args, open(os.path.join(dedo_args.logdir, 'args.pkl'), 'wb'))
print(f'\nDone. Checkpoint: {ckpt_path}')
vec_env.close()
