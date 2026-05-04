"""
Train SAC on HangProcCloth-v1 with PIXEL (RGB) observations.

Counterpart to train_pixels.py (PPO) — same wrapper, same camera, same
adaptive success / reward shaping. SAC is off-policy with a replay
buffer, typically more sample-efficient than PPO on continuous-control
tasks. The replay buffer is image-aware: defaults to 100k transitions
(vs PPO's 1M) since each pixel obs is ~12 KB at 64x64x3, and 100k of
those is already ~1.2 GB.

Default obs is Dict({image, grip}) → SB3 'MultiInputPolicy'. Pass
--no_grip for a pure 'CnnPolicy' baseline.

Usage (from repo root):
  python experiments/hang_obs_exp/scripts/train_pixels_sac.py --use_wandb

  # Tighter buffer (less memory) — useful when iterating on a laptop:
  python experiments/hang_obs_exp/scripts/train_pixels_sac.py \
      --buffer_size 50000 --use_wandb

Outputs saved under logs/hang_obs_exp/<grip_str>_<resolution>_sac/

NOTE on SAC + images: the replay buffer dominates memory. 100k 64x64x3
uint8 transitions ≈ 1.2 GB just for the obs side, plus next_obs ≈
another 1.2 GB. Pass --buffer_size lower if you OOM. Setting
--optimize_memory_usage stores only one obs copy per transition (saves
~half the memory) but is incompatible with HER and a few other features.
"""

import sys, os, argparse, pickle
from copy import deepcopy
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT))

import numpy as np
import torch
import gym
from stable_baselines3 import SAC
from stable_baselines3.common.env_util import DummyVecEnv
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import VecNormalize, VecTransposeImage


sys.path.insert(0, str(Path(__file__).resolve().parent))
from _helpers import RetryResetEnv, build_hole_aware_waypoints  # noqa: E402
from _video_callback import HangVideoCallback  # noqa: E402
from _reward_diagnostics import (  # noqa: E402
    RewardDiagnosticsCallback, dump_run_config, make_final_eval_collector,
    log_final_eval_metrics)

import dedo  # registers gym envs
from dedo.utils.args import get_args_parser, args_postprocess
from dedo.utils.train_utils import init_train
from stable_baselines3.common.callbacks import CallbackList
from experiments.hang_obs_exp.envs.pixel_env import PixelObsWrapper


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
parser = argparse.ArgumentParser(add_help=False)
parser.add_argument('--total_env_steps', type=int, default=1_000_000,
                    help='SAC is more sample-efficient; 1M is usually enough.')
parser.add_argument('--num_envs', type=int, default=1,
                    help='SAC: 1 env is canonical (off-policy gains '
                         'diminish with more parallel envs).')
parser.add_argument('--lr', type=float, default=3e-4)
parser.add_argument('--seed', type=int, default=42)
parser.add_argument('--logdir_root', type=str,
                    default=str(REPO_ROOT / 'logs' / 'hang_obs_exp'))
parser.add_argument('--use_wandb', action='store_true')
parser.add_argument('--n_final_eval_episodes', type=int, default=20)
parser.add_argument('--log_save_interval', type=int, default=50,
                    help='Controls checkpoint / eval / video cadence. '
                         'Checkpoint every (log_save_interval * 10 * 50) '
                         'env steps; eval every 2nd checkpoint; video '
                         'every 4th checkpoint. Lower = more frequent. '
                         'Default 50 → checkpoint @ 25k / eval @ 50k / '
                         'video @ 100k.')
# Camera config (matches train_pixels.py).
parser.add_argument('--cam_resolution', type=int, default=64)
parser.add_argument('--cam_viewmat', type=float, nargs=6,
                    default=[9.0, -25.0, 45.0, 0.0, 0.5, 6.5])
parser.add_argument('--no_grip', action='store_true')
parser.add_argument('--max_episode_len', type=int, default=200)
# SAC-specific knobs. Buffer is image-aware (smaller than the privileged
# default) to keep RAM usage tractable.
parser.add_argument('--buffer_size', type=int, default=100_000)
parser.add_argument('--learning_starts', type=int, default=1_000)
parser.add_argument('--batch_size', type=int, default=256)
parser.add_argument('--tau', type=float, default=0.005)
parser.add_argument('--gamma', type=float, default=0.99)
parser.add_argument('--train_freq', type=int, default=1,
                    help='Train every N env steps.')
parser.add_argument('--gradient_steps', type=int, default=1,
                    help='Gradient updates per train trigger.')
parser.add_argument('--ent_coef', type=str, default='auto',
                    help='SAC entropy coef. "auto" = auto-tuned, or a '
                         'float string like "0.1".')
parser.add_argument('--optimize_memory_usage', action='store_true',
                    help='Store one obs per transition instead of (obs, '
                         'next_obs). ~halves replay buffer memory but '
                         'incompatible with HER and a few other features.')
# Adaptive success + shaping (same defaults as the other scripts).
parser.add_argument('--success_factor', type=float, default=1.2)
parser.add_argument('--no_adaptive_success', action='store_true')
parser.add_argument('--success_bonus', type=float, default=200.0)
parser.add_argument('--fail_penalty', type=float, default=0.0)
parser.add_argument('--vel_penalty', type=float, default=0.0)
# BC pretrain.
parser.add_argument('--bc_episodes', type=int, default=0,
                    help='Target number of scripted demos to KEEP for '
                         'BC pretrain. 0 = skip. The collector retries '
                         'until it has this many demos that pass the '
                         'keep criterion (any if --bc_demos_only_success '
                         'off; only is_success=1 if on), capped at '
                         '~3x attempts. Dataset size is therefore '
                         'deterministic across seeds.')
parser.add_argument('--bc_epochs', type=int, default=20)
parser.add_argument('--bc_lr', type=float, default=1e-3)
parser.add_argument('--bc_demo_path', type=str, default=None,
                    help='Directory of pixel demo_NNN.pkl files. If '
                         'set, skip scripted collection and BC on the '
                         'loaded demos instead. Demos must match the '
                         'current --cam_resolution and --no_grip mode.')
parser.add_argument('--no_save_scripted_demos', action='store_true',
                    help='Skip persisting scripted BC demos to '
                         '<logdir>/scripted_demos/. Default behaviour '
                         'saves them so they can be reused via '
                         '--bc_demo_path; opt out for big sweeps to '
                         'avoid ~100 MB / run of disk.')
parser.add_argument('--bc_demos_only_success', action='store_true',
                    help='Drop scripted demos whose terminal '
                         'is_success=0 from the BC dataset. The hole-'
                         'aware waypoint controller succeeds on most '
                         'but not all procedural cloth shapes; '
                         'filtering yields a cleaner BC dataset at the '
                         'cost of fewer (obs, act) pairs. Strongly '
                         'recommended unless --bc_episodes is small '
                         '(<20) and the scripted success rate is low.')
parser.add_argument('--cpu', action='store_true')
extra_args, remaining = parser.parse_known_args()

if extra_args.no_adaptive_success:
    extra_args.success_factor = None

# Single source of truth: `extra_args.success_factor` flows into the
# training env, the eval env, AND the BC scripted-demo collection env
# below — so `info['is_success']` is identical across all three. When
# adaptive success is off (success_factor=None), the wrapper's override
# block is skipped and `info['is_success']` falls through to dedo's
# default (|last_rwd| < SUCCESS_REWARD_THRESHOLD, ~0.125 m). Print the
# active criterion at startup so logs make this auditable for any run.
_sf_descr = (
    f'adaptive (sf={extra_args.success_factor} * hole_radius)'
    if extra_args.success_factor is not None
    else 'dedo default (|last_rwd| < SUCCESS_REWARD_THRESHOLD, ~0.125 m)')
print(f'[success-criterion] training = eval = BC scripted: {_sf_descr}')

_grip_str = 'pixgrip' if not extra_args.no_grip else 'pixonly'
run_subdir = f'{_grip_str}_{extra_args.cam_resolution}_sac'


# ---------------------------------------------------------------------------
# Build dedo args.
# ---------------------------------------------------------------------------
sys.argv = [
    'train_pixels_sac',
    '--env=HangProcCloth-v1',
    f'--cam_resolution={extra_args.cam_resolution}',
    '--uint8_pixels',
    '--num_envs=0',
    '--total_env_steps=0',
    f'--log_save_interval={extra_args.log_save_interval}',
    '--seed', str(extra_args.seed),
    '--max_episode_len', str(extra_args.max_episode_len),
    '--cam_viewmat',
    str(extra_args.cam_viewmat[0]), str(extra_args.cam_viewmat[1]),
    str(extra_args.cam_viewmat[2]), str(extra_args.cam_viewmat[3]),
    str(extra_args.cam_viewmat[4]), str(extra_args.cam_viewmat[5]),
]
dedo_args, _ = get_args_parser()
args_postprocess(dedo_args)
dedo_args.rl_algo = 'SAC'
dedo_args.seed = extra_args.seed
dedo_args.use_wandb = extra_args.use_wandb
dedo_args.total_env_steps = extra_args.total_env_steps
dedo_args.num_envs = extra_args.num_envs
dedo_args.lr = extra_args.lr
dedo_args.debug = False
dedo_args.viz = False
dedo_args.log_save_interval = extra_args.log_save_interval
dedo_args.disable_logging_video = False

logdir_base = os.path.join(extra_args.logdir_root, run_subdir)
os.makedirs(logdir_base, exist_ok=True)
dedo_args.logdir = logdir_base

np.random.seed(dedo_args.seed)
torch.manual_seed(dedo_args.seed)


# ---------------------------------------------------------------------------
# Env factory.
# ---------------------------------------------------------------------------
def make_wrapped_env(args, monitor_dir=None):
    def _init():
        _args = deepcopy(args)
        _args.debug = False
        _args.viz = False
        env = gym.make(_args.env, args=_args)
        env = RetryResetEnv(env)
        env = PixelObsWrapper(
            env,
            cam_resolution=extra_args.cam_resolution,
            include_grip=not extra_args.no_grip,
            success_factor=extra_args.success_factor,
            success_bonus=extra_args.success_bonus,
            fail_penalty=extra_args.fail_penalty,
            vel_penalty=extra_args.vel_penalty,
        )
        env = Monitor(env, filename=monitor_dir)
        return env
    return _init


# ---------------------------------------------------------------------------
# Build vec envs.  norm_obs=False is required for image obs (uint8); we
# still normalize rewards so SAC's critic targets stay in a sane range.
# ---------------------------------------------------------------------------
n_envs = extra_args.num_envs
vec_env = DummyVecEnv([make_wrapped_env(dedo_args) for _ in range(n_envs)])
vec_env.seed(dedo_args.seed)
# IMPORTANT: VecTransposeImage MUST come before VecNormalize so SAC's
# ReplayBuffer pre-allocates against the channel-first obs shape SB3's
# CnnPolicy / MultiInputPolicy expect. Without this the buffer is
# allocated for HWC but step() returns CHW (or vice versa) → broadcast
# error on the first transition push.
vec_env = VecTransposeImage(vec_env)
vec_env = VecNormalize(vec_env, norm_obs=False, norm_reward=True,
                       clip_obs=10.0)

# Eval env (single, non-vec).
eval_args = deepcopy(dedo_args)
eval_env_raw = gym.make(eval_args.env, args=eval_args)
eval_env_raw = RetryResetEnv(eval_env_raw)
eval_env_raw = PixelObsWrapper(
    eval_env_raw,
    cam_resolution=extra_args.cam_resolution,
    include_grip=not extra_args.no_grip,
    success_factor=extra_args.success_factor,
    success_bonus=extra_args.success_bonus,
    fail_penalty=extra_args.fail_penalty,
    vel_penalty=extra_args.vel_penalty,
)
eval_env_raw = Monitor(eval_env_raw)
eval_env_raw.seed(dedo_args.seed)
eval_env = eval_env_raw

print(f'\n{"="*60}')
print(f'Condition: pixels  ALGO=SAC  '
      f'(grip={"yes" if not extra_args.no_grip else "no"})')
print(f'Image: {extra_args.cam_resolution}x{extra_args.cam_resolution}x3 '
      f'uint8  cam_viewmat={extra_args.cam_viewmat}')
print(f'Steps: {extra_args.total_env_steps:,}  Envs: {n_envs}  '
      f'LR: {extra_args.lr}')
print(f'Buffer: {extra_args.buffer_size:,}  '
      f'learning_starts: {extra_args.learning_starts}  '
      f'batch: {extra_args.batch_size}')
print(f'Logdir: {logdir_base}')
print(f'{"="*60}\n')


# ---------------------------------------------------------------------------
# Init wandb / logdir.
# ---------------------------------------------------------------------------
dedo_args.logdir, dedo_args.device = init_train('SAC', dedo_args)
if extra_args.cpu:
    dedo_args.device = 'cpu'

if dedo_args.use_wandb:
    import wandb
    if wandb.run is not None:
        sf = extra_args.success_factor
        sb = extra_args.success_bonus
        fp = extra_args.fail_penalty
        vp = extra_args.vel_penalty
        sf_tag = f'_sf{sf:g}' if sf is not None else '_sf_default'
        sb_tag = f'_sb{sb:g}' if sb else ''
        fp_tag = f'_fp{fp:g}' if fp else ''
        vp_tag = f'_vp{vp:g}' if vp else ''
        bc_tag = '_bc' if extra_args.bc_episodes > 0 else ''
        grip_tag = '_grip' if not extra_args.no_grip else '_pix'
        wandb.run.name = (f'{wandb.run.name}_pixels{extra_args.cam_resolution}'
                          f'{grip_tag}_sac'
                          f'{sf_tag}{sb_tag}{fp_tag}{vp_tag}{bc_tag}')
        wandb.run.tags = list(wandb.run.tags or []) + [
            'algo=sac',
            'obs=pixels',
            f'include_grip={"yes" if not extra_args.no_grip else "no"}',
            f'cam_resolution={extra_args.cam_resolution}',
            f'success_factor={sf if sf is not None else "default"}',
            f'success_bonus={sb}',
            f'fail_penalty={fp}',
            f'vel_penalty={vp}',
            f'bc={"yes" if bc_tag else "no"}',
        ]


# ---------------------------------------------------------------------------
# SAC. CNN feature extractor is shared between actor and critic by SB3
# default for image obs (memory-efficient, slightly worse asymptote than
# separate extractors but reasonable for this use).
# ---------------------------------------------------------------------------
try:
    ent_coef_arg = float(extra_args.ent_coef)
except ValueError:
    ent_coef_arg = extra_args.ent_coef

policy_name = 'MultiInputPolicy' if not extra_args.no_grip else 'CnnPolicy'
policy_kwargs = dict(net_arch=[256, 256])
rl_kwargs = dict(
    learning_rate=dedo_args.lr,
    device=dedo_args.device,
    tensorboard_log=dedo_args.logdir,
    verbose=1,
    policy_kwargs=policy_kwargs,
    buffer_size=extra_args.buffer_size,
    learning_starts=extra_args.learning_starts,
    batch_size=extra_args.batch_size,
    tau=extra_args.tau,
    gamma=extra_args.gamma,
    train_freq=extra_args.train_freq,
    gradient_steps=extra_args.gradient_steps,
    ent_coef=ent_coef_arg,
    optimize_memory_usage=extra_args.optimize_memory_usage,
)
agent = SAC(policy_name, vec_env, **rl_kwargs)

num_steps_between_save = dedo_args.log_save_interval * 10 * 50
_sf = extra_args.success_factor
_sf_str = f'sf{_sf:g}' if _sf is not None else 'sf_off'
_video_basename = (f'eval_pixels{extra_args.cam_resolution}'
                   f'{"_grip" if not extra_args.no_grip else ""}_sac_{_sf_str}')
if extra_args.success_bonus:
    _video_basename += f'_sb{extra_args.success_bonus:g}'
if extra_args.fail_penalty:
    _video_basename += f'_fp{extra_args.fail_penalty:g}'
if extra_args.vel_penalty:
    _video_basename += f'_vp{extra_args.vel_penalty:g}'
_video_basename += f'_seed{extra_args.seed}'

video_cb = HangVideoCallback(eval_env, dedo_args.logdir, n_envs, dedo_args,
                             num_steps_between_save=num_steps_between_save,
                             viz=False, debug=False,
                             video_basename=_video_basename)
diag_cb = RewardDiagnosticsCallback(window=100)
cb = CallbackList([video_cb, diag_cb])

dump_run_config(extra_args, dedo_args, dedo_args.logdir,
                use_wandb=dedo_args.use_wandb)


# ---------------------------------------------------------------------------
# Optional BC pretrain on scripted hole-aware demos. Targets SAC's actor
# only (deterministic_action = tanh(mu(latent_pi)) — same pattern as
# train_privileged_sac.py).
# ---------------------------------------------------------------------------
def _collect_pixel_demos(args, num_episodes, only_success=False,
                         max_attempt_factor=3, save_dir=None):
    """Roll out scripted hole-aware waypoints in the pixel obs env.

    `num_episodes` is the **target number of demos kept**, not the
    number of attempts. The collector retries until that many demos
    pass the keep criterion (any rollout if `only_success=False`; only
    those with `is_success=1` if `only_success=True`), capped at
    `num_episodes * max_attempt_factor` attempts. Dataset size is
    therefore deterministic across seeds.

    `save_dir`: if non-None, persist each kept demo as
    `<save_dir>/demo_NNN.pkl` (one episode per pkl) so it can be
    reloaded later via `--bc_demo_path`. Same schema as train_pixels.py
    (PPO) so demos can be shared across PPO and SAC runs."""
    from dedo.demo_preset import build_traj, merge_traj
    from dedo.envs.deform_env import DeformEnv

    raw = gym.make(args.env, args=deepcopy(args))
    raw = RetryResetEnv(raw)
    raw = PixelObsWrapper(
        raw,
        cam_resolution=extra_args.cam_resolution,
        include_grip=not extra_args.no_grip,
        success_factor=extra_args.success_factor,
        success_bonus=0.0, fail_penalty=0.0, vel_penalty=0.0,
    )
    raw.seed(args.seed + 1000)

    if save_dir is not None:
        os.makedirs(save_dir, exist_ok=True)

    obs_buf, act_buf = [], []
    succeeded = 0
    n_kept, n_dropped = 0, 0
    target_kept = num_episodes
    max_attempts = max(num_episodes * max_attempt_factor, num_episodes + 5)
    attempts = 0
    while n_kept < target_kept and attempts < max_attempts:
        attempts += 1
        obs = raw.reset()
        underlying = raw
        while hasattr(underlying, 'env'):
            underlying = underlying.env
            if isinstance(underlying, DeformEnv):
                break
        ctrl_freq = args.sim_freq / args.sim_steps_per_action

        preset_wp = build_hole_aware_waypoints(underlying)
        if preset_wp is None:
            print(f'[BC] attempt {attempts} (kept {n_kept}/{target_kept}): '
                  f'no hole loop, retrying')
            continue
        try:
            _, vel_a = build_traj(underlying, preset_wp, 'a',
                                  anchor_idx=0, ctrl_freq=ctrl_freq, robot=None)
            _, vel_b = build_traj(underlying, preset_wp, 'b',
                                  anchor_idx=1, ctrl_freq=ctrl_freq, robot=None)
            traj = merge_traj(vel_a, vel_b)
        except Exception as e:
            print(f'[BC] attempt {attempts} (kept {n_kept}/{target_kept}): '
                  f'build_traj failed ({e!r}), retrying')
            continue

        last = np.zeros_like(traj[0])
        ep_obs, ep_act = [], []
        ep_rwd, ep_succ = 0.0, 0
        step = 0
        while True:
            act_unscaled = traj[step] if step < len(traj) else last
            normalized = np.clip(
                act_unscaled / DeformEnv.MAX_ACT_VEL, -1.0, 1.0
            ).astype(np.float32)
            ep_obs.append(obs)
            ep_act.append(normalized)
            obs, rwd, done, info = raw.step(normalized)
            ep_rwd += float(rwd)
            if 'is_success' in info:
                ep_succ = max(ep_succ, int(info['is_success']))
            if done:
                break
            step += 1
        if only_success and not ep_succ:
            n_dropped += 1
            print(f'[BC] attempt {attempts} (kept {n_kept}/{target_kept})  '
                  f'len={len(ep_obs)}  rwd={ep_rwd:.1f}  success=0  '
                  f'(dropped, --bc_demos_only_success)')
            continue
        obs_buf.extend(ep_obs)
        act_buf.extend(ep_act)
        succeeded += ep_succ
        n_kept += 1

        # Persist this episode so it can be reloaded via --bc_demo_path.
        # Same payload schema as the PPO pixel collector.
        if save_dir is not None:
            demo_idx = n_kept - 1
            pkl_path = os.path.join(save_dir, f'demo_{demo_idx:03d}.pkl')
            if isinstance(ep_obs[0], dict):
                ep_obs_stacked = {
                    k: np.stack([o[k] for o in ep_obs], axis=0)
                    for k in ep_obs[0].keys()}
            else:
                ep_obs_stacked = np.stack(ep_obs, axis=0)
            payload = {
                'obs': ep_obs_stacked,
                'acts': np.asarray(ep_act, dtype=np.float32),
                'reward': float(ep_rwd),
                'success': int(ep_succ),
                'obs_type': 'pixels',
                'cam_resolution': extra_args.cam_resolution,
                'include_grip': bool(not extra_args.no_grip),
                'success_factor': extra_args.success_factor,
                'len': len(ep_act),
                'source': 'scripted',
            }
            with open(pkl_path, 'wb') as f:
                pickle.dump(payload, f)

        print(f'[BC] attempt {attempts} (kept {n_kept}/{target_kept})  '
              f'len={len(ep_obs)}  rwd={ep_rwd:.1f}  success={ep_succ}')

    raw.close()
    if n_kept < target_kept:
        print(f'[BC] WARNING: only collected {n_kept}/{target_kept} demos '
              f'after {attempts} attempts (cap={max_attempts}). Either '
              f'increase --bc_episodes, raise the attempt cap, or lower '
              f'--success_factor.')
    elif only_success:
        print(f'[BC] kept {n_kept}/{target_kept} successful demos in '
              f'{attempts} attempts ({n_dropped} dropped, '
              f'~{n_kept/max(attempts,1):.0%} scripted success rate)')

    if not obs_buf:
        return None, np.zeros((0, 0), dtype=np.float32), 0
    if isinstance(obs_buf[0], dict):
        stacked = {k: np.stack([o[k] for o in obs_buf], axis=0)
                   for k in obs_buf[0].keys()}
    else:
        stacked = np.stack(obs_buf, axis=0)
    return (stacked,
            np.asarray(act_buf, dtype=np.float32),
            succeeded)


def _load_manual_pixel_demos(demo_dir, cam_resolution, include_grip,
                             only_success=False):
    """Load per-episode pixel demos written by `_collect_pixel_demos`.
    Mirrors train_pixels.py's loader so demos are interchangeable
    between PPO and SAC pixel runs. See the PPO version for full
    docstring."""
    obs_buf, act_buf = [], []
    n_files, n_success_demos = 0, 0
    sf_train = extra_args.success_factor
    sf_mismatches, sf_unknown = 0, 0
    for fname in sorted(os.listdir(demo_dir)):
        if not (fname.startswith('demo_') and fname.endswith('.pkl')):
            continue
        path = os.path.join(demo_dir, fname)
        with open(path, 'rb') as f:
            d = pickle.load(f)
        if d.get('obs_type') != 'pixels':
            print(f'[BC] {fname}: obs_type={d.get("obs_type")!r} '
                  f'not pixels, skipping')
            continue
        if d.get('cam_resolution') != cam_resolution:
            print(f'[BC] {fname}: cam_resolution='
                  f'{d.get("cam_resolution")} != {cam_resolution}, '
                  f'skipping')
            continue
        if bool(d.get('include_grip')) != bool(include_grip):
            print(f'[BC] {fname}: include_grip='
                  f'{d.get("include_grip")} != {include_grip}, skipping')
            continue

        if 'success_factor' in d:
            if d['success_factor'] != sf_train:
                sf_mismatches += 1
        else:
            sf_unknown += 1

        if only_success and not d.get('success', 0):
            print(f'[BC] {fname}: success=0, skipping')
            continue

        n_files += 1
        n_success_demos += int(d.get('success', 0))
        obs_buf.append(d['obs'])
        act_buf.append(d['acts'])
        print(f'[BC] loaded {fname}  len={d.get("len", len(d["acts"]))}  '
              f'rwd={d.get("reward", 0):.2f}  '
              f'success={d.get("success", 0)}')

    if sf_mismatches > 0 or sf_unknown > 0:
        print(f'[BC] WARNING: {sf_mismatches} demo(s) recorded under a '
              f'different success_factor than training '
              f'(training sf={sf_train}); {sf_unknown} demo(s) have no '
              f'recorded success_factor (legacy pkls). Their `success` '
              f'flags may not reflect the training criterion — '
              f'--bc_demos_only_success could keep/drop the wrong demos.')

    if not obs_buf:
        return None, np.zeros((0, 0), dtype=np.float32), 0, 0
    if isinstance(obs_buf[0], dict):
        keys = obs_buf[0].keys()
        stacked = {k: np.concatenate([o[k] for o in obs_buf], axis=0)
                   for k in keys}
    else:
        stacked = np.concatenate(obs_buf, axis=0)
    return (stacked,
            np.concatenate(act_buf, axis=0).astype(np.float32),
            n_success_demos, n_files)


def _bc_pretrain_sac(agent, demo_obs, demo_acts, epochs, batch_size, lr):
    """BC the SAC actor's mean-action head. Uses obs_to_tensor so image
    preprocessing (uint8→float, NHWC→NCHW transpose, /255 scaling) is
    handled by SB3's standard pipeline."""
    import torch.nn.functional as F

    actor = agent.policy.actor
    optimizer = torch.optim.Adam(actor.parameters(), lr=lr)

    if isinstance(demo_obs, dict):
        n = len(next(iter(demo_obs.values())))
    else:
        n = len(demo_obs)
    print(f'[BC] training SAC actor on {n} (obs, act) pairs '
          f'for {epochs} epochs, batch={batch_size}')

    act_t_full = torch.as_tensor(demo_acts, device=agent.device)

    def _slice_obs(obs, idx):
        if isinstance(obs, dict):
            return {k: v[idx] for k, v in obs.items()}
        return obs[idx]

    for epoch in range(epochs):
        perm = np.random.permutation(n)
        total_loss, count = 0.0, 0
        for i in range(0, n, batch_size):
            idx = perm[i:i + batch_size]
            obs_batch = _slice_obs(demo_obs, idx)
            obs_tensor, _ = agent.policy.obs_to_tensor(obs_batch)
            a_batch = act_t_full[torch.as_tensor(idx, device=agent.device)]

            features = actor.extract_features(obs_tensor)
            latent_pi = actor.latent_pi(features)
            mean_actions = actor.mu(latent_pi)
            predicted = torch.tanh(mean_actions)
            loss = F.mse_loss(predicted, a_batch)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * len(idx)
            count += len(idx)
        avg = total_loss / max(count, 1)
        print(f'[BC] epoch {epoch+1}/{epochs}  mse={avg:.5f}')
        if dedo_args.use_wandb:
            import wandb
            wandb.log({'bc/mse': avg, 'bc/epoch': epoch + 1})


if extra_args.bc_demo_path:
    print(f'\n=== BC pretrain: loading manual pixel demos from '
          f'{extra_args.bc_demo_path} ===')
    demo_obs, demo_acts, n_success, n_demos = _load_manual_pixel_demos(
        extra_args.bc_demo_path,
        cam_resolution=extra_args.cam_resolution,
        include_grip=not extra_args.no_grip,
        only_success=extra_args.bc_demos_only_success)
elif extra_args.bc_episodes > 0:
    _scripted_demos_dir = (
        None if extra_args.no_save_scripted_demos
        else os.path.join(dedo_args.logdir, 'scripted_demos'))
    print(f'\n=== BC pretrain: collecting {extra_args.bc_episodes} '
          f'scripted pixel demos '
          f'(only_success={extra_args.bc_demos_only_success}) ===')
    if _scripted_demos_dir is not None:
        print(f'[BC] saving scripted demos to {_scripted_demos_dir}')
    demo_obs, demo_acts, n_success = _collect_pixel_demos(
        dedo_args, extra_args.bc_episodes,
        only_success=extra_args.bc_demos_only_success,
        save_dir=_scripted_demos_dir)
    n_demos = extra_args.bc_episodes
else:
    demo_obs, demo_acts, n_success, n_demos = (
        None, np.zeros((0, 0), dtype=np.float32), 0, 0)

if demo_obs is not None and len(demo_acts) > 0:
    n_pairs = (len(next(iter(demo_obs.values())))
               if isinstance(demo_obs, dict) else len(demo_obs))
    _bc_pretrain_sac(agent, demo_obs, demo_acts,
                     epochs=extra_args.bc_epochs,
                     batch_size=128, lr=extra_args.bc_lr)
    if dedo_args.use_wandb:
        import wandb
        wandb.log({'bc/n_pairs': n_pairs,
                   'bc/n_success_demos': n_success,
                   'bc/n_demos': n_demos})
    print('[BC] note: SAC critic starts random; expect actor to drift '
          'from BC init for ~50k steps until critic stabilizes.')
elif extra_args.bc_demo_path or extra_args.bc_episodes > 0:
    print('[BC] no usable demos — skipping BC pretrain')


# ---------------------------------------------------------------------------
# Train.
# ---------------------------------------------------------------------------
print('Start SAC pixel training ...')
agent.learn(total_timesteps=extra_args.total_env_steps, callback=cb)

ckpt_path = os.path.join(dedo_args.logdir, 'agent.zip')
agent.save(ckpt_path)
agent.save_replay_buffer(os.path.join(dedo_args.logdir, 'replay_buffer.pkl'))
vec_env.save(os.path.join(dedo_args.logdir, 'vec_normalize.pkl'))
pickle.dump(dedo_args, open(os.path.join(dedo_args.logdir, 'args.pkl'), 'wb'))
print(f'\nDone. Checkpoint: {ckpt_path}')


# ---------------------------------------------------------------------------
# Final eval.
# ---------------------------------------------------------------------------
from stable_baselines3.common.evaluation import evaluate_policy

print(f'\nRunning final eval ({extra_args.n_final_eval_episodes} episodes)...')
collector = make_final_eval_collector()
mean_rwd, std_rwd = evaluate_policy(
    agent, eval_env, n_eval_episodes=extra_args.n_final_eval_episodes,
    deterministic=True, callback=collector, return_episode_rewards=False)

final_metrics = log_final_eval_metrics(
    collector, use_wandb=dedo_args.use_wandb, prefix='final_eval')
final_metrics['final_eval/mean_reward'] = float(mean_rwd)
final_metrics['final_eval/std_reward'] = float(std_rwd)

print(f'Final eval — mean_rwd={mean_rwd:.3f} ± {std_rwd:.3f}  '
      f"success_rate={final_metrics.get('final_eval/success_rate', float('nan')):.3f}  "
      f"(n={int(final_metrics.get('final_eval/n_episodes', 0))})")
print('  Per-metric means over the eval set:')
for k in sorted(final_metrics):
    if k.endswith('__std') or k in (
            'final_eval/mean_reward', 'final_eval/std_reward',
            'final_eval/success_rate', 'final_eval/n_episodes'):
        continue
    print(f'    {k:48s} = {final_metrics[k]:.4f}')

if dedo_args.use_wandb:
    import wandb
    wandb.log(final_metrics)
    wandb.finish()

vec_env.close()
