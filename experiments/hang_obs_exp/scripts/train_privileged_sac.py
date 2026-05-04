"""
Train SAC on HangProcCloth-v1 with privileged ground-truth cloth observations.

Counterpart to train_privileged.py (PPO). Same wrappers, same adaptive
success / reward shaping, same BC pretrain pipeline — only the RL algo
changes. SAC is off-policy with a replay buffer, typically 5-10× more
sample-efficient than PPO on continuous control with low-dim obs.

Usage (from repo root):
  python experiments/hang_obs_exp/scripts/train_privileged_sac.py \
      --obs_mode hole_centroid --use_wandb

  # With manual-demo BC pretrain:
  python experiments/hang_obs_exp/scripts/train_privileged_sac.py \
      --obs_mode hole_centroid --use_wandb \
      --bc_demo_path logs/hang_obs_exp/manual_demos

Outputs saved under logs/hang_obs_exp/<obs_mode>_sac/

NOTE on BC + SAC: BC initializes the actor with demo behavior, but SAC's
critic starts random. The first SAC updates use a random Q, so the actor
can drift away from the BC initialization until the critic stabilizes.
This usually self-corrects within ~50k steps but can hurt vs pure SAC if
demos are very high-quality. If that happens, set --bc_episodes 0 and
--bc_demo_path '' to disable BC.
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
from stable_baselines3.common.vec_env import VecNormalize


sys.path.insert(0, str(Path(__file__).resolve().parent))
from _helpers import RetryResetEnv, build_hole_aware_waypoints  # noqa: E402

import dedo  # registers gym envs
from dedo.utils.args import get_args_parser, args_postprocess
from dedo.utils.train_utils import init_train
from stable_baselines3.common.callbacks import CallbackList
from _video_callback import HangVideoCallback  # noqa: E402
from _reward_diagnostics import (  # noqa: E402
    RewardDiagnosticsCallback, dump_run_config, make_final_eval_collector,
    log_final_eval_metrics)

from experiments.hang_obs_exp.envs.privileged_env import PrivilegedObsWrapper


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
parser = argparse.ArgumentParser(add_help=False)
parser.add_argument('--obs_mode', type=str, default='hole_centroid',
                    choices=['hole_centroid', 'hole_vertices', 'full_mesh'])
parser.add_argument('--total_env_steps', type=int, default=1_000_000,
                    help='SAC is more sample-efficient than PPO; 1M is '
                         'usually enough.')
parser.add_argument('--num_envs', type=int, default=1,
                    help='SAC typically uses 1 env (off-policy gains '
                         'diminish with more parallel envs).')
parser.add_argument('--lr', type=float, default=3e-4)
parser.add_argument('--seed', type=int, default=42)
parser.add_argument('--logdir_root', type=str,
                    default=str(REPO_ROOT / 'logs' / 'hang_obs_exp'))
parser.add_argument('--use_wandb', action='store_true')
parser.add_argument('--n_final_eval_episodes', type=int, default=20)
# SAC-specific knobs.
parser.add_argument('--buffer_size', type=int, default=1_000_000)
parser.add_argument('--learning_starts', type=int, default=1_000)
parser.add_argument('--batch_size', type=int, default=256)
parser.add_argument('--tau', type=float, default=0.005)
parser.add_argument('--gamma', type=float, default=0.99)
parser.add_argument('--train_freq', type=int, default=1,
                    help='Train every N env steps.')
parser.add_argument('--gradient_steps', type=int, default=1,
                    help='Gradient updates per train trigger.')
parser.add_argument('--ent_coef', type=str, default='auto',
                    help='SAC entropy coef. "auto" = auto-tuned. '
                         'Or float like "0.1".')
# Adaptive success + shaping (same defaults as train_privileged.py).
parser.add_argument('--success_factor', type=float, default=1.2)
parser.add_argument('--no_adaptive_success', action='store_true',
                    help='Disable the adaptive success override entirely '
                         '(equivalent to success_factor=None). dedo '
                         'is_success is used as-is and bonus/penalty are '
                         'inert. Use to compare against base reward.')
parser.add_argument('--success_bonus', type=float, default=200.0)
parser.add_argument('--fail_penalty', type=float, default=0.0)
parser.add_argument('--vel_penalty', type=float, default=0.0,
                    help='Per-step penalty proportional to cloth mean '
                         'vertex displacement (proxy for cloth speed). '
                         'Same knob and meaning as train_privileged.py '
                         '(PPO) so SAC and PPO runs can be compared on '
                         'identical reward functions. 0 = off.')
# BC pretrain.
parser.add_argument('--bc_episodes', type=int, default=0,
                    help='If >0 and --bc_demo_path not set: collect '
                         'scripted hole-aware demos for BC pretrain.')
parser.add_argument('--bc_epochs', type=int, default=20)
parser.add_argument('--bc_lr', type=float, default=1e-3)
parser.add_argument('--bc_demo_path', type=str, default=None,
                    help='Directory of demo_NNN.pkl files from '
                         'record_demo.py.')
parser.add_argument('--bc_demos_only_success', action='store_true')
extra_args, remaining = parser.parse_known_args()

# `--no_adaptive_success` is the single switch for "use dedo's base reward
# and success unchanged". The wrapper treats success_factor=None as the
# disable signal, so we flip it here once for all downstream consumers.
if extra_args.no_adaptive_success:
    extra_args.success_factor = None


# ---------------------------------------------------------------------------
# Build dedo args.
# ---------------------------------------------------------------------------
sys.argv = [
    'train_privileged_sac',
    '--env=HangProcCloth-v1',
    '--cam_resolution=0',
    '--num_envs=0',
    '--total_env_steps=0',
    '--log_save_interval=50',
    '--seed', str(extra_args.seed),
    # Lock cam_viewmat against preset_override_util — see train_privileged.py.
    '--cam_viewmat', '9.0', '-25.0', '45.0', '0.0', '0.5', '6.5',
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
dedo_args.log_save_interval = 50
dedo_args.disable_logging_video = False

obs_mode = extra_args.obs_mode
logdir_base = os.path.join(extra_args.logdir_root, f'{obs_mode}_sac')
os.makedirs(logdir_base, exist_ok=True)
dedo_args.logdir = logdir_base

np.random.seed(dedo_args.seed)
torch.manual_seed(dedo_args.seed)


# ---------------------------------------------------------------------------
# Env factory.
# ---------------------------------------------------------------------------
def make_wrapped_env(args, obs_mode_str, monitor_dir=None):
    def _init():
        _args = deepcopy(args)
        _args.debug = False
        _args.viz = False
        env = gym.make(_args.env, args=_args)
        env = RetryResetEnv(env)
        env = PrivilegedObsWrapper(env, obs_mode=obs_mode_str,
                                    success_factor=extra_args.success_factor,
                                    success_bonus=extra_args.success_bonus,
                                    fail_penalty=extra_args.fail_penalty,
                                    vel_penalty=extra_args.vel_penalty)
        env = Monitor(env, filename=monitor_dir)
        return env
    return _init


# ---------------------------------------------------------------------------
# Build vec envs.
# ---------------------------------------------------------------------------
n_envs = extra_args.num_envs
vec_env = DummyVecEnv([make_wrapped_env(dedo_args, obs_mode)
                       for _ in range(n_envs)])
vec_env.seed(dedo_args.seed)
vec_env = VecNormalize(vec_env, norm_obs=True, norm_reward=True, clip_obs=10.0)

# Eval env with synced obs normalization (same pattern as PPO script).
eval_args = deepcopy(dedo_args)
eval_args.cam_viewmat = [9.0, -25.0, 45.0, 0.0, 0.5, 6.5]
eval_env_raw = gym.make(eval_args.env, args=eval_args)
eval_env_raw = RetryResetEnv(eval_env_raw)
eval_env_raw = PrivilegedObsWrapper(eval_env_raw, obs_mode=obs_mode,
                                     success_factor=extra_args.success_factor,
                                     success_bonus=extra_args.success_bonus,
                                     fail_penalty=extra_args.fail_penalty,
                                     vel_penalty=extra_args.vel_penalty)
eval_env_raw = Monitor(eval_env_raw)
eval_env_raw.seed(dedo_args.seed)


class _SyncObsNorm(gym.ObservationWrapper):
    def __init__(self, env, vec_normalize):
        super().__init__(env)
        self._vn = vec_normalize

    def observation(self, obs):
        return self._vn.normalize_obs(np.asarray(obs, dtype=np.float32))


eval_env = _SyncObsNorm(eval_env_raw, vec_env)

obs_shape = vec_env.observation_space.shape
print(f'\n{"="*60}')
print(f'Condition: privileged/{obs_mode}  ALGO=SAC')
print(f'Obs shape: {obs_shape}  Action shape: {vec_env.action_space.shape}')
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
        bc_tag = ('_bc' if (extra_args.bc_demo_path or
                            extra_args.bc_episodes > 0) else '')
        wandb.run.name = (f'{wandb.run.name}_sac_256x256'
                          f'{sf_tag}{sb_tag}{fp_tag}{vp_tag}{bc_tag}')
        wandb.run.save()
        wandb.run.tags = list(wandb.run.tags or []) + [
            'algo=sac',
            f'obs_mode={obs_mode}',
            f'success_factor={sf}',
            f'success_bonus={sb}',
            f'fail_penalty={fp}',
            f'vel_penalty={vp}',
            f'bc={"yes" if bc_tag else "no"}',
        ]


# ---------------------------------------------------------------------------
# SAC.
# ---------------------------------------------------------------------------
# ent_coef can be 'auto' or a float string.
try:
    ent_coef_arg = float(extra_args.ent_coef)
except ValueError:
    ent_coef_arg = extra_args.ent_coef

rl_kwargs = dict(
    learning_rate=dedo_args.lr,
    device=dedo_args.device,
    tensorboard_log=dedo_args.logdir,
    verbose=1,
    policy_kwargs=dict(net_arch=[256, 256]),
    buffer_size=extra_args.buffer_size,
    learning_starts=extra_args.learning_starts,
    batch_size=extra_args.batch_size,
    tau=extra_args.tau,
    gamma=extra_args.gamma,
    train_freq=extra_args.train_freq,
    gradient_steps=extra_args.gradient_steps,
    ent_coef=ent_coef_arg,
)
agent = SAC('MlpPolicy', vec_env, **rl_kwargs)

# Eval cadence: keep PPO-comparable so wandb plots line up.
num_steps_between_save = dedo_args.log_save_interval * 10 * 50
_sf = extra_args.success_factor
_sf_str = f'sf{_sf:g}' if _sf is not None else 'sf_off'
_video_basename = f'eval_{obs_mode}_sac_{_sf_str}'
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

# Persist a self-describing config.json + push to wandb.config so future
# debugging never has to guess what reward this run optimized.
dump_run_config(extra_args, dedo_args, dedo_args.logdir,
                use_wandb=dedo_args.use_wandb)


# ---------------------------------------------------------------------------
# Demo collection / loading (same as PPO version).
# ---------------------------------------------------------------------------
def _collect_demo_rollouts(args, obs_mode_str, num_episodes):
    from dedo.demo_preset import build_traj, merge_traj
    from dedo.envs.deform_env import DeformEnv

    raw = gym.make(args.env, args=deepcopy(args))
    raw = RetryResetEnv(raw)
    raw = PrivilegedObsWrapper(raw, obs_mode=obs_mode_str,
                                success_factor=extra_args.success_factor,
                                success_bonus=0.0, fail_penalty=0.0,
                                vel_penalty=0.0)
    raw.seed(args.seed + 1000)

    obs_buf, act_buf = [], []
    succeeded = 0
    for ep in range(num_episodes):
        obs = raw.reset()
        underlying = raw
        while hasattr(underlying, 'env'):
            underlying = underlying.env
            if isinstance(underlying, DeformEnv):
                break
        ctrl_freq = args.sim_freq / args.sim_steps_per_action

        preset_wp = build_hole_aware_waypoints(underlying)
        if preset_wp is None:
            print(f'[BC] demo {ep+1}: no hole loop, skipping')
            continue
        try:
            _, vel_a = build_traj(underlying, preset_wp, 'a',
                                  anchor_idx=0, ctrl_freq=ctrl_freq, robot=None)
            _, vel_b = build_traj(underlying, preset_wp, 'b',
                                  anchor_idx=1, ctrl_freq=ctrl_freq, robot=None)
            traj = merge_traj(vel_a, vel_b)
        except Exception as e:
            print(f'[BC] demo {ep+1}: build_traj failed ({e!r})')
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
            ep_obs.append(np.asarray(obs, dtype=np.float32))
            ep_act.append(normalized)
            obs, rwd, done, info = raw.step(normalized)
            ep_rwd += float(rwd)
            if 'is_success' in info:
                ep_succ = max(ep_succ, int(info['is_success']))
            if done:
                break
            step += 1
        obs_buf.extend(ep_obs)
        act_buf.extend(ep_act)
        succeeded += ep_succ
        print(f'[BC] demo {ep+1}/{num_episodes}  '
              f'len={len(ep_obs)}  rwd={ep_rwd:.1f}  success={ep_succ}')

    raw.close()
    return (np.asarray(obs_buf, dtype=np.float32),
            np.asarray(act_buf, dtype=np.float32),
            succeeded)


def _load_manual_demos(demo_dir, obs_mode_str, only_success=False):
    obs_buf, act_buf = [], []
    n_files, n_success_demos = 0, 0
    for fname in sorted(os.listdir(demo_dir)):
        if not (fname.startswith('demo_') and fname.endswith('.pkl')):
            continue
        path = os.path.join(demo_dir, fname)
        with open(path, 'rb') as f:
            d = pickle.load(f)

        if only_success and not d.get('success', 0):
            print(f'[BC] {fname}: success=0, skipping')
            continue

        if isinstance(d.get('obs'), dict):
            if obs_mode_str not in d['obs']:
                print(f'[BC] {fname}: missing {obs_mode_str!r} obs, skipping')
                continue
            obs_arr = d['obs'][obs_mode_str]
        else:
            if d.get('obs_mode') != obs_mode_str:
                print(f'[BC] {fname}: legacy demo recorded as '
                      f'{d.get("obs_mode")!r}, skipping')
                continue
            obs_arr = d['obs']

        n_files += 1
        n_success_demos += int(d.get('success', 0))
        obs_buf.append(obs_arr)
        act_buf.append(d['acts'])
        print(f'[BC] loaded {fname}  len={d.get("len", len(d["acts"]))}  '
              f'rwd={d.get("reward", 0):.2f}  '
              f'success={d.get("success", 0)}')

    if not obs_buf:
        return (np.zeros((0, 0), dtype=np.float32),
                np.zeros((0, 0), dtype=np.float32),
                0, 0)
    return (np.concatenate(obs_buf, axis=0).astype(np.float32),
            np.concatenate(act_buf, axis=0).astype(np.float32),
            n_success_demos, n_files)


def _bc_pretrain_sac(agent, demo_obs, demo_acts, vec_normalize,
                     epochs, batch_size, lr):
    """BC the SAC actor's mean-action head. Loss = MSE(tanh(mu), demo_act)
    since SAC's deterministic action is tanh(mu(latent_pi))."""
    import torch.nn.functional as F
    vec_normalize.obs_rms.update(demo_obs)
    norm_obs = vec_normalize.normalize_obs(demo_obs).astype(np.float32)

    obs_t = torch.as_tensor(norm_obs, device=agent.device)
    act_t = torch.as_tensor(demo_acts, device=agent.device)

    actor = agent.policy.actor
    optimizer = torch.optim.Adam(actor.parameters(), lr=lr)
    n = len(demo_obs)
    print(f'[BC] training SAC actor on {n} (obs, act) pairs '
          f'for {epochs} epochs, batch={batch_size}')
    for epoch in range(epochs):
        perm = torch.randperm(n, device=agent.device)
        total_loss, count = 0.0, 0
        for i in range(0, n, batch_size):
            idx = perm[i:i + batch_size]
            o, a = obs_t[idx], act_t[idx]
            features = actor.extract_features(o)
            latent_pi = actor.latent_pi(features)
            mean_actions = actor.mu(latent_pi)
            # SAC's deterministic action is tanh(mean_actions).
            predicted = torch.tanh(mean_actions)
            loss = F.mse_loss(predicted, a)
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
    print(f'\n=== BC pretrain: loading manual demos from '
          f'{extra_args.bc_demo_path} ===')
    demo_obs, demo_acts, n_success, n_demos = _load_manual_demos(
        extra_args.bc_demo_path, obs_mode,
        only_success=extra_args.bc_demos_only_success)
elif extra_args.bc_episodes > 0:
    print(f'\n=== BC pretrain: collecting {extra_args.bc_episodes} '
          f'scripted demo rollouts ===')
    demo_obs, demo_acts, n_success = _collect_demo_rollouts(
        dedo_args, obs_mode, extra_args.bc_episodes)
    n_demos = extra_args.bc_episodes
else:
    demo_obs = np.zeros((0, 0), dtype=np.float32)
    demo_acts = np.zeros((0, 0), dtype=np.float32)
    n_success, n_demos = 0, 0

if len(demo_obs) > 0:
    _bc_pretrain_sac(agent, demo_obs, demo_acts, vec_env,
                     epochs=extra_args.bc_epochs,
                     batch_size=256, lr=extra_args.bc_lr)
    if dedo_args.use_wandb:
        import wandb
        wandb.log({'bc/n_pairs': len(demo_obs),
                   'bc/n_success_demos': n_success,
                   'bc/n_demos': n_demos})
    print('[BC] note: SAC critic starts random; expect actor to drift '
          'from BC init for ~50k steps until critic stabilizes.')
elif extra_args.bc_demo_path or extra_args.bc_episodes > 0:
    print('[BC] no usable demos found — skipping BC pretrain')


# ---------------------------------------------------------------------------
# Train.
# ---------------------------------------------------------------------------
print('Start SAC privileged training ...')
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
