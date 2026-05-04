"""
Train PPO on HangProcCloth-v1 with PIXEL (RGB) observations.

Counterpart to train_privileged.py / train_pointcloud.py for the obs-
modality comparison. Renders a 64x64 RGB image from the front-angled
camera (yaw=45, pitch=-25 — the "full-observability" config from
scripts/train_full.sh, where both the cloth's hole and the hanger goal
are visible). Uses SB3 'MultiInputPolicy' over a Dict observation
{image, grip} so the CNN doesn't have to re-derive gripper state from
pixels — the privileged baselines also see gripper proprio, so this
keeps the comparison fair. Use --no_grip to drop gripper state and run
pure pixel obs with 'CnnPolicy'.

Adaptive success / reward shaping (success_factor, success_bonus,
fail_penalty, vel_penalty) defaults match train_privileged.py so the
runs are cross-comparable. Eval videos logged to wandb via
HangVideoCallback (renders independently of the obs pipeline so
resolution / quality is decoupled from cam_resolution).

Usage (from repo root):
  # Default: Dict obs (image + grip), MultiInputPolicy.
  python experiments/hang_obs_exp/scripts/train_pixels.py --use_wandb

  # Pure-pixel CnnPolicy.
  python experiments/hang_obs_exp/scripts/train_pixels.py --no_grip --use_wandb

  # Bigger image (CNN sees more):
  python experiments/hang_obs_exp/scripts/train_pixels.py \
      --cam_resolution 96 --use_wandb

Outputs saved under logs/hang_obs_exp/pixels[_<resolution>]/
"""

import sys, os, argparse, pickle
from copy import deepcopy
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT))

import numpy as np
import torch
import gym
from stable_baselines3 import PPO
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
parser.add_argument('--total_env_steps', type=int, default=3_000_000)
parser.add_argument('--num_envs', type=int, default=4)
parser.add_argument('--lr', type=float, default=3e-4)
parser.add_argument('--seed', type=int, default=42)
parser.add_argument('--logdir_root', type=str,
                    default=str(REPO_ROOT / 'logs' / 'hang_obs_exp'))
parser.add_argument('--use_wandb', action='store_true')
parser.add_argument('--n_final_eval_episodes', type=int, default=20)
# Camera config: defaults to the "full-observability" front-angled view from
# scripts/train_full.sh (yaw=45, pitch=-25 — both hole and hanger visible).
# Pass a different 6-tuple to test a partial-observability camera.
parser.add_argument('--cam_resolution', type=int, default=64,
                    help='Square RGB resolution. 64 matches train_full.sh; '
                         '96 / 84 give the CNN more spatial detail at the '
                         'cost of compute.')
parser.add_argument('--cam_viewmat', type=float, nargs=6,
                    default=[9.0, -25.0, 45.0, 0.0, 0.5, 6.5],
                    help='[distance, pitch, yaw, posX, posY, posZ]. Default '
                         'is the front-angled view from train_full.sh.')
parser.add_argument('--no_grip', action='store_true',
                    help='Drop gripper proprio from obs. Use pure CnnPolicy '
                         'over a (H, W, 3) uint8 image. Default keeps grip '
                         'in a Dict obs alongside the image (MultiInputPolicy).')
# Adaptive success + shaping (same defaults as train_privileged.py).
parser.add_argument('--success_factor', type=float, default=1.2,
                    help='If set, override env success threshold with '
                         'dist < success_factor * hole_radius (adaptive).')
parser.add_argument('--no_adaptive_success', action='store_true',
                    help='Disable the adaptive success override entirely.')
parser.add_argument('--success_bonus', type=float, default=200.0)
parser.add_argument('--fail_penalty', type=float, default=0.0)
parser.add_argument('--vel_penalty', type=float, default=0.0,
                    help='Per-step penalty proportional to cloth mean '
                         'vertex displacement. Mirrors train_privileged.py.')
# BC pretrain (scripted hole-aware demos rolled out in pixel obs space).
parser.add_argument('--bc_episodes', type=int, default=0,
                    help='Scripted demo rollouts to BC-pretrain on. '
                         '0 = skip (default; vision BC is finicky and a '
                         'random PPO init is usually fine if reward is '
                         'shaped).')
parser.add_argument('--bc_epochs', type=int, default=20)
parser.add_argument('--bc_lr', type=float, default=1e-3)
# PPO knobs.
parser.add_argument('--n_steps', type=int, default=2048,
                    help='Rollout buffer per env. Smaller than the '
                         'privileged default (4096) since per-step CNN '
                         'inference is slower and we want more frequent '
                         'updates.')
parser.add_argument('--batch_size', type=int, default=256)
parser.add_argument('--n_epochs', type=int, default=10)
parser.add_argument('--max_episode_len', type=int, default=200)
parser.add_argument('--cpu', action='store_true',
                    help='Force CPU. Vision PPO usually wants GPU; this is '
                         'an escape hatch.')
extra_args, remaining = parser.parse_known_args()

if extra_args.no_adaptive_success:
    extra_args.success_factor = None

# Distinguish pixel runs by resolution + grip-mode in the logdir name.
_grip_str = 'pixgrip' if not extra_args.no_grip else 'pixonly'
run_subdir = f'{_grip_str}_{extra_args.cam_resolution}'

# ---------------------------------------------------------------------------
# Build dedo args. cam_resolution>0 + uint8_pixels are required for pixel
# obs; the wrapper enforces this on construction but we set it here too
# to be explicit.
# ---------------------------------------------------------------------------
sys.argv = [
    'train_pixels',
    '--env=HangProcCloth-v1',
    f'--cam_resolution={extra_args.cam_resolution}',
    '--uint8_pixels',
    '--num_envs=0',
    '--total_env_steps=0',
    '--log_save_interval=50',
    '--seed', str(extra_args.seed),
    '--max_episode_len', str(extra_args.max_episode_len),
    # Lock cam_viewmat against preset_override_util — every env reset()
    # would otherwise clobber it with procedural_hang_cloth's preset
    # (yaw=314, target z=5.3), which hides the hanger once the cloth drops.
    '--cam_viewmat',
    str(extra_args.cam_viewmat[0]), str(extra_args.cam_viewmat[1]),
    str(extra_args.cam_viewmat[2]), str(extra_args.cam_viewmat[3]),
    str(extra_args.cam_viewmat[4]), str(extra_args.cam_viewmat[5]),
]
dedo_args, _ = get_args_parser()
args_postprocess(dedo_args)
dedo_args.rl_algo = 'PPO'
dedo_args.seed = extra_args.seed
dedo_args.use_wandb = extra_args.use_wandb
dedo_args.total_env_steps = extra_args.total_env_steps
dedo_args.num_envs = extra_args.num_envs
dedo_args.lr = extra_args.lr
dedo_args.debug = False
dedo_args.viz = False
dedo_args.log_save_interval = 50
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
# Build vec envs. norm_obs=False is the right default for image obs:
#   - The CNN expects raw uint8 pixels and rescales internally.
#   - The grip slice is already divided by 20 in the wrapper, so it's in
#     ~[-1, 1] without VecNormalize's running-mean whitening.
# We do still normalize REWARDS — HangProcCloth rewards range over ~[-300,
# +300] depending on shaping params, and PPO benefits from advantage
# scaling.
# ---------------------------------------------------------------------------
n_envs = extra_args.num_envs
vec_env = DummyVecEnv([make_wrapped_env(dedo_args) for _ in range(n_envs)])
vec_env.seed(dedo_args.seed)
# IMPORTANT: VecTransposeImage MUST come before VecNormalize. SB3's
# auto-transpose-on-CnnPolicy logic places the transpose around whatever
# is the outermost env at PPO/SAC __init__ time. If VecNormalize is on
# the outside, the transpose only modifies obs_space metadata for the
# top wrapper and SAC's ReplayBuffer (which allocates from
# observation_space.shape but reads obs through VecNormalize's
# pass-through) ends up with mismatched HWC vs CHW shapes. Applying the
# transpose here fixes both algos and is a no-op for non-image obs.
vec_env = VecTransposeImage(vec_env)
vec_env = VecNormalize(vec_env, norm_obs=False, norm_reward=True,
                       clip_obs=10.0)

# Eval env (single, non-vec) — same wrapper stack so HangVideoCallback can
# render directly. No obs normalization to sync since norm_obs=False above.
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

obs_shape = vec_env.observation_space
print(f'\n{"="*60}')
print(f'Condition: pixels  (grip={"yes" if not extra_args.no_grip else "no"})')
print(f'Obs space: {obs_shape}')
print(f'Action shape: {vec_env.action_space.shape}')
print(f'Image: {extra_args.cam_resolution}x{extra_args.cam_resolution}x3 '
      f'uint8  cam_viewmat={extra_args.cam_viewmat}')
print(f'Steps: {extra_args.total_env_steps:,}  Envs: {n_envs}  '
      f'LR: {extra_args.lr}')
print(f'Logdir: {logdir_base}')
print(f'{"="*60}\n')


# ---------------------------------------------------------------------------
# Init wandb / logdir.
# ---------------------------------------------------------------------------
dedo_args.logdir, dedo_args.device = init_train('PPO', dedo_args)
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
        grip_tag = '_grip' if not extra_args.no_grip else '_pix'
        wandb.run.name = (f'{wandb.run.name}_pixels'
                          f'{extra_args.cam_resolution}{grip_tag}'
                          f'{sf_tag}{sb_tag}{fp_tag}{vp_tag}')
        wandb.run.save()
        wandb.run.tags = list(wandb.run.tags or []) + [
            'obs=pixels',
            f'include_grip={"yes" if not extra_args.no_grip else "no"}',
            f'cam_resolution={extra_args.cam_resolution}',
            f'success_factor={sf if sf is not None else "default"}',
            f'success_bonus={sb}',
            f'fail_penalty={fp}',
            f'vel_penalty={vp}',
        ]


# ---------------------------------------------------------------------------
# PPO. MultiInputPolicy when obs is Dict, CnnPolicy for image-only.
# net_arch=[256, 256] for the post-feature-extractor MLP head matches the
# privileged scripts. SB3's NatureCNN is the default features extractor;
# leaving it at the default is the right call unless we want to swap in
# IMPALA-CNN or similar.
# ---------------------------------------------------------------------------
policy_name = 'MultiInputPolicy' if not extra_args.no_grip else 'CnnPolicy'
policy_kwargs = dict(net_arch=[256, 256])
rl_kwargs = dict(
    learning_rate=dedo_args.lr,
    device=dedo_args.device,
    tensorboard_log=dedo_args.logdir,
    verbose=1,
    policy_kwargs=policy_kwargs,
    n_steps=extra_args.n_steps,
    batch_size=extra_args.batch_size,
    n_epochs=extra_args.n_epochs,
    gae_lambda=0.95, gamma=0.99,
)
agent = PPO(policy_name, vec_env, **rl_kwargs)

num_steps_between_save = dedo_args.log_save_interval * 10 * 50
_sf = extra_args.success_factor
_sf_str = f'sf{_sf:g}' if _sf is not None else 'sf_off'
_video_basename = (f'eval_pixels{extra_args.cam_resolution}'
                   f'{"_grip" if not extra_args.no_grip else ""}_{_sf_str}')
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
# Optional BC pretrain on scripted hole-aware demos. Walk the dedo demo
# preset waypoints, capture (obs, action) pairs in pixel obs space, and
# regress the policy mean onto the demo actions. Skipped by default
# (--bc_episodes 0); vision BC is finicky and shaped reward usually
# converges without it.
# ---------------------------------------------------------------------------
def _collect_pixel_demos(args, num_episodes):
    from dedo.demo_preset import build_traj, merge_traj
    from dedo.envs.deform_env import DeformEnv

    raw = gym.make(args.env, args=deepcopy(args))
    raw = RetryResetEnv(raw)
    raw = PixelObsWrapper(
        raw,
        cam_resolution=extra_args.cam_resolution,
        include_grip=not extra_args.no_grip,
        success_factor=extra_args.success_factor,
        success_bonus=0.0,  # don't shape demo reward — BC uses (obs, act)
        fail_penalty=0.0,
        vel_penalty=0.0,
    )
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
            ep_obs.append(obs)
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

    # Stack obs into either an array (image-only) or dict-of-arrays (Dict obs).
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


def _bc_pretrain(agent, demo_obs, demo_acts, epochs, batch_size, lr):
    """Supervised regression of policy mean action onto demo actions.

    Handles both Box (image-only) and Dict (image+grip) obs by routing
    through SB3's policy.obs_to_tensor, which preprocesses each component
    correctly (uint8→float for images, channel-first transpose, etc.).
    """
    import torch.nn.functional as F

    policy = agent.policy
    optimizer = torch.optim.Adam(policy.parameters(), lr=lr)

    if isinstance(demo_obs, dict):
        n = len(next(iter(demo_obs.values())))
    else:
        n = len(demo_obs)
    print(f'[BC] training on {n} (obs, action) pairs '
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
            obs_tensor, _ = policy.obs_to_tensor(obs_batch)
            a_batch = act_t_full[torch.as_tensor(idx, device=agent.device)]

            features = policy.extract_features(obs_tensor)
            latent_pi, _ = policy.mlp_extractor(features)
            mean_a = policy.action_net(latent_pi)
            loss = F.mse_loss(mean_a, a_batch)
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


if extra_args.bc_episodes > 0:
    print(f'\n=== BC pretrain: collecting {extra_args.bc_episodes} '
          f'scripted pixel demos ===')
    demo_obs, demo_acts, n_success = _collect_pixel_demos(
        dedo_args, extra_args.bc_episodes)
    if demo_obs is not None and len(demo_acts) > 0:
        n_pairs = (len(next(iter(demo_obs.values())))
                   if isinstance(demo_obs, dict) else len(demo_obs))
        print(f'[BC] collected {n_pairs} (obs,act) pairs from '
              f'{extra_args.bc_episodes} demos ({n_success} succeeded)')
        _bc_pretrain(agent, demo_obs, demo_acts,
                     epochs=extra_args.bc_epochs,
                     batch_size=128, lr=extra_args.bc_lr)
        if dedo_args.use_wandb:
            import wandb
            wandb.log({'bc/n_pairs': n_pairs,
                       'bc/n_success_demos': n_success,
                       'bc/n_demos': extra_args.bc_episodes})
    else:
        print('[BC] no usable demos — skipping BC pretrain')


print('Start pixel RL training ...')
agent.learn(total_timesteps=extra_args.total_env_steps, callback=cb)

ckpt_path = os.path.join(dedo_args.logdir, 'agent.zip')
agent.save(ckpt_path)
vec_env.save(os.path.join(dedo_args.logdir, 'vec_normalize.pkl'))
pickle.dump(dedo_args, open(os.path.join(dedo_args.logdir, 'args.pkl'), 'wb'))
print(f'\nDone. Checkpoint: {ckpt_path}')


# ---------------------------------------------------------------------------
# Final eval: deterministic, log mean reward + success rate.
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
