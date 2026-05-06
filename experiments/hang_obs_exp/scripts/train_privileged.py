"""
Train PPO on HangProcCloth-v1 with privileged ground-truth cloth observations.

Four conditions launched from this script (select via --obs_mode):

  hole_centroid          — 18-dim: gripper + hole centroid + hanger goal   [fastest]
  hole_centroid_corners  — 30-dim: gripper + centroid + 4 cloth corners + goal
  hole_vertices          — 132-dim: gripper + all hole-boundary vertices
  full_mesh              — ~762-dim: gripper + all cloth vertex positions

Usage (from repo root):
  python experiments/hang_obs_exp/scripts/train_privileged.py --obs_mode hole_centroid
  python experiments/hang_obs_exp/scripts/train_privileged.py --obs_mode hole_centroid_corners
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
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import VecNormalize


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
from experiments.hang_obs_exp.envs.privileged_env import PrivilegedObsWrapper


# ---------------------------------------------------------------------------
# Parse our extra arg on top of dedo's args
# ---------------------------------------------------------------------------
parser = argparse.ArgumentParser(add_help=False)
parser.add_argument('--obs_mode', type=str, default='hole_centroid',
                    choices=['hole_centroid', 'hole_centroid_corners',
                             'hole_vertices', 'full_mesh'])
parser.add_argument('--total_env_steps', type=int, default=3_000_000)
parser.add_argument('--num_envs', type=int, default=4)
parser.add_argument('--lr', type=float, default=3e-4)
parser.add_argument('--seed', type=int, default=42)
parser.add_argument('--logdir_root', type=str,
                    default=str(REPO_ROOT / 'logs' / 'hang_obs_exp'))
parser.add_argument('--use_wandb', action='store_true',
                    help='Log metrics to wandb')
parser.add_argument('--n_final_eval_episodes', type=int, default=50,
                    help='Number of deterministic eval episodes at end '
                         'of training. SE shrinks as 1/sqrt(n); n=50 '
                         'gives SE≈0.07 at p=0.5 — a reliable summary '
                         'number for cross-run comparison. Cheap '
                         '(end-of-run only).')
parser.add_argument('--n_eval_episodes_during_training', type=int, default=30,
                    help='Number of deterministic eval episodes per '
                         'in-training eval pass. Drives the standard '
                         'error of eval/success_rate: n=30 → SE≈0.09 '
                         'at p=0.5 (vs the legacy n=10 → SE≈0.16, '
                         'which is most of the chart noise). Each '
                         'episode is ~max_episode_len env steps.')
parser.add_argument('--eval_seed_lock', dest='eval_seed_lock',
                    action='store_true', default=True,
                    help='Default. Re-seed the eval env to '
                         '`seed + 9999` at the start of every eval '
                         'pass so the same N procedural cloths are '
                         'evaluated every checkpoint. Eval-rate trace '
                         'then reflects only policy change, not env '
                         'resampling — much cleaner for cross-run '
                         'comparison. Generalization signal still '
                         'comes from the final eval (different seed '
                         'offset, --n_final_eval_episodes). Pass '
                         '--no_eval_seed_lock to disable.')
parser.add_argument('--no_eval_seed_lock', dest='eval_seed_lock',
                    action='store_false',
                    help='Disable eval seed locking; eval env RNG '
                         'state advances naturally between eval '
                         'passes (legacy behavior).')
parser.add_argument('--log_save_interval', type=int, default=50,
                    help='Controls checkpoint / eval / video cadence. '
                         'Checkpoint every (log_save_interval * 10 * 50) '
                         'env steps; eval every 2nd checkpoint; video '
                         'every 4th checkpoint. Lower = more frequent. '
                         'Default 50 → checkpoint @ 25k / eval @ 50k / '
                         'video @ 100k. Try 20 for ~2.5x faster '
                         'feedback during early training.')
parser.add_argument('--bc_episodes', type=int, default=50,
                    help='Target number of scripted demos to KEEP for BC '
                         'pretrain (0 to skip). The collector retries '
                         'until it has this many demos that pass the '
                         'keep criterion (any if --bc_demos_only_success '
                         'off; only is_success=1 if on), capped at '
                         '~3x attempts. Dataset size is therefore '
                         'deterministic across seeds.')
parser.add_argument('--bc_epochs', type=int, default=20,
                    help='BC pretrain epochs over collected demos')
parser.add_argument('--bc_lr', type=float, default=1e-3,
                    help='BC pretrain learning rate')
parser.add_argument('--bc_demo_path', type=str, default=None,
                    help='Path to a directory of demo_NNN.pkl files (from '
                         'record_demo.py). If set, skip scripted demo '
                         'collection and BC on these manual demos instead.')
parser.add_argument('--bc_demos_only_success', action='store_true',
                    help='Filter BC demos to only those that fire '
                         'is_success at terminal step. Applies to BOTH '
                         'manual demos (loaded from --bc_demo_path) and '
                         'scripted demos collected via --bc_episodes. '
                         'Strongly recommended for scripted demos: the '
                         'hole-aware waypoint controller succeeds ~60-80% '
                         'of the time, and dropping the failures leaves '
                         'a cleaner BC dataset.')
parser.add_argument('--success_factor', type=float, default=1.2,
                    help='If set, override env success threshold with '
                         'dist < success_factor * hole_radius (adaptive). '
                         'e.g. 0.8 = hole-radius-proportional. Applied to '
                         'training, eval, and final-eval envs.')
parser.add_argument('--no_adaptive_success', action='store_true',
                    help='Disable the adaptive success override entirely. '
                         'Equivalent to passing success_factor=None: dedo '
                         'is_success is used as-is and success_bonus / '
                         'fail_penalty have no effect (they are gated on '
                         'the adaptive criterion). Use this for runs that '
                         'compare against the unmodified base reward.')
parser.add_argument('--success_bonus', type=float, default=200.0,
                    help='Extra reward added at terminal step when the '
                         'adaptive success criterion fires. Makes PPO '
                         'actually optimize success, not just distance. '
                         '0 = no shaping. Suggested 100-400; dedo per-'
                         'step rewards have magnitude ~10-20.')
parser.add_argument('--fail_penalty', type=float, default=0.0,
                    help='Extra negative reward at terminal step when '
                         'success does NOT fire. Use together with '
                         'success_bonus for a clearer success/fail gap.')
parser.add_argument('--vel_penalty', type=float, default=0.0,
                    help='Per-step penalty proportional to cloth mean '
                         'vertex displacement (proxy for cloth speed). '
                         'reward -= vel_penalty * mean_per_vertex_disp. '
                         'Use to slow down whippy trajectories. Only '
                         'applied during the policy phase. 0 = off; try '
                         '1-10 for a meaningful effect.')
parser.add_argument('--boundary_penalty', type=float, default=0.0,
                    help='Terminal penalty applied when an episode ends '
                         'via dedo workspace-bound exit (gripper past '
                         'gripper_lims) rather than natural max-steps '
                         'timeout. Closes the "lift the cloth high to '
                         'cut the episode short" reward-hacking exit: '
                         'without this, PPO can trigger the workspace '
                         'bound to skip the per-step distance penalty '
                         'over the remaining steps, parking the hole '
                         'above the peg without ever threading. The '
                         'penalty must dominate the marginal benefit of '
                         'cutting ~150 dithering steps at ~-0.25/step; '
                         '0 = off (default). Suggested 100-400. Applied '
                         'identically to training, eval, and BC scripted-'
                         'demo collection envs so all three see the '
                         'same reward.')
parser.add_argument('--z_overshoot_penalty', type=float, default=0.0,
                    help='Per-step penalty proportional to the cloth\'s '
                         'hole-centroid z above (peg_z + z_overshoot_slack). '
                         'Targets the orientation-blind reward landscape: '
                         'dedo\'s base reward only measures full-3D '
                         'centroid-to-goal distance, so hovering the hole '
                         'high above the peg minimizes it just as well as '
                         'threading. boundary_penalty alone does not fix '
                         'this — the policy can still lift below '
                         'gripper_lims. 0 = off (default). Suggested 1-5. '
                         'Applied training/eval/BC envs identically.')
parser.add_argument('--z_overshoot_slack', type=float, default=1.0,
                    help='Meters above peg z that count as the "free zone" '
                         'with no overshoot penalty. The cloth starts '
                         'above the peg and approaches from above, so '
                         'a non-trivial slack avoids penalizing the '
                         'natural threading approach. Default 1.0 m. '
                         'Tighten to 0.3-0.5 for more aggressive shaping; '
                         'loosen to 2-3 for longer cloths.')
parser.add_argument('--cpu', action='store_true',
                    help='Force CPU even if CUDA is available. Often faster '
                         'for the small MLP over privileged obs since GPU '
                         'kernel-launch overhead dominates.')
parser.add_argument('--max_episode_len', type=int, default=200,
                    help='Steps per episode before timeout-done fires. '
                         'dedo default is 200; lower (e.g. 100) cuts '
                         'wall-clock and avoids dithering after the cloth '
                         'has already reached the pole region.')
extra_args, remaining = parser.parse_known_args()

# `--no_adaptive_success` is the single switch for "use dedo's base reward
# and success unchanged". The wrapper treats success_factor=None as the
# disable signal, so we flip it here once for all downstream consumers
# (env factory, eval env, demo collector, banner, wandb tags).
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

# Build dedo args with cam_resolution=0 (wrapper takes care of geometry obs)
sys.argv = [
    'train_privileged',
    '--env=HangProcCloth-v1',
    '--cam_resolution=0',
    '--num_envs=0',
    '--total_env_steps=0',
    f'--log_save_interval={extra_args.log_save_interval}',
    '--seed', str(extra_args.seed),
    '--max_episode_len', str(extra_args.max_episode_len),
    # Lock cam_viewmat against preset_override_util — every env reset()
    # would otherwise clobber it with procedural_hang_cloth's preset
    # (yaw=314, target z=5.3), which hides the hanger once the cloth drops.
    # Listing the flag here makes preset_override_util skip cam_viewmat.
    '--cam_viewmat', '9.0', '-25.0', '45.0', '0.0', '0.5', '6.5',
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
dedo_args.log_save_interval = extra_args.log_save_interval
# eval_env gets its own camera (built below) for debugging videos, so leave
# this enabled — CustomCallback will log a video every 4th save.
dedo_args.disable_logging_video = False

obs_mode = extra_args.obs_mode
logdir_base = os.path.join(extra_args.logdir_root, obs_mode)
os.makedirs(logdir_base, exist_ok=True)
dedo_args.logdir = logdir_base

np.random.seed(dedo_args.seed)
torch.manual_seed(dedo_args.seed)


# ---------------------------------------------------------------------------
# Factory: wrapped env (DummyVecEnv — all envs run in-process)
# ---------------------------------------------------------------------------
def make_wrapped_env(args, obs_mode_str, monitor_dir=None):
    def _init():
        _args = deepcopy(args)
        _args.debug = False
        _args.viz = False
        env = gym.make(_args.env, args=_args)
        # Wrap underlying env first so retry catches loadSoftBody failures
        # before PrivilegedObsWrapper / Monitor see them.
        env = RetryResetEnv(env)
        env = PrivilegedObsWrapper(env, obs_mode=obs_mode_str,
                                    success_factor=extra_args.success_factor,
                                    success_bonus=extra_args.success_bonus,
                                    fail_penalty=extra_args.fail_penalty,
                                    vel_penalty=extra_args.vel_penalty,
                                    boundary_penalty=extra_args.boundary_penalty,
                                    z_overshoot_penalty=extra_args.z_overshoot_penalty,
                                    z_overshoot_slack=extra_args.z_overshoot_slack)
        # Monitor records ep rewards/lengths so SB3 logs rollout/ep_rew_mean.
        env = Monitor(env, filename=monitor_dir)
        return env
    return _init


# ---------------------------------------------------------------------------
# Build vec envs
# ---------------------------------------------------------------------------
n_envs = extra_args.num_envs
env_fns = [make_wrapped_env(dedo_args, obs_mode) for _ in range(n_envs)]
vec_env = DummyVecEnv(env_fns)
vec_env.seed(dedo_args.seed)
# VecNormalize: running obs mean/std + reward normalization. Big PPO unlock
# when reward magnitudes are far from 0 (HangProcCloth: ~-300) and obs
# components have very different scales.
vec_env = VecNormalize(vec_env, norm_obs=True, norm_reward=True, clip_obs=10.0)

# Eval env: keep cam_resolution=0 so the underlying obs is gripper-only
# (matches PrivilegedObsWrapper's expectations), but set cam_viewmat to the
# front-angled view so CustomCallback's render() calls produce debug videos.
# render() reads cam_viewmat dynamically and accepts any width/height.
eval_args = deepcopy(dedo_args)
eval_args.cam_viewmat = [9.0, -25.0, 45.0, 0.0, 0.5, 6.5]
eval_env_raw = gym.make(eval_args.env, args=eval_args)
eval_env_raw = RetryResetEnv(eval_env_raw)
eval_env_raw = PrivilegedObsWrapper(eval_env_raw, obs_mode=obs_mode,
                                     success_factor=extra_args.success_factor,
                                    success_bonus=extra_args.success_bonus,
                                    fail_penalty=extra_args.fail_penalty,
                                    vel_penalty=extra_args.vel_penalty,
                                    boundary_penalty=extra_args.boundary_penalty,
                                    z_overshoot_penalty=extra_args.z_overshoot_penalty,
                                    z_overshoot_slack=extra_args.z_overshoot_slack)
eval_env_raw = Monitor(eval_env_raw)
eval_env_raw.seed(dedo_args.seed)


# Sync the eval env's obs normalization with the training VecNormalize so
# the policy sees consistently-normalized inputs at eval time. Kept as a
# single (non-vec) env so CustomCallback's render(width=..., height=...)
# call still works for video logging.
class _SyncObsNorm(gym.ObservationWrapper):
    def __init__(self, env, vec_normalize):
        super().__init__(env)
        self._vn = vec_normalize

    def observation(self, obs):
        return self._vn.normalize_obs(np.asarray(obs, dtype=np.float32))


eval_env = _SyncObsNorm(eval_env_raw, vec_env)

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
if extra_args.cpu:
    dedo_args.device = 'cpu'
    print(f'[device] forced CPU via --cpu flag '
          f'(torch.cuda.is_available()={torch.cuda.is_available()})')

# Append "_256x256" tag to the wandb run name to mark the bigger-net config.
if dedo_args.use_wandb:
    import wandb
    if wandb.run is not None:
        sf = extra_args.success_factor
        sb = extra_args.success_bonus
        fp = extra_args.fail_penalty
        vp = extra_args.vel_penalty
        bp = extra_args.boundary_penalty
        zp = extra_args.z_overshoot_penalty
        zs = extra_args.z_overshoot_slack
        sf_tag = f'_sf{sf:g}' if sf is not None else '_sf_default'
        sb_tag = f'_sb{sb:g}' if sb else ''
        fp_tag = f'_fp{fp:g}' if fp else ''
        vp_tag = f'_vp{vp:g}' if vp else ''
        bp_tag = f'_bp{bp:g}' if bp else ''
        zp_tag = f'_zp{zp:g}s{zs:g}' if zp else ''
        wandb.run.name = (
            f'{wandb.run.name}_256x256{sf_tag}{sb_tag}{fp_tag}{vp_tag}{bp_tag}{zp_tag}')
        wandb.run.tags = list(wandb.run.tags or []) + [
            f'success_factor={sf if sf is not None else "default"}',
            f'success_bonus={sb}',
            f'fail_penalty={fp}',
            f'vel_penalty={vp}',
            f'boundary_penalty={bp}',
            f'z_overshoot_penalty={zp}',
            f'z_overshoot_slack={zs}',
            f'obs_mode={obs_mode}',
        ]

rl_kwargs = {
    'learning_rate': dedo_args.lr,
    'device': dedo_args.device,
    'tensorboard_log': dedo_args.logdir,
    'verbose': 1,
    # Bigger network — defaults [64, 64] are too small, especially for
    # full_mesh (~762-dim).
    'policy_kwargs': dict(net_arch=[256, 256]),
    # More samples per update for smoother gradients (4 envs * 4096 = 16384).
    'n_steps': 4096,
    'batch_size': 256,
    'n_epochs': 10,
    'gae_lambda': 0.95,
    'gamma': 0.99,
}
agent = PPO('MlpPolicy', vec_env, **rl_kwargs)

# Match run_rl_sb3.py PPO cadence: log_save_interval * 10 * 50.
num_steps_between_save = dedo_args.log_save_interval * 10 * 50
# Build a self-describing basename so eval mp4s on disk and in wandb media
# folders are identifiable without checking the surrounding metadata.
_sf = extra_args.success_factor
_sf_str = f'sf{_sf:g}' if _sf is not None else 'sf_off'
_video_basename = f'eval_{obs_mode}_{_sf_str}'
if extra_args.success_bonus:
    _video_basename += f'_sb{extra_args.success_bonus:g}'
if extra_args.fail_penalty:
    _video_basename += f'_fp{extra_args.fail_penalty:g}'
if extra_args.vel_penalty:
    _video_basename += f'_vp{extra_args.vel_penalty:g}'
if extra_args.boundary_penalty:
    _video_basename += f'_bp{extra_args.boundary_penalty:g}'
if extra_args.z_overshoot_penalty:
    _video_basename += (f'_zp{extra_args.z_overshoot_penalty:g}'
                        f's{extra_args.z_overshoot_slack:g}')
_video_basename += f'_seed{extra_args.seed}'
video_cb = HangVideoCallback(eval_env, dedo_args.logdir, n_envs, dedo_args,
                             num_steps_between_save=num_steps_between_save,
                             viz=False, debug=False,
                             video_basename=_video_basename,
                             n_eval_episodes=extra_args.n_eval_episodes_during_training,
                             eval_seed_lock=extra_args.eval_seed_lock,
                             eval_seed=dedo_args.seed + 9999)
diag_cb = RewardDiagnosticsCallback(window=100)
cb = CallbackList([video_cb, diag_cb])

# Persist a self-describing config.json + push to wandb.config so future
# debugging never has to guess what reward this run optimized.
dump_run_config(extra_args, dedo_args, dedo_args.logdir,
                use_wandb=dedo_args.use_wandb)

# ---------------------------------------------------------------------------
# Behavior cloning pretrain on scripted demos (huge unlock for sparse-reward
# cloth tasks — random PPO exploration almost never threads the hanger).
# Demos use dedo's preset waypoint trajectories from `cloth/apron_0.obj`
# (HangProcCloth always uses the apron preset regardless of which procedural
# cloth was generated — it's a fixed-target trajectory).
# ---------------------------------------------------------------------------
def _collect_demo_rollouts(args, obs_mode_str, num_episodes,
                           only_success=False, max_attempt_factor=3,
                           save_dir=None):
    """Roll out the scripted hole-aware waypoint controller and return
    (obs, act) pairs.

    `num_episodes` is the **target number of demos kept**, NOT the number
    of attempts. We loop until we have collected that many demos that pass
    the keep criterion (any rollout if `only_success=False`; only those
    with `info['is_success']=1` if `only_success=True`). Build_traj
    failures and dropped failed-success demos do NOT count toward the
    target — they trigger a retry — so dataset size is identical across
    runs / seeds, regardless of how often the scripted controller misses.

    `max_attempt_factor` caps the retry budget at
    `num_episodes * max_attempt_factor` to avoid an infinite loop when
    the scripted controller's success rate is pathologically low. Hitting
    the cap prints a warning and returns whatever was collected.

    `save_dir`: if non-None, write each kept demo to
    `<save_dir>/demo_NNN.pkl` in the same payload format `record_demo.py`
    uses, so the demos can be (a) inspected post-hoc and (b) reused
    across seeds/algos via `--bc_demo_path <save_dir>` (which routes
    through `_load_manual_demos`). Saving is a side effect; it does not
    change the in-memory return value."""
    from dedo.demo_preset import build_traj, merge_traj
    from dedo.envs.deform_env import DeformEnv

    raw = gym.make(args.env, args=deepcopy(args))
    raw = RetryResetEnv(raw)
    raw = PrivilegedObsWrapper(raw, obs_mode=obs_mode_str,
                                success_factor=extra_args.success_factor,
                                    success_bonus=extra_args.success_bonus,
                                    fail_penalty=extra_args.fail_penalty,
                                    vel_penalty=extra_args.vel_penalty,
                                    boundary_penalty=extra_args.boundary_penalty,
                                    z_overshoot_penalty=extra_args.z_overshoot_penalty,
                                    z_overshoot_slack=extra_args.z_overshoot_slack)
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
        # Walk down to the underlying DeformEnv to read sim_freq for build_traj.
        underlying = raw
        while hasattr(underlying, 'env'):
            underlying = underlying.env
            if isinstance(underlying, DeformEnv):
                break
        ctrl_freq = args.sim_freq / args.sim_steps_per_action

        # Build hole-aware waypoints per episode.
        preset_wp = build_hole_aware_waypoints(underlying)
        if preset_wp is None:
            print(f'[BC] attempt {attempts} (kept {n_kept}/{target_kept}): '
                  f'no hole loop on cloth, retrying')
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

        last_action = np.zeros_like(traj[0])
        ep_obs, ep_act = [], []
        step, ep_rwd, ep_success = 0, 0.0, 0
        while True:
            act_unscaled = traj[step] if step < len(traj) else last_action
            # Normalize to [-1, 1] PPO action range.
            normalized = np.clip(act_unscaled / DeformEnv.MAX_ACT_VEL, -1.0, 1.0)
            ep_obs.append(np.asarray(obs, dtype=np.float32))
            ep_act.append(np.asarray(normalized, dtype=np.float32))
            obs, rwd, done, info = raw.step(normalized.astype(np.float32))
            ep_rwd += float(rwd)
            if 'is_success' in info:
                ep_success = max(ep_success, int(info['is_success']))
            if done:
                break
            step += 1
        if only_success and not ep_success:
            n_dropped += 1
            print(f'[BC] attempt {attempts} (kept {n_kept}/{target_kept})  '
                  f'len={len(ep_obs)}  rwd={ep_rwd:.1f}  success=0  '
                  f'(dropped, --bc_demos_only_success)')
            continue
        obs_buf.extend(ep_obs)
        act_buf.extend(ep_act)
        succeeded += ep_success
        n_kept += 1

        # Persist this demo using the same pkl schema record_demo.py
        # writes, so _load_manual_demos can read it back unchanged.
        # Scripted demos only know one obs mode at collection time; we
        # store it in the dict-of-modes container with a single entry.
        if save_dir is not None:
            demo_idx = n_kept - 1
            pkl_path = os.path.join(save_dir, f'demo_{demo_idx:03d}.pkl')
            payload = {
                'obs': {obs_mode_str: np.asarray(ep_obs, dtype=np.float32)},
                'acts': np.asarray(ep_act, dtype=np.float32),
                'reward': float(ep_rwd),
                'success': int(ep_success),
                'obs_modes': [obs_mode_str],
                'recorded_in': obs_mode_str,
                'success_factor': extra_args.success_factor,
                'len': len(ep_act),
                'source': 'scripted',
            }
            with open(pkl_path, 'wb') as f:
                pickle.dump(payload, f)

        print(f'[BC] attempt {attempts} (kept {n_kept}/{target_kept})  '
              f'len={len(ep_obs)}  rwd={ep_rwd:.1f}  success={ep_success}')

    raw.close()
    if n_kept < target_kept:
        print(f'[BC] WARNING: only collected {n_kept}/{target_kept} demos '
              f'after {attempts} attempts (cap={max_attempts}). Either '
              f'increase --bc_episodes, raise the attempt cap, or lower '
              f'--success_factor — the scripted controller is missing '
              f'too often on this cloth distribution.')
    elif only_success:
        print(f'[BC] kept {n_kept}/{target_kept} successful demos in '
              f'{attempts} attempts ({n_dropped} dropped, '
              f'~{n_kept/max(attempts,1):.0%} scripted success rate)')
    return (np.array(obs_buf, dtype=np.float32),
            np.array(act_buf, dtype=np.float32),
            succeeded)


def _bc_pretrain(agent, demo_obs, demo_acts, vec_normalize,
                 epochs, batch_size, lr):
    import torch.nn.functional as F
    # Seed VecNormalize obs_rms with demo obs so the policy trains on
    # normalized inputs that match what it'll see during PPO rollouts.
    vec_normalize.obs_rms.update(demo_obs)
    norm_obs = vec_normalize.normalize_obs(demo_obs).astype(np.float32)

    obs_t = torch.as_tensor(norm_obs, device=agent.device)
    act_t = torch.as_tensor(demo_acts, device=agent.device)

    policy = agent.policy
    optimizer = torch.optim.Adam(policy.parameters(), lr=lr)
    n = len(demo_obs)
    print(f'[BC] training on {n} (obs, action) pairs '
          f'for {epochs} epochs, batch={batch_size}')
    for epoch in range(epochs):
        perm = torch.randperm(n, device=agent.device)
        total_loss, count = 0.0, 0
        for i in range(0, n, batch_size):
            idx = perm[i:i + batch_size]
            o, a = obs_t[idx], act_t[idx]
            features = policy.extract_features(o)
            latent_pi, _ = policy.mlp_extractor(features)
            mean_a = policy.action_net(latent_pi)
            loss = F.mse_loss(mean_a, a)
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


def _load_manual_demos(demo_dir, obs_mode_str, only_success=False):
    """Load (obs, act) pairs from a directory of demo_NNN.pkl files written
    by record_demo.py. Pkls store all 3 obs modes; we pick the requested
    one. Falls back to legacy single-mode format with a strict mode match.

    `--bc_demos_only_success` filters by the per-pkl `success` flag, which
    was determined at record time using the recorder's own success_factor.
    If that doesn't match the training success_factor, the filter throws
    out the wrong demos. We emit a single aggregate warning when we see
    any mismatch; legacy pkls without a stored `success_factor` only get
    a soft 'unknown' note."""
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

        if 'success_factor' in d:
            if d['success_factor'] != sf_train:
                sf_mismatches += 1
        else:
            sf_unknown += 1

        if only_success and not d.get('success', 0):
            print(f'[BC] {fname}: success=0, skipping '
                  f'(--bc_demos_only_success)')
            continue

        # New format: d['obs'] is a dict {mode: array}.
        if isinstance(d.get('obs'), dict):
            if obs_mode_str not in d['obs']:
                print(f'[BC] {fname}: missing obs for mode '
                      f'{obs_mode_str!r}, skipping')
                continue
            obs_arr = d['obs'][obs_mode_str]
        # Legacy format: d['obs'] is a single array, d['obs_mode'] is the mode.
        else:
            if d.get('obs_mode') != obs_mode_str:
                print(f'[BC] {fname}: legacy demo recorded as '
                      f'{d.get("obs_mode")!r}, skipping (re-record with '
                      f'updated record_demo.py to share across modes)')
                continue
            obs_arr = d['obs']

        n_files += 1
        n_success_demos += int(d.get('success', 0))
        obs_buf.append(obs_arr)
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
              f'--bc_demos_only_success could keep/drop the wrong demos. '
              f'Re-record with the current --success_factor to align.')

    if not obs_buf:
        return (np.zeros((0, 0), dtype=np.float32),
                np.zeros((0, 0), dtype=np.float32),
                0, 0)
    return (np.concatenate(obs_buf, axis=0).astype(np.float32),
            np.concatenate(act_buf, axis=0).astype(np.float32),
            n_success_demos, n_files)


if extra_args.bc_demo_path:
    print(f'\n=== BC pretrain: loading manual demos from '
          f'{extra_args.bc_demo_path} ===')
    demo_obs, demo_acts, n_success, n_demos = _load_manual_demos(
        extra_args.bc_demo_path, obs_mode,
        only_success=extra_args.bc_demos_only_success)
    print(f'[BC] loaded {len(demo_obs)} (obs,act) pairs from '
          f'{n_demos} manual demos ({n_success} succeeded)')
elif extra_args.bc_episodes > 0:
    # Persist scripted demos under <logdir>/scripted_demos/ in the same
    # pkl format `record_demo.py` writes, so they're (a) inspectable
    # post-hoc and (b) re-usable on a future run via
    # `--bc_demo_path <logdir>/scripted_demos`. Negligible disk cost
    # (~100 KB per demo for hole_centroid mode).
    _scripted_demos_dir = os.path.join(dedo_args.logdir, 'scripted_demos')
    print(f'\n=== BC pretrain: collecting {extra_args.bc_episodes} '
          f'scripted demo rollouts '
          f'(only_success={extra_args.bc_demos_only_success}) ===')
    print(f'[BC] saving scripted demos to {_scripted_demos_dir}')
    demo_obs, demo_acts, n_success = _collect_demo_rollouts(
        dedo_args, obs_mode, extra_args.bc_episodes,
        only_success=extra_args.bc_demos_only_success,
        save_dir=_scripted_demos_dir)
    n_demos = extra_args.bc_episodes
    print(f'[BC] collected {len(demo_obs)} (obs,act) pairs from '
          f'{n_demos} demos ({n_success} succeeded)')
else:
    demo_obs = np.zeros((0, 0), dtype=np.float32)
    demo_acts = np.zeros((0, 0), dtype=np.float32)
    n_success, n_demos = 0, 0

if len(demo_obs) > 0:
    _bc_pretrain(agent, demo_obs, demo_acts, vec_env,
                 epochs=extra_args.bc_epochs,
                 batch_size=256,
                 lr=extra_args.bc_lr)
    if dedo_args.use_wandb:
        import wandb
        wandb.log({'bc/n_pairs': len(demo_obs),
                   'bc/n_success_demos': n_success,
                   'bc/n_demos': n_demos})
elif extra_args.bc_demo_path or extra_args.bc_episodes > 0:
    print('[BC] no usable demos found — skipping BC pretrain')

print('Start privileged RL training ...')
agent.learn(total_timesteps=extra_args.total_env_steps, callback=cb)

ckpt_path = os.path.join(dedo_args.logdir, 'agent.zip')
agent.save(ckpt_path)
# Persist VecNormalize stats so loading the checkpoint later produces the
# same normalized obs the policy was trained on.
vec_env.save(os.path.join(dedo_args.logdir, 'vec_normalize.pkl'))
pickle.dump(dedo_args, open(os.path.join(dedo_args.logdir, 'args.pkl'), 'wb'))
print(f'\nDone. Checkpoint: {ckpt_path}')

# ---------------------------------------------------------------------------
# Final validation rollouts: deterministic, log mean reward + success rate.
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
