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
from _helpers import (RetryResetEnv, build_hole_aware_waypoints,  # noqa: E402
                      build_run_name_suffix, probe_peak_demo_vel)
from _video_callback import HangVideoCallback  # noqa: E402
from _critic_warmup import PPOCriticWarmupCallback  # noqa: E402
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
parser.add_argument('--wandb_run_name', type=str, default=None,
                    help='Custom wandb run name prefix. Replaces wandb '
                         'auto-generated PPO_<TS>_<env>; the auto-suffix '
                         '(arch, lr, reward shape, etc.) is still '
                         'appended. Use this to set the descriptive '
                         '"H: BC-anchor on D-base" portion from the CLI '
                         'instead of editing in the wandb UI later.')
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
parser.add_argument('--max_act_vel', type=str, default=None,
                    help='Override DeformEnv.MAX_ACT_VEL (m/s). Dedo '
                         'default is 10.0, but the scripted controller '
                         'RMS velocity is ~0.3 m/s, so demo actions '
                         'normalized by 10 land at RMS 0.03 in [-1, 1] '
                         '— only ~3%% of the action space is used. '
                         'Pass "auto" to probe 3 cloths and pick MAX_ACT_VEL '
                         '= peak * 1.2 (recommended). Or pass a float; '
                         'CAUTION: a value below the trajectory peak '
                         '(typically 1-3 m/s in the lift/thread phase) '
                         'silently breaks demos because '
                         'clip(act/MAX_ACT_VEL, -1, 1) saturates and the '
                         'gripper can\'t keep up. Watch the [BC] traj '
                         'peak |vel| diagnostic on attempt 1. None '
                         '(default) leaves dedo unchanged.')
parser.add_argument('--net_arch', type=str, default='256,256',
                    help='Comma-separated MLP hidden sizes for the '
                         'shared trunk (used by both actor and critic). '
                         'Default "256,256". Try "512,512" (~4x params) '
                         'if BC mse plateaus too high — useful for '
                         '30-dim corners and can\'t hurt for 18-dim '
                         'hole_centroid. With 300 demos the param/data '
                         'ratio is ~1.2 at 256,256 vs ~5 at 512,512 — '
                         'risk of overfitting BC, mitigated by the '
                         'large epoch count.')
parser.add_argument('--ent_coef', type=float, default=0.0,
                    help='PPO entropy bonus coefficient. SB3 default is '
                         '0.0 (no bonus), which lets log_std collapse '
                         'toward 0 once it\'s unfrozen post-warmup. '
                         'When BC is on with --log_std_init negative, '
                         'set 0.005-0.02 to keep log_std from collapsing '
                         'and exploration intact while the actor is '
                         'being updated. 0 disables.')
parser.add_argument('--critic_warmup_rollouts', type=int, default=0,
                    help='Freeze the actor (mu head + log_std + actor '
                         'MLP trunk) for the first N PPO rollouts so '
                         'the value head can converge on noisy initial '
                         'returns BEFORE the actor moves. Prevents BC '
                         'erasure: with random V(s), the first 1-2 PPO '
                         'updates push mu in noise directions and erase '
                         'BC. Recommended 2 when BC is on; 0 (default) '
                         'disables the warmup. With n_steps=4096 num_envs=4, '
                         'one rollout = 16384 env steps, so 2 = ~32k '
                         'frozen-actor steps.')
parser.add_argument('--ppo_clip_range', type=float, default=0.2,
                    help='PPO probability-ratio clip. SB3 default 0.2 '
                         'allows up to 20%% per-step probability shift '
                         'every gradient update, and with n_epochs=10 '
                         'and a 16384-sample rollout that is 640 updates '
                         'per rollout — enough to walk the actor '
                         'arbitrarily far from BC even with low lr. Try '
                         '0.05-0.1 when BC is on and the post-warmup '
                         'erasure pattern (eval/success_rate falling '
                         'from BC level to ~0) shows up.')
parser.add_argument('--ppo_epochs', type=int, default=10,
                    help='PPO gradient epochs per rollout. SB3 default '
                         '10. Each epoch is a full pass over n_steps * '
                         'num_envs samples in batches of batch_size; '
                         'fewer epochs = fewer gradient steps per '
                         'rollout = less per-rollout drift from the '
                         'BC-warmstarted actor. Try 3-5 when BC erasure '
                         'is the failure mode.')
parser.add_argument('--ppo_target_kl', type=float, default=None,
                    help='PPO early-stop threshold on KL-to-previous-'
                         'policy. SB3 default None (no early stop). '
                         'When set, PPO terminates the n_epochs gradient '
                         'pass once the rolling-batch KL exceeds this '
                         'value — so a single rollout cannot move the '
                         'actor more than ~target_kl away. Soft '
                         'complement to clip_range; recommended 0.01-0.03 '
                         'when BC is on. None disables.')
parser.add_argument('--bc_anchor_batches', type=int, default=0,
                    help='Number of MSE gradient batches against the BC '
                         'demo dataset to take after each PPO rollout. '
                         'Counters the BC-erasure pattern by anchoring '
                         'the actor toward the demos throughout PPO '
                         'training (not just at warmstart). 0 disables. '
                         'Try 4-8 when post-BC eval success drops below '
                         'the BC-pretrain level. No extra env interaction '
                         '— reuses the in-memory demos.')
parser.add_argument('--bc_anchor_batch_size', type=int, default=256,
                    help='Batch size for BC anchor gradient steps.')
parser.add_argument('--bc_anchor_lr', type=float, default=1e-4,
                    help='Learning rate for BC anchor gradient steps. '
                         'Smaller than --bc_lr (BC pretrain default 1e-3) '
                         'because anchor updates run continuously through '
                         'PPO and should not overpower the policy gradient.')
parser.add_argument('--critic_warmup_demo_epochs', type=int, default=0,
                    help='Pretrain V(s) on Monte Carlo returns from BC '
                         'demos for N epochs after BC pretrain, before '
                         'PPO starts. Targets the structural cause of '
                         '"PPO erases BC": with a random V at PPO start, '
                         'A=Q-V is noise and the actor is pushed off the '
                         'BC manifold. Seeding V with demo MC returns '
                         'aligns the critic with BC-visited states from '
                         'step 0. 0 disables; try 50-100. Requires demos '
                         'with per-step rewards (collect fresh via '
                         '--bc_episodes, or use a demo dir whose pkls '
                         'contain the `rewards` field).')
parser.add_argument('--critic_warmup_demo_lr', type=float, default=3e-4,
                    help='Learning rate for demo-based critic warmup.')
parser.add_argument('--critic_warmup_demo_batch_size', type=int, default=256,
                    help='Batch size for demo-based critic warmup.')
parser.add_argument('--log_std_init', type=float, default=None,
                    help='Initial log_std for the PPO actor head. SB3 '
                         'default is 0.0 (std=1.0 in pre-clip space). '
                         'Scripted demos here have |action| ≈ 0.03 in '
                         'normalized [-1, 1] space (waypoint vels ~0.3 '
                         'm/s ÷ MAX_ACT_VEL=10), so default noise std=1.0 '
                         'completely drowns out BC-trained mu at rollout '
                         'time: action = mu + N(0, 1) ≈ N(0, 1) clipped, '
                         'i.e. uniform-ish [-1, 1]. Set to -3.5 (std≈0.030) '
                         'to match demo magnitude so BC is visible from '
                         'step 0; -3.0 (std≈0.050) for slightly more '
                         'exploration. None (default) keeps SB3 default. '
                         'Skipped on resume (saved policy already has its '
                         'own learned log_std).')
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
parser.add_argument('--action_penalty', type=float, default=0.0,
                    help='Per-step penalty on action magnitude. '
                         'reward -= action_penalty * mean(action**2). '
                         'Action is in [-1, 1]^6 so mean(a**2) is in '
                         '[0, 1]; coefs ~0.1-2 give per-step penalties '
                         'comparable to vel_penalty. Discourages bang-'
                         'bang/flailing control. Applied every step '
                         'including terminal. 0 = off. Applied '
                         'identically to training, eval, and BC scripted-'
                         'demo collection envs so all three see the '
                         'same reward.')
parser.add_argument('--pre_settle_coef', type=float, default=0.0,
                    help='Linear penalty on hole-to-goal distance (m) at '
                         'policy handoff, BEFORE the gravity settle. '
                         'reward -= pre_settle_coef * pre_settle_dist_m. '
                         'Counters the "lift cloth high, let it drop onto '
                         'the hanger" exploit: post-settle reward has '
                         'effective coef ~20/m, so a coef of 20 equal-'
                         'weights threading-by-control vs ballistic drop. '
                         '0 = off; start at 20.')
parser.add_argument('--dist_reward_coef', type=float, default=0.0,
                    help='Per-step DENSE distance reward, fires every '
                         'step. reward += coef / (1 + adaptive_dist). '
                         'Repairs the credit-assignment problem where '
                         'legacy reward concentrates ~85%% of episode '
                         'return at the terminal step (after the policy '
                         'stops acting) — gives PPO a continuous "you '
                         'are getting closer" signal. Suggested 0.5–2.0 '
                         '(cumulative-over-200-steps comparable to '
                         'success_bonus). 0 = off (default; preserves '
                         'legacy reward).')
parser.add_argument('--threading_bonus_coef', type=float, default=0.0,
                    help='Per-step bonus when adaptive_dist < threshold '
                         '(cloth on hole). CAVEAT: detection inherits '
                         'the same flakiness as the terminal success '
                         'check (centroid-distance, not topological), '
                         'so prefer dist_reward_coef alone for the '
                         'first denser-reward experiments. 0 = off '
                         '(default).')
parser.add_argument('--final_reward_mult', type=float, default=None,
                    help='Override DeformEnv.FINAL_REWARD_MULT (default '
                         '400). The terminal-step base reward scales as '
                         'final_reward_mult * dist, so 400 makes the '
                         'terminal step ~70%% of episode reward range '
                         'and crushes the per-step dist_reward signal '
                         'whenever the cloth ends up far from hole. '
                         'Reducing to 50–100 equalizes the per-step '
                         'and terminal magnitudes (50 ~ comparable to '
                         'a 200-step dist_reward sum at coef=1). None '
                         '= keep dedo default of 400.')
parser.add_argument('--success_metric', type=str, default='hanging',
                    choices=['hanging', 'topological', 'legacy'],
                    help='Criterion for info["is_success"] / reward '
                         'success_bonus / BC demo filtering. '
                         '"hanging" (default, recommended): three 3D '
                         'checks — hole-centroid xy near peg, some hole '
                         'vertex below peg tip, hole has vertical '
                         'extent. Robust to collapsed cloth. '
                         '"topological": winding number |w|>=0.5 around '
                         'peg axis. Mathematically clean but degenerates '
                         'on collapsed cloth (hole vertices align along '
                         'a vertical line through peg, xy projection '
                         'becomes a line/point). '
                         '"legacy": hole-centroid 3D distance to peg '
                         'tip < success_factor * hole_radius. Has '
                         'documented false negatives (cloth hangs below '
                         'tip) and false positives (cloth lands beside '
                         'peg). All three are computed and emitted to '
                         'wandb for comparison (rwd_diag/success/'
                         '{legacy,topological,hanging}_rate, plus '
                         'pairwise disagreement rates).')
parser.add_argument('--cpu', action='store_true',
                    help='Force CPU even if CUDA is available. Often faster '
                         'for the small MLP over privileged obs since GPU '
                         'kernel-launch overhead dominates.')
parser.add_argument('--max_episode_len', type=int, default=200,
                    help='Steps per episode before timeout-done fires. '
                         'dedo default is 200; lower (e.g. 100) cuts '
                         'wall-clock and avoids dithering after the cloth '
                         'has already reached the pole region.')
parser.add_argument('--load_checkpoint', type=str, default=None,
                    help='Path to a previous run logdir (containing '
                         'agent.zip and vec_normalize.pkl) to resume '
                         'training from. Skips BC pretrain (already baked '
                         'into the saved policy). --total_env_steps is the '
                         'TARGET total — pass the same number you would '
                         'for a fresh run; SAC continues until it reaches '
                         'that target. A NEW wandb run is started, tagged '
                         '`resumed_from=<orig>` so you can group origin + '
                         'resume in the wandb UI.')
extra_args, remaining = parser.parse_known_args()

# `--no_adaptive_success` is the single switch for "use dedo's base reward
# and success unchanged". The wrapper treats success_factor=None as the
# disable signal, so we flip it here once for all downstream consumers
# (env factory, eval env, demo collector, banner, wandb tags).
if extra_args.no_adaptive_success:
    extra_args.success_factor = None

# Parse net_arch up front so it's available when wandb.run.name is
# built (which happens before policy construction below).
_net_arch_list = [int(x) for x in extra_args.net_arch.split(',') if x.strip()]

# Resume-time auto-restore of MAX_ACT_VEL. If the user is resuming from
# a previous run's checkpoint and didn't explicitly pass --max_act_vel,
# read it back from the checkpoint's config.json. Without this, a
# resume run silently falls back to dedo's default 10.0 while the
# original run might have been at e.g. 4.1 — action scales diverge
# between training and continuation rollouts, eval becomes inconsistent
# with the saved policy. An explicit user-supplied --max_act_vel still
# takes precedence (overriding the saved value is sometimes intentional,
# e.g. for sensitivity studies).
if extra_args.load_checkpoint and extra_args.max_act_vel is None:
    import json as _json_resume
    _ckpt_cfg = os.path.join(extra_args.load_checkpoint, 'config.json')
    if os.path.exists(_ckpt_cfg):
        try:
            with open(_ckpt_cfg) as _f:
                _saved_cfg = _json_resume.load(_f)
            # `dump_run_config` writes the active runtime MAX_ACT_VEL
            # under reward_def.dedo_max_act_vel (already resolved from
            # 'auto' or float to a concrete number). The CLI raw value
            # at extra.max_act_vel may be 'auto' or None, so we prefer
            # the resolved class-attribute snapshot.
            _saved_mav = (_saved_cfg.get('reward_def', {})
                          .get('dedo_max_act_vel'))
            if (_saved_mav is not None
                    and abs(float(_saved_mav) - 10.0) > 1e-6):
                extra_args.max_act_vel = str(float(_saved_mav))
                print(f'[resume] auto-restored '
                      f'--max_act_vel={extra_args.max_act_vel} '
                      f'from {_ckpt_cfg}. Pass --max_act_vel explicitly '
                      f'to override.')
            else:
                print(f'[resume] config.json shows MAX_ACT_VEL was at '
                      f'dedo default ({_saved_mav}); no auto-restore '
                      f'needed.')
        except (ValueError, KeyError, OSError) as _e:
            print(f'[resume] WARN: could not read max_act_vel from '
                  f'{_ckpt_cfg}: {_e!r}. MAX_ACT_VEL stays at dedo '
                  f'default — pass --max_act_vel manually if the '
                  f'original run used a non-default value.')
    else:
        print(f'[resume] WARN: no config.json at {_ckpt_cfg}; '
              f'cannot auto-restore MAX_ACT_VEL. Pass --max_act_vel '
              f'explicitly if the original run used a non-default value.')

# Parse the max_act_vel knob into a tag (None | 'auto' | float). Explicit
# floats patch DeformEnv.MAX_ACT_VEL immediately; 'auto' defers until after
# dedo_args is built so we can run a probe through real cloth resets. Reads
# of DeformEnv.MAX_ACT_VEL are dynamic (unscale_vel and the demo collector
# look it up per call), so a single class-attribute write propagates
# everywhere consistently.
_mav_arg = extra_args.max_act_vel
if _mav_arg is None or str(_mav_arg).lower() in ('none', ''):
    _mav_mode = None
elif str(_mav_arg).lower() == 'auto':
    _mav_mode = 'auto'
else:
    _mav_mode = float(_mav_arg)
    from dedo.envs.deform_env import DeformEnv as _DeformEnvForPatch
    _orig_max_act_vel = _DeformEnvForPatch.MAX_ACT_VEL
    _DeformEnvForPatch.MAX_ACT_VEL = _mav_mode
    print(f'[init] DeformEnv.MAX_ACT_VEL: {_orig_max_act_vel} -> '
          f'{_DeformEnvForPatch.MAX_ACT_VEL}')

# Patch DeformEnv.FINAL_REWARD_MULT if requested. Same class-attribute
# pattern as MAX_ACT_VEL — reads in dedo's get_reward() are dynamic, so
# a single write propagates everywhere. Default 400 makes the terminal
# step ~70% of episode reward range; reducing to 50-100 equalizes
# per-step (dist_reward) and terminal magnitudes for healthier credit
# assignment.
if extra_args.final_reward_mult is not None:
    from dedo.envs.deform_env import DeformEnv as _DeformEnvForPatch
    _orig_final_mult = _DeformEnvForPatch.FINAL_REWARD_MULT
    _DeformEnvForPatch.FINAL_REWARD_MULT = float(extra_args.final_reward_mult)
    print(f'[init] DeformEnv.FINAL_REWARD_MULT: {_orig_final_mult} -> '
          f'{_DeformEnvForPatch.FINAL_REWARD_MULT}')

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


# Auto-probe MAX_ACT_VEL if requested. Builds 3 cloths via the real
# pipeline, runs build_traj on each, takes max |velocity|, then sizes
# MAX_ACT_VEL = peak * 1.2 (rounded up to 0.1 m/s). This is the only
# value that's both small enough to make demo |a| RMS meaningful in
# [-1, 1] (improves BC SNR) and large enough to not clip the lift/thread
# phase (which would silently break demos).
if _mav_mode == 'auto':
    print('[init] probing scripted-demo peak velocity (3 cloths)...')
    _peak = probe_peak_demo_vel(dedo_args, n_probes=3)
    if _peak is None:
        print('[init] WARN: all probes failed; leaving MAX_ACT_VEL at '
              'dedo default (10.0). BC will be noisy, but demos won\'t '
              'saturate.')
    else:
        _new_mav = float(np.ceil(_peak * 1.2 * 10) / 10)
        from dedo.envs.deform_env import DeformEnv as _DeformEnvForPatch
        _orig = _DeformEnvForPatch.MAX_ACT_VEL
        _DeformEnvForPatch.MAX_ACT_VEL = _new_mav
        print(f'[init] DeformEnv.MAX_ACT_VEL: {_orig} -> {_new_mav} '
              f'(peak demo |vel| = {_peak:.3f} m/s × 1.2 safety, '
              f'rounded up to 0.1)')


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
                                    action_penalty=extra_args.action_penalty,
                                    pre_settle_coef=extra_args.pre_settle_coef,
                                    dist_reward_coef=extra_args.dist_reward_coef,
                                    threading_bonus_coef=extra_args.threading_bonus_coef,
                                    success_metric=extra_args.success_metric)
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
                                    action_penalty=extra_args.action_penalty,
                                    pre_settle_coef=extra_args.pre_settle_coef,
                                    dist_reward_coef=extra_args.dist_reward_coef,
                                    threading_bonus_coef=extra_args.threading_bonus_coef,
                                    success_metric=extra_args.success_metric)
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

# Build the wandb run name. build_run_name_suffix encodes ALL
# experimental dials (lr, critic warmup, BC anchor, demo-V warmup, PPO
# clip/epochs/target_kl, log_std_init, BC budget, reward shape) so a
# crashed run can be identified from the name alone — no config.json
# hunt required. Defaults are omitted to keep names short on baselines.
if dedo_args.use_wandb:
    import wandb
    if wandb.run is not None:
        sf = extra_args.success_factor
        sb = extra_args.success_bonus
        fp = extra_args.fail_penalty
        vp = extra_args.vel_penalty
        ap = extra_args.action_penalty
        psc = extra_args.pre_settle_coef
        _suffix = build_run_name_suffix(
            extra_args, algo='PPO', obs_kind=obs_mode,
            net_arch=_net_arch_list)
        if extra_args.wandb_run_name:
            wandb.run.name = extra_args.wandb_run_name + _suffix
        else:
            wandb.run.name = wandb.run.name + _suffix
        wandb.run.tags = list(wandb.run.tags or []) + [
            f'success_factor={sf if sf is not None else "default"}',
            f'success_bonus={sb}',
            f'fail_penalty={fp}',
            f'vel_penalty={vp}',
            f'action_penalty={ap}',
            f'pre_settle_coef={psc}',
            f'obs_mode={obs_mode}',
        ]

_policy_kwargs = dict(net_arch=_net_arch_list)
print(f'[init] policy net_arch = {_net_arch_list}')
if extra_args.log_std_init is not None:
    # PPO's DiagGaussianDistribution uses log_std as an nn.Parameter
    # initialized to log_std_init. Lowering it below 0 is essential here:
    # demo |action| ≈ 0.03 in normalized space, so default std=1.0 makes
    # noise overwhelm BC's mu by ~30x and rollouts revert to ~uniform.
    _policy_kwargs['log_std_init'] = float(extra_args.log_std_init)
rl_kwargs = {
    'learning_rate': dedo_args.lr,
    'device': dedo_args.device,
    'tensorboard_log': dedo_args.logdir,
    'verbose': 1,
    # Bigger network — defaults [64, 64] are too small, especially for
    # full_mesh (~762-dim).
    'policy_kwargs': _policy_kwargs,
    # More samples per update for smoother gradients (4 envs * 4096 = 16384).
    'n_steps': 4096,
    'batch_size': 256,
    'n_epochs': int(extra_args.ppo_epochs),
    'gae_lambda': 0.95,
    'gamma': 0.99,
    'ent_coef': float(extra_args.ent_coef),
    'clip_range': float(extra_args.ppo_clip_range),
    'target_kl': (None if extra_args.ppo_target_kl is None
                  else float(extra_args.ppo_target_kl)),
}
_resuming = bool(extra_args.load_checkpoint)
if _resuming:
    ckpt_dir = extra_args.load_checkpoint
    if not os.path.isdir(ckpt_dir):
        raise FileNotFoundError(
            f'--load_checkpoint dir does not exist: {ckpt_dir}')
    agent_path = os.path.join(ckpt_dir, 'agent.zip')
    vn_path = os.path.join(ckpt_dir, 'vec_normalize.pkl')
    if not os.path.exists(agent_path):
        raise FileNotFoundError(f'no agent.zip in {ckpt_dir}')
    # Restore VecNormalize stats IN PLACE before PPO.load so the loaded
    # policy sees normalized obs at the same scale it was trained against.
    if os.path.exists(vn_path):
        _loaded_vn = VecNormalize.load(vn_path, vec_env.venv)
        vec_env.obs_rms = _loaded_vn.obs_rms
        vec_env.ret_rms = _loaded_vn.ret_rms
        vec_env.training = True
        print(f'[resume] loaded VecNormalize stats from {vn_path}')
    else:
        print(f'[resume] WARN: no vec_normalize.pkl at {vn_path}; '
              f'using fresh obs/ret rms (policy will see drifted obs scale)')
    agent = PPO.load(
        agent_path, env=vec_env, device=dedo_args.device,
        tensorboard_log=dedo_args.logdir,
        custom_objects={
            'learning_rate': dedo_args.lr,
            'lr_schedule': lambda _progress: dedo_args.lr,
        })
    # SB3 caches a stale _last_obs across save/load; clearing it forces a
    # fresh env.reset() on entry to learn().
    agent._last_obs = None
    print(f'[resume] loaded PPO policy from {agent_path}; '
          f'num_timesteps={agent.num_timesteps:,}')
    if dedo_args.use_wandb:
        try:
            import wandb
            if wandb.run is not None:
                _orig_name = os.path.basename(ckpt_dir.rstrip('/'))
                wandb.run.tags = list(wandb.run.tags or []) + [
                    f'resumed_from={_orig_name}',
                    f'resume_step={agent.num_timesteps}',
                ]
        except Exception:
            pass
else:
    agent = PPO('MlpPolicy', vec_env, **rl_kwargs)
    if extra_args.log_std_init is not None:
        _init_std = float(np.exp(extra_args.log_std_init))
        print(f'[init] PPO log_std_init = {extra_args.log_std_init} '
              f'(std ≈ {_init_std:.3f}). Pick this comparable to the '
              f'demo |a| RMS so BC mu is visible at deploy; see the '
              f'[BC] traj peak |vel| diagnostic during demo collection '
              f'to estimate it (RMS ≈ peak / 5–10).')

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
if extra_args.action_penalty:
    _video_basename += f'_ap{extra_args.action_penalty:g}'
if extra_args.pre_settle_coef:
    _video_basename += f'_psc{extra_args.pre_settle_coef:g}'
_video_basename += f'_seed{extra_args.seed}'
video_cb = HangVideoCallback(eval_env, dedo_args.logdir, n_envs, dedo_args,
                             num_steps_between_save=num_steps_between_save,
                             viz=False, debug=False,
                             video_basename=_video_basename,
                             n_eval_episodes=extra_args.n_eval_episodes_during_training,
                             eval_seed_lock=extra_args.eval_seed_lock,
                             eval_seed=dedo_args.seed + 9999)
diag_cb = RewardDiagnosticsCallback(window=100)
_cbs = [video_cb, diag_cb]
# Critic warmup: only on a fresh run (saved checkpoints already have a
# trained critic; warmup would needlessly freeze the actor again).
if not _resuming and extra_args.critic_warmup_rollouts > 0:
    _cbs.append(PPOCriticWarmupCallback(
        n_warmup_rollouts=extra_args.critic_warmup_rollouts))
# CallbackList is built AFTER BC pretrain below, so the BC anchor
# callback can be appended to _cbs once demo data is in scope.

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
                                    action_penalty=extra_args.action_penalty,
                                    pre_settle_coef=extra_args.pre_settle_coef,
                                    dist_reward_coef=extra_args.dist_reward_coef,
                                    threading_bonus_coef=extra_args.threading_bonus_coef,
                                    success_metric=extra_args.success_metric)
    raw.seed(args.seed + 1000)

    if save_dir is not None:
        os.makedirs(save_dir, exist_ok=True)

    obs_buf, act_buf = [], []
    rewards_per_ep = []  # one np.ndarray per kept demo (per-step raw rewards)
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

        # Diagnostic: peak waypoint velocity vs the active MAX_ACT_VEL.
        # If peak > MAX_ACT_VEL, the action `np.clip(traj/MAX_ACT_VEL, -1, 1)`
        # saturates and the gripper can't keep up with the trajectory,
        # which silently turns "scripted demo" into "scripted demo that
        # always fails because the lift phase moves at the cap, not the
        # required speed". Print on first attempt only.
        if attempts == 1:
            _peak = float(np.abs(traj).max())
            _mav = float(DeformEnv.MAX_ACT_VEL)
            _flag = ' <- TOO LOW, demos will saturate' if _peak > _mav else ''
            print(f'[BC] traj peak |vel| = {_peak:.3f} m/s; '
                  f'MAX_ACT_VEL = {_mav:.3f} m/s{_flag}')

        last_action = np.zeros_like(traj[0])
        ep_obs, ep_act, ep_rewards = [], [], []
        step, ep_rwd, ep_success = 0, 0.0, 0
        while True:
            act_unscaled = traj[step] if step < len(traj) else last_action
            # Normalize to [-1, 1] PPO action range.
            normalized = np.clip(act_unscaled / DeformEnv.MAX_ACT_VEL, -1.0, 1.0)
            ep_obs.append(np.asarray(obs, dtype=np.float32))
            ep_act.append(np.asarray(normalized, dtype=np.float32))
            obs, rwd, done, info = raw.step(normalized.astype(np.float32))
            ep_rewards.append(float(rwd))
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
        rewards_per_ep.append(np.asarray(ep_rewards, dtype=np.float32))
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
                # Per-step rewards (added so demo-based critic warmup
                # can compute MC returns). 'reward' (scalar episode
                # total) kept for backward compat with record_demo.py.
                'rewards': np.asarray(ep_rewards, dtype=np.float32),
                'reward': float(ep_rwd),
                'success': int(ep_success),
                'obs_modes': [obs_mode_str],
                'recorded_in': obs_mode_str,
                'success_factor': extra_args.success_factor,
                # Action-scale parity invariant. Demos store actions in
                # `clip(traj / MAX_ACT_VEL, -1, 1)` — so loading these
                # demos in a future run with a different MAX_ACT_VEL
                # silently mistrains. _load_manual_demos refuses to
                # proceed when this field disagrees with the active
                # DeformEnv.MAX_ACT_VEL.
                'max_act_vel': float(DeformEnv.MAX_ACT_VEL),
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
            rewards_per_ep,
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
    from dedo.envs.deform_env import DeformEnv as _DeformEnvForCheck
    obs_buf, act_buf = [], []
    rewards_per_ep = []  # filled if pkls have 'rewards' field
    n_files, n_success_demos = 0, 0
    n_pkls_missing_rewards = 0
    sf_train = extra_args.success_factor
    sf_mismatches, sf_unknown = 0, 0
    # MAX_ACT_VEL parity: demos store actions in clip(traj / MAX_ACT_VEL,
    # -1, 1). A training-time MAX_ACT_VEL different from the demos' value
    # means the policy learns the wrong action scale — silent and
    # catastrophic for BC. Track per-pkl values and refuse to proceed
    # when any disagrees with the active training-time class attribute.
    mav_train = float(_DeformEnvForCheck.MAX_ACT_VEL)
    mav_per_pkl = []  # (fname, recorded_mav_or_None) for the post-loop check
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

        # Track MAX_ACT_VEL — checked after the loop so we can emit a
        # single aggregate error rather than spamming per-pkl.
        mav_per_pkl.append((fname, float(d['max_act_vel'])
                            if 'max_act_vel' in d else None))

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
        # Per-step rewards: present in pkls written by the updated
        # _collect_demo_rollouts; absent in pre-update pkls and in
        # legacy record_demo.py output. Critic-warmup-on-demos
        # requires this field.
        if 'rewards' in d:
            rewards_per_ep.append(np.asarray(d['rewards'], dtype=np.float32))
        else:
            n_pkls_missing_rewards += 1
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

    # MAX_ACT_VEL parity enforcement (strict, fails-loud).
    mav_recorded = sorted({m for _, m in mav_per_pkl if m is not None})
    mav_missing = [fn for fn, m in mav_per_pkl if m is None]
    mav_mismatches = [fn for fn, m in mav_per_pkl
                      if m is not None and abs(m - mav_train) > 1e-6]
    if mav_mismatches:
        # Sample a few filenames for the error message so the user can
        # immediately spot-check.
        _examples = ', '.join(mav_mismatches[:3])
        _extra = (f' (+{len(mav_mismatches) - 3} more)'
                  if len(mav_mismatches) > 3 else '')
        raise RuntimeError(
            f'[BC] action-scale mismatch: training MAX_ACT_VEL={mav_train} '
            f'but {len(mav_mismatches)}/{len(mav_per_pkl)} demo pkl(s) '
            f'were recorded under different value(s) {mav_recorded}. '
            f'Demo actions are stored as clip(traj / MAX_ACT_VEL, -1, 1); '
            f'loading them at a different MAX_ACT_VEL would silently '
            f'mistrain the policy (BC pulls actor toward wrong absolute '
            f'velocities). Example mismatched files: {_examples}{_extra}. '
            f'Re-pass --max_act_vel={mav_recorded[0]} to match the demos, '
            f'or re-record them under the current MAX_ACT_VEL.')
    if mav_missing:
        print(f'[BC] WARNING: {len(mav_missing)} demo pkl(s) lack a '
              f'recorded `max_act_vel` field (legacy pkls). Cannot '
              f'verify action-scale parity. Training will proceed under '
              f'the assumption that they were recorded at MAX_ACT_VEL='
              f'{mav_train} (the current setting). If that\'s wrong, '
              f'BC will silently fail. Re-record under current settings '
              f'to remove this warning.')
    if n_pkls_missing_rewards > 0:
        print(f'[BC] NOTE: {n_pkls_missing_rewards} demo pkl(s) lack a '
              f'per-step `rewards` field. Demo-based critic warmup '
              f'(--critic_warmup_demo_epochs) cannot be used with these '
              f'demos; re-collect to enable.')

    # If any pkls were missing the rewards field, drop the partial list
    # (caller treats empty list as "rewards unavailable") so we don't
    # silently train V on a rewards subset that misaligns with obs/acts.
    if n_pkls_missing_rewards > 0:
        rewards_per_ep = []

    if not obs_buf:
        return (np.zeros((0, 0), dtype=np.float32),
                np.zeros((0, 0), dtype=np.float32),
                [], 0, 0)
    return (np.concatenate(obs_buf, axis=0).astype(np.float32),
            np.concatenate(act_buf, axis=0).astype(np.float32),
            rewards_per_ep,
            n_success_demos, n_files)


if _resuming:
    # Resuming an existing run — the saved policy already encodes BC +
    # however many PPO steps were applied. Re-running BC would clobber
    # learned weights.
    demo_obs = np.zeros((0, 0), dtype=np.float32)
    demo_acts = np.zeros((0, 0), dtype=np.float32)
    demo_rewards_per_ep = []
    n_success, n_demos = 0, 0
    print('[resume] skipping BC pretrain (policy already trained)')
elif extra_args.bc_demo_path:
    print(f'\n=== BC pretrain: loading manual demos from '
          f'{extra_args.bc_demo_path} ===')
    (demo_obs, demo_acts, demo_rewards_per_ep,
     n_success, n_demos) = _load_manual_demos(
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
    demo_obs, demo_acts, demo_rewards_per_ep, n_success = (
        _collect_demo_rollouts(
            dedo_args, obs_mode, extra_args.bc_episodes,
            only_success=extra_args.bc_demos_only_success,
            save_dir=_scripted_demos_dir))
    n_demos = extra_args.bc_episodes
    print(f'[BC] collected {len(demo_obs)} (obs,act) pairs from '
          f'{n_demos} demos ({n_success} succeeded)')
else:
    demo_obs = np.zeros((0, 0), dtype=np.float32)
    demo_acts = np.zeros((0, 0), dtype=np.float32)
    demo_rewards_per_ep = []
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

# Demo-based critic warmup: pretrain V(s) on Monte Carlo returns from
# the BC demos before PPO starts. Targets the root cause of post-BC
# erasure (random V => noisy advantages => actor pushed off BC).
if (not _resuming and extra_args.critic_warmup_demo_epochs > 0
        and len(demo_obs) > 0):
    if not demo_rewards_per_ep:
        print('[critic-warmup] SKIPPED: per-step rewards unavailable. '
              'Either re-collect demos via --bc_episodes (saves rewards '
              'in the new pkl format) or use a demo dir whose pkls '
              'contain the `rewards` field.')
    else:
        from _critic_warmup_demos import critic_warmup_on_demos
        _cw_stats = critic_warmup_on_demos(
            agent, demo_obs, demo_rewards_per_ep, vec_env,
            gamma=rl_kwargs['gamma'],
            epochs=extra_args.critic_warmup_demo_epochs,
            batch_size=extra_args.critic_warmup_demo_batch_size,
            lr=extra_args.critic_warmup_demo_lr,
            use_wandb=dedo_args.use_wandb)
        if dedo_args.use_wandb:
            import wandb
            wandb.log({'critic_warmup/final_mse': _cw_stats['final_mse'],
                       'critic_warmup/target_mean': _cw_stats['target_mean'],
                       'critic_warmup/target_std': _cw_stats['target_std'],
                       'critic_warmup/ret_rms_var': _cw_stats['ret_rms_var']})

# BC anchor: continuously pull the actor toward the BC demos during
# PPO. Only meaningful when we have demos in memory (skipped on
# resume since demos aren't reloaded there). See _bc_anchor.py.
if (not _resuming and extra_args.bc_anchor_batches > 0
        and len(demo_obs) > 0):
    from _bc_anchor import BCAnchorCallback
    _cbs.append(BCAnchorCallback(
        demo_obs=demo_obs,
        demo_acts=demo_acts,
        vec_normalize=vec_env,
        n_batches=extra_args.bc_anchor_batches,
        batch_size=extra_args.bc_anchor_batch_size,
        lr=extra_args.bc_anchor_lr,
        verbose=1))
    print(f'[bc-anchor] enabled: {extra_args.bc_anchor_batches} '
          f'batches/rollout, batch_size={extra_args.bc_anchor_batch_size}, '
          f'lr={extra_args.bc_anchor_lr}')

cb = CallbackList(_cbs)

print(f'Start privileged RL training '
      f'({"resuming" if _resuming else "fresh"}; '
      f'num_timesteps={agent.num_timesteps:,}; '
      f'target {extra_args.total_env_steps:,}) ...')
agent.learn(total_timesteps=extra_args.total_env_steps, callback=cb,
            reset_num_timesteps=not _resuming)

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
