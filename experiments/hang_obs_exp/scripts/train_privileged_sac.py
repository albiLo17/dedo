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
from _helpers import (RetryResetEnv, build_hole_aware_waypoints,  # noqa: E402
                      build_run_name_suffix, probe_peak_demo_vel)

import dedo  # registers gym envs
from dedo.utils.args import get_args_parser, args_postprocess
from dedo.utils.train_utils import init_train
from stable_baselines3.common.callbacks import CallbackList
from _video_callback import HangVideoCallback  # noqa: E402
from _critic_warmup import SACCriticWarmupCallback  # noqa: E402
from _reward_diagnostics import (  # noqa: E402
    RewardDiagnosticsCallback, dump_run_config, make_final_eval_collector,
    log_final_eval_metrics)

from experiments.hang_obs_exp.envs.privileged_env import PrivilegedObsWrapper


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
parser = argparse.ArgumentParser(add_help=False)
parser.add_argument('--obs_mode', type=str, default='hole_centroid',
                    choices=['hole_centroid', 'hole_centroid_corners',
                             'hole_vertices', 'full_mesh'])
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
parser.add_argument('--wandb_run_name', type=str, default=None,
                    help='Custom wandb run name prefix; auto-suffix is '
                         'still appended. See train_privileged.py for '
                         'detail.')
parser.add_argument('--n_final_eval_episodes', type=int, default=50,
                    help='Deterministic eval episodes at end of '
                         'training. n=50 → SE≈0.07 at p=0.5; cheap '
                         '(end-of-run only) and gives a reliable '
                         'summary number for cross-run comparison.')
parser.add_argument('--n_eval_episodes_during_training', type=int, default=30,
                    help='Deterministic eval episodes per in-training '
                         'eval pass. Drives the SE of '
                         'eval/success_rate (n=30 → SE≈0.09 at p=0.5; '
                         'legacy n=10 → SE≈0.16 was most of the chart '
                         'noise).')
parser.add_argument('--eval_seed_lock', dest='eval_seed_lock',
                    action='store_true', default=True,
                    help='Default. Re-seed the eval env to '
                         '`seed + 9999` at the start of every eval '
                         'pass so the same N procedural cloths are '
                         'evaluated every checkpoint — much cleaner '
                         'cross-run trace. Generalization signal '
                         'still comes from the final eval (different '
                         'seed offset, --n_final_eval_episodes). Pass '
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
                         'Or float like "0.1". NOTE: with large terminal '
                         'rewards (success_bonus + FINAL_REWARD_MULT) auto '
                         'reliably collapses ent_coef toward 0 within '
                         '~50–200k steps, killing exploration. Pinning it '
                         'at 0.1–0.2 is safer with this reward scale.')
parser.add_argument('--max_act_vel', type=str, default=None,
                    help='Override DeformEnv.MAX_ACT_VEL (m/s). Pass '
                         '"auto" to probe and pick peak * 1.2. Or pass a '
                         'float — values below the trajectory peak '
                         '(typically 1-3 m/s) silently break demos. None '
                         '(default) leaves dedo at 10.0. See '
                         'train_privileged.py for full discussion.')
parser.add_argument('--critic_warmup_steps', type=int, default=0,
                    help='Freeze actor for N env steps after '
                         'learning_starts so Q networks can stabilize '
                         'before the actor moves. Prevents BC erasure. '
                         'Recommended 10000 when BC is on; 0 (default) '
                         'disables.')
parser.add_argument('--log_std_init', type=float, default=None,
                    help='Initial log_std for the SAC actor head. SB3 '
                         'default leaves log_std as a freshly-initialized '
                         'nn.Linear whose output ≈ N(0, ~1), so initial '
                         'exploration noise std ≈ 1 in pre-tanh space. '
                         'Scripted demo actions are ~0.03 in normalized '
                         '[-1, 1] space (waypoint vels ~0.3 m/s ÷ '
                         'MAX_ACT_VEL=10), so the BC-trained mu signal is '
                         'completely drowned out at deploy (SNR ~0.03; '
                         'tanh squashing makes it worse). Set to -3.5 '
                         '(std≈0.030) when BC is on, matching demo '
                         'magnitude so the BC mean is visible from step 0; '
                         'or -3.0 (std≈0.050) for slightly more '
                         'exploration. Pass None to leave log_std at its '
                         'random init.')
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
parser.add_argument('--action_penalty', type=float, default=0.0,
                    help='Per-step penalty on action magnitude. '
                         'reward -= action_penalty * mean(action**2). '
                         'Action is in [-1, 1]^6 so mean(a**2) is in '
                         '[0, 1]; coefs ~0.1-2 give per-step penalties '
                         'comparable to vel_penalty. Discourages bang-'
                         'bang/flailing control. 0 = off.')
parser.add_argument('--pre_settle_coef', type=float, default=0.0,
                    help='Linear penalty on hole-to-goal distance (m) at '
                         'policy handoff, BEFORE the gravity settle. '
                         'reward -= pre_settle_coef * pre_settle_dist_m. '
                         'Counters the "lift cloth high, let gravity drop '
                         'it onto the hanger" exploit. 0 = off; start '
                         'at 20.')
# BC pretrain.
parser.add_argument('--bc_episodes', type=int, default=0,
                    help='If >0 and --bc_demo_path not set: target '
                         'number of scripted hole-aware demos to KEEP '
                         'for BC pretrain. The collector retries until '
                         'it has this many demos that pass the keep '
                         'criterion (any if --bc_demos_only_success '
                         'off; only is_success=1 if on), capped at '
                         '~3x attempts. Dataset size is therefore '
                         'deterministic across seeds.')
parser.add_argument('--bc_epochs', type=int, default=20)
parser.add_argument('--bc_lr', type=float, default=1e-3)
parser.add_argument('--bc_demo_path', type=str, default=None,
                    help='Directory of demo_NNN.pkl files from '
                         'record_demo.py.')
parser.add_argument('--bc_demos_only_success', action='store_true',
                    help='Filter BC demos to only those that fire '
                         'is_success at terminal step. Applies to BOTH '
                         'manual demos (loaded from --bc_demo_path) and '
                         'scripted demos collected via --bc_episodes.')
# Resume.
parser.add_argument('--load_checkpoint', type=str, default=None,
                    help='Path to a previous run logdir (e.g. '
                         'hole_centroid_sac/SAC_<timestamp>_HangProcCloth-v1) '
                         'to resume training from. Loads agent.zip, '
                         'replay_buffer.pkl, and vec_normalize.pkl from '
                         'that directory; skips BC pretrain and '
                         'log_std_init patch (those are already baked in '
                         'to the saved policy). A NEW wandb run is started '
                         'with tags resumed_from=<orig> and '
                         'resume_step=<n_timesteps_at_load> so you can '
                         'group the original + resumed runs in the wandb '
                         'UI for one continuous chart.')
extra_args, remaining = parser.parse_known_args()

# `--no_adaptive_success` is the single switch for "use dedo's base reward
# and success unchanged". The wrapper treats success_factor=None as the
# disable signal, so we flip it here once for all downstream consumers.
if extra_args.no_adaptive_success:
    extra_args.success_factor = None

# Parse max_act_vel into None | 'auto' | float. See train_privileged.py.
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


# ---------------------------------------------------------------------------
# Build dedo args.
# ---------------------------------------------------------------------------
sys.argv = [
    'train_privileged_sac',
    '--env=HangProcCloth-v1',
    '--cam_resolution=0',
    '--num_envs=0',
    '--total_env_steps=0',
    f'--log_save_interval={extra_args.log_save_interval}',
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
dedo_args.log_save_interval = extra_args.log_save_interval
dedo_args.disable_logging_video = False

obs_mode = extra_args.obs_mode
logdir_base = os.path.join(extra_args.logdir_root, f'{obs_mode}_sac')
os.makedirs(logdir_base, exist_ok=True)
dedo_args.logdir = logdir_base

np.random.seed(dedo_args.seed)
torch.manual_seed(dedo_args.seed)


# Auto-probe MAX_ACT_VEL if requested. See train_privileged.py for rationale.
if _mav_mode == 'auto':
    print('[init] probing scripted-demo peak velocity (3 cloths)...')
    _peak = probe_peak_demo_vel(dedo_args, n_probes=3)
    if _peak is None:
        print('[init] WARN: all probes failed; leaving MAX_ACT_VEL at '
              'dedo default (10.0).')
    else:
        _new_mav = float(np.ceil(_peak * 1.2 * 10) / 10)
        from dedo.envs.deform_env import DeformEnv as _DeformEnvForPatch
        _orig = _DeformEnvForPatch.MAX_ACT_VEL
        _DeformEnvForPatch.MAX_ACT_VEL = _new_mav
        print(f'[init] DeformEnv.MAX_ACT_VEL: {_orig} -> {_new_mav} '
              f'(peak demo |vel| = {_peak:.3f} m/s × 1.2 safety, '
              f'rounded up to 0.1)')


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
                                    vel_penalty=extra_args.vel_penalty,
                                    action_penalty=extra_args.action_penalty,
                                    pre_settle_coef=extra_args.pre_settle_coef)
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
                                     vel_penalty=extra_args.vel_penalty,
                                     action_penalty=extra_args.action_penalty,
                                     pre_settle_coef=extra_args.pre_settle_coef)
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
        ap = extra_args.action_penalty
        psc = extra_args.pre_settle_coef
        bc_on = bool(extra_args.bc_demo_path or extra_args.bc_episodes > 0)
        _suffix = build_run_name_suffix(
            extra_args, algo='SAC', obs_kind=f'{obs_mode}_sac',
            net_arch=[256, 256])
        if extra_args.wandb_run_name:
            wandb.run.name = extra_args.wandb_run_name + _suffix
        else:
            wandb.run.name = wandb.run.name + _suffix
        wandb.run.tags = list(wandb.run.tags or []) + [
            'algo=sac',
            f'obs_mode={obs_mode}',
            f'success_factor={sf}',
            f'success_bonus={sb}',
            f'fail_penalty={fp}',
            f'vel_penalty={vp}',
            f'action_penalty={ap}',
            f'pre_settle_coef={psc}',
            f'bc={"yes" if bc_on else "no"}',
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
# ---------------------------------------------------------------------------
# Build agent. Two paths:
#   * Fresh: construct SAC, then optionally patch log_std and run BC.
#   * Resume: SAC.load() + load_replay_buffer() + apply saved
#     VecNormalize stats. Skip log_std patch and BC since both are
#     already baked into the saved policy / replay state.
# ---------------------------------------------------------------------------
_resuming = bool(extra_args.load_checkpoint)

if _resuming:
    ckpt_dir = extra_args.load_checkpoint
    if not os.path.isdir(ckpt_dir):
        raise FileNotFoundError(
            f'--load_checkpoint dir does not exist: {ckpt_dir}')
    agent_path = os.path.join(ckpt_dir, 'agent.zip')
    rb_path = os.path.join(ckpt_dir, 'replay_buffer.pkl')
    vn_path = os.path.join(ckpt_dir, 'vec_normalize.pkl')
    if not os.path.exists(agent_path):
        raise FileNotFoundError(f'no agent.zip in {ckpt_dir}')

    # 1. Restore VecNormalize stats IN PLACE (must happen before SAC.load
    #    so the loaded policy sees normalized obs at the same scale it
    #    was trained against).
    if os.path.exists(vn_path):
        _loaded_vn = VecNormalize.load(vn_path, vec_env.venv)
        vec_env.obs_rms = _loaded_vn.obs_rms
        vec_env.ret_rms = _loaded_vn.ret_rms
        vec_env.training = True
        print(f'[resume] loaded VecNormalize stats from {vn_path}')
    else:
        print(f'[resume] WARN: no vec_normalize.pkl at {vn_path}; '
              f'using fresh obs/ret rms (policy will see drifted obs scale)')

    # 2. Load SAC weights + optimizer + num_timesteps. Pass current rl_kwargs
    #    via custom_objects so e.g. learning_rate / ent_coef from the *new*
    #    command-line override the saved ones (you almost always want this
    #    on resume — that's why you're resuming).
    agent = SAC.load(
        agent_path, env=vec_env, device=dedo_args.device,
        tensorboard_log=dedo_args.logdir,
        custom_objects={
            'learning_rate': dedo_args.lr,
            'lr_schedule': lambda _progress: dedo_args.lr,
            'ent_coef': ent_coef_arg,
        })
    print(f'[resume] loaded SAC policy from {agent_path}; '
          f'num_timesteps={agent.num_timesteps:,}')

    # 3. Replay buffer: critical for SAC. Without it, SAC's first updates
    #    after resume are on whatever buffer SAC.load reconstructs (usually
    #    empty or tiny), making the policy drift quickly.
    if os.path.exists(rb_path):
        agent.load_replay_buffer(rb_path)
        try:
            _rb_size = agent.replay_buffer.size()
        except Exception:
            _rb_size = '?'
        print(f'[resume] loaded replay buffer from {rb_path} '
              f'(size={_rb_size})')
    else:
        print(f'[resume] WARN: no replay_buffer.pkl at {rb_path}; '
              f'SAC will resume with an empty buffer (expect actor drift).')

    # 4. Tag the new wandb run so it's clearly linked to its origin.
    if dedo_args.use_wandb:
        try:
            import wandb
            if wandb.run is not None:
                _orig_name = os.path.basename(ckpt_dir.rstrip('/'))
                wandb.run.tags = list(wandb.run.tags or []) + [
                    f'resumed_from={_orig_name}',
                    f'resume_step={agent.num_timesteps}',
                ]
                wandb.run.notes = (
                    f'Resumed from `{ckpt_dir}` at step '
                    f'{agent.num_timesteps:,}.\n'
                    f'Use the wandb "Group by tag" UI to overlay this '
                    f'run with the origin for a continuous chart.')
        except Exception:
            pass

else:
    # ---- Fresh agent ------------------------------------------------------
    agent = SAC('MlpPolicy', vec_env, **rl_kwargs)

    # -----------------------------------------------------------------------
    # Force-init the actor's log_std head to a small constant.
    #
    # Why: SB3 SAC (non-SDE) builds log_std as `nn.Linear(latent_dim,
    # action_dim)` with default torch init; `policy_kwargs={'log_std_init':
    # ...}` is silently ignored in this code path (it's only consumed by
    # the SDE branch). The resulting initial std is ~1.0 in pre-tanh space,
    # so tanh-squashed actions average |a| ≈ 0.55 before any learning —
    # overwhelming the ~0.03 magnitude of BC-trained mu (demo actions are
    # waypoint vels ~0.3 m/s ÷ MAX_ACT_VEL=10 ≈ 0.03 normalized). Patching
    # the bias to a constant (and zeroing the weight) pins log_std to that
    # constant at initialization; SAC's gradient updates move it from there.
    #
    # Pick log_std_init comparable to the demo |a| RMS (which depends on
    # MAX_ACT_VEL — see the [BC] traj peak |vel| diagnostic during demo
    # collection to estimate it; RMS ≈ peak / 5–10). Not setting this
    # knob preserves SB3 default behavior.
    # -----------------------------------------------------------------------
    if extra_args.log_std_init is not None:
        with torch.no_grad():
            agent.policy.actor.log_std.bias.fill_(
                float(extra_args.log_std_init))
            agent.policy.actor.log_std.weight.zero_()
        _init_std = float(np.exp(extra_args.log_std_init))
        print(f'[init] forced actor.log_std to constant '
              f'{extra_args.log_std_init} (std ≈ {_init_std:.3f}). Pick '
              f'this comparable to demo |a| RMS so BC mu is visible at '
              f'deploy.')

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
if not _resuming and extra_args.critic_warmup_steps > 0:
    _cbs.append(SACCriticWarmupCallback(
        n_warmup_env_steps=extra_args.critic_warmup_steps))
cb = CallbackList(_cbs)

# Persist a self-describing config.json + push to wandb.config so future
# debugging never has to guess what reward this run optimized.
dump_run_config(extra_args, dedo_args, dedo_args.logdir,
                use_wandb=dedo_args.use_wandb)


# ---------------------------------------------------------------------------
# Demo collection / loading (same as PPO version).
# ---------------------------------------------------------------------------
def _collect_demo_rollouts(args, obs_mode_str, num_episodes,
                           only_success=False, max_attempt_factor=3,
                           save_dir=None):
    """Roll out the scripted hole-aware waypoint controller.

    `num_episodes` is the **target number of demos kept**, not the number
    of attempts. The collector retries until that many demos pass the
    keep criterion (any rollout if `only_success=False`; only those with
    `is_success=1` if `only_success=True`), capped at
    `num_episodes * max_attempt_factor` attempts to avoid an infinite
    loop. Dataset size is therefore deterministic across seeds.

    `save_dir`: see train_privileged.py — when set, kept demos are
    persisted as `<save_dir>/demo_NNN.pkl` in the same payload format
    `record_demo.py` writes, so they can be reloaded via
    `--bc_demo_path`."""
    from dedo.demo_preset import build_traj, merge_traj
    from dedo.envs.deform_env import DeformEnv

    raw = gym.make(args.env, args=deepcopy(args))
    raw = RetryResetEnv(raw)
    raw = PrivilegedObsWrapper(raw, obs_mode=obs_mode_str,
                                success_factor=extra_args.success_factor,
                                success_bonus=0.0, fail_penalty=0.0,
                                vel_penalty=0.0,
                                action_penalty=0.0, pre_settle_coef=0.0)
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
            # Diagnostic on first attempt only: warn if MAX_ACT_VEL is too
            # low for the waypoint trajectory's peak speeds.
            if attempts == 1:
                _peak = float(np.abs(traj).max())
                _mav = float(DeformEnv.MAX_ACT_VEL)
                _flag = (' <- TOO LOW, demos will saturate'
                         if _peak > _mav else '')
                print(f'[BC] traj peak |vel| = {_peak:.3f} m/s; '
                      f'MAX_ACT_VEL = {_mav:.3f} m/s{_flag}')
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
            ep_obs.append(np.asarray(obs, dtype=np.float32))
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

        # Persist this demo using the same pkl schema record_demo.py
        # writes, so _load_manual_demos can read it back unchanged.
        if save_dir is not None:
            demo_idx = n_kept - 1
            pkl_path = os.path.join(save_dir, f'demo_{demo_idx:03d}.pkl')
            payload = {
                'obs': {obs_mode_str: np.asarray(ep_obs, dtype=np.float32)},
                'acts': np.asarray(ep_act, dtype=np.float32),
                'reward': float(ep_rwd),
                'success': int(ep_succ),
                'obs_modes': [obs_mode_str],
                'recorded_in': obs_mode_str,
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
    return (np.asarray(obs_buf, dtype=np.float32),
            np.asarray(act_buf, dtype=np.float32),
            succeeded)


def _load_manual_demos(demo_dir, obs_mode_str, only_success=False):
    """See train_privileged.py for the full docstring. The success_factor
    mismatch warning here is identical: filtering by `d['success']` only
    matches training when the recorder used the same success_factor."""
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


if _resuming:
    # On resume, skip BC entirely — the saved policy already encodes
    # whatever BC was applied in the originating run, and re-running BC
    # on top of a partially-RL-trained policy would clobber the RL
    # progress.
    print(f'\n=== resume mode: skipping BC pretrain (policy already '
          f'has BC + {agent.num_timesteps:,} steps of RL applied) ===')
    demo_obs = np.zeros((0, 0), dtype=np.float32)
    demo_acts = np.zeros((0, 0), dtype=np.float32)
    n_success, n_demos = 0, 0
elif extra_args.bc_demo_path:
    print(f'\n=== BC pretrain: loading manual demos from '
          f'{extra_args.bc_demo_path} ===')
    demo_obs, demo_acts, n_success, n_demos = _load_manual_demos(
        extra_args.bc_demo_path, obs_mode,
        only_success=extra_args.bc_demos_only_success)
elif extra_args.bc_episodes > 0:
    # Persist scripted demos under <logdir>/scripted_demos/ so they can
    # be inspected post-hoc and reused on a future run via
    # `--bc_demo_path <logdir>/scripted_demos`.
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
#
# `--total_env_steps` is interpreted as the *target total*, not the
# remainder. With reset_num_timesteps=False on resume, SAC continues from
# its loaded num_timesteps until that target is reached. So if you
# launched the original run with `--total_env_steps 500000`, killed at
# step 100k, and now want to keep going to 500k, just pass
# `--total_env_steps 500000` again. To extend further (e.g. 800k after
# resume), pass that bigger number.
# ---------------------------------------------------------------------------
print('Start SAC privileged training '
      f'({"resuming" if _resuming else "fresh"}; '
      f'num_timesteps starts at {agent.num_timesteps:,}; '
      f'target {extra_args.total_env_steps:,}) ...')
if _resuming and agent.num_timesteps >= extra_args.total_env_steps:
    print(f'[resume] num_timesteps already >= target; nothing to do. '
          f'Pass a higher --total_env_steps to extend training.')
agent.learn(total_timesteps=extra_args.total_env_steps, callback=cb,
            reset_num_timesteps=not _resuming)

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
