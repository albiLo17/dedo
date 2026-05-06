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
from _helpers import (RetryResetEnv, build_hole_aware_waypoints,  # noqa: E402
                      probe_peak_demo_vel)
from _video_callback import HangVideoCallback  # noqa: E402
from _critic_warmup import PPOCriticWarmupCallback  # noqa: E402
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
                         'video @ 100k.')
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
parser.add_argument('--action_penalty', type=float, default=0.0,
                    help='Per-step penalty on action magnitude. '
                         'reward -= action_penalty * mean(action**2). '
                         'Mirrors train_privileged.py. 0 = off.')
parser.add_argument('--pre_settle_coef', type=float, default=0.0,
                    help='Linear penalty on hole-to-goal distance (m) at '
                         'policy handoff, BEFORE the gravity settle. '
                         'reward -= pre_settle_coef * pre_settle_dist_m. '
                         'Mirrors train_privileged.py. 0 = off; start at 20.')
# BC pretrain (scripted hole-aware demos rolled out in pixel obs space).
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
                         'current --cam_resolution and --no_grip mode '
                         '(loader skips mismatches with a warning). '
                         'Reuse demos saved under '
                         '<prior_logdir>/scripted_demos/ across seeds.')
parser.add_argument('--no_save_scripted_demos', action='store_true',
                    help='Skip persisting scripted BC demos to disk. '
                         'Default behaviour saves them to '
                         '<logdir>/scripted_demos/ so they can be '
                         'reused via --bc_demo_path. At 64x64x3 uint8 '
                         'each demo is ~2 MB, so 50 demos ≈ 100 MB '
                         'per run — pass this flag for big sweeps '
                         'where disk pressure matters.')
parser.add_argument('--bc_demos_only_success', action='store_true',
                    help='Drop scripted demos whose terminal '
                         'is_success=0 from the BC dataset. The hole-'
                         'aware waypoint controller succeeds on most '
                         'but not all procedural cloth shapes; '
                         'filtering yields a cleaner BC dataset at the '
                         'cost of fewer (obs, act) pairs. Strongly '
                         'recommended unless --bc_episodes is small '
                         '(<20) and the scripted success rate is low.')
parser.add_argument('--max_act_vel', type=str, default=None,
                    help='Override DeformEnv.MAX_ACT_VEL (m/s). Pass '
                         '"auto" to probe demo trajectories and pick '
                         'peak * 1.2. Or pass a float — but values below '
                         'the trajectory peak (typically 1-3 m/s) silently '
                         'break demos. None (default) leaves dedo at 10.0. '
                         'See train_privileged.py for full discussion.')
parser.add_argument('--net_arch', type=str, default='256,256',
                    help='Comma-separated MLP hidden sizes for the head '
                         'after the CNN feature extractor. Default '
                         '"256,256". Try "512,512" if BC mse plateaus '
                         'too high on the pixel task.')
parser.add_argument('--ent_coef', type=float, default=0.0,
                    help='PPO entropy bonus coefficient. 0.0 (default) '
                         'lets log_std collapse post-warmup; set '
                         '0.005-0.02 to keep exploration intact while '
                         'actor updates. See train_privileged.py for '
                         'rationale.')
parser.add_argument('--critic_warmup_rollouts', type=int, default=0,
                    help='Freeze actor for first N PPO rollouts so V(s) '
                         'can converge before the actor moves. Prevents '
                         'BC erasure. Recommended 2 when BC is on; 0 '
                         '(default) disables. NOTE for pixels: PPO '
                         'shares features_extractor (CNN) between actor '
                         'and critic by default, so the CNN is *not* '
                         'frozen — critic gradients still drift it '
                         'mildly. For perfect BC preservation pass '
                         'share_features_extractor=False (not exposed '
                         'as flag here).')
parser.add_argument('--log_std_init', type=float, default=None,
                    help='Initial log_std for the PPO actor head. SB3 '
                         'default is 0.0 (std=1.0 in pre-clip space). '
                         'Scripted demos here have |action| ≈ 0.03 in '
                         'normalized [-1, 1] space (waypoint vels ~0.3 '
                         'm/s ÷ MAX_ACT_VEL=10), so default noise std=1.0 '
                         'completely drowns out BC-trained mu at rollout '
                         'time. Set to -3.5 (std≈0.030) to match demo '
                         'magnitude so BC is visible from step 0; -3.0 '
                         '(std≈0.050) for slightly more exploration. None '
                         '(default) keeps SB3 default. Skipped on resume.')
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
parser.add_argument('--load_checkpoint', type=str, default=None,
                    help='Path to a previous run logdir (containing '
                         'agent.zip and vec_normalize.pkl) to resume '
                         'training from. Skips BC pretrain (already baked '
                         'into the saved policy). --total_env_steps is '
                         'the TARGET total. A NEW wandb run is started, '
                         'tagged `resumed_from=<orig>` for grouping in UI.')
extra_args, remaining = parser.parse_known_args()

if extra_args.no_adaptive_success:
    extra_args.success_factor = None

# Parse net_arch up front so it's available when wandb.run.name is built
# below (which happens before policy construction).
_net_arch_list = [int(x) for x in extra_args.net_arch.split(',') if x.strip()]

# Parse max_act_vel into None | 'auto' | float. Explicit floats patch
# now; 'auto' defers until after dedo_args is built so we can probe via
# real cloth resets. See train_privileged.py for full discussion.
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
    f'--log_save_interval={extra_args.log_save_interval}',
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
dedo_args.log_save_interval = extra_args.log_save_interval
dedo_args.disable_logging_video = False

logdir_base = os.path.join(extra_args.logdir_root, run_subdir)
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
            action_penalty=extra_args.action_penalty,
            pre_settle_coef=extra_args.pre_settle_coef,
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
    action_penalty=extra_args.action_penalty,
    pre_settle_coef=extra_args.pre_settle_coef,
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
        ap = extra_args.action_penalty
        psc = extra_args.pre_settle_coef
        sf_tag = f'_sf{sf:g}' if sf is not None else '_sf_default'
        sb_tag = f'_sb{sb:g}' if sb else ''
        fp_tag = f'_fp{fp:g}' if fp else ''
        vp_tag = f'_vp{vp:g}' if vp else ''
        ap_tag = f'_ap{ap:g}' if ap else ''
        psc_tag = f'_psc{psc:g}' if psc else ''
        grip_tag = '_grip' if not extra_args.no_grip else '_pix'
        net_tag = '_net' + 'x'.join(str(s) for s in _net_arch_list)
        wandb.run.name = (f'{wandb.run.name}_pixels'
                          f'{extra_args.cam_resolution}{grip_tag}{net_tag}'
                          f'{sf_tag}{sb_tag}{fp_tag}{vp_tag}{ap_tag}{psc_tag}')
        wandb.run.tags = list(wandb.run.tags or []) + [
            'obs=pixels',
            f'include_grip={"yes" if not extra_args.no_grip else "no"}',
            f'cam_resolution={extra_args.cam_resolution}',
            f'success_factor={sf if sf is not None else "default"}',
            f'success_bonus={sb}',
            f'fail_penalty={fp}',
            f'vel_penalty={vp}',
            f'action_penalty={ap}',
            f'pre_settle_coef={psc}',
        ]


# ---------------------------------------------------------------------------
# PPO. MultiInputPolicy when obs is Dict, CnnPolicy for image-only.
# net_arch=[256, 256] for the post-feature-extractor MLP head matches the
# privileged scripts. SB3's NatureCNN is the default features extractor;
# leaving it at the default is the right call unless we want to swap in
# IMPALA-CNN or similar.
# ---------------------------------------------------------------------------
policy_name = 'MultiInputPolicy' if not extra_args.no_grip else 'CnnPolicy'
policy_kwargs = dict(net_arch=_net_arch_list)
print(f'[init] policy net_arch (post-CNN) = {_net_arch_list}')
if extra_args.log_std_init is not None:
    # PPO's DiagGaussianDistribution uses log_std as an nn.Parameter
    # initialized to log_std_init. Lowering it below 0 is essential here:
    # demo |action| ≈ 0.03, so default std=1.0 makes noise overwhelm BC's
    # mu by ~30x and rollouts revert to ~uniform.
    policy_kwargs['log_std_init'] = float(extra_args.log_std_init)
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
    ent_coef=float(extra_args.ent_coef),
)
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
    # VecNormalize wraps a VecTransposeImage here, but the .load(...) call
    # restores running stats onto whatever venv is passed; we use vec_env
    # itself since norm_obs=False (no obs_rms to overwrite, only ret_rms).
    if os.path.exists(vn_path):
        _loaded_vn = VecNormalize.load(vn_path, vec_env.venv)
        vec_env.obs_rms = _loaded_vn.obs_rms
        vec_env.ret_rms = _loaded_vn.ret_rms
        vec_env.training = True
        print(f'[resume] loaded VecNormalize stats from {vn_path}')
    else:
        print(f'[resume] WARN: no vec_normalize.pkl at {vn_path}; '
              f'reward normalization will drift from origin run')
    agent = PPO.load(
        agent_path, env=vec_env, device=dedo_args.device,
        tensorboard_log=dedo_args.logdir,
        custom_objects={
            'learning_rate': dedo_args.lr,
            'lr_schedule': lambda _progress: dedo_args.lr,
        })
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
    agent = PPO(policy_name, vec_env, **rl_kwargs)
    if extra_args.log_std_init is not None:
        _init_std = float(np.exp(extra_args.log_std_init))
        print(f'[init] PPO log_std_init = {extra_args.log_std_init} '
              f'(std ≈ {_init_std:.3f}). Pick this comparable to demo '
              f'|a| RMS so BC mu is visible at deploy.')

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
if not _resuming and extra_args.critic_warmup_rollouts > 0:
    _cbs.append(PPOCriticWarmupCallback(
        n_warmup_rollouts=extra_args.critic_warmup_rollouts))
cb = CallbackList(_cbs)

dump_run_config(extra_args, dedo_args, dedo_args.logdir,
                use_wandb=dedo_args.use_wandb)


# ---------------------------------------------------------------------------
# Optional BC pretrain on scripted hole-aware demos. Walk the dedo demo
# preset waypoints, capture (obs, action) pairs in pixel obs space, and
# regress the policy mean onto the demo actions. Skipped by default
# (--bc_episodes 0); vision BC is finicky and shaped reward usually
# converges without it.
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
    reloaded later via `--bc_demo_path` (which routes through
    `_load_manual_pixel_demos`). Pkl payload mirrors the schema
    record_demo_pcd.py uses, with `obs_type='pixels'` plus
    `cam_resolution` / `include_grip` so the loader can reject
    incompatible demos."""
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
        action_penalty=0.0,
        pre_settle_coef=0.0,
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

        # Diagnostic: peak waypoint velocity vs active MAX_ACT_VEL. If peak
        # > MAX_ACT_VEL the demo gripper saturates during the fast lift/
        # thread phase and demos silently fail to reach the hanger.
        if attempts == 1:
            _peak = float(np.abs(traj).max())
            _mav = float(DeformEnv.MAX_ACT_VEL)
            _flag = ' <- TOO LOW, demos will saturate' if _peak > _mav else ''
            print(f'[BC] traj peak |vel| = {_peak:.3f} m/s; '
                  f'MAX_ACT_VEL = {_mav:.3f} m/s{_flag}')

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

        # Persist this episode as a self-contained pkl so it can be
        # reloaded via --bc_demo_path. Pre-stack the per-step obs (same
        # logic as the bottom of this function) so the on-disk shape
        # matches exactly what the loader will concatenate across
        # episodes — no per-step processing needed at load time.
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


def _load_manual_pixel_demos(demo_dir, cam_resolution, include_grip,
                             only_success=False):
    """Load per-episode pixel demos written by `_collect_pixel_demos`
    (or any future record_demo_pixels.py with a matching schema). Each
    pkl holds ONE pre-stacked episode; we concatenate across episodes
    to return the same (obs, act) shape `_collect_pixel_demos` returns.

    Compatibility: `cam_resolution` and `include_grip` MUST match the
    current run (otherwise the obs shape doesn't even match the policy's
    observation_space). Mismatched demos are skipped with a per-file
    log line. `success_factor` mismatch is a *soft* warning — `success`
    flags may not reflect the training criterion, but the (obs, act)
    pairs themselves are still valid trajectories.

    Returns (obs, acts, n_success_demos, n_files), where obs is the
    same dict-or-array structure that `_collect_pixel_demos` produces.
    Empty result returns (None, zeros, 0, 0) for parity with the
    collector's empty-case return."""
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
                  f'skipping (re-record at matching resolution)')
            continue
        if bool(d.get('include_grip')) != bool(include_grip):
            print(f'[BC] {fname}: include_grip='
                  f'{d.get("include_grip")} != {include_grip}, '
                  f'skipping (Dict-vs-Box obs mismatch)')
            continue

        if 'success_factor' in d:
            if d['success_factor'] != sf_train:
                sf_mismatches += 1
        else:
            sf_unknown += 1

        if only_success and not d.get('success', 0):
            print(f'[BC] {fname}: success=0, skipping '
                  f'(--bc_demos_only_success)')
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
    # Concat episodes into one big (obs, act) pair set, matching the
    # shape `_collect_pixel_demos` returns.
    if isinstance(obs_buf[0], dict):
        keys = obs_buf[0].keys()
        stacked = {k: np.concatenate([o[k] for o in obs_buf], axis=0)
                   for k in keys}
    else:
        stacked = np.concatenate(obs_buf, axis=0)
    return (stacked,
            np.concatenate(act_buf, axis=0).astype(np.float32),
            n_success_demos, n_files)


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


if _resuming:
    demo_obs, demo_acts, n_success, n_demos = (
        None, np.zeros((0, 0), dtype=np.float32), 0, 0)
    print('[resume] skipping BC pretrain (policy already trained)')
elif extra_args.bc_demo_path:
    print(f'\n=== BC pretrain: loading manual pixel demos from '
          f'{extra_args.bc_demo_path} ===')
    demo_obs, demo_acts, n_success, n_demos = _load_manual_pixel_demos(
        extra_args.bc_demo_path,
        cam_resolution=extra_args.cam_resolution,
        include_grip=not extra_args.no_grip,
        only_success=extra_args.bc_demos_only_success)
elif extra_args.bc_episodes > 0:
    # Persist scripted demos under <logdir>/scripted_demos/ unless the
    # user opts out. Pixel demos are larger than privileged ones (~2 MB
    # per demo at 64x64x3 uint8); the --no_save_scripted_demos flag is
    # the escape hatch for big sweeps where disk pressure matters.
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
    print(f'[BC] {n_pairs} (obs,act) pairs from '
          f'{n_demos} demos ({n_success} succeeded)')
    _bc_pretrain(agent, demo_obs, demo_acts,
                 epochs=extra_args.bc_epochs,
                 batch_size=128, lr=extra_args.bc_lr)
    if dedo_args.use_wandb:
        import wandb
        wandb.log({'bc/n_pairs': n_pairs,
                   'bc/n_success_demos': n_success,
                   'bc/n_demos': n_demos})
elif extra_args.bc_demo_path or extra_args.bc_episodes > 0:
    print('[BC] no usable demos — skipping BC pretrain')


print(f'Start pixel RL training '
      f'({"resuming" if _resuming else "fresh"}; '
      f'num_timesteps={agent.num_timesteps:,}; '
      f'target {extra_args.total_env_steps:,}) ...')
agent.learn(total_timesteps=extra_args.total_env_steps, callback=cb,
            reset_num_timesteps=not _resuming)

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
