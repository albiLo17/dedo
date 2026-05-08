"""
Train PPO on HangProcCloth-v1 with a POINT-CLOUD observation (depth-camera
back-projection), as the obs-comparison counterpart to train_privileged.py.

Two intended runs:

  # Without BC (cold-start RL on PCD):
  python experiments/hang_obs_exp/scripts/train_pointcloud.py --use_wandb

  # With BC (scripted hole-aware demos collected in PCD obs space):
  python experiments/hang_obs_exp/scripts/train_pointcloud.py --use_wandb \
      --bc_episodes 100

Adaptive success / reward shaping (success_factor, success_bonus,
fail_penalty) defaults match train_privileged.py so results are
comparable. Eval videos logged to wandb show the captured PCD overlaid
on the RGB frame.

Outputs saved under logs/hang_obs_exp/pointcloud[_bc]/
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
from stable_baselines3.common.vec_env import VecNormalize

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _helpers import RetryResetEnv, build_hole_aware_waypoints  # noqa: E402

import dedo  # registers gym envs
from dedo.utils.args import get_args_parser, args_postprocess
from dedo.utils.train_utils import init_train
from dedo.utils.rl_sb3_utils import CustomCallback

from experiments.hang_obs_exp.envs.pointcloud_env import PointCloudObsWrapper


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
parser.add_argument('--n_eval_episodes', type=int, default=5,
                    help='Episodes per periodic eval during training '
                         '(CustomCallback runs eval every 2 checkpoints). '
                         'Bump up for less-noisy eval/success_rate curves '
                         'at the cost of training wall-clock.')
parser.add_argument('--n_points', type=int, default=512,
                    help='Number of points in the PCD obs (subsampled '
                         'from depth-back-projected world points).')
parser.add_argument('--cam_resolution_pcd', type=int, default=128,
                    help='Render resolution used for the depth → PCD pipe.')
# Adaptive success + shaping (same defaults as train_privileged.py).
parser.add_argument('--success_metric', type=str, default='distance',
                    choices=['distance', 'threading'],
                    help='Which metric defines is_success at episode end. '
                         '"distance" uses hole-centroid → goal_pos '
                         '(dedo default; threshold scaled by '
                         '--success_factor). "threading" uses a geometric '
                         'linking test — True iff a hanger rod pierces the '
                         'cloth-hole loop. Threading is robust to post-'
                         'settle centroid drift and lift-and-drop exploits, '
                         'and ignores --success_factor.')
parser.add_argument('--success_factor', type=float, default=1.2)
parser.add_argument('--success_bonus', type=float, default=200.0)
parser.add_argument('--fail_penalty', type=float, default=0.0)
parser.add_argument('--vel_penalty', type=float, default=0.0,
                    help='Per-step penalty proportional to cloth mean '
                         'vertex displacement. reward -= vel_penalty * '
                         'mean_per_vertex_disp. Discourages whippy '
                         'trajectories. 0 = off; try 1-10.')
parser.add_argument('--pre_settle_coef', type=float, default=0.0,
                    help='Linear penalty on hole-to-goal distance (m) at '
                         'policy handoff, BEFORE make_final_steps. '
                         'reward -= pre_settle_coef * pre_settle_dist_m. '
                         'Counters the "lift high, drop straight down" '
                         'exploit. 0 = off; start at 20.')
parser.add_argument('--action_penalty', type=float, default=0.0,
                    help='Per-step penalty on action magnitude. '
                         'reward -= action_penalty * mean(action**2). '
                         'Discourages bang-bang control. 0 = off.')
# BC pretrain (scripted hole-aware demos in PCD obs space).
parser.add_argument('--bc_episodes', type=int, default=0,
                    help='If >0 AND --bc_demo_path not set: collect this '
                         'many scripted hole-aware demos in the PCD '
                         'wrapper and BC-pretrain on them. 0 = skip BC.')
parser.add_argument('--bc_epochs', type=int, default=20)
parser.add_argument('--bc_lr', type=float, default=1e-3)
parser.add_argument('--bc_demo_path', type=str, default=None,
                    help='Directory of demo_NNN.pkl files from '
                         'record_demo_pcd.py. If set, load manual PCD '
                         'demos and BC on those (overrides bc_episodes).')
parser.add_argument('--bc_demos_only_success', action='store_true',
                    help='When loading manual demos, keep only success=1.')
# Policy / encoder.
parser.add_argument('--policy', type=str, default='mlp',
                    choices=['mlp', 'pointnet2'],
                    help='mlp = SB3 default flat MLP over [grip|flat_pcd]. '
                         'pointnet2 = hierarchical PointNet++ encoder over '
                         'the PCD slice, concat with gripper.')
parser.add_argument('--features_dim', type=int, default=512,
                    help='Output dim of pointnet2 features extractor '
                         '(ignored for --policy mlp).')
parser.add_argument('--cpu', action='store_true',
                    help='Force CPU even if CUDA is available. Useful when '
                         'PointNet++ FPS Python loop is slower on GPU due '
                         'to kernel launch overhead.')
# PPO optimization knobs (defaults tuned for pure-Python pointnet2 — keep
# updates feasible without CUDA-accelerated FPS / ball query).
parser.add_argument('--n_steps', type=int, default=4096,
                    help='Rollout buffer per env. Total samples per '
                         'update = n_steps * num_envs.')
parser.add_argument('--batch_size', type=int, default=512,
                    help='PPO minibatch size. Larger = fewer grad steps '
                         'per epoch but each step has bigger batches.')
parser.add_argument('--n_epochs', type=int, default=4,
                    help='Passes over the rollout per update. Default 4 '
                         '(was 10) for pointnet2-friendly wall-clock; bump '
                         'back to 8-10 for mlp policy if needed.')
extra_args, remaining = parser.parse_known_args()

# Subdir distinguishes BC vs non-BC runs.
_use_bc = (extra_args.bc_demo_path is not None
           or extra_args.bc_episodes > 0)
run_subdir = 'pointcloud_bc' if _use_bc else 'pointcloud'

# Build dedo args. We need cam_resolution>0 for depth render — the wrapper
# enforces this on construction, but we set it here too to be explicit.
sys.argv = [
    'train_pointcloud',
    '--env=HangProcCloth-v1',
    f'--cam_resolution={extra_args.cam_resolution_pcd}',
    '--num_envs=0',
    '--total_env_steps=0',
    '--log_save_interval=50',
    '--seed', str(extra_args.seed),
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
dedo_args.disable_logging_video = False  # we want the PCD-overlay videos
dedo_args.cam_viewmat = [9.0, -25.0, 45.0, 0.0, 0.5, 6.5]

logdir_base = os.path.join(extra_args.logdir_root, run_subdir)
os.makedirs(logdir_base, exist_ok=True)
dedo_args.logdir = logdir_base

np.random.seed(dedo_args.seed)
torch.manual_seed(dedo_args.seed)


# ---------------------------------------------------------------------------
# Env factory
# ---------------------------------------------------------------------------
def make_wrapped_env(args, monitor_dir=None):
    def _init():
        _args = deepcopy(args)
        _args.debug = False
        _args.viz = False
        env = gym.make(_args.env, args=_args)
        env = RetryResetEnv(env)
        env = PointCloudObsWrapper(
            env,
            n_points=extra_args.n_points,
            cam_resolution=extra_args.cam_resolution_pcd,
            success_metric=extra_args.success_metric,
            success_factor=extra_args.success_factor,
            success_bonus=extra_args.success_bonus,
            fail_penalty=extra_args.fail_penalty,
            vel_penalty=extra_args.vel_penalty,
            pre_settle_coef=extra_args.pre_settle_coef,
            action_penalty=extra_args.action_penalty)
        env = Monitor(env, filename=monitor_dir)
        return env
    return _init


# ---------------------------------------------------------------------------
# Training vec env + VecNormalize.
# ---------------------------------------------------------------------------
n_envs = extra_args.num_envs
vec_env = DummyVecEnv([make_wrapped_env(dedo_args) for _ in range(n_envs)])
vec_env.seed(dedo_args.seed)
# For PointNet++, disable per-dim obs whitening — the extractor needs the
# PCD coords to live in real Euclidean space (kNN, FPS, relative-xyz are
# geometry-aware). The wrapper already divides positions by 20, which
# keeps obs in roughly [-1, 1].
_norm_obs = (extra_args.policy != 'pointnet2')
vec_env = VecNormalize(vec_env,
                       norm_obs=_norm_obs,
                       norm_reward=True,
                       clip_obs=10.0)

# Eval env: same wrapper stack; obs normalization synced from training.
eval_args = deepcopy(dedo_args)
eval_env_raw = gym.make(eval_args.env, args=eval_args)
eval_env_raw = RetryResetEnv(eval_env_raw)
eval_env_raw = PointCloudObsWrapper(
    eval_env_raw,
    n_points=extra_args.n_points,
    cam_resolution=extra_args.cam_resolution_pcd,
    success_metric=extra_args.success_metric,
    success_factor=extra_args.success_factor,
    success_bonus=extra_args.success_bonus,
    fail_penalty=extra_args.fail_penalty,
    vel_penalty=extra_args.vel_penalty,
    pre_settle_coef=extra_args.pre_settle_coef,
    action_penalty=extra_args.action_penalty)
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
print(f'Condition: pointcloud{"+BC" if _use_bc else ""}')
print(f'Obs shape: {obs_shape}  Action shape: {vec_env.action_space.shape}')
print(f'PCD: {extra_args.n_points} pts  cam_res={extra_args.cam_resolution_pcd}')
print(f'Steps: {extra_args.total_env_steps:,}  Envs: {n_envs}  LR: {extra_args.lr}')
print(f'Logdir: {logdir_base}')
print(f'{"="*60}\n')


# ---------------------------------------------------------------------------
# Init wandb / logdir.
# ---------------------------------------------------------------------------
dedo_args.logdir, dedo_args.device = init_train('PPO', dedo_args)
if extra_args.cpu:
    dedo_args.device = 'cpu'
    print(f'[device] forced CPU via --cpu flag '
          f'(torch.cuda.is_available()={torch.cuda.is_available()})')

if dedo_args.use_wandb:
    import wandb
    if wandb.run is not None:
        sm = extra_args.success_metric
        sf = extra_args.success_factor
        sb = extra_args.success_bonus
        fp = extra_args.fail_penalty
        vp = extra_args.vel_penalty
        psc = extra_args.pre_settle_coef
        ap = extra_args.action_penalty
        bc_tag = '_bc' if _use_bc else ''
        sm_tag = f'_sm-{sm}'
        sf_tag = f'_sf{sf:g}' if sf is not None else '_sf_default'
        sb_tag = f'_sb{sb:g}' if sb else ''
        fp_tag = f'_fp{fp:g}' if fp else ''
        vp_tag = f'_vp{vp:g}' if vp else ''
        psc_tag = f'_psc{psc:g}' if psc else ''
        ap_tag = f'_ap{ap:g}' if ap else ''
        wandb.run.name = (f'{wandb.run.name}_pcd{extra_args.n_points}'
                          f'{bc_tag}{sm_tag}{sf_tag}{sb_tag}{fp_tag}'
                          f'{vp_tag}{psc_tag}{ap_tag}')
        wandb.run.save()
        wandb.run.tags = list(wandb.run.tags or []) + [
            'obs=pointcloud',
            f'bc={"yes" if _use_bc else "no"}',
            f'success_metric={sm}',
            f'success_factor={sf}',
            f'success_bonus={sb}',
            f'fail_penalty={fp}',
            f'vel_penalty={vp}',
            f'pre_settle_coef={psc}',
            f'action_penalty={ap}',
            f'n_points={extra_args.n_points}',
            f'policy={extra_args.policy}',
        ]
        wandb.run.name = wandb.run.name + f'_{extra_args.policy}'
        wandb.run.save()
        # Promote shaping coefs to top-level config keys (filterable in
        # the wandb runs table; not auto-included from extra_args).
        wandb.config.update({
            'success_metric': sm,
            'shaping_success_factor': sf,
            'shaping_success_bonus': sb,
            'shaping_fail_penalty': fp,
            'shaping_vel_penalty': vp,
            'shaping_pre_settle_coef': psc,
            'shaping_action_penalty': ap,
            'n_points': extra_args.n_points,
            'cam_resolution_pcd': extra_args.cam_resolution_pcd,
            'policy': extra_args.policy,
        }, allow_val_change=True)


# ---------------------------------------------------------------------------
# PPO.
# ---------------------------------------------------------------------------
if extra_args.policy == 'pointnet2':
    from pointnet2_extractor import PointNet2FeaturesExtractor
    policy_kwargs = dict(
        features_extractor_class=PointNet2FeaturesExtractor,
        features_extractor_kwargs=dict(
            n_points=extra_args.n_points,
            grip_dim=12,
            features_dim=extra_args.features_dim,
        ),
        # MLP on top of the PointNet++ features.
        net_arch=[256, 256],
    )
else:
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
agent = PPO('MlpPolicy', vec_env, **rl_kwargs)

num_steps_between_save = dedo_args.log_save_interval * 10 * 50
cb = CustomCallback(eval_env, dedo_args.logdir, n_envs, dedo_args,
                    num_steps_between_save=num_steps_between_save,
                    viz=False, debug=False,
                    n_eval_episodes=extra_args.n_eval_episodes)


# ---------------------------------------------------------------------------
# Optional BC pretrain on scripted hole-aware demos collected in PCD space.
# ---------------------------------------------------------------------------
def _collect_pcd_demos(args, num_episodes):
    from dedo.demo_preset import build_traj, merge_traj
    from dedo.envs.deform_env import DeformEnv

    raw = gym.make(args.env, args=deepcopy(args))
    raw = RetryResetEnv(raw)
    raw = PointCloudObsWrapper(
        raw,
        n_points=extra_args.n_points,
        cam_resolution=extra_args.cam_resolution_pcd,
        success_metric=extra_args.success_metric,
        success_factor=extra_args.success_factor,
        success_bonus=0.0,  # don't shape demo rewards
        fail_penalty=0.0)
    raw.seed(args.seed + 1000)

    obs_buf, act_buf = [], []
    n_success = 0
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
        n_success += ep_succ
        print(f'[BC] demo {ep+1}/{num_episodes}  '
              f'len={len(ep_obs)}  rwd={ep_rwd:.1f}  success={ep_succ}')

    raw.close()
    return (np.asarray(obs_buf, dtype=np.float32),
            np.asarray(act_buf, dtype=np.float32),
            n_success)


def _bc_pretrain(agent, demo_obs, demo_acts, vec_normalize,
                 epochs, batch_size, lr):
    import torch.nn.functional as F
    vec_normalize.obs_rms.update(demo_obs)
    norm_obs = vec_normalize.normalize_obs(demo_obs).astype(np.float32)
    obs_t = torch.as_tensor(norm_obs, device=agent.device)
    act_t = torch.as_tensor(demo_acts, device=agent.device)

    policy = agent.policy
    optimizer = torch.optim.Adam(policy.parameters(), lr=lr)
    n = len(demo_obs)
    print(f'[BC] training on {n} (obs, act) pairs '
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


def _load_manual_pcd_demos(demo_dir, n_points, only_success=False):
    """Load (obs, act) pairs from record_demo_pcd.py outputs."""
    obs_buf, act_buf = [], []
    n_files, n_success_demos = 0, 0
    for fname in sorted(os.listdir(demo_dir)):
        if not (fname.startswith('demo_') and fname.endswith('.pkl')):
            continue
        path = os.path.join(demo_dir, fname)
        with open(path, 'rb') as f:
            d = pickle.load(f)
        if d.get('obs_type') != 'pointcloud':
            print(f'[BC] {fname}: obs_type={d.get("obs_type")!r} '
                  f'not pointcloud, skipping')
            continue
        if d.get('n_points') != n_points:
            print(f'[BC] {fname}: n_points={d.get("n_points")} != '
                  f'{n_points}, skipping (re-record at matching n_points)')
            continue
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
    if not obs_buf:
        return (np.zeros((0, 0), dtype=np.float32),
                np.zeros((0, 0), dtype=np.float32),
                0, 0)
    return (np.concatenate(obs_buf, axis=0).astype(np.float32),
            np.concatenate(act_buf, axis=0).astype(np.float32),
            n_success_demos, n_files)


if extra_args.bc_demo_path:
    print(f'\n=== BC pretrain: loading manual PCD demos from '
          f'{extra_args.bc_demo_path} ===')
    demo_obs, demo_acts, n_success, n_demos = _load_manual_pcd_demos(
        extra_args.bc_demo_path, extra_args.n_points,
        only_success=extra_args.bc_demos_only_success)
    print(f'[BC] loaded {len(demo_obs)} (obs,act) pairs from '
          f'{n_demos} manual demos ({n_success} succeeded)')
elif extra_args.bc_episodes > 0:
    print(f'\n=== BC pretrain: collecting {extra_args.bc_episodes} '
          f'scripted PCD demos ===')
    demo_obs, demo_acts, n_success = _collect_pcd_demos(
        dedo_args, extra_args.bc_episodes)
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
                 batch_size=256, lr=extra_args.bc_lr)
    if dedo_args.use_wandb:
        import wandb
        wandb.log({'bc/n_pairs': len(demo_obs),
                   'bc/n_success_demos': n_success,
                   'bc/n_demos': n_demos})
elif _use_bc:
    print('[BC] no usable demos found — skipping BC pretrain')


# ---------------------------------------------------------------------------
# Train.
# ---------------------------------------------------------------------------
print('Start pointcloud RL training ...')
agent.learn(total_timesteps=extra_args.total_env_steps, callback=cb)

ckpt_path = os.path.join(dedo_args.logdir, 'agent.zip')
agent.save(ckpt_path)
vec_env.save(os.path.join(dedo_args.logdir, 'vec_normalize.pkl'))
pickle.dump(dedo_args, open(os.path.join(dedo_args.logdir, 'args.pkl'), 'wb'))
print(f'\nDone. Checkpoint: {ckpt_path}')


# ---------------------------------------------------------------------------
# Final eval.
# ---------------------------------------------------------------------------
from stable_baselines3.common.evaluation import evaluate_policy

print(f'\nRunning final eval ({extra_args.n_final_eval_episodes} episodes)...')
final_successes = []
final_threadings = []
final_distance_successes = []


def _final_cb(_locals, _globals=None):
    info = _locals.get('info', {})
    if 'is_success' in info:
        final_successes.append(int(info['is_success']))
    if 'is_threaded' in info:
        final_threadings.append(int(info['is_threaded']))
    if 'is_distance_success' in info:
        final_distance_successes.append(int(info['is_distance_success']))


mean_rwd, std_rwd = evaluate_policy(
    agent, eval_env, n_eval_episodes=extra_args.n_final_eval_episodes,
    deterministic=True, callback=_final_cb, return_episode_rewards=False)


def _rate(xs):
    return sum(xs) / len(xs) if xs else float('nan')


final_success_rate = _rate(final_successes)
final_threading_rate = _rate(final_threadings)
final_distance_success_rate = _rate(final_distance_successes)
print(f'Final eval — mean_rwd={mean_rwd:.3f} ± {std_rwd:.3f}  '
      f'success_rate={final_success_rate:.3f}  '
      f'threading_rate={final_threading_rate:.3f}  '
      f'distance_success_rate={final_distance_success_rate:.3f}  '
      f'(n={len(final_successes)})')

if dedo_args.use_wandb:
    import wandb
    wandb.log({
        'final_eval/mean_reward': mean_rwd,
        'final_eval/std_reward': std_rwd,
        'final_eval/success_rate': final_success_rate,
        'final_eval/threading_rate': final_threading_rate,
        'final_eval/distance_success_rate': final_distance_success_rate,
        'final_eval/n_episodes': len(final_successes),
    })
    wandb.finish()

vec_env.close()
