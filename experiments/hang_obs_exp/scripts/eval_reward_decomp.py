"""
eval_reward_decomp.py — load a checkpoint OR demo episodes, run rollouts
through the same PrivilegedObsWrapper that training used, and plot the
reward curve over steps decomposed into all of its shaping components.

Goal: high-visibility view of "what reward is the agent / scripted
controller actually accumulating per step" so reward-shape design can be
sanity-checked against episodes (not just rolling means in wandb).

Four input modes (mutually exclusive):

  --checkpoint <run_logdir>
      Loads agent.zip + vec_normalize.pkl + (config.json | args.pkl) from
      a train_privileged.py run directory and runs `--n_episodes`
      deterministic rollouts. The reward shape (success_factor,
      success_bonus, fail_penalty, vel_penalty, action_penalty,
      pre_settle_coef, obs_mode) is read from config.json so the same
      reward the policy was trained against is what's plotted.

  --demo_dir <scripted_demos_dir>
      Loads every demo_NNN.pkl in the directory, replays the recorded
      actions through a fresh env, and decomposes the rewards. The
      env's reward shape defaults to whatever `success_factor` the
      pkls were recorded under (and zero shaping for the new dials),
      but you can override any of them via --override_*.

  --demo_pkl <demo_NNN.pkl>
      Same as --demo_dir but for a single demo. Useful for deep-diving
      one episode at a time.

  --scripted
      Generate fresh scripted demos on the fly using the hole-aware
      waypoint controller (same logic as view_demo.py and the BC
      collector in train_privileged.py). No replay — re-builds the
      trajectory per episode against each episode's freshly procedural
      cloth. Use this to sanity-check what reward the canonical
      "expert" earns under a candidate reward shape before training a
      policy against it.

Reward decomposition (matches PrivilegedObsWrapper.step):

  total = base
        - action_penalty   (every step,    if action_penalty > 0)
        - vel_penalty      (non-terminal,  if vel_penalty > 0)
        - pre_settle_pen   (terminal,      if pre_settle_coef > 0)
        + terminal_shaping (terminal,      +success_bonus | -fail_penalty)

`base` = the underlying DeformEnv reward, recovered as
`total - shaping_added + action_penalty + vel_penalty + pre_settle_pen`
since the wrapper writes those keys to info whenever its respective
coef is on. The terminal step's `base` is dominated by dedo's
post-settle FINAL_REWARD term (~400 * dist), so the per-step plot shows
that as a single tall bar at the last step — exactly the "drop spike"
the shaping is trying to control.

Usage examples (from repo root):

  # Plot a learned-policy run.
  python experiments/hang_obs_exp/scripts/eval_reward_decomp.py \
      --checkpoint logs/hang_obs_exp/hole_centroid/PPO_260506_030441_HangProcCloth-v1 \
      --n_episodes 5

  # Plot the scripted demos that BC was pretrained on.
  python experiments/hang_obs_exp/scripts/eval_reward_decomp.py \
      --demo_dir logs/hang_obs_exp/hole_centroid/PPO_260506_030441_HangProcCloth-v1/scripted_demos \
      --override_vel_penalty 8.0 --override_pre_settle_coef 20.0

Outputs (written next to the input by default, or under --out_dir):
  reward_decomp_<timestamp>.png            per-episode + aggregate decomposition
  reward_decomp_<timestamp>.csv            per-step, per-episode raw values
  reward_decomp_<timestamp>_videos/        one mp4 per episode (--save_videos,
                                           on by default; pass --no_videos to
                                           skip if you only need the chart).
"""

import sys
import os
import argparse
import json
import pickle
import csv
from copy import deepcopy
from datetime import datetime
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT))

import numpy as np
import gym
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import dedo  # noqa: F401  (registers gym envs)
from dedo.utils.args import get_args_parser, args_postprocess
from dedo.envs.deform_env import DeformEnv

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _helpers import RetryResetEnv, build_hole_aware_waypoints  # noqa: E402
from experiments.hang_obs_exp.envs.privileged_env import PrivilegedObsWrapper

try:
    import cv2  # OpenCV is the only mp4 writer dedo's other replay scripts use
except ImportError:
    cv2 = None


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def parse_args():
    p = argparse.ArgumentParser()
    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument('--checkpoint', type=str, default=None,
                     help='Path to a train_privileged.py run dir '
                          '(contains agent.zip / vec_normalize.pkl / '
                          'config.json).')
    src.add_argument('--demo_dir', type=str, default=None,
                     help='Path to a directory of demo_NNN.pkl files. '
                          'Replays each demo and decomposes its rewards.')
    src.add_argument('--demo_pkl', type=str, default=None,
                     help='Path to a single demo_NNN.pkl. Replays and '
                          'plots that one episode.')
    src.add_argument('--scripted', action='store_true',
                     help='Generate fresh scripted demos via the '
                          'hole-aware waypoint controller (same logic '
                          'as view_demo.py / BC collector). Use this '
                          'to plot what the canonical expert earns '
                          'under a candidate reward shape.')
    p.add_argument('--n_episodes', type=int, default=5,
                   help='Number of rollouts (--checkpoint / --scripted '
                        'mode). For --demo_dir, defaults to all pkls '
                        'in the dir; cap with this flag.')
    p.add_argument('--save_videos', action='store_true', default=True,
                   help='Write one mp4 per rollout to '
                        '<out_dir>/reward_decomp_<ts>_videos/ep_NN.mp4. '
                        'Default on. Pass --no_videos to disable.')
    p.add_argument('--no_videos', dest='save_videos',
                   action='store_false')
    p.add_argument('--video_resolution', type=int, default=400,
                   help='Square render size for the mp4 (default 400). '
                        'Independent of DeformEnv.cam_resolution; we '
                        'always force the env to vector obs and call '
                        'underlying.render(width=H, height=H) for video.')
    p.add_argument('--video_fps', type=int, default=24,
                   help='mp4 framerate (default 24, matches replay_demo.py).')
    p.add_argument('--seed', type=int, default=12345,
                   help='Eval-env seed (offset from this for each ep).')
    p.add_argument('--deterministic', action='store_true', default=True,
                   help='Deterministic policy actions (default on).')
    p.add_argument('--no_deterministic', dest='deterministic',
                   action='store_false')
    p.add_argument('--max_episode_len', type=int, default=None,
                   help='Override env max_episode_len for the eval. '
                        'Defaults to whatever the run / demo used.')
    p.add_argument('--out_dir', type=str, default=None,
                   help='Where to write the .png + .csv. Defaults to '
                        'the input dir.')
    # Reward-shape overrides (apply on top of whatever was loaded).
    for name, dflt, helpstr in [
            ('success_factor', None, 'override adaptive-success factor'),
            ('success_bonus', None, 'override terminal success_bonus'),
            ('fail_penalty', None, 'override terminal fail_penalty'),
            ('vel_penalty', None, 'override per-step vel_penalty coef'),
            ('action_penalty', None, 'override per-step action_penalty coef'),
            ('pre_settle_coef', None, 'override terminal pre_settle_coef'),
            ('dist_reward_coef', None,
             'override per-step dist_reward_coef'),
            ('threading_bonus_coef', None,
             'override per-step threading_bonus_coef'),
            ('obs_mode', None, 'override obs mode used by the wrapper'),
            ('max_act_vel', None, 'override DeformEnv.MAX_ACT_VEL'),
            ('final_reward_mult', None,
             'override DeformEnv.FINAL_REWARD_MULT (default 400)'),
    ]:
        p.add_argument(f'--override_{name}', type=str, default=None,
                       help=helpstr + ' (passed as string; "none" = None).')
    return p.parse_args()


# ---------------------------------------------------------------------------
# Config plumbing
# ---------------------------------------------------------------------------
REWARD_KEYS = ('success_factor', 'success_bonus', 'fail_penalty',
               'vel_penalty', 'action_penalty', 'pre_settle_coef',
               'dist_reward_coef', 'threading_bonus_coef',
               'obs_mode')


def _coerce(name, raw):
    """Coerce CLI override strings into the right type. 'none' -> None."""
    if raw is None:
        return raw
    if str(raw).lower() == 'none':
        return None
    if name == 'obs_mode':
        return str(raw)
    return float(raw)


def load_run_config(checkpoint_dir):
    """Read reward-shape config from config.json (preferred) or args.pkl."""
    cfg_path = os.path.join(checkpoint_dir, 'config.json')
    cfg = {}
    if os.path.exists(cfg_path):
        with open(cfg_path) as f:
            blob = json.load(f)
        extra = blob.get('extra', {}) or {}
        for k in REWARD_KEYS:
            if k in extra:
                cfg[k] = extra[k]
        # The training script's max_act_vel argument is a *string* in the
        # CLI ('auto' | float | None). On disk it's always serialized as
        # the raw input. We don't try to replay 'auto' here — if you want
        # to match the training cap exactly, pass --override_max_act_vel.
        cfg['max_episode_len'] = extra.get('max_episode_len')
        cfg['seed'] = extra.get('seed')
        # If the training run patched FINAL_REWARD_MULT, surface it so
        # the decomp visualization runs with the same terminal-magnitude
        # as the training reward shape.
        if 'final_reward_mult' in extra:
            cfg['final_reward_mult'] = extra['final_reward_mult']
    else:
        # Fall back to args.pkl for at least max_episode_len / seed.
        args_pkl = os.path.join(checkpoint_dir, 'args.pkl')
        if os.path.exists(args_pkl):
            with open(args_pkl, 'rb') as f:
                a = pickle.load(f)
            cfg['max_episode_len'] = getattr(a, 'max_episode_len', None)
            cfg['seed'] = getattr(a, 'seed', None)
        else:
            print(f'[warn] no config.json or args.pkl in {checkpoint_dir}; '
                  f'reward shape will default to zeros — pass --override_*')
    cfg.setdefault('obs_mode', 'hole_centroid')
    cfg.setdefault('success_factor', None)
    for k in ('success_bonus', 'fail_penalty', 'vel_penalty',
              'action_penalty', 'pre_settle_coef',
              'dist_reward_coef', 'threading_bonus_coef'):
        cfg.setdefault(k, 0.0)
    cfg.setdefault('max_episode_len', 200)
    cfg.setdefault('seed', 42)
    return cfg


def load_demo_config(any_demo_pkl):
    """Pull obs_mode + success_factor from a demo pkl. Reward shaping
    coefs are NOT recorded in the demo schema (they're a wrapper-time
    decision), so they default to zero — override via --override_*."""
    with open(any_demo_pkl, 'rb') as f:
        d = pickle.load(f)
    cfg = {
        'obs_mode': d.get('recorded_in') or d.get('obs_mode')
                    or (list(d.get('obs', {}).keys())[0]
                        if isinstance(d.get('obs'), dict) else 'hole_centroid'),
        'success_factor': d.get('success_factor', None),
        'success_bonus': 0.0,
        'fail_penalty': 0.0,
        'vel_penalty': 0.0,
        'action_penalty': 0.0,
        'pre_settle_coef': 0.0,
        'dist_reward_coef': 0.0,
        'threading_bonus_coef': 0.0,
        'max_episode_len': max(d.get('len', 200), 200),
        'seed': 42,
    }
    return cfg


def apply_overrides(cfg, parsed):
    """Stamp --override_* values into the cfg in place."""
    for k in REWARD_KEYS + ('max_act_vel', 'final_reward_mult'):
        raw = getattr(parsed, f'override_{k}')
        if raw is not None:
            cfg[k] = _coerce(k, raw)
    if parsed.max_episode_len is not None:
        cfg['max_episode_len'] = parsed.max_episode_len
    return cfg


# ---------------------------------------------------------------------------
# Env factory (matches train_privileged.py's wrapping order)
# ---------------------------------------------------------------------------
def build_dedo_args(cfg):
    sys.argv = [
        'eval_reward_decomp',
        '--env=HangProcCloth-v1',
        '--cam_resolution=0',
        '--num_envs=0',
        '--total_env_steps=0',
        '--seed', str(int(cfg['seed'])),
        '--max_episode_len', str(int(cfg['max_episode_len'])),
        '--cam_viewmat', '9.0', '-25.0', '45.0', '0.0', '0.5', '6.5',
    ]
    args, _ = get_args_parser()
    args_postprocess(args)
    args.viz = False
    args.debug = False
    return args


def make_env(dedo_args, cfg, seed_offset=0):
    env = gym.make(dedo_args.env, args=deepcopy(dedo_args))
    env = RetryResetEnv(env)
    env = PrivilegedObsWrapper(
        env, obs_mode=cfg['obs_mode'],
        success_factor=cfg['success_factor'],
        success_bonus=float(cfg['success_bonus']),
        fail_penalty=float(cfg['fail_penalty']),
        vel_penalty=float(cfg['vel_penalty']),
        action_penalty=float(cfg['action_penalty']),
        pre_settle_coef=float(cfg['pre_settle_coef']),
        dist_reward_coef=float(cfg.get('dist_reward_coef', 0.0)),
        threading_bonus_coef=float(cfg.get('threading_bonus_coef', 0.0)))
    env.seed(int(dedo_args.seed) + seed_offset)
    return env


# ---------------------------------------------------------------------------
# Video helpers
# ---------------------------------------------------------------------------
def _resolve_underlying(env):
    """Walk down a wrapper stack to the raw DeformEnv (which is the only
    layer with a usable render() implementation). Returns the DeformEnv
    or None if not found."""
    cur = env
    while True:
        if isinstance(cur, DeformEnv):
            return cur
        if hasattr(cur, 'env'):
            cur = cur.env
            continue
        if hasattr(cur, 'envs'):  # DummyVecEnv
            return _resolve_underlying(cur.envs[0])
        return None


def _open_video_writer(path, resolution, fps):
    """Create an mp4 VideoWriter or return None (and warn) if cv2 is
    missing or the writer fails to open. Caller treats None as 'skip
    video for this episode' rather than crashing the rollout."""
    if cv2 is None:
        print('[video] cv2 unavailable; install opencv-python to save '
              'videos. Skipping.')
        return None
    os.makedirs(os.path.dirname(path), exist_ok=True)
    vw = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*'mp4v'), fps,
                         (resolution, resolution))
    if not vw.isOpened():
        print(f'[video] WARN: failed to open {path}; skipping')
        return None
    return vw


def _capture_frame(vw, deform_env, resolution):
    """Render a single frame and write it. RGB-from-pybullet -> BGR-for-cv2
    via the trailing [..., ::-1] swap (matches replay_demo.py)."""
    if vw is None or deform_env is None:
        return
    try:
        img = deform_env.render(mode='rgb_array',
                                width=resolution, height=resolution)
        vw.write(img[..., ::-1])
    except Exception as e:
        print(f'[video] frame capture failed: {e!r}; continuing without')


def _enable_settle_capture(deform_env, resolution):
    """Tell DeformEnv.make_final_steps() to render frames during the
    post-policy gravity settle (~500 sim steps where the cloth either
    catches the apex or slips off). Without this, the video stops at
    policy handoff and you can't see whether the demo actually succeeded
    — which is the whole reason we're plotting reward decomposition."""
    if deform_env is None:
        return
    deform_env._record_settle_frames = True
    deform_env._settle_render_kwargs = dict(
        width=resolution, height=resolution)
    # Default stride=1 captures every recorded sub-step (sim_steps_per_action
    # cadence -> ~62 frames over 500 sim steps), which is fine for ~3 s of
    # settle at 24 fps. Up the stride if videos get bloated.


def _disable_settle_capture(deform_env):
    if deform_env is not None:
        deform_env._record_settle_frames = False


def _capture_settle_frames(vw, info):
    """Drain DeformEnv-rendered settle frames from info into the writer.
    Frames arrive as RGB ndarrays; cv2 wants BGR."""
    if vw is None:
        return
    settle = info.get('settle_frames')
    if not settle:
        return
    for frame in settle:
        try:
            vw.write(np.ascontiguousarray(frame)[..., ::-1])
        except Exception as e:
            print(f'[video] settle-frame write failed: {e!r}')
            return


# ---------------------------------------------------------------------------
# Rollout drivers
# ---------------------------------------------------------------------------
def _decompose_step(reward, info):
    """Extract reward components from one step's (reward, info) tuple.
    Returns (base, action_pen, vel_pen, pre_settle_pen, shaping,
    dist_reward, threading_bonus). Recovers `base` from total reward by
    subtracting all wrapper-added shaping terms."""
    a_pen = float(info.get('action_penalty', 0.0))
    v_pen = float(info.get('vel_penalty', 0.0))
    ps_pen = float(info.get('pre_settle_penalty', 0.0))
    shaping = float(info.get('shaping_added', 0.0))
    dist_rew = float(info.get('dist_reward', 0.0))
    thread_bonus = float(info.get('threading_bonus', 0.0))
    base = (float(reward) + a_pen + v_pen + ps_pen
            - shaping - dist_rew - thread_bonus)
    return base, a_pen, v_pen, ps_pen, shaping, dist_rew, thread_bonus


def _episode_record_init():
    return {
        'step': [], 'reward': [], 'base': [],
        'action_pen': [], 'vel_pen': [], 'pre_settle_pen': [],
        'terminal_shaping': [], 'dist_reward': [], 'threading_bonus': [],
        'is_success': 0,
        'adaptive_dist': None, 'adaptive_thresh': None,
    }


def rollout_policy(checkpoint_dir, cfg, n_episodes, deterministic,
                   video_dir=None, video_resolution=400, video_fps=24):
    """Run a saved PPO checkpoint's policy. Returns list of episode dicts.
    If video_dir is given, writes one mp4 per episode."""
    from stable_baselines3 import PPO
    from stable_baselines3.common.env_util import DummyVecEnv
    from stable_baselines3.common.vec_env import VecNormalize
    from stable_baselines3.common.monitor import Monitor

    dedo_args = build_dedo_args(cfg)

    # Same wrap order as training: Monitor inside VecNormalize so VecNormalize
    # sees the wrapper's shaped reward (matters for VecNormalize.ret_rms,
    # though we set training=False below so it's frozen anyway).
    def _make():
        env = make_env(dedo_args, cfg, seed_offset=0)
        return Monitor(env)
    venv = DummyVecEnv([_make])

    vn_path = os.path.join(checkpoint_dir, 'vec_normalize.pkl')
    if os.path.exists(vn_path):
        venv = VecNormalize.load(vn_path, venv)
        venv.training = False
        venv.norm_reward = False  # we want raw rewards in the plot
        print(f'[load] restored VecNormalize from {vn_path}')
    else:
        print(f'[warn] no vec_normalize.pkl at {vn_path}; policy will see '
              f'un-normalized obs (likely poor performance)')
        venv = VecNormalize(venv, norm_obs=False, norm_reward=False)
        venv.training = False

    agent_path = os.path.join(checkpoint_dir, 'agent.zip')
    agent = PPO.load(agent_path, env=venv, device='cpu')
    print(f'[load] PPO from {agent_path}; '
          f'num_timesteps={agent.num_timesteps:,}')

    # Drive the underlying single env directly so we can read info per step
    # cleanly (DummyVecEnv eats the terminal info into 'terminal_observation'
    # autoreset behavior). VecNormalize.normalize_obs gives us the obs the
    # policy expects.
    inner_env = venv.venv.envs[0]   # Monitor wrapping our PrivilegedObs stack
    deform = _resolve_underlying(inner_env)
    if video_dir is not None:
        _enable_settle_capture(deform, video_resolution)
    episodes = []
    try:
        for ep_idx in range(n_episodes):
            inner_env.seed(int(dedo_args.seed) + ep_idx)
            obs = inner_env.reset()
            # Open a fresh writer per episode so a partial save survives if a
            # later episode crashes the env.
            vw = None
            if video_dir is not None:
                vw = _open_video_writer(
                    os.path.join(video_dir, f'ep_{ep_idx:02d}.mp4'),
                    video_resolution, video_fps)
            # Initial frame: cloth right after reset, before any policy action.
            _capture_frame(vw, deform, video_resolution)
            rec = _episode_record_init()
            step = 0
            while True:
                norm_obs = venv.normalize_obs(np.asarray(obs, dtype=np.float32))
                action, _ = agent.predict(norm_obs[None],
                                          deterministic=deterministic)
                obs, reward, done, info = inner_env.step(action[0])
                base, a_pen, v_pen, ps_pen, shaping, dr, tb = (
                    _decompose_step(reward, info))
                rec['step'].append(step)
                rec['reward'].append(float(reward))
                rec['base'].append(base)
                rec['action_pen'].append(a_pen)
                rec['vel_pen'].append(v_pen)
                rec['pre_settle_pen'].append(ps_pen)
                rec['terminal_shaping'].append(shaping)
                rec['dist_reward'].append(dr)
                rec['threading_bonus'].append(tb)
                if 'is_success' in info:
                    rec['is_success'] = int(info['is_success'])
                if 'adaptive_dist' in info:
                    rec['adaptive_dist'] = float(info['adaptive_dist'])
                if 'adaptive_thresh' in info:
                    rec['adaptive_thresh'] = float(info['adaptive_thresh'])
                if done:
                    # Terminal step: drain the settle frames into the video
                    # so the catch-or-slip moment is visible. Skip the
                    # post-step single frame because the settle sequence
                    # already ends at the post-settle pose.
                    _capture_settle_frames(vw, info)
                else:
                    _capture_frame(vw, deform, video_resolution)
                step += 1
                if done:
                    break
            if vw is not None:
                vw.release()
            episodes.append(rec)
            print(f'[rollout] ep {ep_idx+1}/{n_episodes}  '
                  f'len={len(rec["step"])}  '
                  f'sum_reward={sum(rec["reward"]):.2f}  '
                  f'success={rec["is_success"]}  '
                  f'dist={rec["adaptive_dist"]}')
    finally:
        _disable_settle_capture(deform)
        inner_env.close()
    return episodes


def rollout_demos(demo_paths, cfg, video_dir=None,
                  video_resolution=400, video_fps=24):
    """Replay recorded action sequences. Returns list of episode dicts.
    Optionally writes one mp4 per episode to video_dir."""
    dedo_args = build_dedo_args(cfg)
    env = make_env(dedo_args, cfg, seed_offset=0)
    deform = _resolve_underlying(env)
    if video_dir is not None:
        _enable_settle_capture(deform, video_resolution)

    episodes = []
    try:
        for ep_idx, pkl in enumerate(demo_paths):
            with open(pkl, 'rb') as f:
                d = pickle.load(f)
            acts = np.asarray(d['acts'], dtype=np.float32)
            # Each replay rolls fresh procedural cloth — re-seed for variety
            # but deterministically so reruns reproduce.
            env.seed(int(dedo_args.seed) + ep_idx)
            obs = env.reset()
            vw = None
            if video_dir is not None:
                vw = _open_video_writer(
                    os.path.join(video_dir,
                                 f'ep_{ep_idx:02d}_{Path(pkl).stem}.mp4'),
                    video_resolution, video_fps)
            _capture_frame(vw, deform, video_resolution)
            rec = _episode_record_init()
            for step, a in enumerate(acts):
                obs, reward, done, info = env.step(a)
                base, a_pen, v_pen, ps_pen, shaping, dr, tb = (
                    _decompose_step(reward, info))
                rec['step'].append(step)
                rec['reward'].append(float(reward))
                rec['base'].append(base)
                rec['action_pen'].append(a_pen)
                rec['vel_pen'].append(v_pen)
                rec['pre_settle_pen'].append(ps_pen)
                rec['terminal_shaping'].append(shaping)
                rec['dist_reward'].append(dr)
                rec['threading_bonus'].append(tb)
                if 'is_success' in info:
                    rec['is_success'] = int(info['is_success'])
                if 'adaptive_dist' in info:
                    rec['adaptive_dist'] = float(info['adaptive_dist'])
                if 'adaptive_thresh' in info:
                    rec['adaptive_thresh'] = float(info['adaptive_thresh'])
                if done:
                    _capture_settle_frames(vw, info)
                else:
                    _capture_frame(vw, deform, video_resolution)
                if done:
                    break
            if vw is not None:
                vw.release()
            episodes.append(rec)
            print(f'[replay] ep {ep_idx+1}/{len(demo_paths)} ({Path(pkl).name})  '
                  f'len={len(rec["step"])}  '
                  f'sum_reward={sum(rec["reward"]):.2f}  '
                  f'success={rec["is_success"]}  '
                  f'dist={rec["adaptive_dist"]}')
    finally:
        _disable_settle_capture(deform)
        env.close()
    return episodes


def rollout_scripted(cfg, n_episodes, video_dir=None,
                     video_resolution=400, video_fps=24,
                     max_attempt_factor=3):
    """Generate fresh demos via the hole-aware waypoint controller and
    decompose their rewards. Mirrors view_demo.py's per-episode logic +
    the BC collector's keep-N-rollouts-that-succeed loop logic, except
    here we keep ALL rollouts (success or not) since the point is to
    study the reward, not the BC dataset.

    Returns (episodes, peaks) where peaks is the per-episode peak |vel|
    in m/s — useful for diagnosing whether MAX_ACT_VEL was set safely
    (peak > MAX_ACT_VEL means demo actions saturate and the gripper
    can't keep up)."""
    from dedo.demo_preset import build_traj, merge_traj

    dedo_args = build_dedo_args(cfg)
    env = make_env(dedo_args, cfg, seed_offset=0)
    deform = _resolve_underlying(env)
    if video_dir is not None:
        _enable_settle_capture(deform, video_resolution)
    ctrl_freq = dedo_args.sim_freq / dedo_args.sim_steps_per_action

    episodes = []
    peaks = []
    attempts = 0
    max_attempts = max(n_episodes * max_attempt_factor, n_episodes + 5)
    while len(episodes) < n_episodes and attempts < max_attempts:
        attempts += 1
        ep_idx = len(episodes)
        env.seed(int(dedo_args.seed) + ep_idx + 1000)
        env.reset()
        underlying = deform if deform is not None else _resolve_underlying(env)
        if underlying is None:
            print('[scripted] could not locate DeformEnv; aborting')
            break
        wp = build_hole_aware_waypoints(underlying)
        if wp is None:
            print(f'[scripted] attempt {attempts}: no hole loop on cloth, '
                  f'retrying')
            continue
        try:
            _, va = build_traj(underlying, wp, 'a', anchor_idx=0,
                               ctrl_freq=ctrl_freq, robot=None)
            _, vb = build_traj(underlying, wp, 'b', anchor_idx=1,
                               ctrl_freq=ctrl_freq, robot=None)
            traj = merge_traj(va, vb)
        except Exception as e:
            print(f'[scripted] attempt {attempts}: build_traj failed '
                  f'({e!r}), retrying')
            continue

        ep_peak = float(np.abs(traj).max())
        peaks.append(ep_peak)

        vw = None
        if video_dir is not None:
            vw = _open_video_writer(
                os.path.join(video_dir, f'ep_{ep_idx:02d}_scripted.mp4'),
                video_resolution, video_fps)
        _capture_frame(vw, deform, video_resolution)

        last = np.zeros_like(traj[0])
        rec = _episode_record_init()
        step = 0
        while True:
            act_unscaled = traj[step] if step < len(traj) else last
            normalized = np.clip(
                act_unscaled / DeformEnv.MAX_ACT_VEL, -1.0, 1.0
            ).astype(np.float32)
            obs, reward, done, info = env.step(normalized)
            base, a_pen, v_pen, ps_pen, shaping, dr, tb = (
                _decompose_step(reward, info))
            rec['step'].append(step)
            rec['reward'].append(float(reward))
            rec['base'].append(base)
            rec['action_pen'].append(a_pen)
            rec['vel_pen'].append(v_pen)
            rec['pre_settle_pen'].append(ps_pen)
            rec['terminal_shaping'].append(shaping)
            rec['dist_reward'].append(dr)
            rec['threading_bonus'].append(tb)
            if 'is_success' in info:
                rec['is_success'] = int(info['is_success'])
            if 'adaptive_dist' in info:
                rec['adaptive_dist'] = float(info['adaptive_dist'])
            if 'adaptive_thresh' in info:
                rec['adaptive_thresh'] = float(info['adaptive_thresh'])
            if done:
                _capture_settle_frames(vw, info)
            else:
                _capture_frame(vw, deform, video_resolution)
            step += 1
            if done:
                break
        if vw is not None:
            vw.release()
        episodes.append(rec)
        # Mirror view_demo.py's velocity diagnostic so it's easy to spot
        # MAX_ACT_VEL clipping just from the eval log.
        mav = float(DeformEnv.MAX_ACT_VEL)
        flag = ' <- SATURATING' if ep_peak > mav else ''
        print(f'[scripted] ep {ep_idx+1}/{n_episodes} '
              f'(attempt {attempts})  len={len(rec["step"])}  '
              f'sum_reward={sum(rec["reward"]):.2f}  '
              f'success={rec["is_success"]}  '
              f'dist={rec["adaptive_dist"]}  '
              f'peak|vel|={ep_peak:.3f} m/s '
              f'(MAX_ACT_VEL={mav:.3f}{flag})')

    _disable_settle_capture(deform)
    env.close()
    if len(episodes) < n_episodes:
        print(f'[scripted] WARNING: only got {len(episodes)}/{n_episodes} '
              f'after {attempts} attempts (cap={max_attempts}). '
              f'build_hole_aware_waypoints kept failing — check the cloth '
              f'distribution.')
    if peaks:
        gp = max(peaks)
        sug = float(np.ceil(gp * 1.2 * 10) / 10)
        print(f'[scripted] global peak |vel| = {gp:.3f} m/s; '
              f'suggested --max_act_vel = {sug:.1f}')
    return episodes


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------
COMPONENT_COLORS = {
    'base': '#1f77b4',                   # blue
    'action_pen': '#ff7f0e',             # orange
    'vel_pen': '#2ca02c',                # green
    'pre_settle_pen': '#d62728',         # red
    'terminal_shaping': '#9467bd',       # purple
    'dist_reward': '#17becf',            # cyan (per-step dense distance)
    'threading_bonus': '#bcbd22',        # olive (per-step threading)
    'reward': '#000000',                 # black (sum, for reference)
}
SIGN = {
    'base': +1, 'action_pen': -1, 'vel_pen': -1,
    'pre_settle_pen': -1, 'terminal_shaping': +1,
    'dist_reward': +1, 'threading_bonus': +1, 'reward': +1,
}


def _component_series(rec, name):
    """Signed contribution of a component to per-step total reward."""
    return SIGN[name] * np.asarray(rec[name if name != 'reward' else 'reward'],
                                    dtype=np.float64)


def plot_decomposition(episodes, cfg, out_png, title_extra=''):
    """6-panel layout:
        (R0L) per-step total reward (symlog) — full episode incl. terminal spike
        (R0R) cumulative reward — running Σ
        (R1L) per-step component decomposition, NON-terminal steps only.
              Linear y-axis; reveals the small-magnitude per-step shaping
              (vel_pen, action_pen, base distance reward) that the terminal
              spike would otherwise drown out on a shared axis.
        (R1R) terminal-step component breakdown (one column per episode).
              Stacks pre_settle_pen / terminal_shaping / terminal_base
              so the "drop spike" is fully decomposed in isolation.
        (R2L) per-episode totals (stacked components).
        (R2R) per-episode signed components grouped (mirrors R2L without
              stacking — easier to see which component changes between eps).
    """
    n_ep = len(episodes)
    fig, axes = plt.subplots(3, 2, figsize=(14, 12))

    max_len = max(len(r['step']) for r in episodes)

    # --- (R0L) per-step total reward, symlog so terminal spike + per-step
    # shaping are both visible on a single axis. linthresh=1 means values
    # in [-1, 1] render linearly (preserves zeros) and outside they're
    # log-scaled (compresses the -180 terminal). ---
    ax = axes[0, 0]
    for rec in episodes:
        ax.plot(rec['step'], rec['reward'], color='black', alpha=0.18, lw=0.8)
    pad = np.full((n_ep, max_len), np.nan)
    for i, rec in enumerate(episodes):
        pad[i, :len(rec['reward'])] = rec['reward']
    mean = np.nanmean(pad, axis=0)
    ax.plot(np.arange(max_len), mean, color='black', lw=1.8, label='mean')
    ax.axhline(0, color='gray', lw=0.5, ls='--')
    ax.set_yscale('symlog', linthresh=1.0)
    ax.set_title(f'Per-step total reward  ({n_ep} ep, symlog)')
    ax.set_xlabel('env step'); ax.set_ylabel('reward')
    ax.legend(loc='upper left', fontsize=8)

    # --- (R0R) cumulative reward, linear ---
    ax = axes[0, 1]
    pad = np.full((n_ep, max_len), np.nan)
    for i, rec in enumerate(episodes):
        cum = np.cumsum(rec['reward'])
        ax.plot(rec['step'], cum, color='black', alpha=0.2, lw=0.8)
        pad[i, :len(cum)] = cum
    mean = np.nanmean(pad, axis=0)
    ax.plot(np.arange(max_len), mean, color='black', lw=1.8, label='mean')
    ax.axhline(0, color='gray', lw=0.5, ls='--')
    ax.set_title('Cumulative reward')
    ax.set_xlabel('env step'); ax.set_ylabel('Σ reward')
    ax.legend(loc='upper left', fontsize=8)

    # --- (R1L) per-step component contribution, NON-terminal only ---
    # The terminal step pulls the y-axis by 100x (post-settle FINAL_REWARD
    # in dedo's get_reward()), so we drop it here and break it out in R1R.
    ax = axes[1, 0]
    comp_names = ['base', 'action_pen', 'vel_pen',
                  'pre_settle_pen', 'terminal_shaping',
                  'dist_reward', 'threading_bonus']
    active = []
    for name in comp_names:
        any_nonzero = any(
            np.any(np.asarray(rec[name]) != 0.0) for rec in episodes)
        if any_nonzero:
            active.append(name)
    for name in active:
        pad = np.full((n_ep, max_len), np.nan)
        for i, rec in enumerate(episodes):
            v = _component_series(rec, name).copy()
            # Mask the terminal step so it doesn't blow out the y-axis.
            if len(v) > 0:
                v[-1] = np.nan
            pad[i, :len(v)] = v
        # Drop the series if it's all-nan OR all-zero post-mask: terminal-only
        # components (pre_settle_pen, terminal_shaping) collapse to zeros
        # here and add nothing but legend noise.
        if np.all(np.isnan(pad)) or np.nanmax(np.abs(pad)) == 0.0:
            continue
        mean = np.nanmean(pad, axis=0)
        lo = np.nanpercentile(pad, 25, axis=0)
        hi = np.nanpercentile(pad, 75, axis=0)
        x = np.arange(max_len)
        ax.plot(x, mean, color=COMPONENT_COLORS[name], lw=1.5,
                label=f'{"+" if SIGN[name] > 0 else "-"}{name}')
        ax.fill_between(x, lo, hi, color=COMPONENT_COLORS[name], alpha=0.15)
    ax.axhline(0, color='gray', lw=0.5, ls='--')
    ax.set_title('Per-step contribution, non-terminal (mean ± IQR)')
    ax.set_xlabel('env step'); ax.set_ylabel('signed contribution')
    ax.legend(loc='best', fontsize=8)

    # --- (R1R) terminal-step component breakdown, one bar per episode ---
    ax = axes[1, 1]
    x = np.arange(n_ep)
    pos_b = np.zeros(n_ep)
    neg_b = np.zeros(n_ep)
    for name in comp_names:
        # Terminal step contribution = signed value at the last step.
        v = np.array([SIGN[name] * float(rec[name][-1])
                      if len(rec[name]) > 0 else 0.0
                      for rec in episodes])
        if np.all(v == 0):
            continue
        pos = v > 0; neg = v < 0
        if pos.any():
            ax.bar(x[pos], v[pos], bottom=pos_b[pos],
                   color=COMPONENT_COLORS[name],
                   label=f'{"+" if SIGN[name] > 0 else "-"}{name}',
                   edgecolor='white', linewidth=0.4)
            pos_b[pos] += v[pos]
        if neg.any():
            ax.bar(x[neg], v[neg], bottom=neg_b[neg],
                   color=COMPONENT_COLORS[name],
                   label=(None if pos.any()
                          else f'{"+" if SIGN[name] > 0 else "-"}{name}'),
                   edgecolor='white', linewidth=0.4)
            neg_b[neg] += v[neg]
    term_totals = np.array([rec['reward'][-1] if rec['reward'] else 0.0
                            for rec in episodes])
    ax.scatter(x, term_totals, color='black', marker='_', s=120, zorder=5,
               label='terminal reward')
    ax.axhline(0, color='gray', lw=0.5, ls='--')
    ax.set_title('Terminal-step decomposition (per episode)')
    ax.set_xlabel('episode'); ax.set_ylabel('signed contribution')
    ax.set_xticks(x)
    ax.legend(loc='best', fontsize=8, ncol=2)

    # --- (R2L) per-episode total reward, stacked bar ---
    ax = axes[2, 0]
    sums = {name: np.array([SIGN[name] * float(np.sum(rec[name]))
                            for rec in episodes])
            for name in comp_names}
    totals = np.array([sum(rec['reward']) for rec in episodes])
    successes = np.array([rec['is_success'] for rec in episodes])
    x = np.arange(n_ep)
    pos_bottom = np.zeros(n_ep)
    neg_bottom = np.zeros(n_ep)
    for name in comp_names:
        v = sums[name]
        if np.all(v == 0):
            continue
        pos_mask = v > 0
        neg_mask = v < 0
        if pos_mask.any():
            ax.bar(x[pos_mask], v[pos_mask], bottom=pos_bottom[pos_mask],
                   color=COMPONENT_COLORS[name],
                   label=f'{"+" if SIGN[name] > 0 else "-"}{name}',
                   edgecolor='white', linewidth=0.4)
            pos_bottom[pos_mask] += v[pos_mask]
        if neg_mask.any():
            ax.bar(x[neg_mask], v[neg_mask], bottom=neg_bottom[neg_mask],
                   color=COMPONENT_COLORS[name],
                   label=(None if pos_mask.any()
                          else f'{"+" if SIGN[name] > 0 else "-"}{name}'),
                   edgecolor='white', linewidth=0.4)
            neg_bottom[neg_mask] += v[neg_mask]
    # Net total marker.
    ax.scatter(x, totals, color='black', marker='_', s=120, zorder=5,
               label='net total')
    # Mark successes with a green dot just above each bar's top.
    if successes.any():
        ax.scatter(x[successes == 1], pos_bottom[successes == 1] + 5,
                   marker='*', s=80, color='gold', edgecolor='black',
                   linewidth=0.5, label='success', zorder=6)
    ax.axhline(0, color='gray', lw=0.5, ls='--')
    ax.set_title('Per-episode totals (stacked components)')
    ax.set_xlabel('episode'); ax.set_ylabel('Σ contribution')
    ax.set_xticks(x)
    ax.legend(loc='best', fontsize=8, ncol=2)

    # --- (R2R) per-episode signed components grouped (side-by-side bars) ---
    # Same data as R2L but unstacked, so component-to-component magnitudes
    # are easier to compare across episodes (e.g. "did vel_pen blow up on
    # ep 5?"). Tradeoff: less obvious which way nets compose.
    ax = axes[2, 1]
    active_for_groups = [n for n in comp_names if not np.all(sums[n] == 0)]
    if active_for_groups:
        n_act = len(active_for_groups)
        bar_w = 0.8 / n_act
        for j, name in enumerate(active_for_groups):
            ax.bar(x + (j - n_act / 2) * bar_w + bar_w / 2,
                   sums[name], width=bar_w,
                   color=COMPONENT_COLORS[name],
                   label=f'{"+" if SIGN[name] > 0 else "-"}{name}')
    ax.axhline(0, color='gray', lw=0.5, ls='--')
    ax.set_title('Per-episode signed components (grouped)')
    ax.set_xlabel('episode'); ax.set_ylabel('signed contribution')
    ax.set_xticks(x)
    ax.legend(loc='best', fontsize=8, ncol=2)

    # Suptitle: the reward shape this run was trained against, so reading
    # the plot doesn't require cross-referencing config.json.
    sf = cfg['success_factor']
    dr = cfg.get('dist_reward_coef', 0.0)
    tb = cfg.get('threading_bonus_coef', 0.0)
    extras = ''
    if dr:
        extras += f'  dr={dr:g}'
    if tb:
        extras += f'  tb={tb:g}'
    frm = cfg.get('final_reward_mult')
    if frm not in (None, 'none', ''):
        extras += f'  frm={float(frm):g}'
    title = (
        f'Reward decomposition  ({title_extra})\n'
        f'obs_mode={cfg["obs_mode"]}  '
        f'sf={sf}  sb={cfg["success_bonus"]:g}  '
        f'fp={cfg["fail_penalty"]:g}  vp={cfg["vel_penalty"]:g}  '
        f'ap={cfg["action_penalty"]:g}  psc={cfg["pre_settle_coef"]:g}'
        f'{extras}'
    )
    fig.suptitle(title, fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(out_png, dpi=130)
    plt.close(fig)
    print(f'[plot] wrote {out_png}')


def write_csv(episodes, out_csv):
    fields = ['episode', 'step', 'reward', 'base',
              'action_pen', 'vel_pen', 'pre_settle_pen',
              'terminal_shaping', 'dist_reward', 'threading_bonus',
              'is_success']
    with open(out_csv, 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(fields)
        for ep_i, rec in enumerate(episodes):
            for s in range(len(rec['step'])):
                w.writerow([ep_i, rec['step'][s],
                            f'{rec["reward"][s]:.6f}',
                            f'{rec["base"][s]:.6f}',
                            f'{rec["action_pen"][s]:.6f}',
                            f'{rec["vel_pen"][s]:.6f}',
                            f'{rec["pre_settle_pen"][s]:.6f}',
                            f'{rec["terminal_shaping"][s]:.6f}',
                            f'{rec["dist_reward"][s]:.6f}',
                            f'{rec["threading_bonus"][s]:.6f}',
                            rec['is_success']])
    print(f'[csv]  wrote {out_csv}')


def print_summary(episodes, cfg):
    print('\n' + '=' * 72)
    print('  EPISODE SUMMARY')
    print('=' * 72)
    print(f'  reward shape: sf={cfg["success_factor"]} '
          f'sb={cfg["success_bonus"]} fp={cfg["fail_penalty"]} '
          f'vp={cfg["vel_penalty"]} ap={cfg["action_penalty"]} '
          f'psc={cfg["pre_settle_coef"]}')
    print(f'  {"ep":>3} {"len":>4} {"sum":>10} {"base_sum":>10} '
          f'{"act":>8} {"vel":>8} {"psc":>8} {"shp":>8} {"succ":>4} {"dist":>8}')
    for i, rec in enumerate(episodes):
        d = rec['adaptive_dist']
        print(f'  {i:3d} {len(rec["step"]):4d} '
              f'{sum(rec["reward"]):10.2f} {sum(rec["base"]):10.2f} '
              f'{sum(rec["action_pen"]):8.2f} '
              f'{sum(rec["vel_pen"]):8.2f} '
              f'{sum(rec["pre_settle_pen"]):8.2f} '
              f'{sum(rec["terminal_shaping"]):8.2f} '
              f'{rec["is_success"]:4d} '
              f'{("nan" if d is None else f"{d:.3f}"):>8}')
    sums = np.array([sum(r['reward']) for r in episodes])
    succ = np.array([r['is_success'] for r in episodes])
    print(f'  --- mean Σreward = {sums.mean():.2f} '
          f'± {sums.std():.2f}   '
          f'success_rate = {succ.mean():.2f} ({succ.sum()}/{len(succ)})')
    print('=' * 72 + '\n')


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parsed = parse_args()

    # Resolve input source -> (cfg, output dir, title). We pre-allocate the
    # timestamped output paths BEFORE running rollouts so the video dir for
    # in-rollout writers shares the same suffix as the chart/csv.
    if parsed.checkpoint:
        ckpt = parsed.checkpoint
        cfg = load_run_config(ckpt)
        title_extra = f'ckpt: {Path(ckpt).name}'
        out_dir = parsed.out_dir or ckpt
        source = 'checkpoint'
    elif parsed.scripted:
        # Scripted demos don't need a config source — start from defaults
        # and let the user override reward shape via --override_*.
        cfg = {
            'obs_mode': 'hole_centroid',
            'success_factor': 1.2,  # matches view_demo.py's default
            'success_bonus': 0.0, 'fail_penalty': 0.0,
            'vel_penalty': 0.0, 'action_penalty': 0.0,
            'pre_settle_coef': 0.0,
            'dist_reward_coef': 0.0, 'threading_bonus_coef': 0.0,
            'max_episode_len': 200, 'seed': parsed.seed,
        }
        title_extra = f'scripted ({parsed.n_episodes} ep)'
        out_dir = (parsed.out_dir
                   or str(REPO_ROOT / 'logs' / 'hang_obs_exp' /
                          'reward_decomp_scripted'))
        source = 'scripted'
    elif parsed.demo_pkl:
        demo_paths = [parsed.demo_pkl]
        ref_pkl = parsed.demo_pkl
        base_dir = str(Path(parsed.demo_pkl).parent)
        title_extra = f'demo: {Path(parsed.demo_pkl).name}'
        cfg = load_demo_config(ref_pkl)
        out_dir = parsed.out_dir or base_dir
        source = 'demo_pkl'
    else:
        demo_paths = sorted(
            str(p) for p in Path(parsed.demo_dir).glob('demo_*.pkl'))
        if parsed.n_episodes:
            demo_paths = demo_paths[:parsed.n_episodes]
        if not demo_paths:
            raise SystemExit(f'no demo_*.pkl in {parsed.demo_dir}')
        ref_pkl = demo_paths[0]
        title_extra = (f'demos: {Path(parsed.demo_dir).name} '
                       f'({len(demo_paths)} ep)')
        cfg = load_demo_config(ref_pkl)
        out_dir = parsed.out_dir or parsed.demo_dir
        source = 'demo_dir'

    cfg = apply_overrides(cfg, parsed)
    _patch_max_act_vel(cfg)

    os.makedirs(out_dir, exist_ok=True)
    ts = datetime.now().strftime('%y%m%d_%H%M%S')
    out_png = os.path.join(out_dir, f'reward_decomp_{ts}.png')
    out_csv = os.path.join(out_dir, f'reward_decomp_{ts}.csv')
    video_dir = (os.path.join(out_dir, f'reward_decomp_{ts}_videos')
                 if parsed.save_videos else None)
    if video_dir is not None and cv2 is None:
        print('[video] cv2 not installed; install opencv-python or pass '
              '--no_videos. Continuing with videos disabled.')
        video_dir = None
    if video_dir is not None:
        os.makedirs(video_dir, exist_ok=True)
        print(f'[video] writing per-episode mp4s to {video_dir}')

    if source == 'checkpoint':
        episodes = rollout_policy(
            ckpt, cfg, parsed.n_episodes, parsed.deterministic,
            video_dir=video_dir,
            video_resolution=parsed.video_resolution,
            video_fps=parsed.video_fps)
    elif source == 'scripted':
        episodes = rollout_scripted(
            cfg, parsed.n_episodes,
            video_dir=video_dir,
            video_resolution=parsed.video_resolution,
            video_fps=parsed.video_fps)
    else:  # demo_dir / demo_pkl
        episodes = rollout_demos(
            demo_paths, cfg,
            video_dir=video_dir,
            video_resolution=parsed.video_resolution,
            video_fps=parsed.video_fps)

    plot_decomposition(episodes, cfg, out_png, title_extra=title_extra)
    write_csv(episodes, out_csv)
    print_summary(episodes, cfg)


def _patch_max_act_vel(cfg):
    """If the cfg carries a numeric max_act_vel override, set it on the
    DeformEnv class (matches how train_privileged.py applies the flag —
    one class-attribute write propagates everywhere)."""
    mav = cfg.get('max_act_vel')
    if mav not in (None, 'none', '', 'auto'):
        try:
            from dedo.envs.deform_env import DeformEnv
            old = DeformEnv.MAX_ACT_VEL
            DeformEnv.MAX_ACT_VEL = float(mav)
            print(f'[init] DeformEnv.MAX_ACT_VEL: {old} -> '
                  f'{DeformEnv.MAX_ACT_VEL}')
        except Exception as e:
            print(f'[warn] failed to patch MAX_ACT_VEL: {e!r}')

    # Also handle final_reward_mult patching here so a single helper
    # covers both DeformEnv class-attribute overrides.
    frm = cfg.get('final_reward_mult')
    if frm not in (None, 'none', ''):
        try:
            from dedo.envs.deform_env import DeformEnv
            old = DeformEnv.FINAL_REWARD_MULT
            DeformEnv.FINAL_REWARD_MULT = float(frm)
            print(f'[init] DeformEnv.FINAL_REWARD_MULT: {old} -> '
                  f'{DeformEnv.FINAL_REWARD_MULT}')
        except Exception as e:
            print(f'[warn] failed to patch FINAL_REWARD_MULT: {e!r}')


if __name__ == '__main__':
    main()
