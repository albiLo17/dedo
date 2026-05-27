"""
Train a diffusion-policy behavior-cloning model on dedo HangProcCloth-v1
across observation modalities.

Three obs modes share the same training pipeline, dataset format, and
in-env eval — only the obs encoder swaps:

  --obs_mode state   privileged 18-dim hole_centroid (or whichever
                     --state_key was recorded). Identity encoder.
  --obs_mode rgb     ResNet-18 (GroupNorm) on rendered RGB.
  --obs_mode pcd     PointNet++ (SSG) on back-projected world-coord PCD.

Reads demos written by collect_bc_demos.py:
  python experiments/hang_obs_exp/scripts/collect_bc_demos.py ...
  python experiments/hang_obs_exp/scripts/train_diffusion_bc.py \
      --demo_path logs/hang_obs_exp/bc_demos_v1 \
      --obs_mode state \
      --num_epochs 100 --use_wandb

Outputs (under <logdir_root>/<obs_mode>/<run_id>/):
  policy.pt          — diffusion U-Net + obs encoder + EMA weights
  obs_normalizer.pkl — fitted obs normalizer (so eval can match training)
  config.json        — flags + git rev for reproducibility
"""
from __future__ import annotations

import argparse
import collections
import json
import math
import os
import pickle
import sys
import time
from copy import deepcopy
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT))

import gym
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

import dedo  # noqa: F401
from dedo.envs.deform_env import DeformEnv
from dedo.utils.args import get_args_parser, args_postprocess
from dedo.utils.mesh_utils import get_mesh_data
from dedo.demo_preset import build_traj, merge_traj  # noqa: F401

# CRITICAL parity invariant: training-time actions were clipped by the
# MAX_ACT_VEL active during demo collection (stored per-pkl as
# `max_act_vel`). If the eval env runs under dedo's default 10.0 while
# demos were collected at e.g. 3.5, an action of 0.5 in [-1, 1] becomes
# 5.0 m/s at eval vs 1.75 m/s in training — the gripper moves ~3x too
# fast and every eval episode flies off the trajectory. We read it from
# the demos below and patch DeformEnv.MAX_ACT_VEL before building the
# eval env.

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _helpers import RetryResetEnv, compute_per_episode_max_len  # noqa: E402

# Reuse the camera and success-check helpers so eval-time RGB/PCD/success
# match collection-time bit-for-bit.
from _bc_obs_helpers import (  # noqa: E402
    capture_rgb_depth, depth_to_pcd, proj_matrix,
    check_hanging_on_peg, check_threaded_topological, check_legacy,
    measure_hole_radius, get_hole_indices, get_hole_loops,
    resolve_deform, patch_deform_render_to_obs_camera)

from experiments.hang_obs_exp.envs.privileged_env import (  # noqa: E402
    build_privileged_obs, identify_cloth_corners)

from _diffusion_policy import (  # noqa: E402
    DiffusionPolicy, build_encoder, ObsNormalizer, ActionNormalizer)


PRIV_STATE_MODES = ('hole_centroid', 'hole_centroid_corners',
                    'hole_vertices', 'full_mesh')


# =============================================================================
# Argparse
# =============================================================================
parser = argparse.ArgumentParser()
parser.add_argument('--demo_path', type=str, required=True,
                    help='Directory of demo_NNN.pkl files written by '
                         'collect_bc_demos.py.')
parser.add_argument('--obs_mode', type=str, default='state',
                    choices=['state', 'rgb', 'pcd'],
                    help='Which obs modality to train on. Each demo pkl '
                         'contains all three; this picks the key to load.')
parser.add_argument('--state_key', type=str, default='hole_centroid',
                    choices=list(PRIV_STATE_MODES),
                    help='Which privileged-state field to use when '
                         '--obs_mode=state. 18-dim hole_centroid is the '
                         'default; larger fields give more info but more '
                         'params to fit.')
parser.add_argument('--pretrained_rgb', action='store_true',
                    help='When --obs_mode=rgb, initialize the ResNet-18 '
                         'backbone from ImageNet weights instead of from '
                         'Kaiming init. The subsequent BN->GN swap re-'
                         'initializes the normalization layers, but the '
                         'conv stack (the bulk of the transferable signal) '
                         'is preserved. Ignored for state/pcd modes — '
                         'PointNet++ pretraining options (ShapeNet etc.) '
                         'transfer poorly to deformable cloth.')
parser.add_argument('--logdir_root', type=str,
                    default=str(REPO_ROOT / 'logs' / 'hang_obs_exp' /
                                'diffusion_bc'))
parser.add_argument('--run_name', type=str, default=None,
                    help='Subdirectory name under <logdir_root>/<obs_mode>. '
                         'Default: timestamp + obs_mode + seed.')
parser.add_argument('--seed', type=int, default=2026)

# Filter
parser.add_argument('--only_success', action='store_true', default=True,
                    help='Default: keep only demos with success=1 (under '
                         '--success_metric). Pass --no_only_success to use '
                         'all.')
parser.add_argument('--no_only_success', dest='only_success',
                    action='store_false')
parser.add_argument('--success_metric', type=str, default='hanging',
                    choices=['hanging', 'topological', 'legacy'],
                    help='Which per-demo success field gates filtering '
                         '(success_hanging / _topological / _legacy in pkl).')

# Diffusion-policy hyperparameters (defaults match pusht demo).
parser.add_argument('--obs_horizon', type=int, default=2)
parser.add_argument('--pred_horizon', type=int, default=16)
parser.add_argument('--action_horizon', type=int, default=8)
parser.add_argument('--num_diffusion_iters', type=int, default=100,
                    help='DDPM train timesteps. Inference iterates over '
                         'the same number of steps (could be reduced for '
                         'faster eval but kept = train_iters by default).')

# Training
parser.add_argument('--num_epochs', type=int, default=100)
parser.add_argument('--batch_size', type=int, default=64)
parser.add_argument('--lr', type=float, default=1e-4)
parser.add_argument('--weight_decay', type=float, default=1e-6)
parser.add_argument('--num_warmup_steps', type=int, default=500)
parser.add_argument('--ema_power', type=float, default=0.75)
parser.add_argument('--num_workers', type=int, default=0,
                    help='DataLoader workers. 0 = main thread; safer on '
                         'macOS where multiprocessing pickling sometimes '
                         'chokes on torch tensors.')

# Eval
parser.add_argument('--eval_every_epochs', type=int, default=10,
                    help='Run an in-env eval pass every N epochs (set '
                         '0 to disable mid-training eval).')
parser.add_argument('--n_eval_episodes', type=int, default=10,
                    help='Episodes per mid-training eval pass. SE on '
                         'success_rate at p=0.5 is ~0.16 at n=10, ~0.09 '
                         'at n=30.')
parser.add_argument('--n_final_eval_episodes', type=int, default=50)
parser.add_argument('--video_every_evals', type=int, default=1,
                    help='Capture an eval video every Nth eval pass and '
                         'log it to wandb + save alongside the logdir. '
                         '0 = disabled. 1 = every eval pass (default). '
                         'Each recorded episode adds ~6-10 s of '
                         'capture+encode time; only the first '
                         '--n_video_episodes episodes are recorded per '
                         'eval pass, the remaining ones run unobserved.')
parser.add_argument('--n_video_episodes', type=int, default=3,
                    help='Number of episodes to record per video pass. '
                         'They get concatenated into a single mp4 — '
                         'different clothes within a single eval pass '
                         'because env.reset() advances the procgen sampler. '
                         '(The eval seed is locked across passes, so '
                         'episode 1 of every video shows the same cloth, '
                         'good for tracking learning on a fixed example. '
                         'Higher N shows variety; 3 is the sweet spot '
                         'between cloth variety and eval wall-clock.)')
parser.add_argument('--video_render_size', type=int, default=300,
                    help='Per-frame render H=W (square) for the video. '
                         'Independent of the policy obs resolution — '
                         'this is just what the wandb player shows. '
                         'Matches HangVideoCallback default (PPO runs).')
parser.add_argument('--video_fps', type=int, default=None,
                    help='Playback FPS for eval-rollout MP4s. Default '
                         '(None) auto-sets to the demos\' recorded '
                         'ctrl_freq so playback matches sim wall-clock '
                         'time — same convention as collect_bc_demos.py\'s '
                         '--debug_fps. Pass an explicit int to override '
                         '(e.g. 30 for smoother scrubbing at 2x sim speed; '
                         'collect-time debug videos use 15 by default so '
                         'matching keeps eval and collection videos '
                         'visually comparable).')
parser.add_argument('--settle_frame_stride', type=int, default=2,
                    help='Sub-sample the post-settle gravity-phase frames '
                         'captured inside make_final_steps. 2 matches '
                         'collect_bc_demos.py debug-video stride (~7-8 '
                         'settle frames at 15 Hz / ~15 at 62.5 Hz) so the '
                         'cloth-drape phase is visible in eval videos.')
parser.add_argument('--save_every_epochs', type=int, default=0,
                    help='Save a numbered policy_ep<NNNN>.pt checkpoint '
                         'after each eval pass that lands on this interval, '
                         'in addition to the always-saved final policy.pt. '
                         '0 = disabled. Recommended: match '
                         '--eval_every_epochs so every logged eval point '
                         'has a recoverable policy. Note: best-eval '
                         'tracking (policy_best.pt) is ALWAYS enabled '
                         'regardless of this flag — every eval pass '
                         'compares against the running best and mirrors '
                         'the EMA weights to policy_best.pt on improvement.')
parser.add_argument('--resume', type=str, default=None,
                    help='Path to a checkpoint to resume training from. '
                         'Loads weights, EMA shadow, optimizer state, LR '
                         'scheduler state, and resumes the epoch counter. '
                         'Falls back gracefully if the ckpt was saved by an '
                         'older version that did not persist optimizer/ema/'
                         'scheduler state (a warning is printed and those '
                         'components start fresh). Architecture flags '
                         '(obs_horizon/pred_horizon/etc.) must match the '
                         'ckpt — load_state_dict will error out on shape '
                         'mismatch.')
parser.add_argument('--eval_only', action='store_true',
                    help='Skip training entirely; load --resume checkpoint '
                         'and only run the final eval ('
                         '--n_final_eval_episodes episodes). Useful for '
                         're-scoring a checkpoint after fixing eval-time '
                         'bugs (e.g. demo_speed time extension, action '
                         'clip tightening) without paying for retraining. '
                         'Requires --resume. Per-epoch eval (and '
                         'policy_best.pt tracking) is also skipped — '
                         'only the final-eval block at the end runs.')
parser.add_argument('--eval_all_in', type=str, default=None,
                    help='Directory containing policy_ep<NNNN>.pt '
                         'checkpoints. When set, skips training and runs '
                         'eval (--n_eval_episodes episodes) on each '
                         'checkpoint in epoch order, logging metrics to '
                         'wandb keyed by epoch — recreates the '
                         'mid-training eval curve with updated eval-time '
                         'code (demo_speed time extension, video_fps fix). '
                         'Reuses the same env build across checkpoints '
                         'inside the loop so it\'s faster than re-running '
                         'the script per checkpoint. Mutually exclusive '
                         'with --resume; implies --eval_only.')
parser.add_argument('--max_episode_len', type=int, default=200,
                    help='Safety ceiling on per-episode length. The actual '
                         'per-episode cap at eval is the scripted-traj '
                         'length + --episode_tail_frames (mirrors '
                         'collect_bc_demos.py), with this value as upper '
                         'bound.')
parser.add_argument('--episode_tail_frames', type=int, default=5,
                    help='Brake-tail frames appended after the scripted '
                         'trajectory; the eval env\'s per-episode '
                         'max_episode_len = len(traj) + tail. Must match '
                         'the value used at demo collection time (default '
                         '5 in collect_bc_demos.py).')
parser.add_argument('--eval_cam_resolution', type=int, default=None,
                    help='Camera resolution at eval time. Default = the '
                         'resolution recorded in the demo pkls. Override '
                         'only if you specifically want to test cross-'
                         'resolution generalization.')
parser.add_argument('--success_factor', type=float, default=1.2,
                    help='Adaptive success threshold for eval-time '
                         'criterion. Should match the value collect_bc_'
                         'demos.py used; the script warns on mismatch.')
parser.add_argument('--ctrl_freq', type=float, default=15.0,
                    help='Control frequency (Hz) for the eval env. By '
                         'default the demos\' recorded ctrl_freq overrides '
                         'this (so eval matches collection). For legacy '
                         'pkls with no ctrl_freq field, this value is used '
                         'as a fallback. Implemented via '
                         'sim_steps_per_action = round(sim_freq/ctrl_freq).')
parser.add_argument('--sim_freq', type=int, default=500,
                    help='PyBullet physics frequency for the eval env. '
                         'Default 500 matches dedo. Only used together '
                         'with --ctrl_freq when demos lack a recorded '
                         'ctrl_freq, or to confirm parity for demos '
                         'collected under a non-default sim_freq.')

# Eval-env seed strategy
parser.add_argument('--eval_seed_offset', type=int, default=9999,
                    help='Eval env is seeded to args.seed + offset, so it '
                         'evaluates a fixed set of procedural cloths each '
                         'eval pass — eval-rate trace then reflects only '
                         'policy change, not env resampling.')

# wandb
parser.add_argument('--use_wandb', action='store_true')
parser.add_argument('--wandb_project', type=str, default='hang_bc_diffusion')
parser.add_argument('--wandb_run_name', type=str, default=None)

# Device
parser.add_argument('--device', type=str, default=None,
                    help='Default: cuda if available, else mps if Apple '
                         'silicon, else cpu.')

args = parser.parse_args()

if args.eval_all_in and args.resume:
    parser.error('--eval_all_in and --resume are mutually exclusive; '
                 '--eval_all_in iterates over all policy_ep*.pt in the dir.')
if args.eval_all_in:
    # --eval_all_in always skips training; the per-checkpoint loop replaces
    # the single final-eval block.
    args.eval_only = True
elif args.eval_only and not args.resume:
    parser.error('--eval_only requires --resume to point at a checkpoint.')


# =============================================================================
# Setup
# =============================================================================
np.random.seed(args.seed)
torch.manual_seed(args.seed)

if args.device is None:
    # Prefer CUDA; on Apple silicon, fall back to CPU rather than MPS — MPS
    # silently NaNs the diffusion-policy U-Net (FiLM-modulated Conv1d kernels
    # hit MPS bugs as of torch 2.4). Pass --device mps explicitly to override.
    args.device = 'cuda' if torch.cuda.is_available() else 'cpu'
device = torch.device(args.device)
print(f'[init] device = {device}')

def _fmt_lr(lr):
    """Compact lr format matching _helpers.build_run_name_suffix."""
    s = f'{float(lr):.0e}'  # '1e-04'
    mantissa, _, exp = s.partition('e')
    sign = '-' if exp.startswith('-') else ''
    exp_num = exp.lstrip('+-').lstrip('0') or '0'
    return f'{mantissa}e{sign}{exp_num}'


def build_diffusion_run_suffix(a):
    """Encode all experimental dials into the wandb run name, matching
    train_privileged.py's build_run_name_suffix style. Defaults are
    omitted so baselines stay short and only knobs the user actually
    turned show up in the name."""
    parts = [f'_{a.obs_mode}']
    if a.obs_mode == 'state' and a.state_key != 'hole_centroid':
        parts.append(f'_{a.state_key}')
    if a.obs_mode == 'rgb' and a.pretrained_rgb:
        parts.append('_pre')
    parts.append(f'_lr{_fmt_lr(a.lr)}')
    parts.append(f'_e{a.num_epochs}')
    parts.append(f'_bs{a.batch_size}')
    if a.num_diffusion_iters != 100:
        parts.append(f'_di{a.num_diffusion_iters}')
    if a.pred_horizon != 16:
        parts.append(f'_ph{a.pred_horizon}')
    if a.obs_horizon != 2:
        parts.append(f'_oh{a.obs_horizon}')
    if a.action_horizon != 8:
        parts.append(f'_ah{a.action_horizon}')
    if a.success_metric != 'hanging':
        parts.append(f'_sm-{a.success_metric}')
    if not a.only_success:
        parts.append('_all-demos')
    parts.append(f'_s{a.seed}')
    return ''.join(parts)


_run_suffix = build_diffusion_run_suffix(args)
run_name = args.run_name or (f'diff{time.strftime("%y%m%d-%H%M%S")}'
                              f'{_run_suffix}')
logdir = os.path.join(args.logdir_root, args.obs_mode, run_name)
os.makedirs(logdir, exist_ok=True)
print(f'[init] logdir = {logdir}')

if args.use_wandb:
    import wandb
    # wandb run name: user-provided prefix + the suffix (matches the
    # train_privileged.py pattern so cross-script runs sort comparably).
    if args.wandb_run_name:
        wandb_run_name = args.wandb_run_name + _run_suffix
    else:
        wandb_run_name = run_name
    wandb.init(project=args.wandb_project,
               name=wandb_run_name, config=vars(args))
    # Tags make it easy to filter the runs table by obs_mode and metric.
    wandb.run.tags = list(wandb.run.tags or []) + [
        f'obs_mode={args.obs_mode}',
        f'success_metric={args.success_metric}',
        f'seed={args.seed}',
        f'algo=diffusion_bc',
    ]
    if args.obs_mode == 'state':
        wandb.run.tags = list(wandb.run.tags) + [f'state_key={args.state_key}']


# =============================================================================
# Load demos
# =============================================================================
print(f'\n=== Loading demos from {args.demo_path} ===')
demo_paths = sorted(p for p in os.listdir(args.demo_path)
                    if p.startswith('demo_') and p.endswith('.pkl'))
if not demo_paths:
    raise FileNotFoundError(f'no demo_*.pkl in {args.demo_path}')

succ_key_map = {'hanging': 'success_hanging',
                'topological': 'success_topological',
                'legacy': 'success_legacy'}
succ_key = succ_key_map[args.success_metric]

# Primary obs (state vector / image / pcd) per timestep, plus grip for
# the RGB and PCD modes. State mode keeps grip embedded in its vector;
# RGB and PCD modes load it as a separate auxiliary input so the encoder
# can concatenate visual features with proprioception (matches
# PixelObsWrapper / PointCloudObsWrapper convention).
#
# RGB and PCD modes additionally load a 3-dim `goal` vector (hanger
# pose), parallel to grip. State mode's privileged vector already has
# goal embedded in its last 3 dims — adding it as a separate auxiliary
# input to the visual modes equalizes goal information across the
# three-way comparison so the only modality-specific knowledge gap is
# hole-centroid extraction (the actual privileged advantage).
obs_buf: list = []
grip_buf: list = []
goal_buf: list = []
act_buf: list = []
ep_ends: list = []      # cumulative one-past-end indices
n_skipped = 0
n_total = 0
recorded_cam_resolutions = set()
recorded_pcd_n_points = set()
recorded_success_factors = set()
# Tracking the camera so eval matches collection exactly. cam_viewmat is
# stored in every pkl as [dist, pitch, yaw, tx, ty, tz]; tuple-ifying lets
# us put it in a set to detect mixed-camera demo dirs.
recorded_cam_viewmats: set = set()
recorded_max_act_vels: set = set()  # critical parity invariant; see import block
recorded_ctrl_freqs: set = set()    # control-freq parity (same kind of invariant
                                    # as max_act_vel: trajectory was recorded at
                                    # this Hz; eval env must match or actions
                                    # play back at the wrong speed)
recorded_sim_freqs: set = set()
recorded_sim_steps_per_action: set = set()
# Hanger-goal randomization radius. Treated like max_act_vel: comes from the
# demo pkls (not a CLI flag) since the policy trained on a specific spatial
# distribution and eval must mirror it. Mixed values raise.
recorded_randomize_goal_radii: set = set()
# Demo-speed slowdown factor (1.0 = unmodified, 0.5 = half-speed commanded
# velocities with 2x trajectory length). Demos with non-1.0 demo_speed have
# more steps per episode AND smaller per-step displacements, so the eval env
# needs a proportionally larger max_episode_len to give the policy enough
# time to complete the motion. Treated like max_act_vel: pulled from pkls,
# parity-checked, no CLI override.
recorded_demo_speeds: set = set()
# Per-demo episode lengths. Demos are collected with deform.max_episode_len
# set per-episode to (traj_len + episode_tail_frames) — typically ~51 ctrl
# steps at 15 Hz — so the training distribution covers only this window.
# Eval must terminate inside (or very close to) the same window or the
# policy runs OOD for the tail of every episode and drifts arbitrarily.
recorded_episode_lengths: list = []
needs_grip = args.obs_mode in ('rgb', 'pcd')  # state already has grip
needs_goal = args.obs_mode in ('rgb', 'pcd')  # state already has goal
n_missing_grip = 0
n_missing_goal = 0

for fname in demo_paths:
    path = os.path.join(args.demo_path, fname)
    with open(path, 'rb') as f:
        d = pickle.load(f)
    n_total += 1
    # Filter by success metric.
    if args.only_success:
        sval = d.get(succ_key, d.get('success', 0))
        if not int(sval):
            n_skipped += 1
            continue
    # Pick the obs key.
    if args.obs_mode == 'state':
        if args.state_key not in d['obs']:
            print(f'[demo] {fname}: missing state key {args.state_key!r}, '
                  f'skipping')
            n_skipped += 1
            continue
        obs = d['obs'][args.state_key]
    else:
        if args.obs_mode not in d['obs']:
            print(f'[demo] {fname}: missing obs key {args.obs_mode!r}, '
                  f'skipping (re-collect with collect_bc_demos.py)')
            n_skipped += 1
            continue
        obs = d['obs'][args.obs_mode]
    acts = d['acts']
    if len(obs) != len(acts):
        print(f'[demo] {fname}: obs/act length mismatch '
              f'({len(obs)} vs {len(acts)}), skipping')
        n_skipped += 1
        continue
    obs_buf.append(np.asarray(obs))
    act_buf.append(np.asarray(acts, dtype=np.float32))
    recorded_episode_lengths.append(int(len(acts)))
    # Auxiliary gripper proprioception (only for RGB/PCD; state mode has
    # it baked into its 18-dim vector already).
    if needs_grip:
        grip_arr = d['obs'].get('grip')
        if grip_arr is None:
            # Legacy pkl without grip — fall back to zeros and warn.
            grip_arr = np.zeros((len(acts), 12), dtype=np.float32)
            n_missing_grip += 1
        grip_buf.append(np.asarray(grip_arr, dtype=np.float32))
    # Auxiliary hanger-goal vector (only for RGB/PCD; state mode has it
    # baked into the last 3 dims of its 18-dim vector already).
    if needs_goal:
        goal_arr = d['obs'].get('goal')
        if goal_arr is None:
            # Legacy pkl without goal — fall back to zeros and warn.
            goal_arr = np.zeros((len(acts), 3), dtype=np.float32)
            n_missing_goal += 1
        goal_buf.append(np.asarray(goal_arr, dtype=np.float32))
    # Track metadata used for eval-env construction.
    if 'cam_resolution' in d:
        recorded_cam_resolutions.add(int(d['cam_resolution']))
    if 'pcd_n_points' in d:
        recorded_pcd_n_points.add(int(d['pcd_n_points']))
    if 'success_factor' in d:
        recorded_success_factors.add(float(d['success_factor']))
    if 'cam_viewmat' in d:
        recorded_cam_viewmats.add(tuple(float(x) for x in d['cam_viewmat']))
    if 'max_act_vel' in d:
        recorded_max_act_vels.add(float(d['max_act_vel']))
    # round to 4 dp so 15.151515... and 15.15151515 don't trigger a false
    # mismatch across demos recorded with the same intent.
    if 'ctrl_freq' in d:
        recorded_ctrl_freqs.add(round(float(d['ctrl_freq']), 4))
    if 'sim_freq' in d:
        recorded_sim_freqs.add(int(d['sim_freq']))
    if 'sim_steps_per_action' in d:
        recorded_sim_steps_per_action.add(int(d['sim_steps_per_action']))
    if 'randomize_goal_radius' in d:
        # round to 6 dp so float-repr noise across collection runs doesn't
        # produce false mismatches.
        recorded_randomize_goal_radii.add(
            round(float(d['randomize_goal_radius']), 6))
    if 'demo_speed' in d:
        # round to 6 dp; demo_speed is always 1/N (collected as
        # 1.0 / int(round(1/speed))) so this is plenty.
        recorded_demo_speeds.add(round(float(d['demo_speed']), 6))
    ep_ends.append(sum(len(a) for a in act_buf))

if not obs_buf:
    raise RuntimeError(
        f'no usable demos after filter (only_success={args.only_success}, '
        f'metric={args.success_metric}). Got {n_total} files, skipped '
        f'{n_skipped}.')

obs_all = np.concatenate(obs_buf, axis=0)
acts_all = np.concatenate(act_buf, axis=0)
grip_all = (np.concatenate(grip_buf, axis=0)
            if needs_grip and grip_buf else None)
goal_all = (np.concatenate(goal_buf, axis=0)
            if needs_goal and goal_buf else None)
episode_ends = np.asarray(ep_ends, dtype=np.int64)
print(f'[data] kept {len(obs_buf)}/{n_total} demos  '
      f'(skipped {n_skipped})')
print(f'[data] obs  shape = {obs_all.shape}  dtype={obs_all.dtype}')
if grip_all is not None:
    print(f'[data] grip shape = {grip_all.shape}  '
          f'(grip is auxiliary; concat with visual features in encoder)')
    if n_missing_grip > 0:
        print(f'[data] WARN: {n_missing_grip} demo pkl(s) missing the '
              f'"grip" key. Used zero-grip fallback for those demos — '
              f're-collect with the updated collect_bc_demos.py to fix.')
if goal_all is not None:
    print(f'[data] goal shape = {goal_all.shape}  '
          f'(hanger pose; concat with visual+grip features in encoder)')
    if n_missing_goal > 0:
        print(f'[data] WARN: {n_missing_goal} demo pkl(s) missing the '
              f'"goal" key. Used zero-goal fallback for those demos — '
              f're-collect with the updated collect_bc_demos.py to fix.')
print(f'[data] act  shape = {acts_all.shape}')

# Sanity checks across demo pkls. Mismatch on any of these would cause
# silent eval/training disagreement, so we fail loudly (the user can
# always recollect into a clean directory).
if len(recorded_cam_resolutions) > 1:
    raise RuntimeError(
        f'demos use mixed cam_resolution {recorded_cam_resolutions}. '
        f'Image shapes would not concatenate and eval-time renders would '
        f'differ from training. Re-collect into a clean directory.')
if len(recorded_pcd_n_points) > 1:
    raise RuntimeError(
        f'demos use mixed pcd_n_points {recorded_pcd_n_points}. The PCD '
        f'encoder fixes n_points at construction; mixed values would '
        f'silently misshape eval-time obs. Re-collect into a clean directory.')
if (len(recorded_success_factors) > 0
        and abs(min(recorded_success_factors) - args.success_factor) > 1e-6):
    print(f'[data] WARN: demos collected at success_factor='
          f'{recorded_success_factors}, eval will use '
          f'success_factor={args.success_factor}')
collection_cam_res = (next(iter(recorded_cam_resolutions))
                      if recorded_cam_resolutions else 96)
collection_pcd_n_pts = (next(iter(recorded_pcd_n_points))
                        if recorded_pcd_n_points else 512)
eval_cam_resolution = args.eval_cam_resolution or collection_cam_res

# Pick the cam_viewmat the eval env will use. If the demos all agree, take
# theirs (this is the right behavior — eval should mirror collection). If
# they disagree, warn and take the first one alphabetically (deterministic).
# If no demo records a viewmat (legacy pkls), fall back to the same default
# collect_bc_demos.py now uses, so eval and a fresh re-collect would agree.
_DEFAULT_VIEWMAT = (14.0, -5.0, 45.0, 0.0, 0.0, 5.5)
if len(recorded_cam_viewmats) == 0:
    eval_cam_viewmat = _DEFAULT_VIEWMAT
    print(f'[data] no cam_viewmat in demos (legacy pkls); using default '
          f'{eval_cam_viewmat}')
elif len(recorded_cam_viewmats) == 1:
    eval_cam_viewmat = next(iter(recorded_cam_viewmats))
    print(f'[data] eval cam_viewmat (from demos) = {eval_cam_viewmat}')
else:
    eval_cam_viewmat = sorted(recorded_cam_viewmats)[0]
    print(f'[data] WARN: demos use mixed cam_viewmats '
          f'{sorted(recorded_cam_viewmats)}; eval will use '
          f'{eval_cam_viewmat} (lowest sort order)')

# CRITICAL: patch DeformEnv.MAX_ACT_VEL to match the value demos were
# collected under. This is a class attribute that dedo reads dynamically
# at every action unscale, so a single assignment here propagates to
# every env we build afterwards (training-time reuse of cached vec_env
# obs stats, the eval env below, the final-eval env). Without this, the
# eval env runs at dedo's default 10.0 while demos were collected at
# e.g. 3.5 — actions are silently 3x too fast at eval.
if len(recorded_max_act_vels) == 0:
    print(f'[data] no max_act_vel in demos (legacy pkls); leaving '
          f'DeformEnv.MAX_ACT_VEL at default {DeformEnv.MAX_ACT_VEL}. '
          f'If demos were collected with --max_act_vel != 10, eval will '
          f'be inconsistent. Re-collect to be safe.')
elif len(recorded_max_act_vels) == 1:
    eval_max_act_vel = next(iter(recorded_max_act_vels))
    _orig_mav = DeformEnv.MAX_ACT_VEL
    DeformEnv.MAX_ACT_VEL = eval_max_act_vel
    print(f'[data] eval MAX_ACT_VEL (from demos) = {eval_max_act_vel} '
          f'(patched from dedo default {_orig_mav})')
else:
    raise RuntimeError(
        f'[data] demos use mixed max_act_vel values '
        f'{sorted(recorded_max_act_vels)}. This is a parity hazard: '
        f'every demo\'s actions were normalized by a different value, '
        f'so the training data mixes incompatible action scales. '
        f'Re-collect into a clean directory with a single --max_act_vel.')


# ---------------------------------------------------------------------------
# Pick the eval env's ctrl_freq (sim_freq + sim_steps_per_action). Same parity
# story as MAX_ACT_VEL: demos were recorded at a specific Hz, and the eval env
# must replay actions at the same Hz or trajectories advance at the wrong
# speed (and at high ctrl_freq the gripper can't keep up with the lift phase
# in the time budget). Prefer demo-recorded values; fall back to CLI flags
# for legacy pkls.
# ---------------------------------------------------------------------------
if len(recorded_sim_steps_per_action) > 1:
    raise RuntimeError(
        f'[data] demos use mixed sim_steps_per_action values '
        f'{sorted(recorded_sim_steps_per_action)}. Demos at different '
        f'control frequencies advance through the recorded waypoint plan '
        f'at different speeds — training data is incompatible. '
        f'Re-collect into a clean directory at a single --ctrl_freq.')
if len(recorded_sim_freqs) > 1:
    raise RuntimeError(
        f'[data] demos use mixed sim_freq values '
        f'{sorted(recorded_sim_freqs)}. Re-collect into a clean directory.')

if recorded_sim_steps_per_action and recorded_sim_freqs:
    eval_sim_steps_per_action = next(iter(recorded_sim_steps_per_action))
    eval_sim_freq = next(iter(recorded_sim_freqs))
    eval_ctrl_freq = eval_sim_freq / eval_sim_steps_per_action
    print(f'[data] eval ctrl_freq (from demos) = {eval_ctrl_freq:.3f} Hz '
          f'(sim_freq={eval_sim_freq}, steps/action='
          f'{eval_sim_steps_per_action})')
    if recorded_ctrl_freqs:
        _saved_freq = next(iter(recorded_ctrl_freqs))
        if abs(_saved_freq - eval_ctrl_freq) > 1e-3:
            print(f'[data] WARN: demo ctrl_freq field {_saved_freq} Hz '
                  f'disagrees with sim_freq/sim_steps_per_action '
                  f'({eval_ctrl_freq:.3f} Hz). Using the derived value.')
else:
    # Legacy pkls without ctrl_freq metadata — fall back to CLI flags.
    eval_sim_freq = int(args.sim_freq)
    eval_sim_steps_per_action = max(
        1, int(round(args.sim_freq / args.ctrl_freq)))
    eval_ctrl_freq = eval_sim_freq / eval_sim_steps_per_action
    print(f'[data] no ctrl_freq in demos (legacy pkls); using CLI '
          f'--ctrl_freq={args.ctrl_freq} -> eval ctrl_freq='
          f'{eval_ctrl_freq:.3f} Hz (sim_freq={eval_sim_freq}, '
          f'steps/action={eval_sim_steps_per_action}). If demos were '
          f'collected at a different ctrl_freq, eval will be inconsistent. '
          f'Re-collect to be safe.')

# Auto-resolve --video_fps to match ctrl_freq so eval-rollout MP4s play
# back at real-time sim speed. Frame capture happens once per env.step()
# (i.e. at ctrl_freq), so encoding at video_fps == ctrl_freq is real-time.
# Encoding at video_fps > ctrl_freq makes the video play SPED UP — which
# is what was happening with the previous default of 30 (videos played at
# ~2x sim speed at ctrl_freq~15 Hz). When the user passes --video_fps
# explicitly, honor it.
if args.video_fps is None:
    args.video_fps = max(1, int(round(eval_ctrl_freq)))
    print(f'[data] video_fps auto-set to {args.video_fps} '
          f'(matches eval_ctrl_freq={eval_ctrl_freq:.3f} Hz for real-time '
          f'playback)')
else:
    _expected = max(1, int(round(eval_ctrl_freq)))
    if args.video_fps != _expected:
        _ratio = args.video_fps / eval_ctrl_freq
        print(f'[data] video_fps={args.video_fps} (CLI override); '
              f'eval ctrl_freq={eval_ctrl_freq:.3f} Hz -> playback runs '
              f'at {_ratio:.2f}x sim speed.')


# ---------------------------------------------------------------------------
# Pick the eval env's hanger-goal randomization radius. Same parity story as
# max_act_vel: demos were collected with a specific spatial distribution of
# peg positions and the policy fits that distribution; eval must mirror it
# or the policy is being scored on an OOD goal distribution. Mixed values
# across the demo dir = data is inhomogeneous and we refuse to silently mix.
# Legacy pkls (no field) are interpreted as the v3-and-earlier behavior:
# radius = 0 (fixed goal). The training script does not expose a CLI
# override — the value comes from the demos.
# ---------------------------------------------------------------------------
if len(recorded_randomize_goal_radii) > 1:
    raise RuntimeError(
        f'[data] demos use mixed randomize_goal_radius values '
        f'{sorted(recorded_randomize_goal_radii)}. The policy would be '
        f'trained on a mixture of spatial distributions and there is no '
        f'sensible single eval distribution to score it against. '
        f'Re-collect into a clean directory with a single '
        f'--randomize_goal_radius.')
if recorded_randomize_goal_radii:
    eval_randomize_goal_radius = next(iter(recorded_randomize_goal_radii))
    print(f'[data] eval randomize_goal_radius (from demos) = '
          f'{eval_randomize_goal_radius} m'
          f'{" (off — fixed goal)" if eval_randomize_goal_radius <= 0 else ""}')
else:
    eval_randomize_goal_radius = 0.0
    print(f'[data] no randomize_goal_radius in demos (legacy pkls); '
          f'using 0 (fixed goal, v3-and-earlier behavior)')


# ---------------------------------------------------------------------------
# Demo-speed parity. Same story as max_act_vel: the policy learned actions
# at the recorded speed (smaller magnitudes + more steps when demo_speed<1)
# so eval has to mirror it. Critical knob: eval's per-episode max length is
# scaled by 1/demo_speed so a 0.5-speed-collected policy gets 2x the steps
# to complete its motion. Without this, a slow policy runs out of time
# before reaching the goal — exactly the symptom the user observed.
# Legacy pkls (no field) are interpreted as demo_speed=1.0.
# ---------------------------------------------------------------------------
if len(recorded_demo_speeds) > 1:
    raise RuntimeError(
        f'[data] demos use mixed demo_speed values '
        f'{sorted(recorded_demo_speeds)}. Each value implies a different '
        f'commanded-velocity scale and a different episode-length budget; '
        f'mixing them would train the policy on inconsistent action '
        f'distributions. Re-collect into a clean directory with a single '
        f'--demo_speed.')
if recorded_demo_speeds:
    eval_demo_speed = next(iter(recorded_demo_speeds))
    print(f'[data] eval demo_speed (from demos) = {eval_demo_speed}'
          f'{" (default; no slowdown)" if eval_demo_speed == 1.0 else ""}')
else:
    eval_demo_speed = 1.0
    print(f'[data] no demo_speed in demos (legacy pkls); using 1.0 '
          f'(no slowdown).')


# ---------------------------------------------------------------------------
# Primary per-episode max_episode_len at eval is computed DYNAMICALLY in
# the eval loop (mirrors collect_bc_demos.py: build the scripted traj for
# the current cloth and cap at `len(traj) + episode_tail_frames`). The
# value below is the FALLBACK used when traj construction fails for a
# specific episode (degenerate hole geometry, NaN mesh, etc.) — set to a
# sensible "looks like the training distribution" length derived from the
# longest recorded demo + small buffer. args.max_episode_len remains the
# absolute safety ceiling that even the dynamic per-episode value can't
# exceed.
# ---------------------------------------------------------------------------
_EVAL_EP_LEN_BUFFER = 5  # ~333 ms at 15 Hz; plenty of brake-and-handoff slack
if recorded_episode_lengths:
    _max_demo_len = max(recorded_episode_lengths)
    _mean_demo_len = sum(recorded_episode_lengths) / len(recorded_episode_lengths)
    eval_max_episode_len = min(
        int(_max_demo_len + _EVAL_EP_LEN_BUFFER),
        int(args.max_episode_len))
    print(f'[data] eval max_episode_len FALLBACK = {eval_max_episode_len} '
          f'(longest demo={_max_demo_len}, mean={_mean_demo_len:.1f}, '
          f'buffer=+{_EVAL_EP_LEN_BUFFER}, safety cap={args.max_episode_len}). '
          f'Used only if per-episode scripted-traj construction fails; '
          f'normal eval episodes are sized per-cloth via build_traj+tail.')
else:
    eval_max_episode_len = int(args.max_episode_len)
    print(f'[data] no per-demo episode lengths recorded; eval '
          f'max_episode_len FALLBACK = {eval_max_episode_len} '
          f'(from --max_episode_len). Per-episode dynamic sizing is still '
          f'attempted at eval time via build_traj+episode_tail_frames.')


# =============================================================================
# Fit normalizers
# =============================================================================
obs_normalizer = ObsNormalizer(args.obs_mode)
obs_normalizer.fit(obs_all)
act_normalizer = ActionNormalizer()
act_normalizer.fit(acts_all)


# =============================================================================
# Dataset — sample (obs_horizon, pred_horizon) windows with padding.
# =============================================================================
def _make_sample_indices(episode_ends, pred_horizon,
                         pad_before, pad_after):
    """For each episode, produce one entry per legal sequence start
    (including pad_before/pad_after edge cases). Each entry tells the
    sampler how to slice obs_all + acts_all and where to place the
    valid range inside the fixed-size sequence_length buffer."""
    indices = []
    for i, end in enumerate(episode_ends):
        start = 0 if i == 0 else int(episode_ends[i - 1])
        length = int(end - start)
        min_start = -pad_before
        max_start = length - pred_horizon + pad_after
        for idx in range(min_start, max_start + 1):
            buf_start = max(idx, 0) + start
            buf_end = min(idx + pred_horizon, length) + start
            samp_start = buf_start - (idx + start)
            samp_end = (idx + pred_horizon + start) - buf_end
            indices.append([buf_start, buf_end,
                            samp_start, pred_horizon - samp_end])
    return np.asarray(indices, dtype=np.int64)


class DiffusionBCDataset(Dataset):
    """Returns one (obs_seq, action_seq) window per index.

    For obs_mode='state', obs_seq is an ndarray of shape
    (obs_horizon, state_dim).

    For obs_mode='rgb', obs_seq is a dict
        {'image': uint8 (obs_horizon, H, W, 3),
         'grip':  float32 (obs_horizon, 12),
         'goal':  float32 (obs_horizon,  3)}.

    For obs_mode='pcd', obs_seq is a dict
        {'pcd':  float32 (obs_horizon, n_pts, 3),
         'grip': float32 (obs_horizon, 12),
         'goal': float32 (obs_horizon,  3)}.

    Normalization is applied at sample time. Action_seq is always a
    (pred_horizon, action_dim) ndarray, already normalized to [-1, 1] at
    collection time (the ActionNormalizer is a no-op).
    """

    def __init__(self, obs_all, grip_all, goal_all, acts_all, episode_ends,
                 obs_horizon, pred_horizon, action_horizon, obs_mode,
                 obs_normalizer, act_normalizer):
        self.obs_all = obs_all
        self.grip_all = grip_all  # None for state-mode
        self.goal_all = goal_all  # None for state-mode
        self.acts_all = acts_all
        # pad_after = action_horizon - 1 matches the pusht reference
        # (diffusion_policy_state_pusht_demo.py:731). Each padded window
        # at the tail of an episode covers action positions the policy
        # would actually be asked to execute — vs pad_after=pred_horizon
        # -1 which generates many windows whose pred_horizon trailing
        # actions are mostly "hold last action" padding and waste model
        # capacity.
        self.indices = _make_sample_indices(
            episode_ends, pred_horizon,
            pad_before=obs_horizon - 1,
            pad_after=action_horizon - 1)
        self.obs_horizon = obs_horizon
        self.pred_horizon = pred_horizon
        self.obs_mode = obs_mode
        self.obs_normalizer = obs_normalizer
        self.act_normalizer = act_normalizer

    def __len__(self):
        return len(self.indices)

    @staticmethod
    def _pad(arr, total_len, ss, se):
        if arr.shape[0] == total_len:
            return arr
        out = np.zeros((total_len, *arr.shape[1:]), dtype=arr.dtype)
        if ss > 0:
            out[:ss] = arr[0]
        out[ss:se] = arr
        if se < total_len:
            out[se:] = arr[-1]
        return out

    def __getitem__(self, i):
        bs, be, ss, se = self.indices[i]
        obs_slice = self.obs_all[bs:be]
        act_slice = self.acts_all[bs:be]
        obs_seq = self._pad(obs_slice, self.pred_horizon, ss, se)
        act_seq = self._pad(act_slice, self.pred_horizon, ss, se)
        obs_seq = obs_seq[:self.obs_horizon]  # only first To frames used
        act_seq = self.act_normalizer.apply(act_seq)

        if self.obs_mode == 'state':
            return self.obs_normalizer.apply(obs_seq), act_seq

        # RGB / PCD: build a dict-typed sample with grip + goal alongside.
        grip_slice = self.grip_all[bs:be]
        grip_seq = self._pad(grip_slice, self.pred_horizon, ss, se)
        grip_seq = grip_seq[:self.obs_horizon]
        goal_slice = self.goal_all[bs:be]
        goal_seq = self._pad(goal_slice, self.pred_horizon, ss, se)
        goal_seq = goal_seq[:self.obs_horizon]
        primary_key = 'image' if self.obs_mode == 'rgb' else 'pcd'
        sample = {primary_key: obs_seq, 'grip': grip_seq, 'goal': goal_seq}
        return self.obs_normalizer.apply(sample), act_seq


def _collate(batch):
    """Stack ndarrays into batch tensors; dict obs stack per-key."""
    obs_list = [b[0] for b in batch]
    act_list = [b[1] for b in batch]
    act_t = torch.from_numpy(np.stack(act_list).astype(np.float32))

    if isinstance(obs_list[0], dict):
        out = {}
        for k in obs_list[0]:
            stacked = np.stack([s[k] for s in obs_list])
            if stacked.dtype == np.uint8:
                out[k] = torch.from_numpy(stacked).to(torch.uint8)
            else:
                out[k] = torch.from_numpy(stacked.astype(np.float32))
        return out, act_t

    arr = np.stack(obs_list)
    if arr.dtype == np.uint8:
        obs_t = torch.from_numpy(arr).to(torch.uint8)
    else:
        obs_t = torch.from_numpy(arr.astype(np.float32))
    return obs_t, act_t


def _to_device(obs, device):
    """Move tensor or dict-of-tensors to device."""
    if isinstance(obs, dict):
        return {k: v.to(device, non_blocking=True) for k, v in obs.items()}
    return obs.to(device, non_blocking=True)


dataset = DiffusionBCDataset(
    obs_all, grip_all, goal_all, acts_all, episode_ends,
    obs_horizon=args.obs_horizon, pred_horizon=args.pred_horizon,
    action_horizon=args.action_horizon, obs_mode=args.obs_mode,
    obs_normalizer=obs_normalizer, act_normalizer=act_normalizer)
print(f'[data] dataset windows = {len(dataset)}')

dataloader = DataLoader(
    dataset, batch_size=args.batch_size, shuffle=True,
    num_workers=args.num_workers,
    pin_memory=(device.type == 'cuda'),
    persistent_workers=(args.num_workers > 0),
    collate_fn=_collate)


# =============================================================================
# Build encoder + policy
# =============================================================================
if args.obs_mode == 'state':
    enc_kwargs = {'state_dim': int(obs_all.shape[-1])}
elif args.obs_mode == 'rgb':
    enc_kwargs = {'pretrained': args.pretrained_rgb}
elif args.obs_mode == 'pcd':
    # Use the actual n_pts from the saved data (handles both 256 and 512
    # demos cleanly; PointCloudObsEncoder treats it as a fixed input size).
    enc_kwargs = {'n_points': int(obs_all.shape[-2]), 'feat_dim': 256}

encoder = build_encoder(args.obs_mode, enc_kwargs).to(device)
policy = DiffusionPolicy(
    action_dim=int(acts_all.shape[-1]),
    obs_feat_dim=encoder.feat_dim,
    obs_horizon=args.obs_horizon,
    pred_horizon=args.pred_horizon,
    action_horizon=args.action_horizon,
    num_diffusion_iters=args.num_diffusion_iters,
).to(device)

n_params = sum(p.numel() for p in policy.parameters())
n_enc = sum(p.numel() for p in encoder.parameters())
print(f'[init] policy params  = {n_params:,}')
print(f'[init] encoder params = {n_enc:,}')


# =============================================================================
# Optimizer + scheduler + EMA
# =============================================================================
from diffusers.training_utils import EMAModel
from diffusers.optimization import get_scheduler

trainables = list(policy.parameters()) + list(encoder.parameters())
optimizer = torch.optim.AdamW(
    trainables, lr=args.lr, weight_decay=args.weight_decay)
lr_scheduler = get_scheduler(
    name='cosine', optimizer=optimizer,
    num_warmup_steps=args.num_warmup_steps,
    num_training_steps=len(dataloader) * args.num_epochs)
ema = EMAModel(parameters=trainables, power=args.ema_power)


# =============================================================================
# Resume from checkpoint (optional). Expects a ckpt saved by this script's
# current _save_ema_checkpoint — i.e. one that persists optimizer, EMA,
# and LR scheduler state alongside weights. Old weight-only ckpts will
# raise on the first missing key; recover by training from scratch.
# =============================================================================
resume_start_epoch = 0
resume_step_counter = 0
resume_best_eval_success = float('-inf')
resume_best_eval_epoch = -1
if args.resume:
    if not os.path.exists(args.resume):
        raise FileNotFoundError(f'--resume path does not exist: {args.resume}')
    print(f'\n=== Resuming from {args.resume} ===')
    rckpt = torch.load(args.resume, map_location=device)

    policy.load_state_dict(rckpt['policy_state_dict'])
    encoder.load_state_dict(rckpt['encoder_state_dict'])
    ema.load_state_dict(rckpt['ema_state_dict'])
    optimizer.load_state_dict(rckpt['optimizer_state_dict'])
    lr_scheduler.load_state_dict(rckpt['scheduler_state_dict'])
    resume_start_epoch = int(rckpt['epoch'])
    resume_step_counter = int(rckpt['step_counter'])
    resume_best_eval_success = float(rckpt['best_eval_success'])
    resume_best_eval_epoch = int(rckpt['best_eval_epoch'])

    print(f'  [resume] policy + encoder + EMA + optimizer + scheduler loaded')
    print(f'  [resume] epoch counter -> {resume_start_epoch} '
          f'(will train epochs {resume_start_epoch + 1}..{args.num_epochs})')
    print(f'  [resume] step counter -> {resume_step_counter}')
    print(f'  [resume] best_eval_success seeded at '
          f'{resume_best_eval_success:.3f} (epoch {resume_best_eval_epoch})')

    if resume_start_epoch >= args.num_epochs:
        raise RuntimeError(
            f'ckpt epoch ({resume_start_epoch}) >= --num_epochs '
            f'({args.num_epochs}); nothing to train. Either pass a '
            f'larger --num_epochs or use a ckpt from an earlier epoch.')


# =============================================================================
# In-env evaluation
#
# The eval env mirrors collect_bc_demos.py exactly: gym.make HangProcCloth-v1,
# RetryReset, manually capture obs each step via the same helpers used at
# collection time (so eval-time RGB/PCD are bit-identical to training
# inputs). Success is checked via _check_hanging_on_peg (the default
# metric); all three metrics are also logged for cross-comparison.
# =============================================================================
def _build_eval_env(eval_seed):
    # CRITICAL: sys.argv must stay patched through `gym.make`. dedo's
    # `preset_override_util` (dedo/utils/args.py:180) reads sys.argv at
    # env construction time to decide which args the user "owns"; any
    # arg name NOT present in the current sys.argv gets overwritten with
    # the HangProcCloth preset value (task_info.py:259). For cam_viewmat
    # specifically the preset is yaw=314, target=(-0.4, 0.6, 5.3) — a
    # completely different camera than our demos. If we restore sys.argv
    # before gym.make, eval RGB/PCD render from the wrong viewpoint and
    # every eval episode silently sees novel-camera inputs.
    sys_argv_backup = sys.argv
    sys.argv = [
        'eval',
        '--env=HangProcCloth-v1',
        f'--cam_resolution={eval_cam_resolution}',
        '--num_envs=0',
        '--total_env_steps=0',
        '--seed', str(eval_seed),
        '--max_episode_len', str(args.max_episode_len),
        f'--sim_freq={eval_sim_freq}',
        f'--sim_steps_per_action={eval_sim_steps_per_action}',
        '--cam_viewmat',
        *[f'{x:.6f}' for x in eval_cam_viewmat],
        f'--randomize_goal_radius={eval_randomize_goal_radius}',
    ]
    try:
        dargs, _ = get_args_parser()
        args_postprocess(dargs)
        dargs.debug = False
        dargs.viz = False
        dargs.uint8_pixels = True
        # sys.argv MUST still be patched here so preset_override_util
        # treats cam_viewmat (and the rest) as user-owned.
        e = gym.make(dargs.env, args=dargs)
    finally:
        sys.argv = sys_argv_backup
    # Patch deform.render() at construction so eval-video frames AND the
    # in-step make_final_steps settle frames both use the obs-camera
    # projection (fov=60), matching what capture_rgb_depth feeds the
    # policy. Unconditional: cheap, idempotent, and prevents any future
    # render call in this env from silently using dedo's fov≈90 default.
    patch_deform_render_to_obs_camera(resolve_deform(e))
    e = RetryResetEnv(e)
    e.seed(eval_seed)
    return e, dargs


def _capture_obs_for_policy(deform, obs_mode, state_key,
                            hole_idx, corner_idx):
    """Return one obs sample matching what the dataset saw at training
    time (UN-normalized — the obs_normalizer is applied next).

    state mode: returns a single ndarray (state_dim,).
    rgb / pcd:  returns a dict {primary, grip, goal} with the same keys
                the training dataset returns.
    """
    if obs_mode == 'state':
        return build_privileged_obs(
            deform, state_key, hole_idx, corner_indices=corner_idx)

    # Shared grip capture — matches PixelObsWrapper/PointCloudObsWrapper
    # and collect_bc_demos.py: 12-dim, /WBOX-normalized, clipped to [-2, 2].
    grip = np.asarray(deform.get_grip_obs(), dtype=np.float32)
    grip = np.clip(grip / 20.0, -2.0, 2.0)  # WORKSPACE_BOX_SIZE = 20
    # Hanger goal pose (/WBOX-normalized). Same field collect_bc_demos.py
    # saves into every demo pkl under obs['goal'] — given to RGB/PCD
    # alongside grip so the eval-time obs matches training exactly.
    goal = np.asarray(deform.goal_pos[0], dtype=np.float32) / 20.0

    if obs_mode == 'rgb':
        rgb, _, _, _, _ = capture_rgb_depth(
            deform, eval_cam_resolution, eval_cam_resolution)
        return {'image': rgb, 'grip': grip, 'goal': goal}
    if obs_mode == 'pcd':
        from _bc_obs_helpers import cloth_only_pcd as _cloth_only_pcd
        _, depth, seg, view, proj = capture_rgb_depth(
            deform, eval_cam_resolution, eval_cam_resolution)
        # Match collection-time filtering: keep only cloth-seg points so
        # eval-time PCD has the same distribution the encoder trained on.
        pcd = _cloth_only_pcd(depth, seg, view, proj,
                              deform.deform_id, collection_pcd_n_pts)
        return {'pcd': pcd, 'grip': grip, 'goal': goal}
    raise AssertionError


def _stack_obs_seq(samples):
    """Stack a list of `obs_horizon` per-step obs samples (each either an
    ndarray or a dict) into the (1, To, *) tensor the policy expects."""
    if isinstance(samples[0], dict):
        out = {}
        for k in samples[0]:
            arr = np.stack([s[k] for s in samples])
            if arr.dtype == np.uint8:
                out[k] = torch.from_numpy(arr).unsqueeze(0).to(device)
            else:
                out[k] = torch.from_numpy(
                    arr.astype(np.float32)).unsqueeze(0).to(device)
        return out
    arr = np.stack(samples)
    if arr.dtype == np.uint8:
        return torch.from_numpy(arr).unsqueeze(0).to(device)
    return torch.from_numpy(arr.astype(np.float32)).unsqueeze(0).to(device)


def _apply_normalizer_to_seq(samples, normalizer):
    """Apply obs_normalizer to each frame of a list of samples (dict or
    ndarray). Returns a new list of normalized samples."""
    return [normalizer.apply(s) for s in samples]


@torch.no_grad()
def evaluate_policy(n_episodes: int, eval_seed: int, label: str = 'eval',
                    record_video_episodes: int = 0,
                    video_label: str = ''):
    """Roll out the EMA policy on `n_episodes` deterministic episodes.

    Returns a dict of metrics (success rates per metric, episode reward,
    episode length).

    If `record_video_episodes > 0`, captures rendered RGB frames for the
    first N episodes (including the post-settle phase via dedo's
    `_record_settle_frames` hook), encodes them to a single mp4 in the
    logdir, and logs it to wandb under `<label>/video`. The remaining
    `n_episodes - record_video_episodes` rollouts run without capture so
    the success-rate aggregate isn't biased by the slower recorded
    episodes.
    """
    encoder.eval()
    policy.eval()
    # Apply EMA weights to a temp copy for eval; restore after.
    ema_state = [p.detach().clone() for p in trainables]
    ema.copy_to(trainables)

    e, dargs = _build_eval_env(eval_seed)
    deform = resolve_deform(e)
    # `deform.render` is already patched to use the obs-camera projection
    # (see _build_eval_env above) — policy-phase frames AND
    # make_final_steps settle frames all render through fov=60, matching
    # `capture_rgb_depth`. No per-call patching needed here.

    # Per-episode RGB frame buffers populated only for the first
    # `record_video_episodes` rollouts. Outer list is one entry per
    # recorded episode; inner list is all frames from that episode
    # (policy phase + post-settle).
    recorded_episode_frames = []  # list of list of (H,W,3) uint8

    s_hanging = s_topo = s_legacy = 0
    ep_rwds = []
    ep_lens = []
    for ep in range(n_episodes):
        is_recorded = ep < record_video_episodes
        ep_frames = []  # filled only if is_recorded
        e.reset()
        # Tell dedo to capture post-settle frames during make_final_steps
        # (the 500-step gravity phase that runs inside step() when done
        # fires). MUST be set AFTER e.reset() — collect_bc_demos.py uses
        # the same ordering. Setting before reset would let the env
        # construction path overwrite the flag with the __init__ default
        # (False), so make_final_steps would skip the settle capture and
        # info['settle_frames'] would be missing. With this ordering the
        # flag is fresh-set on the deform every episode and persists
        # through env.step()'s call to make_final_steps.
        if is_recorded:
            deform._record_settle_frames = True
            deform._settle_render_kwargs = dict(
                width=args.video_render_size,
                height=args.video_render_size)
            deform._settle_frame_stride = args.settle_frame_stride
        else:
            deform._record_settle_frames = False
        hole_idx = get_hole_indices(deform)
        if not hole_idx:
            # Skip degenerate clothes — count as failure to keep n_episodes honest.
            ep_rwds.append(0.0)
            ep_lens.append(0)
            continue
        hole_loops = get_hole_loops(deform)
        _, verts0 = get_mesh_data(deform.sim, deform.deform_id)
        corner_idx = identify_cloth_corners(verts0)
        hole_radius = measure_hole_radius(deform, hole_idx)

        # Per-episode max_ep_len: mirror collect_bc_demos.py exactly —
        # build the scripted trajectory for THIS cloth (the demo
        # controller would have run it for len(traj) steps), and cap the
        # episode at min(len(traj) + episode_tail_frames, args.max_episode_len).
        # Falls back to the demo-distribution-derived `eval_max_episode_len`
        # if scripted-traj construction fails (e.g. degenerate hole geometry).
        _per_ep_max = compute_per_episode_max_len(
            deform, ctrl_freq=eval_ctrl_freq,
            tail_frames=args.episode_tail_frames,
            safety_cap=args.max_episode_len)
        if _per_ep_max is None:
            _per_ep_max = eval_max_episode_len
        # Demos collected at demo_speed<1 commanded smaller velocities AND
        # were stretched to 1/demo_speed times the step count. The policy
        # learned that distribution, so eval rollouts also need
        # 1/demo_speed times the steps to complete the same motion.
        # compute_per_episode_max_len rebuilds the scripted traj at FULL
        # speed (it has no knowledge of demo_speed), so we scale here.
        if eval_demo_speed != 1.0 and eval_demo_speed > 0:
            _per_ep_max = int(math.ceil(_per_ep_max / eval_demo_speed))
        deform.max_episode_len = _per_ep_max

        # Prime obs deque with the initial obs (repeated obs_horizon times).
        first_obs = _capture_obs_for_policy(
            deform, args.obs_mode, args.state_key, hole_idx, corner_idx)
        obs_deque = collections.deque(
            [first_obs] * args.obs_horizon, maxlen=args.obs_horizon)

        ep_rwd = 0.0
        step = 0
        done = False
        while not done and step < _per_ep_max:
            # 1) Apply obs normalizer per frame, stack to (1, To, *), run policy.
            normed = _apply_normalizer_to_seq(
                list(obs_deque), obs_normalizer)
            obs_t = _stack_obs_seq(normed)
            naction = policy.predict_action(obs_t, encoder).squeeze(0)
            naction = naction.cpu().numpy()
            # 2) Slice the action_horizon middle chunk.
            start = args.obs_horizon - 1
            end = start + args.action_horizon
            chunk = naction[start:end]
            chunk = act_normalizer.unapply(chunk)
            # 3) Execute chunk open-loop, capture obs each step.
            for a in chunk:
                a = np.clip(a, -1.0, 1.0).astype(np.float32)
                _, rwd, done, info = e.step(a)
                ep_rwd += float(rwd)
                step += 1
                if is_recorded:
                    # Pre-settle policy-phase frame. On the terminal step
                    # (done=True), dedo's make_final_steps already ran
                    # INSIDE this step() call, and settle_frames is in
                    # `info` — we append those below after the loop. We
                    # still capture this one frame to mark the policy
                    # handoff.
                    ep_frames.append(deform.render(
                        mode='rgb_array',
                        width=args.video_render_size,
                        height=args.video_render_size))
                new_obs = _capture_obs_for_policy(
                    deform, args.obs_mode, args.state_key,
                    hole_idx, corner_idx)
                obs_deque.append(new_obs)
                if done or step >= _per_ep_max:
                    break
        # Append the post-settle (gravity-phase) frames captured by
        # dedo inside make_final_steps. `info['settle_frames']` exists
        # iff `_record_settle_frames` was True at the terminal step.
        if is_recorded:
            settle_frames = info.get('settle_frames', []) if isinstance(
                info, dict) else []
            ep_frames.extend(settle_frames)
            recorded_episode_frames.append(ep_frames)
            deform._record_settle_frames = False  # turn off for next ep

        # Score this episode with all three metrics for cross-comparison.
        s_hanging += int(check_hanging_on_peg(
            deform, hole_idx, hole_radius, args.success_factor))
        st, _ = check_threaded_topological(deform, hole_loops)
        s_topo += int(st)
        s_legacy += int(check_legacy(
            deform, hole_idx, hole_radius, args.success_factor))
        ep_rwds.append(ep_rwd)
        ep_lens.append(step)
        print(f'  [{label} ep {ep+1}/{n_episodes}] '
              f'rwd={ep_rwd:6.1f}  len={step:3d}  '
              f'h={s_hanging}/{ep+1} t={s_topo}/{ep+1} l={s_legacy}/{ep+1}')

    # Encode the captured episodes into one mp4 (concat). One file per
    # eval pass is plenty — separate episodes can be told apart by the
    # gripper-anchor reset between them.
    video_path = None
    if recorded_episode_frames:
        flat_frames = [f for ep in recorded_episode_frames for f in ep]
        video_filename = (f'{label}{("_" + video_label) if video_label else ""}'
                          f'.mp4')
        video_path = os.path.join(logdir, video_filename)
        try:
            _write_mp4(flat_frames, video_path, fps=args.video_fps)
            print(f'  [video] wrote {video_path} '
                  f'({len(flat_frames)} frames, '
                  f'{len(recorded_episode_frames)} episodes)')
        except Exception as _video_err:
            print(f'  [video] WARN: mp4 encode failed: {_video_err!r}')
            video_path = None

    e.close()
    # Restore non-EMA weights.
    for p, saved in zip(trainables, ema_state):
        p.data.copy_(saved)
    encoder.train()
    policy.train()
    # Primary success_rate alias: whichever metric the user picked in
    # --success_metric. This is the headline number that should line up
    # with train_privileged.py's final_eval/success_rate so cross-script
    # comparisons stay clean in the wandb workspace.
    primary_rate = {
        'hanging': s_hanging,
        'topological': s_topo,
        'legacy': s_legacy,
    }[args.success_metric] / n_episodes
    metrics = {
        f'{label}/success_rate': primary_rate,
        f'{label}/success_hanging': s_hanging / n_episodes,
        f'{label}/success_topological': s_topo / n_episodes,
        f'{label}/success_legacy': s_legacy / n_episodes,
        f'{label}/mean_reward': float(np.mean(ep_rwds)),
        f'{label}/std_reward': float(np.std(ep_rwds)),
        f'{label}/mean_episode_len': float(np.mean(ep_lens)),
        f'{label}/n_episodes': n_episodes,
    }
    # Stash the on-disk video path for the caller (so it can wandb.log
    # the wandb.Video with the right epoch-tagged caption alongside the
    # numeric metrics, in a single wandb.log call).
    if video_path is not None:
        metrics[f'{label}/_video_path'] = video_path
    return metrics


# =============================================================================
# Persist config
# =============================================================================
with open(os.path.join(logdir, 'config.json'), 'w') as f:
    json.dump({
        'args': vars(args),
        'n_demos_kept': len(obs_buf),
        'n_demos_total': n_total,
        'collection_cam_resolution': collection_cam_res,
        'collection_pcd_n_points': collection_pcd_n_pts,
        'recorded_success_factors': sorted(recorded_success_factors),
        'recorded_ctrl_freqs': sorted(recorded_ctrl_freqs),
        'eval_ctrl_freq': float(eval_ctrl_freq),
        'eval_sim_freq': int(eval_sim_freq),
        'eval_sim_steps_per_action': int(eval_sim_steps_per_action),
        'eval_max_act_vel': float(DeformEnv.MAX_ACT_VEL),
        'eval_max_episode_len_fallback': int(eval_max_episode_len),
        'eval_episode_tail_frames': int(args.episode_tail_frames),
        'demo_episode_lengths_min': (int(min(recorded_episode_lengths))
                                     if recorded_episode_lengths else None),
        'demo_episode_lengths_max': (int(max(recorded_episode_lengths))
                                     if recorded_episode_lengths else None),
        'state_dim': int(obs_all.shape[-1]) if args.obs_mode == 'state' else None,
        'action_dim': int(acts_all.shape[-1]),
        'n_dataset_windows': len(dataset),
    }, f, indent=2)


# =============================================================================
# Checkpoint helper. EMA-swap is centralized here so periodic / best / final
# saves all use the same code path. The helper saves the EMA-weighted policy
# and encoder; live (training) weights are restored on exit so the next
# training step continues from the unaveraged params (EMA is just a smoothed
# shadow, not the train trajectory).
# =============================================================================
_CKPT_METADATA_TEMPLATE = {
    'obs_mode': args.obs_mode,
    'obs_horizon': args.obs_horizon,
    'pred_horizon': args.pred_horizon,
    'action_horizon': args.action_horizon,
    'num_diffusion_iters': args.num_diffusion_iters,
    'action_dim': int(acts_all.shape[-1]),
    'state_dim': int(obs_all.shape[-1]) if args.obs_mode == 'state' else None,
    'pcd_n_points': collection_pcd_n_pts,
    'state_key': args.state_key,
    # The eval env reads these from the demo pkl, but stashing them in the
    # ckpt too means a deploy script can rebuild the env without the demos.
    'eval_cam_viewmat': list(eval_cam_viewmat),
    'eval_cam_resolution': int(eval_cam_resolution),
    'max_act_vel': float(DeformEnv.MAX_ACT_VEL),
    # Eval-time ctrl_freq the env was built under. Demos were recorded
    # at the same Hz, so this lets a deploy script rebuild a parity env
    # without needing the demo pkls.
    'ctrl_freq': float(eval_ctrl_freq),
    'sim_freq': int(eval_sim_freq),
    'sim_steps_per_action': int(eval_sim_steps_per_action),
    # Per-episode eval cap = len(scripted_traj) + episode_tail_frames.
    # Saved so a standalone eval script can rebuild the per-episode
    # termination policy without the demo dir.
    'episode_tail_frames': int(args.episode_tail_frames),
    'max_episode_len_safety_cap': int(args.max_episode_len),
    # Hanger-goal randomization radius the demos were collected under.
    # Saved into the ckpt so eval_diffusion_bc.py can rebuild a parity
    # eval env without the demo dir. 0.0 = fixed goal (legacy v3 behavior).
    'randomize_goal_radius': float(eval_randomize_goal_radius),
}


def _save_ema_checkpoint(path, epoch=None, eval_metrics=None):
    """Snapshot EMA weights + full training state to disk.

    `policy_state_dict` / `encoder_state_dict` contain the EMA-applied
    weights so a deploy script can `torch.load(path)` and use them directly.
    The remaining keys (ema_state_dict, optimizer_state_dict,
    scheduler_state_dict, epoch, step_counter, best_eval_*) persist enough
    state for `--resume` to pick the run back up without drift.
    Live (training) weights are restored on exit so the next training step
    continues from the unaveraged params.
    """
    ema_state_local = [p.detach().clone() for p in trainables]
    ema.copy_to(trainables)
    ckpt = {
        'policy_state_dict': policy.state_dict(),
        'encoder_state_dict': encoder.state_dict(),
        'ema_state_dict': ema.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': lr_scheduler.state_dict(),
        'step_counter': int(step_counter),
        'best_eval_success': float(best_eval_success),
        'best_eval_epoch': int(best_eval_epoch),
        **_CKPT_METADATA_TEMPLATE,
    }
    if epoch is not None:
        ckpt['epoch'] = int(epoch)
    if eval_metrics is not None:
        ckpt['eval_metrics'] = dict(eval_metrics)
    torch.save(ckpt, path)
    for p, saved in zip(trainables, ema_state_local):
        p.data.copy_(saved)


# =============================================================================
# Video helper. Encodes a list of (H, W, 3) uint8 RGB frames to an
# mp4 that wandb / browsers can play back. Mirrors HangVideoCallback's
# convention exactly:
#   - libx264 codec via imageio_ffmpeg's bundled static ffmpeg
#   - yuv420p pixel format (required for browser <video> tags)
#   - +faststart so playback starts before the whole file downloads
# cv2's mp4v fourcc produces an MPEG-4 Simple Profile stream that the
# wandb HTML5 player can't decode — videos would sit on "loading" forever.
# =============================================================================
def _write_mp4(frames, path: str, fps: int = 30) -> None:
    if not frames:
        return
    import imageio  # imageio_ffmpeg is a transitive dep of imageio
    writer = imageio.get_writer(
        path, fps=fps, codec='libx264', quality=8,
        macro_block_size=2, pixelformat='yuv420p',
        ffmpeg_params=['-movflags', '+faststart'])
    for frame in frames:
        writer.append_data(np.ascontiguousarray(frame))
    writer.close()


# Save the obs normalizer once now. It's fitted on the full demo pool at
# startup and never updated during training, so a single file alongside the
# logdir is enough — any periodic / best / final ckpt can be loaded with
# this same normalizer pkl.
with open(os.path.join(logdir, 'obs_normalizer.pkl'), 'wb') as f:
    pickle.dump(obs_normalizer.state_dict(), f)
print(f'[init] wrote obs_normalizer.pkl alongside the logdir')


# =============================================================================
# Train loop
# =============================================================================
print(f'\n=== Training (epochs={args.num_epochs}, '
      f'batches/epoch={len(dataloader)}) ===')

# Best-eval tracking. The "primary" metric is whichever one
# --success_metric selects (alias eval/success_rate). On every eval pass,
# if the current rate beats the running best, mirror the EMA weights to
# policy_best.pt — so a crash or premature kill still leaves you with
# the best-performing snapshot to deploy from.
best_eval_success = resume_best_eval_success
best_eval_epoch = resume_best_eval_epoch
best_ckpt_path = os.path.join(logdir, 'policy_best.pt')

step_counter = resume_step_counter
if args.eval_only:
    print(f'\n=== --eval_only: skipping training loop, jumping straight '
          f'to final eval on {args.resume} ===')
for epoch in (range(0) if args.eval_only
              else range(resume_start_epoch, args.num_epochs)):
    epoch_loss = []
    epoch_start = time.time()
    for obs_b, act_b in dataloader:
        obs_b = _to_device(obs_b, device)
        act_b = act_b.to(device, non_blocking=True)
        loss = policy.compute_loss(obs_b, act_b, encoder)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        lr_scheduler.step()
        ema.step(trainables)
        epoch_loss.append(loss.item())
        step_counter += 1
        if args.use_wandb and (step_counter % 50 == 0):
            wandb.log({'train/loss': float(loss.item()),
                       'train/lr': float(lr_scheduler.get_last_lr()[0]),
                       'train/step': step_counter,
                       'train/epoch': epoch})
    avg_loss = float(np.mean(epoch_loss))
    print(f'[epoch {epoch+1:>3}/{args.num_epochs}] '
          f'loss={avg_loss:.5f}  '
          f'time={time.time()-epoch_start:.1f}s')
    if args.use_wandb:
        wandb.log({'train/epoch_loss': avg_loss,
                   'train/epoch': epoch + 1})

    # Mid-training eval.
    if (args.eval_every_epochs > 0
            and (epoch + 1) % args.eval_every_epochs == 0):
        # Index of this eval pass (1-based). Used both for the
        # video_every_evals modulo check and as a label inside the mp4
        # filename so successive videos don't overwrite each other.
        eval_pass_idx = (epoch + 1) // args.eval_every_epochs
        record_n = (args.n_video_episodes
                    if (args.video_every_evals > 0
                        and eval_pass_idx % args.video_every_evals == 0)
                    else 0)
        print(f'  [eval @ epoch {epoch+1}] running '
              f'{args.n_eval_episodes} episodes'
              f'{f" (recording first {record_n})" if record_n else ""}...')
        eval_metrics = evaluate_policy(
            n_episodes=args.n_eval_episodes,
            eval_seed=args.seed + args.eval_seed_offset,
            label='eval',
            record_video_episodes=record_n,
            video_label=f'ep{epoch+1:04d}')
        # Pop the video path BEFORE printing / wandb-logging so the
        # numeric metrics dict stays clean (wandb.log uses the path to
        # build a wandb.Video object instead of stringifying it).
        video_path = eval_metrics.pop('eval/_video_path', None)
        for k, v in eval_metrics.items():
            print(f'    {k}: {v}')
        if args.use_wandb:
            log_dict = {**eval_metrics, 'train/epoch': epoch + 1}
            if video_path:
                log_dict['eval/video'] = wandb.Video(
                    video_path, fps=args.video_fps,
                    caption=f'epoch {epoch+1} | '
                            f'success_rate={eval_metrics["eval/success_rate"]:.2f}')
            wandb.log(log_dict)

        # --- Periodic numbered checkpoint (opt-in via --save_every_epochs).
        # Saves AFTER the eval so the eval_metrics get embedded in the ckpt
        # dict and a deploy script can see what eval rate this snapshot
        # achieved without consulting the wandb run.
        if (args.save_every_epochs > 0
                and (epoch + 1) % args.save_every_epochs == 0):
            periodic_path = os.path.join(
                logdir, f'policy_ep{epoch+1:04d}.pt')
            _save_ema_checkpoint(
                periodic_path, epoch=epoch + 1, eval_metrics=eval_metrics)
            print(f'  [ckpt] saved periodic {os.path.basename(periodic_path)}')

        # --- Best-eval tracking (always on). Compare the primary success
        # rate (eval/success_rate, which aliases the user's --success_metric)
        # against the running max; mirror to policy_best.pt on improvement.
        cur_eval_succ = float(eval_metrics.get(
            'eval/success_rate', float('-inf')))
        if cur_eval_succ > best_eval_success:
            best_eval_success = cur_eval_succ
            best_eval_epoch = epoch + 1
            _save_ema_checkpoint(
                best_ckpt_path, epoch=epoch + 1, eval_metrics=eval_metrics)
            print(f'  [ckpt] NEW BEST: eval/success_rate='
                  f'{cur_eval_succ:.3f} @ epoch {epoch+1} '
                  f'-> policy_best.pt')
            if args.use_wandb:
                wandb.log({'best/eval_success_rate': best_eval_success,
                           'best/epoch': best_eval_epoch,
                           'train/epoch': epoch + 1})


# =============================================================================
# Final eval — EMA weights, more episodes for tight SE. Runs FIRST so the
# final policy.pt save below can embed the final_metrics in the ckpt dict
# (deploy scripts will know what eval rate this checkpoint achieved without
# loading the wandb run).
# =============================================================================
if args.eval_all_in:
    # ---------------------------------------------------------------------
    # Recreate the training-time eval curve from saved checkpoints. Loads
    # each policy_ep<NNNN>.pt in epoch order, swaps weights into the
    # already-built policy/encoder, runs eval, logs to wandb at step=epoch.
    # ---------------------------------------------------------------------
    import re
    import glob
    pattern = os.path.join(args.eval_all_in, 'policy_ep*.pt')
    found = sorted(glob.glob(pattern))
    if not found:
        raise FileNotFoundError(
            f'no policy_ep*.pt under {args.eval_all_in}; nothing to eval.')
    print(f'\n=== --eval_all_in: re-eval {len(found)} checkpoint(s) '
          f'from {args.eval_all_in} ===')
    for ckpt_path in found:
        m = re.search(r'policy_ep(\d+)\.pt$', ckpt_path)
        ep = int(m.group(1)) if m else -1
        print(f'\n--- ep{ep:04d}: {os.path.basename(ckpt_path)} ---')
        ckpt = torch.load(ckpt_path, map_location=device)
        # policy_state_dict / encoder_state_dict already hold EMA-applied
        # weights (see _save_ema_checkpoint). evaluate_policy will then
        # also apply ema.copy_to internally — that's a no-op since the
        # loaded ema_state_dict matches the loaded EMA-applied weights.
        policy.load_state_dict(ckpt['policy_state_dict'])
        encoder.load_state_dict(ckpt['encoder_state_dict'])
        if 'ema_state_dict' in ckpt:
            ema.load_state_dict(ckpt['ema_state_dict'])
        # One video per checkpoint is enough to inspect motion qualitatively;
        # the success-rate number is the main payload across the curve.
        reeval_record_n = min(args.n_video_episodes,
                              args.n_eval_episodes) if (
            args.video_every_evals > 0) else 0
        reeval_metrics = evaluate_policy(
            n_episodes=args.n_eval_episodes,
            eval_seed=args.seed + args.eval_seed_offset,
            label='reeval',
            record_video_episodes=reeval_record_n,
            video_label=f'ep{ep:04d}')
        reeval_video_path = reeval_metrics.pop('reeval/_video_path', None)
        for k, v in sorted(reeval_metrics.items()):
            print(f'  {k}: {v}')
        if args.use_wandb:
            log_dict = dict(reeval_metrics)
            log_dict['reeval/epoch'] = ep
            if reeval_video_path:
                log_dict['reeval/video'] = wandb.Video(
                    reeval_video_path, fps=args.video_fps,
                    caption=f'ep{ep:04d} | '
                            f'success_rate='
                            f'{reeval_metrics["reeval/success_rate"]:.2f}')
            wandb.log(log_dict, step=ep)
    print('\n=== --eval_all_in: done ===')
else:
    print(f'\n=== Final eval ({args.n_final_eval_episodes} episodes) ===')
    # Always record at least one video at the end of training so deploy
    # decisions don't depend on the wandb mid-training videos.
    final_record_n = max(args.n_video_episodes, 1) \
        if args.video_every_evals > 0 else 0
    final_metrics = evaluate_policy(
        n_episodes=args.n_final_eval_episodes,
        eval_seed=args.seed + args.eval_seed_offset + 1,
        label='final_eval',
        record_video_episodes=final_record_n,
        video_label='final')
    final_video_path = final_metrics.pop('final_eval/_video_path', None)
    print('\nFinal eval:')
    for k, v in sorted(final_metrics.items()):
        print(f'  {k}: {v}')
    if args.use_wandb:
        log_dict = dict(final_metrics)
        if final_video_path:
            log_dict['final_eval/video'] = wandb.Video(
                final_video_path, fps=args.video_fps,
                caption=f'final | '
                        f'success_rate={final_metrics["final_eval/success_rate"]:.2f}')
        wandb.log(log_dict)


# =============================================================================
# Save final checkpoint (EMA weights). Note: policy_best.pt may be a
# different epoch's snapshot if a mid-training eval scored higher than the
# final eval — that's the whole point of best-tracking. Both files are
# valid; deploy from whichever the workflow prefers.
# =============================================================================
if not args.eval_all_in:
    final_ckpt_path = os.path.join(logdir, 'policy.pt')
    _save_ema_checkpoint(
        final_ckpt_path, epoch=args.num_epochs, eval_metrics=final_metrics)
    print(f'\n[ckpt] saved final {final_ckpt_path}')
    if best_eval_epoch > 0:
        print(f'[ckpt] best mid-training eval/success_rate='
              f'{best_eval_success:.3f} at epoch {best_eval_epoch} '
              f'-> policy_best.pt')
    else:
        print(f'[ckpt] no mid-training eval ran (--eval_every_epochs=0); '
              f'policy_best.pt was NOT written.')
if args.use_wandb:
    wandb.finish()
