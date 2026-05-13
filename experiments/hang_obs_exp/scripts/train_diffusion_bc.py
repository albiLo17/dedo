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
from _helpers import RetryResetEnv  # noqa: E402

# Reuse the camera and success-check helpers so eval-time RGB/PCD/success
# match collection-time bit-for-bit.
from _bc_obs_helpers import (  # noqa: E402
    capture_rgb_depth, depth_to_pcd,
    check_hanging_on_peg, check_threaded_topological, check_legacy,
    measure_hole_radius, get_hole_indices, get_hole_loops,
    resolve_deform)

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
parser.add_argument('--max_episode_len', type=int, default=200)
parser.add_argument('--eval_cam_resolution', type=int, default=None,
                    help='Camera resolution at eval time. Default = the '
                         'resolution recorded in the demo pkls. Override '
                         'only if you specifically want to test cross-'
                         'resolution generalization.')
parser.add_argument('--success_factor', type=float, default=1.2,
                    help='Adaptive success threshold for eval-time '
                         'criterion. Should match the value collect_bc_'
                         'demos.py used; the script warns on mismatch.')

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
obs_buf: list = []
grip_buf: list = []
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
needs_grip = args.obs_mode in ('rgb', 'pcd')  # state already has grip
n_missing_grip = 0

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
    # Auxiliary gripper proprioception (only for RGB/PCD; state mode has
    # it baked into its 18-dim vector already).
    if needs_grip:
        grip_arr = d['obs'].get('grip')
        if grip_arr is None:
            # Legacy pkl without grip — fall back to zeros and warn.
            grip_arr = np.zeros((len(acts), 12), dtype=np.float32)
            n_missing_grip += 1
        grip_buf.append(np.asarray(grip_arr, dtype=np.float32))
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
         'grip':  float32 (obs_horizon, 12)}.

    For obs_mode='pcd', obs_seq is a dict
        {'pcd':  float32 (obs_horizon, n_pts, 3),
         'grip': float32 (obs_horizon, 12)}.

    Normalization is applied at sample time. Action_seq is always a
    (pred_horizon, action_dim) ndarray, already normalized to [-1, 1] at
    collection time (the ActionNormalizer is a no-op).
    """

    def __init__(self, obs_all, grip_all, acts_all, episode_ends,
                 obs_horizon, pred_horizon, action_horizon, obs_mode,
                 obs_normalizer, act_normalizer):
        self.obs_all = obs_all
        self.grip_all = grip_all  # None for state-mode
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

        # RGB / PCD: build a dict-typed sample with grip alongside.
        grip_slice = self.grip_all[bs:be]
        grip_seq = self._pad(grip_slice, self.pred_horizon, ss, se)
        grip_seq = grip_seq[:self.obs_horizon]
        primary_key = 'image' if self.obs_mode == 'rgb' else 'pcd'
        sample = {primary_key: obs_seq, 'grip': grip_seq}
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
    obs_all, grip_all, acts_all, episode_ends,
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
    enc_kwargs = {}
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
        '--cam_viewmat',
        *[f'{x:.6f}' for x in eval_cam_viewmat],
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
    e = RetryResetEnv(e)
    e.seed(eval_seed)
    return e, dargs


def _capture_obs_for_policy(deform, obs_mode, state_key,
                            hole_idx, corner_idx):
    """Return one obs sample matching what the dataset saw at training
    time (UN-normalized — the obs_normalizer is applied next).

    state mode: returns a single ndarray (state_dim,).
    rgb / pcd:  returns a dict {primary, grip} with the same keys the
                training dataset returns.
    """
    if obs_mode == 'state':
        return build_privileged_obs(
            deform, state_key, hole_idx, corner_indices=corner_idx)

    # Shared grip capture — matches PixelObsWrapper/PointCloudObsWrapper
    # and collect_bc_demos.py: 12-dim, /WBOX-normalized, clipped to [-2, 2].
    grip = np.asarray(deform.get_grip_obs(), dtype=np.float32)
    grip = np.clip(grip / 20.0, -2.0, 2.0)  # WORKSPACE_BOX_SIZE = 20

    if obs_mode == 'rgb':
        rgb, _, _, _ = capture_rgb_depth(
            deform, eval_cam_resolution, eval_cam_resolution)
        return {'image': rgb, 'grip': grip}
    if obs_mode == 'pcd':
        _, depth, view, proj = capture_rgb_depth(
            deform, eval_cam_resolution, eval_cam_resolution)
        pcd = depth_to_pcd(depth, view, proj, collection_pcd_n_pts)
        return {'pcd': pcd, 'grip': grip}
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
def evaluate_policy(n_episodes: int, eval_seed: int, label: str = 'eval'):
    """Roll out the EMA policy on `n_episodes` deterministic episodes.
    Returns a dict of metrics (success rates per metric, episode reward,
    episode length)."""
    encoder.eval()
    policy.eval()
    # Apply EMA weights to a temp copy for eval; restore after.
    ema_state = [p.detach().clone() for p in trainables]
    ema.copy_to(trainables)

    e, dargs = _build_eval_env(eval_seed)
    deform = resolve_deform(e)

    s_hanging = s_topo = s_legacy = 0
    ep_rwds = []
    ep_lens = []
    for ep in range(n_episodes):
        e.reset()
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

        # Prime obs deque with the initial obs (repeated obs_horizon times).
        first_obs = _capture_obs_for_policy(
            deform, args.obs_mode, args.state_key, hole_idx, corner_idx)
        obs_deque = collections.deque(
            [first_obs] * args.obs_horizon, maxlen=args.obs_horizon)

        ep_rwd = 0.0
        step = 0
        done = False
        while not done and step < args.max_episode_len:
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
                _, rwd, done, _ = e.step(a)
                ep_rwd += float(rwd)
                step += 1
                new_obs = _capture_obs_for_policy(
                    deform, args.obs_mode, args.state_key,
                    hole_idx, corner_idx)
                obs_deque.append(new_obs)
                if done or step >= args.max_episode_len:
                    break

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
        'state_dim': int(obs_all.shape[-1]) if args.obs_mode == 'state' else None,
        'action_dim': int(acts_all.shape[-1]),
        'n_dataset_windows': len(dataset),
    }, f, indent=2)


# =============================================================================
# Train loop
# =============================================================================
print(f'\n=== Training (epochs={args.num_epochs}, '
      f'batches/epoch={len(dataloader)}) ===')

step_counter = 0
for epoch in range(args.num_epochs):
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
        print(f'  [eval @ epoch {epoch+1}] running '
              f'{args.n_eval_episodes} episodes...')
        eval_metrics = evaluate_policy(
            n_episodes=args.n_eval_episodes,
            eval_seed=args.seed + args.eval_seed_offset,
            label='eval')
        for k, v in eval_metrics.items():
            print(f'    {k}: {v}')
        if args.use_wandb:
            wandb.log({**eval_metrics, 'train/epoch': epoch + 1})


# =============================================================================
# Save final checkpoint (EMA weights)
# =============================================================================
ckpt_path = os.path.join(logdir, 'policy.pt')
# Capture EMA weights into a regular state dict for clean loading.
ema_state = [p.detach().clone() for p in trainables]
ema.copy_to(trainables)
torch.save({
    'policy_state_dict': policy.state_dict(),
    'encoder_state_dict': encoder.state_dict(),
    'obs_mode': args.obs_mode,
    'obs_horizon': args.obs_horizon,
    'pred_horizon': args.pred_horizon,
    'action_horizon': args.action_horizon,
    'num_diffusion_iters': args.num_diffusion_iters,
    'action_dim': int(acts_all.shape[-1]),
    'state_dim': int(obs_all.shape[-1]) if args.obs_mode == 'state' else None,
    'pcd_n_points': collection_pcd_n_pts,
    'state_key': args.state_key,
}, ckpt_path)
with open(os.path.join(logdir, 'obs_normalizer.pkl'), 'wb') as f:
    pickle.dump(obs_normalizer.state_dict(), f)
print(f'\n[ckpt] saved {ckpt_path}')

# Restore live weights so the final eval (below) uses EMA.
# (We already copied EMA in; just keep the restore-to-live for parity
# with mid-training eval flow if anything later runs.)

# =============================================================================
# Final eval — EMA weights, more episodes for tight SE.
# =============================================================================
print(f'\n=== Final eval ({args.n_final_eval_episodes} episodes) ===')
final_metrics = evaluate_policy(
    n_episodes=args.n_final_eval_episodes,
    eval_seed=args.seed + args.eval_seed_offset + 1,
    label='final_eval')
print('\nFinal eval:')
for k, v in sorted(final_metrics.items()):
    print(f'  {k}: {v}')
if args.use_wandb:
    wandb.log(final_metrics)
    wandb.finish()

# Restore live (non-EMA) weights post-eval just in case anything else runs.
for p, saved in zip(trainables, ema_state):
    p.data.copy_(saved)
