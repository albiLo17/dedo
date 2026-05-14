"""
Standalone eval-only validator for a trained diffusion-BC checkpoint.

Use case: sanity-check a checkpoint against an arbitrary `--max_episode_len`
WITHOUT needing the original demo dir. Built specifically to validate the
eval-cap fix (eval episodes terminating around the training-distribution
length, e.g. 56 ctrl steps at 15 Hz, instead of dedo's default 200) on an
already-trained policy.

The checkpoint format (written by train_diffusion_bc.py's _save_ema_checkpoint)
embeds all the metadata needed to rebuild the encoder + policy + eval env:
obs_mode, horizons, num_diffusion_iters, cam_viewmat, MAX_ACT_VEL,
sim_freq, etc. The obs_normalizer.pkl lives next to the ckpt in the run
dir; we auto-discover it.

Example:
  python experiments/hang_obs_exp/scripts/eval_diffusion_bc.py \\
      --ckpt logs/hang_obs_exp/diffusion_bc/state/diff260514-073100_state_lr1e-4_e300_bs256_ah4_sm-legacy_s2026/policy_best.pt \\
      --n_episodes 30 --max_episode_len 56
"""
from __future__ import annotations

import argparse
import collections
import os
import pickle
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT))

import gym
import numpy as np
import torch

import dedo  # noqa: F401
from dedo.envs.deform_env import DeformEnv
from dedo.utils.args import get_args_parser, args_postprocess
from dedo.utils.mesh_utils import get_mesh_data

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _helpers import RetryResetEnv  # noqa: E402
from _bc_obs_helpers import (  # noqa: E402
    capture_rgb_depth,
    check_hanging_on_peg, check_threaded_topological, check_legacy,
    measure_hole_radius, get_hole_indices, get_hole_loops,
    resolve_deform)
from experiments.hang_obs_exp.envs.privileged_env import (  # noqa: E402
    build_privileged_obs, identify_cloth_corners)
from _diffusion_policy import (  # noqa: E402
    DiffusionPolicy, build_encoder, ObsNormalizer, ActionNormalizer)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--ckpt', type=str, required=True,
                   help='Path to policy_best.pt / policy_epNNNN.pt. '
                        'obs_normalizer.pkl is auto-discovered in the '
                        'same directory.')
    p.add_argument('--max_episode_len', type=int, default=56,
                   help='Eval-time per-episode cap. Default 56 matches '
                        'the v3 brake-tail demo distribution '
                        '(~51 ctrl steps + 5 frame buffer). Set higher '
                        'to test the OOD-tail failure mode.')
    p.add_argument('--n_episodes', type=int, default=30)
    p.add_argument('--eval_seed', type=int, default=2026 + 9999,
                   help='Matches train_diffusion_bc.py: '
                        'seed + eval_seed_offset(9999).')
    p.add_argument('--success_factor', type=float, default=1.2)
    return p.parse_args()


def main():
    args = parse_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'[init] device = {device}')

    # --- Load ckpt + metadata ----------------------------------------------
    ckpt_path = Path(args.ckpt)
    if not ckpt_path.exists():
        raise FileNotFoundError(f'ckpt not found: {ckpt_path}')
    ckpt = torch.load(str(ckpt_path), map_location=device)
    print(f'[init] loaded ckpt from {ckpt_path}')
    if 'epoch' in ckpt:
        print(f'[init] ckpt epoch = {ckpt["epoch"]}')
    if 'eval_metrics' in ckpt:
        print(f'[init] ckpt embedded eval_metrics = {ckpt["eval_metrics"]}')

    obs_mode = ckpt['obs_mode']
    obs_horizon = int(ckpt['obs_horizon'])
    pred_horizon = int(ckpt['pred_horizon'])
    action_horizon = int(ckpt['action_horizon'])
    num_diffusion_iters = int(ckpt['num_diffusion_iters'])
    action_dim = int(ckpt['action_dim'])
    state_dim = ckpt.get('state_dim')
    state_key = ckpt.get('state_key', 'hole_centroid')
    pcd_n_points = int(ckpt.get('pcd_n_points', 512))
    eval_cam_viewmat = tuple(ckpt['eval_cam_viewmat'])
    eval_cam_resolution = int(ckpt['eval_cam_resolution'])
    eval_max_act_vel = float(ckpt['max_act_vel'])
    eval_sim_freq = int(ckpt['sim_freq'])
    eval_sim_steps_per_action = int(ckpt['sim_steps_per_action'])

    print(f'[init] obs_mode={obs_mode} obs_horizon={obs_horizon} '
          f'pred_horizon={pred_horizon} action_horizon={action_horizon}')
    print(f'[init] cam_viewmat={eval_cam_viewmat} '
          f'cam_resolution={eval_cam_resolution}')
    print(f'[init] MAX_ACT_VEL={eval_max_act_vel} '
          f'sim_freq={eval_sim_freq} '
          f'sim_steps_per_action={eval_sim_steps_per_action}')

    # CRITICAL parity patch — same as train_diffusion_bc.py does at startup.
    _orig_mav = DeformEnv.MAX_ACT_VEL
    DeformEnv.MAX_ACT_VEL = eval_max_act_vel
    print(f'[init] patched DeformEnv.MAX_ACT_VEL: '
          f'{_orig_mav} -> {DeformEnv.MAX_ACT_VEL}')

    # --- Load obs normalizer ----------------------------------------------
    norm_path = ckpt_path.parent / 'obs_normalizer.pkl'
    if not norm_path.exists():
        raise FileNotFoundError(
            f'obs_normalizer.pkl not found in {ckpt_path.parent}. '
            f'Required to match training-time obs statistics.')
    obs_normalizer = ObsNormalizer(obs_mode)
    with open(norm_path, 'rb') as f:
        obs_normalizer.load_state_dict(pickle.load(f))
    print(f'[init] loaded obs_normalizer from {norm_path}')
    act_normalizer = ActionNormalizer()  # no-op; actions live in [-1, 1]

    # --- Build encoder + policy -------------------------------------------
    if obs_mode == 'state':
        enc_kwargs = {'state_dim': int(state_dim)}
    elif obs_mode == 'rgb':
        enc_kwargs = {'pretrained': False}  # weights come from ckpt
    elif obs_mode == 'pcd':
        enc_kwargs = {'n_points': pcd_n_points, 'feat_dim': 256}
    else:
        raise ValueError(f'unknown obs_mode {obs_mode}')

    encoder = build_encoder(obs_mode, enc_kwargs).to(device)
    encoder.load_state_dict(ckpt['encoder_state_dict'])
    encoder.eval()

    policy = DiffusionPolicy(
        action_dim=action_dim,
        obs_feat_dim=encoder.feat_dim,
        obs_horizon=obs_horizon,
        pred_horizon=pred_horizon,
        action_horizon=action_horizon,
        num_diffusion_iters=num_diffusion_iters,
    ).to(device)
    policy.load_state_dict(ckpt['policy_state_dict'])
    policy.eval()
    print(f'[init] policy + encoder loaded; running eval...')

    # --- Build eval env (mirrors train_diffusion_bc._build_eval_env) ------
    sys_argv_backup = sys.argv
    sys.argv = [
        'eval',
        '--env=HangProcCloth-v1',
        f'--cam_resolution={eval_cam_resolution}',
        '--num_envs=0',
        '--total_env_steps=0',
        '--seed', str(args.eval_seed),
        '--max_episode_len', str(args.max_episode_len),
        f'--sim_freq={eval_sim_freq}',
        f'--sim_steps_per_action={eval_sim_steps_per_action}',
        '--cam_viewmat',
        *[f'{x:.6f}' for x in eval_cam_viewmat],
    ]
    try:
        dargs, _ = get_args_parser()
        args_postprocess(dargs)
        dargs.debug = False
        dargs.viz = False
        dargs.uint8_pixels = True
        e = gym.make(dargs.env, args=dargs)
    finally:
        sys.argv = sys_argv_backup
    e = RetryResetEnv(e)
    e.seed(args.eval_seed)
    deform = resolve_deform(e)

    # --- Helpers (mirror train_diffusion_bc._capture_obs_for_policy) ------
    def capture_obs(hole_idx, corner_idx):
        if obs_mode == 'state':
            return build_privileged_obs(
                deform, state_key, hole_idx, corner_indices=corner_idx)
        grip = np.asarray(deform.get_grip_obs(), dtype=np.float32)
        grip = np.clip(grip / 20.0, -2.0, 2.0)
        goal = np.asarray(deform.goal_pos[0], dtype=np.float32) / 20.0
        if obs_mode == 'rgb':
            rgb, _, _, _, _ = capture_rgb_depth(
                deform, eval_cam_resolution, eval_cam_resolution)
            return {'image': rgb, 'grip': grip, 'goal': goal}
        from _bc_obs_helpers import cloth_only_pcd as _cloth_only_pcd
        _, depth, seg, view, proj = capture_rgb_depth(
            deform, eval_cam_resolution, eval_cam_resolution)
        pcd = _cloth_only_pcd(depth, seg, view, proj,
                              deform.deform_id, pcd_n_points)
        return {'pcd': pcd, 'grip': grip, 'goal': goal}

    def stack_obs_seq(samples):
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
        return torch.from_numpy(
            arr.astype(np.float32)).unsqueeze(0).to(device)

    # --- Rollout loop ------------------------------------------------------
    s_hanging = s_topo = s_legacy = 0
    ep_lens = []
    with torch.no_grad():
        for ep in range(args.n_episodes):
            e.reset()
            deform.max_episode_len = args.max_episode_len  # belt-and-suspenders

            hole_idx = get_hole_indices(deform)
            if not hole_idx:
                ep_lens.append(0)
                print(f'  [ep {ep+1}/{args.n_episodes}] '
                      f'degenerate cloth, counted as failure')
                continue
            hole_loops = get_hole_loops(deform)
            _, verts0 = get_mesh_data(deform.sim, deform.deform_id)
            corner_idx = identify_cloth_corners(verts0)
            hole_radius = measure_hole_radius(deform, hole_idx)

            first_obs = capture_obs(hole_idx, corner_idx)
            obs_deque = collections.deque(
                [first_obs] * obs_horizon, maxlen=obs_horizon)

            step = 0
            done = False
            while not done and step < args.max_episode_len:
                normed = [obs_normalizer.apply(s) for s in obs_deque]
                obs_t = stack_obs_seq(normed)
                naction = policy.predict_action(obs_t, encoder).squeeze(0)
                naction = naction.cpu().numpy()
                start = obs_horizon - 1
                end = start + action_horizon
                chunk = act_normalizer.unapply(naction[start:end])
                for a in chunk:
                    a = np.clip(a, -1.0, 1.0).astype(np.float32)
                    _, _, done, info = e.step(a)
                    step += 1
                    new_obs = capture_obs(hole_idx, corner_idx)
                    obs_deque.append(new_obs)
                    if done or step >= args.max_episode_len:
                        break

            h = int(check_hanging_on_peg(
                deform, hole_idx, hole_radius, args.success_factor))
            t, _ = check_threaded_topological(deform, hole_loops)
            l = int(check_legacy(
                deform, hole_idx, hole_radius, args.success_factor))
            s_hanging += h
            s_topo += int(t)
            s_legacy += l
            ep_lens.append(step)
            print(f'  [ep {ep+1}/{args.n_episodes}] len={step:3d}  '
                  f'h={s_hanging}/{ep+1} t={s_topo}/{ep+1} l={s_legacy}/{ep+1}')

    e.close()

    n = args.n_episodes
    print(f'\n=== Eval results (n={n}, max_episode_len={args.max_episode_len}) ===')
    print(f'  legacy        = {s_legacy}/{n} = {s_legacy/n:.3f}')
    print(f'  hanging       = {s_hanging}/{n} = {s_hanging/n:.3f}')
    print(f'  topological   = {s_topo}/{n} = {s_topo/n:.3f}')
    print(f'  mean ep_len   = {np.mean(ep_lens):.1f}')


if __name__ == '__main__':
    main()
