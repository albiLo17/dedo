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
from _helpers import RetryResetEnv, compute_per_episode_max_len  # noqa: E402
from _bc_obs_helpers import (  # noqa: E402
    capture_rgb_depth, proj_matrix,
    check_hanging_on_peg, check_threaded_topological, check_legacy,
    measure_hole_radius, get_hole_indices, get_hole_loops,
    resolve_deform, patch_deform_render_to_obs_camera)
from experiments.hang_obs_exp.envs.privileged_env import (  # noqa: E402
    build_privileged_obs, identify_cloth_corners)
from _diffusion_policy import (  # noqa: E402
    DiffusionPolicy, build_encoder, ObsNormalizer, ActionNormalizer)


def write_mp4(frames, path, fps=30):
    """libx264 / yuv420p / +faststart mp4. Same encoding as
    train_diffusion_bc.py's _write_mp4 (browser-playable, wandb-compatible)."""
    if not frames:
        return
    import imageio
    writer = imageio.get_writer(
        path, fps=fps, codec='libx264', quality=8,
        macro_block_size=2, pixelformat='yuv420p',
        ffmpeg_params=['-movflags', '+faststart'])
    for f in frames:
        writer.append_data(f)
    writer.close()


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--ckpt', type=str, required=True,
                   help='Path to policy_best.pt / policy_epNNNN.pt. '
                        'obs_normalizer.pkl is auto-discovered in the '
                        'same directory.')
    p.add_argument('--max_episode_len', type=int, default=200,
                   help='Hard safety ceiling on per-episode length. '
                        'The actual per-episode cap is computed dynamically '
                        'from the scripted hole-aware trajectory for each '
                        'cloth (= len(traj) + --episode_tail_frames), '
                        'mirroring collect_bc_demos.py exactly. This flag '
                        'is the upper bound the dynamic value cannot exceed.')
    p.add_argument('--episode_tail_frames', type=int, default=None,
                   help='Brake-tail frames appended after the scripted traj. '
                        'Default None = read from ckpt metadata (which '
                        'matches collect-time exactly); explicit value '
                        'overrides. Should match what the demos were '
                        'collected with (5 in v3).')
    p.add_argument('--force_fixed_max_ep_len', type=int, default=None,
                   help='Skip the per-episode dynamic sizing and force '
                        'a single fixed max_episode_len for every episode. '
                        'Useful for A/B testing the OOD-tail failure mode '
                        '(e.g. --force_fixed_max_ep_len 200 reproduces the '
                        'pre-fix behavior).')
    p.add_argument('--n_episodes', type=int, default=30)
    p.add_argument('--eval_seed', type=int, default=2026 + 9999,
                   help='Matches train_diffusion_bc.py: '
                        'seed + eval_seed_offset(9999). Override to see '
                        'a different cloth distribution than the '
                        'training-time eval set.')
    p.add_argument('--success_factor', type=float, default=1.2)
    # ----- Video options -----
    p.add_argument('--n_video_episodes', type=int, default=0,
                   help='Record an mp4 for the first N episodes of the '
                        'rollout. 0 = no video. Each episode gets its own '
                        'mp4 named eval_ep<NNN>_<success_metric>.mp4 in '
                        '--video_dir.')
    p.add_argument('--video_dir', type=str, default=None,
                   help='Directory to write per-episode mp4s. Default = '
                        'a "videos" subdir alongside the ckpt.')
    p.add_argument('--video_render_size', type=int, default=512,
                   help='Per-frame H=W (square) for the recorded videos. '
                        '512 is a good balance of quality and disk size. '
                        'Independent of the policy obs camera resolution.')
    p.add_argument('--video_fps', type=int, default=30)
    p.add_argument('--settle_frame_stride', type=int, default=2,
                   help='Sub-sample the post-settle gravity-phase frames. '
                        'Matches collect_bc_demos.py debug-video default.')
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
    eval_ctrl_freq = float(ckpt.get('ctrl_freq',
                                    eval_sim_freq / eval_sim_steps_per_action))
    # Hanger-goal randomization the ckpt was trained under. Legacy ckpts
    # without this field default to 0 (v3-and-earlier fixed-goal behavior).
    eval_randomize_goal_radius = float(ckpt.get('randomize_goal_radius', 0.0))

    # Tail frames: CLI override wins, then ckpt metadata, then default 5
    # (matches collect_bc_demos.py default).
    if args.episode_tail_frames is not None:
        episode_tail_frames = int(args.episode_tail_frames)
        tail_source = 'CLI'
    elif 'episode_tail_frames' in ckpt:
        episode_tail_frames = int(ckpt['episode_tail_frames'])
        tail_source = 'ckpt'
    else:
        episode_tail_frames = 5
        tail_source = 'default (ckpt has no episode_tail_frames; legacy)'
    print(f'[init] episode_tail_frames = {episode_tail_frames} '
          f'(source: {tail_source})')

    print(f'[init] obs_mode={obs_mode} obs_horizon={obs_horizon} '
          f'pred_horizon={pred_horizon} action_horizon={action_horizon}')
    print(f'[init] cam_viewmat={eval_cam_viewmat} '
          f'cam_resolution={eval_cam_resolution}')
    print(f'[init] MAX_ACT_VEL={eval_max_act_vel} '
          f'sim_freq={eval_sim_freq} '
          f'sim_steps_per_action={eval_sim_steps_per_action}')
    print(f'[init] randomize_goal_radius={eval_randomize_goal_radius} m'
          f'{" (off — fixed goal)" if eval_randomize_goal_radius <= 0 else ""}')

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
        f'--randomize_goal_radius={eval_randomize_goal_radius}',
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
    # Always patch deform.render() to use the obs-camera projection
    # (fov=60) so any recorded mp4 frames + settle frames inside
    # make_final_steps match what the policy actually saw at training
    # time. Unconditional even when recording is off: keeps the invariant
    # robust if a future caller adds a render call.
    patch_deform_render_to_obs_camera(deform)

    # --- Video setup --------------------------------------------------------
    record_videos = args.n_video_episodes > 0
    if record_videos:
        if args.video_dir is None:
            args.video_dir = str(ckpt_path.parent / 'videos')
        os.makedirs(args.video_dir, exist_ok=True)
        print(f'[init] recording {args.n_video_episodes} eval videos '
              f'to {args.video_dir}')

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
    per_ep_maxes = []
    with torch.no_grad():
        for ep in range(args.n_episodes):
            is_recorded = record_videos and ep < args.n_video_episodes
            ep_frames = []
            e.reset()
            # Enable settle-frame capture AFTER reset so the flag isn't
            # clobbered by env initialization (same ordering as
            # collect_bc_demos.py).
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
                ep_lens.append(0)
                print(f'  [ep {ep+1}/{args.n_episodes}] '
                      f'degenerate cloth, counted as failure')
                continue
            hole_loops = get_hole_loops(deform)
            _, verts0 = get_mesh_data(deform.sim, deform.deform_id)
            corner_idx = identify_cloth_corners(verts0)
            hole_radius = measure_hole_radius(deform, hole_idx)

            # Per-episode max_ep_len: mirror collect_bc_demos.py exactly —
            # build the scripted traj this cloth would have used and cap
            # the episode at len(traj) + episode_tail_frames. The CLI
            # --force_fixed_max_ep_len escape hatch is for A/B-testing
            # the OOD-tail failure mode.
            if args.force_fixed_max_ep_len is not None:
                per_ep_max = int(args.force_fixed_max_ep_len)
            else:
                per_ep_max = compute_per_episode_max_len(
                    deform, ctrl_freq=eval_ctrl_freq,
                    tail_frames=episode_tail_frames,
                    safety_cap=args.max_episode_len)
                if per_ep_max is None:
                    per_ep_max = int(args.max_episode_len)
                    print(f'  [ep {ep+1}/{args.n_episodes}] WARN: scripted-'
                          f'traj build failed; falling back to safety cap '
                          f'{per_ep_max}')
            deform.max_episode_len = per_ep_max
            per_ep_maxes.append(per_ep_max)

            first_obs = capture_obs(hole_idx, corner_idx)
            obs_deque = collections.deque(
                [first_obs] * obs_horizon, maxlen=obs_horizon)

            step = 0
            done = False
            while not done and step < per_ep_max:
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
                    if is_recorded:
                        ep_frames.append(deform.render(
                            mode='rgb_array',
                            width=args.video_render_size,
                            height=args.video_render_size))
                    new_obs = capture_obs(hole_idx, corner_idx)
                    obs_deque.append(new_obs)
                    if done or step >= per_ep_max:
                        break
            # Settle phase frames captured by dedo inside make_final_steps.
            if is_recorded:
                settle_frames = (info.get('settle_frames', [])
                                 if isinstance(info, dict) else [])
                ep_frames.extend(settle_frames)
                deform._record_settle_frames = False

            h = int(check_hanging_on_peg(
                deform, hole_idx, hole_radius, args.success_factor))
            t, _ = check_threaded_topological(deform, hole_loops)
            l = int(check_legacy(
                deform, hole_idx, hole_radius, args.success_factor))
            s_hanging += h
            s_topo += int(t)
            s_legacy += l
            ep_lens.append(step)
            video_msg = ''
            if is_recorded and ep_frames:
                fname = (f'eval_ep{ep+1:03d}_seed{args.eval_seed}_'
                         f'l{l}_h{h}_t{int(t)}.mp4')
                fpath = os.path.join(args.video_dir, fname)
                try:
                    write_mp4(ep_frames, fpath, fps=args.video_fps)
                    video_msg = (f'  [video] {fname} '
                                 f'({len(ep_frames)} frames)')
                except Exception as _err:
                    video_msg = f'  [video] WARN: {_err!r}'
            print(f'  [ep {ep+1}/{args.n_episodes}] len={step:3d}  '
                  f'h={s_hanging}/{ep+1} t={s_topo}/{ep+1} l={s_legacy}/{ep+1}'
                  f'{video_msg}')

    e.close()

    n = args.n_episodes
    if args.force_fixed_max_ep_len is not None:
        sizing_desc = f'fixed max_episode_len={args.force_fixed_max_ep_len}'
    elif per_ep_maxes:
        sizing_desc = (f'per-ep dynamic, len(traj)+{episode_tail_frames}, '
                       f'min={min(per_ep_maxes)} max={max(per_ep_maxes)} '
                       f'mean={np.mean(per_ep_maxes):.1f} '
                       f'(safety cap={args.max_episode_len})')
    else:
        sizing_desc = f'no episodes ran'
    print(f'\n=== Eval results (n={n}, sizing: {sizing_desc}) ===')
    print(f'  legacy        = {s_legacy}/{n} = {s_legacy/n:.3f}')
    print(f'  hanging       = {s_hanging}/{n} = {s_hanging/n:.3f}')
    print(f'  topological   = {s_topo}/{n} = {s_topo/n:.3f}')
    print(f'  mean ep_len   = {np.mean(ep_lens):.1f}')


if __name__ == '__main__':
    main()
