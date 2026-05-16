"""
Evaluate the scripted ButtonProc motion planner over N episodes, write
overlaid videos + trajectory pkls for the first --save_first episodes,
and print the success rate.

Per episode:
  - reset env (procedural cloth + procedural button positions),
    build hole-aware per-anchor waypoints, roll trajectory out via
    build_traj/merge_traj.
  - record info['is_success'] (set by DeformEnv.step after
    make_final_steps releases anchors and settles the cloth).
  - compute the post-settle MEAN distance from each hole centroid to
    its goal (this is the quantity the env's success criterion checks).

For the first --save_first N (default 10) episodes we ALSO:
  - render every action step to an RGB frame
  - toggle underlying._record_settle_frames so make_final_steps appends
    settle-phase frames into info['settle_frames']
  - concat trajectory + settle frames, overlay episode #, step counter,
    SUCCESS/FAIL banner + final mean distance, write mp4
  - save (obs, acts, rewards, success, final_dist_m, per_loop_dists)
    as demo_NNN.pkl

Usage (from repo root):
  python experiments/hang_obs_exp/scripts/eval_demo_buttonproc.py
  python experiments/hang_obs_exp/scripts/eval_demo_buttonproc.py \
      --num_episodes 50 --save_first 10 --cam_resolution 480
"""
import sys, os, argparse, pickle
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT))

import numpy as np
import gym
import cv2

import dedo
from dedo.utils.args import get_args_parser, args_postprocess
from dedo.envs.deform_env import DeformEnv
from dedo.demo_preset import build_traj, merge_traj
from dedo.utils.mesh_utils import get_mesh_data

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _helpers import RetryResetEnv, build_button_proc_waypoints  # noqa: E402


# Dedo's built-in threshold (mean per-loop distance under 0.125 m):
#   |reward * FINAL_REWARD_MULT| < 2.5
#   |(-dist / 20) * 400|         < 2.5   →   dist < 0.125 m
DEDO_DEFAULT_SUCCESS_DIST_M = (DeformEnv.SUCESS_REWARD_TRESHOLD
                               * DeformEnv.WORKSPACE_BOX_SIZE
                               / DeformEnv.FINAL_REWARD_MULT)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--num_episodes', type=int, default=50)
    p.add_argument('--save_first', type=int, default=10,
                   help='Write mp4 + pkl for the first N episodes.')
    p.add_argument('--cam_resolution', type=int, default=480,
                   help='mp4 render size (per side). 0 disables video writing.')
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--env', type=str, default='ButtonProc-v0',
                   help='ButtonProc-v0 rerolls textures per reset; v1 pins '
                        'preset textures. Cloth is procedurally generated '
                        'every reset regardless of version.')
    p.add_argument('--success_dist_m', type=float, default=0.8,
                   help='Custom success threshold (m). Episode counts as a '
                        'success iff the mean post-settle distance from each '
                        'hole centroid to its assigned button is below this '
                        f'value. (Dedo\'s built-in threshold is '
                        f'{DEDO_DEFAULT_SUCCESS_DIST_M:.3f} m.)')
    p.add_argument('--logdir', type=str,
                   default=str(REPO_ROOT / 'logs' / 'hang_obs_exp' /
                               'eval_demo_buttonproc'))
    p.add_argument('--viz', action='store_true',
                   help='Also open the pybullet GUI window.')
    return p.parse_args()


def compute_loop_dists(underlying):
    """Return per-loop distance (centroid -> matching goal) for ButtonProc.

    Loop i is matched to goal_pos[i] (NOT nearest — dedo's reward
    function uses index-aligned pairing in deform_env.py:get_reward).
    Returns a list (possibly with NaN for unrecoverable loops)."""
    loops = underlying.args.deform_true_loop_vertices
    n = min(len(loops), len(underlying.goal_pos))
    _, verts = get_mesh_data(underlying.sim, underlying.deform_id)
    verts = np.asarray(verts, dtype=np.float32)
    dists = []
    for i in range(n):
        loop_verts = verts[loops[i]]
        loop_verts = loop_verts[~np.isnan(loop_verts).any(axis=1)]
        if len(loop_verts) == 0:
            dists.append(float('nan'))
            continue
        centroid = loop_verts.mean(axis=0)
        goal = np.asarray(underlying.goal_pos[i], dtype=np.float32)
        dists.append(float(np.linalg.norm(centroid - goal)))
    return dists


def overlay_frame(frame_rgb, *, ep_idx, step_idx, total_steps,
                  phase_label, success, mean_dist_m, per_loop_dists,
                  success_dist_m):
    """Draw overlays on an RGB frame (returns BGR for cv2 VideoWriter)."""
    img = frame_rgb[..., ::-1].copy()  # RGB→BGR
    h, w = img.shape[:2]
    font = cv2.FONT_HERSHEY_SIMPLEX

    cv2.putText(img, f'EP {ep_idx:02d}  step {step_idx:>4d}/{total_steps}',
                (10, 26), font, 0.6, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(img, phase_label,
                (10, 50), font, 0.55, (200, 200, 255), 2, cv2.LINE_AA)

    if success is not None:
        color = (0, 200, 0) if success else (0, 0, 200)  # BGR
        label = 'SUCCESS' if success else 'FAIL'
        per_loop_str = ', '.join(f'{d:.2f}' for d in per_loop_dists)
        dist_str = (f'  mean_dist={mean_dist_m:.3f}m  [{per_loop_str}]'
                    f' (thresh<{success_dist_m:.3f}m)')
        text = label + dist_str
        (tw, th), _ = cv2.getTextSize(text, font, 0.62, 2)
        x = max(10, (w - tw) // 2)
        y = h - 18
        cv2.rectangle(img, (x - 8, y - th - 6), (x + tw + 8, y + 6),
                      (0, 0, 0), -1)
        cv2.putText(img, text, (x, y), font, 0.62, color, 2, cv2.LINE_AA)
    return img


def main():
    extra = parse_args()
    os.makedirs(extra.logdir, exist_ok=True)
    print(f'[eval] env={extra.env}  logdir={extra.logdir}')

    sys.argv = [
        'eval_demo_buttonproc',
        f'--env={extra.env}',
        '--cam_resolution', '0',
        '--num_envs=0',
        '--total_env_steps=0',
        '--seed', str(extra.seed),
    ]
    args, _ = get_args_parser()
    args_postprocess(args)
    args.viz = extra.viz
    args.debug = False

    env = gym.make(args.env, args=args)
    env = RetryResetEnv(env)
    env.seed(extra.seed)

    def get_underlying():
        u = env
        while hasattr(u, 'env'):
            u = u.env
            if isinstance(u, DeformEnv):
                break
        return u

    n_success = 0
    final_mean_dists = []
    rewards = []
    peaks = []
    record_video = extra.cam_resolution > 0

    for ep in range(extra.num_episodes):
        save_this = ep < extra.save_first
        env.reset()
        underlying = get_underlying()

        if save_this and record_video:
            underlying._record_settle_frames = True
            underlying._settle_render_kwargs = dict(
                width=extra.cam_resolution, height=extra.cam_resolution)
            underlying._settle_frame_stride = 2
        else:
            underlying._record_settle_frames = False

        ctrl_freq = args.sim_freq / args.sim_steps_per_action
        preset_wp = build_button_proc_waypoints(underlying)
        if preset_wp is None:
            print(f'[ep {ep:02d}] no loop/goal info, skipping')
            final_mean_dists.append(float('nan'))
            rewards.append(0.0)
            peaks.append(float('nan'))
            continue

        _, vel_a = build_traj(underlying, preset_wp, 'a',
                              anchor_idx=0, ctrl_freq=ctrl_freq, robot=None)
        _, vel_b = build_traj(underlying, preset_wp, 'b',
                              anchor_idx=1, ctrl_freq=ctrl_freq, robot=None)
        traj = merge_traj(vel_a, vel_b)
        last = np.zeros_like(traj[0])
        peaks.append(float(np.abs(traj).max()))

        ep_obs, ep_acts, ep_rewards = [], [], []
        traj_frames = []

        step = 0
        ep_rwd = 0.0
        ep_dedo_success = 0
        info = {}
        done = False
        while not done:
            act_unscaled = traj[step] if step < len(traj) else last
            normalized = np.clip(act_unscaled / DeformEnv.MAX_ACT_VEL,
                                 -1.0, 1.0).astype(np.float32)
            obs, rwd, done, info = env.step(normalized)
            ep_obs.append(np.asarray(obs, dtype=np.float32))
            ep_acts.append(normalized)
            ep_rewards.append(float(rwd))
            ep_rwd += float(rwd)
            if 'is_success' in info:
                ep_dedo_success = max(ep_dedo_success, int(info['is_success']))
            # Same teleport-guard as eval_demo_hangbag: by the time done=True
            # is returned, make_final_steps has already advanced the sim
            # 500 sub-steps; settle_frames already cover that phase.
            if save_this and record_video and not done:
                img = underlying.render(mode='rgb_array',
                                        width=extra.cam_resolution,
                                        height=extra.cam_resolution)
                traj_frames.append(img)
            step += 1

        per_loop = compute_loop_dists(underlying)
        finite = [d for d in per_loop if np.isfinite(d)]
        mean_dist = float(np.mean(finite)) if finite else float('nan')
        ep_success = int(np.isfinite(mean_dist)
                         and mean_dist < extra.success_dist_m)
        final_mean_dists.append(mean_dist)
        rewards.append(ep_rwd)
        n_success += ep_success

        per_loop_str = ', '.join(f'{d:.3f}' for d in per_loop)
        print(f'[ep {ep:02d}] success={ep_success}  '
              f'mean_dist={mean_dist:.3f}m  per_loop=[{per_loop_str}]m  '
              f'(thresh<{extra.success_dist_m:.3f}m  '
              f'dedo_success={ep_dedo_success})  '
              f'reward={ep_rwd:.2f}  peak|vel|={peaks[-1]:.3f} m/s')

        if save_this:
            pkl_path = os.path.join(extra.logdir, f'demo_{ep:03d}.pkl')
            with open(pkl_path, 'wb') as f:
                pickle.dump({
                    'obs': np.asarray(ep_obs, dtype=np.float32),
                    'acts': np.asarray(ep_acts, dtype=np.float32),
                    'rewards': np.asarray(ep_rewards, dtype=np.float32),
                    'success': bool(ep_success),
                    'dedo_success': bool(ep_dedo_success),
                    'final_mean_dist_m': mean_dist,
                    'per_loop_dists_m': per_loop,
                    'success_dist_m': float(extra.success_dist_m),
                    'dedo_success_dist_m': DEDO_DEFAULT_SUCCESS_DIST_M,
                    'final_reward': float(info.get('final_reward', 0.0)),
                    'seed': extra.seed,
                    'env': extra.env,
                    'len': len(ep_acts),
                }, f)

            if record_video:
                settle_frames = info.get('settle_frames', []) or []
                all_frames = [(f, 'TRAJECTORY') for f in traj_frames] + \
                             [(f, 'POST-RELEASE SETTLE') for f in settle_frames]
                total = len(all_frames)
                vid_path = os.path.join(extra.logdir, f'demo_{ep:03d}.mp4')
                vw = cv2.VideoWriter(
                    vid_path, cv2.VideoWriter_fourcc(*'mp4v'), 24,
                    (extra.cam_resolution, extra.cam_resolution))
                for i, (frame_rgb, phase) in enumerate(all_frames):
                    in_settle = phase.startswith('POST')
                    bgr = overlay_frame(
                        frame_rgb, ep_idx=ep, step_idx=i, total_steps=total,
                        phase_label=phase,
                        success=ep_success if in_settle else None,
                        mean_dist_m=mean_dist,
                        per_loop_dists=per_loop,
                        success_dist_m=extra.success_dist_m)
                    vw.write(bgr)
                vw.release()

    rate = 100.0 * n_success / max(extra.num_episodes, 1)
    finite_dists = [d for d in final_mean_dists if np.isfinite(d)]
    finite_peaks = [p for p in peaks if np.isfinite(p)]
    print(f'\n========================================')
    print(f' ButtonProc motion-planner success rate ')
    print(f'   {n_success}/{extra.num_episodes}  '
          f'= {rate:.1f}%')
    print(f'   success threshold: mean_dist < {extra.success_dist_m:.3f} m  '
          f'(dedo default: {DEDO_DEFAULT_SUCCESS_DIST_M:.3f} m)')
    if finite_dists:
        f = np.array(finite_dists)
        print(f'   final mean dist (m): '
              f'mean={f.mean():.3f}  median={np.median(f):.3f}  '
              f'min={f.min():.3f}  max={f.max():.3f}')
    if finite_peaks:
        gp = max(finite_peaks)
        print(f'   global peak |vel|: {gp:.3f} m/s  '
              f'(rec --max_act_vel = {np.ceil(gp * 1.2 * 10) / 10:.1f})')
    print(f'   saved {min(extra.save_first, extra.num_episodes)} '
          f'demo pkl(s)+mp4(s) in {extra.logdir}')
    print(f'========================================')

    env.close()


if __name__ == '__main__':
    main()
