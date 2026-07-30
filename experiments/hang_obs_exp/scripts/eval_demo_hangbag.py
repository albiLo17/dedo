"""
Evaluate the scripted HangBag motion planner over 50 episodes and write
overlaid videos + trajectory pkls for the first 10 episodes.

Per episode:
  - reset env, build bag-handle-aware waypoints, roll trajectory out via
    build_traj/merge_traj
  - record info['is_success'] (set by DeformEnv.step after make_final_steps
    releases the anchors and settles the bag under gravity for
    STEPS_AFTER_DONE sim steps)
  - measure the post-settle distance from primary loop centroid to the
    hook (this is the quantity the success criterion checks: < 0.125 m).

For the first --save_first N (default 10) episodes we ALSO:
  - render every action step to an RGB frame
  - enable underlying._record_settle_frames so make_final_steps appends
    the gravity-settle frames into info['settle_frames']
  - concat trajectory + settle frames into one buffer, overlay episode #,
    step counter, SUCCESS/FAIL banner + final distance, write mp4
  - save (obs, acts, rewards, success, final_dist_m) as demo_NNN.pkl

Usage (from repo root):
  python experiments/hang_obs_exp/scripts/eval_demo_hangbag.py
  python experiments/hang_obs_exp/scripts/eval_demo_hangbag.py \
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
from _helpers import RetryResetEnv, build_bag_handle_waypoints  # noqa: E402


# dedo's built-in success threshold (kept for reference / sanity in logs):
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
    p.add_argument('--env', type=str, default='HangBag-v1',
                   help='Full gym env id. Overridden by --version if set.')
    p.add_argument('--version', type=int, default=None,
                   choices=[0, 1, 2, 3],
                   help='Shortcut for --env=HangBag-v{N}. '
                        'v0=random mesh + textures per reset (108 totes); '
                        'v1=bag0_0 (major 0); v2=bag1_0 (major 1); '
                        'v3=bag2_0 (major 2). Overrides --env when set.')
    p.add_argument('--versions', type=str, default=None,
                   help='Comma-separated list of versions to sweep, e.g. '
                        '"0,1,2,3". Runs --num_episodes for EACH version and '
                        'reports per-version and aggregate success rates. '
                        'Each version\'s outputs land in logdir/v{N}/. '
                        'Takes priority over --version and --env.')
    p.add_argument('--primary_loop_idx', type=int, default=0,
                   help='Which handle loop is targeted to the hook.')
    p.add_argument('--success_dist_m', type=float, default=2.0,
                   help='Custom success threshold (m). Episode counts as a '
                        'success iff the post-settle distance from the primary '
                        'handle loop centroid to the hook is below this value. '
                        f'(Dedo\'s built-in threshold is '
                        f'{DEDO_DEFAULT_SUCCESS_DIST_M:.3f} m.)')
    p.add_argument('--logdir', type=str,
                   default=str(REPO_ROOT / 'logs' / 'hang_obs_exp' /
                               'eval_demo_hangbag'))
    p.add_argument('--viz', action='store_true',
                   help='Also open the pybullet GUI window.')
    return p.parse_args()


def compute_primary_loop_dist(underlying, loop_idx):
    """Distance from primary handle centroid to the hook (goal_pos[0])."""
    loops = underlying.args.deform_true_loop_vertices
    if len(loops) <= loop_idx:
        return float('nan')
    _, verts = get_mesh_data(underlying.sim, underlying.deform_id)
    verts = np.asarray(verts, dtype=np.float32)
    loop_verts = verts[loops[loop_idx]]
    loop_verts = loop_verts[~np.isnan(loop_verts).any(axis=1)]
    if len(loop_verts) == 0:
        return float('nan')
    centroid = loop_verts.mean(axis=0)
    goal = np.asarray(underlying.goal_pos[0], dtype=np.float32)
    return float(np.linalg.norm(centroid - goal))


def overlay_frame(frame_rgb, *, ep_idx, step_idx, total_steps,
                  phase_label, success, final_dist_m, success_dist_m):
    """Draw overlays on an RGB frame (returns BGR for cv2 VideoWriter)."""
    img = frame_rgb[..., ::-1].copy()  # RGB→BGR
    h, w = img.shape[:2]
    font = cv2.FONT_HERSHEY_SIMPLEX

    # Top-left: episode + step + phase
    cv2.putText(img, f'EP {ep_idx:02d}  step {step_idx:>4d}/{total_steps}',
                (10, 26), font, 0.6, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(img, phase_label,
                (10, 50), font, 0.55, (200, 200, 255), 2, cv2.LINE_AA)

    # Bottom banner: SUCCESS / FAIL (only meaningful once we know it; we
    # pass success=None during the trajectory phase to suppress the verdict).
    if success is not None:
        color = (0, 200, 0) if success else (0, 0, 200)  # BGR
        label = 'SUCCESS' if success else 'FAIL'
        dist_str = (f'  dist={final_dist_m:.3f}m'
                    f' (thresh<{success_dist_m:.3f}m)')
        text = label + dist_str
        (tw, th), _ = cv2.getTextSize(text, font, 0.7, 2)
        x = max(10, (w - tw) // 2)
        y = h - 18
        cv2.rectangle(img, (x - 8, y - th - 6), (x + tw + 8, y + 6),
                      (0, 0, 0), -1)
        cv2.putText(img, text, (x, y), font, 0.7, color, 2, cv2.LINE_AA)
    return img


def run_eval(extra, env_name, logdir, label):
    """Run --num_episodes of the motion planner on `env_name`, writing
    pkls/mp4s for the first --save_first to `logdir`. Returns a dict of
    per-version aggregate stats."""
    os.makedirs(logdir, exist_ok=True)
    print(f'\n[eval] {label}: env={env_name}  logdir={logdir}')

    sys.argv = [
        'eval_demo_hangbag',
        f'--env={env_name}',
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

    # Walk down to the bare DeformEnv (needed to toggle _record_settle_frames
    # and to access the live mesh state for the post-settle distance).
    def get_underlying():
        u = env
        while hasattr(u, 'env'):
            u = u.env
            if isinstance(u, DeformEnv):
                break
        return u

    n_success = 0
    successes = []         # bool per episode
    final_dists = []       # post-settle dist (m) per episode
    rewards = []           # total ep reward per episode
    peaks = []             # planned peak |vel| per episode

    record_video = extra.cam_resolution > 0

    for ep in range(extra.num_episodes):
        save_this = ep < extra.save_first
        env.reset()
        underlying = get_underlying()

        # Toggle settle-frame capture only for saved episodes (render cost).
        if save_this and record_video:
            underlying._record_settle_frames = True
            underlying._settle_render_kwargs = dict(
                width=extra.cam_resolution, height=extra.cam_resolution)
            underlying._settle_frame_stride = 2  # ~Nth sub-step; 2 keeps
            # the post-release falling motion visibly smooth (~50 frames
            # for the 500-sub-step gravity-settle phase).
        else:
            underlying._record_settle_frames = False

        ctrl_freq = args.sim_freq / args.sim_steps_per_action
        preset_wp = build_bag_handle_waypoints(
            underlying, primary_loop_idx=extra.primary_loop_idx)
        if preset_wp is None:
            print(f'[ep {ep:02d}] no loop info, skipping')
            successes.append(False)
            final_dists.append(float('nan'))
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
        traj_frames = []  # rendered frames during the planned trajectory

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
            # Skip the render on the done-step: by the time env.step returns
            # done=True, make_final_steps has already run internally (500
            # sub-steps of gravity settle), so rendering here would capture
            # the post-settle pose and the video would teleport before the
            # settle_frames play back the settling. settle_frames already
            # cover the settling phase chronologically.
            if save_this and record_video and not done:
                img = underlying.render(mode='rgb_array',
                                        width=extra.cam_resolution,
                                        height=extra.cam_resolution)
                traj_frames.append(img)
            step += 1

        # Episode done: env has already executed make_final_steps; mesh is
        # in its post-settle state. Measure the real success-criterion dist
        # and apply the user-specified threshold (overrides dedo's 0.125 m).
        final_dist = compute_primary_loop_dist(underlying,
                                               extra.primary_loop_idx)
        ep_success = int(np.isfinite(final_dist)
                         and final_dist < extra.success_dist_m)
        successes.append(bool(ep_success))
        final_dists.append(final_dist)
        rewards.append(ep_rwd)
        n_success += ep_success

        print(f'[ep {ep:02d}] success={ep_success}  '
              f'final_dist={final_dist:.3f}m  '
              f'(thresh<{extra.success_dist_m:.3f}m  '
              f'dedo_success={ep_dedo_success})  '
              f'reward={ep_rwd:.2f}  peak|vel|={peaks[-1]:.3f} m/s')

        if save_this:
            # --- Save trajectory pkl. ---
            pkl_path = os.path.join(logdir, f'demo_{ep:03d}.pkl')
            with open(pkl_path, 'wb') as f:
                pickle.dump({
                    'obs': np.asarray(ep_obs, dtype=np.float32),
                    'acts': np.asarray(ep_acts, dtype=np.float32),
                    'rewards': np.asarray(ep_rewards, dtype=np.float32),
                    'success': bool(ep_success),
                    'dedo_success': bool(ep_dedo_success),
                    'final_dist_m': final_dist,
                    'success_dist_m': float(extra.success_dist_m),
                    'dedo_success_dist_m': DEDO_DEFAULT_SUCCESS_DIST_M,
                    'final_reward': float(info.get('final_reward', 0.0)),
                    'seed': extra.seed,
                    'env': env_name,
                    'primary_loop_idx': extra.primary_loop_idx,
                    'len': len(ep_acts),
                }, f)

            # --- Write mp4 with overlay. ---
            if record_video:
                settle_frames = info.get('settle_frames', []) or []
                all_frames = [(f, 'TRAJECTORY') for f in traj_frames] + \
                             [(f, 'POST-RELEASE SETTLE') for f in settle_frames]
                total = len(all_frames)
                vid_path = os.path.join(logdir, f'demo_{ep:03d}.mp4')
                vw = cv2.VideoWriter(
                    vid_path, cv2.VideoWriter_fourcc(*'mp4v'), 24,
                    (extra.cam_resolution, extra.cam_resolution))
                for i, (frame_rgb, phase) in enumerate(all_frames):
                    # Once we're in the settle phase we know the verdict;
                    # for the trajectory phase, leave it blank so the
                    # viewer can judge the planning before the catch.
                    in_settle = phase.startswith('POST')
                    bgr = overlay_frame(
                        frame_rgb, ep_idx=ep, step_idx=i, total_steps=total,
                        phase_label=phase,
                        success=ep_success if in_settle else None,
                        final_dist_m=final_dist,
                        success_dist_m=extra.success_dist_m)
                    vw.write(bgr)
                vw.release()

    rate = 100.0 * n_success / max(extra.num_episodes, 1)
    finite_dists = [d for d in final_dists if np.isfinite(d)]
    finite_peaks = [p for p in peaks if np.isfinite(p)]
    print(f'\n---------- {label} summary ----------')
    print(f'   {n_success}/{extra.num_episodes}  = {rate:.1f}%')
    print(f'   success threshold: dist < {extra.success_dist_m:.3f} m  '
          f'(dedo default: {DEDO_DEFAULT_SUCCESS_DIST_M:.3f} m)')
    if finite_dists:
        finite = np.array(finite_dists)
        print(f'   final dist (m): '
              f'mean={finite.mean():.3f}  median={np.median(finite):.3f}  '
              f'min={finite.min():.3f}  max={finite.max():.3f}')
    if finite_peaks:
        gp = max(finite_peaks)
        print(f'   global peak |vel|: {gp:.3f} m/s  '
              f'(rec --max_act_vel = {np.ceil(gp * 1.2 * 10) / 10:.1f})')
    print(f'   saved {min(extra.save_first, extra.num_episodes)} '
          f'demo pkl(s)+mp4(s) in {logdir}')

    env.close()

    return {
        'label': label,
        'env': env_name,
        'logdir': logdir,
        'n_success': n_success,
        'n_episodes': extra.num_episodes,
        'rate': rate,
        'final_dists': final_dists,
        'peaks': peaks,
    }


def main():
    extra = parse_args()

    # Decide which versions to run.
    # --versions takes priority, then --version, then --env.
    if extra.versions is not None:
        try:
            versions = [int(v.strip()) for v in extra.versions.split(',')
                        if v.strip()]
        except ValueError:
            raise SystemExit(
                f'--versions must be a comma-separated list of ints, '
                f'got {extra.versions!r}')
        for v in versions:
            if v not in (0, 1, 2, 3):
                raise SystemExit(f'--versions: {v} not in {{0,1,2,3}}')
    elif extra.version is not None:
        versions = [extra.version]
    else:
        versions = None  # single run with extra.env as-is

    base_logdir = extra.logdir

    if versions is None:
        run_eval(extra, extra.env, base_logdir, label=extra.env)
        return

    results = []
    for v in versions:
        env_name = f'HangBag-v{v}'
        logdir = os.path.join(base_logdir, f'v{v}')
        results.append(run_eval(extra, env_name, logdir, label=env_name))

    # Aggregate sweep summary.
    print(f'\n========================================')
    print(f' HangBag sweep: {len(results)} version(s) × '
          f'{extra.num_episodes} ep')
    print(f' success threshold: dist < {extra.success_dist_m:.3f} m')
    print(f'----------------------------------------')
    total_succ = total_eps = 0
    for r in results:
        total_succ += r['n_success']
        total_eps += r['n_episodes']
        finite = [d for d in r['final_dists'] if np.isfinite(d)]
        median = float(np.median(finite)) if finite else float('nan')
        print(f'   {r["label"]:<14} {r["n_success"]:>3d}/{r["n_episodes"]:<3d} '
              f'= {r["rate"]:>5.1f}%   median_dist={median:.3f} m')
    agg_rate = 100.0 * total_succ / max(total_eps, 1)
    print(f'----------------------------------------')
    print(f'   AGGREGATE       {total_succ:>3d}/{total_eps:<3d} '
          f'= {agg_rate:>5.1f}%')
    print(f'========================================')


if __name__ == '__main__':
    main()
