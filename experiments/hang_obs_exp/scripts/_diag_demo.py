"""
Diagnostic runner: executes the SAME scripted policy view_demo.py uses,
but logs hole_centroid + gripper positions + hole->hanger distance every
control step so we can see exactly where the current waypoints fail.

Usage:
  python experiments/hang_obs_exp/scripts/_diag_demo.py --num_episodes 3
"""
import sys, os, argparse
from pathlib import Path
REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT))

import numpy as np
import gym

import dedo  # noqa: F401
from dedo.utils.args import get_args_parser, args_postprocess
from dedo.utils.mesh_utils import get_mesh_data
from dedo.envs.deform_env import DeformEnv
from dedo.demo_preset import build_traj, merge_traj

from experiments.hang_obs_exp.envs.privileged_env import PrivilegedObsWrapper
sys.path.insert(0, str(Path(__file__).resolve().parent))
from _helpers import RetryResetEnv, build_hole_aware_waypoints  # noqa: E402

parser = argparse.ArgumentParser()
parser.add_argument('--num_episodes', type=int, default=3)
parser.add_argument('--seed', type=int, default=42)
parser.add_argument('--log_every', type=int, default=10,
                    help='Print one diagnostic line every N control steps')
parser.add_argument('--success_factor', type=float, default=1.2,
                    help='Adaptive success threshold; pass <0 to disable '
                         'and use dedo\'s fixed 0.125 m.')
parser.add_argument('--randomize_goal_radius', type=float, default=0.0,
                    help='Half-extent of the per-episode xy hanger '
                         'randomization box (meters). 0 keeps the legacy '
                         'fixed-goal behavior. Useful for stress-testing '
                         'the scripted controller\'s reach at the corners '
                         'of the v4 randomization box.')
extra = parser.parse_args()
sf = None if extra.success_factor < 0 else float(extra.success_factor)

sys.argv = ['_diag_demo', '--env=HangProcCloth-v1',
            '--cam_resolution=0', '--num_envs=0', '--total_env_steps=0',
            '--seed', str(extra.seed),
            f'--randomize_goal_radius={extra.randomize_goal_radius}']
args, _ = get_args_parser()
args_postprocess(args)
args.viz = False
args.debug = False

env = gym.make(args.env, args=args)
env = RetryResetEnv(env)
env = PrivilegedObsWrapper(env, obs_mode='hole_centroid', success_factor=sf)
env.seed(extra.seed)
print(f'[_diag_demo] success criterion: sf={sf}')


def hole_centroid_now(underlying, hole_idxs):
    _, verts = get_mesh_data(underlying.sim, underlying.deform_id)
    verts = np.array(verts, dtype=np.float32)
    h = verts[hole_idxs]
    h = h[~np.isnan(h).any(axis=1)]
    if len(h) == 0:
        return None
    return h.mean(axis=0)


for ep in range(extra.num_episodes):
    print(f'\n========== EP {ep} ==========')
    obs = env.reset()
    underlying = env
    while hasattr(underlying, 'env'):
        underlying = underlying.env
        if isinstance(underlying, DeformEnv):
            break

    loops = underlying.args.deform_true_loop_vertices
    hole_idxs = [i for loop in loops for i in loop]

    hole0 = hole_centroid_now(underlying, hole_idxs)
    grip_a0 = np.array(underlying.anchors[underlying.anchor_ids[0]]['pos'],
                       dtype=np.float32)
    grip_b0 = np.array(underlying.anchors[underlying.anchor_ids[1]]['pos'],
                       dtype=np.float32)
    hanger = np.array(underlying.goal_pos[0], dtype=np.float32)
    print(f'init  hanger    = {hanger}')
    print(f'init  hole_cent = {hole0}     dist_to_hanger = '
          f'{np.linalg.norm(hole0 - hanger):.3f}')
    print(f'init  grip_a    = {grip_a0}')
    print(f'init  grip_b    = {grip_b0}')
    print(f'init  delta_a   = grip_a - hole = {grip_a0 - hole0}')
    print(f'init  delta_b   = grip_b - hole = {grip_b0 - hole0}')

    ctrl_freq = args.sim_freq / args.sim_steps_per_action
    preset_wp = build_hole_aware_waypoints(underlying)
    print(f'computed wp_a   = {preset_wp["a"]}')
    print(f'computed wp_b   = {preset_wp["b"]}')

    _, vel_a = build_traj(underlying, preset_wp, 'a',
                          anchor_idx=0, ctrl_freq=ctrl_freq, robot=None)
    _, vel_b = build_traj(underlying, preset_wp, 'b',
                          anchor_idx=1, ctrl_freq=ctrl_freq, robot=None)
    traj = merge_traj(vel_a, vel_b)
    last = np.zeros_like(traj[0])
    print(f'traj len        = {len(traj)} ctrl steps  '
          f'(max_episode_len={args.max_episode_len})')

    step = 0
    closest = (1e9, None, None)
    while True:
        act_unscaled = traj[step] if step < len(traj) else last
        normalized = np.clip(act_unscaled / DeformEnv.MAX_ACT_VEL, -1.0, 1.0)
        obs, rwd, done, info = env.step(normalized.astype(np.float32))

        if step % extra.log_every == 0 or done:
            hc = hole_centroid_now(underlying, hole_idxs)
            ga = np.array(underlying.anchors[underlying.anchor_ids[0]]['pos'],
                          dtype=np.float32)
            d = (np.linalg.norm(hc - hanger)
                 if hc is not None else float('nan'))
            if hc is not None and d < closest[0]:
                closest = (d, step, hc.copy())
            tag = ''
            if step >= len(traj):
                tag = ' [HOLD]'
            elif step >= len(traj) - 1:
                tag = ' [END_TRAJ]'
            print(f'  step {step:>3d}{tag}  hole={hc}  d={d:.3f}  '
                  f'grip_a_z={ga[2]:.3f}')
        if done:
            break
        step += 1
    succ = info.get('is_success', 0) if isinstance(info, dict) else 0
    fr = info.get('final_reward', float('nan')) if isinstance(info, dict) else float('nan')
    adapt_dist = info.get('adaptive_dist', None) if isinstance(info, dict) else None
    adapt_th = info.get('adaptive_thresh', None) if isinstance(info, dict) else None
    h_r = info.get('hole_radius', None) if isinstance(info, dict) else None
    print(f'  END  closest hole->hanger dist seen during traj = '
          f'{closest[0]:.3f} at step {closest[1]}')
    extra_msg = ''
    if adapt_dist is not None and adapt_th is not None:
        extra_msg = (f'  | adaptive: dist={adapt_dist:.3f} thresh<{adapt_th:.3f}'
                     f' (hole_r={h_r:.3f})')
    print(f'  END  is_success={int(bool(succ))}  final_reward={fr:.3f}{extra_msg}')

env.close()
