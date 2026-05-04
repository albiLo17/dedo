"""
Record manual point-cloud demos on HangProcCloth-v1 via keyboard teleop.

Same UX as record_demo.py (pynput, arrow keys + per-anchor nudges, ENTER
to save, R to discard, ESC to quit) but the wrapper is
PointCloudObsWrapper, so each saved (obs, action) pair has the depth-
based PCD as obs — directly usable for BC pretrain in train_pointcloud.py.

REQUIREMENT: pip install pynput

Usage (from repo root):
  python experiments/hang_obs_exp/scripts/record_demo_pcd.py \
      --demos_dir logs/hang_obs_exp/manual_demos_pcd

Saves to <demos_dir>/demo_NNN.pkl with:
  {'obs':       (T, 12 + n_points*3) float32   # PCD obs
   'acts':      (T, 6) float32
   'reward':    float
   'success':   int                            # uses adaptive threshold
   'obs_type':  'pointcloud'
   'n_points':  int                            # so loader can verify shape
   'cam_resolution_pcd': int
   'success_factor': float | None
   'len':       int}
"""
import sys, os, time, argparse, pickle, threading
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT))

import numpy as np
import gym

try:
    from pynput import keyboard as pkb
except ImportError:
    print('ERROR: pynput not installed. Run:\n    pip install pynput\n'
          'then re-run this script.', file=sys.stderr)
    sys.exit(1)

import dedo
from dedo.utils.args import get_args_parser, args_postprocess
from dedo.envs.deform_env import DeformEnv
from dedo.utils.mesh_utils import get_mesh_data

from experiments.hang_obs_exp.envs.pointcloud_env import PointCloudObsWrapper
sys.path.insert(0, str(Path(__file__).resolve().parent))
from _helpers import RetryResetEnv  # noqa: E402


# ---------------------------------------------------------------------------
# pynput-based held-key tracker (same as record_demo.py).
# ---------------------------------------------------------------------------
class KeyState:
    def __init__(self):
        self._held = set()
        self._triggers = []
        self._lock = threading.Lock()
        self._listener = None

    def start(self):
        self._listener = pkb.Listener(on_press=self._on_press,
                                       on_release=self._on_release,
                                       suppress=False)
        self._listener.daemon = True
        self._listener.start()

    def stop(self):
        if self._listener is not None:
            self._listener.stop()

    @staticmethod
    def _tok(key):
        if isinstance(key, pkb.KeyCode) and key.char is not None:
            return key.char.lower()
        if key == pkb.Key.left:  return 'left'
        if key == pkb.Key.right: return 'right'
        if key == pkb.Key.up:    return 'up'
        if key == pkb.Key.down:  return 'down'
        if key == pkb.Key.shift or key == pkb.Key.shift_l or key == pkb.Key.shift_r:
            return 'shift'
        if key == pkb.Key.enter: return 'enter'
        if key == pkb.Key.esc:   return 'esc'
        return None

    def _on_press(self, key):
        tok = self._tok(key)
        if tok is None:
            return
        with self._lock:
            if tok not in self._held:
                self._triggers.append(tok)
            self._held.add(tok)

    def _on_release(self, key):
        tok = self._tok(key)
        if tok is None:
            return
        with self._lock:
            self._held.discard(tok)

    def is_down(self, tok):
        with self._lock:
            return tok in self._held

    def consume_triggers(self):
        with self._lock:
            t = self._triggers
            self._triggers = []
        return t


def compute_action(ks):
    fast = ks.is_down('shift')
    s = 0.6 if fast else 0.25

    dx = (1.0 if ks.is_down('right') else 0.0) - (1.0 if ks.is_down('left') else 0.0)
    dy = (1.0 if ks.is_down('up')    else 0.0) - (1.0 if ks.is_down('down') else 0.0)
    dz = (1.0 if ks.is_down('x')     else 0.0) - (1.0 if ks.is_down('z')    else 0.0)

    a = np.array([dx, dy, dz], dtype=np.float32)
    b = np.array([dx, dy, dz], dtype=np.float32)

    if ks.is_down('a'): a[0] -= 0.7
    if ks.is_down('d'): a[0] += 0.7
    if ks.is_down('s'): a[1] -= 0.7
    if ks.is_down('w'): a[1] += 0.7
    if ks.is_down('q'): a[2] -= 0.7
    if ks.is_down('e'): a[2] += 0.7

    if ks.is_down('j'): b[0] -= 0.7
    if ks.is_down('l'): b[0] += 0.7
    if ks.is_down('k'): b[1] -= 0.7
    if ks.is_down('i'): b[1] += 0.7
    if ks.is_down('u'): b[2] -= 0.7
    if ks.is_down('o'): b[2] += 0.7

    return np.clip(np.concatenate([a, b]) * s, -1.0, 1.0).astype(np.float32)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
parser = argparse.ArgumentParser()
parser.add_argument('--demos_dir', type=str, required=True)
parser.add_argument('--seed', type=int, default=42)
parser.add_argument('--fps', type=float, default=30.0)
parser.add_argument('--max_episode_len', type=int, default=None)
parser.add_argument('--n_points', type=int, default=512)
parser.add_argument('--cam_resolution_pcd', type=int, default=128)
parser.add_argument('--success_factor', type=float, default=1.2,
                    help='Adaptive success threshold factor (same as '
                         'train_pointcloud.py default).')
extra = parser.parse_args()

os.makedirs(extra.demos_dir, exist_ok=True)

# Build dedo args. Need viz=True for the live GUI window AND
# cam_resolution > 0 for PCD depth render.
sys.argv = [
    'record_demo_pcd',
    '--env=HangProcCloth-v1',
    f'--cam_resolution={extra.cam_resolution_pcd}',
    '--num_envs=0',
    '--total_env_steps=0',
    '--seed', str(extra.seed),
    '--viz',
]
args, _ = get_args_parser()
args_postprocess(args)
args.viz = True
args.debug = False
args.cam_viewmat = [9.0, -25.0, 45.0, 0.0, 0.5, 6.5]
if extra.max_episode_len is not None:
    args.max_episode_len = extra.max_episode_len

env = gym.make(args.env, args=args)
env = RetryResetEnv(env)
env = PointCloudObsWrapper(
    env,
    n_points=extra.n_points,
    cam_resolution=extra.cam_resolution_pcd,
    success_factor=extra.success_factor,
    success_bonus=0.0,   # don't shape during recording
    fail_penalty=0.0)
env.seed(extra.seed)

# Underlying DeformEnv for sim handle / mesh access.
underlying = env
while hasattr(underlying, 'env'):
    underlying = underlying.env
    if isinstance(underlying, DeformEnv):
        break

ks = KeyState()
ks.start()

print('\n=== Manual demo recorder — POINT CLOUD (pynput) ===')
print(f'  demos_dir: {extra.demos_dir}')
print(f'  obs: PCD ({extra.n_points} points) + 12 gripper '
      f'(cam_res={extra.cam_resolution_pcd})')
print(f'  success_factor: {extra.success_factor} (adaptive threshold)')
print('  Arrows: lateral xy   z/x: lower/raise   SHIFT: faster')
print('  Per-anchor nudges: w/a/s/d/q/e (a)   i/j/k/l/u/o (b)')
print('  ENTER=save demo   R=discard+reset   ESC=quit')
print('  NOTE: pynput captures keys system-wide.\n')


def next_demo_id():
    existing = [p for p in os.listdir(extra.demos_dir)
                if p.startswith('demo_') and p.endswith('.pkl')]
    nums = []
    for p in existing:
        try:
            nums.append(int(p[len('demo_'):-len('.pkl')]))
        except ValueError:
            pass
    return max(nums) + 1 if nums else 0


def _success_thresh_m():
    return env.success_threshold_m


def _hole_to_goal_dist():
    if not env._hole_vertex_indices:
        return float('nan')
    _, verts = get_mesh_data(underlying.sim, underlying.deform_id)
    verts = np.asarray(verts, dtype=np.float32)
    hv = verts[env._hole_vertex_indices]
    hv = hv[~np.isnan(hv).any(axis=1)]
    if len(hv) == 0:
        return float('nan')
    centroid = hv.mean(axis=0)
    goal = np.asarray(underlying.goal_pos[0], dtype=np.float32)
    return float(np.linalg.norm(centroid - goal))


def _save_demo(ep_obs, ep_act, ep_rwd, ep_success):
    idx = next_demo_id()
    path = os.path.join(extra.demos_dir, f'demo_{idx:03d}.pkl')
    payload = {
        'obs':  np.asarray(ep_obs, dtype=np.float32),
        'acts': np.asarray(ep_act, dtype=np.float32),
        'reward': float(ep_rwd),
        'success': int(ep_success),
        'obs_type': 'pointcloud',
        'n_points': extra.n_points,
        'cam_resolution_pcd': extra.cam_resolution_pcd,
        'success_factor': extra.success_factor,
        'len': len(ep_act),
    }
    with open(path, 'wb') as f:
        pickle.dump(payload, f)
    print(f'[record] saved {path}  len={len(ep_act)}  '
          f'rwd={ep_rwd:.2f}  success={ep_success}')


running = True
while running:
    obs = env.reset()
    ep_obs, ep_act = [], []
    ep_rwd = 0.0
    ep_success = 0
    ks.consume_triggers()
    print(f'[record] new episode (already saved: {next_demo_id()})')
    dt = 1.0 / max(extra.fps, 1.0)

    while True:
        loop_start = time.time()
        triggers = ks.consume_triggers()
        if 'esc' in triggers:
            print('[record] ESC — quitting (current discarded).')
            running = False
            break
        if 'r' in triggers:
            print('[record] R — discarding episode, resetting.')
            break
        if 'enter' in triggers:
            if len(ep_act) == 0:
                print('[record] empty buffer — nothing to save.')
            else:
                _save_demo(ep_obs, ep_act, ep_rwd, ep_success)
            break

        action = compute_action(ks)
        ep_obs.append(np.asarray(obs, dtype=np.float32))
        ep_act.append(action)
        obs, rwd, done, info = env.step(action)
        ep_rwd += float(rwd)
        if 'is_success' in info:
            ep_success = max(ep_success, int(info['is_success']))

        if len(ep_act) % 5 == 0 and not done:
            d = _hole_to_goal_dist()
            thresh = _success_thresh_m()
            marker = '✓' if d < thresh else ' '
            extra_info = ''
            if env.hole_radius is not None:
                extra_info = f'  hole_r={env.hole_radius:.3f}m'
            sys.stdout.write(
                f'\r[record] step={len(ep_act):>4d}  '
                f'dist={d:.3f}m  thresh<{thresh:.3f}m  {marker}'
                f'{extra_info}    ')
            sys.stdout.flush()

        if done:
            sys.stdout.write('\n')
            final_dist = _hole_to_goal_dist()
            thresh = _success_thresh_m()
            print(f'[record] env done — len={len(ep_act)} '
                  f'rwd={ep_rwd:.2f}  success={ep_success}  '
                  f'final_dist={final_dist:.3f}m '
                  f'(needs <{thresh:.3f}m).')
            if ep_success == 0 and final_dist < 0.5:
                print('[record] tip: cloth was close — release keys for '
                      '~1s before ENTER so anchors quiesce during settling.')
            print('  ENTER=save  R=discard  ESC=quit')
            while True:
                triggers = ks.consume_triggers()
                if 'enter' in triggers:
                    _save_demo(ep_obs, ep_act, ep_rwd, ep_success)
                    break
                if 'r' in triggers:
                    print('[record] discarded.')
                    break
                if 'esc' in triggers:
                    running = False
                    break
                time.sleep(0.05)
            break

        elapsed = time.time() - loop_start
        if elapsed < dt:
            time.sleep(dt - elapsed)

ks.stop()
env.close()
print('Done.')
