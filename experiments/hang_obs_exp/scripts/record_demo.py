"""
Record manual demos on HangProcCloth-v1 via keyboard teleop (pynput-based).

Opens pybullet's GUI; you drive BOTH gripper anchors with the keyboard.
We use pynput (system-wide keyboard listener) instead of pybullet's
getKeyboardEvents() because pybullet's GUI intercepts letter keys for its
own debug toggles (wireframe, AABBs, etc.) and held-key tracking is
unreliable on Windows.

REQUIREMENT: pip install pynput

Controls (BOTH anchors move together unless noted):
  Arrow keys      lateral xy translation
  z / x           lower / raise
  SHIFT held      faster (0.6 instead of 0.25 of MAX_ACT_VEL)

  Per-anchor differential nudge (added on top):
    w/s   a-anchor  +y / -y
    a/d   a-anchor  -x / +x
    q/e   a-anchor  -z / +z
    i/k   b-anchor  +y / -y
    j/l   b-anchor  -x / +x
    u/o   b-anchor  -z / +z

  ENTER (Return)  save current episode as a demo
  R               discard current episode and reset
  ESC             quit (saves nothing extra)

Saves to <demos_dir>/demo_NNN.pkl with all 3 obs modes computed from the
same trajectory (so a single recorded demo can train any of the three
privileged conditions).

Usage (from repo root, with the dedo conda env activated):
  python experiments/hang_obs_exp/scripts/record_demo.py \
      --demos_dir logs/hang_obs_exp/manual_demos
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

from experiments.hang_obs_exp.envs.privileged_env import (
    PrivilegedObsWrapper, build_privileged_obs)
sys.path.insert(0, str(Path(__file__).resolve().parent))
from _helpers import RetryResetEnv  # noqa: E402

ALL_MODES = ('hole_centroid', 'hole_vertices', 'full_mesh')


# ---------------------------------------------------------------------------
# pynput-based held-key tracker. Captures keys system-wide so pybullet's
# GUI shortcuts don't eat them. Lock-protected because the listener
# callback runs on a separate thread.
# ---------------------------------------------------------------------------
class KeyState:
    def __init__(self):
        self._held = set()           # set of normalized key tokens
        self._triggers = []          # list of one-shot tokens (cleared on read)
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
        # Letter / number keys -> single lowercase char.
        # Arrow / shift / enter / esc -> a string label.
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

    # Per-anchor differential nudges.
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


# Main --------------------------------------------------------------------
parser = argparse.ArgumentParser()
parser.add_argument('--demos_dir', type=str, required=True,
                    help='Where to save demo_NNN.pkl files')
parser.add_argument('--obs_mode', type=str, default='hole_centroid',
                    choices=['hole_centroid', 'hole_vertices', 'full_mesh'],
                    help='Mode used to STEP the env during recording. The '
                         'pkl always saves all three modes.')
parser.add_argument('--seed', type=int, default=42)
parser.add_argument('--fps', type=float, default=30.0,
                    help='Real-time control loop rate (Hz)')
parser.add_argument('--max_episode_len', type=int, default=None,
                    help='Override env max_episode_len (default = env default). '
                         'Bump this if you need more time per recording.')
parser.add_argument('--success_factor', type=float, default=1.2,
                    help='If set, success threshold becomes '
                         'dist < success_factor * hole_radius (adaptive to '
                         'hole size). e.g. 0.8 = pass if centroid is within '
                         '80%% of hole radius from goal. Default: None '
                         '(use dedo fixed 0.125m).')
extra = parser.parse_args()

os.makedirs(extra.demos_dir, exist_ok=True)

sys.argv = [
    'record_demo',
    '--env=HangProcCloth-v1',
    '--cam_resolution=0',
    '--num_envs=0',
    '--total_env_steps=0',
    '--seed', str(extra.seed),
    '--viz',
]
args, _ = get_args_parser()
args_postprocess(args)
args.viz = True
args.debug = False
if extra.max_episode_len is not None:
    args.max_episode_len = extra.max_episode_len

env = gym.make(args.env, args=args)
env = RetryResetEnv(env)
env = PrivilegedObsWrapper(env, obs_mode=extra.obs_mode,
                            success_factor=extra.success_factor)
env.seed(extra.seed)

underlying = env
while hasattr(underlying, 'env'):
    underlying = underlying.env
    if isinstance(underlying, DeformEnv):
        break

ks = KeyState()
ks.start()

print('\n=== Manual demo recorder (pynput) ===')
print(f'  demos_dir: {extra.demos_dir}')
print(f'  pkls always contain all 3 obs modes ({", ".join(ALL_MODES)})')
print('  Arrows: lateral xy   z/x: lower/raise   SHIFT: faster')
print('  Per-anchor nudges: w/a/s/d/q/e (a)   i/j/k/l/u/o (b)')
print('  ENTER=save demo   R=discard+reset   ESC=quit')
print('  NOTE: pynput captures keys system-wide — you do NOT need to '
      'click into the pybullet window first.\n')


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


def _capture_all_obs():
    return {
        m: build_privileged_obs(underlying, m, env._hole_vertex_indices)
        for m in ALL_MODES
    }


# Live success threshold (m). Pulled from the wrapper so it reflects the
# adaptive (hole-radius-proportional) threshold when --success_factor is
# set, or dedo's fixed 0.125 m when it isn't.
def _success_thresh_m():
    return env.success_threshold_m


def _hole_to_goal_dist():
    """Live dist from hole centroid to hanger goal (meters)."""
    from dedo.utils.mesh_utils import get_mesh_data
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


def _save_demo(ep_obs_per_mode, ep_act, ep_rwd, ep_success):
    idx = next_demo_id()
    path = os.path.join(extra.demos_dir, f'demo_{idx:03d}.pkl')
    payload = {
        'obs': {m: np.asarray(ep_obs_per_mode[m], dtype=np.float32)
                for m in ALL_MODES},
        'acts': np.asarray(ep_act, dtype=np.float32),
        'reward': float(ep_rwd),
        'success': int(ep_success),
        'obs_modes': list(ALL_MODES),
        'recorded_in': extra.obs_mode,
        'len': len(ep_act),
    }
    with open(path, 'wb') as f:
        pickle.dump(payload, f)
    print(f'[record] saved {path}  len={len(ep_act)}  '
          f'rwd={ep_rwd:.2f}  success={ep_success}')


running = True
while running:
    obs = env.reset()
    ep_obs_per_mode = {m: [] for m in ALL_MODES}
    ep_act = []
    ep_rwd = 0.0
    ep_success = 0
    ks.consume_triggers()  # clear any pending triggers from last episode
    print(f'[record] new episode (already saved: {next_demo_id()})')
    dt = 1.0 / max(extra.fps, 1.0)

    while True:
        loop_start = time.time()

        triggers = ks.consume_triggers()
        if 'esc' in triggers:
            print('[record] ESC — quitting (current episode discarded).')
            running = False
            break
        if 'r' in triggers:
            print('[record] R — discarding episode, resetting.')
            break
        if 'enter' in triggers:
            if len(ep_act) == 0:
                print('[record] empty buffer — nothing to save.')
            else:
                _save_demo(ep_obs_per_mode, ep_act, ep_rwd, ep_success)
            break

        action = compute_action(ks)
        per_mode = _capture_all_obs()
        for m in ALL_MODES:
            ep_obs_per_mode[m].append(per_mode[m])
        ep_act.append(action)
        obs, rwd, done, info = env.step(action)
        ep_rwd += float(rwd)
        if 'is_success' in info:
            ep_success = max(ep_success, int(info['is_success']))

        # Live HUD: hole→goal distance vs. live success threshold.
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
                print('[record] tip: cloth was close to threshold but the '
                      '500 post-done settling steps may have nudged it off. '
                      'Try stopping all motion (release keys) for ~1s before '
                      'pressing ENTER, so anchors quiesce before settling.')
            print('  ENTER=save  R=discard  ESC=quit')
            while True:
                triggers = ks.consume_triggers()
                if 'enter' in triggers:
                    _save_demo(ep_obs_per_mode, ep_act, ep_rwd, ep_success)
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
