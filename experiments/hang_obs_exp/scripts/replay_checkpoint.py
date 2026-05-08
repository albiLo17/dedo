"""
replay_checkpoint.py — load a train_privileged.py checkpoint and visualize
one or more rollout episodes:

  • mp4 of the rollout (with the SUCCESS/FAIL badge + settle frames used
    in training-time eval videos)
  • PNG with two subplots — per-step reward, cumulative reward — over
    the episode timeline. Each subplot shows one line per reward
    component (env base + each shaping term) plus a thicker dashed
    'total' line. Larger scatter dots emphasize the terminal step.

In --live mode, the rollout runs in an interactive matplotlib window
with these key controls:
    SPACE / p      pause or resume
    →   / n        single-step forward (works while paused)
    q              skip the rest of the current episode

Usage (from repo root):

  python experiments/hang_obs_exp/scripts/replay_checkpoint.py \
      --run_dir logs/hang_obs_exp/hole_centroid/PPO_<...>

  # Specific intermediate checkpoint (file: agent_step{N:08d}.zip):
  python experiments/hang_obs_exp/scripts/replay_checkpoint.py \
      --run_dir <...> --step 1500000

  # Multiple episodes:
  python experiments/hang_obs_exp/scripts/replay_checkpoint.py \
      --run_dir <...> --n_episodes 5

Outputs default to <run_dir>/replays/; override with --out_dir.

Auto-resolution of wrapper-side config:
  • obs_mode + shaping coefs (success_metric, success_factor,
    success_bonus, fail_penalty, vel_penalty, pre_settle_coef,
    action_penalty) are auto-resolved per field with priority:
        explicit CLI arg
      > <run_dir>/extra_args.pkl  (saved by current train_privileged.py)
      > eval_*.mp4 filename in <run_dir>  (most coefs are encoded there)
      > defaults matching train_privileged.py
  • The resolved values + their source are printed at startup so you can
    verify the policy is being replayed against the same shaping it was
    trained with.
  • dedo args (env name, max_episode_len, …) come from <run_dir>/args.pkl.

Wrapper-side coefs only affect the *plotted* reward decomposition — the
policy's actions are fixed by the loaded weights regardless.

VecNormalize is loaded from the run, with reward normalization disabled
at replay so the plotted rewards are the raw env rewards (incl. shaping).
Obs normalization stays on so the policy sees the same inputs as training.
"""
import sys, os, re, argparse, pickle
from copy import deepcopy
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT))

import numpy as np
import gym
import cv2
import matplotlib
# Use Agg when --live isn't passed so the script runs headless cleanly.
# Checking sys.argv before argparse is a small hack, but matplotlib.use()
# must run before pyplot is imported — and we want one consistent backend
# for the whole process.
if '--live' not in sys.argv:
    matplotlib.use('Agg')
import matplotlib.pyplot as plt

from stable_baselines3 import PPO
from stable_baselines3.common.env_util import DummyVecEnv
from stable_baselines3.common.vec_env import VecNormalize

import dedo  # registers gym envs
from dedo.envs.deform_env import DeformEnv

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _helpers import RetryResetEnv  # noqa: E402
from _video_callback import _draw_status_badge  # noqa: E402

from experiments.hang_obs_exp.envs.privileged_env import PrivilegedObsWrapper


_OBS_MODE_DIRS = ('hole_centroid', 'hole_vertices', 'full_mesh', 'enriched')

# Default wrapper-side shaping config — must match train_privileged.py's
# argparse defaults so a fresh checkpoint without extra_args.pkl plays
# back identically to one trained today.
_WRAPPER_DEFAULTS = dict(
    obs_mode=None,                 # inferred from <run_dir>'s parent
    success_metric='distance',
    success_factor=1.2,
    success_bonus=200.0,
    fail_penalty=0.0,
    vel_penalty=0.0,
    pre_settle_coef=0.0,
    action_penalty=0.0,
)

# Regex for parsing the wrapper config back out of an eval video
# filename. train_privileged.py builds:
#   eval_<obs_mode>_sm-<sm>_sf<sf>[_sb<sb>][_fp<fp>][_vp<vp>][_psc<psc>]
#       [_ap<ap>]_seed<seed>_step<NNNNNNNN>.mp4
# Optional segments are absent when their coef is 0/false-y.
_NUM = r'[-\d.eE+]+'
_VIDEO_NAME_RE = re.compile(
    r'^eval_(?P<obs_mode>[a-z_]+?)_sm-(?P<sm>[a-z]+)'
    rf'_sf(?P<sf>{_NUM})'
    rf'(?:_sb(?P<sb>{_NUM}))?'
    rf'(?:_fp(?P<fp>{_NUM}))?'
    rf'(?:_vp(?P<vp>{_NUM}))?'
    rf'(?:_psc(?P<psc>{_NUM}))?'
    rf'(?:_ap(?P<ap>{_NUM}))?'
    r'_seed(?P<seed>\d+)_step(?P<step>\d+)\.mp4$')

# Reward decomposition. Each entry is (info_key, label, sign, color):
#   contribution[t] = sign * info[t].get(info_key, 0.0)
# and sum-of-contributions[t] should equal env.step's returned reward.
# `base_reward` is the dedo env reward before wrapper-side shaping; the
# others are what PrivilegedObsWrapper subtracts/adds on top.
_REWARD_COMPONENTS = (
    ('base_reward',         'env (base)',     +1, 'tab:blue'),
    ('action_penalty',      'action_pen',     -1, 'tab:purple'),
    ('vel_penalty',         'vel_pen',        -1, 'tab:green'),
    ('pre_settle_penalty',  'pre_settle_pen', -1, 'tab:red'),
    ('shaping_added',       'shaping',        +1, 'tab:olive'),
)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--run_dir', type=str, required=True,
                   help='Directory created by train_privileged.py — must '
                        'contain agent.zip, vec_normalize.pkl, args.pkl.')
    p.add_argument('--step', type=str, default='final',
                   help='Which checkpoint to load. One of: an integer '
                        'step number (loads agent_step{step:08d}.zip + '
                        'matching VecNormalize); "final" (default; '
                        'agent.zip — the very-last save at end of '
                        'training); "latest" (the largest available '
                        'agent_step* — most recent intermediate save '
                        'before training ended); "middle" (median '
                        'agent_step* — useful for spot-checking '
                        'mid-training behavior); "earliest" (smallest '
                        'agent_step*); "list" (print available steps '
                        'and exit).')
    p.add_argument('--n_episodes', type=int, default=1)
    p.add_argument('--out_dir', type=str, default=None,
                   help='Output directory. Default: <run_dir>/replays')
    p.add_argument('--obs_mode', type=str, default=None,
                   choices=list(_OBS_MODE_DIRS),
                   help='Default: auto-resolved from <run_dir>/extra_args.pkl, '
                        'falling back to <run_dir> parent name.')
    # Wrapper-side shaping params. Default None means "auto-resolve" —
    # see _resolve_wrapper_args(): try <run_dir>/extra_args.pkl, then
    # parse from any eval_*.mp4 filename in the run dir, else use the
    # train_privileged.py defaults. Pass any flag explicitly to override.
    p.add_argument('--success_metric', type=str, default=None,
                   choices=['distance', 'threading'])
    p.add_argument('--success_factor', type=float, default=None)
    p.add_argument('--success_bonus', type=float, default=None)
    p.add_argument('--fail_penalty', type=float, default=None)
    p.add_argument('--vel_penalty', type=float, default=None)
    p.add_argument('--pre_settle_coef', type=float, default=None)
    p.add_argument('--action_penalty', type=float, default=None)
    p.add_argument('--seed', type=int, default=42)
    # Replays default to deterministic (mean of the policy's action
    # distribution). PPO's log_std rarely shrinks all the way during
    # training, so stochastic rollouts can look ~random even from a
    # decent policy. For "show me what this policy does", deterministic
    # is what you almost always want.
    p.add_argument('--stochastic', action='store_true',
                   help='Sample actions from the policy distribution '
                        'instead of taking the mean. Default is '
                        'deterministic (mirrors SB3 '
                        'evaluate_policy(deterministic=True)). Use this '
                        'to see how much of the rollout variance is '
                        'from the policy itself vs cloth physics.')
    p.add_argument('--cam_resolution', type=int, default=400)
    p.add_argument('--fps_video', type=int, default=24)
    p.add_argument('--no_video', action='store_true')
    p.add_argument('--no_plot', action='store_true')
    p.add_argument('--cpu', action='store_true')
    p.add_argument('--live', action='store_true',
                   help='Pop up a window with the rollout video on the '
                        'left and per-step / cumulative reward plots on '
                        'the right, updating live as the episode unfolds. '
                        'Window stays up after the last episode until you '
                        'close it; earlier episodes pause briefly then '
                        'advance automatically.')
    p.add_argument('--live_linger', type=float, default=1.5,
                   help='Seconds to keep the final frame visible between '
                        'live episodes (last episode blocks regardless).')
    return p.parse_args()


_AGENT_STEP_RE = re.compile(r'^agent_step(\d{8})\.zip$')


def _list_intermediate_steps(run_dir):
    """Sorted list of step numbers for which both agent_step{N}.zip and
    vec_normalize_step{N}.pkl exist in run_dir."""
    steps = []
    for path in run_dir.glob('agent_step*.zip'):
        m = _AGENT_STEP_RE.match(path.name)
        if m is None:
            continue
        n = int(m.group(1))
        if (run_dir / f'vec_normalize_step{n:08d}.pkl').is_file():
            steps.append(n)
    return sorted(steps)


def _resolve_checkpoint(run_dir, step_arg):
    """Resolve --step's keyword/number into (agent.zip path, VN.pkl path).
    Handles 'final' / 'latest' / 'middle' / 'earliest' / 'list' / int."""
    intermediates = _list_intermediate_steps(run_dir)

    if step_arg == 'list':
        print(f'[replay] checkpoints in {run_dir}:')
        if (run_dir / 'agent.zip').is_file():
            print('[replay]   final  → agent.zip')
        if intermediates:
            print('[replay]   intermediates ({} total): {}'.format(
                len(intermediates),
                ', '.join(f'{n:,}' for n in intermediates)))
            print(f'[replay]     earliest = {intermediates[0]:,}')
            print(f'[replay]     middle   = '
                  f'{intermediates[len(intermediates) // 2]:,}')
            print(f'[replay]     latest   = {intermediates[-1]:,}')
        else:
            print('[replay]   no intermediate agent_step*.zip files found')
        raise SystemExit(0)

    if step_arg == 'final':
        return run_dir / 'agent.zip', run_dir / 'vec_normalize.pkl'

    keyword_to_n = {}
    if intermediates:
        keyword_to_n['earliest'] = intermediates[0]
        keyword_to_n['middle'] = intermediates[len(intermediates) // 2]
        keyword_to_n['latest'] = intermediates[-1]

    if step_arg in keyword_to_n:
        n = keyword_to_n[step_arg]
        print(f'[replay] --step {step_arg} → step {n:,} '
              f'(of {len(intermediates)} intermediate checkpoints)')
    elif step_arg in ('earliest', 'middle', 'latest'):
        raise SystemExit(
            f'--step {step_arg}: no intermediate agent_step*.zip files '
            f'found in {run_dir}. Use --step list to see what\'s available.')
    else:
        try:
            n = int(step_arg)
        except ValueError:
            raise SystemExit(
                f'--step {step_arg!r}: not an int or one of '
                f'final/latest/middle/earliest/list')

    return (run_dir / f'agent_step{n:08d}.zip',
            run_dir / f'vec_normalize_step{n:08d}.pkl')


def _infer_obs_mode(run_dir):
    parent = Path(run_dir).parent.name
    return parent if parent in _OBS_MODE_DIRS else None


def _recover_from_extra_args_pkl(run_dir):
    """Read wrapper-side config from <run_dir>/extra_args.pkl. Newer
    train_privileged.py runs persist it; older ones won't. Returns a
    dict of recovered fields, or None if the file is missing."""
    p = run_dir / 'extra_args.pkl'
    if not p.is_file():
        return None
    with open(p, 'rb') as f:
        ea = pickle.load(f)
    out = {}
    for k in _WRAPPER_DEFAULTS:
        v = getattr(ea, k, None)
        if v is not None:
            out[k] = v
    return out


def _recover_from_video_filename(run_dir):
    """Parse the wrapper config out of any eval_*.mp4 in run_dir
    (train_privileged.py encodes most coefs in the filename). Returns a
    dict of recovered fields, or None if no parseable file exists."""
    for path in sorted(run_dir.glob('eval_*_step*.mp4')):
        m = _VIDEO_NAME_RE.match(path.name)
        if m is None:
            continue
        d = m.groupdict()
        return {
            'obs_mode':         d['obs_mode'],
            'success_metric':   d['sm'],
            'success_factor':   float(d['sf']),
            'success_bonus':    float(d['sb']) if d['sb'] else 0.0,
            'fail_penalty':     float(d['fp']) if d['fp'] else 0.0,
            'vel_penalty':      float(d['vp']) if d['vp'] else 0.0,
            'pre_settle_coef':  float(d['psc']) if d['psc'] else 0.0,
            'action_penalty':   float(d['ap']) if d['ap'] else 0.0,
        }
    return None


def _resolve_wrapper_args(args, run_dir):
    """Fill in any wrapper-side args that the user didn't pass explicitly.

    Priority: CLI value (if not None) > extra_args.pkl > video filename
    > _WRAPPER_DEFAULTS. Logs the source for each resolved field so the
    replay output is honest about what coefs the policy was actually
    trained with vs which were assumed."""
    sources = {}  # field -> source label
    pkl = _recover_from_extra_args_pkl(run_dir)
    vid = _recover_from_video_filename(run_dir) if pkl is None else None
    inferred_obs = _infer_obs_mode(run_dir)

    for field, default in _WRAPPER_DEFAULTS.items():
        if getattr(args, field) is not None:
            sources[field] = 'CLI'
            continue
        if pkl is not None and field in pkl:
            setattr(args, field, pkl[field])
            sources[field] = 'extra_args.pkl'
            continue
        if vid is not None and field in vid:
            setattr(args, field, vid[field])
            sources[field] = 'video filename'
            continue
        if field == 'obs_mode' and inferred_obs is not None:
            setattr(args, field, inferred_obs)
            sources[field] = 'run_dir parent name'
            continue
        setattr(args, field, default)
        sources[field] = 'default'

    if args.obs_mode is None:
        raise SystemExit(
            'could not resolve --obs_mode (no extra_args.pkl, no eval '
            'video, run_dir parent not a known obs mode); pass it explicitly.')

    print('[replay] wrapper config (source → value):')
    for field in _WRAPPER_DEFAULTS:
        print(f'[replay]   {field:<18s} ({sources[field]:<18s}) '
              f'= {getattr(args, field)}')


class _LiveDisplay:
    """Single matplotlib window for live replay: rendered frame on the
    left, per-step reward (top right) + cumulative reward (bottom right).
    Each subplot shows one line per reward component (env base + each
    shaping term) plus a thicker 'total' line. All axes update
    incrementally as the rollout unfolds.

    During the post-policy gravity settle the plots freeze (the policy
    isn't acting; the terminal reward already landed) while the video
    keeps playing — that matches the saved mp4 and makes it visually
    obvious that the settle is gravity-only.
    """

    def __init__(self, cam_resolution, fps_video):
        plt.ion()
        self.fig = plt.figure(figsize=(13, 5))
        gs = self.fig.add_gridspec(2, 2, width_ratios=[1.0, 1.2])
        self.ax_img = self.fig.add_subplot(gs[:, 0])
        self.ax_step = self.fig.add_subplot(gs[0, 1])
        self.ax_cum = self.fig.add_subplot(gs[1, 1], sharex=self.ax_step)
        self.ax_img.set_xticks([])
        self.ax_img.set_yticks([])
        self.ax_step.axhline(0, color='gray', lw=0.5)
        self.ax_cum.axhline(0, color='gray', lw=0.5)
        self.ax_step.set_ylabel('per-step reward')
        self.ax_cum.set_ylabel('cumulative reward')
        self.ax_cum.set_xlabel('step')
        self._im = self.ax_img.imshow(np.zeros(
            (cam_resolution, cam_resolution, 3), dtype=np.uint8))
        # Total line first (dashed, low z-order) so component lines are
        # NOT occluded when they coincide with total. Markers on every
        # line so single-point values (pre_settle_pen, shaping_added —
        # which only fire at the terminal step) are visible.
        self._line_step_total, = self.ax_step.plot(
            [], [], lw=1.4, color='black', linestyle='--', label='total',
            zorder=1, marker='.', markersize=3)
        self._line_cum_total, = self.ax_cum.plot(
            [], [], lw=1.4, color='black', linestyle='--', label='total',
            zorder=1, marker='.', markersize=3)
        self._lines_step = {}
        self._lines_cum = {}
        for key, label, _, color in _REWARD_COMPONENTS:
            self._lines_step[key], = self.ax_step.plot(
                [], [], lw=1.0, color=color, label=label, alpha=0.9,
                zorder=2, marker='.', markersize=3)
            self._lines_cum[key], = self.ax_cum.plot(
                [], [], lw=1.0, color=color, label=label, alpha=0.9,
                zorder=2, marker='.', markersize=3)
        # Scatter dots that track the most recent point on each line.
        # Mid-rollout they act as a "current step" cursor; at episode
        # end they anchor the terminal-step contributions visibly so the
        # user can read off any pre_settle_pen / shaping_added that
        # fires only on the terminal step.
        self._scatters_step = {}
        self._scatters_cum = {}
        for key, _, _, color in _REWARD_COMPONENTS:
            self._scatters_step[key] = self.ax_step.scatter(
                [], [], s=70, facecolor=color, edgecolor='black',
                linewidth=0.6, zorder=4)
            self._scatters_cum[key] = self.ax_cum.scatter(
                [], [], s=70, facecolor=color, edgecolor='black',
                linewidth=0.6, zorder=4)
        self._scatter_step_total = self.ax_step.scatter(
            [], [], s=140, marker='*', facecolor='black',
            edgecolor='white', linewidth=0.6, zorder=5)
        self._scatter_cum_total = self.ax_cum.scatter(
            [], [], s=140, marker='*', facecolor='black',
            edgecolor='white', linewidth=0.6, zorder=5)
        # Inline value annotations next to each scatter dot. Critical
        # for components that only fire at the terminal step
        # (pre_settle_pen, shaping_added) — their lines visually
        # overlap with the gray axhline / each other for the rollout's
        # entire flat-zero portion, so the only signal of their
        # contribution is the labeled value at the terminal.
        self._anns_step = {}
        self._anns_cum = {}
        for key, _, _, color in _REWARD_COMPONENTS:
            self._anns_step[key] = self.ax_step.annotate(
                '', xy=(0, 0), xytext=(6, 0),
                textcoords='offset points', fontsize=7, color=color,
                va='center', zorder=6)
            self._anns_cum[key] = self.ax_cum.annotate(
                '', xy=(0, 0), xytext=(6, 0),
                textcoords='offset points', fontsize=7, color=color,
                va='center', zorder=6)
        self._ann_step_total = self.ax_step.annotate(
            '', xy=(0, 0), xytext=(6, -10),
            textcoords='offset points', fontsize=7, color='black',
            fontweight='bold', va='center', zorder=6)
        self._ann_cum_total = self.ax_cum.annotate(
            '', xy=(0, 0), xytext=(6, -10),
            textcoords='offset points', fontsize=7, color='black',
            fontweight='bold', va='center', zorder=6)

        self.ax_step.legend(fontsize=7, loc='best', ncol=2)
        self.ax_cum.legend(fontsize=7, loc='best', ncol=2)
        self._title = self.ax_img.set_title('episode 0  step 0')
        self._delay = 1.0 / max(fps_video, 1)
        self._ep_idx = 0
        self._n_episodes = 1
        # Interactive controls. Space (or 'p') toggles pause; while
        # paused, right-arrow (or 'n') single-steps. 'q' skips the rest
        # of the current episode. The title shows the current state +
        # a one-line key hint so the controls are discoverable.
        self._paused = False
        self._step_once = False
        self._skip_episode = False
        self.fig.canvas.mpl_connect('key_press_event', self._on_key)
        print('[replay] live controls: SPACE=pause/resume, →=step, '
              'q=skip episode')
        self.fig.tight_layout()
        plt.show(block=False)

    def _on_key(self, event):
        if event.key in (' ', 'p'):
            self._paused = not self._paused
        elif event.key in ('right', 'n'):
            self._step_once = True
        elif event.key == 'q':
            self._skip_episode = True
            self._paused = False  # let update() return promptly

    def reset_episode(self, ep_idx, n_episodes):
        self._ep_idx = ep_idx
        self._n_episodes = n_episodes
        # Clear interactive flags so a pause/skip from the previous
        # episode doesn't carry over.
        self._paused = False
        self._step_once = False
        self._skip_episode = False
        for ln in (*self._lines_step.values(), *self._lines_cum.values(),
                   self._line_step_total, self._line_cum_total):
            ln.set_data([], [])
        empty = np.empty((0, 2))
        for sc in (*self._scatters_step.values(),
                   *self._scatters_cum.values(),
                   self._scatter_step_total, self._scatter_cum_total):
            sc.set_offsets(empty)
        for ann in (*self._anns_step.values(), *self._anns_cum.values(),
                    self._ann_step_total, self._ann_cum_total):
            ann.set_text('')
        self.ax_step.relim(); self.ax_step.autoscale_view()
        self.ax_cum.relim(); self.ax_cum.autoscale_view()
        self._title.set_text(f'episode {ep_idx + 1}/{n_episodes}  step 0')
        self.fig.canvas.draw_idle()
        plt.pause(0.01)

    def update(self, frame, components, total, frozen_plot=False, label=''):
        """components: dict[key -> list of per-step contributions].
        total: list of per-step total rewards (length == len(components[*]))."""
        self._im.set_data(frame)
        n = len(total)
        if not frozen_plot and n > 0:
            steps = np.arange(1, n + 1)
            last_x = n
            for key, _, _, _ in _REWARD_COMPONENTS:
                vals = components[key]
                cum = np.cumsum(vals)
                self._lines_step[key].set_data(steps, vals)
                self._lines_cum[key].set_data(steps, cum)
                self._scatters_step[key].set_offsets(
                    [[last_x, vals[-1]]])
                self._scatters_cum[key].set_offsets(
                    [[last_x, cum[-1]]])
                self._anns_step[key].xy = (last_x, vals[-1])
                self._anns_step[key].set_text(f'{vals[-1]:+.1f}')
                self._anns_cum[key].xy = (last_x, cum[-1])
                self._anns_cum[key].set_text(f'{cum[-1]:+.1f}')
            self._line_step_total.set_data(steps, total)
            cum_total = np.cumsum(total)
            self._line_cum_total.set_data(steps, cum_total)
            self._scatter_step_total.set_offsets([[last_x, total[-1]]])
            self._scatter_cum_total.set_offsets([[last_x, cum_total[-1]]])
            self._ann_step_total.xy = (last_x, total[-1])
            self._ann_step_total.set_text(f'{total[-1]:+.1f}')
            self._ann_cum_total.xy = (last_x, cum_total[-1])
            self._ann_cum_total.set_text(f'{cum_total[-1]:+.1f}')
            self.ax_step.relim(); self.ax_step.autoscale_view()
            self.ax_cum.relim(); self.ax_cum.autoscale_view()
        suffix = f'  ({label})' if label else ''
        paused_tag = '  [PAUSED — SPACE resumes, → steps, q skips]' \
            if self._paused else ''
        self._title.set_text(
            f'episode {self._ep_idx + 1}/{self._n_episodes}  '
            f'step {n}{suffix}{paused_tag}')
        # Pause-aware delay. If paused at entry, sit in a tight plt.pause
        # loop until the user toggles _paused off, single-steps via right-
        # arrow, or skips the episode via 'q'. Otherwise just honor the
        # configured fps. plt.pause keeps the GUI event loop responsive
        # so key callbacks still fire.
        if self._paused:
            while self._paused and not self._step_once \
                    and not self._skip_episode:
                plt.pause(0.05)
            self._step_once = False  # consumed; next call pauses again
        else:
            plt.pause(self._delay)

    def linger(self, seconds):
        if seconds > 0:
            plt.pause(seconds)

    def show_blocking(self):
        plt.ioff()
        plt.show(block=True)
        plt.ion()  # restore for any later episodes

    def close(self):
        plt.ioff()
        plt.close(self.fig)


def _find_deform_env(env):
    seen = set()
    cur = env
    while id(cur) not in seen:
        seen.add(id(cur))
        if isinstance(cur, DeformEnv):
            return cur
        if hasattr(cur, 'env'):
            cur = cur.env
        else:
            return None
    return None


def _build_env(args, dedo_args, obs_mode):
    eval_args = deepcopy(dedo_args)
    eval_args.viz = False
    eval_args.debug = False
    eval_args.cam_resolution = 0
    # Lock the front-angled camera used by train_privileged's eval env so
    # rendered frames match training-time videos.
    eval_args.cam_viewmat = [9.0, -25.0, 45.0, 0.0, 0.5, 6.5]

    def _make():
        env = gym.make(eval_args.env, args=eval_args)
        env = RetryResetEnv(env)
        env = PrivilegedObsWrapper(
            env, obs_mode=obs_mode,
            success_metric=args.success_metric,
            success_factor=args.success_factor,
            success_bonus=args.success_bonus,
            fail_penalty=args.fail_penalty,
            vel_penalty=args.vel_penalty,
            pre_settle_coef=args.pre_settle_coef,
            action_penalty=args.action_penalty)
        return env

    return DummyVecEnv([_make])


def _run_episode(ep_idx, args, vec_env, agent, deform, out_dir, ckpt_stem,
                 live=None):
    if live is not None:
        live.reset_episode(ep_idx, args.n_episodes)

    obs = vec_env.reset()

    # Settle frames feed both the saved mp4 and the live window, so capture
    # them whenever either output path is active.
    record_video = (not args.no_video) or (live is not None)
    if record_video and deform is not None:
        deform._record_settle_frames = True
        deform._settle_render_kwargs = dict(
            width=args.cam_resolution, height=args.cam_resolution)

    rewards, frames = [], []
    components = {key: [] for key, _, _, _ in _REWARD_COMPONENTS}
    done = False
    info = {}
    while not done:
        action, _ = agent.predict(obs, deterministic=not args.stochastic)
        obs, rwds, dones, infos = vec_env.step(action)
        info = infos[0]
        done = bool(dones[0])
        rewards.append(float(rwds[0]))
        for key, _, sign, _ in _REWARD_COMPONENTS:
            components[key].append(sign * float(info.get(key, 0.0)))

        if record_video and deform is not None:
            settle = info.get('settle_frames')
            if settle:
                # Terminal step: replace the post-settle one-shot render
                # with the chronological settle progression. During settle
                # the policy isn't acting, so we freeze the reward plots
                # (the terminal reward — including success_bonus / settle
                # multiplier — already landed in `rewards`).
                n = len(settle)
                for j, f in enumerate(settle):
                    arr = np.ascontiguousarray(f).copy()
                    frames.append(arr)
                    if live is not None:
                        live.update(arr, components, rewards,
                                    frozen_plot=True,
                                    label=f'settle {j + 1}/{n}')
                        if live._skip_episode:
                            break
            else:
                img = deform.render(mode='rgb_array',
                                    width=args.cam_resolution,
                                    height=args.cam_resolution)
                arr = np.ascontiguousarray(img).copy()
                frames.append(arr)
                if live is not None:
                    live.update(arr, components, rewards)

        if live is not None and live._skip_episode:
            print(f'[replay]   ep {ep_idx + 1}: skipped at step '
                  f'{len(rewards)} via "q"')
            break

    if record_video and deform is not None:
        deform._record_settle_frames = False

    success = bool(info.get('is_success', False))
    threading = info.get('is_threaded')
    distance_success = info.get('is_distance_success')
    total_rwd = float(np.sum(rewards))
    print(f'[replay] ep {ep_idx + 1}/{args.n_episodes}  '
          f'len={len(rewards)}  total_rwd={total_rwd:.2f}  '
          f'success={int(success)}  '
          f'threaded={int(threading) if threading is not None else "?"}  '
          f'distance_success={int(distance_success) if distance_success is not None else "?"}')
    # Per-component contribution breakdown — answers "is the total just
    # base_reward, or is shaping doing real work?" without having to
    # read off the chart. (coef 0) tags components whose coef is zero
    # for this run; their lines are flat at zero on the plots.
    coef_attrs = {
        'action_penalty': args.action_penalty,
        'vel_penalty': args.vel_penalty,
        'pre_settle_penalty': args.pre_settle_coef,
    }
    # cum_contrib = total over the episode; terminal_contrib = value at
    # the final step alone (often the same as cum for terminal-only
    # components like pre_settle_pen and shaping_added).
    print(f'[replay]   reward breakdown    cumulative   '
          f'terminal-step')
    for key, label, _, _ in _REWARD_COMPONENTS:
        contrib = float(np.sum(components[key]))
        term_v = float(components[key][-1])
        suffix = ('   (coef 0)'
                  if key in coef_attrs and coef_attrs[key] == 0.0 else '')
        print(f'[replay]     {label:<16s} {contrib:+10.2f}   '
              f'{term_v:+10.2f}{suffix}')
    print(f'[replay]     {"total":<16s} {total_rwd:+10.2f}   '
          f'{rewards[-1]:+10.2f}')

    if frames:
        # Stamp every buffered frame retroactively so playback shows the
        # outcome from frame 1 (matches HangVideoCallback).
        ep_dist = info.get('adaptive_dist')
        ep_thresh = info.get('adaptive_thresh')
        for f in frames:
            _draw_status_badge(f, success, dist=ep_dist, thresh=ep_thresh)

    # Refresh the live window so the user sees the final frame WITH the
    # SUCCESS/FAIL badge before the window blocks / lingers.
    if live is not None and frames:
        live.update(frames[-1], components, rewards, frozen_plot=True,
                    label='SUCCESS' if success else 'FAIL')
        if ep_idx == args.n_episodes - 1:
            print('[replay]   close the window to exit.')
            live.show_blocking()
        else:
            live.linger(args.live_linger)

    stem = (f'replay_{ckpt_stem}_seed{args.seed}_ep{ep_idx:02d}'
            f'_{"stoch" if args.stochastic else "det"}')

    if not args.no_video and frames:
        h, w = frames[0].shape[:2]
        out_mp4 = out_dir / f'{stem}.mp4'
        writer = cv2.VideoWriter(str(out_mp4),
                                 cv2.VideoWriter_fourcc(*'mp4v'),
                                 args.fps_video, (w, h))
        for f in frames:
            writer.write(np.ascontiguousarray(f[..., ::-1]))  # RGB→BGR
        writer.release()
        print(f'[replay]   wrote {out_mp4}')

    if not args.no_plot and rewards:
        out_png = out_dir / f'{stem}.png'
        steps = np.arange(1, len(rewards) + 1)
        terminal_x = len(rewards)
        fig, axes = plt.subplots(2, 1, figsize=(11, 7), sharex=True)
        # Per-step decomposition: thin colored line + dot markers per
        # component, dashed black for the env-returned total. Markers
        # are essential for components that only fire at the terminal
        # step (pre_settle_pen, shaping_added) — without them, a single
        # nonzero point on a 100+ step line is invisible.
        # Total drawn first with low z-order so component lines are NOT
        # occluded when they coincide with the total.
        axes[0].plot(steps, rewards, lw=1.4, color='black',
                     linestyle='--', label='total', zorder=1,
                     marker='.', markersize=3)
        for key, label, _, color in _REWARD_COMPONENTS:
            axes[0].plot(steps, components[key], lw=1.0, color=color,
                         alpha=0.9, label=label, zorder=2,
                         marker='.', markersize=3)
        # Emphasize the terminal step with bigger scatter dots — this is
        # where dedo applies the (max_episode_len - stepnum) multiplier
        # plus the post-settle last_rwd, AND where wrapper-side shaping
        # (pre_settle_pen, shaping_added) lands. Drawn over the line
        # markers with a black edge so each component's terminal value
        # is unambiguous.
        for key, label, _, color in _REWARD_COMPONENTS:
            term_v = components[key][-1]
            axes[0].scatter([terminal_x], [term_v], s=70,
                            facecolor=color, edgecolor='black',
                            linewidth=0.6, zorder=4)
            axes[0].annotate(f'{term_v:+.1f}',
                             xy=(terminal_x, term_v),
                             xytext=(6, 0), textcoords='offset points',
                             fontsize=7, color=color,
                             va='center', zorder=6)
        axes[0].scatter([terminal_x], [rewards[-1]], s=140, marker='*',
                        facecolor='black', edgecolor='white',
                        linewidth=0.6, zorder=5)
        axes[0].annotate(f'{rewards[-1]:+.1f}',
                         xy=(terminal_x, rewards[-1]),
                         xytext=(6, -10), textcoords='offset points',
                         fontsize=7, color='black', fontweight='bold',
                         va='center', zorder=6)
        axes[0].axhline(0, color='gray', lw=0.5)
        axes[0].axvline(terminal_x, color='red', lw=0.6, linestyle='--',
                        alpha=0.5, label='terminal', zorder=0)
        axes[0].set_ylabel('per-step reward')
        axes[0].legend(loc='best', fontsize=8, ncol=2)
        title = (f'{stem}\n'
                 f'len={len(rewards)}  total={total_rwd:.2f}  '
                 f'success={int(success)}')
        if threading is not None:
            title += f'  threaded={int(threading)}'
        if distance_success is not None:
            title += f'  d_succ={int(distance_success)}'
        axes[0].set_title(title)
        # Cumulative: same treatment. Where each line ends = that
        # component's total contribution to the episode return — the
        # terminal scatter dot on each line annotates that value
        # directly.
        cum_total = np.cumsum(rewards)
        axes[1].plot(steps, cum_total, lw=1.4, color='black',
                     linestyle='--', label='total', zorder=1,
                     marker='.', markersize=3)
        for key, label, _, color in _REWARD_COMPONENTS:
            axes[1].plot(steps, np.cumsum(components[key]), lw=1.0,
                         color=color, alpha=0.9, label=label, zorder=2,
                         marker='.', markersize=3)
        for key, label, _, color in _REWARD_COMPONENTS:
            cum_key = float(np.sum(components[key]))
            axes[1].scatter([terminal_x], [cum_key], s=70,
                            facecolor=color, edgecolor='black',
                            linewidth=0.6, zorder=4)
            # Inline numeric label so small contributions (e.g.
            # pre_settle_pen at -7.76 next to a -220 total) are
            # legible regardless of axis scale.
            axes[1].annotate(f'{cum_key:+.1f}',
                             xy=(terminal_x, cum_key),
                             xytext=(6, 0), textcoords='offset points',
                             fontsize=7, color=color,
                             va='center', zorder=6)
        axes[1].scatter([terminal_x], [cum_total[-1]], s=140, marker='*',
                        facecolor='black', edgecolor='white',
                        linewidth=0.6, zorder=5)
        axes[1].annotate(f'{cum_total[-1]:+.1f}',
                         xy=(terminal_x, cum_total[-1]),
                         xytext=(6, -10), textcoords='offset points',
                         fontsize=7, color='black', fontweight='bold',
                         va='center', zorder=6)
        axes[1].axhline(0, color='gray', lw=0.5)
        axes[1].axvline(terminal_x, color='red', lw=0.6, linestyle='--',
                        alpha=0.5, zorder=0)
        axes[1].set_ylabel('cumulative reward')
        axes[1].set_xlabel('step')
        axes[1].legend(loc='best', fontsize=8, ncol=2)
        fig.tight_layout()
        fig.savefig(out_png, dpi=120)
        plt.close(fig)
        print(f'[replay]   wrote {out_png}')


def main():
    args = parse_args()

    run_dir = Path(args.run_dir).resolve()
    if not run_dir.is_dir():
        raise SystemExit(f'run_dir not found: {run_dir}')

    ckpt_path, vn_path = _resolve_checkpoint(run_dir, args.step)
    for p in (ckpt_path, vn_path):
        if not p.is_file():
            raise SystemExit(f'missing file: {p}')

    args_pkl = run_dir / 'args.pkl'
    if not args_pkl.is_file():
        raise SystemExit(f'missing args.pkl in {run_dir}')
    with open(args_pkl, 'rb') as f:
        dedo_args = pickle.load(f)

    out_dir = (Path(args.out_dir) if args.out_dir
               else (run_dir / 'replays'))
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f'[replay] run_dir = {run_dir}')
    print(f'[replay] ckpt    = {ckpt_path.name}')
    print(f'[replay] out_dir = {out_dir}')
    print(f'[replay] actions = '
          f'{"stochastic (sampled)" if args.stochastic else "deterministic (mean)"}')
    _resolve_wrapper_args(args, run_dir)
    obs_mode = args.obs_mode

    vec_env = _build_env(args, dedo_args, obs_mode)
    vec_env.seed(args.seed)
    vec_env = VecNormalize.load(str(vn_path), vec_env)
    # Replay-mode VecNormalize: don't update obs_rms, don't normalize the
    # rewards we plot. Obs normalization stays so the policy sees the
    # same input distribution it trained on.
    vec_env.training = False
    vec_env.norm_reward = False

    device = 'cpu' if args.cpu else 'auto'
    agent = PPO.load(str(ckpt_path), env=vec_env, device=device)
    print(f'[replay] loaded PPO ({agent.num_timesteps:,} timesteps)')

    deform = _find_deform_env(vec_env.envs[0])
    if deform is None:
        if not args.no_video:
            print('[replay] warning: could not locate DeformEnv — disabling video')
            args.no_video = True
        if args.live:
            print('[replay] warning: could not locate DeformEnv — disabling live')
            args.live = False

    live = _LiveDisplay(args.cam_resolution, args.fps_video) if args.live else None
    try:
        for ep in range(args.n_episodes):
            _run_episode(ep, args, vec_env, agent, deform, out_dir,
                         ckpt_path.stem, live=live)
    finally:
        if live is not None:
            live.close()
        vec_env.close()
    print('Done.')


if __name__ == '__main__':
    main()
