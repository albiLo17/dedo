"""
Reward / success diagnostics for hang_obs_exp training scripts.

Provides three small pieces, all importable from the experiment scripts
without touching anything outside of experiments/:

  * RewardDiagnosticsCallback — drains rwd_diag/* keys emitted by the
    obs wrappers (PrivilegedObsWrapper / PointCloudObsWrapper) at each
    terminal step and logs rolling-window means to TensorBoard. With
    `sync_tensorboard=True` in dedo.utils.train_utils.init_train, these
    automatically appear in wandb under the same metric names.

  * dump_run_config — writes a single config.json into the run logdir
    that captures every CLI arg, DEDO arg, and the precise reward
    definition (success_factor / success_bonus / fail_penalty /
    vel_penalty plus the relevant DeformEnv class constants). Pushes
    the same dict into wandb.config and prints a clean banner.

  * collect_final_eval_diagnostics / log_final_eval_diagnostics —
    helpers to aggregate rwd_diag/* values produced during the final
    evaluate_policy pass and log them under final_eval/* so the very
    last bar/scalar in TB and wandb is unambiguous.

Why the wrapper emits and the callback consumes:

  - Wrappers are the only place that knows the exact reward decomposition,
    so they are the natural emitters.
  - Callbacks are the only objects SB3 routinely calls during learn(),
    so they are the natural sinks.
  - Keeping them decoupled means swapping in new reward shaping (or new
    obs wrappers) only requires updating the wrapper; the callback and
    config dump stay generic.

Metric layout in TB / wandb. Two layers per metric:

  (1) Rolling-mean (default), one record per metric per `_on_step`. Path
      = the metric name without prefix:

    reward/episode_total       full reward seen by the agent per episode
    reward/base_sum            sum of all per-step base env rewards
    reward/vel_penalty_sum     cumulative magnitude subtracted by vel_penalty
    reward/terminal_base       base env reward at the terminal step alone
                               (dominant signal: includes the post-settle
                                FINAL_REWARD_MULT term)
    reward/terminal_shaping    +success_bonus or -fail_penalty applied at done
    reward/episode_length

    success/base_rate          dedo's built-in is_success
    success/adaptive_rate      wrapper's adaptive (hole-radius-relative)
    success/active_rate        whichever definition the agent was trained on
    success/disagree_rate      base != adaptive (only emitted when both run)

    task/adaptive_dist         ||hole_centroid - goal|| at terminal step
    task/adaptive_thresh       success_factor * hole_radius
    task/hole_radius           per-episode mean hole radius

  (2) Per-episode RAW (no averaging), one wandb.log call at episode end.
      These are essential for spotting individual successful episodes
      since the rolling-mean charts smear them across 100 eps:

    last_episode/success            0/1, the active success criterion
    last_episode/success_adaptive   0/1 adaptive (== success when sf set)
    last_episode/success_base       0/1 dedo's strict criterion
    last_episode/dist               final hole→goal distance in meters
    last_episode/episode_total      total reward including shaping
    last_episode/episode_length     step count of the just-ended episode
    train/episodes_total            running episode counter (good x-axis)
    train/cumulative_successes      monotone tally of successes so far
"""

from __future__ import annotations

import json
import os
import platform
from collections import defaultdict, deque
from datetime import datetime
from typing import Any, Dict, Iterable, Optional

import numpy as np
import torch

from stable_baselines3.common.callbacks import BaseCallback

# Imported to surface DEDO's reward constants in the run config so anyone
# reading config.json later knows the exact base-env behavior.
from dedo.envs.deform_env import DeformEnv


_RWD_DIAG_PREFIX = 'rwd_diag/'


# =========================================================================
# Callback: drains rwd_diag/* info keys into TensorBoard scalars.
# =========================================================================
class RewardDiagnosticsCallback(BaseCallback):
    """Logs rolling-window means of every rwd_diag/* metric from the env.

    Args:
        window: number of finished episodes to average over per metric.
        verbose: SB3 verbosity (0 silent, 1 prints).
    """

    def __init__(self, window: int = 100, verbose: int = 0):
        super().__init__(verbose)
        self._window = int(window)
        self._buffers: Dict[str, deque] = defaultdict(
            lambda: deque(maxlen=self._window))
        # Total finished episodes / successes seen across all parallel envs.
        # These power the per-episode raw wandb stream (last_episode/*,
        # train/episodes_total, train/cumulative_successes) so charts can
        # show individual successes instead of rolling-window averages.
        self._n_episodes = 0
        self._n_successes = 0

    def _on_step(self) -> bool:
        for info in self.locals.get('infos', []):
            episode_ended = False
            ep_metrics: Dict[str, float] = {}
            for key, value in info.items():
                if not key.startswith(_RWD_DIAG_PREFIX):
                    continue
                if value is None:
                    continue
                # bool -> int so int/float collapse cleanly.
                if isinstance(value, bool):
                    value = int(value)
                if isinstance(value, (int, float, np.floating, np.integer)):
                    metric = key[len(_RWD_DIAG_PREFIX):]
                    val = float(value)
                    self._buffers[metric].append(val)
                    ep_metrics[metric] = val
                    episode_ended = True
            if episode_ended:
                self._n_episodes += 1
                # success/active_rate is the criterion the agent actually
                # trained against; fall back to adaptive then base. Use
                # explicit None checks — `0 or adaptive` would wrongly ignore
                # a legitimate active=0 failure.
                ep_success = ep_metrics.get('success/active_rate')
                if ep_success is None:
                    ep_success = ep_metrics.get('success/adaptive_rate')
                if ep_success is None:
                    ep_success = ep_metrics.get('success/base_rate')
                if ep_success is None:
                    ep_success = 0.0
                if ep_success > 0.5:
                    self._n_successes += 1
                self._log_per_episode_to_wandb(ep_metrics, ep_success)

        # Record current rolling means. SB3 will dump them to TB at the
        # next logger.dump() (every n_steps for PPO, every step-with-log
        # for SAC). Cheap to call every step.
        for metric, buf in self._buffers.items():
            if buf:
                self.logger.record(metric, sum(buf) / len(buf))
        if self._n_episodes:
            self.logger.record('rwd_diag_meta/episodes_seen',
                               self._n_episodes)
            self.logger.record('rwd_diag_meta/window_size',
                               len(next(iter(self._buffers.values())))
                               if self._buffers else 0)

        # Mirror episode counters to TensorBoard on *every* env step (constant
        # between episode ends, jumps when a new episode completes). Why:
        # `train/episodes_total` was previously only sent through wandb.log at
        # episode-end timesteps; TB/W&B metrics like `train/critic_loss` arrive
        # on almost every step. Wandb's workspace "X-axis = train/episodes_total"
        # joins scalars by global step — if a step has no train/episodes_total
        # key, the chart shows "no data". Flooding TB with the current counters
        # every step makes episode-count X-axis work for all series.
        self.logger.record('train/episodes_total', float(self._n_episodes))
        self.logger.record('train/cumulative_successes',
                           float(self._n_successes))
        return True

    def _log_per_episode_to_wandb(self, ep_metrics: Dict[str, float],
                                  ep_success: float) -> None:
        """Emit raw (un-averaged) per-episode values to wandb so charts can
        show individual successful episodes as 0/1 spikes instead of
        100-ep rolling means. Runs only when wandb is the active logger.

        We use wandb.log directly (not self.logger.record) because SB3's
        logger overwrites repeated record() calls between dumps — if 4
        episodes finish in one rollout window and only 1 is a success,
        the 1 gets clobbered by the next 0. Logging directly to wandb
        with step=num_timesteps preserves every episode."""
        try:
            import wandb
        except ImportError:
            return
        if wandb.run is None:
            return

        log_dict: Dict[str, float] = {
            'last_episode/success': float(ep_success),
            'train/episodes_total': float(self._n_episodes),
            'train/cumulative_successes': float(self._n_successes),
        }
        # Mirror a few of the most useful per-episode raw values so users
        # can see individual-episode behavior on the same axes that the
        # rolling-mean charts already use.
        for src, dst in (
                ('success/adaptive_rate', 'last_episode/success_adaptive'),
                ('success/base_rate', 'last_episode/success_base'),
                ('task/adaptive_dist', 'last_episode/dist'),
                ('reward/episode_total', 'last_episode/episode_total'),
                ('reward/episode_length', 'last_episode/episode_length')):
            if src in ep_metrics:
                log_dict[dst] = float(ep_metrics[src])
        try:
            wandb.log(log_dict, step=int(self.num_timesteps))
        except Exception:
            # Swallow wandb errors so a logging glitch never kills training.
            pass


# =========================================================================
# Config dump: one JSON + wandb.config.update + printed banner.
# =========================================================================
def _to_jsonable(v: Any) -> Any:
    """Best-effort conversion for argparse Namespaces, numpy types, paths."""
    if isinstance(v, (str, bool, type(None))):
        return v
    if isinstance(v, (int, float)):
        return v
    if isinstance(v, np.integer):
        return int(v)
    if isinstance(v, np.floating):
        return float(v)
    if isinstance(v, np.ndarray):
        return v.tolist()
    if isinstance(v, (list, tuple)):
        return [_to_jsonable(x) for x in v]
    if isinstance(v, dict):
        return {str(k): _to_jsonable(x) for k, x in v.items()}
    return str(v)


def _namespace_to_dict(ns) -> Dict[str, Any]:
    if ns is None:
        return {}
    if hasattr(ns, '__dict__'):
        return {k: _to_jsonable(v) for k, v in vars(ns).items()}
    if isinstance(ns, dict):
        return {k: _to_jsonable(v) for k, v in ns.items()}
    return {}


def _reward_def_summary(extra_args) -> Dict[str, Any]:
    """The single source of truth for "what reward did this run optimize?".

    Captures both the wrapper-level shaping knobs and the relevant DEDO
    base-env constants so future readers don't have to chase the code.
    """
    sf = getattr(extra_args, 'success_factor', None)
    return {
        'uses_adaptive_success': sf is not None,
        'success_factor': sf,
        'success_bonus': getattr(extra_args, 'success_bonus', 0.0),
        'fail_penalty': getattr(extra_args, 'fail_penalty', 0.0),
        'vel_penalty': getattr(extra_args, 'vel_penalty', 0.0),
        'boundary_penalty': getattr(extra_args, 'boundary_penalty', 0.0),
        'z_overshoot_penalty': getattr(extra_args, 'z_overshoot_penalty', 0.0),
        'z_overshoot_slack': getattr(extra_args, 'z_overshoot_slack', 1.0),
        'obs_mode': getattr(extra_args, 'obs_mode', None),
        # DEDO-side constants (so we can spot if the repo changed them).
        'dedo_base_success_threshold': DeformEnv.SUCESS_REWARD_TRESHOLD,
        'dedo_final_reward_mult': DeformEnv.FINAL_REWARD_MULT,
        'dedo_workspace_box_size': DeformEnv.WORKSPACE_BOX_SIZE,
        'dedo_steps_after_done': DeformEnv.STEPS_AFTER_DONE,
        'dedo_max_act_vel': DeformEnv.MAX_ACT_VEL,
    }


def dump_run_config(extra_args, dedo_args, logdir: str,
                    use_wandb: bool = False,
                    extra_metadata: Optional[Dict[str, Any]] = None) -> str:
    """Persist a fully-self-describing run config to logdir/config.json.

    The same dict is pushed into wandb.config so the wandb run page's
    Config tab shows everything (already-set wandb config keys are not
    overwritten unless they have changed).

    Returns the path written.
    """
    os.makedirs(logdir, exist_ok=True)

    cfg: Dict[str, Any] = {
        'extra': _namespace_to_dict(extra_args),
        'dedo': _namespace_to_dict(dedo_args),
        'reward_def': _reward_def_summary(extra_args),
        'system': {
            'timestamp_iso': datetime.now().isoformat(timespec='seconds'),
            'python': platform.python_version(),
            'platform': platform.platform(),
            'torch': torch.__version__,
            'cuda_available': torch.cuda.is_available(),
            'cuda_device_count': torch.cuda.device_count(),
        },
    }
    if extra_metadata:
        cfg['metadata'] = _to_jsonable(extra_metadata)

    path = os.path.join(logdir, 'config.json')
    with open(path, 'w') as f:
        json.dump(cfg, f, indent=2, sort_keys=True, default=str)

    if use_wandb:
        try:
            import wandb
            if wandb.run is not None:
                # Flatten one level so wandb's Config UI shows hierarchical
                # keys like reward_def.success_factor.
                flat = _flatten(cfg)
                wandb.config.update(flat, allow_val_change=True)
        except ImportError:
            pass

    _print_config_banner(cfg, path)
    return path


def _flatten(d: Dict[str, Any], prefix: str = '') -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    for k, v in d.items():
        key = f'{prefix}.{k}' if prefix else str(k)
        if isinstance(v, dict):
            out.update(_flatten(v, key))
        else:
            out[key] = v
    return out


def _print_config_banner(cfg: Dict[str, Any], path: str) -> None:
    print('\n' + '=' * 72)
    print(f'  RUN CONFIG  (full dump: {path})')
    print('=' * 72)
    rd = cfg.get('reward_def', {})
    print('  reward_def:')
    for k in ('uses_adaptive_success', 'success_factor', 'success_bonus',
             'fail_penalty', 'vel_penalty', 'boundary_penalty',
             'z_overshoot_penalty', 'z_overshoot_slack', 'obs_mode',
             'dedo_base_success_threshold', 'dedo_final_reward_mult'):
        if k in rd:
            print(f'    {k:34s} = {rd[k]}')
    extra = cfg.get('extra', {})
    if extra:
        print('  extra (script-level):')
        for k in sorted(extra):
            print(f'    {k:34s} = {extra[k]}')
    print('=' * 72 + '\n')


# =========================================================================
# Final-eval helpers: aggregate rwd_diag/* over evaluate_policy episodes.
# =========================================================================
class _FinalEvalCollector:
    """Drop-in replacement for the per-episode callback used with
    stable_baselines3.common.evaluation.evaluate_policy. Collects
    is_success and rwd_diag/* values from each terminal info dict so
    the caller can summarize them after eval finishes."""

    def __init__(self) -> None:
        self.successes: list[int] = []
        self.metrics: Dict[str, list[float]] = defaultdict(list)

    def __call__(self, _locals, _globals=None):
        info = _locals.get('info', {})
        if 'is_success' in info:
            self.successes.append(int(info['is_success']))
        for key, value in info.items():
            if not key.startswith(_RWD_DIAG_PREFIX) or value is None:
                continue
            if isinstance(value, bool):
                value = int(value)
            if isinstance(value, (int, float, np.floating, np.integer)):
                metric = key[len(_RWD_DIAG_PREFIX):]
                self.metrics[metric].append(float(value))

    def summary(self) -> Dict[str, float]:
        out: Dict[str, float] = {}
        for k, vs in self.metrics.items():
            if vs:
                out[k] = float(np.mean(vs))
                out[f'{k}__std'] = float(np.std(vs))
        return out


def make_final_eval_collector() -> _FinalEvalCollector:
    """Convenience constructor used by the train scripts."""
    return _FinalEvalCollector()


def log_final_eval_metrics(collector: _FinalEvalCollector,
                           use_wandb: bool = False,
                           prefix: str = 'final_eval') -> Dict[str, float]:
    """Log everything the collector saw to wandb (and return the dict so
    the caller can also write to TB / stdout)."""
    summary = collector.summary()
    n = len(collector.successes)
    success_rate = (sum(collector.successes) / n) if n else float('nan')
    summary[f'success_rate'] = success_rate
    summary['n_episodes'] = n

    namespaced = {f'{prefix}/{k}': v for k, v in summary.items()}
    if use_wandb:
        try:
            import wandb
            if wandb.run is not None:
                wandb.log(namespaced)
        except ImportError:
            pass
    return namespaced
