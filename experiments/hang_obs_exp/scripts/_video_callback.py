"""
HangVideoCallback — replacement for dedo's CustomCallback with two fixes:

1. Eval mp4s are written to disk inside the run's logdir with a descriptive
   filename (`{basename}_step{N}.mp4`) AND logged to wandb with a caption
   that includes the wandb run name. The bare tensorboard `Video` path
   produces nondescript media file names that are hard to identify when
   downloading across runs.

2. Eval video includes the post-policy settle phase. dedo's env runs
   `make_final_steps` (cloth falls under gravity) inside the terminal
   `step()` call, so a vanilla `evaluate_policy(callback=...)` only ever
   sees the post-settle state on the last frame. Here we toggle a flag
   on the underlying DeformEnv that captures rendered frames during the
   settle, and we splice those into the video in chronological order.

Drop-in replacement for `CustomCallback` in train_privileged.py.
"""
import os
import pickle
from collections import deque

import cv2
import numpy as np
import torch

from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.logger import Video

from dedo.utils.train_utils import object_to_str
from dedo.envs.deform_env import DeformEnv


def _find_deform_env(env):
    """Walk down the gym.Wrapper chain to find the underlying DeformEnv."""
    cur = env
    seen = set()
    while id(cur) not in seen:
        seen.add(id(cur))
        if isinstance(cur, DeformEnv):
            return cur
        if hasattr(cur, 'env'):
            cur = cur.env
        else:
            break
    return None


def _draw_status_badge(frame, success, dist=None, thresh=None):
    """Mutate an HxWx3 uint8 RGB frame in place: thin colored border plus a
    'SUCCESS' / 'FAIL' badge in the top-left, optionally annotated with
    the final hole-centroid-to-goal distance and the adaptive success
    threshold (both in meters). Called retroactively after each episode
    terminates so all frames in the episode share the same outcome tag."""
    h, w = frame.shape[:2]
    color = (0, 200, 0) if success else (220, 0, 0)  # RGB
    label = 'SUCCESS' if success else 'FAIL'

    lines = [label]
    if dist is not None and thresh is not None:
        # e.g. 'd=0.42 t=0.51' — both meters, post-settle.
        lines.append(f'd={dist:.2f}m t={thresh:.2f}m')

    # Layout: pad + rows of ~18 px, char width ~9 px at scale=0.5.
    pad = 6
    line_h = 18
    box_h = pad + line_h * len(lines)
    char_w = 9
    box_w = pad + char_w * max(len(s) for s in lines)

    cv2.rectangle(frame, (0, 0), (w - 1, h - 1), color, 4)
    cv2.rectangle(frame, (8, 8), (8 + box_w, 8 + box_h), color, -1)
    for i, line in enumerate(lines):
        y = 8 + line_h * (i + 1) - 4
        cv2.putText(frame, line, (12, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)


class HangVideoCallback(BaseCallback):

    def __init__(self, eval_env, logdir, num_train_envs, args,
                 num_steps_between_save=10000, viz=False, debug=False,
                 video_basename='eval', render_size=300, video_fps=30,
                 n_eval_episodes=10):
        super().__init__(debug)
        self._eval_env = eval_env
        self._logdir = logdir
        self._n_eval_episodes = int(n_eval_episodes)
        self._num_train_envs = num_train_envs
        self._my_args = args
        self._num_steps_between_save = num_steps_between_save
        self._viz = viz
        self._debug = debug
        self._steps_since_save = num_steps_between_save  # save right away
        self._episode_count = 0
        self._success_window = deque(maxlen=100)
        # Parallel windows for the two underlying metrics — both are
        # populated unconditionally by PrivilegedObsWrapper at episode end,
        # regardless of which one was selected as `is_success`. Logging
        # both means a single training run shows on the wandb dashboard
        # whether (a) optimizing distance also drives threading up, and
        # (b) optimizing threading also drives the centroid into the
        # goal region — i.e. agreement vs disagreement of the metrics.
        self._threading_window = deque(maxlen=100)
        self._distance_success_window = deque(maxlen=100)
        # Rolling per-step shaping windows. ~2000 samples ≈ 500 steps × 4
        # envs, so the logged mean tracks the last few rollouts and
        # responds quickly when the policy changes.
        self._action_penalty_window = deque(maxlen=2000)
        self._action_cost_window = deque(maxlen=2000)
        self._vel_penalty_window = deque(maxlen=2000)
        self._cloth_speed_window = deque(maxlen=2000)
        self._eval_count = 0
        self._video_basename = video_basename
        self._render_size = render_size
        self._video_fps = video_fps
        self._deform = _find_deform_env(eval_env)

    def _on_training_start(self) -> None:
        self.logger.record('args', object_to_str(self._my_args))

    def _on_rollout_start(self) -> None:
        pass

    def _on_rollout_end(self) -> None:
        pass

    def _on_training_end(self) -> None:
        pass

    def _on_step(self) -> bool:
        # Track training-env successes (rolling window of 100).
        for info in self.locals.get('infos', []):
            if 'is_success' in info:
                self._episode_count += 1
                self._success_window.append(int(info['is_success']))
                self.logger.record(
                    'rollout/success_rate_100',
                    sum(self._success_window) / len(self._success_window))
                self.logger.record('rollout/episodes', self._episode_count)
            # Both underlying metrics, gated separately so we keep the
            # window populated even if one of them ever goes missing
            # (e.g. a hanger_id resolution failure suppresses is_threaded).
            if 'is_threaded' in info:
                self._threading_window.append(int(info['is_threaded']))
                self.logger.record(
                    'rollout/threading_rate_100',
                    sum(self._threading_window)
                    / len(self._threading_window))
            if 'is_distance_success' in info:
                self._distance_success_window.append(
                    int(info['is_distance_success']))
                self.logger.record(
                    'rollout/distance_success_rate_100',
                    sum(self._distance_success_window)
                    / len(self._distance_success_window))
            # Rolling per-step shaping stats. Recording every step is cheap
            # — SB3 only flushes to disk/wandb at log_interval.
            if 'action_penalty' in info:
                self._action_penalty_window.append(info['action_penalty'])
                self._action_cost_window.append(info['action_cost'])
            if 'vel_penalty' in info:
                self._vel_penalty_window.append(info['vel_penalty'])
                self._cloth_speed_window.append(info['cloth_mean_speed'])

        if self._action_penalty_window:
            self.logger.record(
                'shaping/action_penalty_mean',
                float(np.mean(self._action_penalty_window)))
            self.logger.record(
                'shaping/action_cost_mean',
                float(np.mean(self._action_cost_window)))
        if self._vel_penalty_window:
            self.logger.record(
                'shaping/vel_penalty_mean',
                float(np.mean(self._vel_penalty_window)))
            self.logger.record(
                'shaping/cloth_speed_mean',
                float(np.mean(self._cloth_speed_window)))

        self._steps_since_save += self._num_train_envs
        if self._steps_since_save < self._num_steps_between_save:
            return True

        # Save checkpoint. VecNormalize stats are saved alongside the
        # policy weights so resume_from works mid-training, not just at
        # the end of a completed run.
        if self._logdir is not None:
            self.model.save(os.path.join(self._logdir, 'agent'))
            vn = self.model.get_vec_normalize_env()
            if vn is not None:
                vn.save(os.path.join(self._logdir, 'vec_normalize.pkl'))
            # Versioned per-step checkpoints. Pairs 1:1 with the
            # eval_*_step{N}.mp4 filenames so a recorded video and the
            # policy that produced it can be matched. Local-only: these
            # files live in the run logdir and are never uploaded to
            # wandb (no wandb.save / wandb.Artifact call references them).
            step_tag = f'{self.num_timesteps:08d}'
            self.model.save(os.path.join(
                self._logdir, f'agent_step{step_tag}'))
            if vn is not None:
                vn.save(os.path.join(
                    self._logdir, f'vec_normalize_step{step_tag}.pkl'))
            pickle.dump(self._my_args,
                        open(os.path.join(self._logdir, 'args.pkl'), 'wb'),
                        protocol=pickle.HIGHEST_PROTOCOL)
        self._steps_since_save = 0
        self._eval_count += 1

        # Eval every 2 checkpoints, video every 4 (matches CustomCallback).
        if self._eval_count % 2 != 0:
            return True

        log_video = (not self._my_args.disable_logging_video
                     and self._eval_count % 4 == 0)
        screens = []
        eval_successes = []
        eval_threadings = []
        eval_distance_successes = []

        # Toggle settle-frame capture on the underlying DeformEnv only when
        # we're recording, so training/eval-without-video stays cheap.
        if log_video and self._deform is not None:
            self._deform._record_settle_frames = True
            self._deform._settle_render_kwargs = dict(
                width=self._render_size, height=self._render_size)

        ep_buffer = []  # frames for the current eval episode

        def grab_screens(_locals, _globals=None):
            info = _locals.get('info', {})
            if 'is_success' in info:
                eval_successes.append(int(info['is_success']))
            if 'is_threaded' in info:
                eval_threadings.append(int(info['is_threaded']))
            if 'is_distance_success' in info:
                eval_distance_successes.append(
                    int(info['is_distance_success']))
            if not log_video:
                return
            # Append this step's frames into the per-episode buffer (we
            # don't know success/fail until the terminal step, so we tag
            # all frames retroactively below).
            settle = info.get('settle_frames')
            if settle:
                # Terminal step: replace the post-settle one-shot render
                # with the chronological settle progression. Copy each
                # frame so the badge can be drawn in-place safely.
                ep_buffer.extend(np.ascontiguousarray(f).copy()
                                 for f in settle)
            else:
                screen = self._eval_env.render(
                    mode='rgb_array', width=self._render_size,
                    height=self._render_size)
                ep_buffer.append(np.ascontiguousarray(screen).copy())
            # Episode boundary: stamp every buffered frame with the result
            # and flush to the main screens list.
            done = bool(_locals.get('done', False))
            if 'is_success' in info:
                success = bool(info['is_success'])
            elif done:
                # Crash/early-termination without is_success — count as fail.
                success = False
            else:
                return
            # Adaptive metrics are only set when PrivilegedObsWrapper had a
            # success_factor configured; render them when present.
            ep_dist = info.get('adaptive_dist')
            ep_thresh = info.get('adaptive_thresh')
            for frame in ep_buffer:
                _draw_status_badge(frame, success,
                                   dist=ep_dist, thresh=ep_thresh)
            screens.extend(ep_buffer)
            ep_buffer.clear()

        try:
            evaluate_policy(
                self.model, self._eval_env, callback=grab_screens,
                n_eval_episodes=self._n_eval_episodes,
                deterministic=True)
        finally:
            if log_video and self._deform is not None:
                self._deform._record_settle_frames = False

        if eval_successes:
            self.logger.record(
                'eval/success_rate',
                sum(eval_successes) / len(eval_successes))
        if eval_threadings:
            self.logger.record(
                'eval/threading_rate',
                sum(eval_threadings) / len(eval_threadings))
        if eval_distance_successes:
            self.logger.record(
                'eval/distance_success_rate',
                sum(eval_distance_successes) / len(eval_distance_successes))

        if not screens:
            return True

        # Save mp4 to disk with a descriptive filename.
        h, w = screens[0].shape[:2]
        video_path = os.path.join(
            self._logdir,
            f'{self._video_basename}_step{self.num_timesteps:08d}.mp4')
        writer = cv2.VideoWriter(
            video_path, cv2.VideoWriter_fourcc(*'mp4v'),
            self._video_fps, (w, h))
        for frame in screens:
            writer.write(np.ascontiguousarray(frame[..., ::-1]))  # RGB->BGR
        writer.release()

        # Prefer logging directly to wandb so the media file inherits the
        # run name and our caption. Otherwise fall back to the SB3
        # tensorboard Video path used by CustomCallback.
        wb_run = None
        if self._my_args.use_wandb:
            try:
                import wandb
                wb_run = wandb.run
            except ImportError:
                wb_run = None
        if wb_run is not None:
            import wandb
            wandb.log({
                'eval/video': wandb.Video(
                    video_path, fps=self._video_fps,
                    caption=f'{wb_run.name} step {self.num_timesteps}'),
            }, step=self.num_timesteps)
        else:
            stacked = np.stack([f.transpose(2, 0, 1) for f in screens])
            self.logger.record(
                'trajectory/video',
                Video(torch.ByteTensor([stacked]), fps=self._video_fps),
                exclude=('stdout', 'log', 'json', 'csv'))

        return True
