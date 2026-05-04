"""
Utilities for RL training and eval.


Note: this code is for research i.e. quick experimentation; it has minimal
comments for now, but if we see further interest from the community -- we will
add further comments, unify the style, improve efficiency and add unittests.

@contactrika

"""
import os
import pickle
from collections import deque

import cv2
import torch

from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.logger import Video

from dedo.utils.train_utils import object_to_str


def play(env, num_episodes, rl_agent, debug=False, logdir=None,
         cam_resolution=None, filename="play"):
    vidwriter = None
    if logdir is not None:
        if not os.path.exists(logdir):
            os.mkdir(logdir)
        vidwriter = cv2.VideoWriter(
            os.path.join(logdir, f'{filename}.mp4'), cv2.VideoWriter_fourcc(*'mp4v'),
            24, (cam_resolution, cam_resolution))
    for epsd in range(num_episodes):
        episode_rwd = 0.0
        if debug:
            print('------------ Play episode ', epsd, '------------------')
        obs = env.reset()
        if vidwriter is not None:
            img = env.render(mode='rgb_array', width=cam_resolution,
                             height=cam_resolution)
            vidwriter.write(img[..., ::-1])
        step = 0
        while True:
            # rl_agent.predict() to get acts, not forcing deterministic.
            act, _states = rl_agent.predict(obs)
            next_obs, rwd, done, info = env.step(act)
            episode_rwd += rwd
            if vidwriter is not None:
                img = env.render(mode='rgb_array', width=cam_resolution,
                                 height=cam_resolution)
                vidwriter.write(img[..., ::-1])
            if done:
                if debug:
                    print(f'episode reward {episode_rwd:0.2f}')
                break
            obs = next_obs
            step += 1
        # input('Episode ended; press enter to go on')
    if vidwriter is not None:
        vidwriter.release()


class CustomCallback(BaseCallback):
    """
    A custom callback that runs eval and adds videos to Tensorboard.
    """

    def __init__(self, eval_env, logdir, num_train_envs, args,
                 num_steps_between_save=10000, viz=False, debug=False):
        super(CustomCallback, self).__init__(debug)
        # Those variables will be accessible in the callback
        # (they are defined in the base class)
        # The RL model
        # self.model = None  # type: BaseAlgorithm
        # An alias for self.model.get_env(), the environment used for training
        # self.training_env = None  # type: Union[gym.Env, VecEnv, None]
        # Number of time the callback was called
        # self.n_calls = 0  # type: int
        # self.num_timesteps = 0  # type: int
        # local and global variables
        # self.locals = None  # type: Dict[str, Any]
        # self.globals = None  # type: Dict[str, Any]
        # The logger object, used to report things in the terminal
        # self.logger = None  # stable_baselines3.common.logger
        # # Sometimes, for event callback, it is useful
        # # to have access to the parent object
        # self.parent = None  # type: Optional[BaseCallback]
        self._eval_env = eval_env
        self._logdir = logdir
        self._num_train_envs = num_train_envs
        self._my_args = args
        self._num_steps_between_save = num_steps_between_save
        self._viz = viz
        self._debug = debug
        self._steps_since_save = num_steps_between_save  # save right away
        self._episode_count = 0
        self._success_window = deque(maxlen=100)
        self._eval_count = 0

    def _on_training_start(self) -> None:
        """
        This method is called before the first rollout starts.
        """
        # Log args to tensorboard.
        self.logger.record('args', object_to_str(self._my_args))

    def _on_rollout_start(self) -> None:
        """
        A rollout is the collection of environment interaction
        using the current policy.
        This event is triggered before collecting new samples.
        """
        pass

    def _on_step(self) -> bool:
        """
        This method will be called by the model after each call to `env.step()`.

        For child callback (of an `EventCallback`), this will be called
        when the event is triggered.

        :return: (bool) If the callback returns False, training is aborted early.
        """
        for info in self.locals.get('infos', []):
            if 'is_success' in info:
                self._episode_count += 1
                self._success_window.append(int(info['is_success']))
                self.logger.record('rollout/success_rate_100',
                                   sum(self._success_window) / len(self._success_window))
                self.logger.record('rollout/episodes', self._episode_count)

        self._steps_since_save += self._num_train_envs
        if self._steps_since_save >= self._num_steps_between_save:
            # Save checkpoint.
            if self._logdir is not None:
                self.model.save(os.path.join(self._logdir, 'agent'))
                pickle.dump(self._my_args,
                            open(os.path.join(self._logdir, 'args.pkl'), 'wb'),
                            protocol=pickle.HIGHEST_PROTOCOL)
            self._steps_since_save = 0
            self._eval_count += 1
            # Eval every 2 checkpoints, video every 4.
            if self._eval_count % 2 == 0:
                log_video = (not self._my_args.disable_logging_video
                             and self._eval_count % 4 == 0)
                screens = []
                eval_successes = []

                def grab_screens(_locals, _globals=None):
                    if log_video:
                        screen = self._eval_env.render(
                            mode='rgb_array', width=300, height=300)
                        screens.append(screen.transpose(2, 0, 1))
                    info = _locals.get('info', {})
                    if 'is_success' in info:
                        eval_successes.append(int(info['is_success']))

                evaluate_policy(
                    self.model, self._eval_env, callback=grab_screens,
                    n_eval_episodes=5, deterministic=True)

                if eval_successes:
                    self.logger.record('eval/success_rate',
                                       sum(eval_successes) / len(eval_successes))

                if screens:
                    self.logger.record(
                        'trajectory/video',
                        Video(torch.ByteTensor([screens]), fps=50),
                        exclude=('stdout', 'log', 'json', 'csv'))

        return True

    def _on_rollout_end(self) -> None:
        """
        This event is triggered before updating the policy.
        """
        pass

    def _on_training_end(self) -> None:
        """
        This event is triggered before exiting the `learn()` method.
        """
        pass
