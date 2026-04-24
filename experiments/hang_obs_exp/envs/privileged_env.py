"""
PrivilegedObsWrapper — replaces the camera image with ground-truth cloth geometry.

Three observation modes (set via obs_mode arg):

  'hole_centroid'  (18-dim)  [DEFAULT]
      gripper_pos_vel (12) + hole_centroid_xyz (3) + hanger_goal_xyz (3)
      → minimal oracle: tells the policy exactly where the hole IS and where
        the hanger IS without needing a camera at all.

  'hole_vertices'  (12 + max_hole_verts*3 dim)
      gripper (12) + all hole-boundary vertex positions (padded to max_hole_verts)
      → richer geometry: policy can see hole shape, not just centroid.

  'full_mesh'      (12 + num_verts*3 dim)
      gripper (12) + all cloth vertex positions (~210-220 vertices × 3)
      → complete privileged state: full deformable object geometry.

Usage:
    env = gym.make('HangProcCloth-v1', args=args)
    env = PrivilegedObsWrapper(env, obs_mode='hole_centroid')

The wrapper:
  - forces cam_resolution=0 so no render is ever called (faster)
  - re-reads hole vertices from args after every reset() (they change each episode
    because HangProcCloth generates a new cloth procedurally)
  - normalises all positions into ~[-1,1] using the workspace box size
"""

import numpy as np
import gym
from gym import spaces

from dedo.utils.mesh_utils import get_mesh_data

_WBOX = 20.0
_MAX_HOLE_VERTS = 40


class PrivilegedObsWrapper(gym.ObservationWrapper):
    """
    Wraps a DeformEnv to replace the pixel/low-dim observation with
    privileged ground-truth cloth geometry.
    """

    MODES = ('hole_centroid', 'hole_vertices', 'full_mesh')

    def __init__(self, env, obs_mode: str = 'hole_centroid'):
        assert obs_mode in self.MODES, f'obs_mode must be one of {self.MODES}'
        env.args.cam_resolution = 0
        super().__init__(env)

        self.obs_mode = obs_mode
        self._hole_vertex_indices = []
        self._goal_pos = None

        grip_dim = 12

        if obs_mode == 'hole_centroid':
            obs_dim = grip_dim + 3 + 3
        elif obs_mode == 'hole_vertices':
            obs_dim = grip_dim + _MAX_HOLE_VERTS * 3
        elif obs_mode == 'full_mesh':
            obs_dim = grip_dim + 250 * 3

        self.observation_space = spaces.Box(
            low=-np.ones(obs_dim, dtype=np.float32) * 2.0,
            high=np.ones(obs_dim, dtype=np.float32) * 2.0,
            dtype=np.float32,
        )
        self._obs_dim = obs_dim

    def _get_hole_indices(self):
        if hasattr(self.env.args, 'deform_true_loop_vertices'):
            loops = self.env.args.deform_true_loop_vertices
            return [idx for loop in loops for idx in loop]
        return []

    def _build_obs(self):
        grip = np.array(self.env.get_grip_obs(), dtype=np.float32)
        grip = np.clip(grip / _WBOX, -2.0, 2.0)

        _, verts = get_mesh_data(self.env.sim, self.env.deform_id)
        verts = np.array(verts, dtype=np.float32)

        if self.obs_mode == 'hole_centroid':
            if len(self._hole_vertex_indices) > 0:
                hole_verts = verts[self._hole_vertex_indices]
                hole_verts = hole_verts[~np.isnan(hole_verts).any(axis=1)]
                centroid = hole_verts.mean(axis=0) if len(hole_verts) > 0 \
                           else np.zeros(3, dtype=np.float32)
            else:
                centroid = np.zeros(3, dtype=np.float32)
            goal = np.array(self.env.goal_pos[0], dtype=np.float32)
            obs = np.concatenate([
                grip,
                centroid / _WBOX,
                goal / _WBOX,
            ]).astype(np.float32)

        elif self.obs_mode == 'hole_vertices':
            if len(self._hole_vertex_indices) > 0:
                hole_verts = verts[self._hole_vertex_indices]
                hole_verts = hole_verts[~np.isnan(hole_verts).any(axis=1)]
            else:
                hole_verts = np.zeros((0, 3), dtype=np.float32)
            padded = np.zeros((_MAX_HOLE_VERTS, 3), dtype=np.float32)
            n = min(len(hole_verts), _MAX_HOLE_VERTS)
            if n > 0:
                padded[:n] = hole_verts[:n] / _WBOX
            obs = np.concatenate([grip, padded.reshape(-1)]).astype(np.float32)

        elif self.obs_mode == 'full_mesh':
            flat_verts = (verts / _WBOX).reshape(-1)
            full = np.zeros(250 * 3, dtype=np.float32)
            n = min(len(flat_verts), 250 * 3)
            full[:n] = flat_verts[:n]
            obs = np.concatenate([grip, full]).astype(np.float32)

        obs = np.clip(obs, self.observation_space.low, self.observation_space.high)
        return obs

    def reset(self):
        self.env.reset()
        self._hole_vertex_indices = self._get_hole_indices()
        self._goal_pos = self.env.goal_pos.copy()
        return self._build_obs()

    def observation(self, obs):
        return self._build_obs()

    def step(self, action):
        _, reward, done, info = self.env.step(action)
        obs = self._build_obs()
        return obs, reward, done, info
