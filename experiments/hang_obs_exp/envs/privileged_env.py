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


def build_privileged_obs(deform_env, obs_mode, hole_vertex_indices):
    """Compute the privileged obs for a given mode from the underlying
    DeformEnv state. Mode-independent inputs (gripper, mesh, goal) → mode-
    specific concatenation. Used by PrivilegedObsWrapper and by the demo
    recorder so a single recorded demo can be replayed in any obs mode."""
    grip = np.array(deform_env.get_grip_obs(), dtype=np.float32)
    grip = np.clip(grip / _WBOX, -2.0, 2.0)

    _, verts = get_mesh_data(deform_env.sim, deform_env.deform_id)
    verts = np.array(verts, dtype=np.float32)

    if obs_mode == 'hole_centroid':
        if len(hole_vertex_indices) > 0:
            hole_verts = verts[hole_vertex_indices]
            hole_verts = hole_verts[~np.isnan(hole_verts).any(axis=1)]
            centroid = (hole_verts.mean(axis=0) if len(hole_verts) > 0
                        else np.zeros(3, dtype=np.float32))
        else:
            centroid = np.zeros(3, dtype=np.float32)
        goal = np.array(deform_env.goal_pos[0], dtype=np.float32)
        obs = np.concatenate([
            grip, centroid / _WBOX, goal / _WBOX
        ]).astype(np.float32)

    elif obs_mode == 'hole_vertices':
        if len(hole_vertex_indices) > 0:
            hole_verts = verts[hole_vertex_indices]
            hole_verts = hole_verts[~np.isnan(hole_verts).any(axis=1)]
        else:
            hole_verts = np.zeros((0, 3), dtype=np.float32)
        padded = np.zeros((_MAX_HOLE_VERTS, 3), dtype=np.float32)
        n = min(len(hole_verts), _MAX_HOLE_VERTS)
        if n > 0:
            padded[:n] = hole_verts[:n] / _WBOX
        obs = np.concatenate([grip, padded.reshape(-1)]).astype(np.float32)

    elif obs_mode == 'full_mesh':
        flat_verts = (verts / _WBOX).reshape(-1)
        full = np.zeros(250 * 3, dtype=np.float32)
        n = min(len(flat_verts), 250 * 3)
        full[:n] = flat_verts[:n]
        obs = np.concatenate([grip, full]).astype(np.float32)

    else:
        raise ValueError(f'unknown obs_mode {obs_mode!r}')

    return np.clip(obs, -2.0, 2.0)


class PrivilegedObsWrapper(gym.ObservationWrapper):
    """
    Wraps a DeformEnv to replace the pixel/low-dim observation with
    privileged ground-truth cloth geometry.
    """

    MODES = ('hole_centroid', 'hole_vertices', 'full_mesh')

    def __init__(self, env, obs_mode: str = 'hole_centroid',
                 success_factor: float = None,
                 success_bonus: float = 0.0,
                 fail_penalty: float = 0.0,
                 vel_penalty: float = 0.0):
        """
        success_factor: if not None, override the env's fixed success
        threshold (0.125 m) with an ADAPTIVE one — success requires
        hole-centroid-to-goal distance < success_factor * hole_radius,
        where hole_radius is the mean distance from the hole centroid to
        the hole-loop vertices, measured at reset. ~0.6-1.0 is reasonable;
        smaller is stricter. None keeps dedo's fixed criterion unchanged.

        success_bonus: extra reward added at the terminal step IFF the
        adaptive success criterion fires. 0 = no shaping (default; PPO
        only sees dedo's distance-based reward). >0 = sparse positive
        bonus that biases PPO toward landing inside the threshold rather
        than just minimizing distance. Suggested ~100-400 (dedo's per-
        step distance reward magnitude is at most ~20).

        fail_penalty: extra negative reward added at terminal step iff
        success criterion does NOT fire. 0 = off (default). Together
        with success_bonus, you can express "+B for success, -P for
        failure" — useful when you want the policy to clearly prefer the
        success-region even at the cost of a longer path.

        vel_penalty: per-step penalty proportional to the cloth's mean
        per-vertex displacement between consecutive policy steps (proxy
        for cloth speed). reward -= vel_penalty * mean(‖v_t - v_{t-1}‖).
        Only applied during the policy phase, not the post-policy settle
        (which is physics-driven and not policy-controllable). 0 = off.
        Per-step displacements are ~0.05–0.5 m at default sim_steps_per_action,
        so coefs ~1–10 yield per-step penalties of ~-0.1 to -5, comparable
        to dedo's per-step distance reward magnitude.
        """
        assert obs_mode in self.MODES, f'obs_mode must be one of {self.MODES}'
        env.args.cam_resolution = 0
        super().__init__(env)

        self.obs_mode = obs_mode
        self._hole_vertex_indices = []
        self._goal_pos = None
        self._success_factor = success_factor
        self._success_bonus = float(success_bonus)
        self._fail_penalty = float(fail_penalty)
        self._vel_penalty = float(vel_penalty)
        self._prev_verts = None
        self._hole_radius = None

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
        return build_privileged_obs(
            self.env, self.obs_mode, self._hole_vertex_indices)

    def _measure_hole_radius(self):
        """Mean distance from hole centroid to hole-loop vertices, in m.
        Captured at reset so the adaptive threshold reflects the hole
        geometry at episode start (it shouldn't change much after)."""
        if not self._hole_vertex_indices:
            return 0.0
        _, verts = get_mesh_data(self.env.sim, self.env.deform_id)
        verts = np.asarray(verts, dtype=np.float32)
        hv = verts[self._hole_vertex_indices]
        hv = hv[~np.isnan(hv).any(axis=1)]
        if len(hv) == 0:
            return 0.0
        centroid = hv.mean(axis=0)
        return float(np.mean(np.linalg.norm(hv - centroid, axis=1)))

    def reset(self):
        self.env.reset()
        self._hole_vertex_indices = self._get_hole_indices()
        self._goal_pos = self.env.goal_pos.copy()
        self._hole_radius = self._measure_hole_radius()
        if self._vel_penalty > 0.0:
            _, verts = get_mesh_data(self.env.sim, self.env.deform_id)
            self._prev_verts = np.asarray(verts, dtype=np.float32)
        else:
            self._prev_verts = None
        return self._build_obs()

    def observation(self, obs):
        return self._build_obs()

    def step(self, action):
        _, reward, done, info = self.env.step(action)
        obs = self._build_obs()
        # Cloth-velocity penalty: discourages whippy trajectories. Only
        # active during the policy phase — on the terminal step, the
        # underlying step() already ran make_final_steps, so the verts
        # delta would mix policy motion with gravity-driven settle motion
        # which the policy can't control. Skip it there.
        if self._vel_penalty > 0.0 and not done:
            _, verts_now = get_mesh_data(self.env.sim, self.env.deform_id)
            verts_now = np.asarray(verts_now, dtype=np.float32)
            if self._prev_verts is not None and \
                    self._prev_verts.shape == verts_now.shape:
                disp = np.linalg.norm(verts_now - self._prev_verts, axis=1)
                disp = disp[~np.isnan(disp)]
                if len(disp) > 0:
                    mean_speed = float(disp.mean())
                    pen = self._vel_penalty * mean_speed
                    reward = float(reward) - pen
                    info['vel_penalty'] = pen
                    info['cloth_mean_speed'] = mean_speed
            self._prev_verts = verts_now
        # Adaptive success: override env's fixed 0.125m threshold with one
        # proportional to the hole's effective radius. Optionally inject
        # a terminal bonus/penalty into the reward so PPO actually
        # optimizes the (adaptive) success criterion, not just distance.
        if (done and 'is_success' in info
                and self._success_factor is not None
                and self._hole_radius is not None):
            _, verts = get_mesh_data(self.env.sim, self.env.deform_id)
            verts = np.asarray(verts, dtype=np.float32)
            hv = verts[self._hole_vertex_indices] \
                if self._hole_vertex_indices else np.zeros((0, 3))
            hv = hv[~np.isnan(hv).any(axis=1)]
            if len(hv) > 0:
                centroid = hv.mean(axis=0)
                goal = np.asarray(self.env.goal_pos[0], dtype=np.float32)
                dist = float(np.linalg.norm(centroid - goal))
                thresh = self._hole_radius * self._success_factor
                is_success = bool(dist < thresh)
                info['is_success'] = is_success
                info['adaptive_dist'] = dist
                info['adaptive_thresh'] = thresh
                info['hole_radius'] = self._hole_radius

                shaping = 0.0
                if is_success and self._success_bonus != 0.0:
                    shaping += self._success_bonus
                elif (not is_success) and self._fail_penalty != 0.0:
                    shaping -= self._fail_penalty
                if shaping != 0.0:
                    reward = float(reward) + shaping
                    info['shaping_added'] = shaping
        return obs, reward, done, info

    @property
    def hole_radius(self):
        return self._hole_radius

    @property
    def success_threshold_m(self):
        """Currently active success-distance threshold in meters."""
        if self._success_factor is None:
            return 0.125  # dedo default: SUCESS_REWARD_TRESHOLD * 20 / 400
        if self._hole_radius is None:
            return float('nan')
        return self._hole_radius * self._success_factor
