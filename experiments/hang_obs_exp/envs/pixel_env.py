"""
PixelObsWrapper — RGB image observation wrapper.

Counterpart to PrivilegedObsWrapper / PointCloudObsWrapper for the obs-
modality comparison: the policy sees a (H, W, 3) uint8 image rendered by
the env's camera (no privileged geometry, no point cloud). Optionally
concatenates 12-dim gripper proprioception alongside the image as a Dict
observation, which is a much stronger baseline than pure pixels — the
CNN doesn't have to re-derive gripper position from the rendered image.

Two observation layouts:

  include_grip=True  (DEFAULT)   →  Dict({
                                         'image': Box(0, 255, (H, W, 3), uint8),
                                         'grip':  Box(-2, 2,  (12,),    float32),
                                     })
                                     Use SB3 'MultiInputPolicy'.

  include_grip=False             →  Box(0, 255, (H, W, 3), uint8)
                                     Use SB3 'CnnPolicy'.

The wrapper applies the same reward shaping (adaptive success,
success_bonus, fail_penalty, vel_penalty) and emits the same rwd_diag/*
diagnostic keys as PrivilegedObsWrapper / PointCloudObsWrapper so the
existing RewardDiagnosticsCallback and dump_run_config helpers work
without any modification.

Forces uint8 pixels on the underlying DeformEnv (SB3 CnnPolicy expects
uint8 [0,255] inputs and scales them internally).
"""
import numpy as np
import gym
from gym import spaces

from dedo.utils.mesh_utils import get_mesh_data

_GRIP_DIM = 12
_WBOX = 20.0  # workspace normalization (matches PrivilegedObsWrapper / PCD)


class PixelObsWrapper(gym.Wrapper):
    """Pixel-observation wrapper with optional gripper proprioception.

    NOTE: subclasses gym.Wrapper (not ObservationWrapper) because we also
    rewrite reward in step() — ObservationWrapper would route step() back
    through observation() in a way that complicates the reward path.
    """

    def __init__(self, env,
                 cam_resolution: int = 64,
                 include_grip: bool = True,
                 success_factor: float = None,
                 success_bonus: float = 0.0,
                 fail_penalty: float = 0.0,
                 vel_penalty: float = 0.0):
        # Force the camera + uint8 pixel mode dedo expects for SB3 CnnPolicy.
        env.args.cam_resolution = cam_resolution
        env.args.uint8_pixels = True
        # flat_obs would collapse the (H, W, 3) image into a 1D vector, which
        # defeats the CNN. Disable defensively in case the user passed it.
        env.args.flat_obs = False
        super().__init__(env)

        self._cam_resolution = cam_resolution
        self._include_grip = bool(include_grip)
        self._success_factor = success_factor
        self._success_bonus = float(success_bonus)
        self._fail_penalty = float(fail_penalty)
        self._vel_penalty = float(vel_penalty)
        self._prev_verts = None

        self._hole_vertex_indices = []
        self._goal_pos = None
        self._hole_radius = None

        # Per-episode reward bookkeeping for the diagnostics callback.
        self._ep_step_count = 0
        self._ep_base_reward_sum = 0.0
        self._ep_vel_penalty_sum = 0.0  # accumulates magnitude (>= 0)

        # Resolve the actual DeformEnv once for sim/mesh access. Anything
        # wrapped between us and DeformEnv (e.g. RetryResetEnv) blocks
        # attribute access through the gym.Wrapper chain.
        from dedo.envs.deform_env import DeformEnv  # local: avoid hard dep at import time
        self._deform = env
        while hasattr(self._deform, 'env') and not isinstance(
                self._deform, DeformEnv):
            self._deform = self._deform.env

        image_space = spaces.Box(
            low=0, high=255,
            shape=(cam_resolution, cam_resolution, 3),
            dtype=np.uint8,
        )
        if self._include_grip:
            grip_space = spaces.Box(
                low=-np.ones(_GRIP_DIM, dtype=np.float32) * 2.0,
                high=np.ones(_GRIP_DIM, dtype=np.float32) * 2.0,
                dtype=np.float32,
            )
            self.observation_space = spaces.Dict({
                'image': image_space,
                'grip': grip_space,
            })
        else:
            self.observation_space = image_space

    # ------------------------------------------------------------------
    # Hole / goal bookkeeping (same logic as PrivilegedObsWrapper).
    # ------------------------------------------------------------------
    def _get_hole_indices(self):
        if hasattr(self._deform.args, 'deform_true_loop_vertices'):
            loops = self._deform.args.deform_true_loop_vertices
            return [idx for loop in loops for idx in loop]
        return []

    def _measure_hole_radius(self):
        if not self._hole_vertex_indices:
            return 0.0
        _, verts = get_mesh_data(self._deform.sim, self._deform.deform_id)
        verts = np.asarray(verts, dtype=np.float32)
        hv = verts[self._hole_vertex_indices]
        hv = hv[~np.isnan(hv).any(axis=1)]
        if len(hv) == 0:
            return 0.0
        c = hv.mean(axis=0)
        return float(np.mean(np.linalg.norm(hv - c, axis=1)))

    # ------------------------------------------------------------------
    # Obs assembly.
    # ------------------------------------------------------------------
    def _grip_vec(self):
        grip = np.asarray(self._deform.get_grip_obs(), dtype=np.float32)
        return np.clip(grip / _WBOX, -2.0, 2.0)

    def _wrap_obs(self, image_obs):
        """The raw obs from DeformEnv is the image; we just wrap it into
        a Dict if include_grip is set."""
        # Belt-and-suspenders: dedo's get_obs() should already return
        # uint8 since we set args.uint8_pixels=True, but VecNormalize and
        # SB3's checks both insist on dtype matching observation_space.
        img = image_obs
        if img.dtype != np.uint8:
            img = np.clip(img, 0, 255).astype(np.uint8)
        if self._include_grip:
            return {'image': img, 'grip': self._grip_vec()}
        return img

    def reset(self, **kwargs):
        raw_obs = self.env.reset(**kwargs)
        self._hole_vertex_indices = self._get_hole_indices()
        self._goal_pos = self._deform.goal_pos.copy()
        self._hole_radius = self._measure_hole_radius()
        if self._vel_penalty > 0.0:
            _, verts = get_mesh_data(self._deform.sim, self._deform.deform_id)
            self._prev_verts = np.asarray(verts, dtype=np.float32)
        else:
            self._prev_verts = None
        self._ep_step_count = 0
        self._ep_base_reward_sum = 0.0
        self._ep_vel_penalty_sum = 0.0
        return self._wrap_obs(raw_obs)

    def step(self, action):
        raw_obs, reward, done, info = self.env.step(action)
        obs = self._wrap_obs(raw_obs)

        base_reward = float(reward)
        base_is_success = info.get('is_success', None)

        # ------------------------------------------------------------------
        # Cloth-velocity penalty (non-terminal only). Mirrors
        # PrivilegedObsWrapper / PointCloudObsWrapper so the same coef has
        # the same meaning across obs modes.
        # ------------------------------------------------------------------
        vel_pen = 0.0
        if self._vel_penalty > 0.0 and not done:
            _, verts_now = get_mesh_data(
                self._deform.sim, self._deform.deform_id)
            verts_now = np.asarray(verts_now, dtype=np.float32)
            if (self._prev_verts is not None
                    and self._prev_verts.shape == verts_now.shape):
                disp = np.linalg.norm(verts_now - self._prev_verts, axis=1)
                disp = disp[~np.isnan(disp)]
                if len(disp) > 0:
                    mean_speed = float(disp.mean())
                    vel_pen = self._vel_penalty * mean_speed
                    reward = base_reward - vel_pen
                    info['vel_penalty'] = vel_pen
                    info['cloth_mean_speed'] = mean_speed
            self._prev_verts = verts_now

        # ------------------------------------------------------------------
        # Adaptive success override + terminal shaping (identical logic to
        # PrivilegedObsWrapper / PointCloudObsWrapper).
        # ------------------------------------------------------------------
        adaptive_dist = None
        adaptive_thresh = None
        adaptive_is_success = None
        terminal_shaping = 0.0

        if (done and 'is_success' in info
                and self._success_factor is not None
                and self._hole_radius is not None):
            _, verts = get_mesh_data(self._deform.sim, self._deform.deform_id)
            verts = np.asarray(verts, dtype=np.float32)
            hv = (verts[self._hole_vertex_indices]
                  if self._hole_vertex_indices else np.zeros((0, 3)))
            hv = hv[~np.isnan(hv).any(axis=1)]
            if len(hv) > 0:
                centroid = hv.mean(axis=0)
                goal = np.asarray(self._deform.goal_pos[0], dtype=np.float32)
                adaptive_dist = float(np.linalg.norm(centroid - goal))
                adaptive_thresh = float(
                    self._hole_radius * self._success_factor)
                adaptive_is_success = bool(adaptive_dist < adaptive_thresh)

                info['is_success'] = adaptive_is_success
                info['adaptive_dist'] = adaptive_dist
                info['adaptive_thresh'] = adaptive_thresh
                info['hole_radius'] = self._hole_radius

                if adaptive_is_success and self._success_bonus != 0.0:
                    terminal_shaping = self._success_bonus
                elif (not adaptive_is_success) and self._fail_penalty != 0.0:
                    terminal_shaping = -self._fail_penalty
                if terminal_shaping != 0.0:
                    reward = float(reward) + terminal_shaping
                    info['shaping_added'] = terminal_shaping

        # ------------------------------------------------------------------
        # Per-episode accumulators + diagnostic emission on done.
        # ------------------------------------------------------------------
        self._ep_step_count += 1
        self._ep_base_reward_sum += base_reward
        self._ep_vel_penalty_sum += vel_pen

        if done:
            self._emit_episode_diagnostics(
                info,
                terminal_base=base_reward,
                terminal_shaping=terminal_shaping,
                total_reward=float(reward),
                base_is_success=base_is_success,
                adaptive_is_success=adaptive_is_success,
                adaptive_dist=adaptive_dist,
                adaptive_thresh=adaptive_thresh,
            )

        return obs, reward, done, info

    # ----------------------------------------------------------------------
    # Diagnostics emission. Same metric names as the other wrappers.
    # ----------------------------------------------------------------------
    def _emit_episode_diagnostics(self, info, *,
                                  terminal_base, terminal_shaping,
                                  total_reward,
                                  base_is_success, adaptive_is_success,
                                  adaptive_dist, adaptive_thresh):
        info['rwd_diag/reward/episode_total'] = float(
            self._ep_base_reward_sum
            - self._ep_vel_penalty_sum
            + terminal_shaping
        )
        info['rwd_diag/reward/base_sum'] = float(self._ep_base_reward_sum)
        info['rwd_diag/reward/vel_penalty_sum'] = float(
            self._ep_vel_penalty_sum)
        info['rwd_diag/reward/terminal_base'] = float(terminal_base)
        info['rwd_diag/reward/terminal_shaping'] = float(terminal_shaping)
        info['rwd_diag/reward/episode_length'] = int(self._ep_step_count)

        if base_is_success is not None:
            info['rwd_diag/success/base_rate'] = int(bool(base_is_success))
        if adaptive_is_success is not None:
            info['rwd_diag/success/adaptive_rate'] = int(
                bool(adaptive_is_success))
        if base_is_success is not None and adaptive_is_success is not None:
            info['rwd_diag/success/disagree_rate'] = int(
                bool(base_is_success) != bool(adaptive_is_success))
        active = (adaptive_is_success
                  if adaptive_is_success is not None else base_is_success)
        if active is not None:
            info['rwd_diag/success/active_rate'] = int(bool(active))

        if adaptive_dist is not None:
            info['rwd_diag/task/adaptive_dist'] = adaptive_dist
        if adaptive_thresh is not None:
            info['rwd_diag/task/adaptive_thresh'] = adaptive_thresh
        if self._hole_radius is not None:
            info['rwd_diag/task/hole_radius'] = float(self._hole_radius)

    # ------------------------------------------------------------------
    # Properties (parity with the other obs wrappers).
    # ------------------------------------------------------------------
    @property
    def hole_radius(self):
        return self._hole_radius

    @property
    def success_threshold_m(self):
        if self._success_factor is None:
            return 0.125  # cosmetic only; see note in privileged_env.py
        if self._hole_radius is None:
            return float('nan')
        return self._hole_radius * self._success_factor
