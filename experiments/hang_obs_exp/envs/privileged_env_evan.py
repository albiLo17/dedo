"""
PrivilegedObsWrapper — replaces the camera image with ground-truth cloth geometry.

Four observation modes (set via obs_mode arg):

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

  'enriched'       (40-dim)
      gripper (12) + hole geometry summary (8: centroid 3, normal 3, radius 1,
      eccentricity 1) + cloth geometry summary (9: bbox_min 3, bbox_max 3,
      centroid 3) + task-relative vectors (7: gripper→hole 3, hole→goal 3,
      hole→goal distance 1) + hanger goal (3) + time progress (1)
      → compact mid-level oracle: the policy sees hole orientation /
        radius / shape and cloth extent without paying the full-mesh cost.

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

    elif obs_mode == 'enriched':
        # --- Hole geometry summary (8 dim) -----------------------------
        if len(hole_vertex_indices) > 0:
            hv = verts[hole_vertex_indices]
            hv = hv[~np.isnan(hv).any(axis=1)]
        else:
            hv = np.zeros((0, 3), dtype=np.float32)

        if len(hv) >= 3:
            hole_centroid = hv.mean(axis=0)
            centered = hv - hole_centroid
            # SVD on the centered loop: right-singular vectors = principal
            # axes; smallest one is the plane normal.
            try:
                _, sv, vt = np.linalg.svd(centered, full_matrices=False)
                hole_normal = vt[2].astype(np.float32)
                # Sign convention: choose normal pointing toward +z so it's
                # stable across episodes (loops can come back with either
                # winding from PCA otherwise).
                if hole_normal[2] < 0:
                    hole_normal = -hole_normal
                # In-plane eccentricity: ratio of major / minor axis.
                # Clip to a finite range so the obs stays well-conditioned.
                ecc = float(sv[0] / max(sv[1], 1e-6))
                ecc = min(ecc, 10.0)
                # Effective radius: mean dist from centroid to loop verts.
                hole_radius = float(np.mean(np.linalg.norm(centered, axis=1)))
            except np.linalg.LinAlgError:
                hole_normal = np.zeros(3, dtype=np.float32)
                ecc = 1.0
                hole_radius = 0.0
        else:
            hole_centroid = np.zeros(3, dtype=np.float32)
            hole_normal = np.zeros(3, dtype=np.float32)
            ecc = 1.0
            hole_radius = 0.0

        # --- Cloth geometry summary (9 dim) ----------------------------
        valid_verts = verts[~np.isnan(verts).any(axis=1)]
        if len(valid_verts) > 0:
            cloth_min = valid_verts.min(axis=0)
            cloth_max = valid_verts.max(axis=0)
            cloth_centroid = valid_verts.mean(axis=0)
        else:
            cloth_min = np.zeros(3, dtype=np.float32)
            cloth_max = np.zeros(3, dtype=np.float32)
            cloth_centroid = np.zeros(3, dtype=np.float32)

        # --- Task-relative vectors (7 dim) -----------------------------
        goal = np.asarray(deform_env.goal_pos[0], dtype=np.float32)
        # Gripper world position: get_grip_obs returns [pos_a(3) vel_a(3)
        # pos_b(3) vel_b(3)]; midpoint of the two anchor positions is the
        # most useful "where the gripper is" summary.
        grip_raw = np.asarray(deform_env.get_grip_obs(), dtype=np.float32)
        gripper_pos = 0.5 * (grip_raw[0:3] + grip_raw[6:9])
        grip_to_hole = (hole_centroid - gripper_pos).astype(np.float32)
        hole_to_goal = (goal - hole_centroid).astype(np.float32)
        hole_to_goal_dist = float(np.linalg.norm(hole_to_goal))

        # --- Time progress (1 dim) -------------------------------------
        # Finite-horizon PPO benefits from knowing how much time is left;
        # stepnum is reset by DeformEnv on each reset().
        max_len = max(int(getattr(deform_env, 'max_episode_len', 200)), 1)
        progress = float(min(getattr(deform_env, 'stepnum', 0) / max_len, 1.0))

        obs = np.concatenate([
            grip,                                           # 12
            hole_centroid / _WBOX,                          # 3
            hole_normal,                                    # 3 (already unit)
            np.array([hole_radius / _WBOX], dtype=np.float32),   # 1
            np.array([ecc / 10.0], dtype=np.float32),       # 1 (normalized)
            cloth_min / _WBOX,                              # 3
            cloth_max / _WBOX,                              # 3
            cloth_centroid / _WBOX,                         # 3
            grip_to_hole / _WBOX,                           # 3
            hole_to_goal / _WBOX,                           # 3
            np.array([hole_to_goal_dist / _WBOX], dtype=np.float32),  # 1
            goal / _WBOX,                                   # 3
            np.array([progress], dtype=np.float32),         # 1
        ]).astype(np.float32)

    else:
        raise ValueError(f'unknown obs_mode {obs_mode!r}')

    return np.clip(obs, -2.0, 2.0)


class PrivilegedObsWrapper(gym.ObservationWrapper):
    """
    Wraps a DeformEnv to replace the pixel/low-dim observation with
    privileged ground-truth cloth geometry.
    """

    MODES = ('hole_centroid', 'hole_vertices', 'full_mesh', 'enriched')

    def __init__(self, env, obs_mode: str = 'hole_centroid',
                 success_factor: float = None,
                 success_bonus: float = 0.0,
                 fail_penalty: float = 0.0,
                 vel_penalty: float = 0.0,
                 pre_settle_coef: float = 0.0,
                 action_penalty: float = 0.0):
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

        action_penalty: per-step penalty on the magnitude of the policy's
        action vector. reward -= action_penalty * mean(action**2). Action
        is in [-1, 1]^6 (PPO normalized), so mean(a**2) is in [0, 1] —
        coefs ~0.1-2 yield per-step penalties of ~-0.05 to -2, comparable
        to vel_penalty's range. Mirrors mujoco-style action-cost shaping;
        used to discourage flailing/bang-bang control. 0 = off.

        pre_settle_coef: linear penalty on the hole-to-goal distance (in
        meters) at the terminal step, BEFORE make_final_steps runs.
        reward -= pre_settle_coef * pre_settle_dist_m. Counters the
        "lift cloth high, let gravity drop it onto the hanger" exploit:
        the post-settle reward currently rewards a ballistic alignment
        equally to a threaded one, but this term only rewards being
        close to the goal at policy-handoff. 0 = off. The post-settle
        reward has effective coef FINAL_REWARD_MULT/WORKSPACE_BOX_SIZE
        = 400/20 = 20 per meter, so coef ~10–30 is a meaningful
        equal-or-greater counterweight; start at 20.
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
        self._pre_settle_coef = float(pre_settle_coef)
        self._action_penalty = float(action_penalty)
        self._prev_verts = None
        self._hole_radius = None

        grip_dim = 12

        if obs_mode == 'hole_centroid':
            obs_dim = grip_dim + 3 + 3
        elif obs_mode == 'hole_vertices':
            obs_dim = grip_dim + _MAX_HOLE_VERTS * 3
        elif obs_mode == 'full_mesh':
            obs_dim = grip_dim + 250 * 3
        elif obs_mode == 'enriched':
            # 12 grip + 8 hole + 9 cloth + 7 task-rel + 3 goal + 1 time = 40
            obs_dim = grip_dim + 8 + 9 + 7 + 3 + 1

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
        # Action-magnitude penalty: discourages bang-bang/flailing control.
        # Computed on the post-clip action that was actually applied.
        # Applied every step (including terminal) — the action was chosen
        # by the policy and is well-defined regardless of the settle phase.
        if self._action_penalty > 0.0:
            a = np.asarray(action, dtype=np.float32)
            act_cost = float(np.mean(a * a))
            pen = self._action_penalty * act_cost
            reward = float(reward) - pen
            info['action_penalty'] = pen
            info['action_cost'] = act_cost
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
        # Pre-settle distance penalty: linear shaping on the hole-to-goal
        # distance at policy handoff (BEFORE make_final_steps drops the
        # cloth under gravity). Counters the "lift high, drop straight
        # down" exploit by rewarding only positions reached via control.
        if (done and self._pre_settle_coef > 0.0
                and 'pre_settle_dist_m' in info):
            pre_dist = float(info['pre_settle_dist_m'])
            pen = self._pre_settle_coef * pre_dist
            reward = float(reward) - pen
            info['pre_settle_penalty'] = pen
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