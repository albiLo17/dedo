"""
PrivilegedObsWrapper — replaces the camera image with ground-truth cloth geometry.

Four observation modes (set via obs_mode arg):

  'hole_centroid'  (18-dim)  [DEFAULT]
      gripper_pos_vel (12) + hole_centroid_xyz (3) + hanger_goal_xyz (3)
      → minimal oracle: tells the policy exactly where the hole IS and where
        the hanger IS without needing a camera at all.

  'hole_centroid_corners'  (30-dim)
      gripper (12) + hole_centroid (3) + 4 cloth-corner positions (12) + goal (3)
      → adds rigid-body framing of the cloth on top of the centroid: the
        policy can disambiguate cloth pose / orientation from a tiny extra
        observation. Corners are identified once at reset via PCA-extremes
        on the cloth mesh and ordered deterministically (CCW around the
        centroid in the cloth's principal-plane frame), so the same physical
        corner always lands in the same slot of the obs.

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
_NUM_CORNERS = 4


def identify_cloth_corners(verts):
    """Identify the 4 corners of a roughly rectangular cloth mesh.

    Strategy: the cloth lies on a 2D manifold; SVD on the centered mesh
    gives the two principal in-plane axes. The 4 corners are the vertices
    that maximize each of the 4 sign-combinations of (proj_a1, proj_a2).
    The PCA axis signs are flipped so axis-1 has positive y and axis-2
    has positive z in world frame — this pins the corner-slot assignment
    across episodes (same physical corner → same slot of the obs), since
    HangProcCloth always lays the cloth in roughly the world y-z plane.

    Returns a list of 4 vertex indices in CCW order around the cloth
    centroid, or [] if corner detection fails (degenerate mesh, all-NaN,
    or the principal axes happen to align with world x — all unlikely
    for HangProcCloth's procedural cloths)."""
    verts = np.asarray(verts, dtype=np.float32)
    valid_mask = ~np.isnan(verts).any(axis=1)
    valid_verts = verts[valid_mask]
    if len(valid_verts) < 4:
        return []

    centered = valid_verts - valid_verts.mean(axis=0)
    _, _, vt = np.linalg.svd(centered, full_matrices=False)
    a1, a2 = vt[0].copy(), vt[1].copy()
    if a1[1] < 0:
        a1 = -a1
    if a2[2] < 0:
        a2 = -a2

    proj1 = centered @ a1
    proj2 = centered @ a2
    sign_combos = [(1.0, 1.0), (-1.0, 1.0), (-1.0, -1.0), (1.0, -1.0)]
    valid_idx = np.where(valid_mask)[0]
    corner_indices = []
    seen = set()
    for s1, s2 in sign_combos:
        scores = s1 * proj1 + s2 * proj2
        order = np.argsort(-scores)
        for k in order:
            cand = int(valid_idx[k])
            if cand not in seen:
                corner_indices.append(cand)
                seen.add(cand)
                break
    if len(corner_indices) != _NUM_CORNERS:
        return []
    return corner_indices


def build_privileged_obs(deform_env, obs_mode, hole_vertex_indices,
                         corner_indices=None):
    """Compute the privileged obs for a given mode from the underlying
    DeformEnv state. Mode-independent inputs (gripper, mesh, goal) → mode-
    specific concatenation. Used by PrivilegedObsWrapper and by the demo
    recorder so a single recorded demo can be replayed in any obs mode.

    `corner_indices`: only consumed by the 'hole_centroid_corners' mode.
    If None, corners are detected on-the-fly from the current mesh — slower
    but lets external callers (e.g. record_demo) avoid threading the indices
    through. The wrapper itself caches the indices at reset so per-step
    calls are O(1)."""
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

    elif obs_mode == 'hole_centroid_corners':
        if len(hole_vertex_indices) > 0:
            hole_verts = verts[hole_vertex_indices]
            hole_verts = hole_verts[~np.isnan(hole_verts).any(axis=1)]
            centroid = (hole_verts.mean(axis=0) if len(hole_verts) > 0
                        else np.zeros(3, dtype=np.float32))
        else:
            centroid = np.zeros(3, dtype=np.float32)

        if corner_indices is None:
            corner_indices = identify_cloth_corners(verts)
        corners_arr = np.zeros((_NUM_CORNERS, 3), dtype=np.float32)
        if len(corner_indices) == _NUM_CORNERS:
            picked = verts[corner_indices]
            picked = np.where(np.isnan(picked), 0.0, picked)
            corners_arr[:] = picked

        goal = np.array(deform_env.goal_pos[0], dtype=np.float32)
        obs = np.concatenate([
            grip,
            centroid / _WBOX,
            (corners_arr / _WBOX).reshape(-1),
            goal / _WBOX,
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

    MODES = ('hole_centroid', 'hole_centroid_corners',
             'hole_vertices', 'full_mesh')

    def __init__(self, env, obs_mode: str = 'hole_centroid',
                 success_factor: float = None,
                 success_bonus: float = 0.0,
                 fail_penalty: float = 0.0,
                 vel_penalty: float = 0.0,
                 action_penalty: float = 0.0,
                 pre_settle_coef: float = 0.0,
                 dist_reward_coef: float = 0.0,
                 threading_bonus_coef: float = 0.0):
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
        used to discourage flailing/bang-bang control. Applied every step
        including terminal (action is well-defined). 0 = off.

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

        dist_reward_coef: per-step dense distance reward, fires every
        step (including terminal). reward += dist_reward_coef / (1 +
        adaptive_dist), where adaptive_dist is the same hole-centroid-
        to-goal distance the terminal success check uses. Designed to
        repair the credit-assignment problem where 99% of the reward
        signal arrives at the terminal step (after the policy stops
        acting) — this gives PPO a continuous "you're getting closer"
        signal that's nonzero throughout the episode and largest
        exactly when the cloth is on the hole. Suggested 0.5–2.0;
        cumulative-over-200-steps is comparable to the success_bonus
        magnitude. 0 = off (default, preserves the old reward shape).

        threading_bonus_coef: per-step bonus that fires every step the
        cloth is within the success threshold (adaptive_dist <
        adaptive_thresh). Tells the policy "stay here" once it threads,
        rather than passing through. Equivalent to a non-sparse success
        signal that doesn't depend on post-settle physics.

        CAVEAT — the underlying detection is the same hole-centroid-to-
        goal distance check used by the terminal success criterion, so
        it inherits the same flakiness (cloth can be near the peg
        without being topologically threaded; cloth can thread briefly
        during a swing without staying). For the first denser-reward
        experiments, prefer dist_reward_coef alone (which doesn't need
        a threading detection — it's pure distance) and leave this at
        0. Re-enable after a better threading metric (e.g. peg-z
        between top/bottom hole-loop vertices, or winding number)
        replaces the centroid-distance check.

        Disabled if success_factor is None. 0 = off (default).
        """
        assert obs_mode in self.MODES, f'obs_mode must be one of {self.MODES}'
        env.args.cam_resolution = 0
        super().__init__(env)

        self.obs_mode = obs_mode
        self._hole_vertex_indices = []
        self._corner_indices = []
        self._goal_pos = None
        self._success_factor = success_factor
        self._success_bonus = float(success_bonus)
        self._fail_penalty = float(fail_penalty)
        self._vel_penalty = float(vel_penalty)
        self._action_penalty = float(action_penalty)
        self._pre_settle_coef = float(pre_settle_coef)
        self._dist_reward_coef = float(dist_reward_coef)
        self._threading_bonus_coef = float(threading_bonus_coef)
        self._prev_verts = None
        self._hole_radius = None
        # Per-episode counter for diagnostics: how many steps were
        # within threshold? The new threading_bonus + this counter
        # together let us read off "how often is the cloth threaded
        # during an episode" which post-settle success fails to
        # capture (cloth can thread at step 80 then slip during settle).
        self._ep_threading_steps = 0
        self._ep_dist_reward_sum = 0.0
        self._ep_threading_bonus_sum = 0.0

        # Per-episode reward bookkeeping for the diagnostics callback.
        self._ep_step_count = 0
        self._ep_base_reward_sum = 0.0
        self._ep_vel_penalty_sum = 0.0      # accumulates magnitude (>= 0)
        self._ep_action_penalty_sum = 0.0   # accumulates magnitude (>= 0)

        grip_dim = 12

        if obs_mode == 'hole_centroid':
            obs_dim = grip_dim + 3 + 3
        elif obs_mode == 'hole_centroid_corners':
            obs_dim = grip_dim + 3 + _NUM_CORNERS * 3 + 3
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
            self.env, self.obs_mode, self._hole_vertex_indices,
            corner_indices=self._corner_indices)

    def _compute_adaptive_dist(self, verts=None):
        """Return (adaptive_dist, adaptive_thresh) for the current sim
        state, or (None, None) if the hole geometry is unavailable.

        Single source of truth for both per-step shaping (dist_reward,
        threading_bonus) and the terminal success check, so they always
        agree. `verts` may be passed in to avoid a redundant
        `get_mesh_data` query when the caller already has them (the
        step() path queries verts once and reuses).
        """
        if not self._hole_vertex_indices or self._hole_radius is None:
            return None, None
        if verts is None:
            _, verts = get_mesh_data(self.env.sim, self.env.deform_id)
            verts = np.asarray(verts, dtype=np.float32)
        hv = verts[self._hole_vertex_indices]
        hv = hv[~np.isnan(hv).any(axis=1)]
        if len(hv) == 0:
            return None, None
        centroid = hv.mean(axis=0)
        goal = np.asarray(self.env.goal_pos[0], dtype=np.float32)
        dist = float(np.linalg.norm(centroid - goal))
        if self._success_factor is None:
            thresh = None
        else:
            thresh = float(self._hole_radius * self._success_factor)
        return dist, thresh

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
        # Cache corner indices once per episode: cloth identity is fixed
        # within an episode (only positions change), so PCA-extreme detection
        # on the freshly-reset mesh picks corners that remain valid for the
        # whole rollout.
        _, _verts0 = get_mesh_data(self.env.sim, self.env.deform_id)
        self._corner_indices = identify_cloth_corners(_verts0)
        if self._vel_penalty > 0.0:
            _, verts = get_mesh_data(self.env.sim, self.env.deform_id)
            self._prev_verts = np.asarray(verts, dtype=np.float32)
        else:
            self._prev_verts = None
        self._ep_step_count = 0
        self._ep_base_reward_sum = 0.0
        self._ep_vel_penalty_sum = 0.0
        self._ep_action_penalty_sum = 0.0
        self._ep_threading_steps = 0
        self._ep_dist_reward_sum = 0.0
        self._ep_threading_bonus_sum = 0.0
        return self._build_obs()

    def observation(self, obs):
        return self._build_obs()

    def step(self, action):
        _, reward, done, info = self.env.step(action)
        obs = self._build_obs()

        # Capture base values BEFORE any wrapper modification, so diagnostics
        # can attribute every reward component cleanly.
        base_reward = float(reward)
        base_is_success = info.get('is_success', None)

        # ------------------------------------------------------------------
        # (1) Action-magnitude penalty (every step, including terminal).
        #     The action was chosen by the policy and is well-defined
        #     regardless of the settle phase. mean(a**2) in [0, 1] for
        #     a in [-1, 1]^6.
        # ------------------------------------------------------------------
        act_pen = 0.0
        if self._action_penalty > 0.0:
            a = np.asarray(action, dtype=np.float32)
            act_cost = float(np.mean(a * a))
            act_pen = self._action_penalty * act_cost
            reward = float(reward) - act_pen
            info['action_penalty'] = act_pen
            info['action_cost'] = act_cost

        # ------------------------------------------------------------------
        # (2) Cloth-velocity penalty (non-terminal only).
        #     The terminal step's verts delta would mix policy motion with
        #     gravity-driven settle motion the policy can't control, so we
        #     skip it there.
        # ------------------------------------------------------------------
        vel_pen = 0.0
        if self._vel_penalty > 0.0 and not done:
            _, verts_now = get_mesh_data(self.env.sim, self.env.deform_id)
            verts_now = np.asarray(verts_now, dtype=np.float32)
            if (self._prev_verts is not None
                    and self._prev_verts.shape == verts_now.shape):
                disp = np.linalg.norm(verts_now - self._prev_verts, axis=1)
                disp = disp[~np.isnan(disp)]
                if len(disp) > 0:
                    mean_speed = float(disp.mean())
                    vel_pen = self._vel_penalty * mean_speed
                    reward = float(reward) - vel_pen
                    info['vel_penalty'] = vel_pen
                    info['cloth_mean_speed'] = mean_speed
            self._prev_verts = verts_now

        # ------------------------------------------------------------------
        # (2.5) Per-step dense distance reward and threading bonus.
        #
        #   Repairs the credit-assignment problem: under the legacy reward
        #   shape, ~85% of episode return arrives at the terminal step
        #   (dedo's FINAL_REWARD_MULT=400 dominates per-step base ~0.01).
        #   PPO can't tell which mid-episode action helped, so it walks
        #   the policy on noise. These two terms put a continuous signal
        #   at every step that's largest exactly when the cloth is on
        #   the hole.
        #
        #     dist_reward   = dist_reward_coef / (1 + adaptive_dist)
        #     threading_bonus = threading_bonus_coef  if adaptive_dist
        #                       < adaptive_thresh
        #
        #   adaptive_dist is the same hole-centroid-to-goal distance the
        #   terminal success check uses, so the per-step signal points
        #   at the same target as the eventual reward.
        # ------------------------------------------------------------------
        dist_reward = 0.0
        threading_bonus = 0.0
        cur_adaptive_dist = None
        cur_adaptive_thresh = None
        if self._dist_reward_coef > 0.0 or self._threading_bonus_coef > 0.0:
            cur_adaptive_dist, cur_adaptive_thresh = (
                self._compute_adaptive_dist())
            if cur_adaptive_dist is not None:
                if self._dist_reward_coef > 0.0:
                    dist_reward = (
                        self._dist_reward_coef / (1.0 + cur_adaptive_dist))
                    reward = float(reward) + dist_reward
                    info['dist_reward'] = dist_reward
                    self._ep_dist_reward_sum += dist_reward
                if (self._threading_bonus_coef > 0.0
                        and cur_adaptive_thresh is not None
                        and cur_adaptive_dist < cur_adaptive_thresh):
                    threading_bonus = self._threading_bonus_coef
                    reward = float(reward) + threading_bonus
                    info['threading_bonus'] = threading_bonus
                    self._ep_threading_bonus_sum += threading_bonus
                    self._ep_threading_steps += 1
                # Always expose the per-step adaptive_dist for diagnostics
                # even when no shaping fires from it. Lets eval_reward_decomp
                # plot the cloth-to-hole distance trajectory directly.
                info['adaptive_dist_step'] = cur_adaptive_dist
                if cur_adaptive_thresh is not None:
                    info['adaptive_thresh_step'] = cur_adaptive_thresh

        # ------------------------------------------------------------------
        # (3) Pre-settle distance penalty (terminal only).
        #     Linear shaping on the hole-to-goal distance at policy
        #     handoff (BEFORE make_final_steps drops the cloth under
        #     gravity). Counters the "lift high, drop straight down"
        #     exploit by rewarding only positions reached via control.
        # ------------------------------------------------------------------
        pre_settle_pen = 0.0
        if (done and self._pre_settle_coef > 0.0
                and 'pre_settle_dist_m' in info):
            pre_dist = float(info['pre_settle_dist_m'])
            pre_settle_pen = self._pre_settle_coef * pre_dist
            reward = float(reward) - pre_settle_pen
            info['pre_settle_penalty'] = pre_settle_pen

        # ------------------------------------------------------------------
        # (4) Adaptive success override + terminal shaping.
        # ------------------------------------------------------------------
        adaptive_dist = None
        adaptive_thresh = None
        adaptive_is_success = None
        terminal_shaping = 0.0

        if (done and 'is_success' in info
                and self._success_factor is not None
                and self._hole_radius is not None):
            # Reuse the per-step computation if it ran this step, else
            # compute fresh. Cuts a redundant get_mesh_data call when
            # both per-step shaping and terminal shaping are active.
            if cur_adaptive_dist is not None:
                adaptive_dist = cur_adaptive_dist
                adaptive_thresh = cur_adaptive_thresh
            else:
                adaptive_dist, adaptive_thresh = (
                    self._compute_adaptive_dist())
            if adaptive_dist is not None and adaptive_thresh is not None:
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
        # (5) Update per-episode accumulators and emit diagnostics on done.
        # ------------------------------------------------------------------
        self._ep_step_count += 1
        self._ep_base_reward_sum += base_reward
        self._ep_vel_penalty_sum += vel_pen
        self._ep_action_penalty_sum += act_pen

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
                pre_settle_pen=pre_settle_pen,
                dist_reward_sum=self._ep_dist_reward_sum,
                threading_bonus_sum=self._ep_threading_bonus_sum,
                threading_steps=self._ep_threading_steps,
            )

        return obs, reward, done, info

    # ----------------------------------------------------------------------
    # Diagnostics emission. Keys live under 'rwd_diag/<group>/<metric>' and
    # are drained by RewardDiagnosticsCallback into TB/wandb. Only emitted
    # at terminal step so the callback can compute per-episode statistics.
    # ----------------------------------------------------------------------
    def _emit_episode_diagnostics(self, info, *,
                                  terminal_base, terminal_shaping,
                                  total_reward,
                                  base_is_success, adaptive_is_success,
                                  adaptive_dist, adaptive_thresh,
                                  pre_settle_pen=0.0,
                                  dist_reward_sum=0.0,
                                  threading_bonus_sum=0.0,
                                  threading_steps=0):
        info['rwd_diag/reward/episode_total'] = float(
            self._ep_base_reward_sum
            - self._ep_vel_penalty_sum
            - self._ep_action_penalty_sum
            + dist_reward_sum
            + threading_bonus_sum
            + terminal_shaping
            - pre_settle_pen
        )
        info['rwd_diag/reward/dist_reward_sum'] = float(dist_reward_sum)
        info['rwd_diag/reward/threading_bonus_sum'] = float(
            threading_bonus_sum)
        info['rwd_diag/task/threading_steps'] = int(threading_steps)
        info['rwd_diag/task/threading_fraction'] = (
            float(threading_steps) / max(self._ep_step_count, 1))
        info['rwd_diag/reward/base_sum'] = float(self._ep_base_reward_sum)
        info['rwd_diag/reward/vel_penalty_sum'] = float(
            self._ep_vel_penalty_sum)
        info['rwd_diag/reward/action_penalty_sum'] = float(
            self._ep_action_penalty_sum)
        info['rwd_diag/reward/terminal_base'] = float(terminal_base)
        info['rwd_diag/reward/terminal_shaping'] = float(terminal_shaping)
        info['rwd_diag/reward/pre_settle_penalty'] = float(pre_settle_pen)
        info['rwd_diag/reward/episode_length'] = int(self._ep_step_count)

        # Successes: which definition each run uses, and disagreement rate.
        if base_is_success is not None:
            info['rwd_diag/success/base_rate'] = int(bool(base_is_success))
        if adaptive_is_success is not None:
            info['rwd_diag/success/adaptive_rate'] = int(
                bool(adaptive_is_success))
        if base_is_success is not None and adaptive_is_success is not None:
            info['rwd_diag/success/disagree_rate'] = int(
                bool(base_is_success) != bool(adaptive_is_success))
        # The success label the agent was actually trained against.
        active = (adaptive_is_success
                  if adaptive_is_success is not None else base_is_success)
        if active is not None:
            info['rwd_diag/success/active_rate'] = int(bool(active))

        # Task geometry (only meaningful when adaptive computation ran).
        if adaptive_dist is not None:
            info['rwd_diag/task/adaptive_dist'] = adaptive_dist
        if adaptive_thresh is not None:
            info['rwd_diag/task/adaptive_thresh'] = adaptive_thresh
        if self._hole_radius is not None:
            info['rwd_diag/task/hole_radius'] = float(self._hole_radius)

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
