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
                 boundary_penalty: float = 0.0,
                 z_overshoot_penalty: float = 0.0,
                 z_overshoot_slack: float = 1.0):
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

        z_overshoot_penalty: per-step penalty proportional to how far
        the cloth's hole centroid is ABOVE the peg's z-coordinate
        (= goal_pos[0][2]), beyond a slack zone. 0 = off (default).
        Why this exists: dedo's base reward measures full-3D distance
        ||hole_centroid - goal_pos|| but is orientation-blind — it
        rewards getting the centroid coordinate near the peg from any
        direction, including straight above. Together with the
        anchor-release final reward where gravity sometimes drops the
        cloth onto the peg, this creates a "park the hole high above
        the peg" local minimum that the boundary_penalty doesn't fix
        (the policy can lift just below gripper_lims and still over-
        shoot the peg by several meters). This shaping kills the lift
        directly: per-step `reward -= z_overshoot_penalty *
        max(0, hole_z - (peg_z + z_overshoot_slack))`. Suggested 1-5.

        z_overshoot_slack: meters above the peg z that count as the
        "free zone" with no overshoot penalty. The cloth starts hanging
        from the grippers a few meters above the peg, and threading
        approaches the peg from above, so a non-trivial slack avoids
        penalizing the natural approach. Default 1.0 m. Tighten to
        0.3-0.5 if you want more aggressive shaping; loosen to 2-3 if
        the cloth is noticeably long.

        boundary_penalty: terminal penalty applied IFF the episode ended
        via dedo's workspace-bound exit (gripper position past
        gripper_lims) rather than natural max_episode_len timeout. 0 = off.
        Why this exists: dedo's `step()` sets done=True both for natural
        timeout AND for workspace-bound exit, then amplifies the
        terminal-step reward by `(max_episode_len - stepnum)` only on
        the boundary path. PPO can exploit this by learning a
        "lift the cloth high" policy that triggers the workspace exit,
        cutting the episode short and avoiding the per-step distance
        penalty that would accumulate over the remaining steps. The
        cloth never actually threads — it just gets parked above the
        peg. Adding a fixed penalty on the boundary-exit terminal makes
        that strategy strictly worse than running to natural timeout.
        Suggested 100-400; the magnitude must dominate the marginal
        benefit of cutting short ~150 dithering steps at ~-0.25/step.
        Detected post-hoc via the underlying env's stepnum: any done
        with `stepnum <= max_episode_len` after step() returns is the
        boundary exit (natural timeout exits with stepnum =
        max_episode_len + 1).
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
        self._boundary_penalty = float(boundary_penalty)
        self._z_overshoot_penalty = float(z_overshoot_penalty)
        self._z_overshoot_slack = float(z_overshoot_slack)
        self._prev_verts = None
        self._hole_radius = None

        # Per-episode reward bookkeeping for the diagnostics callback.
        self._ep_step_count = 0
        self._ep_base_reward_sum = 0.0
        self._ep_vel_penalty_sum = 0.0  # accumulates magnitude (>= 0)
        self._ep_z_overshoot_pen_sum = 0.0  # accumulates magnitude (>= 0)
        self._ep_z_overshoot_sum = 0.0      # accumulates raw overshoot in m

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
        self._ep_z_overshoot_pen_sum = 0.0
        self._ep_z_overshoot_sum = 0.0
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
        # (1) Cloth-velocity penalty (non-terminal only).
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
                    reward = base_reward - vel_pen
                    info['vel_penalty'] = vel_pen
                    info['cloth_mean_speed'] = mean_speed
            self._prev_verts = verts_now

        # ------------------------------------------------------------------
        # (1b) Cloth-z overshoot penalty (non-terminal only).
        #      Targets the orientation-blind reward landscape: dedo's base
        #      reward only measures ||hole_centroid - goal_pos||, which is
        #      minimized just as well by hovering the hole high above the
        #      peg as by threading. The boundary penalty closes the
        #      workspace-exit hack but doesn't touch sub-boundary lifts.
        #      This term penalizes the lift directly: per-step loss
        #      proportional to (hole_centroid_z - peg_z) above a slack
        #      zone. Skipped on terminal step so it doesn't double-count
        #      against the dedo settle that has already been integrated
        #      into the terminal reward.
        # ------------------------------------------------------------------
        z_pen = 0.0
        z_overshoot = 0.0
        if self._z_overshoot_penalty > 0.0 and not done:
            _, verts_z = get_mesh_data(self.env.sim, self.env.deform_id)
            verts_z = np.asarray(verts_z, dtype=np.float32)
            if self._hole_vertex_indices:
                hv_z = verts_z[self._hole_vertex_indices]
                hv_z = hv_z[~np.isnan(hv_z).any(axis=1)]
                if len(hv_z) > 0:
                    hole_z = float(hv_z[:, 2].mean())
                    goal_z = float(self.env.goal_pos[0][2])
                    z_overshoot = max(
                        0.0, hole_z - (goal_z + self._z_overshoot_slack))
                    if z_overshoot > 0.0:
                        z_pen = self._z_overshoot_penalty * z_overshoot
                        reward = float(reward) - z_pen
                        info['z_overshoot'] = z_overshoot
                        info['z_overshoot_penalty'] = z_pen

        # ------------------------------------------------------------------
        # (2) Adaptive success override + terminal shaping.
        # ------------------------------------------------------------------
        adaptive_dist = None
        adaptive_thresh = None
        adaptive_is_success = None
        terminal_shaping = 0.0

        if (done and 'is_success' in info
                and self._success_factor is not None
                and self._hole_radius is not None):
            _, verts = get_mesh_data(self.env.sim, self.env.deform_id)
            verts = np.asarray(verts, dtype=np.float32)
            hv = (verts[self._hole_vertex_indices]
                  if self._hole_vertex_indices else np.zeros((0, 3)))
            hv = hv[~np.isnan(hv).any(axis=1)]
            if len(hv) > 0:
                centroid = hv.mean(axis=0)
                goal = np.asarray(self.env.goal_pos[0], dtype=np.float32)
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
        # (3) Workspace-exit early-termination penalty.
        #     dedo's `step()` sets done=True both for natural timeout
        #     (stepnum >= max_episode_len) AND for workspace-bound
        #     violation (gripper past gripper_lims). Without
        #     intervention, the latter is exploitable: a "lift the cloth
        #     high" policy triggers the bound, ending the episode early
        #     and skipping the per-step distance penalty that would
        #     accumulate over the remaining ~150 dithering steps.
        #
        #     Detection: dedo increments stepnum at the END of step(),
        #     so after the call returns, natural timeout exits with
        #     stepnum = max_episode_len + 1 while a boundary exit at
        #     internal step k exits with stepnum = k + 1
        #     (<= max_episode_len). Boundary exits are tagged regardless
        #     of self._boundary_penalty so the violation rate is
        #     observable in diagnostics even with the penalty disabled.
        # ------------------------------------------------------------------
        boundary_violation = False
        boundary_pen_applied = 0.0
        if done and self.env.stepnum <= self.env.max_episode_len:
            boundary_violation = True
            info['boundary_violation'] = True
            if self._boundary_penalty != 0.0:
                boundary_pen_applied = self._boundary_penalty
                reward = float(reward) - boundary_pen_applied
                info['boundary_penalty'] = boundary_pen_applied

        # ------------------------------------------------------------------
        # (4) Update per-episode accumulators and emit diagnostics on done.
        # ------------------------------------------------------------------
        self._ep_step_count += 1
        self._ep_base_reward_sum += base_reward
        self._ep_vel_penalty_sum += vel_pen
        self._ep_z_overshoot_pen_sum += z_pen
        self._ep_z_overshoot_sum += z_overshoot

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
                boundary_violation=boundary_violation,
                boundary_pen_applied=boundary_pen_applied,
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
                                  boundary_violation=False,
                                  boundary_pen_applied=0.0):
        info['rwd_diag/reward/episode_total'] = float(
            self._ep_base_reward_sum
            - self._ep_vel_penalty_sum
            - self._ep_z_overshoot_pen_sum
            + terminal_shaping
            - boundary_pen_applied
        )
        info['rwd_diag/reward/base_sum'] = float(self._ep_base_reward_sum)
        info['rwd_diag/reward/vel_penalty_sum'] = float(
            self._ep_vel_penalty_sum)
        info['rwd_diag/reward/z_overshoot_penalty_sum'] = float(
            self._ep_z_overshoot_pen_sum)
        info['rwd_diag/reward/terminal_base'] = float(terminal_base)
        info['rwd_diag/reward/terminal_shaping'] = float(terminal_shaping)
        info['rwd_diag/reward/boundary_penalty'] = float(boundary_pen_applied)
        info['rwd_diag/reward/episode_length'] = int(self._ep_step_count)
        # 0/1 per-episode flag → rolling mean = workspace-exit rate.
        info['rwd_diag/boundary/violation_rate'] = int(bool(boundary_violation))
        # Mean per-step raw overshoot in meters: a behavior signal that
        # works even when penalty=0 (so users can decide whether to turn
        # the knob on by inspecting an unshaped baseline).
        ep_len = max(self._ep_step_count, 1)
        info['rwd_diag/task/z_overshoot_mean'] = float(
            self._ep_z_overshoot_sum / ep_len)

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
