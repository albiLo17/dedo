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

# Hanger URDF + globalScaling, used by the threading-success metric below.
# Must match SCENE_INFO['hangcloth']['entities']['urdf/hanger.urdf'] in
# dedo/utils/task_info.py — change both together.
_HANGER_GLOBAL_SCALING = 10.0


# ---------------------------------------------------------------------------
# Threading success metric: True iff a hanger rod pierces the cloth's hole
# loop. This is a topological linking test — robust to hole-centroid drift
# after settle and to "lift cloth high, drop it onto goal" exploits, which
# both leave the rod-vs-loop linking unchanged.
#
# Algorithm (per loop):
#   1. Fit a least-squares plane to the loop verts (SVD → centroid + normal).
#   2. For each hanger rod segment, find where (if at all) it crosses that
#      plane.
#   3. Project the crossing point and the loop into the plane's 2D basis,
#      then run point-in-polygon. Any rod inside any loop → success.
# Cloth holes from procedural_hang_cloth are small + near-planar after
# settle, so the planar-polygon approximation matches a Gauss linking
# integral in practice while staying O(loops * rods).
# ---------------------------------------------------------------------------
def _hanger_rod_segments_base_frame(scale=_HANGER_GLOBAL_SCALING):
    """Hanger rod segments in the hanger's BASE frame, derived from
    urdf/hanger.urdf. Each entry is a (2, 3) array [p0, p1]."""
    # rod_link_top: vertical cylinder length=0.05, origin offset (0,0,0.05).
    top = np.array([[0.0, 0.0, 0.025], [0.0, 0.0, 0.075]], dtype=np.float32)
    # rod_link_left: cylinder length=0.15, rpy=(0, 1.2566, 0), xyz=(-0.07, 0, 0).
    angle_l, half_l = 1.25663706144, 0.075
    cl, sl = np.cos(angle_l), np.sin(angle_l)
    left = np.array([
        [+sl * half_l - 0.07, 0.0, +cl * half_l],
        [-sl * half_l - 0.07, 0.0, -cl * half_l],
    ], dtype=np.float32)
    # rod_link_right: cylinder length=0.15, rpy=(0, 1.8849, 0), xyz=(0.07, 0, 0).
    angle_r, half_r = 1.88495559215, 0.075
    cr, sr = np.cos(angle_r), np.sin(angle_r)
    right = np.array([
        [+sr * half_r + 0.07, 0.0, +cr * half_r],
        [-sr * half_r + 0.07, 0.0, -cr * half_r],
    ], dtype=np.float32)
    return [scale * top, scale * left, scale * right]


def _segments_to_world(segments_base, base_pos, base_quat, sim):
    R = np.asarray(sim.getMatrixFromQuaternion(base_quat),
                   dtype=np.float32).reshape(3, 3)
    t = np.asarray(base_pos, dtype=np.float32)
    return [(R @ seg[0] + t, R @ seg[1] + t) for seg in segments_base]


def _point_in_polygon_2d(pt, poly):
    """Ray-cast point-in-polygon. poly: (N, 2) ordered vertices, N >= 3."""
    n = len(poly)
    if n < 3:
        return False
    px, py = float(pt[0]), float(pt[1])
    inside = False
    j = n - 1
    for i in range(n):
        xi, yi = float(poly[i, 0]), float(poly[i, 1])
        xj, yj = float(poly[j, 0]), float(poly[j, 1])
        if (yi > py) != (yj > py):
            denom = yj - yi
            if abs(denom) > 1e-12:
                x_int = (xj - xi) * (py - yi) / denom + xi
                if px < x_int:
                    inside = not inside
        j = i
    return inside


def _segment_pierces_loop(p0, p1, hv):
    """One rod segment vs one hole loop. Returns True iff the segment
    crosses the loop's least-squares plane *inside* the loop polygon."""
    if len(hv) < 3:
        return False
    centroid = hv.mean(axis=0)
    centered = hv - centroid
    try:
        _, _, vt = np.linalg.svd(centered, full_matrices=False)
    except np.linalg.LinAlgError:
        return False
    normal, u, v = vt[2], vt[0], vt[1]
    poly_2d = np.stack([centered @ u, centered @ v], axis=1)
    # Sort verts by angle around centroid so the polygon is well-defined
    # even if deform_true_loop_vertices arrives unordered.
    angles = np.arctan2(poly_2d[:, 1], poly_2d[:, 0])
    poly_2d = poly_2d[np.argsort(angles)]

    d0 = float(np.dot(p0 - centroid, normal))
    d1 = float(np.dot(p1 - centroid, normal))
    if d0 * d1 > 0:
        return False  # both endpoints on the same side
    if abs(d0 - d1) < 1e-9:
        return False  # parallel to plane
    t = d0 / (d0 - d1)
    if not (0.0 <= t <= 1.0):
        return False
    x = p0 + t * (p1 - p0)
    rel = x - centroid
    return _point_in_polygon_2d(np.array([rel @ u, rel @ v]), poly_2d)


def is_cloth_threaded(deform_env, hole_loops, hanger_id,
                      scale=_HANGER_GLOBAL_SCALING):
    """True iff any hanger rod pierces any cloth-hole loop.

    hole_loops: list[list[int]] — vertex indices for each hole loop
        (i.e. args.deform_true_loop_vertices, kept per-loop, NOT
        flattened — multi-hole cloths thread independently).
    hanger_id: pybullet body id of the hanger URDF.
    """
    if not hole_loops or hanger_id is None:
        return False
    sim = deform_env.sim
    _, verts = get_mesh_data(sim, deform_env.deform_id)
    verts = np.asarray(verts, dtype=np.float32)
    base_pos, base_quat = sim.getBasePositionAndOrientation(hanger_id)
    segs_world = _segments_to_world(
        _hanger_rod_segments_base_frame(scale=scale),
        base_pos, base_quat, sim)
    for loop in hole_loops:
        if len(loop) < 3:
            continue
        hv = verts[loop]
        hv = hv[~np.isnan(hv).any(axis=1)]
        for p0, p1 in segs_world:
            if _segment_pierces_loop(p0, p1, hv):
                return True
    return False


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
    SUCCESS_METRICS = ('distance', 'threading')

    def __init__(self, env, obs_mode: str = 'hole_centroid',
                 success_metric: str = 'distance',
                 success_factor: float = None,
                 success_bonus: float = 0.0,
                 fail_penalty: float = 0.0,
                 vel_penalty: float = 0.0,
                 pre_settle_coef: float = 0.0,
                 action_penalty: float = 0.0):
        """
        success_metric: 'distance' (default; current behavior — uses
        hole-centroid → goal_pos distance, optionally overridden with a
        radius-proportional adaptive threshold via success_factor) or
        'threading' (uses a geometric linking test — True iff a hanger
        rod passes through the cloth-hole loop). Threading is robust to
        post-settle centroid drift and to lift-and-drop exploits, since
        both leave the rod-vs-loop topology unchanged. With 'threading',
        success_factor is ignored.

        success_factor: if not None, override the env's fixed success
        threshold (0.125 m) with an ADAPTIVE one — success requires
        hole-centroid-to-goal distance < success_factor * hole_radius,
        where hole_radius is the mean distance from the hole centroid to
        the hole-loop vertices, measured at reset. ~0.6-1.0 is reasonable;
        smaller is stricter. None keeps dedo's fixed criterion unchanged.
        Only consulted when success_metric == 'distance'.

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
        assert success_metric in self.SUCCESS_METRICS, \
            f'success_metric must be one of {self.SUCCESS_METRICS}'
        env.args.cam_resolution = 0
        super().__init__(env)

        self.obs_mode = obs_mode
        self._success_metric = success_metric
        self._hole_loops = []           # list[list[int]] — per-loop indices
        self._hole_vertex_indices = []  # flat concatenation, used by obs builders
        self._hanger_id = None
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

    def _get_hole_loops(self):
        """Per-loop vertex indices. Threading test needs the loops kept
        separate (multi-hole cloths thread independently); the obs builders
        only want a flat concatenation."""
        if hasattr(self.env.args, 'deform_true_loop_vertices'):
            return [list(loop) for loop in
                    self.env.args.deform_true_loop_vertices]
        return []

    def _get_hanger_id(self):
        """SCENE_INFO['hangcloth']['entities'] iterates {hanger.urdf,
        tallrod.urdf} in insertion order, so DeformEnv.rigid_ids[0] is
        the hanger body. Used by the threading-success test."""
        rigid_ids = getattr(self.env, 'rigid_ids', None)
        if rigid_ids is not None and len(rigid_ids) > 0:
            return rigid_ids[0]
        return None

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
        self._hole_loops = self._get_hole_loops()
        self._hole_vertex_indices = [i for loop in self._hole_loops
                                     for i in loop]
        self._hanger_id = self._get_hanger_id()
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
        # Snapshot the env-side reward before any wrapper-side shaping is
        # applied, so downstream consumers (e.g. replay_checkpoint.py) can
        # plot each component of the reward separately. Sum of components
        # = info['base_reward'] - action_penalty - vel_penalty
        #                       - pre_settle_penalty + shaping_added.
        info['base_reward'] = float(reward)
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
        # Episode-end success metric & shaping. Two metrics are tracked:
        #   - threading: rod-pierces-loop linking test (topological).
        #   - distance:  hole-centroid → goal_pos distance, optionally
        #                radius-proportional via success_factor.
        # `is_success` is set to whichever the configured success_metric
        # selects; the other is logged alongside under its own key.
        if done:
            threading_success = None
            if self._hole_loops and self._hanger_id is not None:
                threading_success = is_cloth_threaded(
                    self.env, self._hole_loops, self._hanger_id)
                info['is_threaded'] = bool(threading_success)

            # info['is_success'] arrives from dedo as the fixed-threshold
            # distance test; with success_factor we recompute it adaptively.
            distance_success = info.get('is_success')
            if (self._success_factor is not None
                    and self._hole_radius is not None
                    and self._hole_vertex_indices):
                _, verts = get_mesh_data(self.env.sim, self.env.deform_id)
                verts = np.asarray(verts, dtype=np.float32)
                hv = verts[self._hole_vertex_indices]
                hv = hv[~np.isnan(hv).any(axis=1)]
                if len(hv) > 0:
                    centroid = hv.mean(axis=0)
                    goal = np.asarray(self.env.goal_pos[0], dtype=np.float32)
                    dist = float(np.linalg.norm(centroid - goal))
                    thresh = self._hole_radius * self._success_factor
                    distance_success = bool(dist < thresh)
                    info['adaptive_dist'] = dist
                    info['adaptive_thresh'] = thresh
                    info['hole_radius'] = self._hole_radius
            if distance_success is not None:
                info['is_distance_success'] = bool(distance_success)

            if (self._success_metric == 'threading'
                    and threading_success is not None):
                chosen = bool(threading_success)
            else:
                chosen = (bool(distance_success)
                          if distance_success is not None else False)
            info['is_success'] = chosen
            info['success_metric'] = self._success_metric

            # Shaping fires only when an adaptive metric is in effect — so
            # default behavior (no success_factor, distance mode) leaves
            # dedo's untouched is_success and emits no extra reward.
            shape_active = (self._success_metric == 'threading'
                            or self._success_factor is not None)
            if shape_active:
                shaping = 0.0
                if chosen and self._success_bonus != 0.0:
                    shaping += self._success_bonus
                elif (not chosen) and self._fail_penalty != 0.0:
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
