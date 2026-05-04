"""
PointCloudObsWrapper — replaces the env's obs with a fixed-size point cloud
captured from a depth render + back-projection.

Mirrors PrivilegedObsWrapper's adaptive success / reward shaping
(success_factor, success_bonus, fail_penalty) so the same metrics and
training pipeline can be reused.

The wrapper also overrides render() to overlay the captured PCD points
on the RGB frame, so eval-mode videos logged to wandb show what the
policy is actually seeing.

Obs layout:
    [12 gripper_pos_vel] + [n_points * 3 flat (x,y,z)]
"""
import numpy as np
import gym
import pybullet
from gym import spaces

from dedo.utils.mesh_utils import get_mesh_data
from dedo.envs.deform_env import DeformEnv

_GRIP_DIM = 12
_DEFAULT_N_POINTS = 512
_WBOX = 20.0     # workspace normalization (matches PrivilegedObsWrapper)
_PCD_NEAR = 0.1
_PCD_FAR = 30.0


def _proj_matrix(near=_PCD_NEAR, far=_PCD_FAR):
    return pybullet.computeProjectionMatrixFOV(
        fov=60.0, aspect=1.0, nearVal=near, farVal=far)


class PointCloudObsWrapper(gym.ObservationWrapper):
    """Replaces obs with a depth-derived point cloud."""

    def __init__(self, env,
                 n_points: int = _DEFAULT_N_POINTS,
                 cam_resolution: int = 128,
                 success_factor: float = None,
                 success_bonus: float = 0.0,
                 fail_penalty: float = 0.0):
        # PCD wrapper requires a working camera — force cam_resolution.
        env.args.cam_resolution = cam_resolution
        super().__init__(env)

        self._n_points = n_points
        self._cam_res_pcd = cam_resolution
        self._success_factor = success_factor
        self._success_bonus = float(success_bonus)
        self._fail_penalty = float(fail_penalty)

        self._hole_vertex_indices = []
        self._goal_pos = None
        self._hole_radius = None
        self._last_pcd_world = None  # cached for overlay viz

        # Resolve the actual DeformEnv once. Anything wrapped between us
        # and the inner env (e.g. RetryResetEnv) blocks attribute access
        # to underscore-prefixed members like `_cam_viewmat`.
        self._deform = env
        while hasattr(self._deform, 'env') and not isinstance(
                self._deform, DeformEnv):
            self._deform = self._deform.env

        obs_dim = _GRIP_DIM + n_points * 3
        self.observation_space = spaces.Box(
            low=-np.ones(obs_dim, dtype=np.float32) * 2.0,
            high=np.ones(obs_dim, dtype=np.float32) * 2.0,
            dtype=np.float32,
        )

    # ------------------------------------------------------------------
    # Hole / goal bookkeeping (mirrors PrivilegedObsWrapper).
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
    # PCD capture / projection helpers.
    # ------------------------------------------------------------------
    def _camera_matrices(self):
        view = self._deform._cam_viewmat   # 16-tuple (column-major)
        proj = _proj_matrix()
        return view, proj

    @staticmethod
    def _vp_inverse(view, proj):
        v = np.asarray(view, dtype=np.float64).reshape(4, 4, order='F')
        p = np.asarray(proj, dtype=np.float64).reshape(4, 4, order='F')
        return np.linalg.inv(p @ v), v, p

    def _capture_pcd(self):
        """Render depth at cam_res_pcd, back-project, return (N, 3) world."""
        cam_res = self._cam_res_pcd
        view, proj = self._camera_matrices()
        try:
            _, _, _, depth_buf, _ = self._deform.sim.getCameraImage(
                width=cam_res, height=cam_res,
                viewMatrix=view, projectionMatrix=proj,
                renderer=pybullet.ER_BULLET_HARDWARE_OPENGL)
        except Exception:
            self._last_pcd_world = np.zeros((self._n_points, 3),
                                             dtype=np.float32)
            return self._last_pcd_world

        depth = np.asarray(depth_buf, dtype=np.float64).reshape(cam_res, cam_res)

        try:
            inv_vp, _, _ = self._vp_inverse(view, proj)
        except np.linalg.LinAlgError:
            self._last_pcd_world = np.zeros((self._n_points, 3),
                                             dtype=np.float32)
            return self._last_pcd_world

        ys, xs = np.meshgrid(
            np.arange(cam_res), np.arange(cam_res), indexing='ij')
        u = (xs.astype(np.float64) + 0.5) / cam_res * 2.0 - 1.0
        v = 1.0 - (ys.astype(np.float64) + 0.5) / cam_res * 2.0
        z = depth * 2.0 - 1.0
        clip = np.stack([u, v, z, np.ones_like(z)], axis=-1).reshape(-1, 4)
        world_h = clip @ inv_vp.T
        world = world_h[:, :3] / world_h[:, 3:4]

        valid = (depth.reshape(-1) < 0.999) & ~np.isnan(world).any(axis=1)
        valid_pts = world[valid]

        n = len(valid_pts)
        if n == 0:
            sampled = np.zeros((self._n_points, 3), dtype=np.float32)
        elif n >= self._n_points:
            idx = np.random.choice(n, self._n_points, replace=False)
            sampled = valid_pts[idx].astype(np.float32)
        else:
            idx = np.random.choice(n, self._n_points, replace=True)
            sampled = valid_pts[idx].astype(np.float32)

        self._last_pcd_world = sampled.copy()
        return sampled

    # ------------------------------------------------------------------
    # Obs assembly.
    # ------------------------------------------------------------------
    def _build_obs(self):
        grip = np.asarray(self._deform.get_grip_obs(), dtype=np.float32)
        grip = np.clip(grip / _WBOX, -2.0, 2.0)
        pcd = self._capture_pcd() / _WBOX
        return np.clip(
            np.concatenate([grip, pcd.reshape(-1)]).astype(np.float32),
            -2.0, 2.0)

    def reset(self):
        self.env.reset()
        self._hole_vertex_indices = self._get_hole_indices()
        self._goal_pos = self._deform.goal_pos.copy()
        self._hole_radius = self._measure_hole_radius()
        return self._build_obs()

    def observation(self, obs):
        return self._build_obs()

    def step(self, action):
        _, reward, done, info = self.env.step(action)
        obs = self._build_obs()

        # Adaptive success + reward shaping (same logic as PrivilegedObsWrapper).
        if (done and 'is_success' in info
                and self._success_factor is not None
                and self._hole_radius is not None):
            _, verts = get_mesh_data(self._deform.sim, self._deform.deform_id)
            verts = np.asarray(verts, dtype=np.float32)
            hv = verts[self._hole_vertex_indices] \
                if self._hole_vertex_indices else np.zeros((0, 3))
            hv = hv[~np.isnan(hv).any(axis=1)]
            if len(hv) > 0:
                centroid = hv.mean(axis=0)
                goal = np.asarray(self._deform.goal_pos[0], dtype=np.float32)
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

    # ------------------------------------------------------------------
    # Render with PCD overlay (CustomCallback uses this for eval videos).
    # ------------------------------------------------------------------
    def render(self, mode='rgb_array', width=300, height=300):
        img = self._deform.render(mode=mode, width=width, height=height)
        if mode != 'rgb_array' or self._last_pcd_world is None:
            return img
        try:
            return self._overlay_pcd(img, width, height)
        except Exception:
            return img

    def _overlay_pcd(self, img, width, height):
        view, proj = self._camera_matrices()
        v = np.asarray(view, dtype=np.float64).reshape(4, 4, order='F')
        p = np.asarray(proj, dtype=np.float64).reshape(4, 4, order='F')
        pts = self._last_pcd_world
        pts_h = np.concatenate(
            [pts, np.ones((len(pts), 1), dtype=np.float64)], axis=1)
        clip = (p @ v @ pts_h.T).T
        ok = clip[:, 3] != 0.0
        clip = clip[ok]
        ndc = clip[:, :3] / clip[:, 3:4]
        u_px = ((ndc[:, 0] + 1.0) * 0.5 * width).astype(np.int32)
        v_px = ((1.0 - (ndc[:, 1] + 1.0) * 0.5) * height).astype(np.int32)
        in_view = ((u_px >= 0) & (u_px < width)
                   & (v_px >= 0) & (v_px < height)
                   & (ndc[:, 2] > -1) & (ndc[:, 2] < 1))
        u_px, v_px = u_px[in_view], v_px[in_view]
        out = img.copy()
        if out.dtype != np.uint8:
            out = (out * 255).astype(np.uint8) if out.max() <= 1.0 \
                else out.astype(np.uint8)
        # Magenta dots, 3x3 thick.
        for du in (-1, 0, 1):
            for dv in (-1, 0, 1):
                u2 = np.clip(u_px + du, 0, width - 1)
                v2 = np.clip(v_px + dv, 0, height - 1)
                out[v2, u2] = [255, 0, 255]
        return out

    # ------------------------------------------------------------------
    # Properties (parity with PrivilegedObsWrapper).
    # ------------------------------------------------------------------
    @property
    def hole_radius(self):
        return self._hole_radius

    @property
    def success_threshold_m(self):
        if self._success_factor is None:
            return 0.125
        if self._hole_radius is None:
            return float('nan')
        return self._hole_radius * self._success_factor
