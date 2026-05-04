"""
Shared, side-effect-free utilities for hang_obs_exp scripts.

This module exists so record_demo.py / view_demo.py / train_privileged.py
can share helpers WITHOUT importing each other (they are scripts that run
training at import time, which would re-trigger pipelines unintentionally).
"""
import gym
import numpy as np
import pybullet


# ---------------------------------------------------------------------------
# Retry wrapper: dedo's procedural cloth generator occasionally produces a
# mesh that pybullet's loadSoftBody refuses (raises pybullet.error). reset()
# starts with reset_bullet() which clears sim state, so a retry is safe and
# will resample a new cloth.
# ---------------------------------------------------------------------------
class RetryResetEnv(gym.Wrapper):
    def __init__(self, env, max_retries=20):
        super().__init__(env)
        self._max_retries = max_retries

    def reset(self, **kwargs):
        last_err = None
        for attempt in range(self._max_retries):
            try:
                return self.env.reset(**kwargs)
            except pybullet.error as e:
                last_err = e
                print(f'[RetryReset] attempt {attempt+1}/{self._max_retries}: '
                      f'{e!r} — resampling cloth')
        raise last_err

    def step(self, action):
        try:
            return self.env.step(action)
        except pybullet.error as e:
            print(f'[RetryReset] pybullet.error during step: {e!r} — '
                  f'forcing episode end')
            obs = self.reset()
            return obs, 0.0, True, {'pybullet_step_error': True}


# ---------------------------------------------------------------------------
# Per-episode hole-aware waypoint builder. Reads privileged sim state
# (hole centroid, hanger goal, gripper init) and returns a {'a': ..., 'b': ...}
# dict consumable by dedo.demo_preset.build_traj.
# ---------------------------------------------------------------------------
def build_hole_aware_waypoints(underlying):
    from dedo.utils.mesh_utils import get_mesh_data

    if not hasattr(underlying.args, 'deform_true_loop_vertices'):
        return None
    loops = underlying.args.deform_true_loop_vertices
    idxs = [i for loop in loops for i in loop]
    if len(idxs) == 0:
        return None

    _, verts = get_mesh_data(underlying.sim, underlying.deform_id)
    verts = np.array(verts, dtype=np.float32)
    hole_verts = verts[idxs]
    hole_verts = hole_verts[~np.isnan(hole_verts).any(axis=1)]
    if len(hole_verts) == 0:
        return None
    hole_centroid = hole_verts.mean(axis=0)

    hanger = np.array(underlying.goal_pos[0], dtype=np.float32)

    anc_ids = list(underlying.anchors.keys())
    grip_a = np.array(underlying.anchors[anc_ids[0]]['pos'], dtype=np.float32)
    grip_b = np.array(underlying.anchors[anc_ids[1]]['pos'], dtype=np.float32)

    offset_xy = hanger[:2] - hole_centroid[:2]

    z_hover = hanger[2] + 2.0
    z_thread = hanger[2] + 0.3
    z_settle = hanger[2] - 0.8

    wp_a = [
        [grip_a[0] + offset_xy[0], grip_a[1] + offset_xy[1], z_hover, 1.5],
        [grip_a[0] + offset_xy[0], grip_a[1] + offset_xy[1], z_thread, 1.0],
        [grip_a[0] + offset_xy[0], grip_a[1] + offset_xy[1], z_settle, 0.5],
    ]
    wp_b = [
        [grip_b[0] + offset_xy[0], grip_b[1] + offset_xy[1], z_hover, 1.5],
        [grip_b[0] + offset_xy[0], grip_b[1] + offset_xy[1], z_thread, 1.0],
        [grip_b[0] + offset_xy[0], grip_b[1] + offset_xy[1], z_settle, 0.5],
    ]
    return {'a': wp_a, 'b': wp_b}
