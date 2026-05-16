"""
Shared obs-capture + success-check helpers for the BC collection /
training scripts.

Lives in its own module (not collect_bc_demos.py) so train_diffusion_bc.py
can import these without re-triggering the collector's argparse + main
loop at module-import time.

Camera capture: pybullet.getCameraImage is called directly with explicit
width/height — we don't rely on `dedo_env.args.cam_resolution`, so the
same code path runs at collection AND eval time, producing bit-identical
RGB/PCD obs for the policy.

Success checks: three metrics — hanging (3D, robust to collapsed cloth;
the recommended default), topological (winding-number around peg axis),
legacy (centroid 3D distance). Each is a standalone function that reads
the underlying DeformEnv at the current sim state; all three are
recorded into every demo pkl so the training script can filter by any.
"""
from __future__ import annotations

import numpy as np
import pybullet

from dedo.envs.deform_env import DeformEnv
from dedo.utils.mesh_utils import get_mesh_data


_PCD_NEAR = 0.1
_PCD_FAR = 30.0


# ---------------------------------------------------------------------------
# Camera helpers.
# ---------------------------------------------------------------------------
def proj_matrix(near=_PCD_NEAR, far=_PCD_FAR, fov=60.0, aspect=1.0):
    return pybullet.computeProjectionMatrixFOV(
        fov=fov, aspect=aspect, nearVal=near, farVal=far)


def patch_deform_render_to_obs_camera(deform_env):
    """Replace `deform_env.render()` so captured frames use the SAME
    projection as the obs camera (fov=60, near=0.1, far=30 — i.e.
    `proj_matrix()`). The view matrix already comes from
    `deform_env._cam_viewmat` (built from `args.cam_viewmat`) and is
    unchanged.

    Why this exists: without the patch, `deform_env.render()` uses dedo's
    `DEFAULT_CAM_PROJECTION` (fov≈90), so any saved video would have a
    visibly wider field of view than the policy's actual obs camera. The
    policy was trained on fov=60 RGB / depth, so videos should look the
    same. This call site also covers `make_final_steps` — pybullet's
    settle-phase frames go through `self.render()` too, so the post-
    policy gravity-drape phase ends up consistent with the policy phase.

    Apply this UNCONDITIONALLY at env construction (right after
    `gym.make`) — it's idempotent, cheap (replaces one method
    reference), and makes any future render call in this env safe by
    default. Gating it on a flag like `record_videos > 0` makes the
    invariant fragile: anyone who adds a debug `deform.render()` later
    silently gets the wrong fov.

    Returns the patched method (handy if a caller wants to hold onto a
    direct reference, e.g. for a 3-panel video builder).
    """
    import types as _types
    matched_proj = proj_matrix()

    def _matched_render(self, mode='rgb_array', width=300, height=300):
        assert mode == 'rgb_array'
        _, _, rgba, _, _ = self.sim.getCameraImage(
            width=width, height=height,
            renderer=pybullet.ER_BULLET_HARDWARE_OPENGL,
            viewMatrix=self._cam_viewmat,
            projectionMatrix=matched_proj)
        return np.asarray(rgba)[:, :, :3]

    deform_env.render = _types.MethodType(_matched_render, deform_env)
    return deform_env.render


def capture_rgb_depth(deform_env, width, height):
    """Render RGB + depth (+ seg mask) using deform_env's cached cam_viewmat.

    Returns (rgb uint8 (H,W,3), depth float64 (H,W) ∈ [0,1),
             seg int32 (H,W), view, proj).

    `seg[i, j]` is pybullet's per-pixel object ID. -1 = background (e.g.
    sky, no hit). For dedo HangProcCloth-v1, valid IDs include the
    deform (cloth) body, the procedural peg/pole/flag rigid bodies, and
    the anchors. The deform's ID is `deform_env.deform_id` — filter the
    seg mask to only those pixels to get a cloth-only depth buffer
    suitable for back-projection.
    """
    view = deform_env._cam_viewmat
    proj = proj_matrix()
    _, _, rgb_raw, depth_buf, seg_raw = deform_env.sim.getCameraImage(
        width=width, height=height,
        viewMatrix=view, projectionMatrix=proj,
        renderer=pybullet.ER_BULLET_HARDWARE_OPENGL)
    rgb = np.asarray(rgb_raw, dtype=np.uint8).reshape(height, width, 4)[:, :, :3]
    depth = np.asarray(depth_buf, dtype=np.float64).reshape(height, width)
    seg = np.asarray(seg_raw, dtype=np.int32).reshape(height, width)
    return rgb, depth, seg, view, proj


def depth_to_pcd(depth, view, proj, n_points, mask=None):
    """Back-project a non-linear depth buffer to world coords, mask out
    background (depth ≈ 1.0) and NaN, then sub-/over-sample to exactly
    n_points. Returns (n_points, 3) float32 in WORLD meters.

    If `mask` is provided (boolean array of depth's shape), restrict the
    back-projection to pixels where `mask` is True (use e.g. a seg-mask
    filter to keep only cloth points and drop the peg/pole/flag).
    """
    H, W = depth.shape
    v = np.asarray(view, dtype=np.float64).reshape(4, 4, order='F')
    p = np.asarray(proj, dtype=np.float64).reshape(4, 4, order='F')
    try:
        inv_vp = np.linalg.inv(p @ v)
    except np.linalg.LinAlgError:
        return np.zeros((n_points, 3), dtype=np.float32)
    ys, xs = np.meshgrid(np.arange(H), np.arange(W), indexing='ij')
    u = (xs.astype(np.float64) + 0.5) / W * 2.0 - 1.0
    vv = 1.0 - (ys.astype(np.float64) + 0.5) / H * 2.0
    z = depth * 2.0 - 1.0
    clip = np.stack([u, vv, z, np.ones_like(z)], axis=-1).reshape(-1, 4)
    world_h = clip @ inv_vp.T
    world = world_h[:, :3] / world_h[:, 3:4]
    valid = (depth.reshape(-1) < 0.999) & ~np.isnan(world).any(axis=1)
    if mask is not None:
        valid = valid & np.asarray(mask).reshape(-1).astype(bool)
    pts = world[valid]
    n = len(pts)
    if n == 0:
        return np.zeros((n_points, 3), dtype=np.float32)
    idx = (np.random.choice(n, n_points, replace=False) if n >= n_points
           else np.random.choice(n, n_points, replace=True))
    return pts[idx].astype(np.float32)


def cloth_only_pcd(depth, seg, view, proj, deform_id, n_points):
    """Back-project depth to world points keeping ONLY pixels whose
    segmentation ID matches the cloth body. Filters out peg / pole /
    flag / anchors / background. Returns (n_points, 3) float32."""
    mask = (seg == int(deform_id))
    return depth_to_pcd(depth, view, proj, n_points, mask=mask)


# ---------------------------------------------------------------------------
# Hole bookkeeping (mirrors PrivilegedObsWrapper).
# ---------------------------------------------------------------------------
def get_hole_indices(deform_env):
    if hasattr(deform_env.args, 'deform_true_loop_vertices'):
        loops = deform_env.args.deform_true_loop_vertices
        return [idx for loop in loops for idx in loop]
    return []


def get_hole_loops(deform_env):
    if hasattr(deform_env.args, 'deform_true_loop_vertices'):
        return [list(loop) for loop in
                deform_env.args.deform_true_loop_vertices]
    return []


def measure_hole_radius(deform_env, hole_vertex_indices):
    if not hole_vertex_indices:
        return 0.0
    _, verts = get_mesh_data(deform_env.sim, deform_env.deform_id)
    verts = np.asarray(verts, dtype=np.float32)
    hv = verts[hole_vertex_indices]
    hv = hv[~np.isnan(hv).any(axis=1)]
    if len(hv) == 0:
        return 0.0
    c = hv.mean(axis=0)
    return float(np.mean(np.linalg.norm(hv - c, axis=1)))


# ---------------------------------------------------------------------------
# Three success metrics. Each takes the current sim state and returns a
# bool (and optionally a diagnostic scalar).
# ---------------------------------------------------------------------------
def check_threaded_topological(deform_env, hole_loops):
    """Signed-winding-number check; threaded iff |w| >= 0.5 around peg."""
    if not hole_loops:
        return False, 0.0
    _, verts = get_mesh_data(deform_env.sim, deform_env.deform_id)
    verts = np.asarray(verts, dtype=np.float32)
    peg_xy = np.asarray(deform_env.goal_pos[0], dtype=np.float32)[:2]
    max_abs_winding = 0.0
    for loop in hole_loops:
        if len(loop) < 3:
            continue
        lv = verts[loop]
        mask = ~np.isnan(lv).any(axis=1)
        if mask.sum() < 3:
            continue
        xy = lv[mask][:, :2]
        rel = xy - peg_xy
        r = np.linalg.norm(rel, axis=1)
        if (r < 1e-6).any():
            continue
        angles = np.arctan2(rel[:, 1], rel[:, 0])
        ext = np.concatenate([angles, angles[:1]])
        diffs = np.diff(ext)
        diffs = (diffs + np.pi) % (2 * np.pi) - np.pi
        w = float(diffs.sum()) / (2 * np.pi)
        if abs(w) > max_abs_winding:
            max_abs_winding = abs(w)
    return max_abs_winding >= 0.5, max_abs_winding


def check_hanging_on_peg(deform_env, hole_vertex_indices,
                         hole_radius, success_factor):
    """3D-robust hanging check: lateral xy alignment + vertex descended
    past peg tip + nonzero vertical extent. Robust to collapsed-cloth
    degeneracy that breaks winding-number."""
    if not hole_vertex_indices or hole_radius is None or hole_radius == 0:
        return False
    _, verts = get_mesh_data(deform_env.sim, deform_env.deform_id)
    verts = np.asarray(verts, dtype=np.float32)
    hv = verts[hole_vertex_indices]
    hv = hv[~np.isnan(hv).any(axis=1)]
    if len(hv) < 3:
        return False
    centroid = hv.mean(axis=0)
    peg = np.asarray(deform_env.goal_pos[0], dtype=np.float32)
    lat_dist = float(np.linalg.norm(centroid[:2] - peg[:2]))
    lat_close = lat_dist < hole_radius * success_factor
    descended = float(hv[:, 2].min()) < peg[2]
    z_range = float(hv[:, 2].max() - hv[:, 2].min())
    has_extent = z_range > (0.5 * hole_radius)
    return bool(lat_close and descended and has_extent)


def check_legacy(deform_env, hole_vertex_indices,
                 hole_radius, success_factor):
    """Centroid 3D distance < success_factor * hole_radius."""
    if not hole_vertex_indices or hole_radius is None or hole_radius == 0:
        return False
    _, verts = get_mesh_data(deform_env.sim, deform_env.deform_id)
    verts = np.asarray(verts, dtype=np.float32)
    hv = verts[hole_vertex_indices]
    hv = hv[~np.isnan(hv).any(axis=1)]
    if len(hv) == 0:
        return False
    centroid = hv.mean(axis=0)
    goal = np.asarray(deform_env.goal_pos[0], dtype=np.float32)
    return float(np.linalg.norm(centroid - goal)) < hole_radius * success_factor


# ---------------------------------------------------------------------------
# Walk wrapper chain down to the underlying DeformEnv.
# ---------------------------------------------------------------------------
def resolve_deform(env):
    e = env
    while hasattr(e, 'env') and not isinstance(e, DeformEnv):
        e = e.env
    return e
