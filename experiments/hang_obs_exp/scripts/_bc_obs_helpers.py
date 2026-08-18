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
# Renderer selection. ER_BULLET_HARDWARE_OPENGL needs an X display, which a
# headless VM does not have, so collection there must use the CPU rasterizer.
# That is safe for everything we train on: verified on HangProcClothReal-v1 at
# 384x384 that the two renderers agree BIT-EXACTLY on the depth buffer and the
# segmentation mask (max abs depth diff 0.0, identical masks, 3391/3391 cloth
# pixels). Depth is pure geometry - only shading differs, and no model consumes
# the RGB. Set DEDO_PYBULLET_RENDERER=tiny on headless machines.
def _renderer():
    import os
    if os.environ.get('DEDO_PYBULLET_RENDERER', '').lower().startswith('tiny'):
        return pybullet.ER_TINY_RENDERER
    return pybullet.ER_BULLET_HARDWARE_OPENGL


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
            renderer=_renderer(),
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
        renderer=_renderer())
    rgb = np.asarray(rgb_raw, dtype=np.uint8).reshape(height, width, 4)[:, :, :3]
    depth = np.asarray(depth_buf, dtype=np.float64).reshape(height, width)
    seg = np.asarray(seg_raw, dtype=np.int32).reshape(height, width)
    # Lens blocker, if one is configured. Applied HERE so every consumer —
    # pcd, rgb, the served state estimator — degrades from the same masked
    # image, rather than each call site having to remember to mask.
    rgb = np.ascontiguousarray(rgb)
    apply_image_occlusion(rgb, depth, seg)
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


# ---------------------------------------------------------------------------
# Image-space occlusion: a FIXED rectangle of the camera image is blocked, as
# if something sat in front of the lens.
#
# This is deliberately different from add_occluder() below, which puts a 3D box
# on the camera->hole ray and re-aims it at the cloth every step. That models a
# world obstacle and its occlusion level follows the cloth around. A lens
# blocker instead stays put in IMAGE coordinates, so how much of the cloth it
# hides depends on where the cloth happens to be — which is what a real camera
# with something in the way actually does.
#
# Applied by blanking the SEGMENTATION mask (to background) and the RGB inside
# the rectangle, so every downstream modality degrades consistently: cloth
# points inside it vanish from the point cloud, and the rgb encoder sees a dead
# patch. Depth is left alone; the seg mask is what gates the cloud.
#
# Privileged modes (state / mesh) read the simulator, not the camera, so they
# are untouched by construction — which is itself the useful contrast.
# ---------------------------------------------------------------------------
_IMG_OCC = {'frac': 0.0, 'side': 'bottom'}


def set_image_occlusion(frac=0.0, side='bottom'):
    """Set the blocked image fraction (0 = none) and which region."""
    _IMG_OCC['frac'] = float(max(0.0, min(1.0, frac)))
    _IMG_OCC['side'] = side


def get_image_occlusion():
    return dict(_IMG_OCC)


def image_occlusion_box(h, w, frac=None, side=None):
    """Pixel bounds (r0, r1, c0, c1) of the blocked rectangle, or None."""
    frac = _IMG_OCC['frac'] if frac is None else frac
    side = _IMG_OCC['side'] if side is None else side
    if frac <= 0.0:
        return None
    if side in ('bottom', 'top'):
        n = int(round(h * frac))
        return (h - n, h, 0, w) if side == 'bottom' else (0, n, 0, w)
    if side in ('left', 'right'):
        n = int(round(w * frac))
        return (0, h, 0, n) if side == 'left' else (0, h, w - n, w)
    if side == 'center':
        # centred square covering `frac` of the image AREA
        import math
        sh = int(round(h * math.sqrt(frac)))
        sw = int(round(w * math.sqrt(frac)))
        r0, c0 = (h - sh) // 2, (w - sw) // 2
        return (r0, r0 + sh, c0, c0 + sw)
    raise ValueError(f'unknown occlusion side {side!r}')


def apply_image_occlusion(rgb, depth, seg):
    """Blank the configured rectangle in place. Returns the blocked box."""
    box = image_occlusion_box(seg.shape[0], seg.shape[1])
    if box is None:
        return None
    r0, r1, c0, c1 = box
    seg[r0:r1, c0:c1] = -1          # background: drops these pixels from the cloud
    if rgb is not None:
        rgb[r0:r1, c0:c1] = 0
    return box


def cloth_only_pcd(depth, seg, view, proj, deform_id, n_points):
    """Back-project depth to world points keeping ONLY pixels whose
    segmentation ID matches the cloth body. Filters out peg / pole /
    flag / anchors / background. Returns (n_points, 3) float32."""
    mask = (seg == int(deform_id))
    return depth_to_pcd(depth, view, proj, n_points, mask=mask)


def add_occluder(deform_env, hole_vertex_indices, size, frac=0.45,
                 rgba=(0.15, 0.15, 0.18, 1.0)):
    """Put a VISUAL-ONLY blocker on the camera->hole ray. Returns a body id.

    Two properties make this the right occlusion knob:

      1. NO COLLISION SHAPE. The cloth can pass straight through it, so the
         physics and therefore the task are untouched - only what the sensors
         see changes. A colliding occluder would confound "harder to see" with
         "harder to do", and every curve would be uninterpretable.
      2. IT IS IN THE SCENE, not in a modality. RGB, depth, the segmentation
         mask and hence the point cloud all lose the same information, because
         they all come from the same render. Masking points (or pixels) would
         penalise one modality and not another, which is exactly the comparison
         we are trying to make.

    `size` is the cube half-extent; sweeping it sweeps how much of the hole is
    hidden. `frac` places it along the camera->hole segment (0.45 = closer to
    the camera, so a small object blocks a large solid angle).

    Remove it with `remove_occluder` at episode end - it is attached to the
    sim, not the env, so a reset does not necessarily clear it.
    """
    pos = _occluder_position(deform_env, hole_vertex_indices, frac)
    if pos is None:
        return None
    return _spawn_occluder(deform_env, pos, [size, size, size], rgba)


def _spawn_occluder(deform_env, pos, half_extents, rgba):
    sim = deform_env.sim
    vis = sim.createVisualShape(pybullet.GEOM_BOX,
                                halfExtents=list(half_extents), rgbaColor=rgba)
    return sim.createMultiBody(baseMass=0.0, baseCollisionShapeIndex=-1,
                               baseVisualShapeIndex=vis, basePosition=pos.tolist())


def add_static_occluder(deform_env, target_world, half_extents, frac=0.45,
                        rgba=(0.15, 0.15, 0.18, 1.0)):
    """A WORLD-FIXED visual obstacle: placed once, aimed at `target_world`,
    then never moved. Returns a body id.

    Different experiment from `add_occluder` + `move_occluder`, on purpose.
    The tracking occluder re-aims at the hole every step, which holds
    observability roughly CONSTANT — the right tool for sweeping a controlled
    visibility axis, but it means the cloth is hidden the same way for the whole
    episode and a belief never gets a moment where memory is worth more than the
    current frame.

    A static object in the scene is the realistic case: the cloth moves past it,
    so occlusion is a CONSEQUENCE of the task motion rather than a scripted
    condition. Visibility then varies within the episode all by itself, and the
    interesting question becomes whether an estimator can ride through the dips.

    Aim it at the GOAL PEG rather than the hole's start pose: the hang task
    lifts the cloth toward the peg, so a box on the camera->peg ray is clear
    early (cloth still low) and blocks exactly the threading phase, which is
    both the hardest part of the task and the part where a stale belief hurts
    most. Aiming at the hole's t=0 pose instead is what made the original
    non-tracking mode barely occlude at all (0.44 -> 0.39): the cloth simply
    climbs out from behind it.

    Still visual-only (no collision shape), still scene-level, so the physics
    and the task are untouched and every modality loses the same information.
    """
    target = np.asarray(target_world, dtype=np.float64).reshape(3)
    v = np.asarray(deform_env._cam_viewmat, dtype=np.float64).reshape(4, 4, order='F')
    cam_pos = -v[:3, :3].T @ v[:3, 3]
    pos = cam_pos + frac * (target - cam_pos)
    return _spawn_occluder(deform_env, pos, half_extents, rgba)


def _occluder_position(deform_env, hole_vertex_indices, frac):
    """World point `frac` of the way from the camera to the hole centroid."""
    idx = np.asarray(hole_vertex_indices, dtype=np.int64)
    _, verts = get_mesh_data(deform_env.sim, deform_env.deform_id)
    verts = np.asarray(verts, dtype=np.float64)
    idx = idx[idx < len(verts)]
    hv = verts[idx]
    hv = hv[np.isfinite(hv).all(axis=1)]
    if len(hv) == 0:
        return None
    target = hv.mean(axis=0)

    # Camera position from the cached viewmat: the view matrix maps world->cam,
    # so the camera origin in world coords is -R^T t.
    v = np.asarray(deform_env._cam_viewmat, dtype=np.float64).reshape(4, 4, order='F')
    cam_pos = -v[:3, :3].T @ v[:3, 3]
    return cam_pos + frac * (target - cam_pos)


def move_occluder(deform_env, body_id, hole_vertex_indices, frac=0.45):
    """Re-aim an existing occluder at the hole's CURRENT position.

    A box placed once at reset stops occluding: the hang task lifts the cloth
    a long way, so the hole climbs out from behind it and the measured
    visibility barely moves (0.44 -> 0.39 at half-extent 0.30, where a
    single-frame calibration predicted 0.50 -> 0.17). Re-aiming each step
    holds the observability level roughly constant for the whole episode,
    which is what a controlled sweep of the visibility axis needs.

    Still visual-only, still scene-level: the box has no collision shape, so
    the physics and the task are untouched and every modality loses the same
    information. Pass `--no_eval_occluder_track` for the alternative reading —
    a fixed obstacle in the world, which is realistic but sweeps a much
    narrower range of visibility.
    """
    if body_id is None:
        return
    pos = _occluder_position(deform_env, hole_vertex_indices, frac)
    if pos is None:
        return
    deform_env.sim.resetBasePositionAndOrientation(
        body_id, pos.tolist(), [0.0, 0.0, 0.0, 1.0])


def remove_occluder(deform_env, body_id):
    if body_id is not None:
        try:
            deform_env.sim.removeBody(body_id)
        except Exception:
            pass


def hole_visibility(deform_env, hole_vertex_indices, depth, seg, view, proj,
                    depth_tol=None, tol_frac=0.05):
    """Fraction of the hole loop the CAMERA can actually see, in [0, 1].

    This is the x-axis for any occlusion experiment. Reporting an occluder
    setting instead ("panel width 3") is not comparable across scenes, cameras
    or cloths; the fraction of task-critical structure actually observed is.
    It is also the same quantity the state-estimation arm sweeps, so the
    perception and policy results land on one axis.

    A hole vertex counts as visible when it projects inside the image AND the
    depth buffer at that pixel agrees with the vertex's own depth to within
    `depth_tol`. That tolerance MUST scale with the scene: an absolute value
    tuned on the real-metre env reports 0% visible on the ~22x env even from
    the training viewpoint, because at 128x128 a pixel spans a real slice of a
    slanted surface and the nearest depth sample differs from the vertex's own
    depth by far more than a few millimetres. Default: `tol_frac` of the
    cloth's bounding-box diagonal, which is scene-scale free. The depth test is what distinguishes "behind
    the cloth / behind an occluder" from "in view" - without it a vertex hidden
    by its own cloth would count as visible and the axis would be meaningless.

    Returns (visible_fraction, n_visible, n_total). Returns (nan, 0, 0) when the
    cloth has no hole, so callers can drop those episodes rather than silently
    scoring them 0.
    """
    idx = np.asarray(hole_vertex_indices, dtype=np.int64)
    if idx.size == 0:
        return float('nan'), 0, 0
    _, verts = get_mesh_data(deform_env.sim, deform_env.deform_id)
    verts = np.asarray(verts, dtype=np.float64)
    idx = idx[idx < len(verts)]
    hv = verts[idx]
    hv = hv[np.isfinite(hv).all(axis=1)]
    if len(hv) == 0:
        return float('nan'), 0, 0

    if depth_tol is None:
        extent = float(np.linalg.norm(verts.max(axis=0) - verts.min(axis=0)))
        depth_tol = max(tol_frac * extent, 1e-6)

    H, W = depth.shape
    v = np.asarray(view, dtype=np.float64).reshape(4, 4, order='F')
    p = np.asarray(proj, dtype=np.float64).reshape(4, 4, order='F')
    vp = p @ v
    hom = np.concatenate([hv, np.ones((len(hv), 1))], axis=1)
    clip = hom @ vp.T
    w = clip[:, 3:4]
    ok_w = (np.abs(w[:, 0]) > 1e-9)
    ndc = np.divide(clip[:, :3], np.where(np.abs(w) < 1e-9, 1.0, w))

    # NDC -> pixel, matching depth_to_pcd's convention exactly (u right,
    # v flipped) so this test and the back-projection agree.
    px = ((ndc[:, 0] + 1.0) * 0.5 * W).astype(np.int64)
    py = ((1.0 - (ndc[:, 1] + 1.0) * 0.5) * H).astype(np.int64)
    in_img = ok_w & (px >= 0) & (px < W) & (py >= 0) & (py < H) & \
        (ndc[:, 2] >= -1.0) & (ndc[:, 2] <= 1.0)
    if not in_img.any():
        return 0.0, 0, int(len(hv))

    pxi, pyi = px[in_img], py[in_img]
    # Compare in LINEAR world depth: the buffer is non-linear, so a fixed
    # tolerance on raw buffer values would mean different things near vs far.
    near, far = _PCD_NEAR, _PCD_FAR
    buf = depth[pyi, pxi]
    z_buf = 2.0 * near * far / (far + near - (2.0 * buf - 1.0) * (far - near))
    z_vert = 2.0 * near * far / (
        far + near - (2.0 * ((ndc[in_img, 2] + 1.0) * 0.5) - 1.0) * (far - near))
    seen = np.abs(z_buf - z_vert) <= depth_tol
    n_vis = int(seen.sum())
    n_tot = int(len(hv))
    return n_vis / max(n_tot, 1), n_vis, n_tot


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


# Hole-frame reliability gates. A procedurally-carved hole can come out
# nearly-degenerate (that is why the topological success metric was rejected
# as the primary one), and a normal fitted to a collapsed loop is noise.
# Callers fall back to a position-only test when `reliable` is False.
HOLE_MAX_PLANARITY = 0.35
HOLE_MIN_RADIUS = 0.25


def hole_frame(deform_env, hole_vertex_indices):
    """Centroid, plane normal, radius and planarity of the cloth's hole.

    The normal is the smallest principal direction of the loop vertices; its
    SIGN IS ARBITRARY, so compare directions with |cos|, never cos.
    `planarity` is s3/s2 — 0 for a perfectly planar loop, ~1 for a collapsed
    one. Returns (centroid, normal, radius, planarity, reliable); centroid and
    normal are None when there is no usable loop at all.
    """
    if not hole_vertex_indices:
        return None, None, 0.0, 1.0, False
    _, verts = get_mesh_data(deform_env.sim, deform_env.deform_id)
    verts = np.asarray(verts, dtype=np.float32)
    hv = verts[hole_vertex_indices]
    hv = hv[~np.isnan(hv).any(axis=1)]
    if len(hv) < 3:
        return None, None, 0.0, 1.0, False

    centroid = hv.mean(axis=0)
    radius = float(np.mean(np.linalg.norm(hv - centroid, axis=1)))
    _, s, vt = np.linalg.svd(hv - centroid, full_matrices=False)
    normal = np.asarray(vt[2], dtype=np.float32)
    planarity = float(s[2] / s[1]) if s[1] > 1e-9 else 1.0
    reliable = bool(planarity <= HOLE_MAX_PLANARITY and radius >= HOLE_MIN_RADIUS)
    return centroid, normal, radius, planarity, reliable


# ---------------------------------------------------------------------------
# Mesh sanity. The differential-anchor term in HoleServo pulls the two
# anchors apart, so an over-stretched spring mesh is a live failure mode
# rather than a theoretical one. This is a HARD GATE at collection time --
# an exploded episode is dropped, not recorded -- because a blown-up cloth
# still renders a plausible-looking point cloud and would silently poison
# the dataset.
# ---------------------------------------------------------------------------
def mesh_is_sane(deform_env, rest_extent, max_growth=2.5):
    """(sane, diagnostic_str). Rejects NaNs, escapes and runaway stretch."""
    _, verts = get_mesh_data(deform_env.sim, deform_env.deform_id)
    verts = np.asarray(verts, dtype=np.float32)
    if len(verts) == 0:
        return False, 'empty mesh'
    if not np.isfinite(verts).all():
        return False, f'{int((~np.isfinite(verts)).any(axis=1).sum())} non-finite verts'
    box = DeformEnv.WORKSPACE_BOX_SIZE
    if np.abs(verts).max() > box:
        return False, f'vert outside workspace box ({np.abs(verts).max():.1f} > {box})'
    extent = float(np.linalg.norm(verts.max(axis=0) - verts.min(axis=0)))
    if rest_extent > 1e-6 and extent > max_growth * rest_extent:
        return False, (f'extent {extent:.2f} = {extent / rest_extent:.2f}x rest '
                       f'{rest_extent:.2f} (max {max_growth}x)')
    return True, f'extent {extent:.2f} ({extent / max(rest_extent, 1e-6):.2f}x rest)'


def mesh_extent(deform_env):
    """Diagonal of the cloth's AABB — the reference for mesh_is_sane."""
    _, verts = get_mesh_data(deform_env.sim, deform_env.deform_id)
    verts = np.asarray(verts, dtype=np.float32)
    verts = verts[~np.isnan(verts).any(axis=1)]
    if len(verts) == 0:
        return 0.0
    return float(np.linalg.norm(verts.max(axis=0) - verts.min(axis=0)))


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
