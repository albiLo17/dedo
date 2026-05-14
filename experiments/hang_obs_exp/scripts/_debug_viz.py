"""
Debug visualization for BC demo collection.

Two artifacts per "debug" demo:

  1. demo_NNN_grid.png — multi-row PNG sampling N timesteps + the
     post-settle frame. Each row has 4 panels (RGB | depth heatmap |
     PCD overlay on RGB | valid-depth mask). Same format as
     `_diag_pcd_framing.py` — use it to eyeball whether the cloth
     stays in frame, whether PCD coverage is uniform, whether
     background is eating budget.

  2. demo_NNN_video.mp4 — per-control-step combined frame with three
     panels horizontally:
       - sim render (high-res RGB) with hole-centroid marker overlay
       - obs RGB (the 128² actually fed into the policy) with the
         back-projected PCD points drawn on top
       - PCD top-down scatter (x vs y, colored by z)
     Lets you scrub through the maneuver and see all three obs
     modalities + privileged state advance together.

Designed to be cheap enough to enable on the first few demos of
every collection run without slowing it down much (~5-10 s overhead
per debug demo).
"""
from __future__ import annotations

import os
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pybullet

import matplotlib
matplotlib.use('Agg')  # headless save
import matplotlib.pyplot as plt


# ---------------------------------------------------------------------------
# Hole centroid in world coordinates.
# ---------------------------------------------------------------------------
def hole_centroid_world(deform, hole_vertex_indices) -> Optional[np.ndarray]:
    """Compute the 3D world-space centroid of the cloth's hole loop,
    matching what `build_privileged_obs(obs_mode='hole_centroid')`
    uses — except this is the raw (unscaled) world point so we can
    project it to screen for the video overlay."""
    if not hole_vertex_indices:
        return None
    from dedo.utils.mesh_utils import get_mesh_data
    _, verts = get_mesh_data(deform.sim, deform.deform_id)
    verts = np.asarray(verts, dtype=np.float32)
    hole_verts = verts[hole_vertex_indices]
    hole_verts = hole_verts[~np.isnan(hole_verts).any(axis=1)]
    if len(hole_verts) == 0:
        return None
    return hole_verts.mean(axis=0)


# ---------------------------------------------------------------------------
# Generic projection: world point -> screen (u, v, in_view).
# ---------------------------------------------------------------------------
def project_world_to_screen(
        point_world: np.ndarray, view, proj, width: int, height: int
) -> Tuple[int, int, bool]:
    v = np.asarray(view, dtype=np.float64).reshape(4, 4, order='F')
    p = np.asarray(proj, dtype=np.float64).reshape(4, 4, order='F')
    pt_h = np.concatenate([np.asarray(point_world, dtype=np.float64),
                           [1.0]])
    clip = p @ v @ pt_h
    if clip[3] == 0.0:
        return 0, 0, False
    ndc = clip[:3] / clip[3]
    u = int((ndc[0] + 1.0) * 0.5 * width)
    vv = int((1.0 - (ndc[1] + 1.0) * 0.5) * height)
    in_view = (0 <= u < width and 0 <= vv < height
               and -1.0 < ndc[2] < 1.0)
    return u, vv, in_view


def overlay_pcd_on_rgb(rgb: np.ndarray, pcd_world: np.ndarray,
                       view, proj) -> np.ndarray:
    """Paint magenta 3x3 dots on the RGB for each PCD point that
    projects into frame. Same routine as `_diag_pcd_framing.py`."""
    H, W = rgb.shape[:2]
    if len(pcd_world) == 0:
        return rgb.copy()
    v = np.asarray(view, dtype=np.float64).reshape(4, 4, order='F')
    p = np.asarray(proj, dtype=np.float64).reshape(4, 4, order='F')
    pts_h = np.concatenate(
        [pcd_world, np.ones((len(pcd_world), 1))], axis=1)
    clip = (p @ v @ pts_h.T).T
    ok = clip[:, 3] != 0.0
    clip = clip[ok]
    ndc = clip[:, :3] / clip[:, 3:4]
    u = ((ndc[:, 0] + 1.0) * 0.5 * W).astype(np.int32)
    vv = ((1.0 - (ndc[:, 1] + 1.0) * 0.5) * H).astype(np.int32)
    in_view = ((u >= 0) & (u < W) & (vv >= 0) & (vv < H)
               & (ndc[:, 2] > -1) & (ndc[:, 2] < 1))
    u, vv = u[in_view], vv[in_view]
    out = rgb.copy()
    for du in (-1, 0, 1):
        for dv in (-1, 0, 1):
            u2 = np.clip(u + du, 0, W - 1)
            v2 = np.clip(vv + dv, 0, H - 1)
            out[v2, u2] = [255, 0, 255]
    return out


# ---------------------------------------------------------------------------
# PCD rendered from the SAME camera the dataset uses to capture RGB/depth.
# This is what the policy actually consumes: the 3D points back-projected
# from this exact view, then forward-projected back to 2D screen space to
# visualize their distribution. Aligned 1:1 with the RGB obs so the viewer
# can compare obs RGB ↔ PCD coverage at a glance.
# ---------------------------------------------------------------------------
def pcd_camera_view_image(pcd_world: np.ndarray, view, proj,
                          size: int = 300,
                          colormap_by: str = 'depth') -> np.ndarray:
    """Project world-space PCD through (view, proj) and paint dots
    colored by depth (NDC z, near=cool, far=warm). Black background.
    Same camera as the obs RGB → geometrically aligned with the
    middle panel of the debug video.

    `colormap_by='depth'` uses NDC z (camera-relative depth).
    `colormap_by='world_z'` uses world z (height above ground).
    """
    img = np.zeros((size, size, 3), dtype=np.uint8)
    if len(pcd_world) == 0:
        return img
    v = np.asarray(view, dtype=np.float64).reshape(4, 4, order='F')
    p = np.asarray(proj, dtype=np.float64).reshape(4, 4, order='F')
    pts_h = np.concatenate(
        [pcd_world, np.ones((len(pcd_world), 1))], axis=1)
    clip = (p @ v @ pts_h.T).T
    ok = clip[:, 3] != 0.0
    clip = clip[ok]
    pcd_kept = pcd_world[ok]
    ndc = clip[:, :3] / clip[:, 3:4]
    u = ((ndc[:, 0] + 1.0) * 0.5 * size).astype(np.int32)
    vv = ((1.0 - (ndc[:, 1] + 1.0) * 0.5) * size).astype(np.int32)
    in_view = ((u >= 0) & (u < size) & (vv >= 0) & (vv < size)
               & (ndc[:, 2] > -1) & (ndc[:, 2] < 1))
    u = u[in_view]
    vv = vv[in_view]
    pcd_kept = pcd_kept[in_view]
    ndc_z = ndc[in_view, 2]
    if colormap_by == 'world_z':
        t = np.clip((pcd_kept[:, 2] - 4.0) / 8.0, 0.0, 1.0)
    else:  # depth
        t = np.clip((ndc_z + 1.0) * 0.5, 0.0, 1.0)
    r = (np.clip(2.0 * t, 0.0, 1.0) * 255).astype(np.uint8)
    g = (np.clip(1.0 - np.abs(2.0 * t - 1.0), 0.0, 1.0) * 255).astype(np.uint8)
    b = (np.clip(2.0 * (1.0 - t), 0.0, 1.0) * 255).astype(np.uint8)
    # 3x3 dots so points are visible after a downstream nearest-neighbor
    # upscale to the video panel size.
    color = np.stack([r, g, b], axis=-1)
    for du in (-1, 0, 1):
        for dv in (-1, 0, 1):
            u2 = np.clip(u + du, 0, size - 1)
            v2 = np.clip(vv + dv, 0, size - 1)
            img[v2, u2] = color
    return img


# ---------------------------------------------------------------------------
# Render an annotated sim RGB frame: high-res deform.sim camera image with
# a small colored circle at the projected hole-centroid position.
# ---------------------------------------------------------------------------
def render_sim_with_centroid(
        deform, view, proj, centroid_world: Optional[np.ndarray],
        size: int = 300) -> np.ndarray:
    _, _, rgb_raw, _, _ = deform.sim.getCameraImage(
        width=size, height=size,
        viewMatrix=view, projectionMatrix=proj,
        renderer=pybullet.ER_BULLET_HARDWARE_OPENGL)
    rgb = np.asarray(rgb_raw, dtype=np.uint8).reshape(size, size, 4)[:, :, :3].copy()
    if centroid_world is not None:
        u, v_, in_view = project_world_to_screen(
            centroid_world, view, proj, size, size)
        if in_view:
            # 5-pixel-radius hollow circle, bright cyan.
            for du in range(-5, 6):
                for dv in range(-5, 6):
                    r2 = du * du + dv * dv
                    if 16 <= r2 <= 36:
                        uu = np.clip(u + du, 0, size - 1)
                        vv = np.clip(v_ + dv, 0, size - 1)
                        rgb[vv, uu] = [0, 255, 255]
    return rgb


# ---------------------------------------------------------------------------
# Combined video frame builder.
# ---------------------------------------------------------------------------
def _resize_nearest(img: np.ndarray, size: int) -> np.ndarray:
    """Cheap nearest-neighbor upscale. Avoid the cv2 dependency since
    not all installs have it; numpy indexing is plenty for visualization."""
    H, W = img.shape[:2]
    if H == size and W == size:
        return img
    ys = (np.linspace(0, H, size, endpoint=False)).astype(np.int32)
    xs = (np.linspace(0, W, size, endpoint=False)).astype(np.int32)
    return img[ys[:, None], xs[None, :]]


def _draw_label(img: np.ndarray, text: str) -> np.ndarray:
    """Burn a tiny black bar with the label text into the top of the
    image. matplotlib-style font would be expensive per frame; instead
    we just draw a small black strip and the caller relies on the
    panel order being stable."""
    out = img.copy()
    out[:14, :, :] = 20  # dark strip; caller knows panel order
    # No actual text rendering — too slow per frame. The panel order is
    # documented in the build function below.
    return out


def build_video_frame(
        sim_rgb: np.ndarray,
        obs_rgb_with_overlay: np.ndarray,
        pcd_topdown: np.ndarray,
        size: int = 300) -> np.ndarray:
    """Concatenate (sim | obs+PCD | top-down PCD) horizontally to one
    (size, 3*size, 3) uint8 frame."""
    a = _resize_nearest(sim_rgb, size)
    b = _resize_nearest(obs_rgb_with_overlay, size)
    c = _resize_nearest(pcd_topdown, size)
    # Thin divider between panels so the eye can tell them apart.
    div = np.full((size, 2, 3), 255, dtype=np.uint8)
    return np.concatenate([a, div, b, div, c], axis=1)


# ---------------------------------------------------------------------------
# MP4 writer (libx264, yuv420p, +faststart — matches the diffusion script
# and the PPO HangVideoCallback so all wandb videos play uniformly).
# ---------------------------------------------------------------------------
def write_video_mp4(frames: List[np.ndarray], out_path: str,
                    fps: int = 15) -> None:
    if not frames:
        return
    import imageio  # imageio_ffmpeg is a transitive dep of imageio
    writer = imageio.get_writer(
        out_path, fps=fps, codec='libx264', quality=8,
        macro_block_size=2, pixelformat='yuv420p',
        ffmpeg_params=['-movflags', '+faststart'])
    for frame in frames:
        writer.append_data(np.ascontiguousarray(frame))
    writer.close()


# ---------------------------------------------------------------------------
# Per-demo action diagnostics. The collector tells us how many of the
# `len(acts)` steps were active trajectory frames vs `last_action` hold
# frames; we plot ||a|| per step, mark the boundary, and break out the
# 3-axis components for each of the 2 grippers.
# ---------------------------------------------------------------------------
ACTION_HOLD_THRESHOLD = 0.05  # ||act|| below this is "stationary"


def summarize_action_stream(
        acts: np.ndarray, traj_len: int) -> Dict[str, float]:
    """Compute per-episode action-distribution stats. Returns a flat dict
    of scalars suitable for printing or wandb logging."""
    acts = np.asarray(acts, dtype=np.float32)
    norms = np.linalg.norm(acts, axis=-1)  # per-step L2 over 6-dim action
    ep_len = len(acts)
    hold = max(ep_len - int(traj_len), 0)
    active = ep_len - hold
    stationary = int((norms < ACTION_HOLD_THRESHOLD).sum())
    return {
        'ep_len': int(ep_len),
        'traj_len': int(traj_len),
        'n_active': int(active),
        'n_hold': int(hold),
        'hold_frac': float(hold / max(ep_len, 1)),
        'n_stationary': stationary,
        'stationary_frac': float(stationary / max(ep_len, 1)),
        'peak_abs_a': float(np.abs(acts).max()) if ep_len > 0 else 0.0,
        'mean_norm_a': float(norms.mean()) if ep_len > 0 else 0.0,
        'final_action_norm': (float(norms[traj_len - 1])
                              if 0 < traj_len <= ep_len else 0.0),
    }


def save_actions_plot(acts: np.ndarray, traj_len: int,
                      out_path: str, title: str) -> None:
    """Time-series plot of per-step action magnitude + 6 component
    traces. Vertical line marks the traj→hold boundary so the viewer
    can immediately see how long the "hold last action" phase lasted.

    `acts` is (T, 6) normalized to [-1, 1]; columns are dx_a, dy_a,
    dz_a, dx_b, dy_b, dz_b (gripper A then gripper B).
    """
    acts = np.asarray(acts, dtype=np.float32)
    T, D = acts.shape
    fig, axes = plt.subplots(2, 1, figsize=(11, 6), sharex=True)

    # Top panel: ||action|| over time.
    norms = np.linalg.norm(acts, axis=-1)
    axes[0].plot(norms, color='black', lw=1.2, label='||action||')
    axes[0].axhline(ACTION_HOLD_THRESHOLD, color='gray', ls=':', lw=0.8,
                    label=f'stationary thresh ({ACTION_HOLD_THRESHOLD})')
    if 0 < traj_len <= T:
        axes[0].axvline(traj_len - 0.5, color='red', ls='--', lw=1.0,
                        label=f'traj→hold boundary (step {traj_len})')
    axes[0].set_ylabel('||action||')
    axes[0].set_ylim(0, max(1.05, float(norms.max() * 1.05)))
    axes[0].legend(loc='upper right', fontsize=8)
    axes[0].grid(alpha=0.3)

    # Bottom panel: per-component action traces.
    component_labels = (['A.dx', 'A.dy', 'A.dz', 'B.dx', 'B.dy', 'B.dz']
                        if D == 6 else [f'a[{i}]' for i in range(D)])
    component_colors = ['tab:blue', 'tab:orange', 'tab:green',
                        'tab:red', 'tab:purple', 'tab:brown']
    for i, lbl in enumerate(component_labels):
        axes[1].plot(acts[:, i], lw=1.0, alpha=0.85,
                     color=component_colors[i % len(component_colors)],
                     label=lbl)
    if 0 < traj_len <= T:
        axes[1].axvline(traj_len - 0.5, color='red', ls='--', lw=1.0)
    axes[1].axhline(0, color='gray', lw=0.5)
    axes[1].set_xlabel('control step')
    axes[1].set_ylabel('action component')
    axes[1].set_ylim(-1.05, 1.05)
    axes[1].legend(loc='upper right', fontsize=8, ncol=3)
    axes[1].grid(alpha=0.3)

    fig.suptitle(title, fontsize=11)
    plt.tight_layout()
    plt.savefig(out_path, dpi=110, bbox_inches='tight')
    plt.close(fig)


# ---------------------------------------------------------------------------
# Multi-row PNG grid. Same layout as _diag_pcd_framing.py:
#   one row per (sampled timestep + post-settle); 4 cols per row.
# ---------------------------------------------------------------------------
def save_grid_png(
        rows: List[Dict[str, Any]],
        out_path: str,
        title: str) -> None:
    """`rows` is a list of dicts, each with keys:
        label    : str        (e.g. 'step 12/45' or 'post-settle')
        rgb      : (H,W,3) u8
        depth    : (H,W)   f
        pcd      : (N,3)   f  — world coords (or None to skip overlay)
        view     : pybullet viewMatrix
        proj     : pybullet projMatrix
        in_frame : float 0-1   (cloth-mesh fraction in frame; or None)
        n_valid  : int          (depth pixels < 0.999)
    """
    n_rows = len(rows)
    fig, axes = plt.subplots(n_rows, 4, figsize=(14, 3.4 * n_rows))
    if n_rows == 1:
        axes = axes[None, :]
    for ri, r in enumerate(rows):
        rgb = r['rgb']
        depth = r['depth']
        pcd = r.get('pcd')
        view = r.get('view')
        proj = r.get('proj')
        H, W = rgb.shape[:2]
        overlay = (overlay_pcd_on_rgb(rgb, pcd, view, proj)
                   if pcd is not None and view is not None
                   else rgb.copy())

        axes[ri, 0].imshow(rgb)
        axes[ri, 0].set_title(
            f'{r["label"]}\nRGB  ({W}×{H})')
        axes[ri, 0].axis('off')

        axes[ri, 1].imshow(depth, cmap='viridis', vmin=0.0, vmax=1.0)
        axes[ri, 1].set_title(
            f'depth  (1.0 = bg)\n{r.get("n_valid", "?")} valid px')
        axes[ri, 1].axis('off')

        axes[ri, 2].imshow(overlay)
        in_frame = r.get('in_frame')
        if in_frame is None:
            in_frame_str = ''
        else:
            in_frame_str = f'\ncloth in frame: {in_frame*100:.1f}%'
        axes[ri, 2].set_title(
            f'PCD overlay (N={len(pcd) if pcd is not None else 0})'
            f'{in_frame_str}')
        axes[ri, 2].axis('off')

        valid_mask = (depth < 0.999).astype(np.float32)
        axes[ri, 3].imshow(valid_mask, cmap='gray')
        axes[ri, 3].set_title('valid depth mask\n(white = used for PCD)')
        axes[ri, 3].axis('off')

    fig.suptitle(title, fontsize=11, y=1.0)
    plt.tight_layout()
    plt.savefig(out_path, dpi=110, bbox_inches='tight')
    plt.close(fig)
