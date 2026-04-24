"""
Axis-aligned bounding box (AABB) computation and visualization for point clouds,
with hole detection via 2-D occupancy grid analysis.

Usage (standalone on a saved .npy file):
    python -m dedo.utils.bbox_utils --pcd path/to/pcd.npy
"""
import argparse
from collections import deque
from typing import List, Tuple

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection


# ---------------------------------------------------------------------------
# AABB helpers
# ---------------------------------------------------------------------------

def compute_aabb(pcd: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Return (min_corner, max_corner), each shape (3,)."""
    return pcd.min(axis=0), pcd.max(axis=0)


def aabb_corners(mn: np.ndarray, mx: np.ndarray) -> np.ndarray:
    """Return the 8 corners of the AABB, shape (8, 3)."""
    return np.array([
        [mn[0], mn[1], mn[2]],
        [mx[0], mn[1], mn[2]],
        [mx[0], mx[1], mn[2]],
        [mn[0], mx[1], mn[2]],
        [mn[0], mn[1], mx[2]],
        [mx[0], mn[1], mx[2]],
        [mx[0], mx[1], mx[2]],
        [mn[0], mx[1], mx[2]],
    ])


def aabb_faces(corners: np.ndarray) -> List[List[np.ndarray]]:
    """Return the 6 faces of the AABB as lists of vertices for Poly3DCollection."""
    idx = [[0,1,2,3], [4,5,6,7], [0,1,5,4],
           [2,3,7,6], [0,3,7,4], [1,2,6,5]]
    return [[corners[j] for j in face] for face in idx]


# ---------------------------------------------------------------------------
# Hole detection
# ---------------------------------------------------------------------------

def find_holes(pcd: np.ndarray, grid_res: int = 64, search_radius: int = 10,
               min_hole_cells: int = 6, dilation: int = 2,
               debug_save: str = None) -> List[Tuple[np.ndarray, np.ndarray]]:
    """
    Detect holes in a cloth point cloud.

    Algorithm
    ---------
    1. Project the cloth onto its dominant 2-D plane (PCA).
    2. Rasterise into an occupancy grid.
    3. For each empty cell, scan outward in all 4 cardinal directions within
       `search_radius` cells.  If occupied cells are found in every direction,
       the empty cell is inside the cloth — i.e. it is part of a hole.
    4. Label connected hole cells (BFS) and discard tiny clusters.
    5. Map each cluster back to world-space and return its AABB.

    Parameters
    ----------
    pcd           : (N, 3) cloth-only point cloud
    grid_res      : resolution of the 2-D occupancy grid
    search_radius : how far (in cells) to look in each direction
    min_hole_cells: minimum cluster size to keep (filters noise)
    """
    if len(pcd) < 10:
        return []

    # --- 1. PCA dominant plane ---
    mean = pcd.mean(axis=0)
    centered = pcd - mean
    _, _, vh = np.linalg.svd(centered, full_matrices=False)
    axes2 = vh[:2]   # (2, 3) two principal axes
    normal = vh[2]   # (3,)   out-of-plane axis

    proj = centered @ axes2.T   # (N, 2)

    # --- 2. Occupancy grid ---
    mn2 = proj.min(axis=0)
    mx2 = proj.max(axis=0)
    pad = (mx2 - mn2) * 0.05
    mn2 -= pad; mx2 += pad
    span = mx2 - mn2

    ij = np.floor((proj - mn2) / span * grid_res).astype(int)
    ij = np.clip(ij, 0, grid_res - 1)
    occupied = np.zeros((grid_res, grid_res), dtype=bool)
    occupied[ij[:, 0], ij[:, 1]] = True

    # Dilate occupied grid to fill small gaps between adjacent mesh vertices.
    # Without this, every 1-2 cell gap between cloth vertices looks like a hole.
    # Real holes are much larger and survive dilation.
    dilated = occupied.copy()
    for _ in range(dilation):
        dilated[1:,  :] |= dilated[:-1, :]
        dilated[:-1, :] |= dilated[1:,  :]
        dilated[:,  1:] |= dilated[:, :-1]
        dilated[:, :-1] |= dilated[:,  1:]

    # --- 3. Mark empty cells that are surrounded on all 4 sides ---
    hole_mask = np.zeros((grid_res, grid_res), dtype=bool)
    for r in range(grid_res):
        for c in range(grid_res):
            if dilated[r, c]:
                continue
            up    = dilated[max(0, r - search_radius):r,              c].any()
            down  = dilated[r + 1:r + 1 + search_radius,              c].any()
            left  = dilated[r, max(0, c - search_radius):c           ].any()
            right = dilated[r, c + 1:c + 1 + search_radius           ].any()
            if up and down and left and right:
                hole_mask[r, c] = True

    # --- 4. BFS connected components on hole_mask ---
    labels = np.zeros((grid_res, grid_res), dtype=int)
    current_label = 0
    for start_r in range(grid_res):
        for start_c in range(grid_res):
            if not hole_mask[start_r, start_c] or labels[start_r, start_c]:
                continue
            current_label += 1
            q = deque([(start_r, start_c)])
            labels[start_r, start_c] = current_label
            while q:
                r, c = q.popleft()
                for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                    nr, nc = r + dr, c + dc
                    if 0 <= nr < grid_res and 0 <= nc < grid_res:
                        if hole_mask[nr, nc] and not labels[nr, nc]:
                            labels[nr, nc] = current_label
                            q.append((nr, nc))

    # --- debug: save a 3-panel image showing occupancy / hole mask / labels ---
    if debug_save is not None:
        fig, axes = plt.subplots(1, 3, figsize=(12, 4))
        axes[0].imshow(dilated.T, origin='lower', cmap='gray')
        axes[0].set_title(f'Occupied+dilated  (d={dilation})')
        axes[1].imshow(hole_mask.T, origin='lower', cmap='hot')
        axes[1].set_title(f'Hole mask  (radius={search_radius})')
        axes[2].imshow(labels.T, origin='lower', cmap='tab10')
        axes[2].set_title(f'Labels  ({current_label} components, min={min_hole_cells})')
        plt.tight_layout()
        plt.savefig(debug_save, dpi=150)
        plt.close(fig)
        return [], [], proj, occupied, dilated, hole_mask, labels

    # --- 5. Map each cluster to a 3-D AABB ---
    depth_proj = centered @ normal
    depth_mn, depth_mx = float(depth_proj.min()), float(depth_proj.max())

    hole_boxes = []
    hole_boxes_2d = []   # (mn_pca2, mx_pca2) in PCA space, for 2-D visualization
    for lbl in range(1, current_label + 1):
        cells = np.argwhere(labels == lbl)
        if len(cells) < min_hole_cells:
            continue

        mn_g = cells.min(axis=0).astype(float)
        mx_g = (cells.max(axis=0) + 1).astype(float)
        mn_pca2 = mn_g / grid_res * span + mn2
        mx_pca2 = mx_g / grid_res * span + mn2
        hole_boxes_2d.append((mn_pca2, mx_pca2))

        world_corners = []
        for u in (mn_pca2[0], mx_pca2[0]):
            for v in (mn_pca2[1], mx_pca2[1]):
                for d in (depth_mn, depth_mx):
                    world_corners.append(u * axes2[0] + v * axes2[1] + d * normal + mean)
        world_corners = np.array(world_corners)
        hole_boxes.append((world_corners.min(axis=0), world_corners.max(axis=0)))

    return hole_boxes, hole_boxes_2d, proj, occupied, dilated, hole_mask, labels


# ---------------------------------------------------------------------------
# Visualization
# ---------------------------------------------------------------------------

def save_pca_grid_frame(pcd: np.ndarray, save_path: str) -> None:
    """Save a 4-panel frame: raw PCA scatter | occupancy grid | dilated grid | bboxes."""
    import matplotlib.patches as mpatches
    _, hole_boxes_2d, proj, occupied, dilated, _, _ = find_holes(pcd)

    fig, axes = plt.subplots(1, 4, figsize=(16, 4))

    axes[0].scatter(proj[:, 0], proj[:, 1], s=1.0, c='steelblue', alpha=0.5)
    axes[0].set_aspect('equal')
    axes[0].set_title('PCA projection (raw)', fontsize=9)
    axes[0].invert_yaxis()

    axes[1].imshow(occupied.T, origin='lower', cmap='gray')
    axes[1].set_title('Occupancy grid', fontsize=9)
    axes[1].axis('off')

    axes[2].imshow(dilated.T, origin='lower', cmap='gray')
    axes[2].set_title('Dilated grid', fontsize=9)
    axes[2].axis('off')

    # 4th panel: PCA scatter + bounding boxes
    axes[3].scatter(proj[:, 0], proj[:, 1], s=1.0, c='steelblue', alpha=0.4)
    cloth_mn, cloth_mx = proj.min(axis=0), proj.max(axis=0)
    w, h = cloth_mx - cloth_mn
    axes[3].add_patch(mpatches.Rectangle(
        cloth_mn, w, h, linewidth=1.5, edgecolor='cyan', facecolor='none'))
    for mn2, mx2 in hole_boxes_2d:
        hw, hh = mx2 - mn2
        axes[3].add_patch(mpatches.Rectangle(
            mn2, hw, hh, linewidth=1.2, edgecolor='orange', facecolor='none'))
    axes[3].set_aspect('equal')
    axes[3].set_title(f'Bounding boxes ({len(hole_boxes_2d)} holes)', fontsize=9)
    axes[3].invert_yaxis()

    plt.tight_layout()
    plt.savefig(save_path, dpi=100)
    plt.close(fig)

def visualize_aabb(pcd: np.ndarray, ids: np.ndarray = None,
                   ax: plt.Axes = None, ax_2d: plt.Axes = None,
                   save_path: str = None) -> None:
    """
    Plot the cloth point cloud, its outer AABB, and detected hole AABBs.

    Parameters
    ----------
    pcd    : (N, 3) cloth-only point cloud
    ids    : (N,) integer object-id array for coloring (optional)
    ax     : existing 3-D matplotlib Axes (optional)
    ax_2d  : existing 2-D matplotlib Axes for the grid view (optional)
    """
    mn, mx = compute_aabb(pcd)
    corners = aabb_corners(mn, mx)
    faces = aabb_faces(corners)
    hole_boxes, _hole_boxes_2d, _proj, _occupied, dilated, hole_mask, labels = find_holes(pcd)

    standalone = ax is None
    if standalone:
        fig = plt.figure(figsize=(7, 6))
        ax = fig.add_subplot(111, projection='3d')
    else:
        fig = ax.get_figure()

    # --- point cloud ---
    if ids is not None:
        id_remap = {v: i for i, v in enumerate(np.unique(ids))}
        c = [id_remap[v] for v in ids]
        ax.scatter(pcd[:, 0], pcd[:, 1], pcd[:, 2],
                   marker='.', s=1.5, c=c, cmap='PRGn', alpha=0.4)
    else:
        ax.scatter(pcd[:, 0], pcd[:, 1], pcd[:, 2],
                   marker='.', s=1.5, color='steelblue', alpha=0.4)

    # --- outer bounding box (cyan) ---
    box = Poly3DCollection(faces, alpha=0.06, facecolor='cyan',
                           edgecolor='navy', linewidth=0.8)
    ax.add_collection3d(box)

    # --- hole bounding boxes (orange) ---
    for h_mn, h_mx in hole_boxes:
        h_corners = aabb_corners(h_mn, h_mx)
        h_faces = aabb_faces(h_corners)
        hole_col = Poly3DCollection(h_faces, alpha=0.15, facecolor='orange',
                                    edgecolor='red', linewidth=1.0)
        ax.add_collection3d(hole_col)

    dims = mx - mn
    n_holes = len(hole_boxes)
    ax.set_title(
        f'AABB  W={dims[0]:.2f} D={dims[1]:.2f} H={dims[2]:.2f} | holes={n_holes}',
        fontsize=8
    )
    ax.set_xlabel('X'); ax.set_ylabel('Y'); ax.set_zlabel('Z')
    ax.view_init(elev=20, azim=130)

    # --- 2-D grid panel: same 3-panel image as the debug output ---
    if ax_2d is not None:
        fig_2d, axes_2d = plt.subplots(1, 3, figsize=(6, 2))
        axes_2d[0].imshow(dilated.T, origin='lower', cmap='gray')
        axes_2d[0].set_title('Cloth', fontsize=7)
        axes_2d[0].axis('off')
        axes_2d[1].imshow(hole_mask.T, origin='lower', cmap='hot')
        axes_2d[1].set_title('Holes', fontsize=7)
        axes_2d[1].axis('off')
        axes_2d[2].imshow(labels.T, origin='lower', cmap='tab10')
        axes_2d[2].set_title(f'Labels ({n_holes})', fontsize=7)
        axes_2d[2].axis('off')
        fig_2d.tight_layout(pad=0.3)
        fig_2d.canvas.draw()
        w, h = fig_2d.canvas.get_width_height()
        grid_img = np.frombuffer(fig_2d.canvas.tostring_rgb(), dtype=np.uint8).reshape(h, w, 3)
        plt.close(fig_2d)
        ax_2d.imshow(grid_img)
        ax_2d.set_xticks([]); ax_2d.set_yticks([])
        ax_2d.set_title('2D grid', fontsize=8)

    center = (mn + mx) / 2
    half = dims.max() / 2 * 1.1
    ax.set_xlim(center[0] - half, center[0] + half)
    ax.set_ylim(center[1] - half, center[1] + half)
    ax.set_zlim(center[2] - half, center[2] + half)

    if save_path is not None:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        if standalone:
            plt.close(fig)
    elif standalone:
        plt.show()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--pcd', required=True, help='Path to .npy file, shape (N,3)')
    parser.add_argument('--save', default=None, help='Save plot to this path')
    parser.add_argument('--grid_res', type=int, default=64)
    args = parser.parse_args()

    pcd = np.load(args.pcd)
    assert pcd.ndim == 2 and pcd.shape[1] == 3, f'Expected (N,3), got {pcd.shape}'

    mn, mx = compute_aabb(pcd)
    dims = mx - mn
    print(f'Min corner : {mn}')
    print(f'Max corner : {mx}')
    print(f'Dimensions : W={dims[0]:.4f}  D={dims[1]:.4f}  H={dims[2]:.4f}')

    holes = find_holes(pcd, grid_res=args.grid_res)
    print(f'Holes found: {len(holes)}')
    for i, (h_mn, h_mx) in enumerate(holes):
        print(f'  hole {i}: mn={h_mn}  mx={h_mx}')

    visualize_aabb(pcd, save_path=args.save)
