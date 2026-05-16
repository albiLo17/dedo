"""
Interactive visualizer for GSE demo HDF5 files.

Four panels update together as you scrub a timestep slider:
  top-left   — RGB image
  top-right  — deformed mesh (vertex positions + faces at time t)
  bot-left   — point cloud (depth back-projection at time t)
  bot-right  — canonical mesh (rest positions, constant per demo)

A radio-button panel at the bottom lets you switch between demos.

Usage (from repo root):
    python experiments/hang_obs_exp/scripts/visualize_gse_demos.py \
        --h5 logs/hang_obs_exp/gse_demos_v1.h5
"""
from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path

import h5py
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, RadioButtons
from mpl_toolkits.mplot3d.art3d import Poly3DCollection


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

@dataclass
class DemoData:
    rest_positions: np.ndarray  # (V, 3) float32
    faces: np.ndarray           # (F, 3) int64
    positions: np.ndarray       # (T, V, 3) float32
    pcd: np.ndarray             # (T, N, 3) float32
    rgb: np.ndarray             # (T, H, W, 3) uint8
    split: str


def load_all_demos(h5_path: str) -> dict[str, DemoData]:
    demos: dict[str, DemoData] = {}
    with h5py.File(h5_path, 'r') as f:
        for split in ('training', 'validation'):
            if split not in f:
                continue
            for cloth_key in sorted(f[split].keys()):
                grp = f[split][cloth_key]
                rest_pos = grp['rest_positions'][:]
                faces = grp['faces'][:]
                traj = grp['trajectory_0']
                step_keys = sorted(k for k in traj if k.startswith('step'))
                positions, pcd_frames, rgb_frames = [], [], []
                for sk in step_keys:
                    sg = traj[sk]
                    positions.append(sg['positions'][:])
                    pcd_frames.append(sg['pointclouds']['cam_0'][:])
                    rgb_frames.append(sg['rgb'][:])
                key = f'{split}/{cloth_key}'
                demos[key] = DemoData(
                    rest_positions=rest_pos.astype(np.float32),
                    faces=faces.astype(np.int64),
                    positions=np.stack(positions).astype(np.float32),
                    pcd=np.stack(pcd_frames).astype(np.float32),
                    rgb=np.stack(rgb_frames),
                    split=split,
                )
                print(f'  {key}: T={len(step_keys)}, '
                      f'V={rest_pos.shape[0]}, F={faces.shape[0]}, '
                      f'N={pcd_frames[0].shape[0]}, '
                      f'rgb={rgb_frames[0].shape}')
    return demos


# ---------------------------------------------------------------------------
# 3-D drawing helpers
# ---------------------------------------------------------------------------

def _apply_bounds(ax, lo: np.ndarray, hi: np.ndarray):
    ctr = (lo + hi) * 0.5
    half = max((hi - lo).max() * 0.5, 1e-4)
    ax.set_xlim(ctr[0] - half, ctr[0] + half)
    ax.set_ylim(ctr[1] - half, ctr[1] + half)
    ax.set_zlim(ctr[2] - half, ctr[2] + half)
    ax.set_xlabel('x', fontsize=6, labelpad=1)
    ax.set_ylabel('y', fontsize=6, labelpad=1)
    ax.set_zlabel('z', fontsize=6, labelpad=1)
    ax.tick_params(labelsize=5, pad=1)


def draw_mesh(ax, positions: np.ndarray, faces: np.ndarray,
              title: str, facecolor: str,
              bounds: tuple[np.ndarray, np.ndarray]):
    elev, azim = ax.elev, ax.azim
    ax.cla()
    tris = positions[faces]  # (F, 3, 3)
    poly = Poly3DCollection(tris, alpha=0.45,
                            facecolor=facecolor,
                            edgecolor='#606060', linewidth=0.15)
    ax.add_collection3d(poly)
    _apply_bounds(ax, *bounds)
    ax.set_title(title, fontsize=9)
    ax.view_init(elev=elev, azim=azim)


def draw_pcd(ax, pcd: np.ndarray, title: str,
             bounds: tuple[np.ndarray, np.ndarray]):
    elev, azim = ax.elev, ax.azim
    ax.cla()
    ax.scatter(pcd[:, 0], pcd[:, 1], pcd[:, 2],
               s=2, c='#ff7043', alpha=0.75, depthshade=True)
    _apply_bounds(ax, *bounds)
    ax.set_title(title, fontsize=9)
    ax.view_init(elev=elev, azim=azim)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--h5', required=True,
                        help='Path to the GSE HDF5 file')
    args = parser.parse_args()

    print(f'Loading {args.h5} …')
    demos = load_all_demos(args.h5)
    if not demos:
        print('No demos found. Run collect_bc_demos_GSE.py first.')
        sys.exit(1)
    demo_keys = list(demos.keys())
    print(f'Loaded {len(demo_keys)} demo(s).')

    # Precompute per-demo spatial bounds (mesh and pcd separately so
    # axes limits stay constant as the timestep scrolls).
    def _bounds(d: DemoData):
        all_mesh = np.concatenate([d.positions.reshape(-1, 3),
                                   d.rest_positions], axis=0)
        mesh_lo, mesh_hi = all_mesh.min(0), all_mesh.max(0)
        all_pcd = d.pcd.reshape(-1, 3)
        pcd_lo, pcd_hi = all_pcd.min(0), all_pcd.max(0)
        return (mesh_lo, mesh_hi), (pcd_lo, pcd_hi)

    bounds_cache = {k: _bounds(v) for k, v in demos.items()}

    # -----------------------------------------------------------------------
    # Figure layout
    # -----------------------------------------------------------------------
    fig = plt.figure(figsize=(16, 11))
    fig.suptitle('GSE Demo Viewer', fontsize=12, fontweight='bold')

    # Main 2×2 grid for the four panels
    plt.subplots_adjust(left=0.05, right=0.97,
                        top=0.93, bottom=0.20,
                        wspace=0.30, hspace=0.38)

    ax_rgb   = fig.add_subplot(2, 2, 1)
    ax_mesh  = fig.add_subplot(2, 2, 2, projection='3d')
    ax_pcd   = fig.add_subplot(2, 2, 3, projection='3d')
    ax_canon = fig.add_subplot(2, 2, 4, projection='3d')

    # Timestep slider (centre-bottom)
    ax_slider = fig.add_axes([0.20, 0.10, 0.58, 0.03])
    d0 = demos[demo_keys[0]]
    T0 = d0.positions.shape[0]
    slider = Slider(ax_slider, 'Timestep', 0, T0 - 1,
                    valinit=0, valstep=1, color='steelblue')

    # Demo radio buttons (left of slider)
    radio_h = min(0.06 * len(demo_keys) + 0.03, 0.14)
    ax_radio = fig.add_axes([0.20, 0.10 - radio_h - 0.02,
                             0.58, radio_h])
    radio = RadioButtons(ax_radio, demo_keys, active=0,
                         activecolor='steelblue')
    # Shrink the radio-button font for long keys
    for lbl in radio.labels:
        lbl.set_fontsize(8)

    # -----------------------------------------------------------------------
    # Shared mutable state
    # -----------------------------------------------------------------------
    state = {'key': demo_keys[0], 't': 0}

    # -----------------------------------------------------------------------
    # Update callbacks
    # -----------------------------------------------------------------------
    def _update(t: int, key: str, redraw_canon: bool):
        d = demos[key]
        mesh_bounds, pcd_bounds = bounds_cache[key]

        # RGB
        ax_rgb.cla()
        ax_rgb.imshow(d.rgb[t])
        ax_rgb.set_title(f'RGB   t={t}', fontsize=9)
        ax_rgb.axis('off')

        # Deformed mesh
        draw_mesh(ax_mesh, d.positions[t], d.faces,
                  title=f'Mesh   t={t}',
                  facecolor='#4fc3f7',
                  bounds=mesh_bounds)

        # Point cloud
        draw_pcd(ax_pcd, d.pcd[t],
                 title=f'Point cloud   t={t}',
                 bounds=pcd_bounds)

        # Canonical mesh — only redrawn on demo switch (rest_positions is
        # constant across timesteps so there's no need to clear it every frame)
        if redraw_canon:
            draw_mesh(ax_canon, d.rest_positions, d.faces,
                      title='Canonical mesh (rest)',
                      facecolor='#a5d6a7',
                      bounds=mesh_bounds)

        fig.canvas.draw_idle()

    def on_slider(val):
        t = int(slider.val)
        state['t'] = t
        _update(t, state['key'], redraw_canon=False)

    def on_demo(label):
        state['key'] = label
        d = demos[label]
        T = d.positions.shape[0]
        # Update slider range for the new demo
        slider.valmax = T - 1
        slider.ax.set_xlim(0, T - 1)
        slider.set_val(0)
        state['t'] = 0
        _update(0, label, redraw_canon=True)

    slider.on_changed(on_slider)
    radio.on_clicked(on_demo)

    # Initial draw
    _update(0, demo_keys[0], redraw_canon=True)

    plt.show()


if __name__ == '__main__':
    main()
