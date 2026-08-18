"""What a demo dataset actually contains, and what the scripted expert does.

  1. cloth meshes — the procedural variety the policies train on: every episode
     re-randomizes size and hole placement, so a "cloth" is a fresh topology
     rather than a pose of one mesh.
  2. goal positions — where the policy is asked to put the hole, split by
     episode kind, with the measured real hanger tip overlaid.
  3. start pose — where the two grippers hold the cloth at t=0, against the
     measured real start. Every real demo starts from the same pose (sd 0.1 mm),
     so this is a point the sim distribution either covers or does not.
  4. expert diagnosis (--diag_dir) — success rate and hole-height traces across
     the four geometries from _diag_expert.sh. The traces are plotted RELATIVE
     to each episode's own peg tip, so episodes with different goal jitter
     overlay and the expert's three fixed waypoint heights (+1.8 / +0.4 / -0.5)
     are single horizontal lines.

    python viz_v5_dataset.py --demo_path logs/hang_obs_exp/bc_demos_v5 \
        --out_dir logs/hang_obs_exp/v5_report
    python viz_v5_dataset.py --demo_path .../bc_demos_v6 --tag v6 \
        --diag_dir .../diag_expert --out_dir logs/hang_obs_exp/v6_report
"""
import argparse
import glob
import os
import pickle

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt                                  # noqa: E402
from matplotlib.patches import Polygon                           # noqa: E402
from matplotlib.collections import PatchCollection               # noqa: E402

SCALE_MM = 45.0
WBOX = 20.0
REAL_GOAL = np.array([1.366, -1.266, 6.213])     # measured hanger tip, sim units
# Measured start pose of the two grippers on the rig, sim units. All 8 real
# demos start here to within 0.1 mm, so this is a point, not a distribution.
REAL_ANCHOR = np.array([[0.597, 3.547, 9.562],
                        [-3.987, 2.848, 9.673]])
# The expert's three waypoint heights, as offsets from the peg tip in
# reference-scene units (_helpers.py build_hole_aware_waypoints, s=1 here).
WP_UP = {'hover': 1.8, 'thread': 0.4, 'hold': -0.5}
GEOMETRIES = {
    'A': 'A  v5 control\npeg [0,0,8.2], rod under it',
    'B': 'B  v6 as shipped\npeg moved; cloth + rod not',
    'C': 'C  cloth follows peg\nrod still grounded',
    'D': 'D  real-aligned start\nrod still grounded',
    'E': 'E  TRUE translation\npeg + cloth + rod',
    'F': 'F  real-aligned start\nrod moved too',
}
# Height of the support-rod top above the peg tip, per geometry. tallrod.urdf
# is exactly 8.0 sim units long at globalScaling=10, so wherever the rod base
# is, its top is base+8.0. The shipped preset paired hanger z=8.0 with rod z=0
# so the post ended just under the peg; moving only the hanger left the bare
# post standing 1.79 units ABOVE the target the cloth is aimed at.
ROD_TOP_REL = {'A': -0.198, 'B': 1.789, 'C': 1.789,
               'D': 1.789, 'E': -0.198, 'F': -0.198}


def load(path, keys=('obs', 'cloth_faces', 'episode_kind', 'hole_radius',
                     'chain_goal_positions', 'cloth_width', 'success_legacy',
                     'cloth_shape', 'node_density',
                     'success_hanging', 'peg_nominal',
                     'deform_init_pos_nominal')):
    with open(path, 'rb') as f:
        d = pickle.load(f)
    return {k: d.get(k) for k in keys}


def state_vec(d):
    """obs['hole_centroid'] is the 18-dim state, not the 3-dim centroid:
    [anchor0 pos, anchor0 vel, anchor1 pos, anchor1 vel, hole centroid, goal].
    Returned in sim units (T, 18)."""
    return np.asarray(d['obs']['hole_centroid'], np.float64) * WBOX


def mesh_panel(ax, verts, faces, title):
    """Face-on view of one cloth. The hole is the gap in the triangulation."""
    # The cloth hangs in the XZ plane, so x-z IS the face-on view.
    pts = verts[:, [0, 2]]
    tris = [Polygon(pts[f], closed=True) for f in faces]
    pc = PatchCollection(tris, facecolor='#8fb8de', edgecolor='#41668c',
                         linewidths=0.25, alpha=0.9)
    ax.add_collection(pc)
    ax.set_xlim(pts[:, 0].min() - .3, pts[:, 0].max() + .3)
    ax.set_ylim(pts[:, 1].min() - .3, pts[:, 1].max() + .3)
    ax.set_aspect('equal')
    ax.set_title(title, fontsize=8)
    ax.set_xticks([])
    ax.set_yticks([])


def start_pose_figure(files, out_path, tag):
    """Where the grippers hold the cloth at t=0, sim vs the measured rig."""
    a0, a1 = [], []
    for fp in files:
        s = state_vec(load(fp))[0]
        a0.append(s[0:3])
        a1.append(s[6:9])
    a0, a1 = np.asarray(a0), np.asarray(a1)

    fig, ax = plt.subplots(1, 3, figsize=(18, 5.6))
    for i, (j, k, lj, lk) in enumerate([(0, 2, 'x', 'z'), (1, 2, 'y', 'z'),
                                        (0, 1, 'x', 'y')]):
        for arr, col, name in ((a0, '#4c78a8', 'anchor 0'),
                               (a1, '#e45756', 'anchor 1')):
            ax[i].scatter(arr[:, j], arr[:, k], s=9, alpha=.35, c=col,
                          label=f'sim {name} (n={len(arr)})')
        for m, (col, name) in enumerate((('#1b3f66', 'anchor 0'),
                                         ('#8c1f1f', 'anchor 1'))):
            ax[i].scatter([REAL_ANCHOR[m, j]], [REAL_ANCHOR[m, k]], marker='*',
                          s=420, c=col, edgecolors='k', zorder=6,
                          label=f'REAL {name}')
        ax[i].set(xlabel=f'sim {lj}', ylabel=f'sim {lk}',
                  title=f'gripper start pose · {lj}-{lk}')
        ax[i].grid(alpha=.25)
        ax[i].set_aspect('equal', adjustable='datalim')
    ax[0].legend(fontsize=7, loc='best')

    # The number that matters is how far the real start sits from the middle of
    # the sim distribution, in units of that distribution's own spread: a gap of
    # 2 sd means the rig start is effectively outside what the policy has seen.
    bits = []
    for m, arr in enumerate((a0, a1)):
        mu, sd = arr.mean(0), arr.std(0)
        gap = np.linalg.norm(REAL_ANCHOR[m] - mu)
        z = np.abs(REAL_ANCHOR[m] - mu) / np.maximum(sd, 1e-6)
        bits.append(f'anchor{m}: sim mean {np.round(mu, 2)} sd {np.round(sd, 2)}'
                    f' | real {REAL_ANCHOR[m]} | gap {gap*SCALE_MM:.0f} mm'
                    f' = {np.round(z, 1)} sd per axis')
    mid_sim = ((a0 + a1) / 2).mean(0)
    mid_real = REAL_ANCHOR.mean(0)
    sep_sim = np.linalg.norm(a0 - a1, axis=1)
    sep_real = np.linalg.norm(REAL_ANCHOR[0] - REAL_ANCHOR[1])
    bits.append(f'midpoint gap {np.linalg.norm(mid_real-mid_sim)*SCALE_MM:.0f} mm'
                f'   |   gripper separation sim {sep_sim.mean():.2f}'
                f'+/-{sep_sim.std():.2f} vs real {sep_real:.2f}')
    fig.suptitle(f'{tag}: gripper start pose vs the rig (1 sim unit = 45 mm)\n'
                 + '\n'.join(bits), fontsize=8.5)
    fig.tight_layout(rect=(0, 0, 1, 0.90))
    fig.savefig(out_path, dpi=120)
    print(f'wrote {out_path}')
    print('\n=== gripper start pose ===')
    for b in bits:
        print('  ' + b)


def expert_figure(diag_dir, out_path):
    """Success by geometry, and the hole-height trace that explains it."""
    data = {}
    for gid in GEOMETRIES:
        files = sorted(glob.glob(os.path.join(diag_dir, gid, 'demo_*.pkl')))
        if not files:
            continue
        rows = []
        for fp in files:
            d = load(fp)
            s = state_vec(d)
            rows.append({
                'ok': bool(d['success_legacy']),
                'dz': s[:, 14] - s[:, 17],   # hole z minus THIS episode's peg z
                'peg': np.asarray(d['peg_nominal'], float),
                'start': np.asarray(d['deform_init_pos_nominal'], float),
            })
        data[gid] = rows
    if not data:
        print(f'no diagnostic pkls under {diag_dir}')
        return

    fig = plt.figure(figsize=(19, 9.2))
    gs = fig.add_gridspec(2, len(data), height_ratios=[1.0, 1.35], hspace=.34)

    axb = fig.add_subplot(gs[0, :])
    ids = list(data)
    ps = [np.mean([r['ok'] for r in data[g]]) for g in ids]
    ns = [len(data[g]) for g in ids]
    err = [1.96 * np.sqrt(max(p * (1 - p), 1e-9) / n) for p, n in zip(ps, ns)]
    bars = axb.bar(range(len(ids)), ps, yerr=err, capsize=5,
                   color=['#54a24b', '#e45756', '#4c78a8', '#ff9d1e'][:len(ids)])
    for i, (b, p, n) in enumerate(zip(bars, ps, ns)):
        axb.text(b.get_x() + b.get_width() / 2, p + err[i] + .012,
                 f'{p*100:.0f}%  ({int(round(p*n))}/{n})', ha='center',
                 fontsize=10, fontweight='bold')
    axb.set_xticks(range(len(ids)))
    axb.set_xticklabels([f'{g}\n{GEOMETRIES[g]}' for g in ids], fontsize=9)
    axb.set_ylabel('expert threading success (legacy)')
    axb.set_ylim(0, max(max(ps) + max(err) + .10, .2))
    axb.grid(axis='y', alpha=.25)
    axb.set_title('Scripted expert success by scene geometry — same seed, same '
                  'cloths, same jitter; only the peg and the cloth start differ',
                  fontsize=11)

    ylo = min(min(r['dz'].min() for r in rs) for rs in data.values())
    yhi = max(max(r['dz'].max() for r in rs) for rs in data.values())
    for i, gid in enumerate(ids):
        ax = fig.add_subplot(gs[1, i])
        for r in data[gid]:
            t = np.linspace(0, 1, len(r['dz']))
            ax.plot(t, r['dz'], lw=.7, alpha=.30,
                    c='#54a24b' if r['ok'] else '#e45756')
        for name, up in WP_UP.items():
            ax.axhline(up, ls='--', lw=1.1, c='#333')
            ax.text(1.005, up, f' {name} {up:+.1f}', fontsize=7, va='center')
        ax.axhline(0, lw=1.6, c='k')
        ax.text(1.005, 0, ' peg tip', fontsize=7, va='center', fontweight='bold')
        rod = ROD_TOP_REL.get(gid, 0.0)
        if rod > 0:
            # The bare post standing above the peg tip. Anything the expert
            # tries to do in this band runs the cloth into the post instead of
            # onto the hanger.
            ax.axhspan(0, rod, color='#c8a020', alpha=.32, zorder=0)
        ax.axhline(rod, lw=1.4, c='#8a6d10')
        ax.text(1.005, rod, ' rod top', fontsize=7, va='center', c='#8a6d10',
                fontweight='bold' if rod > 0 else 'normal')

        start = np.mean([r['dz'][0] for r in data[gid]])
        ax.set_title(GEOMETRIES[gid], fontsize=9)
        ax.text(.03, .03,
                f'hole starts {start:+.2f}\nhover asks {WP_UP["hover"]-start:+.2f}\n'
                f'rod top {rod:+.2f}'
                + ('\nPOST BLOCKS TARGET' if rod > 0 else ''),
                transform=ax.transAxes, fontsize=7.5, va='bottom',
                bbox=dict(fc='white', ec='#999', alpha=.85, pad=2.5))
        ax.set_xlabel('episode progress')
        if i == 0:
            ax.set_ylabel('hole-centroid z MINUS peg-tip z (sim units)')
        else:
            ax.set_yticklabels([])
        ax.set_xlim(0, 1.16)
        ax.grid(alpha=.22)
    # Clipped rather than data-fit: a handful of failures drop the cloth to the
    # floor at -8, which would squash the +/-2 band where the threading actually
    # happens down to a few pixels.
    for ax in fig.axes[1:]:
        ax.set_ylim(max(ylo - .3, -3.5), min(max(yhi, 2.2) + .3, 5.0))
    fig.savefig(out_path, dpi=120, bbox_inches='tight')
    print(f'wrote {out_path}')

    print('\n=== expert by geometry ===')
    for g, p, n in zip(ids, ps, ns):
        r0 = data[g][0]
        start = np.mean([r['dz'][0] for r in data[g]])
        print(f'  {g}  peg {np.round(r0["peg"], 2)}  start '
              f'{np.round(r0["start"], 2)}  success {p*100:5.1f}% (n={n})  '
              f'hole starts {start:+.2f} rel. peg, hover asks '
              f'{WP_UP["hover"] - start:+.2f}')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--demo_path', required=True)
    ap.add_argument('--out_dir', required=True)
    ap.add_argument('--n_meshes', type=int, default=12)
    ap.add_argument('--max_demos', type=int, default=300)
    ap.add_argument('--tag', default='v5', help='output filename prefix')
    ap.add_argument('--diag_dir', default=None,
                    help='_diag_expert.sh output dir (A/B/C/D subdirs)')
    a = ap.parse_args()
    os.makedirs(a.out_dir, exist_ok=True)
    files = sorted(glob.glob(os.path.join(a.demo_path, 'demo_*.pkl')))[:a.max_demos]

    # ---- figure 1: cloth meshes ------------------------------------------
    step = max(1, len(files) // a.n_meshes)
    picks = files[::step][:a.n_meshes]
    ncol = 4
    nrow = int(np.ceil(len(picks) / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(3.1 * ncol, 3.0 * nrow))
    axes = np.atleast_1d(axes).ravel()
    widths, radii = [], []
    for ax, fp in zip(axes, picks):
        d = load(fp)
        fm = d['obs'].get('full_mesh')
        faces = d['cloth_faces']
        if fm is None or faces is None:
            ax.axis('off')
            continue
        # full_mesh is [12 grip || V*3 positions], /WBOX-normalized
        v = np.asarray(fm[0], np.float64)[12:].reshape(-1, 3) * WBOX
        keep = np.abs(v).sum(1) > 0            # padded vertex slots are zero
        v = v[keep]
        faces = np.asarray(faces)
        faces = faces[(faces < len(v)).all(1)]
        hr = float(d['hole_radius'] or 0) * SCALE_MM
        widths.append(float(d['cloth_width'] or 0) * SCALE_MM / 1000)
        radii.append(hr)
        mesh_panel(ax, v, faces,
                   f'{d.get("cloth_shape", "rect")} · density '
                   f'{d.get("node_density", 15)}\n'
                   f'{len(v)} verts · hole r {hr:.0f} mm')
    for ax in axes[len(picks):]:
        ax.axis('off')
    fig.suptitle(f'{a.tag} training cloths — every episode re-randomizes '
                 f'size, node count and hole placement', y=0.995)
    fig.tight_layout()
    p1 = os.path.join(a.out_dir, f'{a.tag}_meshes.png')
    fig.savefig(p1, dpi=120)
    print(f'wrote {p1}')

    # ---- figure 2: goal distribution -------------------------------------
    g_thread, g_chain, kinds = [], [], []
    for fp in files:
        d = load(fp)
        g = np.asarray(d['obs']['goal'], np.float64) * WBOX
        kinds.append(d['episode_kind'])
        (g_chain if d['episode_kind'] == 'chain' else g_thread).append(g)
    gt = np.concatenate(g_thread) if g_thread else np.zeros((0, 3))
    gc = np.concatenate(g_chain) if g_chain else np.zeros((0, 3))
    n_thread = sum(1 for k in kinds if k == 'thread')

    fig, ax = plt.subplots(1, 3, figsize=(18, 5.4))
    for i, (j, k, lj, lk) in enumerate([(0, 2, 'x', 'z'), (1, 2, 'y', 'z'),
                                        (0, 1, 'x', 'y')]):
        ax[i].scatter(gc[::7, j], gc[::7, k], s=2.5, alpha=.20, c='#e45756',
                      label=f'chain goals (n={len(gc)})')
        ax[i].scatter(gt[::7, j], gt[::7, k], s=6, alpha=.55, c='#4c78a8',
                      label=f'thread goals / peg (n={len(gt)})')
        ax[i].scatter([REAL_GOAL[j]], [REAL_GOAL[k]], marker='*', s=420,
                      c='#ff9d1e', edgecolors='k', zorder=6,
                      label='measured real hanger tip')
        ax[i].set(xlabel=f'sim {lj}', ylabel=f'sim {lk}',
                  title=f'goal positions · {lj}-{lk}')
        ax[i].grid(alpha=.25)
        ax[i].set_aspect('equal', adjustable='datalim')
    ax[0].legend(fontsize=8, loc='upper left')
    fig.suptitle(f'Where the policy is asked to put the hole — '
                 f'{n_thread} thread + {len(kinds)-n_thread} chain episodes '
                 f'(1 sim unit = 45 mm)')
    fig.tight_layout()
    p2 = os.path.join(a.out_dir, f'{a.tag}_goals.png')
    fig.savefig(p2, dpi=120)
    print(f'wrote {p2}')

    # ---- numbers ---------------------------------------------------------
    print('\n=== goal coverage ===')
    for name, g in (('thread (peg)', gt), ('chain (sampled)', gc)):
        if len(g) == 0:
            continue
        print(f'{name:18s} n={len(g):7d}  '
              f'x [{g[:,0].min():6.2f},{g[:,0].max():6.2f}]  '
              f'y [{g[:,1].min():6.2f},{g[:,1].max():6.2f}]  '
              f'z [{g[:,2].min():6.2f},{g[:,2].max():6.2f}]')
    allg = np.concatenate([gt, gc]) if len(gt) and len(gc) else (gt if len(gt) else gc)
    d_real = np.linalg.norm(allg - REAL_GOAL, axis=1)
    print(f'\nreal hanger tip {np.round(REAL_GOAL,2)} — nearest dataset goal '
          f'{d_real.min()*SCALE_MM:.1f} mm away; '
          f'{100*(d_real < 1.0).mean():.1f}% of goals within 1 sim unit (45 mm)')
    print(f'cloth width  mean {np.mean(widths):.3f} m  '
          f'range [{np.min(widths):.3f}, {np.max(widths):.3f}]  (real 0.209 m)')
    print(f'hole radius  mean {np.mean(radii):.1f} mm '
          f'range [{np.min(radii):.1f}, {np.max(radii):.1f}]  (real 25.0 mm)')

    # ---- figure 3: gripper start pose vs the rig --------------------------
    start_pose_figure(files, os.path.join(a.out_dir, f'{a.tag}_start_pose.png'),
                      a.tag)

    # ---- figure 4: expert success + hole-height traces by geometry --------
    if a.diag_dir:
        expert_figure(a.diag_dir,
                      os.path.join(a.out_dir, f'{a.tag}_expert_geometry.png'))


if __name__ == '__main__':
    main()
