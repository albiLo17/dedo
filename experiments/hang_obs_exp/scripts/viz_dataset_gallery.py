"""Build a self-contained HTML gallery for a collected state-est dataset.

Numbers alone can't tell you whether a cloth dataset is sound - you have to look
at the meshes, the holes, and the point clouds. This renders:

  1. rest-mesh gallery      topology diversity: every cloth is its own mesh, with
                            its own hole placement and vertex count
  2. episode filmstrips     GT mesh vs the partial cloud it must be recovered
                            from, across an episode
  3. coverage overlay       where the observed points actually land on the mesh
  4. debug videos           the collector's 3-panel MP4s, inlined

Everything is embedded as data URIs, so the output is one portable .html file
(no external assets) that can be attached to Notion or opened directly.

Usage:
  python experiments/hang_obs_exp/scripts/viz_dataset_gallery.py \
      --h5 experiments/hang_obs_exp/data/state_est/dedo_hang_real.h5 \
      --out experiments/hang_obs_exp/data/state_est/gallery.html
"""
import argparse
import base64
import glob
import io
import json
import os

import h5py
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d.art3d import Line3DCollection, Poly3DCollection

C_BLUE, C_ORANGE, C_AQUA = '#2a78d6', '#eb6834', '#1baf7a'
INK, INK_2, GRID = '#0b0b0b', '#52514e', '#d8d7d2'
SURFACE = '#fcfcfb'


def _png_uri(fig, dpi=110):
    buf = io.BytesIO()
    fig.savefig(buf, format='png', dpi=dpi, facecolor=fig.get_facecolor(),
                bbox_inches='tight')
    plt.close(fig)
    return 'data:image/png;base64,' + base64.b64encode(buf.getvalue()).decode()


def _equal_3d(ax, pts):
    """Cubic aspect - matplotlib 3D otherwise stretches axes independently and
    a draped cloth reads as a different shape than it is. 0.46 rather than a
    loose 0.55 because mpl3d already reserves generous margins; any larger and
    the cloth is a stamp in the middle of an empty panel."""
    c = pts.mean(axis=0)
    r = max(np.ptp(pts, axis=0).max(), 1e-6) * 0.46
    ax.set_xlim(c[0] - r, c[0] + r)
    ax.set_ylim(c[1] - r, c[1] + r)
    ax.set_zlim(c[2] - r, c[2] + r)
    ax.set_axis_off()
    try:
        ax.set_box_aspect((1, 1, 1))
    except Exception:
        pass


def _draw_mesh(ax, verts, faces, hole_idx=None, face_color='#c9d9ef',
               edge_color='#7c9dc7', pts=None):
    tris = verts[faces]
    ax.add_collection3d(Poly3DCollection(
        tris, facecolor=face_color, edgecolor=edge_color,
        linewidths=0.25, alpha=0.92))
    if hole_idx is not None and len(hole_idx):
        hv = verts[hole_idx]
        ax.scatter(hv[:, 0], hv[:, 1], hv[:, 2], s=9, c=C_ORANGE,
                   depthshade=False, zorder=6)
    if pts is not None and len(pts):
        ax.scatter(pts[:, 0], pts[:, 1], pts[:, 2], s=1.4, c=C_AQUA,
                   depthshade=False, alpha=0.75)
    _equal_3d(ax, verts)


def rest_gallery(f, split, n=8, seed=0):
    """One panel per cloth: the rest mesh with its hole loop marked."""
    keys = sorted(f[split].keys())
    if not keys:
        return None, []
    rng = np.random.default_rng(seed)
    pick = [keys[i] for i in rng.choice(len(keys), min(n, len(keys)), replace=False)]
    cols = min(4, len(pick))
    rows = int(np.ceil(len(pick) / cols))
    fig = plt.figure(figsize=(3.1 * cols, 3.0 * rows))
    fig.patch.set_facecolor(SURFACE)
    meta = []
    for i, ck in enumerate(pick):
        g = f[split][ck]
        verts = g['rest_positions'][:]
        faces = g['faces'][:]
        hole = np.asarray(g.attrs.get('hole_vertex_indices', []), dtype=int)
        ax = fig.add_subplot(rows, cols, i + 1, projection='3d')
        _draw_mesh(ax, verts, faces, hole)
        ax.set_title(f'{ck}\n{len(verts)} verts · {len(hole)} hole verts',
                     fontsize=7.5, color=INK, pad=1)
        ax.view_init(elev=18, azim=-62)
        meta.append(dict(cloth=ck, verts=int(len(verts)), hole=int(len(hole))))
    fig.suptitle('Rest meshes — every episode is its own topology '
                 '(orange = hole loop)', fontsize=10, color=INK, x=0.01, ha='left')
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    return _png_uri(fig), meta


def filmstrip(f, split, cloth_key, n_frames=5):
    """GT mesh (top) against the partial cloud it must be recovered from
    (bottom), sampled across the episode."""
    g = f[split][cloth_key]
    t = g['trajectory_0']
    steps = sorted(k for k in t if k.startswith('step'))
    if len(steps) < 2:
        return None
    idx = np.linspace(0, len(steps) - 1, min(n_frames, len(steps))).astype(int)
    faces = g['faces'][:]
    hole = np.asarray(g.attrs.get('hole_vertex_indices', []), dtype=int)
    fig = plt.figure(figsize=(2.7 * len(idx), 5.6))
    fig.patch.set_facecolor(SURFACE)
    for j, si in enumerate(idx):
        sg = t[steps[si]]
        pos = sg['positions'][:]
        pcd = sg['pointclouds']['cam_0'][:]
        ax = fig.add_subplot(2, len(idx), j + 1, projection='3d')
        _draw_mesh(ax, pos, faces, hole)
        ax.view_init(elev=16, azim=-62)
        ax.set_title(f'step {si}', fontsize=7.5, color=INK, pad=1)
        ax2 = fig.add_subplot(2, len(idx), len(idx) + j + 1, projection='3d')
        ax2.scatter(pcd[:, 0], pcd[:, 1], pcd[:, 2], s=1.6, c=C_AQUA,
                    depthshade=False, alpha=0.8)
        _equal_3d(ax2, pos)
        ax2.view_init(elev=16, azim=-62)
        npx = sg.attrs.get('valid_cloth_px', -1)
        ax2.set_title(f'{npx} unique px', fontsize=7, color=INK_2, pad=1)
    fig.suptitle(f'{cloth_key} — GT mesh (top) vs the single-view cloud the '
                 f'estimator sees (bottom)', fontsize=10, color=INK,
                 x=0.01, ha='left')
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    return _png_uri(fig)


def overlay(f, split, cloth_key, step_frac=0.6):
    """Mesh + observed cloud in one axes: shows WHICH side is observed and how
    much of the surface the single view misses."""
    g = f[split][cloth_key]
    t = g['trajectory_0']
    steps = sorted(k for k in t if k.startswith('step'))
    sg = t[steps[int(step_frac * (len(steps) - 1))]]
    pos, pcd = sg['positions'][:], sg['pointclouds']['cam_0'][:]
    faces = g['faces'][:]
    hole = np.asarray(g.attrs.get('hole_vertex_indices', []), dtype=int)
    fig = plt.figure(figsize=(10.5, 3.4))
    fig.patch.set_facecolor(SURFACE)
    for j, (elev, azim, label) in enumerate((
            (16, -62, 'camera-ish'), (16, 28, 'rotated 90°'), (78, -62, 'top-down'))):
        ax = fig.add_subplot(1, 3, j + 1, projection='3d')
        _draw_mesh(ax, pos, faces, hole, pts=pcd)
        ax.view_init(elev=elev, azim=azim)
        ax.set_title(label, fontsize=8, color=INK_2, pad=1)
    fig.suptitle(f'{cloth_key} — observed points (aqua) on the GT mesh; the '
                 f'unobserved side is what the belief has to invent',
                 fontsize=10, color=INK, x=0.01, ha='left')
    fig.tight_layout(rect=[0, 0, 1, 0.9])
    return _png_uri(fig)


def video_uris(h5_path, limit=3, max_mb=6.0):
    """Inline the collector's 3-panel debug MP4s as data URIs."""
    pre = os.path.splitext(h5_path)[0]
    out = []
    for p in sorted(glob.glob(pre + '_simrgb_*.mp4'))[:limit]:
        mb = os.path.getsize(p) / 1e6
        if mb > max_mb:
            print(f'  skipping {os.path.basename(p)} ({mb:.1f} MB > {max_mb})')
            continue
        with open(p, 'rb') as fh:
            out.append((os.path.basename(p),
                        'data:video/mp4;base64,'
                        + base64.b64encode(fh.read()).decode()))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--h5', required=True)
    ap.add_argument('--out', default='')
    ap.add_argument('--n_rest', type=int, default=8)
    ap.add_argument('--n_episodes', type=int, default=2)
    args = ap.parse_args()
    out = args.out or (os.path.splitext(args.h5)[0] + '_gallery.html')

    stats_path = os.path.splitext(args.h5)[0] + '_stats.json'
    stats = json.load(open(stats_path)) if os.path.exists(stats_path) else None

    with h5py.File(args.h5, 'r') as f:
        splits = [s for s in ('training', 'validation') if s in f and len(f[s])]
        n_cloths = {s: len(f[s]) for s in splits}
        rest_png, rest_meta = rest_gallery(f, splits[0], n=args.n_rest)
        val_png, _ = (rest_gallery(f, 'validation', n=4, seed=3)
                      if 'validation' in splits else (None, []))
        strips, ovs = [], []
        for s in splits:
            for ck in sorted(f[s].keys())[:args.n_episodes]:
                st = filmstrip(f, s, ck)
                if st:
                    strips.append((f'{s}/{ck}', st))
                    ovs.append((f'{s}/{ck}', overlay(f, s, ck)))
                if len(strips) >= args.n_episodes:
                    break
            if len(strips) >= args.n_episodes:
                break

    vids = video_uris(args.h5)

    cfg = (stats or {}).get('config', {})
    eps = (stats or {}).get('episodes', [])
    tot_frames = int(np.sum([e['num_steps'] for e in eps])) if eps else 0
    cov = np.asarray([e['valid_px_mean'] for e in eps], float) if eps else np.array([0.])
    npts = cfg.get('pcd_n_points', 2048)

    def section(title, body, lede=''):
        led = f'<p class="lede">{lede}</p>' if lede else ''
        return f'<section><h2>{title}</h2>{led}{body}</section>'

    pct_full = 100 * np.mean(cov >= npts)
    cov_state = 'ok' if pct_full >= 75 else ('warn' if pct_full >= 40 else 'bad')
    rz = cfg.get('randomize', {})

    def stat(value, label, state=''):
        chip = f' <span class="chip {state}"></span>' if state else ''
        return (f'<div class="stat"><div class="num">{value}{chip}</div>'
                f'<div class="lbl">{label}</div></div>')

    stats_strip = '<div class="strip">' + ''.join([
        stat(f'{sum(n_cloths.values())}', 'cloths (= topologies)'),
        stat(f'{tot_frames:,}', 'frames'),
        stat(f'{np.median(cov):.0f}', f'median unique cloth px', cov_state),
        stat(f'{pct_full:.0f}%', f'episodes ≥ {npts}-pt budget', cov_state),
        stat(f"{cfg.get('n_train','?')}/{cfg.get('n_val','?')}", 'train / val cloths'),
        stat(f"{cfg.get('elapsed_s',0)/60:.0f}m", 'collection wall-clock'),
    ]) + '</div>'

    html = [f"""<header>
<p class="eyebrow">{cfg.get('env','?')} · seed {cfg.get('seed','?')}</p>
<h1>DEDO hang — collected dataset</h1>
<p class="sub">One collection feeding both the state estimator and the dynamics
model. Randomized per episode: cloth geometry, camera yaw/pitch/distance, cloth
physics, peg pose (xy + height), depth noise and dropout.</p>
{stats_strip}
</header>"""]

    html.append(section(
        'Topology diversity',
        f'<figure><img src="{rest_png}" alt="rest meshes"></figure>'
        + (f'<p class="cap">validation cloths — never seen in training</p>'
           f'<figure><img src="{val_png}" alt="validation rest meshes"></figure>'
           if val_png else ''),
        lede='Each episode regenerates the cloth, so <b>every cloth group is a '
             'different mesh</b> — its own vertex count, aspect ratio and hole '
             'placement. That is what makes the train/validation split a '
             'held-out <b>topology</b> split rather than held-out trajectories '
             'of one garment.'))

    if strips:
        html.append(section(
            'What the estimator is asked to do',
            ''.join(f'<p class="cap">{name}</p>'
                    f'<figure><img src="{uri}" alt="filmstrip"></figure>'
                    for name, uri in strips),
            lede='Top row is the ground-truth mesh; bottom row is the single-view '
                 'point cloud it must be recovered from. The per-frame count is '
                 '<b>unique</b> cloth pixels — clouds are always padded to the '
                 'point budget, so that number is the only place real coverage '
                 'is visible.'))
    if ovs:
        html.append(section(
            'Single-view coverage',
            ''.join(f'<p class="cap">{name}</p>'
                    f'<figure><img src="{uri}" alt="coverage overlay"></figure>'
                    for name, uri in ovs),
            lede='The same frame from three angles, with observed points in aqua '
                 'on the ground-truth mesh. The unobserved side is what a belief '
                 'has to invent — and where the hole goes when it turns away.'))
    if vids:
        html.append(section(
            'Collection footage',
            ''.join(f'<p class="cap">{n}</p>'
                    f'<figure><video controls loop muted playsinline src="{u}">'
                    f'</video></figure>' for n, u in vids),
            lede='Three panels per frame: the simulator render, the back-projected '
                 'cloud overlaid on the observation RGB, and that cloud seen from '
                 'the camera. If the overlay drifts off the cloth, the capture '
                 'geometry is wrong.'))
    if rest_meta:
        rows = ''.join(f"<tr><td>{m['cloth']}</td><td>{m['verts']}</td>"
                       f"<td>{m['hole']}</td></tr>" for m in rest_meta)
        html.append(section('Sampled cloths', f"""<div class="tablewrap"><table>
<thead><tr><th>cloth</th><th>vertices</th><th>hole verts</th></tr></thead>
<tbody>{rows}</tbody></table></div>"""))

    # Utility review page: the palette is the data encoding itself (mesh blue,
    # hole orange, observed aqua) so the page and the figures read as one system.
    # Neutrals carry a slight cool bias rather than a flat grey.
    style = """<style>
:root{
  --ground:#fbfbfa; --panel:#ffffff; --ink:#12131a; --muted:#5a5d68;
  --rule:#e3e4e6; --mesh:#2a78d6; --hole:#eb6834; --obs:#1baf7a;
  --ok:#1baf7a; --warn:#eda100; --bad:#e34948;
  color-scheme:light;
}
@media(prefers-color-scheme:dark){:root:where(:not([data-theme="light"])){
  --ground:#131416; --panel:#191b1e; --ink:#f2f3f5; --muted:#a2a6b0;
  --rule:#2b2d31; --mesh:#3987e5; --hole:#d95926; --obs:#199e70;
  color-scheme:dark;
}}
:root[data-theme="dark"]{
  --ground:#131416; --panel:#191b1e; --ink:#f2f3f5; --muted:#a2a6b0;
  --rule:#2b2d31; --mesh:#3987e5; --hole:#d95926; --obs:#199e70;
  color-scheme:dark;
}
*{box-sizing:border-box}
body{margin:0;background:var(--ground);color:var(--ink);
  font:15px/1.6 ui-sans-serif,-apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif;
  -webkit-font-smoothing:antialiased}
.wrap{max-width:1120px;margin:0 auto;padding:40px 24px 72px}
header{display:flex;flex-direction:column;gap:10px;margin-bottom:8px}
.eyebrow{margin:0;font:600 11px/1 ui-monospace,SFMono-Regular,Menlo,monospace;
  letter-spacing:.11em;text-transform:uppercase;color:var(--muted)}
h1{margin:0;font-size:clamp(25px,3.4vw,34px);line-height:1.15;
  letter-spacing:-.021em;text-wrap:balance}
.sub{margin:0;color:var(--muted);max-width:64ch}
.strip{display:grid;grid-template-columns:repeat(auto-fit,minmax(148px,1fr));
  gap:1px;background:var(--rule);border:1px solid var(--rule);border-radius:9px;
  overflow:hidden;margin-top:14px}
.stat{background:var(--panel);padding:13px 15px}
.num{font:600 21px/1.15 ui-monospace,SFMono-Regular,Menlo,monospace;
  font-variant-numeric:tabular-nums;display:flex;align-items:center;gap:7px}
.lbl{margin-top:3px;font-size:11.5px;color:var(--muted);letter-spacing:.01em}
.chip{width:8px;height:8px;border-radius:50%;display:inline-block;flex:none}
.chip.ok{background:var(--ok)} .chip.warn{background:var(--warn)}
.chip.bad{background:var(--bad)}
section{margin-top:42px}
h2{margin:0 0 6px;font-size:17px;letter-spacing:-.012em;
  padding-bottom:8px;border-bottom:1px solid var(--rule)}
.lede{margin:10px 0 16px;color:var(--muted);max-width:70ch}
.lede b{color:var(--ink);font-weight:600}
.cap{margin:20px 0 6px;font:500 12px/1.4 ui-monospace,SFMono-Regular,Menlo,monospace;
  color:var(--muted);letter-spacing:.02em}
figure{margin:0}
img,video{max-width:100%;height:auto;display:block;border-radius:8px;
  border:1px solid var(--rule);background:var(--panel)}
video{width:100%}
.tablewrap{overflow-x:auto}
table{border-collapse:collapse;font:13px/1.5 ui-monospace,SFMono-Regular,Menlo,monospace;
  font-variant-numeric:tabular-nums;min-width:320px}
td,th{padding:6px 20px 6px 0;text-align:left;border-bottom:1px solid var(--rule)}
th{color:var(--muted);font-weight:600;font-size:11px;letter-spacing:.06em;
  text-transform:uppercase}
a{color:var(--mesh)}
</style>"""
    with open(out, 'w') as fh:
        fh.write(f'<title>DEDO hang dataset</title>{style}'
                 f'<div class="wrap">' + '\n'.join(html) + '</div>')
    print(f'wrote {out} ({os.path.getsize(out)/1e6:.1f} MB, '
          f'{len(strips)} filmstrips, {len(vids)} videos)')


if __name__ == '__main__':
    main()
