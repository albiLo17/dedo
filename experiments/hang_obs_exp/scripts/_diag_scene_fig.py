"""Side-by-side of what the peg assembly actually looks like in each geometry.

Everything else in this diagnosis is a number. This is the picture: the same
camera, the same cloth, three scenes. In the v5 scene the hanger sits ON TOP of
the support post, so the cloth's target is the highest thing around. In the v6
scene the hanger was lowered to the measured real tip and the post was not, so
a bare post stands 1.79 sim units (80 mm) above the target the expert aims for.

Frames come from the collector's own debug videos, so this is the scene as the
episodes actually ran, not a re-render.

    python _diag_scene_fig.py --diag_dir .../diag_expert --out .../scene.png
"""
import argparse
import glob
import os
import subprocess

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt                                  # noqa: E402
import matplotlib.image as mpimg                                 # noqa: E402

PANELS = [
    ('A', 'A — v5 scene\nhanger ON TOP of the post', -0.198),
    ('B', 'B — v6 as shipped\npost spikes ABOVE the hanger', 1.789),
    ('E', 'E — rod moved with the peg\nhanger back on top', -0.198),
]


def first_frame(diag_dir, gid, tmp):
    vids = sorted(glob.glob(os.path.join(diag_dir, gid, 'debug_viz',
                                         'demo_*_video.mp4')))
    if not vids:
        return None
    out = os.path.join(tmp, f'{gid}.png')
    r = subprocess.run(['ffmpeg', '-y', '-loglevel', 'error', '-i', vids[0],
                        '-vframes', '1', out], capture_output=True)
    return out if r.returncode == 0 else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--diag_dir', required=True)
    ap.add_argument('--out', required=True)
    a = ap.parse_args()
    tmp = os.path.join(os.path.dirname(a.out), '_frames')
    os.makedirs(tmp, exist_ok=True)

    have = [(g, t, rod) for g, t, rod in PANELS
            if first_frame(a.diag_dir, g, tmp)]
    fig, axes = plt.subplots(1, len(have), figsize=(5.6 * len(have), 4.4))
    axes = np.atleast_1d(axes)
    for ax, (gid, title, rod) in zip(axes, have):
        img = mpimg.imread(os.path.join(tmp, f'{gid}.png'))
        # The debug frame is a 3-up strip (sim | segmented | point cloud); the
        # leftmost third is the plain sim view, which is the one to show.
        ax.imshow(img[:, :img.shape[1] // 3])
        ax.set_title(title, fontsize=11,
                     color='#b3261e' if rod > 0 else '#1e6b2f')
        ax.text(.5, -.06, f'rod top {rod:+.2f} relative to the goal point',
                transform=ax.transAxes, ha='center', fontsize=9.5,
                color='#b3261e' if rod > 0 else '#1e6b2f',
                fontweight='bold' if rod > 0 else 'normal')
        ax.axis('off')
    fig.suptitle('The peg assembly in each geometry — same camera, same cloth. '
                 'tallrod.urdf is exactly 8.0 sim units tall, so the preset '
                 'paired hanger z=8.0 with rod z=0;\nv6 lowered the hanger to '
                 'the measured tip and left the rod on the floor.', fontsize=10)
    fig.tight_layout(rect=(0, 0.02, 1, 0.93))
    fig.savefig(a.out, dpi=130)
    print(f'wrote {a.out}')


if __name__ == '__main__':
    main()
