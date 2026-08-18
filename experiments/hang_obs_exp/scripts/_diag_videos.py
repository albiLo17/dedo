"""Pick the videos worth watching out of a _diag_expert.sh run.

The collector writes one video per episode for the first --debug_viz_first_n
attempts, named demo_NNN_video.mp4, and the outcome lives in the matching
demo_NNN.pkl. This pairs them, keeps a few of each outcome per geometry, and
builds A-vs-B side-by-sides on the SAME episode index — which, because all four
geometries share --seed, means the same cloth, the same hole and the same
jitter draws, so the only thing that differs on screen is the scene geometry.

    python _diag_videos.py --diag_dir .../diag_expert --out_dir .../videos
"""
import argparse
import glob
import os
import pickle
import shutil
import subprocess


LABELS = {
    'A': 'A: v5 scene', 'B': 'B: post left on the floor',
    'C': 'C: cloth moved, post not', 'D': 'D: rig start, post not moved',
    'E': 'E: post moved with the peg', 'F': 'F: rig start + post moved',
    'F2': 'F2: post moved, dz fixed', 'G': 'G: no post at all',
    'H': 'H: + mesh variation', 'I': 'I: no post + mesh variation',
}


def outcomes(gdir):
    """{episode index: success_legacy} for one geometry."""
    out = {}
    for fp in sorted(glob.glob(os.path.join(gdir, 'demo_*.pkl'))):
        idx = int(os.path.basename(fp)[5:-4])
        with open(fp, 'rb') as f:
            out[idx] = bool(pickle.load(f)['success_legacy'])
    return out


def video_for(gdir, idx):
    p = os.path.join(gdir, 'debug_viz', f'demo_{idx:03d}_video.mp4')
    return p if os.path.exists(p) else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--diag_dir', required=True)
    ap.add_argument('--out_dir', required=True)
    ap.add_argument('--n_each', type=int, default=3)
    ap.add_argument('--geometries', default='ABCD')
    ap.add_argument('--pairs', default=None,
                    help='Comma-separated LEFT:RIGHT side-by-sides on the same '
                         'episode, e.g. "A:B,B:E,F2:G".')
    a = ap.parse_args()
    os.makedirs(a.out_dir, exist_ok=True)

    res = {}
    for gid in (a.geometries.split(',') if ',' in a.geometries
                else list(a.geometries)):
        gdir = os.path.join(a.diag_dir, gid)
        if not os.path.isdir(gdir):
            continue
        res[gid] = outcomes(gdir)
        kept = {'ok': 0, 'fail': 0}
        for idx in sorted(res[gid]):
            tag = 'ok' if res[gid][idx] else 'fail'
            if kept[tag] >= a.n_each:
                continue
            src = video_for(gdir, idx)
            if src is None:
                continue          # outside the --debug_viz_first_n window
            dst = os.path.join(a.out_dir, f'{gid}_{tag}_ep{idx:03d}.mp4')
            shutil.copyfile(src, dst)
            kept[tag] += 1
        print(f'{gid}: copied {kept["ok"]} success + {kept["fail"]} failure '
              f'videos (of {sum(res[gid].values())}/{len(res[gid])} successes)')

    # --- side-by-side pairs on the SAME episode ------------------------------
    # Because every geometry ran on the same --seed, episode i is the same
    # cloth, the same hole and the same jitter draw in both runs -- so a pair
    # differs only by the thing under test. Prefer an episode where the left
    # succeeds and the right does not; fall back to the lowest shared index so
    # there is always something to watch.
    for spec in (a.pairs or '').split(',') if a.pairs else []:
        left, _, right = spec.partition(':')
        left, right = left.strip(), right.strip()
        if left not in res or right not in res:
            print(f'skipping pair {left}:{right} — missing run')
            continue
        ldir = os.path.join(a.diag_dir, left)
        rdir = os.path.join(a.diag_dir, right)
        shared = [i for i in sorted(res[left])
                  if i in res[right] and video_for(ldir, i) and video_for(rdir, i)]
        if not shared:
            print(f'skipping pair {left}:{right} — no episode has both videos')
            continue
        contrast = [i for i in shared if res[left][i] and not res[right][i]]
        for i in (contrast or shared)[:1]:
            va, vb = video_for(ldir, i), video_for(rdir, i)
            dst = os.path.join(a.out_dir, f'{left}vs{right}_ep{i:03d}.mp4')
            la = LABELS.get(left, left)
            lb = LABELS.get(right, right)
            # scale2ref pads the shorter clip's frame size to match; the clips
            # can also differ in length, so tpad holds the last frame of
            # whichever ends first rather than cutting the comparison short.
            # tpad holds the last frame of whichever clip ends first, so the
            # two stay aligned instead of one cutting the comparison short.
            def _dt(txt):
                esc = txt.replace('\\', '').replace(':', '\\:').replace(',', '\\,')
                return (f'drawtext=text={esc}:x=8:y=8:fontsize=15:'
                        f'fontcolor=white:box=1:boxcolor=black@0.65')
            cmd = [
                'ffmpeg', '-y', '-loglevel', 'error', '-i', va, '-i', vb,
                '-filter_complex',
                f'[0:v]tpad=stop_mode=clone:stop_duration=6,{_dt(la)}[l];'
                f'[1:v]tpad=stop_mode=clone:stop_duration=6,{_dt(lb)}[r];'
                f'[l][r]hstack=inputs=2,trim=duration=30',
                '-pix_fmt', 'yuv420p', dst,
            ]
            r = subprocess.run(cmd, capture_output=True, text=True)
            if r.returncode == 0:
                note = (f'{left} ok, {right} failed'
                        if (res[left][i] and not res[right][i])
                        else f'{left}={res[left][i]} {right}={res[right][i]}')
                print(f'wrote {dst}  (episode {i}: {note})')
            else:
                print(f'ffmpeg failed for {left}vs{right} ep {i}:\n{r.stderr[-800:]}')

    print(f'\nvideos in {a.out_dir}:')
    for f in sorted(os.listdir(a.out_dir)):
        print('  ', f, f'{os.path.getsize(os.path.join(a.out_dir, f))/1e6:.1f} MB')


if __name__ == '__main__':
    main()
