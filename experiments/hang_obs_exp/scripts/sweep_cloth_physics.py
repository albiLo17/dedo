"""
Side-by-side physics-sweep VIDEO harness for the main (sim-scale)
HangProcCloth.

Runs the scripted hole-aware hang trajectory for every (bending, damping)
combination on the SAME procedural cloth (identical seed -> identical mesh
and holes, so only the physics differs), records the FULL episode
(trajectory + post-trajectory gravity settle) and writes:

  <out>/cell_b<bend>_d<damp>.mp4   — one video per combination
  <out>/sweep_grid.mp4             — all combos tiled into one synchronized
                                     grid video (rows=bending, cols=damping)

The grid video is the side-by-side: every cell plays the same moment of
the same cloth at once, so "flowiness" (billow / whip / settle) is directly
comparable. An exploded/failed cell shows a red FAILED tile.

Example:
  python experiments/hang_obs_exp/scripts/sweep_cloth_physics.py \
      --bending 1,10,30 --damping 0.01,0.1,0.5 --seed 42

Output defaults inside the repo (experiments/hang_obs_exp/cloth_sweep)
so it shows up in the editor file tree, not /tmp.
"""
import argparse
import os
import sys

import numpy as np

# --- harness args (parsed before we hijack sys.argv for dedo) -------------
_p = argparse.ArgumentParser(description=__doc__,
                             formatter_class=argparse.RawDescriptionHelpFormatter)
_p.add_argument('--bending', type=str, default='1,10,30',
                help='Comma list of springBendingStiffness values (rows).')
_p.add_argument('--damping', type=str, default='0.01,0.1,0.5',
                help='Comma list of springDampingStiffness values (cols).')
_p.add_argument('--elastic', type=float, default=50.0,
                help='springElasticStiffness, held fixed across the sweep.')
_p.add_argument('--seed', type=int, default=42,
                help='Seed — fixed so the cloth mesh is identical per cell.')
_p.add_argument('--sim_freq', type=int, default=500,
                help='Higher = smaller timestep; raise if stiff cells explode.')
_p.add_argument('--steps_per_action', type=int, default=8)
_p.add_argument('--render_size', type=int, default=300)
_p.add_argument('--frame_stride', type=int, default=2,
                help='Record every Nth control step during the trajectory.')
_p.add_argument('--settle_stride', type=int, default=2,
                help='Record every Nth sampled sub-step during the settle.')
_p.add_argument('--fps', type=int, default=20, help='Output video fps.')
_p.add_argument('--waypoint_scale', type=float, default=1.0,
                help='1.0 for the sim-scale HangProcCloth (matches collector).')
_p.add_argument('--out', type=str,
                default=os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                     '..', 'cloth_sweep'),
                help='Output dir. Default is inside the repo '
                     '(experiments/hang_obs_exp/cloth_sweep) so it shows '
                     'up in the editor file tree, not /tmp.')
extra = _p.parse_args()

bending_vals = [float(x) for x in extra.bending.split(',')]
damping_vals = [float(x) for x in extra.damping.split(',')]
extra.out = os.path.abspath(extra.out)
os.makedirs(extra.out, exist_ok=True)

# --- build dedo args (sys.argv hijack pattern from collect_bc_demos.py) ---
sys.argv = [
    'sweep_cloth_physics',
    '--env=HangProcCloth-v1',
    '--cam_resolution=-1',           # low-dim obs; we render() directly
    '--seed', str(extra.seed),
    '--max_episode_len', '1000',     # overridden per-episode to len(traj)
    f'--sim_freq={extra.sim_freq}',
    f'--sim_steps_per_action={extra.steps_per_action}',
]
from dedo.utils.args import get_args_parser, args_postprocess  # noqa: E402
from dedo.utils.task_info import DEFORM_INFO                    # noqa: E402
from dedo.envs.deform_env import DeformEnv                      # noqa: E402
import gym                                                      # noqa: E402
import cv2                                                      # noqa: E402
import imageio                                                  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _helpers import RetryResetEnv, build_hole_aware_waypoints  # noqa: E402
from dedo.demo_preset import build_traj, merge_traj             # noqa: E402

args, _ = get_args_parser()
args_postprocess(args)
args.debug = False
args.viz = False

# Snapshot the static preset keys. gen_procedural_hang_cloth derives its
# temp .obj filename from np.random; since we reseed to the SAME seed every
# cell (for an identical mesh), that filename collides across cells, and
# gen_* only refreshes DEFORM_INFO[path]=base.copy() when the path is a NEW
# key. So after cell 1 every later cell would reuse cell 1's cached physics.
# Pruning these procedural keys before each cell forces a fresh copy of the
# (newly mutated) base preset — identical cloth, but the intended physics.
_ORIG_DEFORM_KEYS = set(DEFORM_INFO.keys())


def _unwrap(env):
    u = env
    while hasattr(u, 'env'):
        u = u.env
        if isinstance(u, DeformEnv):
            return u
    return u


def _label(img, text):
    img = np.ascontiguousarray(img)
    cv2.putText(img, text, (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                (0, 0, 0), 3, cv2.LINE_AA)
    cv2.putText(img, text, (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                (255, 255, 0), 1, cv2.LINE_AA)
    return img


def run_cell(bending, damping):
    """Run one (bending, damping) episode; return list of RGB frames
    spanning the trajectory AND the post-trajectory gravity settle."""
    # Drop stale procedural temp entries so the mutated base preset below
    # is freshly copied into the (seed-collided) temp .obj key this cell.
    for k in list(DEFORM_INFO.keys()):
        if k not in _ORIG_DEFORM_KEYS:
            del DEFORM_INFO[k]

    DEFORM_INFO['procedural_hang_cloth']['deform_elastic_stiffness'] = extra.elastic
    DEFORM_INFO['procedural_hang_cloth']['deform_bending_stiffness'] = bending
    DEFORM_INFO['procedural_hang_cloth']['deform_damping_stiffness'] = damping

    np.random.seed(extra.seed)  # identical procedural cloth across cells
    env = gym.make(args.env, args=args)
    env = RetryResetEnv(env)
    env.seed(extra.seed)
    env.reset()
    deform = _unwrap(env)

    # Built-in hook: capture frames during make_final_steps (the gravity
    # settle that runs internally when the episode ends).
    deform._record_settle_frames = True
    deform._settle_render_kwargs = dict(width=extra.render_size,
                                        height=extra.render_size)
    deform._settle_frame_stride = extra.settle_stride

    wp = build_hole_aware_waypoints(deform, waypoint_scale=extra.waypoint_scale)
    if wp is None:
        env.close()
        raise RuntimeError('build_hole_aware_waypoints returned None')
    ctrl_freq = args.sim_freq / args.sim_steps_per_action
    _, va = build_traj(deform, wp, 'a', anchor_idx=0,
                       ctrl_freq=ctrl_freq, robot=None)
    _, vb = build_traj(deform, wp, 'b', anchor_idx=1,
                       ctrl_freq=ctrl_freq, robot=None)
    traj = merge_traj(va, vb)
    deform.max_episode_len = len(traj)

    frames = []
    for step in range(len(traj)):
        act = np.clip(traj[step] / DeformEnv.MAX_ACT_VEL,
                      -1.0, 1.0).astype(np.float32)
        _, _, done, info = env.step(act)
        if step % extra.frame_stride == 0:
            frames.append(deform.render(mode='rgb_array',
                                        width=extra.render_size,
                                        height=extra.render_size))
        if done:
            frames.extend(info.get('settle_frames', []))
            break
    env.close()
    return [np.asarray(f, dtype=np.uint8) for f in frames]


def main():
    nrows, ncols = len(bending_vals), len(damping_vals)
    cells = {}          # (bi, di) -> list[frame] or None
    for bi, b in enumerate(bending_vals):
        for di, d in enumerate(damping_vals):
            tag = f'b={b:g} d={d:g}'
            print(f'[sweep] running {tag} ...', flush=True)
            try:
                fr = run_cell(b, d)
                cells[(bi, di)] = fr
                print(f'[sweep]   done {tag} ({len(fr)} frames)', flush=True)
                path = os.path.join(extra.out, f'cell_b{b:g}_d{d:g}.mp4')
                imageio.mimwrite(path, [_label(f.copy(), tag) for f in fr],
                                 fps=extra.fps, codec='libx264', quality=8,
                                 macro_block_size=None)
                print(f'[sweep]   wrote {path}', flush=True)
            except Exception as e:
                cells[(bi, di)] = None
                print(f'[sweep]   FAILED {tag}: {e!r}', flush=True)

    # --- synchronized grid video ----------------------------------------
    lengths = [len(v) for v in cells.values() if v]
    if not lengths:
        print('[sweep] all cells failed; no grid video.', flush=True)
        return
    n_frames = max(lengths)
    rs = extra.render_size
    fail_tile = np.zeros((rs, rs, 3), dtype=np.uint8)
    cv2.putText(fail_tile, 'FAILED', (rs // 2 - 70, rs // 2),
                cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 0, 0), 2, cv2.LINE_AA)

    grid_path = os.path.join(extra.out, 'sweep_grid.mp4')
    writer = imageio.get_writer(grid_path, fps=extra.fps, codec='libx264',
                                quality=8, macro_block_size=None)
    for t in range(n_frames):
        rows = []
        for bi, b in enumerate(bending_vals):
            row = []
            for di, d in enumerate(damping_vals):
                fr = cells.get((bi, di))
                if fr:
                    img = fr[min(t, len(fr) - 1)].copy()  # hold last frame
                else:
                    img = fail_tile.copy()
                row.append(_label(img, f'b={b:g} d={d:g}'))
            rows.append(np.hstack(row))
        writer.append_data(np.vstack(rows))
    writer.close()
    print(f'[sweep] wrote {grid_path}  ({n_frames} frames, '
          f'{nrows}x{ncols} grid)', flush=True)


if __name__ == '__main__':
    main()
