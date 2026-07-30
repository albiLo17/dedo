"""
Diagnostic: visualize the camera view across the xy randomization box
for the HangProcCloth hanger goal. Renders an n_grid x n_grid mosaic
where each cell shows the scene with the hanger displaced to that
(dx, dy) corner of the [-r, +r]^2 box.

Decoupled from env-side randomization on purpose: directly re-poses
the hanger + tallrod via setBasePositionAndOrientation post-reset, so
you can validate any --cam_viewmat / --radius pair before committing
to data collection code.

Usage:
  python experiments/hang_obs_exp/scripts/_diag_goal_randomization.py \
      --radius 1.5 \
      --cam_viewmat 14 -5 45 0 0 5.5 \
      --save_path logs/hang_obs_exp/diag/cam_v3_r1.5.png

  # Re-use for a new viewpoint you want to try:
  python experiments/hang_obs_exp/scripts/_diag_goal_randomization.py \
      --radius 2.0 \
      --cam_viewmat 12 -25 90 0 0 6.0 \
      --save_path logs/hang_obs_exp/diag/cam_perp_r2.0.png
"""
import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT))

import numpy as np
import gym
import matplotlib.pyplot as plt

import dedo  # noqa: F401
from dedo.utils.args import get_args_parser, args_postprocess
from dedo.utils.anchor_utils import create_anchor_geom
from dedo.envs.deform_env import DeformEnv

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _bc_obs_helpers import patch_deform_render_to_obs_camera  # noqa: E402


def _parse():
    p = argparse.ArgumentParser()
    p.add_argument('--radius', type=float, default=1.5,
                   help='Half-extent of the xy randomization box (meters)')
    p.add_argument('--n_grid', type=int, default=3,
                   help='Mosaic side length: n_grid x n_grid camera renders')
    p.add_argument('--cam_viewmat', type=float, nargs=6,
                   default=(14.0, -5.0, 45.0, 0.0, 0.0, 5.5),
                   help='[dist, pitch, yaw, tx, ty, tz] — matches collect script')
    p.add_argument('--cam_resolution', type=int, default=256,
                   help='Render resolution per panel (256 default; bigger '
                        'than the 128 used at training so cropping is obvious)')
    p.add_argument('--seed', type=int, default=42,
                   help='Procedural cloth seed; reset before every panel so '
                        'the cloth is identical and only the peg moves')
    p.add_argument('--save_path', type=str,
                   default='diag_goal_randomization.png')
    p.add_argument('--no_mark_goal', action='store_true',
                   help='Disable the green marker at goal_pos')
    return p.parse_args()


def _unwrap_deform(env):
    while hasattr(env, 'env'):
        env = env.env
        if isinstance(env, DeformEnv):
            return env
    return env if isinstance(env, DeformEnv) else None


def main():
    extra = _parse()
    r, n = extra.radius, extra.n_grid

    coords = np.linspace(-r, r, n)
    # dy reversed so row 0 (top of mosaic) is +y (far from a yaw=45 camera).
    dxdy_grid = [(float(dx), float(dy))
                 for dy in coords[::-1] for dx in coords]

    sys.argv = [
        '_diag_goal_randomization',
        '--env=HangProcCloth-v1',
        '--cam_resolution', str(extra.cam_resolution),
        '--cam_viewmat', *[f'{x:.6f}' for x in extra.cam_viewmat],
        '--num_envs=0', '--total_env_steps=0',
        '--seed', str(extra.seed),
    ]
    args, _ = get_args_parser()
    args_postprocess(args)
    args.viz = False
    args.debug = False

    env = gym.make(args.env, args=args)
    env.seed(extra.seed)
    deform = _unwrap_deform(env)
    assert deform is not None, 'env unwrap failed: not a DeformEnv'
    # Patch deform.render() so the mosaic uses the obs-camera projection
    # (fov=60) instead of dedo's DEFAULT_CAM_PROJECTION (fov≈90). Without
    # this the mosaic would show a wider field of view than the actual
    # training-time obs camera, so a peg that looked "comfortably in
    # frame" in the diag could still be at the edge of the obs frame.
    patch_deform_render_to_obs_camera(deform)

    env.reset()
    nominal_hanger, _ = deform.sim.getBasePositionAndOrientation(
        deform.rigid_ids[0])
    nominal_tallrod, _ = deform.sim.getBasePositionAndOrientation(
        deform.rigid_ids[1])
    nominal_goal = np.asarray(deform.goal_pos[0], dtype=np.float32).copy()
    print(f'[diag] nominal hanger  = {nominal_hanger}')
    print(f'[diag] nominal tallrod = {nominal_tallrod}')
    print(f'[diag] nominal goal    = {nominal_goal}')
    print(f'[diag] sampling {n}x{n} grid in [-{r}, +{r}]^2 xy')

    frames = []
    for (dx, dy) in dxdy_grid:
        env.seed(extra.seed)  # identical cloth across panels
        env.reset()
        deform = _unwrap_deform(env)

        new_hanger = (nominal_hanger[0] + dx,
                      nominal_hanger[1] + dy,
                      nominal_hanger[2])
        new_tallrod = (nominal_tallrod[0] + dx,
                       nominal_tallrod[1] + dy,
                       nominal_tallrod[2])
        deform.sim.resetBasePositionAndOrientation(
            deform.rigid_ids[0], new_hanger, [0, 0, 0, 1])
        deform.sim.resetBasePositionAndOrientation(
            deform.rigid_ids[1], new_tallrod, [0, 0, 0, 1])

        new_goal = nominal_goal + np.array([dx, dy, 0], dtype=np.float32)
        deform.goal_pos[0] = new_goal
        if deform.goal_pos.shape[0] > 1:
            # HangProcCloth duplicates goal_pos for the two-hole reward
            # convention; mirror that here.
            deform.goal_pos[1] = new_goal

        if not extra.no_mark_goal:
            create_anchor_geom(deform.sim, new_goal.tolist(), mass=0.0,
                               radius=0.15, rgba=(0, 1, 0, 1),
                               use_collision=False)

        deform.sim.stepSimulation()
        rgb = deform.render(mode='rgb_array',
                            width=extra.cam_resolution,
                            height=extra.cam_resolution)
        frames.append(rgb)

    env.close()

    fig, axes = plt.subplots(n, n, figsize=(3.4 * n, 3.4 * n))
    axes = np.atleast_2d(axes)
    for ax, frame, (dx, dy) in zip(axes.flat, frames, dxdy_grid):
        ax.imshow(frame)
        ax.set_title(f'dx={dx:+.2f}, dy={dy:+.2f}', fontsize=10)
        ax.axis('off')
    cam = extra.cam_viewmat
    fig.suptitle(
        f'Hanger randomization preview — radius={r:.2f}m   '
        f'cam=(dist={cam[0]}, pitch={cam[1]}, yaw={cam[2]}, '
        f'target=({cam[3]}, {cam[4]}, {cam[5]}))',
        fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    out = Path(extra.save_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=110, bbox_inches='tight')
    print(f'[diag] wrote {out}')


if __name__ == '__main__':
    main()
