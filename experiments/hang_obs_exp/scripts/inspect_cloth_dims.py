"""
Quick cloth-dimensions probe for HangProcCloth.

Prints world-space dimensions (meters) for procedurally generated cloth meshes:
  - mesh AABB min/max and axis spans (x/y/z)
  - AABB diagonal length
  - optional per-hole loop extents when true loop vertices are available

Usage (from repo root):
  python experiments/hang_obs_exp/scripts/inspect_cloth_dims.py
  python experiments/hang_obs_exp/scripts/inspect_cloth_dims.py --num_samples 5
  python experiments/hang_obs_exp/scripts/inspect_cloth_dims.py --seed 7
"""
import argparse
import sys
from pathlib import Path

import gym
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT))

import dedo  # noqa: F401
from dedo.utils.args import get_args_parser, args_postprocess
from dedo.utils.mesh_utils import get_mesh_data

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _helpers import RetryResetEnv  # noqa: E402


def _aabb_stats(pts):
    pmin = pts.min(axis=0)
    pmax = pts.max(axis=0)
    span = pmax - pmin
    diag = float(np.linalg.norm(span))
    return pmin, pmax, span, diag


def _fmt_xyz(v):
    return f"x={v[0]:.3f} y={v[1]:.3f} z={v[2]:.3f}"


def _build_dedo_args(seed, env_name):
    sys.argv = [
        "inspect_cloth_dims",
        f"--env={env_name}",
        "--cam_resolution=0",
        "--num_envs=0",
        "--total_env_steps=0",
        "--viz=False",
        "--debug=False",
        "--seed",
        str(seed),
    ]
    args, _ = get_args_parser()
    args_postprocess(args)
    args.viz = False
    args.debug = False
    return args


def _print_hole_stats(env, verts):
    if not hasattr(env.args, "deform_true_loop_vertices"):
        print("  hole_loops: unavailable")
        return
    loops = env.args.deform_true_loop_vertices
    if not loops:
        print("  hole_loops: []")
        return
    print(f"  hole_loops: {len(loops)}")
    for i, loop in enumerate(loops):
        if not loop:
            print(f"    loop[{i}] empty")
            continue
        lp = verts[np.asarray(loop, dtype=np.int64)]
        lp = lp[~np.isnan(lp).any(axis=1)]
        if len(lp) == 0:
            print(f"    loop[{i}] all NaN")
            continue
        _, _, span, diag = _aabb_stats(lp)
        centroid = lp.mean(axis=0)
        radius = float(np.mean(np.linalg.norm(lp - centroid, axis=1)))
        print(
            f"    loop[{i}] span({_fmt_xyz(span)})  "
            f"diag={diag:.3f}m  mean_radius={radius:.3f}m"
        )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--env", type=str, default="HangProcCloth-v1")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num_samples", type=int, default=1)
    extra = parser.parse_args()

    args = _build_dedo_args(extra.seed, extra.env)
    env = gym.make(args.env, args=args)
    env = RetryResetEnv(env)
    env.seed(extra.seed)

    print(f"[inspect] env={extra.env}  seed={extra.seed}  samples={extra.num_samples}")
    print("[inspect] dimensions are in world-space meters")

    try:
        for sample_idx in range(extra.num_samples):
            env.reset()
            _, verts = get_mesh_data(env.sim, env.deform_id)
            verts = np.asarray(verts, dtype=np.float32)
            verts = verts[~np.isnan(verts).any(axis=1)]
            if len(verts) == 0:
                print(f"\n--- sample {sample_idx} ---")
                print("mesh has no finite vertices")
                continue

            pmin, pmax, span, diag = _aabb_stats(verts)
            print(f"\n--- sample {sample_idx} ---")
            print(f"  deform_obj: {env.deform_obj}")
            print(f"  vertex_count: {len(verts)}")
            print(f"  aabb_min: {_fmt_xyz(pmin)}")
            print(f"  aabb_max: {_fmt_xyz(pmax)}")
            print(f"  span: {_fmt_xyz(span)}")
            print(f"  aabb_diag: {diag:.3f}m")
            if hasattr(env, "goal_pos"):
                gp = np.asarray(env.goal_pos, dtype=np.float32)
                print(f"  goal_pos[0]: {_fmt_xyz(gp[0])}")

            _print_hole_stats(env, verts)
    finally:
        env.close()


if __name__ == "__main__":
    main()
