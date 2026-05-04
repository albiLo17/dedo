"""
Quick cloth-dimensions probe for HangProcCloth.

Prints world-space dimensions (meters) for procedurally generated cloth meshes:
  - mesh AABB min/max and axis spans (x/y/z)
  - AABB diagonal length
  - optional per-hole loop extents when true loop vertices are available
  - hole-centroid stability under no-op stepping (drift/jitter)

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
        "--seed",
        str(seed),
    ]
    args, _ = get_args_parser()
    args_postprocess(args)
    args.viz = False
    args.debug = False
    return args


def _get_loops(env):
    if not hasattr(env.args, "deform_true_loop_vertices"):
        return []
    return env.args.deform_true_loop_vertices


def _loop_points(verts, loop):
    if not loop:
        return np.zeros((0, 3), dtype=np.float32)
    lp = verts[np.asarray(loop, dtype=np.int64)]
    lp = lp[~np.isnan(lp).any(axis=1)]
    return lp


def _print_hole_stats(env, verts):
    loops = _get_loops(env)
    if not loops:
        print("  hole_loops: unavailable or empty")
        return
    print(f"  hole_loops: {len(loops)}")
    for i, loop in enumerate(loops):
        if not loop:
            print(f"    loop[{i}] empty")
            continue
        lp = _loop_points(verts, loop)
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


def _hole_centroids_from_verts(env, verts):
    loops = _get_loops(env)
    loop_centroids = []
    valid_counts = []
    for loop in loops:
        lp = _loop_points(verts, loop)
        valid_counts.append(int(len(lp)))
        if len(lp) == 0:
            loop_centroids.append(None)
        else:
            loop_centroids.append(lp.mean(axis=0))
    valid = [c for c in loop_centroids if c is not None]
    merged = np.mean(valid, axis=0) if valid else None
    return loop_centroids, merged, valid_counts


def _norm(a, b):
    return float(np.linalg.norm(np.asarray(a) - np.asarray(b)))


def _print_centroid_stability(env, num_probe_steps):
    loops = _get_loops(env)
    if not loops:
        print("  centroid_stability: skipped (no hole loops)")
        return
    if num_probe_steps <= 0:
        print("  centroid_stability: skipped (--probe_steps <= 0)")
        return

    goal = np.asarray(env.goal_pos[0], dtype=np.float32)
    zero_action = np.zeros(env.action_space.shape, dtype=np.float32)

    merged_path = []
    merged_d = []
    per_loop_path = [[] for _ in loops]
    per_loop_d = [[] for _ in loops]
    nan_steps_per_loop = np.zeros(len(loops), dtype=np.int64)
    done_seen = False

    # Include t=0 snapshot.
    _, verts0 = get_mesh_data(env.sim, env.deform_id)
    verts0 = np.asarray(verts0, dtype=np.float32)
    loop_c, merged_c, _ = _hole_centroids_from_verts(env, verts0)
    if merged_c is not None:
        merged_path.append(merged_c)
        merged_d.append(_norm(merged_c, goal))
    for i, c in enumerate(loop_c):
        if c is None:
            nan_steps_per_loop[i] += 1
        else:
            per_loop_path[i].append(c)
            per_loop_d[i].append(_norm(c, goal))

    # No-op rollout probe.
    for _ in range(num_probe_steps):
        _, _, done, _ = env.step(zero_action)
        if done:
            done_seen = True
            break
        _, verts = get_mesh_data(env.sim, env.deform_id)
        verts = np.asarray(verts, dtype=np.float32)
        loop_c, merged_c, _ = _hole_centroids_from_verts(env, verts)
        if merged_c is not None:
            merged_path.append(merged_c)
            merged_d.append(_norm(merged_c, goal))
        for i, c in enumerate(loop_c):
            if c is None:
                nan_steps_per_loop[i] += 1
            else:
                per_loop_path[i].append(c)
                per_loop_d[i].append(_norm(c, goal))

    print(
        f"  centroid_stability: steps={len(merged_path)} "
        f"(requested {num_probe_steps + 1}, done_early={done_seen})"
    )
    if merged_path:
        merged_path = np.asarray(merged_path, dtype=np.float32)
        merged_d = np.asarray(merged_d, dtype=np.float32)
        delta = np.diff(merged_path, axis=0)
        jitter = float(np.mean(np.linalg.norm(delta, axis=1))) if len(delta) > 0 else 0.0
        drift = _norm(merged_path[-1], merged_path[0])
        print(
            f"    merged_centroid: d0={merged_d[0]:.3f}m  "
            f"d_last={merged_d[-1]:.3f}m  d_range=[{merged_d.min():.3f}, {merged_d.max():.3f}]m"
        )
        print(
            f"    merged_motion: drift={drift:.3f}m  "
            f"mean_step_jitter={jitter:.4f}m"
        )
    else:
        print("    merged_centroid: no valid samples")

    for i in range(len(loops)):
        nvalid = len(per_loop_path[i])
        if nvalid == 0:
            print(f"    loop[{i}] centroid: no valid samples")
            continue
        cpath = np.asarray(per_loop_path[i], dtype=np.float32)
        dpath = np.asarray(per_loop_d[i], dtype=np.float32)
        cdelta = np.diff(cpath, axis=0)
        cjitter = float(np.mean(np.linalg.norm(cdelta, axis=1))) if len(cdelta) > 0 else 0.0
        cdrift = _norm(cpath[-1], cpath[0])
        print(
            f"    loop[{i}] centroid: valid={nvalid}  nan_steps={int(nan_steps_per_loop[i])}  "
            f"d_range=[{dpath.min():.3f}, {dpath.max():.3f}]m  "
            f"drift={cdrift:.3f}m  jitter={cjitter:.4f}m"
        )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--env", type=str, default="HangProcCloth-v1")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num_samples", type=int, default=1)
    parser.add_argument(
        "--probe_steps",
        type=int,
        default=60,
        help="No-op steps to probe hole-centroid stability after reset.",
    )
    extra = parser.parse_args()

    args = _build_dedo_args(extra.seed, extra.env)
    env = gym.make(args.env, args=args)
    env = RetryResetEnv(env)
    env.seed(extra.seed)

    print(f"[inspect] env={extra.env}  seed={extra.seed}  samples={extra.num_samples}")
    print("[inspect] dimensions are in world-space meters")
    print(f"[inspect] centroid probe: {extra.probe_steps} no-op steps/sample")

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
            _print_centroid_stability(env, extra.probe_steps)
    finally:
        env.close()


if __name__ == "__main__":
    main()
