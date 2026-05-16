"""
Collect scripted-controller demos and save in the HDF5 format expected by
UniClothDiff's ClothStateEstVariableDataset.

Each kept rollout becomes one cloth entry with one trajectory. The data
needed for GPS state estimation is recorded per timestep:

  positions   — (V, 3) float32 mesh vertex positions in world coords
  pointclouds — (N, 3) float32 point cloud from back-projected depth buffer

Mesh topology (faces) and rest positions are captured once at episode reset
from the procedurally-generated .obj file and initial sim state.

Output HDF5 schema (one file, two top-level splits):
  training/
    cloth_000/
      rest_positions: (V, 3)  float32
      faces:          (F, 3)  int64    (0-indexed triangle faces)
      trajectory_0/
        step_0000/
          positions:   (V, 3)  float32
          pointclouds/
            cam_0:     (N, 3)  float32
        step_0001/ ...
    cloth_001/ ...
  validation/
    cloth_NNN/ ...

Usage (from repo root):
  python experiments/hang_obs_exp/scripts/collect_bc_demos_GSE.py \
      --output_h5 logs/hang_obs_exp/gse_demos_v1.h5 \
      --n_demos 150 \
      --cam_resolution 128 --pcd_n_points 512 \
      --max_act_vel 4.0 \
      --success_metric legacy --success_factor 1.2 \
      --ctrl_freq 15 --max_episode_len 120 \
      --debug_viz_first_n 3 --debug_viz_every 25 \
      --seed 2026
"""
import argparse
import os
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT))

import gym
import h5py
import numpy as np

import dedo  # noqa: F401  (registers gym envs)
from dedo.envs.deform_env import DeformEnv
from dedo.utils.args import get_args_parser, args_postprocess
from dedo.utils.mesh_utils import get_mesh_data
from dedo.demo_preset import build_traj, merge_traj

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _helpers import RetryResetEnv, build_hole_aware_waypoints  # noqa: E402
from _bc_obs_helpers import (  # noqa: E402
    capture_rgb_depth, depth_to_pcd,
    get_hole_indices, get_hole_loops, measure_hole_radius,
    check_hanging_on_peg, check_threaded_topological, check_legacy,
    resolve_deform)
from _debug_viz import (  # noqa: E402
    hole_centroid_world, overlay_pcd_on_rgb, pcd_topdown_image,
    render_sim_with_centroid, build_video_frame, write_video_mp4,
    save_grid_png, save_actions_plot, summarize_action_stream)

from experiments.hang_obs_exp.envs.privileged_env import (  # noqa: E402
    identify_cloth_corners)


# ---------------------------------------------------------------------------
# Argparse
# ---------------------------------------------------------------------------
parser = argparse.ArgumentParser()
parser.add_argument('--output_h5', type=str, required=True,
                    help='Path to the output HDF5 file. Created if missing; '
                         'existing cloths are preserved and new ones appended.')
parser.add_argument('--n_demos', type=int, default=100,
                    help='Target number of demos to keep.')
parser.add_argument('--val_ratio', type=float, default=0.1,
                    help='Fraction of kept demos to place in the validation '
                         'split. Applied at the end of collection by moving '
                         'the last round(n_kept * val_ratio) cloths from '
                         'training/ to validation/.')
parser.add_argument('--only_success', action='store_true', default=True)
parser.add_argument('--no_only_success', dest='only_success',
                    action='store_false')
parser.add_argument('--success_metric', type=str, default='hanging',
                    choices=['hanging', 'topological', 'legacy'])
parser.add_argument('--success_factor', type=float, default=1.2)
parser.add_argument('--seed', type=int, default=2026)
parser.add_argument('--max_episode_len', type=int, default=200)
parser.add_argument('--cam_resolution', type=int, default=96)
parser.add_argument('--pcd_n_points', type=int, default=512)
parser.add_argument('--max_act_vel', type=float, default=10.0)
parser.add_argument('--cam_viewmat', type=float, nargs=6,
                    default=[14.0, -5.0, 45.0, 0.0, 0.0, 5.5])
parser.add_argument('--ctrl_freq', type=float, default=15.0)
parser.add_argument('--sim_freq', type=int, default=500)
parser.add_argument('--debug_viz_first_n', type=int, default=3)
parser.add_argument('--debug_viz_every', type=int, default=0)
parser.add_argument('--debug_render_size', type=int, default=300)
parser.add_argument('--debug_fps', type=int, default=15)
parser.add_argument('--debug_n_grid_samples', type=int, default=5)
extra = parser.parse_args()

_steps_per_action = max(1, int(round(extra.sim_freq / extra.ctrl_freq)))
_actual_ctrl_freq = extra.sim_freq / _steps_per_action
if abs(_actual_ctrl_freq - extra.ctrl_freq) / extra.ctrl_freq > 0.05:
    print(f'[init] WARN: requested ctrl_freq={extra.ctrl_freq} Hz rounds '
          f'to sim_steps_per_action={_steps_per_action} -> actual '
          f'{_actual_ctrl_freq:.3f} Hz (>5% deviation).')
else:
    print(f'[init] ctrl_freq={extra.ctrl_freq} Hz -> '
          f'sim_steps_per_action={_steps_per_action} '
          f'(actual {_actual_ctrl_freq:.3f} Hz)')

output_dir = os.path.dirname(os.path.abspath(extra.output_h5))
os.makedirs(output_dir, exist_ok=True)

# ---------------------------------------------------------------------------
# Build dedo args + env
# ---------------------------------------------------------------------------
sys.argv = [
    'collect_bc_demos_GSE',
    '--env=HangProcCloth-v1',
    f'--cam_resolution={extra.cam_resolution}',
    '--num_envs=0',
    '--total_env_steps=0',
    '--seed', str(extra.seed),
    '--max_episode_len', str(extra.max_episode_len),
    f'--sim_freq={extra.sim_freq}',
    f'--sim_steps_per_action={_steps_per_action}',
    '--cam_viewmat',
    *[str(x) for x in extra.cam_viewmat],
]
args, _ = get_args_parser()
args_postprocess(args)
args.debug = False
args.viz = False
args.uint8_pixels = True

_orig_mav = DeformEnv.MAX_ACT_VEL
DeformEnv.MAX_ACT_VEL = float(extra.max_act_vel)
if extra.max_act_vel != _orig_mav:
    print(f'[init] DeformEnv.MAX_ACT_VEL: {_orig_mav} -> {DeformEnv.MAX_ACT_VEL}')

env = gym.make(args.env, args=args)
env = RetryResetEnv(env)
env.seed(extra.seed)
deform = resolve_deform(env)

np.random.seed(extra.seed)

ctrl_freq = args.sim_freq / args.sim_steps_per_action


# ---------------------------------------------------------------------------
# Cloth numbering: continue past existing cloths in the HDF5.
# ---------------------------------------------------------------------------
def _next_cloth_id(h5_path: str) -> int:
    if not os.path.exists(h5_path):
        return 0
    ids = []
    with h5py.File(h5_path, 'r') as f:
        for split in ('training', 'validation'):
            if split not in f:
                continue
            for key in f[split]:
                try:
                    ids.append(int(key.split('_')[1]))
                except (ValueError, IndexError):
                    pass
    return max(ids) + 1 if ids else 0


def _read_obj_faces(obj_path: str) -> np.ndarray:
    """Return 0-indexed (F, 3) int64 face array from a .obj file."""
    faces = []
    with open(obj_path) as f:
        for line in f:
            if line.startswith('f '):
                parts = line.strip().split()[1:]
                faces.append([int(p.split('/')[0]) - 1 for p in parts])
    return np.array(faces, dtype=np.int64)


# ---------------------------------------------------------------------------
# Debug visualization
# ---------------------------------------------------------------------------
_debug_enabled = (extra.debug_viz_first_n > 0 or extra.debug_viz_every > 0)
_debug_dir = os.path.join(output_dir, 'debug_viz') if _debug_enabled else None
if _debug_dir is not None:
    os.makedirs(_debug_dir, exist_ok=True)


def _should_debug(kept_index: int) -> bool:
    if kept_index < extra.debug_viz_first_n:
        return True
    if extra.debug_viz_every > 0 and (kept_index + 1) % extra.debug_viz_every == 0:
        return True
    return False


# ---------------------------------------------------------------------------
# Main collection loop
# ---------------------------------------------------------------------------
print(f'\n=== GSE demo collection ===')
print(f'  output_h5:       {extra.output_h5}')
print(f'  target n_demos:  {extra.n_demos}')
print(f'  val_ratio:       {extra.val_ratio}')
print(f'  only_success:    {extra.only_success}  ({extra.success_metric})')
print(f'  cam_resolution:  {extra.cam_resolution}')
print(f'  pcd_n_points:    {extra.pcd_n_points}')
print(f'  MAX_ACT_VEL:     {DeformEnv.MAX_ACT_VEL}')
print(f'  ctrl_freq:       {_actual_ctrl_freq:.3f} Hz '
      f'(sim_freq={extra.sim_freq}, steps/action={_steps_per_action})')
if _debug_enabled:
    print(f'  debug_viz:       first {extra.debug_viz_first_n}'
          f'{f" + every {extra.debug_viz_every}th" if extra.debug_viz_every > 0 else ""}'
          f' -> {_debug_dir}')
else:
    print(f'  debug_viz:       disabled')
print(f'  starting cloth_id at: {_next_cloth_id(extra.output_h5)}\n')

_action_stats_kept: list = []

n_kept = 0
n_dropped_failed = 0
attempts = 0
max_attempts = max(extra.n_demos * 5, 30)
start_time = time.time()

# Track cloth keys written this run so we can split train/val at the end.
_kept_cloth_keys: list = []

while n_kept < extra.n_demos and attempts < max_attempts:
    attempts += 1
    env.reset()
    hole_idx = get_hole_indices(deform)
    if not hole_idx:
        print(f'[demo] attempt {attempts}: no hole loop on cloth, retrying')
        continue
    hole_loops = get_hole_loops(deform)
    num_verts, verts0 = get_mesh_data(deform.sim, deform.deform_id)
    rest_positions = np.array(verts0, dtype=np.float32)  # (V, 3)
    faces = _read_obj_faces(deform.deform_obj)            # (F, 3)
    corner_idx = identify_cloth_corners(rest_positions)
    hole_radius = measure_hole_radius(deform, hole_idx)

    wp = build_hole_aware_waypoints(deform)
    if wp is None:
        print(f'[demo] attempt {attempts}: build waypoints failed, retrying')
        continue
    try:
        _, va = build_traj(deform, wp, 'a', anchor_idx=0,
                           ctrl_freq=ctrl_freq, robot=None)
        _, vb = build_traj(deform, wp, 'b', anchor_idx=1,
                           ctrl_freq=ctrl_freq, robot=None)
        traj = merge_traj(va, vb)
    except Exception as e:
        print(f'[demo] attempt {attempts}: build_traj failed ({e!r}), retrying')
        continue

    if attempts == 1:
        peak = float(np.abs(traj).max())
        flag = ' <- TOO LOW, demos will saturate' \
            if peak > DeformEnv.MAX_ACT_VEL else ''
        print(f'[demo] traj peak |vel| = {peak:.3f} m/s; '
              f'MAX_ACT_VEL = {DeformEnv.MAX_ACT_VEL:.3f} m/s{flag}')

    # Per-episode buffers (GSE-relevant only).
    ep_pos = []   # list of (V, 3) vertex positions per step
    ep_pcd = []   # list of (N, 3) point clouds per step
    ep_rgb = []   # list of (H, W, 3) uint8 frames — not used for training,
                  # kept so mp4s can be rendered offline from the HDF5
    ep_act = []   # kept for action-stat summary and traj_len tracking
    ep_rwd = []

    _is_debug_attempt = _debug_enabled and _should_debug(n_kept)
    ep_debug_depth = [] if _is_debug_attempt else None
    ep_debug_centroid = [] if _is_debug_attempt else None
    ep_debug_video_frames = [] if _is_debug_attempt else None
    last_action = np.zeros_like(traj[0])
    step = 0
    done = False
    traj_len_for_demo = int(len(traj))

    while not done:
        # Capture mesh positions and point cloud BEFORE step.
        _, verts_t = get_mesh_data(deform.sim, deform.deform_id)
        ep_pos.append(np.array(verts_t, dtype=np.float32))  # (V, 3)

        rgb, depth, view, proj, seg = capture_rgb_depth(
            deform, extra.cam_resolution, extra.cam_resolution,
            return_seg=True)
        pcd_world = depth_to_pcd(depth, view, proj, extra.pcd_n_points,
                                 seg_mask=seg, cloth_id=deform.deform_id)
        ep_rgb.append(rgb)
        ep_pcd.append(pcd_world)

        if _is_debug_attempt:
            ep_debug_depth.append(depth.copy())
            centroid_w = hole_centroid_world(deform, hole_idx)
            ep_debug_centroid.append(centroid_w)
            sim_panel = render_sim_with_centroid(
                deform, view, proj, centroid_w, size=extra.debug_render_size)
            obs_overlay = overlay_pcd_on_rgb(rgb, pcd_world, view, proj)
            topdown = pcd_topdown_image(pcd_world, size=extra.debug_render_size)
            ep_debug_video_frames.append(build_video_frame(
                sim_panel, obs_overlay, topdown, size=extra.debug_render_size))

        act_unscaled = traj[step] if step < len(traj) else last_action
        act = np.clip(act_unscaled / DeformEnv.MAX_ACT_VEL,
                      -1.0, 1.0).astype(np.float32)
        ep_act.append(act)
        _, rwd, done, _ = env.step(act)
        ep_rwd.append(float(rwd))
        last_action = act_unscaled
        step += 1

    if _is_debug_attempt:
        rgb_ps, depth_ps, view_ps, proj_ps, seg_ps = capture_rgb_depth(
            deform, extra.cam_resolution, extra.cam_resolution,
            return_seg=True)
        pcd_ps = depth_to_pcd(depth_ps, view_ps, proj_ps, extra.pcd_n_points,
                              seg_mask=seg_ps, cloth_id=deform.deform_id)
        centroid_ps = hole_centroid_world(deform, hole_idx)
        _post_settle_panel = {
            'rgb': rgb_ps, 'depth': depth_ps, 'pcd': pcd_ps,
            'view': view_ps, 'proj': proj_ps, 'centroid': centroid_ps,
        }
        sim_panel_ps = render_sim_with_centroid(
            deform, view_ps, proj_ps, centroid_ps, size=extra.debug_render_size)
        obs_overlay_ps = overlay_pcd_on_rgb(rgb_ps, pcd_ps, view_ps, proj_ps)
        topdown_ps = pcd_topdown_image(pcd_ps, size=extra.debug_render_size)
        ep_debug_video_frames.append(build_video_frame(
            sim_panel_ps, obs_overlay_ps, topdown_ps, size=extra.debug_render_size))
    else:
        _post_settle_panel = None

    success_hanging = check_hanging_on_peg(
        deform, hole_idx, hole_radius, extra.success_factor)
    success_topological, max_winding = check_threaded_topological(
        deform, hole_loops)
    success_legacy = check_legacy(
        deform, hole_idx, hole_radius, extra.success_factor)

    success_by_metric = {'hanging': int(success_hanging),
                         'topological': int(success_topological),
                         'legacy': int(success_legacy)}
    keep_success = success_by_metric[extra.success_metric]
    ep_reward_total = float(np.sum(ep_rwd))

    if extra.only_success and not keep_success:
        n_dropped_failed += 1
        print(f'[demo] attempt {attempts} (kept {n_kept}/{extra.n_demos})  '
              f'len={len(ep_act)}  rwd={ep_reward_total:.1f}  '
              f'h={success_hanging} t={success_topological} '
              f'l={success_legacy}  (dropped)')
        continue

    # Write cloth to HDF5 (all go to training/ for now; split at the end).
    cloth_id = _next_cloth_id(extra.output_h5)
    cloth_key = f'cloth_{cloth_id:03d}'
    with h5py.File(extra.output_h5, 'a') as f:
        cloth_grp = f.require_group(f'training/{cloth_key}')
        cloth_grp.create_dataset('rest_positions', data=rest_positions)
        cloth_grp.create_dataset('faces', data=faces)
        traj_grp = cloth_grp.create_group('trajectory_0')
        for t, (pos, pcd, rgb_t) in enumerate(zip(ep_pos, ep_pcd, ep_rgb)):
            step_grp = traj_grp.create_group(f'step_{t:04d}')
            step_grp.create_dataset('positions', data=pos)
            step_grp.create_dataset('rgb', data=rgb_t)
            pcd_grp = step_grp.create_group('pointclouds')
            pcd_grp.create_dataset('cam_0', data=pcd)

    _kept_cloth_keys.append(cloth_key)

    act_stats = summarize_action_stream(
        np.asarray(ep_act, dtype=np.float32), traj_len_for_demo)
    _action_stats_kept.append(act_stats)

    n_kept += 1
    print(f'[demo] attempt {attempts} (kept {n_kept}/{extra.n_demos})  '
          f'len={act_stats["ep_len"]}  '
          f'active={act_stats["n_active"]} hold={act_stats["n_hold"]} '
          f'({act_stats["hold_frac"]*100:.0f}% hold)  '
          f'rwd={ep_reward_total:.1f}  '
          f'h={success_hanging} t={success_topological} '
          f'l={success_legacy}  -> training/{cloth_key}')

    if _is_debug_attempt and _post_settle_panel is not None:
        n_steps = len(ep_pcd)
        sample_ix = np.linspace(0, n_steps - 1, extra.debug_n_grid_samples,
                                dtype=np.int64).tolist()
        rows = []
        for ix in sample_ix:
            n_valid = int((ep_debug_depth[ix] < 0.999).sum())
            rows.append({
                'label': f'step {ix}/{n_steps - 1}',
                'rgb': ep_rgb[ix],
                'depth': ep_debug_depth[ix],
                'pcd': ep_pcd[ix],
                'view': deform._cam_viewmat,
                'proj': None,
                'n_valid': n_valid,
                'in_frame': None,
            })
        from _bc_obs_helpers import proj_matrix as _proj_matrix
        _shared_proj = _proj_matrix()
        for r in rows:
            r['proj'] = _shared_proj
        rows.append({
            'label': 'post-settle',
            'rgb': _post_settle_panel['rgb'],
            'depth': _post_settle_panel['depth'],
            'pcd': _post_settle_panel['pcd'],
            'view': _post_settle_panel['view'],
            'proj': _post_settle_panel['proj'],
            'n_valid': int((_post_settle_panel['depth'] < 0.999).sum()),
            'in_frame': None,
        })
        grid_path = os.path.join(_debug_dir, f'{cloth_key}_grid.png')
        cam_str = (f'dist={extra.cam_viewmat[0]}, '
                   f'pitch={extra.cam_viewmat[1]}, '
                   f'yaw={extra.cam_viewmat[2]}')
        succ_str = (f'h={int(success_hanging)}/t={int(success_topological)}'
                    f'/l={int(success_legacy)}')
        save_grid_png(
            rows, grid_path,
            title=(f'{cloth_key}  |  cam: {cam_str}  |  '
                   f'ctrl_freq={_actual_ctrl_freq:.2f} Hz  |  '
                   f'success {succ_str}  |  '
                   f'len={act_stats["ep_len"]} active={act_stats["n_active"]} '
                   f'hold={act_stats["n_hold"]} '
                   f'({act_stats["hold_frac"]*100:.0f}%)'))

        video_path = os.path.join(_debug_dir, f'{cloth_key}_video.mp4')
        try:
            write_video_mp4(ep_debug_video_frames, video_path,
                            fps=extra.debug_fps)
        except Exception as _e:
            print(f'  [debug_viz] WARN: video write failed: {_e!r}')

        acts_path = os.path.join(_debug_dir, f'{cloth_key}_actions.png')
        save_actions_plot(
            np.asarray(ep_act, dtype=np.float32),
            traj_len_for_demo, acts_path,
            title=(f'{cloth_key}  actions  |  '
                   f'len={act_stats["ep_len"]} '
                   f'(active={act_stats["n_active"]}, '
                   f'hold={act_stats["n_hold"]})  '
                   f'peak|a|={act_stats["peak_abs_a"]:.2f}  '
                   f'mean||a||={act_stats["mean_norm_a"]:.2f}'))
        print(f'  [debug_viz] wrote {os.path.basename(grid_path)}, '
              f'{os.path.basename(video_path)}, '
              f'{os.path.basename(acts_path)}')


env.close()
elapsed = time.time() - start_time
print(f'\nDone. kept={n_kept}, dropped={n_dropped_failed}, '
      f'attempts={attempts}, elapsed={elapsed:.0f}s '
      f'({elapsed/max(n_kept,1):.1f}s/demo)')

# ---------------------------------------------------------------------------
# Train / validation split: move the last round(n_kept * val_ratio) cloths
# written THIS run from training/ to validation/.
# ---------------------------------------------------------------------------
if _kept_cloth_keys and extra.val_ratio > 0.0:
    n_val = max(1, round(len(_kept_cloth_keys) * extra.val_ratio))
    val_keys = _kept_cloth_keys[-n_val:]
    print(f'\nSplitting: {len(_kept_cloth_keys) - n_val} train, '
          f'{n_val} val cloths.')
    with h5py.File(extra.output_h5, 'a') as f:
        f.require_group('validation')
        for key in val_keys:
            f.copy(f'training/{key}', f'validation/{key}')
            del f[f'training/{key}']
            print(f'  training/{key} -> validation/{key}')

# ---------------------------------------------------------------------------
# Action-stat summary
# ---------------------------------------------------------------------------
if _action_stats_kept:
    _lens = np.array([s['ep_len'] for s in _action_stats_kept])
    _actives = np.array([s['n_active'] for s in _action_stats_kept])
    _holds = np.array([s['n_hold'] for s in _action_stats_kept])
    _hold_fracs = np.array([s['hold_frac'] for s in _action_stats_kept])
    _peaks = np.array([s['peak_abs_a'] for s in _action_stats_kept])
    _stationary_fracs = np.array([s['stationary_frac']
                                  for s in _action_stats_kept])
    print(f'\n=== Action-stat summary (across {len(_action_stats_kept)} '
          f'kept demos) ===')
    print(f'  episode length     : mean={_lens.mean():.1f}  '
          f'min={_lens.min()}  max={_lens.max()}')
    print(f'  active steps       : mean={_actives.mean():.1f}  '
          f'min={_actives.min()}  max={_actives.max()}')
    print(f'  hold steps         : mean={_holds.mean():.1f}  '
          f'min={_holds.min()}  max={_holds.max()}')
    print(f'  hold fraction      : mean={_hold_fracs.mean()*100:.1f}%  '
          f'min={_hold_fracs.min()*100:.1f}%  '
          f'max={_hold_fracs.max()*100:.1f}%')
    print(f'  stationary frac    : mean={_stationary_fracs.mean()*100:.1f}%')
    print(f'  peak |a| / demo    : mean={_peaks.mean():.2f}  '
          f'min={_peaks.min():.2f}  max={_peaks.max():.2f}  '
          f'(1.0 = saturated)')
    if _peaks.max() >= 0.99:
        n_sat = int((_peaks >= 0.99).sum())
        print(f'  NOTE: {n_sat}/{len(_peaks)} demo(s) have peak |a| >= 0.99 — '
              f'actions saturated. Bump --max_act_vel to avoid.')

if n_kept < extra.n_demos:
    print(f'WARNING: target {extra.n_demos} not reached.')
