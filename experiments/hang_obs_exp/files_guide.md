# hang_obs_exp — files guide

A pointer-map to everything under [experiments/hang_obs_exp/](./).
For the *why* behind the runs, see [RUNS.md](RUNS.md) (PPO+BC sweep) and
[diffusion_runs.md](diffusion_runs.md) (diffusion-BC cross-modality).
This file is just the *what* — one paragraph per script + the command
that's actually useful to run.

---

## Top-level docs

| file | purpose |
| ---- | ------- |
| [README.md](README.md) | Original "observability conditions" framing (full / partial / lowdim cameras). Predates the privileged + diffusion saga; kept for context. |
| [RUNS.md](RUNS.md) | PPO+BC privileged-obs sweep journal (Runs A–K, reward-decomp diagnosis, success-metric saga). Read this before touching `train_privileged.py`. |
| [diffusion_runs.md](diffusion_runs.md) | Diffusion-BC state/RGB/PCD comparison journal (v1 → v4 randomized-hanger). Read before touching `collect_bc_demos.py` / `train_diffusion_bc.py`. |
| [frame_transforms.md](frame_transforms.md) | Sim-world / camera / Franka-world coordinate frames and how trajectory pkls are transformed between them. |
| [scripts/reward_guide.md](scripts/reward_guide.md) | Reward-shape reference: every `--success_factor` / `--success_bonus` / `--pre_settle_coef` / `--dist_reward_coef` / `--final_reward_mult` knob explained. |
| [scripts/wandb_charts_ref.md](scripts/wandb_charts_ref.md) | Wandb chart cheat-sheet — which metrics matter, what they mean, what's a red flag. |
| [configs/cam_full.json](configs/cam_full.json), [configs/cam_partial.json](configs/cam_partial.json) | Stored camera viewmat presets for the original full/partial conditions. |

---

## Observation-wrapper envs ([envs/](envs/))

Each wrapper plugs into `gym.make('HangProcCloth-v1', ...)` and replaces
the env's obs with a different modality. They all share the same
adaptive success / reward shaping (`success_factor`, `success_bonus`,
`fail_penalty`, `vel_penalty`, `pre_settle_coef`, `dist_reward_coef`,
`final_reward_mult`) so metrics are cross-comparable.

| file | obs | use case |
| ---- | --- | -------- |
| [envs/privileged_env.py](envs/privileged_env.py) | 18/30/132/762-dim privileged state (hole_centroid / hole_centroid_corners / hole_vertices / full_mesh) | `train_privileged.py`, `collect_bc_demos.py --obs_mode state`. The canonical wrapper — emits all `rwd_diag/*` keys, computes all three success metrics. |
| [envs/pixel_env.py](envs/pixel_env.py) | (H, W, 3) uint8 RGB, optionally `Dict({image, grip})` | `train_pixels.py`, `train_pixels_sac.py`. SB3 `MultiInputPolicy` (default) or `CnnPolicy` with `--no_grip`. |
| [envs/pointcloud_env.py](envs/pointcloud_env.py) | flat 12 + n_points*3 vector, world-coord PCD from depth back-projection | `train_pointcloud.py`. Overrides `render()` to draw PCD on RGB for debug videos. |
| [envs/privileged_env_evan.py](envs/privileged_env_evan.py) | Same as `privileged_env.py` + an `enriched` 40-dim mode | Used by `train_privileged_evan.py`. Diverged fork; not in active rotation. |

---

## Main training scripts

### Active (used by the current sweeps)

| file | what it trains | journal |
| ---- | -------------- | ------- |
| [scripts/train_privileged.py](scripts/train_privileged.py) | PPO+BC on privileged obs. All RUNS.md runs A–K live here. Implements `--critic_warmup_rollouts`, `--critic_warmup_demo_epochs`, `--bc_anchor_*`, `--ppo_clip_range`, `--dist_reward_coef`, `--final_reward_mult`. | [RUNS.md](RUNS.md) |
| [scripts/collect_bc_demos.py](scripts/collect_bc_demos.py) | Scripted hole-aware controller → demo pkls with all 4 state modes + RGB + PCD + grip + goal per timestep. One collection feeds all three diffusion modes. | [diffusion_runs.md](diffusion_runs.md) |
| [scripts/train_diffusion_bc.py](scripts/train_diffusion_bc.py) | Diffusion-policy BC across `state` / `rgb` / `pcd` obs modes. Reads pkls from `collect_bc_demos.py`. Builds a parity-checked eval env (cam, MAX_ACT_VEL, ctrl_freq from pkl). | [diffusion_runs.md](diffusion_runs.md) |
| [scripts/eval_diffusion_bc.py](scripts/eval_diffusion_bc.py) | Standalone eval of a saved `policy_best.pt` — rebuilds env entirely from ckpt metadata, supports `--max_episode_len` override (the eval-cap bug fix). Also writes per-episode traj pkls via `--save_traj_dir`. | [diffusion_runs.md v3 fix section](diffusion_runs.md) |

### Sibling experiments (less active)

| file | notes |
| ---- | ----- |
| [scripts/train_pixels.py](scripts/train_pixels.py) | PPO on RGB (the P1 run in RUNS.md). Same BC + reward pipeline as `train_privileged.py`. |
| [scripts/train_pixels_sac.py](scripts/train_pixels_sac.py) | SAC counterpart to `train_pixels.py`; image-replay-buffer-aware (~1.2 GB at 100k transitions). |
| [scripts/train_pointcloud.py](scripts/train_pointcloud.py) | PPO on PCD using `pointnet2_extractor.py`. BC via `record_demo_pcd.py` outputs. |
| [scripts/train_privileged_sac.py](scripts/train_privileged_sac.py) | SAC on privileged obs. Mirrors `train_privileged.py` arg-for-arg. |
| [scripts/train_privileged_evan.py](scripts/train_privileged_evan.py) | Fork with the `enriched` obs mode. |
| [scripts/diffusion_policy_state_pusht_demo.py](scripts/diffusion_policy_state_pusht_demo.py) | Pusht reference notebook (downloaded from Colab). Source of truth for `_diffusion_policy.py` architecture. Don't run it — it's reference. |

---

## Demo collection / recording / replay / view

### Scripted demos

| file | purpose |
| ---- | ------- |
| [scripts/view_demo.py](scripts/view_demo.py) | Run the hole-aware scripted controller on HangProcCloth — live GUI (`--viz`) or save mp4s. Visual sanity-check before kicking off BC. |
| [scripts/view_demo_hangbag.py](scripts/view_demo_hangbag.py), [scripts/view_demo_buttonproc.py](scripts/view_demo_buttonproc.py) | Same template, different tasks (HangBag, ButtonProc). |
| [scripts/eval_demo_hangbag.py](scripts/eval_demo_hangbag.py), [scripts/eval_demo_buttonproc.py](scripts/eval_demo_buttonproc.py) | Run scripted controller N×, record success rate, save overlaid mp4s + per-episode pkls. |

### Teleop demos

| file | purpose |
| ---- | ------- |
| [scripts/record_demo.py](scripts/record_demo.py) | Keyboard teleop (pynput) for HangProcCloth. ARROW + zx + per-anchor wsadqe / ijklou. ENTER saves, R discards. Records all 3 obs modes per step. |
| [scripts/record_demo_pcd.py](scripts/record_demo_pcd.py) | Same teleop UX, but the wrapper is `PointCloudObsWrapper` so saved obs is PCD-only. |
| [scripts/replay_demo.py](scripts/replay_demo.py) | Replay a recorded `demo_NNN.pkl` in the pybullet GUI (`--viz`) or as mp4 (`--logdir`). Trajectories diverge slightly due to non-deterministic cloth dynamics. |
| [scripts/view_demo.py](scripts/view_demo.py) (already above) | Also handles replay-style display for fresh scripted demos. |

### Interactive inspection (viser)

| file | purpose |
| ---- | ------- |
| [scripts/view_demo_inspect.py](scripts/view_demo_inspect.py) | Headless PyBullet + Viser browser UI at `http://localhost:8080`. Live cloth surface, per-anchor frames + velocity arrows, sim-to-Franka transform sliders, ZED real-cloud overlay. The sim-to-real calibration workhorse. |
| [scripts/zed_pcd_publisher.py](scripts/zed_pcd_publisher.py) | Runs on the ZED host: captures + downsamples ZED point cloud and streams it (TCP) or writes it atomically to disk. `view_demo_inspect.py` reads either source. |

---

## Diffusion-BC supporting modules

| file | purpose |
| ---- | ------- |
| [scripts/_diffusion_policy.py](scripts/_diffusion_policy.py) | Shared model code: ConditionalUnet1D + FiLM, `StateObsEncoder` / `RGBObsEncoder` (ResNet-18 GroupNorm, optional ImageNet pretrain) / `PointCloudObsEncoder` (PointNet++ SSG, radii 0.1/0.3 tuned for cloth scale), `ObsNormalizer`, DDPM scheduler config. |
| [scripts/_bc_obs_helpers.py](scripts/_bc_obs_helpers.py) | Camera capture (RGB + PCD via cloth-only seg-mask filter), the three success-metric functions (hanging / topological / legacy), `patch_deform_render_to_obs_camera`. Lives separate from `collect_bc_demos.py` so `train_diffusion_bc.py` can import without re-triggering the collector's argparse. |
| [scripts/pointnet2_extractor.py](scripts/pointnet2_extractor.py) | SB3 features-extractor wrapping PointNet++. Two backends: erikwijmans CUDA ops or vendored pure-Python. Backend auto-selected; `POINTNET2_FORCE_PURE_PYTHON=1` to force the slow path. |
| [scripts/third_party/pointnet2_utils.py](scripts/third_party/pointnet2_utils.py) | Vendored pure-Python PointNet++ ops (from yanx27/Pointnet_Pointnet2_pytorch). Used when CUDA ops aren't available. |

---

## PPO+BC supporting modules (used only by `train_privileged*.py`)

| file | purpose |
| ---- | ------- |
| [scripts/_bc_anchor.py](scripts/_bc_anchor.py) | DAPG-style callback: after each PPO rollout, run K MSE minibatches against the BC demo dataset. Run H in [RUNS.md](RUNS.md). |
| [scripts/_critic_warmup.py](scripts/_critic_warmup.py) | Freeze the actor for the first K rollouts so V catches up before the actor moves. `--critic_warmup_rollouts`. |
| [scripts/_critic_warmup_demos.py](scripts/_critic_warmup_demos.py) | Offline-pretrain V on demo MC returns BEFORE PPO starts. The "structural fix" of Run I. `--critic_warmup_demo_epochs`. |
| [scripts/_video_callback.py](scripts/_video_callback.py) | Replacement for dedo's CustomCallback. Splices in `make_final_steps` settle frames so eval videos include the gravity-drape phase, and writes the video to a descriptive filename for wandb. |
| [scripts/_reward_diagnostics.py](scripts/_reward_diagnostics.py) | `RewardDiagnosticsCallback` (logs `rwd_diag/*` rolling means), `dump_run_config` (writes per-run `config.json`), final-eval aggregation helpers. |

---

## Diagnostics

| file | purpose |
| ---- | ------- |
| [scripts/_diag_demo.py](scripts/_diag_demo.py) | Run scripted controller N× and print hole_centroid / gripper / hole→goal distance every K control steps. Used to debug waypoint failures. |
| [scripts/_diag_pcd_framing.py](scripts/_diag_pcd_framing.py) | Multi-row PNG: RGB \| depth \| PCD overlay \| valid-mask, sampled across a scripted trajectory. Used to pick `cam_viewmat` (winner: yaw=45, pitch=-5, dist=14, target=(0,0,5.5)). |
| [scripts/_diag_goal_randomization.py](scripts/_diag_goal_randomization.py) | n×n mosaic of the scene with the hanger at each (dx, dy) corner of the [-r, +r]² randomization box. Used to validate `--randomize_goal_radius` against `--cam_viewmat` before v4 collection. |
| [scripts/_debug_viz.py](scripts/_debug_viz.py) | Per-demo grid PNG (timestep × {RGB, depth, PCD overlay, valid-mask}) + 3-panel mp4 (sim render \| obs RGB+PCD overlay \| PCD top-down). Wired into `collect_bc_demos.py` via `--debug_viz_first_n / --debug_viz_every`. |
| [scripts/inspect_cloth_dims.py](scripts/inspect_cloth_dims.py) | Procedural-cloth AABB + per-hole extents + hole-centroid drift under no-op stepping. |
| [scripts/eval_reward_decomp.py](scripts/eval_reward_decomp.py) | Plot per-step reward decomposed into components (`base`, `terminal_shaping`, `vel_penalty`, `dist_reward`, `threading_bonus`, `pre_settle`, ...). Four input modes: `--checkpoint <run_dir>`, `--demo_dir`, `--demo_pkl`, `--scripted`. Built during the J→K reward-redesign work. |
| [scripts/smoke_test.py](scripts/smoke_test.py) | 2000-step PPO smoke runs across the three legacy camera conditions (full / partial / lowdim) to verify the pipeline runs end-to-end. |
| [scripts/gen_screenshots.py](scripts/gen_screenshots.py) | Render one RGB screenshot per legacy camera condition into `viz_output/`. Original observability-framing artifact. |

---

## Trajectory transformation / sim-to-real

| file | purpose |
| ---- | ------- |
| [scripts/project_traj_to_camframe.py](scripts/project_traj_to_camframe.py) | Reproject a trajectory pkl from sim-world frame to sim-camera frame (OpenCV convention: +Z forward, +X right, +Y down). Recovers the transform from `cam_viewmat`, no live sim needed. Adds `cam_extrinsics_sim` + `cam_intrinsics` metadata. |
| [scripts/transform_trajectory.py](scripts/transform_trajectory.py) | Bakes the viser viewer's sim→Franka-world transform (`world = offset + R_z(yaw) @ (sim_scale * p_sim)`) into a pkl offline. Defaults match `view_demo_inspect.py`'s sliders. |
| [scripts/verify_camframe.py](scripts/verify_camframe.py) | Viser-only (no gym/pybullet): overlay an original world-frame pkl and its camframe-projected version. Live alignment-residual readout; ≈0 ⇒ transform correct. |
| [scripts/viz_traj_pcd.py](scripts/viz_traj_pcd.py) | From a `--save_traj_dir` pkl: render a 2×3 keyframes PNG + animated 3D PCD mp4 with gripper paths. |

---

## Shell / launch scripts

| file | purpose |
| ---- | ------- |
| [scripts/launch_l4.sh](scripts/launch_l4.sh) | tmux launcher for overnight L4 runs (Runs A, E, F, plus pixel runs). Activates `dedo38` conda env. |
| [scripts/launch_m4.sh](scripts/launch_m4.sh) | Same for M4 Pro Macbook (Runs B, C, D). Currently mostly historical — runs migrated to L4 due to the M4 sim early-termination bug. |
| [scripts/eval.sh](scripts/eval.sh) | Evaluate a saved checkpoint under the legacy camera conditions. Not used for privileged/diffusion runs. |
| [scripts/train_all.sh](scripts/train_all.sh) | Run all 6 legacy obs conditions in parallel (original observability framing). |
| [scripts/train_full.sh](scripts/train_full.sh), [scripts/train_partial.sh](scripts/train_partial.sh), [scripts/train_lowdim.sh](scripts/train_lowdim.sh) | One-shot launchers for the 3 legacy camera conditions. |
| [scripts/train_privileged_all.sh](scripts/train_privileged_all.sh), [scripts/train_privileged_all.ps1](scripts/train_privileged_all.ps1) | Run hole_centroid / hole_vertices / full_mesh sequentially (bash / Windows). |
| [scripts/visualize.sh](scripts/visualize.sh) | Wraps `gen_screenshots.py`. |

---

## Shared helpers

| file | purpose |
| ---- | ------- |
| [scripts/_helpers.py](scripts/_helpers.py) | `RetryResetEnv` (cloth-mesh load retry), `build_hole_aware_waypoints`, `build_bag_handle_waypoints`, `build_button_proc_waypoints`, `compute_per_episode_max_len`. Side-effect-free so any script can import it without re-running training. |

---

## Useful commands cheat-sheet

The condensed list. For full launch blocks (multi-tmux v3 / v4 runs), see
[diffusion_runs.md](diffusion_runs.md).

### Look at a scripted demo

```bash
# Live pybullet GUI
python experiments/hang_obs_exp/scripts/view_demo.py --viz

# Save mp4s of a few demos
python experiments/hang_obs_exp/scripts/view_demo.py --num_episodes 3
```

### Pick / debug a camera

```bash
# Single trajectory, RGB | depth | PCD | mask grid
python experiments/hang_obs_exp/scripts/_diag_pcd_framing.py \
    --out /tmp/framing.png \
    --dist 14 --pitch -5 --yaw 45 --tx 0 --ty 0 --tz 5.5

# Validate goal-randomization at a given radius / camera
python experiments/hang_obs_exp/scripts/_diag_goal_randomization.py \
    --radius 1.5 --cam_viewmat 14 -5 45 0 0 5.5 \
    --save_path logs/hang_obs_exp/diag/cam_v3_r1.5.png
```

### Collect demos (diffusion-BC, v3/v4)

```bash
# v3 fixed-hanger, 1000 demos
python experiments/hang_obs_exp/scripts/collect_bc_demos.py \
    --demos_dir logs/hang_obs_exp/bc_demos_15hz_pcd2048_n1000_v1 \
    --n_demos 1000 --cam_resolution 128 --pcd_n_points 2048 \
    --max_act_vel 4.0 --ctrl_freq 15 --max_episode_len 200 \
    --episode_tail_frames 5 --success_metric legacy --success_factor 1.2 \
    --debug_viz_first_n 3 --debug_viz_every 100 --seed 2026

# v4 adds --randomize_goal_radius 1.5
```

### Train diffusion-BC

```bash
export DEMOS=~/github/dedo/logs/hang_obs_exp/bc_demos_15hz_pcd2048_n1000_v1

# state
python experiments/hang_obs_exp/scripts/train_diffusion_bc.py \
    --demo_path $DEMOS --obs_mode state --state_key hole_centroid \
    --action_horizon 4 --num_epochs 300 --batch_size 256 --lr 1e-4 \
    --success_metric legacy --success_factor 1.2 \
    --eval_every_epochs 20 --n_eval_episodes 30 --n_final_eval_episodes 100 \
    --save_every_epochs 20 --use_wandb --seed 2026

# rgb (add --pretrained_rgb; bs=64)
# pcd (no extra flag; bs=128)
```

### Eval a saved ckpt (no demo dir needed)

```bash
python experiments/hang_obs_exp/scripts/eval_diffusion_bc.py \
    --ckpt logs/hang_obs_exp/diffusion_bc/state/<run>/policy_best.pt \
    --n_episodes 30 --max_episode_len 56
```

### Train privileged PPO (D-base recipe — see RUNS.md)

```bash
python experiments/hang_obs_exp/scripts/train_privileged.py \
    --obs_mode hole_centroid \
    --max_act_vel 4.1 --log_std_init -2.7 \
    --lr 5e-5 --critic_warmup_rollouts 2 \
    --success_factor 1.2 --success_bonus 100 \
    --pre_settle_coef 20 --vel_penalty 8 \
    --bc_episodes 300 --bc_demos_only_success --bc_epochs 60 \
    --net_arch 512,512 \
    --total_env_steps 3000000 --seed 42 --cpu --use_wandb
```

Add `--dist_reward_coef 1.0 --final_reward_mult 50 --vel_penalty 0` for
the Run K dense-reward redesign;
`--critic_warmup_demo_epochs 50 --critic_warmup_demo_lr 3e-4` for the
Run I demo-V warmup; `--bc_anchor_batches 4 --bc_anchor_lr 1e-4` for
the Run H BC anchor.

### Visualize reward decomposition on a checkpoint

```bash
python experiments/hang_obs_exp/scripts/eval_reward_decomp.py \
    --checkpoint logs/hang_obs_exp/hole_centroid/<run_dir> \
    --n_episodes 5

# Or on freshly-generated scripted demos under a proposed reward shape
python experiments/hang_obs_exp/scripts/eval_reward_decomp.py --scripted \
    --override_dist_reward_coef 1.0 --override_final_reward_mult 50
```

### Sim-to-real workspace inspection

```bash
# Headless PyBullet + viser at http://localhost:8080
python experiments/hang_obs_exp/scripts/view_demo_inspect.py

# Stream the ZED cloud into the viewer (on the ZED host)
python experiments/hang_obs_exp/scripts/zed_pcd_publisher.py \
    --mode tcp --host 0.0.0.0 --port 5556 --max-points 15000

# Bake the viewer's sim->Franka transform into a trajectory offline
python experiments/hang_obs_exp/scripts/transform_trajectory.py \
    --in_pkl <traj.pkl> --out world_traj.pkl

# Reproject a trajectory into camera frame for sim-to-real handoff
python experiments/hang_obs_exp/scripts/project_traj_to_camframe.py \
    --pkl <traj.pkl>

# Verify a camframe transform overlays back on the original
python experiments/hang_obs_exp/scripts/verify_camframe.py \
    --orig <traj.pkl> --cam <traj_camframe.pkl>
```
