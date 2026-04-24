# HangProcCloth-v1 Observability Experiment

This experiment tests whether RL performance on the cloth-hanging task depends on
the camera view preserving the spatial relationship between the cloth hole and the
hanger hook.

**Claim being tested:**  
> Successful hanging requires an observation that preserves the spatial structure of
> the hole relative to the hanger. Degrading this information degrades performance.

---

## Experiment Conditions

Six conditions are compared, spanning from no visual information to full privileged state.

### Camera conditions (pixel observations, 64×64 RGB)

| Condition | Script | Camera | What the agent sees |
|-----------|--------|--------|---------------------|
| **full** | `train_full.sh` | `dist=9, pitch=-25, yaw=45` | Front-angled; hole and hanger hook both in frame |
| **partial** | `train_partial.sh` | `dist=8, pitch=-82, yaw=45` | Near-top-down; hole visible but depth to hook is lost |
| **lowdim** | `train_lowdim.sh` | none (`cam_resolution=0`) | Gripper positions only (12-dim), no cloth or hanger info |

### Privileged conditions (ground-truth geometry, no camera)

| Condition | `--obs_mode` | Obs dim | Description |
|-----------|-------------|---------|-------------|
| **hole_centroid** | `hole_centroid` | 18 | Gripper + hole centroid XYZ + hanger goal XYZ |
| **hole_vertices** | `hole_vertices` | 132 | Gripper + all hole-boundary vertices (padded to 40) |
| **full_mesh** | `full_mesh` | ~762 | Gripper + all cloth vertex positions (~250 verts) |

Privileged conditions serve as upper-bound oracles: the policy receives exact geometric
information about the hole and hanger, bypassing any observability limitation.

---

## Setup

```bash
conda activate dedo
cd /path/to/dedo  # repo root
pip install -e .
```

All scripts are run from the **repo root**. Log outputs default to
`logs/hang_obs_exp/<condition>/` relative to the repo root.

Override the log directory with the `LOGDIR` environment variable:
```bash
LOGDIR=/my/custom/path bash experiments/hang_obs_exp/scripts/train_full.sh
```

Override the Python interpreter with the `PYTHON` environment variable:
```bash
PYTHON=/path/to/env/bin/python bash experiments/hang_obs_exp/scripts/train_full.sh
```

---

## Training

### Camera conditions

```bash
# Condition A — full observability (front-angled camera)
bash experiments/hang_obs_exp/scripts/train_full.sh

# Condition B — partial observability (near-top-down camera)
bash experiments/hang_obs_exp/scripts/train_partial.sh

# Condition C — low-dim baseline (no camera)
bash experiments/hang_obs_exp/scripts/train_lowdim.sh
```

All three use PPO with 4 parallel envs and 2M total environment steps.
Checkpoints and TensorBoard logs are saved under `logs/hang_obs_exp/<condition>/`.

### Privileged conditions

```bash
python experiments/hang_obs_exp/scripts/train_privileged.py --obs_mode hole_centroid
python experiments/hang_obs_exp/scripts/train_privileged.py --obs_mode hole_vertices
python experiments/hang_obs_exp/scripts/train_privileged.py --obs_mode full_mesh
```

Additional flags:
```
--total_env_steps INT   (default: 2000000)
--num_envs INT          (default: 4)
--lr FLOAT              (default: 3e-4)
--seed INT              (default: 42)
--logdir_root PATH      (default: logs/hang_obs_exp)
```

### Smoke test

Verify the pipeline runs end-to-end before committing to a full run:
```bash
python experiments/hang_obs_exp/scripts/smoke_test.py
```
Runs 2000 steps per camera condition (~1 min each) and prints PASS/FAIL per condition.

---

## Evaluation

### Camera conditions

```bash
bash experiments/hang_obs_exp/scripts/eval.sh <checkpoint_dir> [<output_dir>]
```

`<checkpoint_dir>` is the timestamped subdirectory created during training, e.g.:
```bash
bash experiments/hang_obs_exp/scripts/eval.sh \
    logs/hang_obs_exp/full/PPO_260412_231040_HangProcCloth-v1
```

This runs the saved policy for one episode using a high-resolution camera (200×200)
and saves a video to `<output_dir>` (default: `/tmp/dedo_eval`).

The script also accepts a relative path resolved from the repo root:
```bash
bash experiments/hang_obs_exp/scripts/eval.sh \
    logs/hang_obs_exp/partial/PPO_260412_231042_HangProcCloth-v1 \
    logs/hang_obs_exp/partial/eval_videos
```

> **Note:** `eval.sh` is for camera conditions only. It calls `dedo.run_rl_sb3 --play`,
> which reconstructs the env from the saved `args.pkl`. The `PrivilegedObsWrapper` is
> not re-applied automatically, so privileged checkpoints need a custom eval script.

### TensorBoard

```bash
tensorboard --logdir logs/hang_obs_exp
```

Each condition appears as a separate run. The key metric is
`eval/mean_reward` (logged by `CustomCallback` every `log_save_interval` steps).

---

## Visualizations

Generate side-by-side screenshots of what each camera sees:
```bash
bash experiments/hang_obs_exp/scripts/visualize.sh
```
Output images are saved to `experiments/hang_obs_exp/viz_output/`:
- `observability_comparison.png` — grid of all conditions at steps 0, 10, 30
- `A_full_step*.png`, `B_partial_step*.png`, `preset_default_step*.png` — individual frames

---

## File Structure

```
experiments/hang_obs_exp/
  envs/
    privileged_env.py       # PrivilegedObsWrapper (hole_centroid / hole_vertices / full_mesh)
  configs/
    cam_full.json           # Camera matrix reference for full-observability condition
    cam_partial.json        # Camera matrix reference for partial-observability condition
  scripts/
    train_full.sh           # Train Condition A (camera, full obs)
    train_partial.sh        # Train Condition B (camera, partial obs)
    train_lowdim.sh         # Train Condition C (no camera, gripper only)
    train_privileged.py     # Train privileged conditions (hole_centroid / hole_vertices / full_mesh)
    eval.sh                 # Evaluate a camera-condition checkpoint
    smoke_test.py           # Quick end-to-end sanity check (~3 min)
    gen_screenshots.py      # Render comparison images
    visualize.sh            # Wrapper to run gen_screenshots.py
  viz_output/               # Generated by visualize.sh (not committed)
  README.md
```

Training outputs go to `logs/hang_obs_exp/` at the repo root (not committed).

---

## Training Configuration

| Parameter | Value |
|-----------|-------|
| Algorithm | PPO (Stable Baselines3) |
| Policy | MlpPolicy (flat pixel vector for camera conds) |
| Image resolution | 64×64×3 = 12,288-dim (flattened) |
| Parallel envs | 4 (SubprocVecEnv for camera; DummyVecEnv for privileged) |
| Total env steps | 2,000,000 |
| Learning rate | 3e-4 |
| Seed | 42 |
| Episode length | 200 steps |
