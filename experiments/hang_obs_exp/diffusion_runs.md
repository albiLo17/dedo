# HangProcCloth — Diffusion-BC cross-modality runs

Companion to [RUNS.md](RUNS.md) (PPO+BC sweep on privileged hole_centroid).
Where RUNS.md investigates *can we keep PPO from erasing BC*, this doc
covers a separate question:

> **Given a fixed scripted-demo dataset, how much does the obs modality
> matter for behavior cloning?** Diffusion policy on three modalities —
> privileged state, RGB, point cloud — all trained on the same demos, all
> evaluated under the same legacy success criterion that D/K used.

The intent is a clean three-way comparison: same trajectories, same
camera, same gripper proprioception, same reward / success metric — only
the visual observation pipeline differs.

## Saga at a glance (for slides)

1. **Pipeline scaffolding.** Built `collect_bc_demos.py` and
   `train_diffusion_bc.py` mirroring the diffusion-policy paper's
   pusht-state demo, adapted to HangProcCloth and dedo conventions.
   Auto-collects demos with all 4 privileged state modes + RGB + PCD +
   gripper proprioception per timestep, so one dataset trains any of
   the three obs modes.
2. **Camera framing diagnosis.** Investigated whether the legacy
   `cam_viewmat=[9, -25, 45, 0, 0.5, 6.5]` (used throughout PPO runs)
   keeps the cloth fully in frame at every trajectory phase. It
   doesn't — the cloth crops at the top of the image during the lift
   phase. Built `_diag_pcd_framing.py` to capture 4-panel grids
   (RGB | depth | PCD overlay | valid mask) across the trajectory under
   user-specified camera configs. Compared (a) the legacy diagonal, (b)
   yaw=90 perpendicular, (c) yaw=45 low-pitch. **Winner: yaw=45,
   pitch=-5, dist=14, target=(0, 0, 5.5)** — cloth never crops, PCD
   valid-pixel count stays ~constant at ~1000 across the trajectory,
   hole stays visible at most timesteps.
3. **MAX_ACT_VEL rescale.** Dedo's default normalization at 10 m/s
   compresses scripted-demo actions to ~[-0.03, 0.03] (3 % of the
   available range), making BC effectively learn near-zero outputs.
   Measured the scripted controller's peak velocity (~3 m/s) and tightened
   `MAX_ACT_VEL` to 3.5 m/s — actions now span most of [-1, 1] while
   the physical motions are unchanged. The value is recorded per demo
   pkl and threaded into the eval env automatically.
4. **Parity audits.** Two silent bugs caught and fixed pre-launch:
   - `sys.argv` was being restored before `gym.make` in the eval env
     builder, so dedo's `preset_override_util` overwrote `cam_viewmat`
     with HangProcCloth's preset yaw=314 — eval would have rendered
     from the wrong camera. Fixed by keeping sys.argv patched through
     the env construction.
   - The pure-Python PointNet++ backend used BatchNorm; at eval batch=1
     the running stats were brittle to distribution shift. Swapped to
     GroupNorm via `_replace_bn_with_gn`.
   See [diffusion_runs_changelog](#changes-that-landed-before-launch)
   below.
5. **Success metric: legacy (matches D/K).** The "hanging-on-peg"
   metric (RUNS.md's third attempt) was not used as the primary, because
   its thresholds (lat_threshold, min_extent) are calibrated against
   intuition rather than labeled video and may have its own
   false-positive / false-negative modes. Picking `--success_metric
   legacy` matches D/K precedent and keeps cross-script comparisons
   directly compatible. All three metrics (hanging, topological,
   legacy) are still computed and logged at every eval pass so analysis
   can re-anchor on a different metric later without recollecting.

---

## Demo dataset

One collection, shared by all three runs:

```bash
python experiments/hang_obs_exp/scripts/collect_bc_demos.py \
    --demos_dir logs/hang_obs_exp/bc_demos_legacy_v1 \
    --n_demos 150 \
    --cam_resolution 128 --pcd_n_points 512 \
    --max_act_vel 3.5 \
    --success_metric legacy --success_factor 1.2 \
    --seed 2026
```

- **150 successful demos** (legacy criterion). Scripted hole-aware
  controller; collector retries failures so 150 is the kept count, not
  attempts.
- **Camera** — diagonal, low-pitch, zoomed out:
  `cam_viewmat = (14.0, -5.0, 45.0, 0.0, 0.0, 5.5)`. Same view used by
  RGB and PCD modes, both at training and eval.
- **RGB** — 128×128 uint8 rendered via dedo's pybullet `getCameraImage`.
- **PCD** — 512 points back-projected from the same depth buffer into
  world coordinates (in meters; normalization happens in
  `ObsNormalizer.fit` over the training pool).
- **Gripper proprioception** (12-dim, pos+vel for both anchors,
  /WBOX-normalized) saved as the `grip` key. State mode has it baked
  inline as dims 0-11 of the 18-dim hole_centroid vector; RGB and PCD
  modes use it as a separate auxiliary input concatenated with visual
  features inside the encoder.
- **Action normalization**: `MAX_ACT_VEL = 3.5 m/s`. Stored per pkl
  under `max_act_vel`; the training script reads it and patches
  `DeformEnv.MAX_ACT_VEL` at eval time so demo-time and eval-time action
  scales agree.
- Each pkl is ~6 MB for 200 steps; full dataset ~1 GB.

---

## Run-name convention

Generated by `build_diffusion_run_suffix` at
[train_diffusion_bc.py:204-232](scripts/train_diffusion_bc.py#L204-L232).
Format:

```
diff<YYMMDD-HHMMSS>_<obs_mode>[_<state_key>]_lr<lr>_e<epochs>_bs<bs>
[_di<diffusion_iters>][_ph<pred_horizon>][_oh<obs_horizon>][_ah<action_horizon>]
[_sm-<metric>]_s<seed>
```

Defaults are omitted so baseline runs stay short. Example launched names:

```
diff260513-212605_state_lr1e-4_e200_bs256_sm-legacy_s2026
diff260513-212605_rgb_lr1e-4_e200_bs64_sm-legacy_s2026
diff260513-212605_pcd_lr1e-4_e200_bs128_sm-legacy_s2026
```

Wandb tags applied automatically: `obs_mode=<...>`,
`success_metric=legacy`, `seed=2026`, `algo=diffusion_bc`. Filter by tag
in the wandb runs table to compare modes.

---

## Hyperparameters shared across all three runs

| Knob | Value | Notes |
| ---- | ----- | ----- |
| obs_horizon | 2 | matches pusht reference; conditions on last 2 frames |
| pred_horizon | 16 | diffusion predicts 16 actions per forward pass |
| action_horizon | 8 | of the 16 predicted, execute the middle 8 open-loop, then re-predict |
| num_diffusion_iters | 100 | DDPM forward+reverse steps; `clip_sample=True` |
| beta_schedule | squaredcos_cap_v2 | per pusht recipe |
| Optimizer | AdamW lr=1e-4, weight_decay=1e-6 | |
| LR schedule | cosine with 500 warmup steps | |
| EMA | power=0.75 | shadow weights used for all evals + final ckpt |
| num_epochs | 200 | |
| eval_every_epochs | 20 | → 10 logged eval points across training |
| n_eval_episodes | 30 | SE on success rate at p=0.5 is ~0.09 |
| n_final_eval_episodes | 50 | tighter SE for the headline number |
| save_every_epochs | (off by default — see runs below) | |
| eval_seed_offset | 9999 | locked eval set during training; +1 for final eval |
| success_metric | legacy | matches D/K; all 3 metrics also logged |
| success_factor | 1.2 | matches D/K |
| max_act_vel (read from pkl) | 3.5 | |

---

## Per-run details

### state — privileged 18-dim hole_centroid

**Hypothesis.** Privileged state gives the upper bound: the policy knows
the hole position and the hanger position directly. Diffusion BC on this
modality measures how much performance is achievable from the demo
distribution before any visual ambiguity enters. Expected to match or
beat the BC-pretrain success rates from RUNS.md (0.5-0.77 on
hole_centroid), since the diffusion model has higher action-distribution
expressivity than the PPO Gaussian.

**Encoder.** Identity. The 18-dim obs vector is fed directly as
`global_cond_dim = obs_horizon * 18 = 36` into the ConditionalUnet1D.

**Wandb run name.**
```
diff260513-212605_state_lr1e-4_e200_bs256_sm-legacy_s2026
```

**Command.**
```bash
python experiments/hang_obs_exp/scripts/train_diffusion_bc.py \
    --demo_path logs/hang_obs_exp/bc_demos_legacy_v1 \
    --obs_mode state --state_key hole_centroid \
    --success_metric legacy --success_factor 1.2 \
    --num_epochs 200 --batch_size 256 --lr 1e-4 --num_workers 2 \
    --eval_every_epochs 20 --n_eval_episodes 30 \
    --n_final_eval_episodes 50 \
    --use_wandb --wandb_project hang_bc_diffusion \
    --seed 2026
```

**Status.** Launched in tmux session `diff-state` on the L4 (started
2026-05-13).

---

### rgb — ResNet-18 (GroupNorm) + grip

**Hypothesis.** Same demos, same trajectories, but the policy now has to
extract hole/peg location from rendered pixels rather than receiving the
ground-truth state. With 150 demos and a 128×128 image at the chosen
camera, this is a small-data visuomotor BC problem — expected to land
meaningfully below the state baseline but well above zero, since the
diagonal camera keeps both cloth and hanger in frame across the
trajectory and the gripper proprioception is provided alongside.

**Encoder.** ResNet-18 from torchvision (no pretraining — dedo render
distribution shares nothing with ImageNet). All `BatchNorm2d` layers
replaced with `GroupNorm(num_groups=16)` so eval-time batch=1 inference
doesn't depend on running stats. Image is `uint8 (H, W, 3)`, converted
to `float / 255` and permuted to NCHW inside the encoder. Output: 512-d
image features. Gripper proprioception passes through a small projection
(`Linear(12 → 12) + ReLU`) and concatenates to a **524-d** per-frame
feature. `global_cond_dim = 2 * 524 = 1048`.

**Wandb run name.**
```
diff260513-212605_rgb_lr1e-4_e200_bs64_sm-legacy_s2026
```

**Command.**
```bash
python experiments/hang_obs_exp/scripts/train_diffusion_bc.py \
    --demo_path logs/hang_obs_exp/bc_demos_legacy_v1 \
    --obs_mode rgb \
    --success_metric legacy --success_factor 1.2 \
    --num_epochs 200 --batch_size 64 --lr 1e-4 --num_workers 2 \
    --eval_every_epochs 20 --n_eval_episodes 30 \
    --n_final_eval_episodes 50 \
    --use_wandb --wandb_project hang_bc_diffusion \
    --seed 2026
```

**Status.** Launched in tmux session `diff-rgb` on the L4 (started
2026-05-13).

---

### pcd — PointNet++ SSG + grip

**Hypothesis.** Point clouds avoid the "extract 3D from 2D pixels"
ambiguity that RGB suffers — the policy gets back-projected world-coord
geometry of whatever the depth buffer captured. With 512 points per
frame, PointNet++ should let the encoder focus on the geometric cloth /
peg relationship rather than learning a visual feature hierarchy.
Expected to land between state and rgb in terms of headline success
rate — better than pixels (lower visual ambiguity) but worse than
privileged state (back-projection misses occluded geometry).

**Encoder.** PointNet++ SSG (set-abstraction) — CUDA backend
(erikwijmans/Pointnet2_PyTorch) on the L4 after a custom build with
`TORCH_CUDA_ARCH_LIST="8.9"` (Ada Lovelace). Falls back to a vendored
pure-Python implementation on Mac / CPU dev. Three SA layers (128 →
32 → global pool, mlp dims 128 → 256 → 256), producing a 256-d pcd
feature. Same grip projection as RGB. Per-frame feature: **268-d**.
`global_cond_dim = 2 * 268 = 536`. PCD normalized per-axis (subtract
training mean, scale by max half-range) to land in roughly [-1, 1]^3 —
required because PointNet++ ball-query radii (0.2 → 0.4) assume a
unit-cube scale.

**Wandb run name.**
```
diff260513-212605_pcd_lr1e-4_e200_bs128_sm-legacy_s2026
```

**Command.**
```bash
python experiments/hang_obs_exp/scripts/train_diffusion_bc.py \
    --demo_path logs/hang_obs_exp/bc_demos_legacy_v1 \
    --obs_mode pcd \
    --success_metric legacy --success_factor 1.2 \
    --num_epochs 200 --batch_size 128 --lr 1e-4 --num_workers 2 \
    --eval_every_epochs 20 --n_eval_episodes 30 \
    --n_final_eval_episodes 50 \
    --use_wandb --wandb_project hang_bc_diffusion \
    --seed 2026
```

**Status.** Launched in tmux session `diff-pcd` on the L4 (started
2026-05-13).

---

## Output artifacts per run

Each logdir at `logs/hang_obs_exp/diffusion_bc/<obs_mode>/<run_name>/`
contains:

| file | contents |
| ---- | -------- |
| `config.json` | every CLI arg + collection metadata + git rev + active DeformEnv constants (MAX_ACT_VEL, FINAL_REWARD_MULT, etc.) |
| `obs_normalizer.pkl` | fitted normalizer stats — written **once at start** so any intermediate ckpt is self-sufficient at deploy time |
| `policy.pt` | final EMA weights at epoch 200, plus the metadata (obs_mode, obs_horizon, pred_horizon, etc.) needed to reconstruct the model + the embedded `final_eval` metrics |
| `policy_best.pt` | **always written** — mirror of EMA weights at whichever mid-training eval scored highest `eval/success_rate`. May correspond to an earlier epoch than 200 if the model peaked mid-training. |
| `policy_ep<NNNN>.pt` | **opt-in via `--save_every_epochs`** — periodic numbered ckpts at each eval pass. Off by default in the launched commands above. |
| (stdout via tee) `~/github/dedo/logs/hang_obs_exp/diffusion_bc/<mode>.log` | full training log including the parity-check lines and per-eval episode prints |

To deploy from a saved ckpt:

```python
import torch, pickle
from experiments.hang_obs_exp.scripts._diffusion_policy import (
    DiffusionPolicy, build_encoder, ObsNormalizer)

ckpt = torch.load('policy_best.pt', map_location='cpu')
norm = ObsNormalizer(ckpt['obs_mode'])
with open('obs_normalizer.pkl', 'rb') as f:
    norm.load_state_dict(pickle.load(f))

# Reconstruct encoder + policy from metadata. obs_mode-specific kwargs
# follow the same pattern as build_encoder in train_diffusion_bc.py.
```

---

## Wandb workspace setup (recommended)

Create three panels grouped by `obs_mode`:
- **Training**: `train/epoch_loss` vs `train/epoch`. Should be a clean
  downward curve. Wildly different magnitudes per mode are expected
  (different obs_feat_dim, different prediction targets).
- **Eval (the headline)**: `eval/success_rate` vs `train/epoch`. This
  is the cross-mode comparison plot. With `success_metric=legacy` it's
  legacy success. The other two rates (`eval/success_hanging`,
  `eval/success_topological`) live in the metric panel alongside.
- **Best tracking**: `best/eval_success_rate` vs `best/epoch`. Monotone
  non-decreasing — corresponds to whichever eval pass mirrored to
  `policy_best.pt`.
- **Final eval**: `final_eval/success_rate` (single point), should
  match the `eval/success_rate` at epoch 200 within sampling noise.
  The eval set differs by one seed offset, so cross-set generalization
  signal lives here.

Filter the runs table by tag `algo=diffusion_bc` + `success_metric=legacy`
to keep these three together. Run-name suffix (`_state_`, `_rgb_`,
`_pcd_`) distinguishes them.

---

## Changes that landed before launch

Recorded here for reproducibility — each was a silent bug or fairness
gap that would have invalidated the comparison if shipped.

1. **Gripper proprioception added to RGB and PCD obs.** Originally the
   visual modes saw only the image / point cloud. State mode embedded
   gripper state in the first 12 of its 18 dims. Without parity, state
   had an unfair advantage. Now all three modes carry a 12-dim grip
   input (state inline, RGB/PCD as a separate dict key consumed by the
   encoder).
2. **MAX_ACT_VEL threaded from demo pkl into eval env.** Demos
   collected at MAX_ACT_VEL=3.5 stored normalized actions. Without
   patching the eval env's class attribute, the env would have
   unscaled the policy's actions by dedo's default 10.0 — eval gripper
   velocities ~3× what training data prescribed. Now read from pkl
   and patched before any eval env is built.
3. **cam_viewmat preserved through `gym.make`.** The eval env builder
   originally restored `sys.argv` before `gym.make`, so dedo's
   `preset_override_util` (which reads sys.argv at construction time
   to decide which args the user "owned") would overwrite cam_viewmat
   with the HangProcCloth preset (yaw=314, target z=5.3). Fixed by
   keeping sys.argv patched through env construction inside a
   try/finally.
4. **PointNet++ BatchNorm → GroupNorm.** Pure-Python backend uses BN
   which depends on running-stats at eval batch=1. Replaced
   recursively via `_replace_bn_with_gn` so eval is invariant to batch
   size and procgen-cloth distribution shift.
5. **Dataset `pad_after`.** Was `pred_horizon - 1` (deviated from
   pusht reference); switched to `action_horizon - 1` so padded tail
   windows correspond to actions the policy will actually be asked to
   execute, instead of wasting model capacity on "hold last action"
   tails.
6. **Strict mismatch handling** on demo-dir metadata. Mixed
   `cam_resolution`, `pcd_n_points`, or `max_act_vel` across pkls in
   the same demo dir now raises `RuntimeError` instead of warning —
   silent shape / scale mismatches caused real bugs in earlier
   iterations.
7. **Checkpoint scheme.** Added `--save_every_epochs` (off by default)
   and always-on `policy_best.pt` mirroring on each eval improvement,
   so a run that's killed mid-way still leaves a recoverable policy
   at the best-performing epoch rather than nothing.

These also benefit `train_privileged.py` (the PPO sweep) — equivalent
MAX_ACT_VEL guards (strict check in `_load_manual_demos`, auto-restore
on `--load_checkpoint`) were added to that script in the same audit
pass. See [RUNS.md](RUNS.md) for context.

---

## Next steps after the three runs land

1. Compare `final_eval/success_rate` across state / rgb / pcd. The gap
   between state and rgb is the visuomotor BC tax for this task; the
   gap between rgb and pcd quantifies depth-augmentation value.
2. Multi-seed the winner. The launched runs all use `seed=2026`. For
   paper-quality SE, replicate with `--seed 7, 42, 1234` and average.
   Wandb groups runs by name suffix so they sort cleanly.
3. Sanity-check `eval/success_legacy` vs `eval/success_hanging` per
   mode. If they diverge dramatically, that's a useful artifact for
   the success-metric discussion (RUNS.md "Success-metric discovery").
4. If the visual modes underperform expectations, the first
   experimental knob to turn is **dataset size** (collect 300-500
   demos) rather than architecture — diffusion policy paper consistently
   shows BC scaling with data when the architecture is correct.
