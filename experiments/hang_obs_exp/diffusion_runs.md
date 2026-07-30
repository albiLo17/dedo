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
6. **v1 launched, partial results, kill + relaunch as v2 (2026-05-14).**
   v1 ran for ~125 / 46 / 48 epochs on state / rgb / pcd before being
   killed. State saturated at 1.0 legacy success in the first eval
   pass; pcd was at 50% legacy / 80-90% hanging and climbing; rgb was
   stuck at 0 across every metric (random ResNet-18 init on 150
   demos = visuomotor BC out of data regime). Three changes for v2:
   sim ctrl freq 30 → 15 Hz (longer-horizon actions per step), RGB
   uses ImageNet pretrain (new `--pretrained_rgb` flag), and the new
   checkpoint/video infra (best-eval ckpt, periodic ckpts, eval mp4s)
   that hadn't existed at v1 launch time gets used. See "v1 partial
   results" + "v2 — relaunch" sections below.
7. **v3 — fairness, PCD audit, data scale, chunking fix
   (2026-05-13).** Eight substantive changes before the comparison
   runs ship: (a) **pull-taut bug fix**: scripted-controller demos at
   15 Hz held the trailing waypoint velocity for the entire
   post-trajectory tail (~5 s at `max_episode_len=120`), training a
   sim2real anti-pattern of "drag the cloth taut against the peg."
   Now zero-action hold + a 5-frame brake, with per-episode
   `max_episode_len = traj_len + tail` so demos end at ~51 control
   steps with 88% active frames instead of 37% active at v2. (b) **PCD
   architecture audit**: cloth-only via pybullet seg-mask filter
   (drops ~30-40% of points previously wasted on the static
   peg/pole/flag), point count bumped 512 → 2048, PointNet++ SSG
   ball-query radii tuned from defaults `0.2 / 0.4` (designed for
   unit-cube-filling objects) to `0.1 / 0.3` so the multi-scale
   hierarchy actually resolves local vs. regional cloth geometry
   instead of collapsing to two global features. (c) **Goal
   conditioning for RGB and PCD**: privileged obs already includes the
   3-dim hanger goal pose in its last 3 dims; v3 adds the same vector
   to RGB and PCD encoders via a small projection head so the only
   asymmetry left across modalities is hole-centroid extraction (the
   actual privileged advantage). (d) **Same-camera projection
   matching in debug viz** so settle frames inside `make_final_steps`
   render with the same fov=60 as the obs camera (was using dedo's
   default DEFAULT_CAM_PROJECTION fov≈90). (e) **Demo scale 150 →
   1000**: v2 eval (with brake-tail demos already applied) showed
   state stalled at 0.5 vs v1's 1.0 — the gap cleanly tracks the 4×
   drop in obs-action pairs (~7k v2 at 15 Hz × 51 steps vs ~30k v1
   at 30 Hz × 200 steps), not the design redesign itself. v3 collects
   1000 demos for ~50k pairs (beyond v1 scale), since visual modes have
   more headroom to absorb data than identity-encoder state. (f)
   **`action_horizon` 8 → 4**: at 15 Hz, an 8-step chunk is 533 ms of
   open-loop execution (v1 at 30 Hz had 267 ms), and with ~51-step
   episodes that's only ~6 obs-conditioned decisions per rollout vs
   v1's ~25. Drop to 4 to restore decision density; keep
   `pred_horizon=16` for diffusion planning depth.
8. **v4 — randomized hanger xy (2026-05-15).** v3's fixed peg
   location at world `(0, 0, 8)` lets the visual modes "cheat" by
   memorizing where the peg appears in the image; goal conditioning
   becomes load-bearing only for hole-centroid extraction. v4 samples
   `(dx, dy) ~ Uniform[-r, +r]^2` at every reset and shifts the hanger
   URDF + tallrod URDF + `goal_pos` together by the same delta, so the
   peg appears at a different world location each episode. The
   scripted controller already reads `goal_pos[0]` for its waypoint
   targets, the per-step `obs['goal']` capture already reads it for
   normalization, and the three encoders already consume the goal —
   so the only changes needed were the env-side sampling
   ([deform_env.py:240-260](../../dedo/envs/deform_env.py#L240-L260))
   and parity plumbing through collect/train/eval. **Radius not yet
   tuned** — `--randomize_goal_radius` is a CLI flag with a
   placeholder default of 1.5 m below; validate camera framing at
   that radius (or your chosen alternative) with
   [_diag_goal_randomization.py](scripts/_diag_goal_randomization.py)
   before launching collection. See "v4 — randomized hanger xy"
   section for design notes + launch commands.

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

**Status (v1).** Killed at ~epoch 125/200 (2026-05-14). The eval curve
had already saturated at 1.0 legacy success rate by the first eval pass
(epoch 20). Superseded by v2 (see below).

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

**Status (v1).** Killed at ~epoch 46/200 (2026-05-14). The eval curve
was flat at 0 / 0 / 0 across legacy, hanging, topological — RGB never
got off the floor. Root cause: random Kaiming init on a 11 M-param
ResNet-18 against 150 demos × ~80 steps is well outside the
data-coverage regime where visuomotor BC converges from scratch.
Superseded by v2 with `--pretrained_rgb` (ImageNet init).

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

**Status (v1).** Killed at ~epoch 48/200 (2026-05-14). Eval rate was at
50% legacy / 80-90% hanging / 15% topological — already learning, but
killed alongside the others to keep all three on the same v2 dataset.
Notable: PCD was the only mode where `eval/success_hanging` >
`eval/success_legacy` (the opposite of state), suggesting the PCD
policy produced "threaded but loose" rollouts that hanging caught but
legacy missed. Worth a closer look in the v2 videos.

---

## v1 partial results (before kill)

The three v1 runs ran for different epoch counts on a shared L4 (state
trains fastest, so it got further). All three were killed simultaneously
on 2026-05-14 to share a clean v2 dataset and infra. Approximate wandb
state at kill time:

| mode | epochs done | `eval/success_legacy` | `eval/success_hanging` | `eval/success_topological` |
| ---- | ----------- | --------------------- | ---------------------- | -------------------------- |
| state | ~125 | 1.0 (since epoch 20) | ~0.95 | ~0.30 |
| rgb | ~46 | 0.0 | 0.0 | 0.0 |
| pcd | ~48 | ~0.50 | ~0.80-0.90 | ~0.15 |

**Two observations worth keeping for the writeup:**

1. **State 1.0 legacy / 0.30 topological is the collapsed-cloth
   pathology in action.** Every state-mode rollout passed the
   centroid-distance check (legacy), most also passed the hanging-on-peg
   3D check, but most also failed the winding-number check. This is
   exactly what RUNS.md's "Why topological breaks on this task"
   subsection predicts — the cloth collapses flat against the peg at
   the end of the episode, hole-loop vertices align along a vertical
   line through the peg axis, the xy-projection becomes degenerate,
   and the winding sum rounds to ~0. **Don't anchor analysis on
   topological for this task.**
2. **PCD's hanging > legacy reversal is interesting.** The PCD policy
   gets ~80-90% on hanging but only ~50% on legacy. The PCD policy
   likely produces a different threading style than state — perhaps it
   ends episodes with the cloth lower (the depth-back-projection
   doesn't see the peg base from this camera angle as crisply), so
   the centroid is further from the peg in 3D (legacy fails) but
   geometrically still hanging from the peg (hanging passes). Spot-
   checking the v2 videos will tell us.

The RGB 0% across all metrics is the unsurprising story: random-init
ResNet-18 on 150 demos × 80 steps × 128² pixels = the BC-from-scratch
regime that the diffusion-policy paper specifically uses ImageNet
pretrain to escape. v2 fixes this.

---

## v2 — relaunch with sim-rate, RGB pretrain, richer artifacts

Three changes from v1, decided 2026-05-14 after seeing the v1 curves:

1. **Sim control frequency 30 → 15 Hz** (via `--ctrl_freq 15`). Halves
   the number of control steps per episode (from ~80 to ~40-60) so
   each policy decision covers a longer real-world interval. The
   scripted controller's three-phase trajectory (1.4 s lift + 1.0 s
   thread + 0.6 s hold = 3 s) lands in ~45 control steps at 15 Hz
   instead of ~90 at 30 Hz, so `--max_episode_len 120` now gives
   plenty of margin without padding the back half with "hold" steps.
   This also widens the action-magnitude distribution per step (the
   waypoint controller's velocity-per-control-step roughly doubles),
   which together with the `--max_act_vel 4.0` bump should give the
   diffusion policy a clearer action signal to fit.
2. **RGB ImageNet pretrain.** New `--pretrained_rgb` flag loads
   `IMAGENET1K_V1` weights into the ResNet-18 conv stack before the
   BN→GN swap. The BN→GN replacement zeroes the normalization layers'
   gamma/beta/running stats, but the conv kernels carry over — which
   is the bulk of the transferable signal per the diffusion-policy
   paper's recipe. Wired through `train_diffusion_bc.py:100, 267, 726`
   to `_diffusion_policy.py:135-138` (`RGBObsEncoder(pretrained=True)`).
   The `_pre` token in the run-name suffix marks runs that used it.
3. **Richer eval artifacts.** v1 launched before `--save_every_epochs`,
   `policy_best.pt` mirroring, and eval-video logging existed. v2
   inherits all three: periodic numbered ckpts every 20 epochs, an
   always-on best-eval ckpt that updates whenever `eval/success_rate`
   improves, and an mp4 logged to wandb each eval pass (covers policy
   phase + the 500-step post-settle gravity phase, sub-sampled at
   stride 5).

### v2 demo collection

```bash
python experiments/hang_obs_exp/scripts/collect_bc_demos.py \
    --demos_dir logs/hang_obs_exp/bc_demos_15hz_v1 \
    --n_demos 150 \
    --cam_resolution 128 --pcd_n_points 512 \
    --max_act_vel 4.0 \
    --success_metric legacy --success_factor 1.2 \
    --ctrl_freq 15 --max_episode_len 120 \
    --debug_viz_first_n 3 --debug_viz_every 25 \
    --seed 2026
```

Deltas from the v1 collection command:

| flag | v1 | v2 | reason |
| ---- | -- | -- | ------ |
| `--max_act_vel` | 3.5 | 4.0 | small headroom bump for the 15 Hz higher-magnitude steps |
| `--ctrl_freq` | (dedo default 30) | 15 | longer-horizon actions, fewer steps per episode |
| `--max_episode_len` | 200 | 120 | sized for the new ctrl rate; scripted controller fits in ~45 steps |
| `--debug_viz_*` | (off) | first 3 every 25 | demo-collection debug videos so we can eyeball what's being recorded |

The pkl schema additionally now stores `ctrl_freq`, `sim_freq`, and
`sim_steps_per_action` so the training script can verify the eval env
matches the collection rate (the existing `max_act_vel` / `cam_viewmat`
parity checks already prevent silent mismatches on those axes; the new
fields extend the same guard pattern to the control rate).

### v2 launch commands

```bash
export DEMOS=~/github/dedo/logs/hang_obs_exp/bc_demos_15hz_v1
```

```bash
tmux new-session -d -s diff-state "bash -lc 'source ~/miniforge3/etc/profile.d/conda.sh && conda activate dedo38 && cd ~/github/dedo && python experiments/hang_obs_exp/scripts/train_diffusion_bc.py --demo_path $DEMOS --obs_mode state --state_key hole_centroid --success_metric legacy --success_factor 1.2 --num_epochs 200 --batch_size 256 --lr 1e-4 --num_workers 2 --eval_every_epochs 20 --n_eval_episodes 30 --n_final_eval_episodes 50 --save_every_epochs 20 --use_wandb --wandb_project hang_bc_diffusion --seed 2026 2>&1 | tee logs/hang_obs_exp/diffusion_bc/state.log; echo DONE; sleep infinity'"

tmux new-session -d -s diff-rgb "bash -lc 'source ~/miniforge3/etc/profile.d/conda.sh && conda activate dedo38 && cd ~/github/dedo && python experiments/hang_obs_exp/scripts/train_diffusion_bc.py --demo_path $DEMOS --obs_mode rgb --pretrained_rgb --success_metric legacy --success_factor 1.2 --num_epochs 200 --batch_size 64 --lr 1e-4 --num_workers 2 --eval_every_epochs 20 --n_eval_episodes 30 --n_final_eval_episodes 50 --save_every_epochs 20 --use_wandb --wandb_project hang_bc_diffusion --seed 2026 2>&1 | tee logs/hang_obs_exp/diffusion_bc/rgb.log; echo DONE; sleep infinity'"

tmux new-session -d -s diff-pcd "bash -lc 'source ~/miniforge3/etc/profile.d/conda.sh && conda activate dedo38 && cd ~/github/dedo && python experiments/hang_obs_exp/scripts/train_diffusion_bc.py --demo_path $DEMOS --obs_mode pcd --success_metric legacy --success_factor 1.2 --num_epochs 200 --batch_size 128 --lr 1e-4 --num_workers 2 --eval_every_epochs 20 --n_eval_episodes 30 --n_final_eval_episodes 50 --save_every_epochs 20 --use_wandb --wandb_project hang_bc_diffusion --seed 2026 2>&1 | tee logs/hang_obs_exp/diffusion_bc/pcd.log; echo DONE; sleep infinity'"
```

Expected v2 wandb run names:
```
diff<TS>_state_lr1e-4_e200_bs256_sm-legacy_s2026
diff<TS>_rgb_pre_lr1e-4_e200_bs64_sm-legacy_s2026          ← _pre token marks ImageNet init
diff<TS>_pcd_lr1e-4_e200_bs128_sm-legacy_s2026
```

### Open metric-validation question (followup, not blocking)

We haven't independently verified which of `legacy`, `hanging`, or
`topological` matches human-eyeball "did the cloth thread the peg" most
closely. RUNS.md documents the legacy false-positive (cloth lands
beside peg, centroid happens close) and false-negative (cloth threads
deep, centroid 3D distance dominated by z) modes, and proposes the
hanging-on-peg three-check as a fix — but the hanging metric's
`lat_threshold` / `min_extent` thresholds are calibrated against
intuition, not labeled video. Now that v2 logs eval videos
automatically (mid-training and final, mp4 per eval pass), a half-day
followup is:

1. After the v2 runs land, collect the final-eval videos for all three
   modes (n=50 episodes each).
2. Hand-label each episode as visually-threaded / not-threaded.
3. Build a 3-way confusion matrix vs each of the three metrics.
4. Pick the metric with the lowest hand-label disagreement as the
   canonical one going forward, and re-anchor reporting if needed.

The runs themselves don't depend on the answer — we log all three rates
per eval pass, so the analysis can re-anchor without retraining.

---

## v3 — fairness, PCD audit, data scale, chunking fix

Decided 2026-05-13 after auditing v2's PCD encoder configuration,
inspecting v2 debug videos, **and** seeing v2's eval curves stall well
short of v1 even with v3-style brake-tail demos already in place (state
0.5 vs v1's 1.0; pcd peak ~0.13 with subsequent overfit drop; rgb
floored). Eight substantive changes; each addresses a specific risk to
the cross-modality comparison. Items 1–6 were the original v3 fairness
+ architecture pass; items 7–8 were added after the v2 eval read
revealed the data scale + chunking issues.

### What changed and why

1. **Pull-taut behavior eliminated.** v2's
   `collect_bc_demos.py` filled the post-trajectory tail by holding
   `last_action` (the final waypoint velocity ~0.55 in normalized
   action space, pointing -y/-z). At 15 Hz × 75 hold frames that was
   ~5 s of PD-driven drag against the peg — produced 100% success but
   trained a sim2real-fragile "strain the cloth against the goal"
   maneuver. v3 commands **zero velocity** after the trajectory ends
   and the planned `--episode_tail_frames=5` of zero-action gives the
   PD controller time to brake the anchor before the gravity-settle
   phase fires. Per-episode `max_episode_len = traj_len + tail` so
   episodes are ~51 steps instead of 121, with 88% active frames
   instead of 37%. Empirically the success rate held (5/6 → ~80%) and
   the topological-success rate *improved* (4 of 5 vs ~1 of 5 in v2)
   because the cloth drapes cleanly under gravity instead of being
   dragged past.

2. **Cloth-only PCD via pybullet segmentation mask.** pybullet's
   `getCameraImage` returns a per-pixel seg-mask alongside depth.
   `_bc_obs_helpers.cloth_only_pcd` filters the depth back-projection
   to pixels where `seg_id == deform.deform_id`, so the saved PCD
   contains 100% cloth surface points instead of ~60% cloth + ~30%
   peg/pole/flag + ~10% base. The peg's geometry is constant across
   episodes (HangProcCloth randomizes only the cloth), so dropping it
   loses no information the policy needs.

3. **PCD point budget 512 → 2048.** With cloth-only filtering, the
   2048-point budget concentrates ~6× more surface density on the
   deformable object the policy actually reasons about. Disk cost
   per demo grows from ~4 MB → ~6 MB; full 150-demo dataset stays
   under 1 GB.

4. **PointNet++ ball-query radii tuned to cloth-only normalized
   scale.** v2 used the canonical SSG radii (0.2 / 0.4) which were
   designed for unit-cube-filling objects (ShapeNet etc.) and on our
   data caused the multi-scale hierarchy to collapse — layer-1 ball-
   query at radius 0.2 covered the entire cloth, layer-2 at 0.4
   covered the entire scene. The empirical per-frame cloth extent
   in normalized space is ~0.9 × 0.4 × 1.0 (cloth diameter ≈ 1.4),
   so radii **0.1 / 0.3** correctly resolve local (hole-edge,
   wrinkles) vs. regional (cloth pose, hole position) features
   before the group-all global pool. See [`_diffusion_policy.py:195-219`](scripts/_diffusion_policy.py#L195-L219)
   for the calibration comment.

5. **Goal conditioning for RGB and PCD.** Privileged
   `hole_centroid` obs already embeds the 3D hanger pose in its last
   3 dims. v1/v2 visual modes only saw image/PCD + 12-dim gripper
   proprio, so they had to implicitly learn the (fixed) peg location
   from data — a free signal the privileged policy got. v3 saves the
   hanger goal as a separate `goal` key in every demo pkl and the
   `RGBObsEncoder` / `PointCloudObsEncoder` consume it via a small
   `_GoalProjection` (3 → 8) concatenated with the visual feature.
   Now the only privileged-only signal is the cloth-derived hole
   centroid, which is the actual hypothesis under test.

6. **Settle-frame camera matches obs camera in debug videos.** v2's
   `deform.render()` used dedo's `DEFAULT_CAM_PROJECTION` (fov≈90)
   while obs RGB used `proj_matrix()` (fov=60). Settle frames in the
   debug mp4s appeared zoomed out relative to the policy-phase frames.
   v3 monkey-patches `deform.render` at `collect_bc_demos.py` startup
   so both code paths share fov=60. Cosmetic-only — no training-time
   effect.

7. **Demo scale 150 → 1000.** The v2 eval runs (which already had the
   v3 brake-tail fix applied — only items 2–8 here postdate them) saw
   state stall at ~0.5 success vs v1's 1.0, pcd peak at ~0.13 then
   overfit-decay, rgb_pre floor near 0. Tracing back to data: at 15 Hz
   × ~51-step demos, 150 demos produces ~7k obs-action pairs vs v1's
   ~30k at 30 Hz × 200-step demos. **4× less training data is the
   dominant variable**, not anything inherent to the 15 Hz / brake-tail
   redesign. The loss curve confirms: all six v2 runs flattened by step
   ~50 to a noise floor ~0.015–0.02 regardless of obs mode, the textbook
   small-dataset memorization fingerprint. v3 collects **1000 demos**
   for ~50k pairs — beyond v1 scale, since visual modes (RGB, PCD) have
   more headroom to absorb data than the 18-dim identity-encoder state
   baseline does. Disk cost ~6 GB; collection wall-clock ~75–90 min on
   L4 (linear in demo count, still cheap relative to per-mode training).

8. **`action_horizon` 8 → 4 at 15 Hz.** At 15 Hz, an 8-step
   open-loop chunk spans 533 ms vs v1's 267 ms at 30 Hz (8 / 30).
   Compounded with v3's ~51-step episodes, that left the policy with
   only ~6 obs-conditioned decision points per rollout vs v1's ~25.
   Fine-grained threading is precisely the regime where re-anchoring
   to fresh observations matters — small denoising errors that v1 could
   correct twice per second, v2 had to live with for half a second.
   v3 sets `--action_horizon 4` to restore the per-second decision
   density v1 had, while keeping `pred_horizon=16` so the diffusion
   model still plans over a 1067 ms horizon (DP-paper recipe: predict
   more than you execute, throw away the tail).

### Schema additions to demo pkls (v3)

| key | shape | description |
| --- | ----- | ----------- |
| `obs['pcd']` | (T, 2048, 3) | **cloth-only** points (was 512 with peg) |
| `obs['goal']` | (T, 3) | hanger pose / WBOX (constant per episode) |
| `ctrl_freq` | float | 15.15 Hz actual (500/33) |
| `sim_freq` | int | 500 (PyBullet step rate) |
| `sim_steps_per_action` | int | 33 |
| `max_act_vel` | float | 4.0 |
| `episode_tail_frames` | — | not stored directly; visible via `len - traj_len` |

### v3 demo collection

```bash
python experiments/hang_obs_exp/scripts/collect_bc_demos.py \
    --demos_dir logs/hang_obs_exp/bc_demos_15hz_pcd2048_n1000_v1 \
    --n_demos 1000 \
    --cam_resolution 128 --pcd_n_points 2048 \
    --max_act_vel 4.0 \
    --success_metric legacy --success_factor 1.2 \
    --ctrl_freq 15 --max_episode_len 200 --episode_tail_frames 5 \
    --debug_viz_first_n 3 --debug_viz_every 100 \
    --seed 2026
```

Deltas from the v2 collection command:

| flag | v2 | v3 | reason |
| ---- | -- | -- | ------ |
| `--n_demos` | 150 | **1000** | restore data scale lost to the 15 Hz × shorter-episode redesign; ~50k pairs vs v2's ~7k, beyond v1's ~30k |
| `--pcd_n_points` | 512 | 2048 | 4× cloth-surface density now that peg points are filtered out |
| `--max_episode_len` | 120 (fixed) | 200 (safety cap) | per-episode length is now `traj_len + tail` (~51); 200 is just an upper bound |
| `--episode_tail_frames` | (n/a, hold filled tail) | 5 | zero-action brake phase before gravity settle; replaces the trailing-velocity hold |
| `--debug_viz_every` | 25 | 100 | scaled with demo count — at 1000 demos, every-25 was ~40 mp4s; every-100 gives ~10 spot-check videos |
| `--demos_dir` | `bc_demos_15hz_v1` | `bc_demos_15hz_pcd2048_n1000_v1` | `pcd2048_n1000` tokens self-document density + demo count |
| (under the hood) PCD content | cloth + peg + flag + base | **cloth only** (seg-mask filter) | concentrates point budget on the deformable target |
| (under the hood) hold action | trailing-velocity hold | **zero-action hold** | eliminates the pull-taut sim2real anti-pattern |
| (under the hood) `goal` field | not saved | **3-dim hanger pose** | parity with privileged obs for RGB/PCD encoders |

Expected dataset characteristics: ~51 control steps per demo (45
scripted + 5 brake + dedo's 500-tick gravity settle), ~6 MB per pkl
(2048 cloth-only float32 + 128² uint8 RGB + state + grip + goal),
**~6 GB total disk for 1000 demos**. Wall-clock **~75–90 min on L4**.

### v3 launch commands

Three-stage pipeline: collect → train → final eval. Stage 2 needs the
collection from stage 1 to be on disk first (the training script
validates demo-dir metadata before building any models). New flags vs
v2: `--action_horizon 4` (was default 8) to restore decision density
at 15 Hz; `--num_epochs 300` (was 200) since ~7× more data per epoch
means saturation lands later in epoch count; `--n_final_eval_episodes
100` (was 50) for tighter SE on the headline cross-modality numbers
that the whole experiment exists to compare.

```bash
# Stage 1 — collect 1000 demos. Wait for this tmux to print DONE
# (~75–90 min on L4) before launching stage 2.

tmux new-session -d -s diff-collect "bash -lc 'source ~/miniforge3/etc/profile.d/conda.sh && conda activate dedo38 && cd ~/github/dedo && python experiments/hang_obs_exp/scripts/collect_bc_demos.py --demos_dir logs/hang_obs_exp/bc_demos_15hz_pcd2048_n1000_v1 --n_demos 1000 --cam_resolution 128 --pcd_n_points 2048 --max_act_vel 4.0 --success_metric legacy --success_factor 1.2 --ctrl_freq 15 --max_episode_len 200 --episode_tail_frames 5 --debug_viz_first_n 3 --debug_viz_every 100 --seed 2026 2>&1 | tee logs/hang_obs_exp/diffusion_bc/collect.log; echo DONE; sleep infinity'"

# Stage 2 — train the three modes in parallel.

export DEMOS=~/github/dedo/logs/hang_obs_exp/bc_demos_15hz_pcd2048_n1000_v1

tmux new-session -d -s diff-state "bash -lc 'source ~/miniforge3/etc/profile.d/conda.sh && conda activate dedo38 && cd ~/github/dedo && python experiments/hang_obs_exp/scripts/train_diffusion_bc.py --demo_path $DEMOS --obs_mode state --state_key hole_centroid --action_horizon 4 --success_metric legacy --success_factor 1.2 --num_epochs 300 --batch_size 256 --lr 1e-4 --num_workers 2 --eval_every_epochs 20 --n_eval_episodes 30 --n_final_eval_episodes 100 --save_every_epochs 20 --use_wandb --wandb_project hang_bc_diffusion --seed 2026 2>&1 | tee logs/hang_obs_exp/diffusion_bc/state.log; echo DONE; sleep infinity'"

tmux new-session -d -s diff-rgb "bash -lc 'source ~/miniforge3/etc/profile.d/conda.sh && conda activate dedo38 && cd ~/github/dedo && python experiments/hang_obs_exp/scripts/train_diffusion_bc.py --demo_path $DEMOS --obs_mode rgb --pretrained_rgb --action_horizon 4 --success_metric legacy --success_factor 1.2 --num_epochs 300 --batch_size 64 --lr 1e-4 --num_workers 2 --eval_every_epochs 20 --n_eval_episodes 30 --n_final_eval_episodes 100 --save_every_epochs 20 --use_wandb --wandb_project hang_bc_diffusion --seed 2026 2>&1 | tee logs/hang_obs_exp/diffusion_bc/rgb.log; echo DONE; sleep infinity'"

tmux new-session -d -s diff-pcd "bash -lc 'source ~/miniforge3/etc/profile.d/conda.sh && conda activate dedo38 && cd ~/github/dedo && python experiments/hang_obs_exp/scripts/train_diffusion_bc.py --demo_path $DEMOS --obs_mode pcd --action_horizon 4 --success_metric legacy --success_factor 1.2 --num_epochs 300 --batch_size 128 --lr 1e-4 --num_workers 2 --eval_every_epochs 20 --n_eval_episodes 30 --n_final_eval_episodes 100 --save_every_epochs 20 --use_wandb --wandb_project hang_bc_diffusion --seed 2026 2>&1 | tee logs/hang_obs_exp/diffusion_bc/pcd.log; echo DONE; sleep infinity'"
```

Expected v3 wandb run names (`_pre` in the rgb suffix marks the
ImageNet init carried over from v2; `_ah4` marks the shorter action
chunk; PCD now uses `pcd2048` density but the suffix doesn't change
because the encoder reads point count from `obs.shape[-2]` at
construction):

```
diff<TS>_state_lr1e-4_e300_bs256_ah4_sm-legacy_s2026
diff<TS>_rgb_pre_lr1e-4_e300_bs64_ah4_sm-legacy_s2026
diff<TS>_pcd_lr1e-4_e300_bs128_ah4_sm-legacy_s2026
```

### v3 eval-cap bug + fix (2026-05-14)

The first v3 state run (`diff260514-073100`) hit **0.1 legacy success at
epoch 20** — wildly off the expected privileged-state ceiling, especially
worse than v2's 0.5 stall. Pulled the first eval video and saw the
gripper anchors pulling the cloth **taut against the peg at episode end**,
exactly the v2 pull-taut anti-pattern that the v3 brake-tail fix was
supposed to eliminate.

**Root cause.** [collect_bc_demos.py](scripts/collect_bc_demos.py) sets
`deform.max_episode_len = len(scripted_traj) + episode_tail_frames`
per-episode (≈51 ctrl steps for v3), but the eval env in
`train_diffusion_bc.py` was constructed with `args.max_episode_len = 200`
and never per-episode-overridden. So every eval episode ran for **~4×
longer than the training distribution** — steps 51-200 fed the diffusion
policy obs windows it had never seen, and the policy drifted into
cumulative lateral force impulses on the anchors. By step 200 the
gripper was way off-axis from the peg, and the gravity-settle phase
(which doesn't release anchors, only stops applying external force on
them — see [deform_env.py:412](../../dedo/envs/deform_env.py#L412)) just
froze the cloth at that stretched configuration.

The bug was masked through v2 because v2 demos had the same
trailing-velocity tail that pull-taut training data — the policy's OOD
behavior past step 51 happened to *match* the training distribution by
luck. v3's clean brake-tail demos exposed it.

**Validation.** Wrote a standalone
[eval_diffusion_bc.py](scripts/eval_diffusion_bc.py) that loads
`policy_best.pt` directly (no demo dir needed; reads env config from
ckpt metadata). On the broken v3 state ckpt:

| `--max_episode_len` | first 12 episodes (legacy success) |
| ------------------- | ---------------------------------- |
| 200 (the bug) | 0 / 3 (matches the wandb 0.1 rate) |
| 56 (in-distribution) | **12 / 12** |

So the policy itself was fine all along — the eval was just measuring
the wrong thing.

**Fix.** Two-layer:

1. **Per-episode dynamic sizing** in
   [evaluate_policy](scripts/train_diffusion_bc.py): after `e.reset()`,
   build the scripted hole-aware trajectory the demo controller WOULD
   have run for the current cloth (`build_hole_aware_waypoints` +
   `build_traj` + `merge_traj`, same calls as `collect_bc_demos.py`),
   set `deform.max_episode_len = len(traj) + args.episode_tail_frames`.
   This mirrors collect-time per-episode termination exactly. New shared
   helper `compute_per_episode_max_len` in
   [_helpers.py](scripts/_helpers.py) used by both the in-training eval
   and the standalone `eval_diffusion_bc.py`.

2. **`args.max_episode_len`** stays as the hard safety ceiling
   (default 200) so a runaway cloth can't run unbounded; demoted from
   "the eval cap" to "the worst-case ceiling."

`episode_tail_frames` defaults to 5 (matches `collect_bc_demos.py`
default) and is now embedded in every ckpt's metadata so
`eval_diffusion_bc.py` can recover it without the demo dir.

**Other small things that fell out of this audit:**

- Added `--resume` flag to `train_diffusion_bc.py` (loads weights + EMA
  shadow + optimizer + scheduler state from a ckpt and continues the
  epoch counter; ckpts saved by the updated `_save_ema_checkpoint`
  persist this state). Useful for crash recovery on future runs.
- `_save_ema_checkpoint` now persists `step_counter`, `best_eval_success`,
  and `best_eval_epoch` alongside the model state.
- `make_final_steps()` doesn't release the gripper anchors (commented
  out in dedo to avoid jerk forces); the gravity-settle drape relies on
  the anchors having their own mass
  (`ANCHOR_MASS=0.1 kg`, [anchor_utils.py:18](../../dedo/utils/anchor_utils.py#L18))
  plus the cloth's distributed weight. This works as long as the anchors
  enter settle at a sensible position (which the fix now guarantees).

**Relaunch.** Same launch commands as the v3 block below — `--resume` is
opt-in, and `--episode_tail_frames` defaults to 5, so the existing
tmux commands work unchanged. The original `diff260514-073100*` runs
were killed; the relaunched ones supersede them.

### Encoder feat_dim across modes (v3)

| mode | encoder | primary feat | + grip | + goal | total |
| ---- | ------- | ------------ | ------ | ------ | ----- |
| state | identity | 18 | (embedded) | (embedded) | 18 |
| rgb | ResNet-18 GroupNorm (ImageNet) | 512 | 12 | 8 | **532** |
| pcd | PointNet++ SSG (radii 0.1 / 0.3), 2048 cloth pts | 256 | 12 | 8 | **276** |

global_cond_dim for the U-Net = `obs_horizon * feat_dim`: 36 (state),
1064 (rgb), 552 (pcd).

### Fairness invariants now enforced across modalities

| invariant | how |
| --------- | --- |
| Same trajectory data | one collection, three obs keys in every pkl |
| Same gripper proprio | 12-dim `grip` saved & consumed by all 3 encoders (state: embedded, rgb/pcd: separate input) |
| **Same goal info** | 3-dim `goal` saved & consumed by all 3 (state: embedded, rgb/pcd: `_GoalProjection`) |
| Same camera | one `cam_viewmat`; RGB and PCD use identical render path |
| Same action normalization | `MAX_ACT_VEL` patched from pkl into eval env |
| Same ctrl freq | `sim_freq` + `sim_steps_per_action` patched from pkl |
| Same success metric | `--success_metric legacy`, all 3 also logged for cross-anchor |

The **only** modality-specific knowledge gap is now: privileged
state gets the cloth-derived 3D hole centroid; RGB and PCD have to
extract it from raw inputs. That is the experimental quantity under
test.

---

## v4 — randomized hanger xy

Decided 2026-05-15. v1–v3 used a fixed hanger pose at world
`(0, 0, 8)` (hanger crossbar) / `(0, 0, 8.2)` (goal_pos). The peg
therefore appears at the same image-space pixel range in every demo,
which means RGB and PCD encoders can in principle memorize that the
threading target lives at a known location and treat the goal-cond
input as redundant. v4 breaks that shortcut: every reset samples
`(dx, dy) ~ Uniform[-r, +r]^2` (default placeholder `r = 1.5 m`,
unvalidated — see below) and shifts the hanger URDF + tallrod URDF +
`goal_pos[*]` together by the same delta. Same procedural cloth, same
gripper start pose, same camera viewmat — only the peg moves.

### Hypothesis

- **state** should still saturate near 1.0: privileged obs already
  embeds the goal in its last 3 dims, so the policy gets the new peg
  pose for free each episode.
- **rgb** and **pcd**: with v3 architecture the goal vector reaches
  the encoder via `_GoalProjection`, but the encoder might not have
  *needed* to use it under a fixed peg. v4 makes goal conditioning
  load-bearing: the rate gap between v3 (fixed) and v4 (randomized)
  is a proxy for "how much did the visual encoder rely on memorized
  peg location vs. the goal input."
- Cross-modality gap on v4 (state minus rgb/pcd) is the cleaner
  measure of the privileged advantage — under v3 the privileged
  advantage was conflated with the visual-modes' ability to memorize
  geometry.

### Open question — radius not yet tuned

The 1.5 m placeholder is a reasonable mid-range starting point under
the v3 camera (yaw=45, pitch=-5, dist=14, target=(0, 0, 5.5)) but has
**not** been visually verified yet. Two failure modes to check before
collection:

1. **Camera framing.** At the box corners (`±1.5, ±1.5`), the peg
   could crop at the image edge, or the cloth (anchored at fixed
   gripper start positions) could end up outside the camera frustum.
2. **Scripted-controller reach.** The waypoints in
   `build_hole_aware_waypoints` are computed as offsets from
   `goal_pos`; if the goal moves to `+1.5x`, the gripper has to
   traverse further, and the velocity-clipped trajectory may not
   reach the threading pose in the time budget. The collector retries
   failures, so the visible signal is a drop in scripted success rate
   (logged as `kept/attempts` at the end of collection).

Use [_diag_goal_randomization.py](scripts/_diag_goal_randomization.py)
to validate camera framing for any (radius, cam_viewmat) pair before
committing — it renders a 3×3 mosaic showing the scene at the 9 grid
points (center + 4 edges + 4 corners) of the randomization box, with
a green marker at the actual `goal_pos`. The script works
independently of the env-side randomization (re-poses rigid bodies
via pybullet directly), so you can also use it to pre-screen new
`--cam_viewmat` candidates for future datasets.

```bash
# Validate v4 framing at the default 1.5 m radius
python experiments/hang_obs_exp/scripts/_diag_goal_randomization.py \
    --radius 1.5 \
    --save_path logs/hang_obs_exp/diag/cam_v3_r1.5.png

# Compare radii at the same viewpoint
python experiments/hang_obs_exp/scripts/_diag_goal_randomization.py \
    --radius 1.0 \
    --save_path logs/hang_obs_exp/diag/cam_v3_r1.0.png
python experiments/hang_obs_exp/scripts/_diag_goal_randomization.py \
    --radius 2.0 \
    --save_path logs/hang_obs_exp/diag/cam_v3_r2.0.png
```

Pick the largest radius that keeps the peg fully in frame at all 4
corners *and* lets the scripted controller maintain ≥80% success at
collection time (eyeball the `kept/attempts` ratio on a 50-demo
pilot run). The number locked in for v4 collection will be recorded
in the demo pkls under `randomize_goal_radius` and propagated to
training/eval automatically.

### What changed in code

Five touchpoints for the randomization itself, all minimal:

| where | change |
| ----- | ------ |
| [dedo/utils/args.py](../../dedo/utils/args.py) | new flag `--randomize_goal_radius` (float, default 0.0). |
| [dedo/envs/deform_env.py](../../dedo/envs/deform_env.py) `load_objects` | when `scene_name == 'hangcloth'` and radius > 0, sample `goal_dxy ~ Uniform[-r, +r]^2`, apply to hanger + tallrod `basePosition` and to every entry of `goal_poses`. Reproducible via the existing np.random seed; the dxy is stashed on `self._last_goal_dxy` for introspection. |
| [collect_bc_demos.py](scripts/collect_bc_demos.py) | argparser flag + sys.argv passthrough + per-pkl metadata field `randomize_goal_radius`. The per-step `obs['goal']` capture is unchanged — it already reads `deform.goal_pos[0] / 20.0` each step, which now varies per episode. |
| [train_diffusion_bc.py](scripts/train_diffusion_bc.py) | parity tracker `recorded_randomize_goal_radii` reads the field from every pkl; mixed values across the demo dir raise (same strict pattern as `max_act_vel`). `eval_randomize_goal_radius` is picked from the demos, threaded into the eval-env `sys.argv`, and saved into the ckpt metadata. No CLI flag — the radius comes from the demos because the policy fits that specific spatial distribution. |
| [eval_diffusion_bc.py](scripts/eval_diffusion_bc.py) | reads `randomize_goal_radius` from ckpt metadata (defaults to 0.0 for legacy ckpts) and patches eval-env sys.argv. |

In the same pass, the camera-match patch that was previously inlined
(and conditional on the recording flag) in three places was
factored into a single helper
[`patch_deform_render_to_obs_camera`](scripts/_bc_obs_helpers.py)
and applied **unconditionally at env construction** in
`collect_bc_demos.py`, `train_diffusion_bc.py._build_eval_env`,
`eval_diffusion_bc.py`, and `_diag_goal_randomization.py`. This
guarantees that every `deform.render()` call (policy-phase frames,
make_final_steps settle frames, debug panels, mosaic tiles) uses
fov=60 — the same projection as `capture_rgb_depth` feeds the policy
encoders. The view matrix already came from `args.cam_viewmat`
(parity-checked against the demos), so the eval-video camera is now
bit-identically the obs camera by construction. Previously this was
only true when the recording branch was hit; a missed branch could
have silently produced fov≈90 frames.

The scripted hole-aware controller required **zero changes** — it
already reads `underlying.goal_pos[0]` at
[_helpers.py:87](scripts/_helpers.py#L87) and computes waypoints as
offsets from it, so per-episode goal movement is automatic. Reward,
legacy/hanging/topological success metrics, and the `obs['goal']`
auxiliary input all read `goal_pos[0]` too — verified by grep, none
hardcode `(0, 0, 8.2)`.

### v4 demo collection

```bash
python experiments/hang_obs_exp/scripts/collect_bc_demos.py \
    --demos_dir logs/hang_obs_exp/bc_demos_15hz_pcd2048_n1000_randgoal_v1 \
    --n_demos 1000 \
    --cam_resolution 128 --pcd_n_points 2048 \
    --max_act_vel 4.0 \
    --success_metric legacy --success_factor 1.2 \
    --ctrl_freq 15 --max_episode_len 200 --episode_tail_frames 5 \
    --randomize_goal_radius 1.5 \
    --debug_viz_first_n 3 --debug_viz_every 100 \
    --seed 2026
```

Deltas from v3 collection:

| flag | v3 | v4 | reason |
| ---- | -- | -- | ------ |
| `--randomize_goal_radius` | n/a | **1.5** (placeholder) | per-episode xy randomization of the hanger pose; tune before launch via `_diag_goal_randomization.py` |
| `--demos_dir` | `bc_demos_15hz_pcd2048_n1000_v1` | `bc_demos_15hz_pcd2048_n1000_randgoal_v1` | `randgoal` token marks the radius>0 dataset |

Everything else identical to v3 (same camera, same MAX_ACT_VEL, same
2048 cloth-only PCD, same brake-tail).

### v4 launch commands

```bash
# Stage 1 — collect 1000 demos. Wait for this to print DONE
# (~75-90 min on L4, possibly longer if the scripted controller's
# success rate drops near the corners of the randomization box).

tmux new-session -d -s diff-collect "bash -lc 'source ~/miniforge3/etc/profile.d/conda.sh && conda activate dedo38 && cd ~/github/dedo && python experiments/hang_obs_exp/scripts/collect_bc_demos.py --demos_dir logs/hang_obs_exp/bc_demos_15hz_pcd2048_n1000_randgoal_v1 --n_demos 1000 --cam_resolution 128 --pcd_n_points 2048 --max_act_vel 4.0 --success_metric legacy --success_factor 1.2 --ctrl_freq 15 --max_episode_len 200 --episode_tail_frames 5 --randomize_goal_radius 1.5 --debug_viz_first_n 3 --debug_viz_every 100 --seed 2026 2>&1 | tee logs/hang_obs_exp/diffusion_bc/collect.log; echo DONE; sleep infinity'"

# Stage 2 — train the three modes in parallel. NB: no train-side
# --randomize_goal_radius flag exists; the value is read from the
# demo pkls and parity-checked across the directory.

export DEMOS=~/github/dedo/logs/hang_obs_exp/bc_demos_15hz_pcd2048_n1000_randgoal_v1

tmux new-session -d -s diff-state "bash -lc 'source ~/miniforge3/etc/profile.d/conda.sh && conda activate dedo38 && cd ~/github/dedo && python experiments/hang_obs_exp/scripts/train_diffusion_bc.py --demo_path $DEMOS --obs_mode state --state_key hole_centroid --action_horizon 4 --success_metric legacy --success_factor 1.2 --num_epochs 300 --batch_size 256 --lr 1e-4 --num_workers 2 --eval_every_epochs 20 --n_eval_episodes 30 --n_final_eval_episodes 100 --save_every_epochs 20 --use_wandb --wandb_project hang_bc_diffusion --seed 2026 2>&1 | tee logs/hang_obs_exp/diffusion_bc/state.log; echo DONE; sleep infinity'"

tmux new-session -d -s diff-rgb "bash -lc 'source ~/miniforge3/etc/profile.d/conda.sh && conda activate dedo38 && cd ~/github/dedo && python experiments/hang_obs_exp/scripts/train_diffusion_bc.py --demo_path $DEMOS --obs_mode rgb --pretrained_rgb --action_horizon 4 --success_metric legacy --success_factor 1.2 --num_epochs 300 --batch_size 64 --lr 1e-4 --num_workers 2 --eval_every_epochs 20 --n_eval_episodes 30 --n_final_eval_episodes 100 --save_every_epochs 20 --use_wandb --wandb_project hang_bc_diffusion --seed 2026 2>&1 | tee logs/hang_obs_exp/diffusion_bc/rgb.log; echo DONE; sleep infinity'"

tmux new-session -d -s diff-pcd "bash -lc 'source ~/miniforge3/etc/profile.d/conda.sh && conda activate dedo38 && cd ~/github/dedo && python experiments/hang_obs_exp/scripts/train_diffusion_bc.py --demo_path $DEMOS --obs_mode pcd --action_horizon 4 --success_metric legacy --success_factor 1.2 --num_epochs 300 --batch_size 128 --lr 1e-4 --num_workers 2 --eval_every_epochs 20 --n_eval_episodes 30 --n_final_eval_episodes 100 --save_every_epochs 20 --use_wandb --wandb_project hang_bc_diffusion --seed 2026 2>&1 | tee logs/hang_obs_exp/diffusion_bc/pcd.log; echo DONE; sleep infinity'"
```

**Eval-video cadence + camera (relevant for every run, called out
here because v4 is the first run where video sanity-checking
genuinely matters — varied goals = visible spatial signal).** The
launch commands above don't pass any `--video_*` flag, so defaults
apply: `--video_every_evals=1` × `--n_video_episodes=3` ×
`--eval_every_epochs=20` = an mp4 with 3 episodes uploaded to wandb
every 20 epochs, plus an extra mp4 at final eval. The recorded
frames go through the same camera (view matrix + fov=60 projection)
as the policy's obs encoder — guaranteed by
`patch_deform_render_to_obs_camera` applied unconditionally at env
construction — so the videos are a direct visual transcript of what
the policy is conditioning on. To slow the cadence (e.g. record
every other eval pass), pass `--video_every_evals=2`; to disable
training-time videos entirely, `--video_every_evals=0` (the final-
eval pass still records one mp4 unconditionally).

Expected v4 wandb run names — identical scheme to v3 (the run-name
suffix doesn't encode the randomization radius since that's a dataset
property, not a model knob, and downstream filtering happens via the
`bc_demos_*_randgoal_v1` dataset path stored in `config.json`):

```
diff<TS>_state_lr1e-4_e300_bs256_ah4_sm-legacy_s2026
diff<TS>_rgb_pre_lr1e-4_e300_bs64_ah4_sm-legacy_s2026
diff<TS>_pcd_lr1e-4_e300_bs128_ah4_sm-legacy_s2026
```

To keep v3 and v4 runs distinguishable in the wandb runs table, use
the `--wandb_run_name` flag at launch (e.g. `--wandb_run_name v4`) to
prefix the suffix, or filter by `config.demo_path` containing
`randgoal`.

### Fairness invariants — v4 update

Same table as v3 with one row added:

| invariant | how |
| --------- | --- |
| Same trajectory data | one collection, three obs keys in every pkl |
| Same gripper proprio | 12-dim `grip` saved & consumed by all 3 encoders |
| Same goal info | 3-dim `goal` saved & consumed by all 3 (state: embedded, rgb/pcd: `_GoalProjection`) — **now varies per episode** |
| Same camera | one `cam_viewmat`; RGB and PCD use identical render path |
| Same action normalization | `MAX_ACT_VEL` patched from pkl into eval env |
| Same ctrl freq | `sim_freq` + `sim_steps_per_action` patched from pkl |
| Same success metric | `--success_metric legacy`, all 3 also logged |
| **Same goal-randomization radius** | `randomize_goal_radius` patched from pkl into eval env; mixed values across demo dir raise |

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
4. If the visual modes still underperform after v3's 1000-demo
   baseline, the next experimental knob is **dataset size again**
   (collect 2000–3000 demos) rather than architecture — diffusion
   policy paper consistently shows BC scaling with data when the
   architecture is correct, and v3 itself is the existence proof that
   stepping up data 4× was the right first move.
