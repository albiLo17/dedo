# Reward Guide

The HangProcCloth-v1 reward system. Last updated 2026-05-05 after the
`boundary_penalty` / `z_overshoot_penalty` removal.

---

## TL;DR

**Validated PPO config** (reaches **0.7 eval success rate**):
```bash
python experiments/hang_obs_exp/scripts/train_privileged.py \
    --obs_mode hole_centroid \
    --success_bonus 100 \
    --pre_settle_coef 20 \
    --vel_penalty 8 \
    --success_factor 1.2 \
    --use_wandb
```

**Five shaping knobs**, all default to `0` (off):

| Knob | When it fires | Sign |
|---|---|---|
| `success_bonus` | terminal, IFF success | + |
| `fail_penalty` | terminal, IFF failure | − |
| `vel_penalty` | every non-terminal step | − |
| `action_penalty` | every step (incl. terminal) | − |
| `pre_settle_coef` | terminal | − |

Plus `success_factor` (default 1.2) — *not* a reward, the criterion that gates
success vs failure.

---

## Variables and units (read this before the formulas)

Every symbol in this guide is unambiguously one of these. **Nothing is
silently divided or scaled** in the wrapper terms — only `r_dedo_base` and
`last_rwd` (both inside dedo's env) divide by `WORKSPACE_BOX_SIZE = 20`.

| Symbol | Meaning | Units | Typical value |
|---|---|---|---|
| `dist` | hole-centroid → goal Euclidean distance | **meters** | 0.1 to 10 |
| `pre_settle_dist_m` | `dist` at policy handoff, BEFORE the gravity settle | **raw meters** | 0.3 to 5 |
| `hole_radius` | mean dist from hole centroid to hole-loop verts (measured at reset) | meters | 0.4 to 0.6 |
| `mean_vertex_disp_m` | mean per-vertex displacement between consecutive policy steps | **raw meters / step** | 0.01 to 0.3 |
| `action` | policy output, 6-dim | dimensionless ∈ [-1, 1] | — |
| `mean(action²)` | the squared-mean used by `action_penalty` | dimensionless ∈ [0, 1] | 0.05 to 0.5 |
| `r_dedo_base` | dedo's per-step distance reward (`= -dist/20`) | reward points | -1.0 to -0.005 |
| `last_rwd` | dedo's post-settle reward (`= (-dist_after_settle/20) × 400 = -20 × dist_after_settle`) | reward points | -100 to -2 |
| `stepnum` | step index inside dedo's env | int | 0 to 200 |

**Three concrete unit checks** to anchor your intuition:

- `pre_settle_coef = 20` × `pre_settle_dist_m = 1.0 m` → reward penalty of **−20**.
- `vel_penalty = 8` × `mean_vertex_disp_m = 0.05 m/step` → **−0.4 per step** (or **−80 over 200 steps**).
- `action_penalty = 0.5` × `mean(action²) = 0.3` → **−0.15 per step** (or **−30 over 200 steps**).

---

## The full reward formula

### Per-step reward (every step, `done=False`)

```
r_t = r_dedo_base_t                              # = −dist_t / 20            in [-1.0, -0.005]
    − action_penalty × mean(action_t²)           # in [0, action_penalty]
    − vel_penalty   × mean_vertex_disp_m_t       # in [0, vel_penalty × ~0.5]
```

### Terminal step (the step where `done=True`, applied IN ADDITION to the per-step formula above)

```
r_terminal_extra = r_dedo_base_t × (max_episode_len − stepnum)    # ×0 on natural timeout
                 + last_rwd                                        # = −20 × dist_after_settle (m)
                 + success_bonus                                   # IFF adaptive success fires
                 − fail_penalty                                    # IFF success does NOT fire
                 − pre_settle_coef × pre_settle_dist_m             # raw meters; default coef=0
```

Three things to remember about the terminal step:

1. **The early-termination multiplier `(max_episode_len − stepnum)` is ZERO on natural timeout.** dedo increments `stepnum` to `max_episode_len + 1` before computing, so this term only amplifies reward on workspace-bound early exit. With a negative base reward, that amplification is also negative — that's dedo's built-in workspace-exit deterrent.
2. **`vel_penalty` does NOT fire on the terminal step.** The terminal verts delta would mix policy motion with the gravity settle (which the policy can't control). `action_penalty` does still fire, because the action is well-defined regardless.
3. **`success_bonus` and `fail_penalty` are mutually exclusive.** Exactly one fires per terminal (or neither, if `success_factor=None`).

---

## Worked example — successful PPO episode at the validated config

Config: `success_bonus=100`, `pre_settle_coef=20`, `vel_penalty=8`,
`success_factor=1.2`. Imagine a successful episode where the cloth is
moved into position over 200 steps. Numbers are illustrative but in the
right ballpark for the validated 0.7-success run.

Setup at this episode:
- Avg per-step `dist` ≈ 2.0 m (gradually closes from ~4 m to ~0.5 m)
- `pre_settle_dist_m` = 0.5 m
- `dist_after_settle` = 0.3 m
- `mean_vertex_disp_m` ≈ 0.05 m/step (cloth moving smoothly)
- `hole_radius` = 0.5 m → success threshold = 1.2 × 0.5 = **0.6 m**
- 0.3 m < 0.6 m → **success fires**

| Term | Computation | Episode contribution |
|---|---|---|
| Per-step base sum | 200 × (−2.0 / 20) | **−20** |
| Per-step vel_penalty sum | 200 × (8 × 0.05) | **−80** |
| Per-step action_penalty sum | 0 (knob off) | **0** |
| Terminal: early-term multiplier | ×0 (natural timeout) | **0** |
| Terminal: `last_rwd` | −20 × 0.3 m | **−6** |
| Terminal: pre_settle penalty | 20 × 0.5 m | **−10** |
| Terminal: `success_bonus` | +100 (success fired) | **+100** |
| | **Total per-episode reward:** | **−16** |

### Same setup, but **failed** episode

`pre_settle_dist_m` = 2.0 m, `dist_after_settle` = 1.5 m. 1.5 m > 0.6 m → **success does NOT fire**.

| Term | Computation | Episode contribution |
|---|---|---|
| Per-step base sum | 200 × (−3.0 / 20)  *(higher avg dist)* | **−30** |
| Per-step vel_penalty sum | 200 × (8 × 0.05) | **−80** |
| Terminal: `last_rwd` | −20 × 1.5 m | **−30** |
| Terminal: pre_settle penalty | 20 × 2.0 m | **−40** |
| Terminal: `success_bonus` | 0 | **0** |
| | **Total per-episode reward:** | **−180** |

**Success vs fail gap: ~164 reward points.** Of that, +100 comes from
`success_bonus` directly; the remaining ~64 comes from the dist-related
terms (`base_sum`, `last_rwd`, `pre_settle_penalty` all swing in the
same direction). That swing is the gradient PPO actually optimizes.

---

## Constants from `dedo/envs/deform_env.py`

| Constant | Value | Where it shows up |
|---|---|---|
| `MAX_ACT_VEL` | 10.0 m/s | `action ∈ [-1, 1]^6` is rescaled to `±10 m/s` per anchor |
| `WORKSPACE_BOX_SIZE` | 20.0 m | divisor in `r_dedo_base = −dist/20`; also gripper_lims |
| `FINAL_REWARD_MULT` | 400 | `last_rwd = get_reward() × 400` |
| `SUCESS_REWARD_TRESHOLD` | 2.5 | dedo's strict success: `\|last_rwd\| < 2.5` ⇔ `dist < 0.125 m` |
| `STEPS_AFTER_DONE` | 500 | gravity-settle substeps after policy handoff |
| `max_episode_len` | 200 | (CLI override `--max_episode_len`) |

---

## Order of operations inside the wrapper

The wrappers ([`privileged_env.py`](../envs/privileged_env.py),
[`pixel_env.py`](../envs/pixel_env.py)) apply the knobs in this exact
sequence inside `step()`. The order matters for the `info` keys logged.

```
1. (obs, reward, done, info) = self.env.step(action)
   ↓ snapshot base_reward = reward
2. reward −= action_penalty × mean(action²)            # every step
   ↓ info['action_penalty'], info['action_cost']
3. if not done:
       reward −= vel_penalty × mean_vertex_disp_m
       ↓ info['vel_penalty'], info['cloth_mean_speed']
4. if done:
       reward −= pre_settle_coef × info['pre_settle_dist_m']
       ↓ info['pre_settle_penalty']
5. if done and success_factor is set:
       compute adaptive_is_success
       reward += success_bonus  OR  reward −= fail_penalty
       ↓ info['is_success'], info['shaping_added']
6. if done: emit rwd_diag/* keys for the diagnostics callback
```

Want to verify a single step? Sum the `info` keys after a `step()`. They
should account for every difference between `base_reward` and the final
returned `reward`.

---

## Per-knob reference

Each knob has the same structure: **default**, **what it does**,
**suggested value**, **diagnostic key** to verify it's firing.

### `success_factor` (success criterion gate, NOT a reward)

| | |
|---|---|
| Default | 1.2 |
| Replaces | dedo's strict criterion `\|last_rwd\| < 2.5` (≈ `dist < 0.125 m`) |
| With | `dist_after_settle < success_factor × hole_radius` |
| Effective threshold at sf=1.2 | ~0.5–0.7 m (since hole_radius ~0.4–0.6 m) |
| Stricter | `sf=0.8` for "actually threading" semantics |
| Disable | `--no_adaptive_success` falls back to dedo's strict criterion; `success_bonus` / `fail_penalty` go inert |
| Diagnostic | `rwd_diag/task/adaptive_thresh`, `rwd_diag/success/disagree_rate` |

### `success_bonus`

| | |
|---|---|
| Default in scripts | 200.0 |
| **Validated value** | **100.0** (used in the 0.7-success PPO run) |
| When | terminal step, IFF adaptive success fires |
| Effect on reward | `reward += success_bonus` |
| Magnitude check | per-step base sum ≈ −20 to −100 over 200 steps; `last_rwd` ≈ −2 to −100. So `+100` is comparable to the cumulative dist signal — strong but not crushing. |
| Why not 200 | Variance of the bonus-induced reward scales with `bonus²`. Dropping 200 → 100 cuts terminal-shaping variance 4× without hurting convergence. |
| Diagnostic | `rwd_diag/reward/terminal_shaping` (signed; +bonus or −penalty depending on outcome) |

### `fail_penalty`

| | |
|---|---|
| Default | 0.0 |
| When | terminal step, IFF success does NOT fire |
| Effect on reward | `reward −= fail_penalty` |
| Use case | sharpens the success/fail gap when paired with `success_bonus`. Only worth turning on if you observe near-miss plateaus. |
| Suggested if needed | 50–100 |
| Diagnostic | same `rwd_diag/reward/terminal_shaping` (negative when fail fires) |

### `vel_penalty`

| | |
|---|---|
| Default | 0.0 |
| **Validated value** | **8.0** |
| When | every NON-terminal step (skipped at terminal — gravity settle would contaminate the verts delta) |
| Effect on reward | `reward −= vel_penalty × mean_vertex_disp_m` |
| Per-step magnitude | typical `mean_vertex_disp_m` = 0.01–0.3 m/step. At `vp=8`, that's **−0.08 to −2.4 per step** (≈ **−16 to −480 per episode**). |
| Why so high | Earlier guidance suggested 1.0; that's too soft once `pre_settle_coef` pulls the cloth into the peg region — the policy responds by flapping the cloth there. 8.0 is empirically tuned. |
| Use case | suppress whippy / oscillating cloth motion |
| Diagnostic | `info['vel_penalty']`, `info['cloth_mean_speed']`; `rwd_diag/reward/vel_penalty_sum` |

### `action_penalty`

| | |
|---|---|
| Default | 0.0 |
| When | **every step including terminal** |
| Effect on reward | `reward −= action_penalty × mean(action²)` |
| Per-step magnitude | `mean(a²)` ∈ [0, 1] (dimensionless). At `ap=0.5`, that's at most **−0.5 per step**. |
| Use case | discourage bang-bang / flailing control |
| Suggested if needed | 0.1–2.0 |
| Diagnostic | `info['action_penalty']`, `info['action_cost']`; `rwd_diag/reward/action_penalty_sum` |

### `pre_settle_coef`

| | |
|---|---|
| Default | 0.0 |
| **Validated value** | **20.0** |
| When | terminal step only |
| Effect on reward | `reward −= pre_settle_coef × pre_settle_dist_m` |
| Magnitude | `pre_settle_dist_m` is **raw meters**. At coef=20, **−20 per meter of pre-settle distance**. Typical 0.5–3 m → −10 to −60. |
| Why coef=20 specifically | Post-settle reward `last_rwd` has effective coefficient `FINAL_REWARD_MULT/WBOX = 400/20 = 20/m`. Setting `pre_settle_coef=20` makes "1 m of pre-settle distance" cost the same as "1 m of post-settle distance". |
| What it solves | Without this, the policy converges to "lift cloth high above peg, let gravity drop it" — same post-settle reward as actually threading. This term penalizes the lift directly. |
| Diagnostic | `info['pre_settle_penalty']`, `info['pre_settle_dist_m']`; `rwd_diag/reward/pre_settle_penalty` |

---

## Should I worry about the terminal-vs-per-step magnitude gap?

**For PPO: no, empirically.** The validated config reaches 0.7 success
despite terminal magnitudes (~±200) dwarfing the per-step path (~±100
cumulative). PPO with VecNormalize + GAE absorbs the gap.

**For SAC: yes, always pin `--ent_coef 0.2`.** Auto-entropy reads the
big terminal as "policy is confident" and collapses to <0.01 within
~100k steps, killing exploration. Documented in `wandb_charts_ref.md`.

**Should I lower `FINAL_REWARD_MULT` from 400?** Don't preemptively.
Reasons to revisit:
1. SAC instability persists after pinning ent_coef.
2. You want >70% success, ruled out representation issues, and the
   remaining failures are precision near-misses (compare
   `rwd_diag/success/base_rate` vs `adaptive_rate`).
3. Cross-seed reproducibility for a paper.

If you do experiment, the cleanest path is a wrapper-level
`final_reward_scale` knob that multiplies `last_rwd` after dedo emits
it (and rescales the base-success threshold). Suggested
`final_reward_scale=0.05` (effective mult = 20), with `success_bonus`
dropped to ~10–20 to match. Not implemented today.

---

## Default recipes

### Privileged PPO — validated (0.7 eval success)

```bash
python experiments/hang_obs_exp/scripts/train_privileged.py \
    --obs_mode hole_centroid \
    --success_factor 1.2 \
    --success_bonus 100 \
    --pre_settle_coef 20 \
    --vel_penalty 8 \
    --use_wandb
```

### Privileged SAC — starting point (mirrors PPO + SAC pins)

```bash
python experiments/hang_obs_exp/scripts/train_privileged_sac.py \
    --obs_mode hole_centroid \
    --success_factor 1.2 \
    --success_bonus 100 \
    --pre_settle_coef 20 \
    --vel_penalty 8 \
    --ent_coef 0.2 \
    --log_std_init -2.0 \
    --use_wandb
```

### Pixel PPO/SAC

Same flags; swap the script. Pixel runs may want slightly higher
`action_penalty` (0.5) — CNN policies tend toward more erratic actions
early.

### Diagnostic / unshaped baseline

```bash
python experiments/hang_obs_exp/scripts/train_privileged.py \
    --obs_mode hole_centroid \
    --no_adaptive_success \
    --success_bonus 0 \
    --use_wandb
```

dedo base reward only. Use to measure shaping contribution.

---

## Known failure modes

### 1. "Lift the cloth high and drop it" — SOLVED by `pre_settle_coef`

| | |
|---|---|
| Symptom | `eval/success_rate` looks fine, but the eval video shows ballistic drops. `last_episode/dist` and `pre_settle_dist_m` are both >1 m. |
| Cause | Without `pre_settle_coef`, dedo rewards a "drop onto peg" the same as "thread by control". |
| Fix | `--pre_settle_coef 20`. Raise to 30–40 if persists. |

### 2. Workspace-exit early termination

| | |
|---|---|
| Symptom | `last_episode/episode_length` < 200 across many episodes; `rollout/success_rate` spikes briefly then crashes. |
| Cause | dedo sets `done=True` on workspace-bound exit; without an extra deterrent, skipping ~150 dithering steps can outweigh the early-term reward multiplier. |
| Fix | `pre_settle_coef > 0` (penalizes the high pre-settle dist these exits produce). If still bad, raise to 40+, or restore a custom boundary penalty. |

### 3. Whippy / bang-bang trajectories

| | |
|---|---|
| Symptom | eval video shows juddering gripper / oscillating cloth. `info['cloth_mean_speed']` > 0.3 sustained. |
| Cause | no smoothness prior in default policies. |
| Fix | `--vel_penalty 8 --action_penalty 0.5` |

### 4. SAC `ent_coef` collapse

| | |
|---|---|
| Symptom | `train/ent_coef` drops below 0.01 in <100k steps; success flatlines. |
| Cause | terminal magnitudes (~±300) drive Q values up; auto-entropy interprets as "confident" and over-commits. |
| Fix | `--ent_coef 0.2` (fixed). PPO unaffected. |

### 5. Adaptive vs base success drift

| | |
|---|---|
| Symptom | `rwd_diag/success/disagree_rate` > 30% — adaptive fires often but base barely. |
| Cause | `sf=1.2 × hole_radius ~ 0.6 m` is much looser than dedo's `0.125 m`. Volume gap ~24×. |
| Status | Feature, not bug. Adaptive gives PPO denser signal. Re-eval with `--no_adaptive_success` for paper numbers. |

---

## Tuning guide

| Phenomenon | Knob | Direction |
|---|---|---|
| Lift-and-drop pattern | `pre_settle_coef` | up (start 20) |
| Bang-bang / flailing actions | `action_penalty` | up (start 0.5) |
| Whippy cloth | `vel_penalty` | up (start 8 with `pre_settle_coef`) |
| Stuck at near-miss plateau | `fail_penalty` | up (start 50) |
| Sparse signal, slow learning | `success_bonus` | up (try 200) |
| SAC entropy collapse | `--ent_coef` | pin at 0.2 |
| Workspace-exit spikes | `pre_settle_coef` | up (40+) |
| PPO not converging at all | `success_bonus` | up |
| SAC unstable | `success_bonus` | down toward 50; pin ent_coef |

---

## Diagnostics checklist

All `rwd_diag/*` keys are 100-ep moving averages drained by
`RewardDiagnosticsCallback`. Per-episode raw values are under
`last_episode/*`.

| Key | What it tells you |
|---|---|
| `rwd_diag/reward/episode_total` | True per-episode return after all shaping. |
| `rwd_diag/reward/base_sum` | Sum of `r_dedo_base` over the episode. Magnitude ≈ avg dist × 200 / 20. |
| `rwd_diag/reward/vel_penalty_sum` | Magnitude of velocity penalty applied. ~0 when knob is 0. |
| `rwd_diag/reward/action_penalty_sum` | Same for action penalty. |
| `rwd_diag/reward/pre_settle_penalty` | Single-step pre-settle penalty at terminal. |
| `rwd_diag/reward/terminal_base` | Base reward at terminal step (= dedo's early-term-multiplied per-step + `last_rwd`). |
| `rwd_diag/reward/terminal_shaping` | Signed: +`success_bonus` or −`fail_penalty` actually applied. |
| `rwd_diag/success/active_rate` | Criterion the agent trained against. |
| `rwd_diag/success/base_rate` | Strict dedo criterion (always ≤ adaptive_rate). |
| `rwd_diag/success/disagree_rate` | Fraction of episodes where adaptive ≠ base. |
| `rwd_diag/task/adaptive_dist` | **Leading indicator.** Mean final hole-to-goal distance. |
| `rwd_diag/task/adaptive_thresh` | Reference line for adaptive_dist. |
| `rwd_diag/task/hole_radius` | Should be roughly constant per cloth distribution. |

**Sanity check**: at convergence,
`episode_total ≈ base_sum + terminal_base + terminal_shaping
− vel_penalty_sum − action_penalty_sum − pre_settle_penalty`.
If those don't sum to `episode_total`, a knob is double-counting or
silently dropped.

---

## What's gone (for the record)

Pre-2026-05-05, three more knobs existed. They've been removed.

| Removed knob | What it did | Replaced by |
|---|---|---|
| `boundary_penalty` | Fixed terminal penalty on workspace-bound exit | `pre_settle_coef` + dedo's built-in early-term multiplier |
| `z_overshoot_penalty` | Per-step penalty on hole-z above peg-z (orientation-blind base reward fix) | `pre_settle_coef` (penalizes pre-settle dist on every axis) |
| `z_overshoot_slack` | Free zone above peg-z | (gone with z_overshoot_penalty) |

Diagnostic keys also gone:
`rwd_diag/boundary/violation_rate`,
`rwd_diag/task/z_overshoot_mean`,
`rwd_diag/reward/boundary_penalty`,
`rwd_diag/reward/z_overshoot_penalty_sum`,
`info['boundary_violation']`,
`info['z_overshoot']`,
`info['z_overshoot_penalty']`.
