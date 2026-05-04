Here's a complete chart guide based on what's actually being logged.

---

## The 6 charts that matter most

If you're glancing at wandb every hour, these are the ones to check first, in this order:

| # | Chart | Plain-English meaning |
|---|---|---|
| 1 | `eval/success_rate` | The headline. Deterministic 10-episode eval success rate. **This is the number that goes in your paper.** |
| 2 | `rollout/success_rate_100` | Live success rate from training rollouts (100-ep moving avg). Noisier than eval but updates every episode. |
| 3 | `rwd_diag/task/adaptive_dist` | Mean hole-centroid → goal distance at episode end (in meters). **Leading indicator** — drops well before success rate climbs. |
| 4 | `train/critic_loss` | SAC critic stability. Should stabilize to a finite value. **Explosion = kill the run.** |
| 5 | `train/ent_coef` | SAC's auto-tuned exploration noise. Should converge to ~0.05–0.5. **Drops to 0 too fast = premature commitment.** |
| 6 | `eval/video` (panel, not chart) | The mp4 grid. Watch one every couple of hours; tells you *why* the metrics look the way they do. |

Pin these 6 to the top of the wandb panel using ⋯ → "Move to top" so they're above the fold every time you open the run page.

---

## Full chart reference

### `rollout/*` — training-time data (every episode)

| Chart | What it is | Expected behavior |
|---|---|---|
| `rollout/ep_rew_mean` | Mean episode return over last 100 training episodes (SB3 default) | Climbs from very-negative (~−300 with the apron cloth) toward less-negative or positive. Plateau is fine. |
| `rollout/ep_len_mean` | Mean episode length | Should stay near `max_episode_len=200`. Drops mean early-termination from `out-of-workspace` clipping — bad sign. |
| `rollout/success_rate_100` | 100-ep moving avg of `info['is_success']` | Starts at 0–5%, climbs gradually. **Noisy at small ep counts** (first ~100 eps it's basically meaningless). |
| `rollout/episodes` | Total episodes seen | Linear in step — useful only for sanity-checking that envs are stepping at all. |

### `eval/*` — periodic deterministic eval (10 episodes per save)

| Chart | What it is | Expected behavior |
|---|---|---|
| `eval/success_rate` | Success rate over 10 deterministic eval episodes | **Cleaner than `rollout/success_rate_100`** because deterministic. Updates every 2nd checkpoint (every 20k steps with `--log_save_interval 20`). Only ~10 episodes per data point so individual values jitter ±10–20 %; trend line is what matters. |
| `eval/video` | mp4 panel | One mp4 every 4th checkpoint (every 40k steps with `--log_save_interval 20`). Each video shows 10 deterministic episodes back-to-back, with SUCCESS/FAIL badges drawn per episode and the post-policy "settle" frames spliced in chronologically. |

### `rwd_diag/reward/*` — reward decomposition (per-episode 100-ep moving avg)

These are emitted by `PrivilegedObsWrapper._emit_episode_diagnostics` and rolling-averaged by `RewardDiagnosticsCallback`. They tell you what shaping terms are actually firing.

| Chart | What it is |
|---|---|
| `rwd_diag/reward/episode_total` | True total reward per episode (after vel_penalty + terminal shaping). The "real" objective. |
| `rwd_diag/reward/base_sum` | Sum of dedo's per-step distance reward over the whole episode. Always negative. |
| `rwd_diag/reward/vel_penalty_sum` | Sum of velocity penalty over the episode (≥ 0). 0 when `vel_penalty=0`. |
| `rwd_diag/reward/terminal_base` | The single-step base reward at the terminal step (driven by `FINAL_REWARD_MULT=400`, dominates `base_sum` magnitude). |
| `rwd_diag/reward/terminal_shaping` | Terminal `+success_bonus` or `-fail_penalty` actually applied. Tells you the success-bonus is firing — should be increasingly nonzero as success rate climbs. |
| `rwd_diag/reward/episode_length` | Per-episode step count. Mostly 200; if it drops, episodes are terminating early (out-of-workspace). |

### `rwd_diag/success/*` — what counts as success (per-episode 100-ep avg)

| Chart | What it is |
|---|---|
| `rwd_diag/success/active_rate` | The success criterion the agent **actually trained against**. Equal to `adaptive_rate` since you set `success_factor=1.2`. **This is the same series as `rollout/success_rate_100`**; they're redundant on purpose for cross-checking. |
| `rwd_diag/success/adaptive_rate` | Your adaptive criterion: `dist < 1.2 × hole_radius`. |
| `rwd_diag/success/base_rate` | Dedo's strict criterion: `\|final_reward\| < 2.5` (≈ `dist < 0.125 m`). Will be lower than `adaptive_rate` because it's stricter. Useful to know how many "almost successes" your runs have under the dedo definition. |
| `rwd_diag/success/disagree_rate` | Fraction of episodes where the two criteria disagree. Tells you how soft your adaptive criterion is vs dedo's strict one. |

### `rwd_diag/task/*` — geometric state (per-episode 100-ep avg)

| Chart | What it is | Expected behavior |
|---|---|---|
| `rwd_diag/task/adaptive_dist` | Mean hole-centroid → goal distance at episode end (m) | **Most important leading indicator.** Should monotonically decrease as policy improves. Stalls before success rate climbs. |
| `rwd_diag/task/adaptive_thresh` | The adaptive threshold value (= `success_factor × hole_radius`) | Roughly constant per cloth distribution (~0.5–0.7 m). Mostly useful as a horizontal reference line vs `adaptive_dist`. |
| `rwd_diag/task/hole_radius` | Mean hole radius across recent episodes (m) | Should be ~constant; sudden changes mean the cloth distribution shifted. |

### `train/*` — SAC internals (every gradient update)

| Chart | What it is | Red flag |
|---|---|---|
| `train/actor_loss` | Actor (policy) loss = `−Q(s, a) − α H[π]` | Should trend negative-and-stable. Wild oscillations mean LR is too high. |
| `train/critic_loss` | Critic MSE on Bellman targets | **Hard cap your eyes here.** Should stabilize <100 typically. Above 1000 and growing → run is collapsing, kill it. |
| `train/ent_coef` | Auto-tuned entropy temperature α | Starts at 1.0, drifts to ~0.05–0.5 for this task. **Decays to ~0 in <100k steps = bad** (premature exploitation). |
| `train/ent_coef_loss` | Loss for the α optimizer | Hovers near 0; magnitude tells you how aggressively α is being adjusted. Mostly diagnostic. |
| `train/learning_rate` | Constant 3e-4 unless you scheduled it | — |
| `train/n_updates` | Total gradient updates so far | Linear in env steps × `gradient_steps`. Sanity check only. |

### `time/*` — throughput / wall-clock

| Chart | What it is | Expected behavior |
|---|---|---|
| `time/total_timesteps` | Env steps elapsed | Equal to wandb step. Linear in wall-clock if not throttled. |
| `time/fps` | Env steps per wall-clock second | **Watch this for thermal throttling.** On the M4 you should see ~10 sps starting, may degrade ~20–30% over time. Sustained drops below 5 sps = something's wrong (FileProvider, swap, etc.). |
| `time/episodes` | Total episodes (= ~steps/200) | — |

### `bc/*` — BC pretrain (only if `--bc_episodes > 0`)

These have **`bc/epoch` as their natural x-axis**, not step. They live in step 0 (since BC runs before `agent.learn()`).

| Chart | What it is |
|---|---|
| `bc/mse` | MSE between actor mean action and demo action, per epoch | Should monotonically decline from ~0.5–1.0 at epoch 1 to ~0.01–0.05 at epoch 30. Plateau >0.1 = demos are inconsistent. |
| `bc/epoch` | Epoch counter (1..bc_epochs) | Use as x-axis for `bc/mse`. |
| `bc/n_pairs` | Total (obs, act) pairs in BC dataset | Single value, ~10000 for 50 demos. |
| `bc/n_demos` | Number of demos collected | — |
| `bc/n_success_demos` | Successful demos in dataset | If this is 0 with `--bc_demos_only_success` not set, your BC dataset is purely failed demos (the Anti-BC). |

### `final_eval/*` — once at end of training (20-episode deterministic eval)

Logged after `agent.learn()` finishes. Single data point per run.

| Chart | What it is |
|---|---|
| `final_eval/success_rate` | The **paper number**: 20-episode deterministic success rate. |
| `final_eval/mean_reward` | Mean episode return over those 20 eps |
| `final_eval/std_reward` | Stddev — if huge, the policy is bimodal (sometimes solves it, sometimes whiffs catastrophically). |
| `final_eval/n_episodes` | Always = `--n_final_eval_episodes` (default 20). |
| `final_eval/task/adaptive_dist` | Mean final distance over the 20 eval eps. |
| `final_eval/success/{base,adaptive,disagree,active}_rate` | Same fields as `rwd_diag/success/*`, but specifically over the 20 final-eval episodes. |
| `final_eval/reward/*` | Same as `rwd_diag/reward/*` but over final-eval. |

### `args` — config dump (no chart, sidebar only)

The script also calls `wandb.config.update(...)` with the flattened run config. Look in the **Config tab** (left sidebar of the run page) — you'll see entries like:

- `extra.success_factor: 1.2`
- `extra.success_bonus: 200.0`
- `reward_def.uses_adaptive_success: true`
- `dedo.max_episode_len: 200`
- `system.platform: macOS-14...`

This is the single source of truth for "what reward / hyperparams did this run optimize." If you ever look at a chart and wonder "wait, what was `success_factor` here?", that's the place.

---

## X-axis recommendations

Top right of any chart → "Edit panel" → "Edit X-axis" (or change globally via the workspace settings ⚙️):

| For these charts | Use x-axis | Why |
|---|---|---|
| Everything **except** `bc/*` | `Step` (default — = `global_step` = env steps) | Cross-run comparable; same scale across PPO and SAC. |
| `bc/mse`, `bc/epoch_loss` | `bc/epoch` | The natural x-axis for BC; otherwise all 30 points stack at step 0. |
| Performance debugging (M4 throttling, FileProvider stalls) | `_runtime` (wall-clock seconds) | If `time/fps` looks fine vs Step but the run takes forever in `_runtime`, something is making each step expensive in wall-clock. |
| Cross-run baselining (multiple seeds) | `Step` with a 100-pt smoothing | Helps you eyeball "is this seed the lucky one or are all seeds rising?" |

Set the **smoothing slider** to ~0.7 globally for the noisy episode-level charts (`rollout/*`, `rwd_diag/*`). For the per-update `train/*` charts, lower smoothing is fine since they're already aggregated by SB3. SAC logs every gradient update so they're dense.

---

## What to watch for at each phase

### 0–5k steps (learning_starts phase)

The actor is uniform random `[-1, 1]^6`. **Don't read into anything.** Reward is very negative, success near 0, train losses are 0 (no updates yet).

What to verify:
- `time/fps` is ~10 — you're actually stepping the sim
- BC banner printed nicely if BC was on (only relevant when you re-enable BC)

### 5k–100k steps (early SAC)

This is where the run is most likely to silently fail.

| Symptom | Likely cause | Action |
|---|---|---|
| `train/critic_loss` exploding past 1000 | Exploration too wild + high LR | Kill, retry with `--lr 1e-4` |
| `train/ent_coef` hits 0.01 by step 20k | Reward magnitudes too big, actor over-confident | Kill, retry with `--ent_coef 0.2` (fixed, no auto) |
| `rollout/ep_len_mean` drops below 150 | Out-of-workspace clipping → many early terminations | Symptom of cloth flying away. Kill, retry with `--vel_penalty 1.0` |
| `time/fps` halves over 20k steps | Thermal throttle or FileProvider | Move logdir off Drive (you did), check thermal |

What you want to see:
- `rwd_diag/task/adaptive_dist` starting to drop (this is the very first sign of learning, before success_rate moves)
- `train/critic_loss` rising then leveling off (typical: ramps to 50–200 by step 50k, then flat)
- `train/ent_coef` slowly decaying (0.5 → 0.2 over 50k steps)

### 100k–500k steps (the make-or-break window)

You should be seeing the success rate climb in this window. Specifically:

- By step **200k**: `eval/success_rate` should be at least 5–10%, `rwd_diag/task/adaptive_dist` clearly below `adaptive_thresh`
- By step **500k**: `eval/success_rate` ≥ 30% if the run is going to converge

If at 300k your `adaptive_dist` is flat AND `eval/success_rate` is stuck at <5%, the run is stuck in a local minimum and won't recover — kill it.

### 500k–1.5M steps (convergence)

`eval/success_rate` should be plateauing. The interesting question becomes: how stable is the plateau?

- **Stable plateau** (low variance between consecutive evals): policy is converged, you've got your number. The `final_eval/*` block at the end will be a clean 20-episode reading.
- **Oscillating plateau** (jumping ±20% between consecutive evals): SAC critic is unstable. The policy is OK on average but sensitive to which mini-batch it just saw. Lowering `--lr` to 1e-4 for the next run helps.
- **Catastrophic forgetting** (rate climbs then crashes back to 0): rare but happens with auto-entropy. Stop at the best checkpoint; future runs use `--ent_coef 0.1` fixed.

---

## A workspace setup that makes this all easier

In the wandb run page, click the ⚙️ next to the workspace name and create three sections:

1. **Headline** — drag in `eval/success_rate`, `rollout/success_rate_100`, `rwd_diag/task/adaptive_dist`, `eval/video`. This is the only section you need to look at most of the time.
2. **Health** — drag in `train/critic_loss`, `train/ent_coef`, `train/actor_loss`, `time/fps`. Glance once an hour.
3. **Decomposition** — drag in `rwd_diag/reward/*` charts. Look at this only when something looks weird in **Headline**.

The default wandb panel layout dumps everything alphabetically and is genuinely hard to read on small screens. The 3-section setup pays for itself within the first run.