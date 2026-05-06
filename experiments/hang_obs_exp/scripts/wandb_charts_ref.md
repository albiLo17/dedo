Here's a complete chart guide based on what's actually being logged.

---

## The 7 charts that matter most

If you're glancing at wandb every hour, these are the ones to check first, in this order:

| # | Chart | Plain-English meaning |
|---|---|---|
| 1 | `eval/success_rate` | The headline. Deterministic 10-episode eval success rate. **This is the number that goes in your paper.** |
| 2 | `train/cumulative_successes` | **Monotone tally of all successful episodes since run start.** Each step UP = one specific episode succeeded. Reads exactly like "how many times has the agent solved it so far?" — no rolling-average smearing. Use this whenever you want "did *that* episode succeed?". |
| 3 | `last_episode/success` | Raw 0/1 spike per episode (1 = success, 0 = fail). Lets you visually count individual successes and see the gaps between them. Sparse and spiky on purpose. |
| 4 | `rwd_diag/task/adaptive_dist` *or* `last_episode/dist` | Mean hole-centroid → goal distance at episode end (m). **Leading indicator** — drops well before success rate climbs. Use the `rwd_diag/...` rolling version for trend; use `last_episode/dist` for per-episode "near-misses". |
| 5 | `train/critic_loss` | SAC critic stability. Should stabilize to a finite value. **Explosion = kill the run.** |
| 6 | `train/ent_coef` | SAC's auto-tuned exploration noise. Should converge to ~0.05–0.5. **Drops below 0.01 = premature commitment, run is collapsing.** Pin entropy with `--ent_coef 0.2` if it keeps happening (see "Known failure modes" below). |
| 7 | `eval/video` (panel, not chart) | The mp4 grid. Watch one every couple of hours; tells you *why* the metrics look the way they do. |

Pin these 7 to the top of the wandb panel using ⋯ → "Move to top" so they're above the fold every time you open the run page.

**Recommended X-axis for charts 2–4**: `train/episodes_total` (set globally via workspace settings ⚙️ → X-axis). For SAC-internals (charts 5–6), keep `Step`. See the X-axis section below for why `time/episodes` from SB3 is *not* available.

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

### `last_episode/*` — raw per-episode values (no averaging)

Logged by `RewardDiagnosticsCallback._log_per_episode_to_wandb` directly via `wandb.log` (bypasses the SB3 logger to avoid record-overwrite-before-dump losses). One data point per episode end. Use these whenever you want individual-episode resolution that the rolling-mean charts can't give you.

| Chart | What it is |
|---|---|
| `last_episode/success` | 0 or 1, the *active* success criterion for the just-finished episode. Sparse spike train. |
| `last_episode/success_adaptive` | 0/1 of the wrapper's adaptive criterion (`dist < success_factor × hole_radius`). |
| `last_episode/success_base` | 0/1 of dedo's strict `\|final_reward\| < 2.5` criterion. Will be ≤ adaptive. |
| `last_episode/dist` | Final hole-centroid → goal distance in meters for the just-ended episode. Pair with `task/adaptive_thresh` reference line. |
| `last_episode/episode_total` | Total reward (incl. shaping) for the just-ended episode. Quick way to see "did terminal_shaping fire". |
| `last_episode/episode_length` | Step count of the just-ended episode. Drops below 200 = early-termination via out-of-workspace clip. |

### `train/*` (custom) — episode counters for x-axis

Logged from the same callback path as `last_episode/*`. These exist *specifically* to give you a useful x-axis (SB3's `time/episodes` is `exclude="tensorboard"` so it never reaches wandb — see X-axis section).

| Chart | What it is |
|---|---|
| `train/episodes_total` | Cumulative episode counter. Use as the global wandb X-axis for everything except `bc/*`. |
| `train/cumulative_successes` | Cumulative count of successful episodes. **Each step up = one specific episode succeeded** — the cleanest way to count successes in a run. |

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
| **Default for everything except `bc/*` and `train/*` SAC-internals** | `train/episodes_total` | Reads as "completed deployments". Step 8000 = episode ~50 vs. "8k of 1.5M" reads cleaner as "50 of ~7500". Logged per-episode by `RewardDiagnosticsCallback`. |
| `bc/mse`, `bc/epoch_loss` | `bc/epoch` | The natural x-axis for BC; otherwise all 30 points stack at step 0. |
| `train/critic_loss`, `train/actor_loss`, `train/ent_coef` | `train/n_updates` | Honest x-axis for SAC internals — nothing happens before `learning_starts`, and `n_updates` only ticks during gradient updates. |
| Performance debugging (M4 throttling, FileProvider stalls) | `_runtime` (wall-clock seconds) | If `time/fps` looks fine vs Step but the run takes forever in `_runtime`, something is making each step expensive in wall-clock. |
| Cross-algo baselining (PPO vs SAC) | `Step` (= env steps) | Cross-algo comparable since both use the same env-step axis. |

### Why we don't use `time/episodes`

SB3 1.2.0's SAC internally logs `time/episodes` and `time/total timesteps` with `exclude="tensorboard"`, which means they never reach the TB events file and (because wandb syncs from TB) they never reach wandb either. So those metrics simply don't exist on the wandb side, even though they show up nicely formatted in your stdout dump tables. Our `train/episodes_total` is the workaround — same content, logged **on every env step** via `RewardDiagnosticsCallback` → TensorBoard → wandb sync so it appears on the same global-step rows as `train/critic_loss` etc. Without that per-step mirror, wandb could only see `train/episodes_total` on episode-completion steps if we used `wandb.log` alone, and choosing it as the chart X-axis would show **no data** for dense metrics (nothing to join). If you still see "no data", refresh the run page after ~1 episode (~200 steps post-reset) and confirm `train/episodes_total` appears under the run's **Scalars** / metric search.

Set the **smoothing slider** to ~0.7 globally for the noisy episode-level charts (`rollout/*`, `rwd_diag/*`). For the per-update `train/*` charts, lower smoothing is fine since they're already aggregated by SB3. SAC logs every gradient update so they're dense. **Set smoothing to 0** on `last_episode/*` and `train/cumulative_successes` — those are exactly the charts where you want to see individual episode events sharply.

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
| First success at step 1–2k (BC), then `last_episode/success` flatlines at 0 for thousands of episodes after `ent_coef` falls below 0.01 | The "ent_coef collapse" — auto-entropy commits to whatever the BC seed pointed at and stops exploring. Confirmed seen in privileged hole_centroid SAC at ~step 240k. | Kill. Re-run with `--ent_coef 0.2 --log_std_init -2`. Pinning entropy prevents the collapse; lower log_std init makes BC immediately useful. |

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

1. **Headline** — drag in `eval/success_rate`, `train/cumulative_successes`, `last_episode/success`, `rwd_diag/task/adaptive_dist`, `eval/video`. This is the only section you need to look at most of the time. Set workspace X-axis to `train/episodes_total`. Set smoothing to 0 for `last_episode/*` and `train/cumulative_successes`; ~0.7 for the rest.
2. **Health** — drag in `train/critic_loss`, `train/ent_coef`, `train/actor_loss`, `time/fps`. Glance once an hour. **`train/ent_coef` is the canary** — if it drops below 0.01 the run is collapsing, kill it.
3. **Decomposition** — drag in `rwd_diag/reward/*` charts plus `last_episode/dist`, `last_episode/episode_length`. Look at this only when something looks weird in **Headline**.

The default wandb panel layout dumps everything alphabetically and is genuinely hard to read on small screens. The 3-section setup pays for itself within the first run.

---

## Resuming a SAC run (`--load_checkpoint`)

`train_privileged_sac.py` supports proper resume from any saved checkpoint. The `_video_callback.py` checkpoint cadence (every `log_save_interval × 500` env steps) writes everything needed:

- `agent.zip` — policy + value + log_std + Adam state + `num_timesteps`
- `replay_buffer.pkl` — SAC's off-policy buffer (critical; without it SAC's first updates after resume run on an empty buffer and the policy drifts fast)
- `vec_normalize.pkl` — obs/reward running stats (without it the loaded policy sees obs at a slightly different scale and quietly degrades)
- `args.pkl` — original dedo args

To resume:

```bash
caffeinate -dimsu python experiments/hang_obs_exp/scripts/train_privileged_sac.py \
    --logdir_root "$LOGDIR_ROOT" \
    --obs_mode hole_centroid \
    --total_env_steps 500000 \
    --load_checkpoint hole_centroid_sac/SAC_<orig_timestamp>_HangProcCloth-v1 \
    --use_wandb
```

What happens:

1. The script reads `agent.zip` / `replay_buffer.pkl` / `vec_normalize.pkl` from the checkpoint dir.
2. **BC pretrain is skipped** (the saved policy already encodes BC + however many steps of RL). Re-applying BC would clobber RL progress.
3. **`--log_std_init` is skipped** (whatever value it had during the original run is in the saved weights; SAC has been moving it via gradient steps since then).
4. `--total_env_steps` is interpreted as the *target total*, not the remainder. To run from step 100k → step 500k, pass `--total_env_steps 500000` again.
5. CLI hyperparams that shadow saved ones (`--lr`, `--ent_coef`) **override** the saved values via SB3's `custom_objects`. This lets you change LR or pin entropy mid-run if you need to.
6. A NEW wandb run is created (see "wandb resume semantics" below for why) with tags `resumed_from=<orig_name>` and `resume_step=<num_timesteps_at_load>`.

### wandb resume semantics: always a NEW run

I deliberately *don't* use `wandb.init(resume="must", id=...)`. Three reasons it's brittle for SAC checkpoint resume:

1. **Step monotonicity.** Wandb requires logged `step` to be ≥ the last logged step in the run. SB3's checkpoint cadence means resume rolls back to `step ≈ last_save - log_save_interval × 500`, which is *less* than the original wandb run's last step. Wandb either drops the new logs or interleaves duplicates with the original, both confusing.
2. **Sync-tensorboard makes it worse.** Our wandb integration goes through TB sync. New TB events at "lower" step numbers than the original can corrupt the chart in ways that depend on wandb client version.
3. **The compare-runs UI handles linked runs natively.** Group-by-tag (`resumed_from=...`) in the wandb workspace gives you a clean overlay of original + resume on the same chart, with distinct colors per resume. That's *more* readable than one mashed-together run, especially across multiple bike-commute resume cycles.

Operationally: in your wandb run page, group by tag `resumed_from`, and pick `train/episodes_total` as the X-axis. The chart will look like one continuous run.

### Bike-commute workflow with tmux

```bash
# Before bike: gracefully stop, let it save final ckpt
tmux send-keys -t sac C-c
sleep 30                    # let it save replay_buffer.pkl etc.
tmux kill-session -t sac

# After bike: resume into a NEW tmux session
ORIG=hole_centroid_sac/SAC_<orig_timestamp>_HangProcCloth-v1
tmux new -s sac
caffeinate -dimsu python experiments/hang_obs_exp/scripts/train_privileged_sac.py \
    --logdir_root "$LOGDIR_ROOT" \
    --obs_mode hole_centroid \
    --total_env_steps 500000 \
    --load_checkpoint "$ORIG" \
    --use_wandb
# Ctrl-b d to detach
```

You'll lose at most `log_save_interval × 500 = 10000` env steps of training (the un-checkpointed tail before kill), which is ~2 minutes of compute on the M4. Negligible.

---

## SAC + BC interaction notes (stuff that bit us)

Three subtle behaviors that cost real debugging time, recorded here so they don't bite again:

1. **SB3 SAC's `policy_kwargs={'log_std_init': ...}` is silently ignored** when `use_sde=False` (the default). The non-SDE code path builds `log_std = nn.Linear(...)` with vanilla torch init, so the actual initial output std is ~1.0 in pre-tanh space. BC trains `mu` to ~0.15 magnitude on demo actions, so the deployed sampled action `tanh(mu + N(0, ~1)) ≈ tanh-noise` overwhelms BC's signal at deploy. **Fix in `train_privileged_sac.py`**: `--log_std_init -2.0` patches the actor's `log_std.bias` to a constant after construction, giving std ≈ 0.135 ≈ demo |action|. BC bias becomes immediately visible.

2. **Auto-entropy collapses under large terminal rewards.** With `success_bonus=200` and `FINAL_REWARD_MULT=400` your terminal reward is ±300+, while non-terminal rewards live in ±0.5 — a 1000× scale gap. SAC's auto-tuned `ent_coef` interprets this as "already very confident, drive entropy down" and decays to <0.01 within 100–200k steps, killing exploration. PPO doesn't have this knob and is unaffected. **Fix**: `--ent_coef 0.2` (fixed, no auto) in any run with `success_bonus > 50`.

3. **`ep_len_mean` is *not* always `max_episode_len`.** Out-of-workspace early termination clips episodes (often to 60–80 steps) when the policy fails badly. This shifts the meaning of "100-ep moving average" charts (e.g. `rollout/success_rate_100`) — at step 8k with `ep_len_mean ≈ 80`, the 100-ep window has saturated; with `ep_len_mean ≈ 200`, only 40 episodes have happened. Translation between Step and "episodes seen" requires knowing the live mean. The `train/episodes_total` x-axis sidesteps this entirely.