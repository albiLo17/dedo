Here's a complete chart guide based on what's actually being logged.

---

## The 7 charts that matter most

If you're glancing at wandb every hour, these are the ones to check first, in this order:


| #   | Chart                                          | Plain-English meaning                                                                                                                                                                                                                                                                          |
| --- | ---------------------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| 1   | `eval/success_rate` (+ `eval/success_rate_se`) | The headline. Deterministic 10-episode eval success rate; the `_se` companion is the binomial standard error (`sqrt(p(1−p)/n)`), so two consecutive evals differing by less than ~`2 × se `are statistically indistinguishable. **`eval/success_rate` is the number that goes in your paper.** |
| 2   | `train/cumulative_successes`                   | **Monotone tally of all successful episodes since run start.** Each step UP = one specific episode succeeded. Reads exactly like "how many times has the agent solved it so far?" — no rolling-average smearing. Use this whenever you want "did *that* episode succeed?".                     |
| 3   | `last_episode/success`                         | Raw 0/1 spike per episode (1 = success, 0 = fail). Lets you visually count individual successes and see the gaps between them. Sparse and spiky on purpose.                                                                                                                                    |
| 4   | `task/adaptive_dist` *or* `last_episode/dist`  | Mean hole-centroid → goal distance at episode end (m). **Leading indicator** — drops well before success rate climbs. Use the rolling `task/...` version for trend; use `last_episode/dist` for per-episode "near-misses".                                                                     |
| 5   | `train/critic_loss`                            | SAC critic stability. Should stabilize to a finite value. **Explosion = kill the run.**                                                                                                                                                                                                        |
| 6   | `train/ent_coef`                               | SAC's auto-tuned exploration noise. Should converge to ~0.05–0.5. **Drops below 0.01 = premature commitment, run is collapsing.** Pin entropy with `--ent_coef 0.2` if it keeps happening (see "Known failure modes" below).                                                                   |
| 7   | `eval/video` (panel, not chart)                | The mp4 grid. Watch one every couple of hours; tells you *why* the metrics look the way they do.                                                                                                                                                                                               |


Pin these 7 to the top of the wandb panel using ⋯ → "Move to top" so they're above the fold every time you open the run page.

**Recommended X-axis for charts 2–4**: `train/episodes_total` (set globally via workspace settings ⚙️ → X-axis). For SAC-internals (charts 5–6), keep `Step`. See the X-axis section below for why `time/episodes` from SB3 is *not* available.

> **Naming note (post 2026-05-05):** the `RewardDiagnosticsCallback` strips the `rwd_diag/` prefix before logging to TensorBoard, so the actual wandb keys are `reward/*`, `success/*`, `task/*` (not `rwd_diag/reward/*`). Old runs and old screenshots may show the longer prefix; the new tables below use the actual key names.

---

## Full chart reference

### `rollout/*` — training-time data (every episode)


| Chart                      | What it is                                                        | Expected behavior                                                                                                 |
| -------------------------- | ----------------------------------------------------------------- | ----------------------------------------------------------------------------------------------------------------- |
| `rollout/ep_rew_mean`      | Mean episode return over last 100 training episodes (SB3 default) | Climbs from very-negative (~−300 with the apron cloth) toward less-negative or positive. Plateau is fine.         |
| `rollout/ep_len_mean`      | Mean episode length                                               | Should stay near `max_episode_len=200`. Drops mean early-termination from `out-of-workspace` clipping — bad sign. |
| `rollout/success_rate_100` | 100-ep moving avg of `info['is_success']`                         | Starts at 0–5%, climbs gradually. **Noisy at small ep counts** (first ~100 eps it's basically meaningless).       |
| `rollout/episodes`         | Total episodes seen                                               | Linear in step — useful only for sanity-checking that envs are stepping at all.                                   |


### `eval/*` — periodic deterministic eval (10 episodes per save)

Emitted by `HangVideoCallback._on_step` in [_video_callback.py](_video_callback.py) when an eval pass fires. The eval pass calls `evaluate_policy(..., deterministic=True, return_episode_rewards=True)` so reward sampling noise is removed. With `--eval_seed_lock` on, the same N procedural cloths are re-evaluated every checkpoint, which makes the curve readable instead of noisy. **Cadence:** every 2nd checkpoint = every `2 × log_save_interval × 500` env steps (default `log_save_interval=20` → every 20k steps).


| Chart                      | What it is                                                                                                                                                                                                                                                                  | Expected behavior                                                                                                                                                                                                                                                                                                           |
| -------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `eval/success_rate`        | Binomial proportion of successes over the eval pass: `successes / n_eval_episodes`. Same definition as `is_success` in the wrapper (= `success/active_rate`).                                                                                                               | **Cleaner than `rollout/success_rate_100`** because deterministic. Only ~10 episodes per data point so individual values jitter ±10–20 %; trend line is what matters. Sample with `eval/success_rate_se` to read confidence.                                                                                                |
| `eval/success_rate_se`     | Standard error of the binomial proportion: `sqrt(p × (1−p) / n)`. With `n=10` and `p=0.5` this is **~0.16**, with `p=0.9` it's **~0.09**, with `p=1.0` it's `0.0`.                                                                                                          | Use as ±1σ band around `eval/success_rate` to judge whether a jump is real. Two consecutive evals differing by less than `2 × se` are statistically indistinguishable at `n_eval_episodes=10`. Increase `--n_eval_episodes` if the band is too wide to resolve the trend you care about.                                    |
| `eval/n_episodes`          | Number of episodes in the eval pass. Equal to `--n_eval_episodes` (default 10).                                                                                                                                                                                             | Constant. Sanity-check only; if it drifts something is wrong.                                                                                                                                                                                                                                                               |
| `eval/episode_reward_mean` | Mean total episode reward (after all shaping) across the eval pass. Same units as `reward/episode_total` but deterministic and only over the eval set.                                                                                                                      | Tracks `eval/success_rate` loosely — depends on shaping magnitudes. With `success_bonus=200` and `FINAL_REWARD_MULT=400`, a successful episode is worth roughly +200 to +500, a failure ~−300 to −800. So a 50% success rate gives mean ≈ −100, 100% ≈ +300.                                                                |
| `eval/episode_reward_se`   | Sample standard error of the eval-set episode rewards: `std(rewards, ddof=1) / sqrt(n)`.                                                                                                                                                                                    | The honest "how noisy is the eval reward" answer. **Bimodal policies blow this up** — if `episode_reward_mean` looks fine but `episode_reward_se` is huge, the policy is sometimes succeeding and sometimes catastrophically failing rather than reliably middling. Cross-check with `final_eval/std_reward` at end of run. |
| `eval/episode_length_mean` | Mean episode length over the eval pass.                                                                                                                                                                                                                                     | Should be ≈ `max_episode_len=200`. Drops below ~150 mean = the policy is triggering out-of-workspace early termination on a meaningful fraction of eval episodes (cloth flying away, etc.).                                                                                                                                 |
| `eval/video`               | mp4 panel. One mp4 every 4th checkpoint (every 40k steps with `--log_save_interval 20`). Each video shows `n_eval_episodes` deterministic episodes back-to-back, with SUCCESS/FAIL badges drawn per episode and the post-policy "settle" frames spliced in chronologically. | Watch one every couple of hours; tells you *why* the metrics look the way they do.                                                                                                                                                                                                                                          |
| `trajectory/video`         | Fallback path used only when `--use_wandb` is off (logs through SB3 → TB Video instead of `wandb.Video`). Same content as `eval/video`.                                                                                                                                     | If you're seeing this and you expected `eval/video`, your run was started without `--use_wandb`.                                                                                                                                                                                                                            |


### `reward/*` — reward decomposition (per-episode 100-ep moving avg)

Emitted by `PrivilegedObsWrapper._emit_episode_diagnostics` (or `PixelObsWrapper`'s equivalent) under the `rwd_diag/reward/*` info-key namespace, then logged by `RewardDiagnosticsCallback` with the prefix stripped. They tell you what shaping terms are actually firing.


| Chart                       | What it is                                                                                                                                                                                          |
| --------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `reward/episode_total`      | True total reward per episode after all shaping (`base_sum − vel_penalty_sum − action_penalty_sum + terminal_shaping − pre_settle_penalty`). The "real" objective.                                  |
| `reward/base_sum`           | Sum of dedo's per-step distance reward over the whole episode. Always negative.                                                                                                                     |
| `reward/vel_penalty_sum`    | Sum of velocity penalty over the episode (≥ 0). 0 when `vel_penalty=0`.                                                                                                                             |
| `reward/action_penalty_sum` | Sum of action-magnitude penalty over the episode (≥ 0). 0 when `action_penalty=0`. **Logged on every terminal step (incl. terminal action), unlike `vel_penalty_sum` which is non-terminal-only.**  |
| `reward/pre_settle_penalty` | Single-step pre-settle distance penalty applied at terminal (`pre_settle_coef × pre_settle_dist_m`). Magnitude ≥ 0 when knob is on; 0 otherwise. Per-episode 100-ep mean of this single-step value. |
| `reward/terminal_base`      | The single-step base reward at the terminal step (driven by `FINAL_REWARD_MULT=400`, dominates `base_sum` magnitude).                                                                               |
| `reward/terminal_shaping`   | Terminal `+success_bonus` or `-fail_penalty` actually applied. Tells you the success-bonus is firing — should be increasingly nonzero as success rate climbs.                                       |
| `reward/episode_length`     | Per-episode step count. Mostly 200; if it drops, episodes are terminating early (out-of-workspace).                                                                                                 |


### `success/*` — what counts as success (per-episode 100-ep avg)


| Chart                   | What it is                                                                                                                                                                                                                                                  |
| ----------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `success/active_rate`   | The success criterion the agent **actually trained against**. Equal to `adaptive_rate` when `success_factor` is set, else equal to `base_rate`. **This is the same series as `rollout/success_rate_100`**; they're redundant on purpose for cross-checking. |
| `success/adaptive_rate` | Your adaptive criterion: `dist < success_factor × hole_radius` (default `1.2 × hole_radius`).                                                                                                                                                               |
| `success/base_rate`     | Dedo's strict criterion: `|final_reward| < 2.5` (≈ `dist < 0.125 m`). Will be lower than `adaptive_rate` because it's stricter. Useful to know how many "almost successes" your runs have under the dedo definition.                                        |
| `success/disagree_rate` | Fraction of episodes where the two criteria disagree. Tells you how soft your adaptive criterion is vs dedo's strict one.                                                                                                                                   |


### `task/*` — geometric state (per-episode 100-ep avg)


| Chart                  | What it is                                                      | Expected behavior                                                                                                          |
| ---------------------- | --------------------------------------------------------------- | -------------------------------------------------------------------------------------------------------------------------- |
| `task/adaptive_dist`   | Mean hole-centroid → goal distance at episode end (m)           | **Most important leading indicator.** Should monotonically decrease as policy improves. Stalls before success rate climbs. |
| `task/adaptive_thresh` | The adaptive threshold value (= `success_factor × hole_radius`) | Roughly constant per cloth distribution (~0.5–0.7 m). Mostly useful as a horizontal reference line vs `adaptive_dist`.     |
| `task/hole_radius`     | Mean hole radius across recent episodes (m)                     | Should be ~constant; sudden changes mean the cloth distribution shifted.                                                   |


### `rwd_diag_meta/`* — diagnostics-callback bookkeeping


| Chart                         | What it is                                                                                                                                                                                       |
| ----------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| `rwd_diag_meta/episodes_seen` | Cumulative count of finished episodes the diagnostics callback has consumed. Sanity-check that wrappers are emitting `rwd_diag/*` keys at all.                                                   |
| `rwd_diag_meta/window_size`   | Current depth of the 100-ep rolling buffer. Will be < 100 for the first ~100 episodes, then constant at 100. Use to gauge whether `reward/*` / `success/*` rolling means are at full window yet. |


### `train/*` — SAC internals (every gradient update)

Logged automatically by SB3's `SAC.train()`. They appear at the *first* env step where a gradient happens (≥ `learning_starts`, default 1000) and update every `gradient_steps` thereafter. With `train_freq=1, gradient_steps=1` (defaults) that's one update per env step.


| Chart                 | What it is                                                                                                                                                                             | Red flag                                                                                                                                                                                                                                                                                                                                                                                                                           |
| --------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `train/actor_loss`    | Actor (policy) loss = `α · log π(a|s) − min(Q1, Q2)(s, a)`. Pushed lower (more negative) when Q rises along the policy's chosen actions.                                               | Should trend negative-and-stable. Wild oscillations mean LR is too high. Spikes upward when entropy collapses (α → 0 makes the `log π` term vanish, leaving raw Q-values).                                                                                                                                                                                                                                                         |
| `train/critic_loss`   | Mean squared TD error on Bellman targets, averaged over both Q heads (Q1, Q2).                                                                                                         | **Hard cap your eyes here.** Should stabilize <100 typically. Above 1000 and growing → run is collapsing, kill it. Expect a brief ramp from ~0 to peak in the first ~10k updates as the random Q networks fit returns, then a slow decay. The `SACCriticWarmupCallback` (default 10k env steps, see `_critic_warmup.py`) keeps the actor frozen during this ramp so the Q heads can converge before the actor follows their noise. |
| `train/ent_coef`      | Auto-tuned entropy temperature α (the SAC exploration knob). Starts at `--ent_coef` initial value (default `auto` = 1.0), evolves to satisfy `−H[π] ≈ target_entropy = −|action_dim|`. | Starts at 1.0, drifts to ~0.05–0.5 for this task. **Decays to ~0 in <100k steps = bad** (premature exploitation). Pin entropy with `--ent_coef 0.2` (fixed value, no `auto_`) when this happens.                                                                                                                                                                                                                                   |
| `train/ent_coef_loss` | Loss the α optimizer is minimizing: `−α · (log π + target_entropy)`. With auto-α, only this loss makes α move.                                                                         | Hovers near 0; magnitude tells you how aggressively α is being adjusted. Constantly large positive → the policy is too low-entropy and α is being pushed up. Constantly large negative → too high-entropy and α is being pushed down. Mostly diagnostic.                                                                                                                                                                           |
| `train/learning_rate` | Optimizer LR for actor + critic.                                                                                                                                                       | Constant `--lr` (default 3e-4) unless you scheduled it.                                                                                                                                                                                                                                                                                                                                                                            |
| `train/n_updates`     | Total gradient updates so far. Linear in `(env_steps − learning_starts) × gradient_steps`.                                                                                             | Useful as the X-axis for `train/`* (since nothing happens before `learning_starts`).                                                                                                                                                                                                                                                                                                                                               |


### `train/*` — PPO internals (every PPO.train() call)

PPO scripts (`train_privileged.py`, `train_pixels.py`, `train_pointcloud.py`) log a different set. PPO's `train()` runs once per rollout (= `n_steps × num_envs` env steps; defaults `n_steps=4096, num_envs=4` → every 16k env steps). All metrics below are averaged over the inner SGD loop (`n_epochs × num_minibatches` iterations).


| Chart                        | What it is                                                                                                            | Red flag                                                                                                                                                                                                              |
| ---------------------------- | --------------------------------------------------------------------------------------------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `train/policy_gradient_loss` | Negative of the clipped surrogate objective (so lower = better).                                                      | Trends mildly negative. Sudden positive jumps = the new policy is worse than the old at the rollout's actions.                                                                                                        |
| `train/value_loss`           | MSE between value-head predictions and the rollout returns.                                                           | Should decrease as the value head fits returns. If it grows for 100k+ steps the value head can't keep up — consider lowering `--lr` or raising `vf_coef`.                                                             |
| `train/entropy_loss`         | Negative mean policy entropy.                                                                                         | Becomes less negative (entropy shrinks) as the policy commits. **Approaches 0 too fast = premature commitment** — raise `ent_coef` or use `--log_std_init` to seed lower entropy from the start (see SAC + BC notes). |
| `train/approx_kl`            | Approximation of `KL(π_old || π_new)` over the update.                                                                | A standard "is the update too big" gauge. SB3 default `target_kl=None`. If you see values >0.1 routinely the trust region is being violated; lower `--lr` or raise `n_minibatches`.                                   |
| `train/clip_fraction`        | Fraction of samples whose probability ratio was clipped to `[1−ε, 1+ε]`.                                              | Healthy range ~0.05–0.3. Above ~0.5 = updates are mostly clipped (PPO's safety net is doing all the work, learning will stall).                                                                                       |
| `train/clip_range`           | The current clip parameter ε. Constant (default 0.2) unless scheduled.                                                | —                                                                                                                                                                                                                     |
| `train/clip_range_vf`        | Value-function clip range, if `clip_range_vf` is set.                                                                 | —                                                                                                                                                                                                                     |
| `train/explained_variance`   | `1 − Var(returns − values) / Var(returns)`; how much variance in returns the value head explains.                     | **The cleanest "is the value head working" signal.** Rises from ~0 toward ~1. Negative or stuck at 0 means value predictions are no better than predicting the mean — investigate before reading the policy charts.   |
| `train/loss`                 | Total loss = policy + vf + entropy losses, weighted.                                                                  | Mostly diagnostic; the components above are more readable.                                                                                                                                                            |
| `train/std`                  | Per-action-dim mean std of the diagonal Gaussian (PPO with `DiagGaussianDistribution`). Reflects `log_std` parameter. | Should slowly decrease as the policy commits. Pinned by `--log_std_init` at run start (default 0.0 → std=1; we use `-3.5` → std≈0.030 to match BC scale).                                                             |
| `train/learning_rate`        | Optimizer LR.                                                                                                         | Constant `--lr` unless scheduled.                                                                                                                                                                                     |
| `train/n_updates`            | Total inner SGD steps so far. Equal to `(rollouts × n_epochs)` for PPO.                                               | Sanity check.                                                                                                                                                                                                         |


### `time/`* — throughput / wall-clock


| Chart                  | What it is                                                                                                                                                                                                  | Expected behavior                                                                                                                                                                            |
| ---------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `time/total_timesteps` | Env steps elapsed.                                                                                                                                                                                          | Equal to wandb step. Linear in wall-clock if not throttled.                                                                                                                                  |
| `time/fps`             | Env steps per wall-clock second, computed by SB3 over the last log dump interval.                                                                                                                           | **Watch this for thermal throttling.** On the M4 you should see ~10 sps starting, may degrade ~20–30% over time. Sustained drops below 5 sps = something's wrong (FileProvider, swap, etc.). |
| `time/iterations`      | (PPO only) Number of `PPO.train()` calls completed = number of rollouts.                                                                                                                                    | Linear in `total_timesteps / (n_steps × num_envs)`.                                                                                                                                          |
| `time/episodes`        | Total episodes (≈ `steps / ep_len_mean`). **Note:** SB3 marks this `exclude="tensorboard"` for SAC, so it does *not* reach wandb on SAC runs — see `train/episodes_total` instead. PPO runs may surface it. | —                                                                                                                                                                                            |
| `time/time_elapsed`    | (Some SB3 versions) wall-clock seconds since `learn()` started.                                                                                                                                             | Use for cross-checking `time/fps` × `time/total_timesteps` ≈ wall time.                                                                                                                      |


### `last_episode/`* — raw per-episode values (no averaging)

Logged by `RewardDiagnosticsCallback._log_per_episode_to_wandb` directly via `wandb.log` (bypasses the SB3 logger to avoid record-overwrite-before-dump losses). One data point per episode end. Use these whenever you want individual-episode resolution that the rolling-mean charts can't give you.


| Chart                           | What it is                                                                                                                 |
| ------------------------------- | -------------------------------------------------------------------------------------------------------------------------- |
| `last_episode/success`          | 0 or 1, the *active* success criterion for the just-finished episode. Sparse spike train.                                  |
| `last_episode/success_adaptive` | 0/1 of the wrapper's adaptive criterion (`dist < success_factor × hole_radius`).                                           |
| `last_episode/success_base`     | 0/1 of dedo's strict `|final_reward| < 2.5` criterion. Will be ≤ adaptive.                                                 |
| `last_episode/dist`             | Final hole-centroid → goal distance in meters for the just-ended episode. Pair with `task/adaptive_thresh` reference line. |
| `last_episode/episode_total`    | Total reward (incl. shaping) for the just-ended episode. Quick way to see "did terminal_shaping fire".                     |
| `last_episode/episode_length`   | Step count of the just-ended episode. Drops below 200 = early-termination via out-of-workspace clip.                       |


### `train/`* (custom) — episode counters for x-axis

Logged from the same callback path as `last_episode/*`. These exist *specifically* to give you a useful x-axis (SB3's `time/episodes` is `exclude="tensorboard"` so it never reaches wandb — see X-axis section).


| Chart                        | What it is                                                                                                                                 |
| ---------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------ |
| `train/episodes_total`       | Cumulative episode counter. Use as the global wandb X-axis for everything except `bc/`*.                                                   |
| `train/cumulative_successes` | Cumulative count of successful episodes. **Each step up = one specific episode succeeded** — the cleanest way to count successes in a run. |


### `bc/`* — BC pretrain (only if `--bc_episodes > 0`)

These have `**bc/epoch` as their natural x-axis**, not step. They live in step 0 (since BC runs before `agent.learn()`).


| Chart                | What it is                                               |
| -------------------- | -------------------------------------------------------- |
| `bc/mse`             | MSE between actor mean action and demo action, per epoch |
| `bc/epoch`           | Epoch counter (1..bc_epochs)                             |
| `bc/n_pairs`         | Total (obs, act) pairs in BC dataset                     |
| `bc/n_demos`         | Number of demos collected                                |
| `bc/n_success_demos` | Successful demos in dataset                              |


### `final_eval/`* — once at end of training (deterministic eval)

Logged after `agent.learn()` finishes by `make_final_eval_collector` + `log_final_eval_metrics` in [_reward_diagnostics.py](_reward_diagnostics.py). Single data point per run, written via `wandb.log` at the final step. Episode count is `--n_final_eval_episodes` (default **50** for newer scripts; older runs default to 20).

**How it's computed:** `_FinalEvalCollector` walks every terminal `info` dict produced by `evaluate_policy`, capturing `is_success` and every `rwd_diag/`* numeric value. At the end:

- `final_eval/success_rate` = `sum(successes) / n`
- `final_eval/mean_reward`, `final_eval/std_reward` = mean & population std (`np.std` with `ddof=0`) of episode totals from `evaluate_policy`
- `final_eval/n_episodes` = `len(successes)` = `--n_final_eval_episodes`
- For **every** `rwd_diag/<metric>` key seen at any terminal step, two keys are emitted: `final_eval/<metric>` (= `np.mean`) and `final_eval/<metric>__std` (= `np.std`, ddof=0).

The `__std` fields are the whole point of this block. They distinguish a confidently-mediocre policy from a bimodal one: a 50% success rate looks identical for "always 50% confident" vs "0% on half / 100% on the other half" if you only have the mean — but `final_eval/success/active_rate__std` will be ~0 for the first and ~0.5 for the second.

#### Fixed keys (always present)


| Chart                     | What it is                                                                                                                                                                                                |
| ------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `final_eval/success_rate` | The **paper number**: deterministic success rate over the eval set. Identical to `final_eval/success/active_rate` (both come from `is_success`).                                                          |
| `final_eval/mean_reward`  | Mean episode return over the eval set, computed from `evaluate_policy`'s `episode_rewards`.                                                                                                               |
| `final_eval/std_reward`   | Population stddev of those returns — if huge, the policy is bimodal (sometimes solves it, sometimes whiffs catastrophically). Cross-check with `eval/episode_reward_se` from the last training-time eval. |
| `final_eval/n_episodes`   | Always equals `--n_final_eval_episodes`.                                                                                                                                                                  |


#### Auto-generated keys (one per `rwd_diag/`* info key, with a `__std` companion)

For every metric the wrappers emit at terminal step (see "Wrapper info keys" below), you'll see both `final_eval/<metric>` and `final_eval/<metric>__std`. The full list, assuming both privileged and pixel wrappers:


| Family      | `final_eval/...` keys                                                                                                                             | `__std` companion |
| ----------- | ------------------------------------------------------------------------------------------------------------------------------------------------- | ----------------- |
| `reward/*`  | `episode_total`, `base_sum`, `vel_penalty_sum`, `action_penalty_sum`, `pre_settle_penalty`, `terminal_base`, `terminal_shaping`, `episode_length` | yes (each)        |
| `success/*` | `base_rate`, `adaptive_rate`, `disagree_rate`, `active_rate`                                                                                      | yes (each)        |
| `task/*`    | `adaptive_dist`, `adaptive_thresh`, `hole_radius`                                                                                                 | yes (each)        |


Reading guide:

- `final_eval/reward/episode_total__std` huge but `final_eval/success_rate` near 1.0 → success bonus dominates, value spread is mostly determined by terminal_base on the small failures. Not a worry.
- `final_eval/success/active_rate__std` near 0.5 with `final_eval/success_rate` ≈ 0.5 → bimodal policy, *not* a "consistently 50%" policy. Investigate which cloths it's failing on (use `--eval_seed_lock` and watch the eval video).
- `final_eval/task/adaptive_dist__std` large vs the mean → policy is inconsistent in *where* it ends up, even on episodes that "succeed" by the loose adaptive criterion. Usually correlates with `success/disagree_rate__std` being non-zero.
- `final_eval/reward/episode_length__std` non-trivial → some episodes early-terminate (out-of-workspace), others don't. Diagnostic for "policy occasionally throws the cloth off the table".

**Older scripts (`train_privileged_evan.py`)** use a simpler one-shot eval and only emit `final_eval/{success_rate, mean_reward, std_reward, n_episodes}` — none of the `__std` decomposition.

### Wrapper info keys (the source of `reward/`*, `success/*`, `task/*`, and `final_eval/*`)

These are not charts you read directly; they're the per-step `info` dict keys emitted by the env wrappers. The diagnostics callback collects them at terminal step, computes per-episode aggregates (sums for `reward/*_sum`, single-shot for `terminal_*` and `task/*`, etc.), and either records to TB (rolling means) or pipes through `_FinalEvalCollector` (final eval). Documented here because if you suspect a chart is missing or wrong, this is the data layer to inspect.


| Info key             | Emitted by                                            | Role                                                                                                 | Charts that consume it                                                                                                                                  |
| -------------------- | ----------------------------------------------------- | ---------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `is_success`         | All wrappers                                          | Active success criterion (adaptive if `success_factor` set, else dedo base).                         | `eval/success_rate`, `final_eval/success_rate`, `last_episode/success`, `success/active_rate`, `train/cumulative_successes`, `rollout/success_rate_100` |
| `adaptive_dist`      | All wrappers (when `success_factor` set)              | Hole-centroid → goal distance at this step (m). Set every step, but only the terminal value matters. | `task/adaptive_dist`, `last_episode/dist`, `final_eval/task/adaptive_dist[__std]`                                                                       |
| `adaptive_thresh`    | All wrappers (when `success_factor` set)              | `success_factor × hole_radius` (m).                                                                  | `task/adaptive_thresh`, `final_eval/task/adaptive_thresh[__std]`                                                                                        |
| `hole_radius`        | All wrappers (when `success_factor` set)              | Effective hole radius for the current cloth (m).                                                     | `task/hole_radius`, `final_eval/task/hole_radius[__std]`                                                                                                |
| `shaping_added`      | All wrappers (when shaping fires at terminal)         | Per-step shaping added: `+success_bonus` on success, `−fail_penalty` on fail.                        | `reward/terminal_shaping`, `final_eval/reward/terminal_shaping[__std]`                                                                                  |
| `pre_settle_dist_m`  | All wrappers                                          | Hole→goal distance measured **before** the post-policy gravity settle. Diagnostic only.              | (used by `_video_callback` to label settle frames; logged via `reward/pre_settle_penalty` after multiplication by `pre_settle_coef`)                    |
| `pre_settle_penalty` | Privileged + Pixel wrappers                           | Single-step penalty `pre_settle_coef × pre_settle_dist_m`, applied at terminal.                      | `reward/pre_settle_penalty`, `final_eval/reward/pre_settle_penalty[__std]`                                                                              |
| `vel_penalty`        | All wrappers (per non-terminal step)                  | Per-step velocity penalty actually applied.                                                          | Summed by callback into `reward/vel_penalty_sum`.                                                                                                       |
| `action_penalty`     | Pixel + Privileged wrappers (per step incl. terminal) | Per-step action-magnitude penalty applied.                                                           | Summed by callback into `reward/action_penalty_sum`.                                                                                                    |
| `cloth_mean_speed`   | Pixel + PointCloud wrappers                           | Mean cloth-vertex speed (m/s). Diagnostic for cloth velocity.                                        | Not currently rolled into a chart — surfaces only as a per-step info value. Useful when debugging velocity-penalty tuning.                              |
| `action_cost`        | Pixel + Privileged wrappers                           | Cumulative action penalty across the episode. Diagnostic.                                            | Not directly charted; effectively the same series as `reward/action_penalty_sum`.                                                                       |


If you turn on a new shaping knob and don't see the corresponding `reward/<name>` chart move, the first thing to check is whether the wrapper is emitting the corresponding `info` key — `print(info)` in a one-step rollout is faster than reading the chart. The callback only logs what it sees.

### Critic warmup callback (no chart, but shapes `train/critic_loss`)

The `SACCriticWarmupCallback` (default `n_warmup_env_steps=10000`) and `PPOCriticWarmupCallback` (default `n_warmup_rollouts=2`) freeze the actor's parameters for the first chunk of training so the value/Q heads can stabilize before they start moving the actor. They emit **no metrics** (only stdout banners), but they distort the early shape of `train/critic_loss`, `train/actor_loss` (= 0 for SAC during freeze since no actor gradients flow), and `train/policy_gradient_loss` (PPO).

**What you'll see in the charts because of this:**

- `train/critic_loss` ramps from ~0 to its peak by ~step 10k (SAC) or after ~2 rollouts (PPO), uninterrupted by actor drift. This is *expected*, not a bug.
- `train/actor_loss` is identically 0 (SAC) or constant (PPO) during the freeze window — also expected. The policy is unchanged through warmup.
- The first time `train/actor_loss` starts moving is the moment of unfreeze. Cross-reference with the stdout `[critic_warmup] ... unfroze actor` banner for the exact env step.

Disable with `--critic_warmup_env_steps 0` (SAC) or `--critic_warmup_rollouts 0` (PPO) if you want to read pre-warmup actor behavior directly.

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


| For these charts                                                     | Use x-axis                      | Why                                                                                                                                                           |
| -------------------------------------------------------------------- | ------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **Default for everything except `bc/`* and `train/*` SAC-internals** | `train/episodes_total`          | Reads as "completed deployments". Step 8000 = episode ~50 vs. "8k of 1.5M" reads cleaner as "50 of ~7500". Logged per-episode by `RewardDiagnosticsCallback`. |
| `bc/mse`, `bc/epoch_loss`                                            | `bc/epoch`                      | The natural x-axis for BC; otherwise all 30 points stack at step 0.                                                                                           |
| `train/critic_loss`, `train/actor_loss`, `train/ent_coef`            | `train/n_updates`               | Honest x-axis for SAC internals — nothing happens before `learning_starts`, and `n_updates` only ticks during gradient updates.                               |
| Performance debugging (M4 throttling, FileProvider stalls)           | `_runtime` (wall-clock seconds) | If `time/fps` looks fine vs Step but the run takes forever in `_runtime`, something is making each step expensive in wall-clock.                              |
| Cross-algo baselining (PPO vs SAC)                                   | `Step` (= env steps)            | Cross-algo comparable since both use the same env-step axis.                                                                                                  |


### Why we don't use `time/episodes`

SB3 1.2.0's SAC internally logs `time/episodes` and `time/total timesteps` with `exclude="tensorboard"`, which means they never reach the TB events file and (because wandb syncs from TB) they never reach wandb either. So those metrics simply don't exist on the wandb side, even though they show up nicely formatted in your stdout dump tables. Our `train/episodes_total` is the workaround — same content, logged **on every env step** via `RewardDiagnosticsCallback` → TensorBoard → wandb sync so it appears on the same global-step rows as `train/critic_loss` etc. Without that per-step mirror, wandb could only see `train/episodes_total` on episode-completion steps if we used `wandb.log` alone, and choosing it as the chart X-axis would show **no data** for dense metrics (nothing to join). If you still see "no data", refresh the run page after ~~1 episode (~~200 steps post-reset) and confirm `train/episodes_total` appears under the run's **Scalars** / metric search.

Set the **smoothing slider** to ~0.7 globally for the noisy episode-level charts (`rollout/`*, `reward/*`, `success/*`, `task/*`). For the per-update `train/*` charts, lower smoothing is fine since they're already aggregated by SB3. SAC logs every gradient update so they're dense. **Set smoothing to 0** on `last_episode/`* and `train/cumulative_successes` — those are exactly the charts where you want to see individual episode events sharply.

---

## What to watch for at each phase

### 0–5k steps (learning_starts phase)

The actor is uniform random `[-1, 1]^6`. **Don't read into anything.** Reward is very negative, success near 0, train losses are 0 (no updates yet).

What to verify:

- `time/fps` is ~10 — you're actually stepping the sim
- BC banner printed nicely if BC was on (only relevant when you re-enable BC)

### 5k–100k steps (early SAC)

This is where the run is most likely to silently fail.


| Symptom                                                                                                                                 | Likely cause                                                                                                                                                         | Action                                                                                                                                       |
| --------------------------------------------------------------------------------------------------------------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------- |
| `train/critic_loss` exploding past 1000                                                                                                 | Exploration too wild + high LR                                                                                                                                       | Kill, retry with `--lr 1e-4`                                                                                                                 |
| `train/ent_coef` hits 0.01 by step 20k                                                                                                  | Reward magnitudes too big, actor over-confident                                                                                                                      | Kill, retry with `--ent_coef 0.2` (fixed, no auto)                                                                                           |
| `rollout/ep_len_mean` drops below 150                                                                                                   | Out-of-workspace clipping → many early terminations                                                                                                                  | Symptom of cloth flying away. Kill, retry with `--vel_penalty 1.0`                                                                           |
| `time/fps` halves over 20k steps                                                                                                        | Thermal throttle or FileProvider                                                                                                                                     | Move logdir off Drive (you did), check thermal                                                                                               |
| First success at step 1–2k (BC), then `last_episode/success` flatlines at 0 for thousands of episodes after `ent_coef` falls below 0.01 | The "ent_coef collapse" — auto-entropy commits to whatever the BC seed pointed at and stops exploring. Confirmed seen in privileged hole_centroid SAC at ~step 240k. | Kill. Re-run with `--ent_coef 0.2 --log_std_init -2`. Pinning entropy prevents the collapse; lower log_std init makes BC immediately useful. |


What you want to see:

- `task/adaptive_dist` starting to drop (this is the very first sign of learning, before success_rate moves)
- `train/critic_loss` rising then leveling off (typical: ramps to 50–200 by step 50k, then flat)
- `train/ent_coef` slowly decaying (0.5 → 0.2 over 50k steps)

### 100k–500k steps (the make-or-break window)

You should be seeing the success rate climb in this window. Specifically:

- By step **200k**: `eval/success_rate` should be at least 5–10%, `task/adaptive_dist` clearly below `task/adaptive_thresh`
- By step **500k**: `eval/success_rate` ≥ 30% if the run is going to converge

If at 300k your `adaptive_dist` is flat AND `eval/success_rate` is stuck at <5%, the run is stuck in a local minimum and won't recover — kill it.

### 500k–1.5M steps (convergence)

`eval/success_rate` should be plateauing. The interesting question becomes: how stable is the plateau?

- **Stable plateau** (low variance between consecutive evals): policy is converged, you've got your number. The `final_eval/`* block at the end will be a clean 20-episode reading.
- **Oscillating plateau** (jumping ±20% between consecutive evals): SAC critic is unstable. The policy is OK on average but sensitive to which mini-batch it just saw. Lowering `--lr` to 1e-4 for the next run helps.
- **Catastrophic forgetting** (rate climbs then crashes back to 0): rare but happens with auto-entropy. Stop at the best checkpoint; future runs use `--ent_coef 0.1` fixed.

---

## A workspace setup that makes this all easier

In the wandb run page, click the ⚙️ next to the workspace name and create three sections:

1. **Headline** — drag in `eval/success_rate` *(with `eval/success_rate_se` overlaid as a band if your wandb chart supports it)*, `train/cumulative_successes`, `last_episode/success`, `task/adaptive_dist`, `eval/video`. This is the only section you need to look at most of the time. Set workspace X-axis to `train/episodes_total`. Set smoothing to 0 for `last_episode/`* and `train/cumulative_successes`; ~0.7 for the rest.
2. **Health** — drag in `train/critic_loss`, `train/ent_coef`, `train/actor_loss`, `time/fps`, `eval/episode_length_mean`. Glance once an hour. `**train/ent_coef` is the canary** — if it drops below 0.01 the run is collapsing, kill it. (PPO runs: swap in `train/explained_variance`, `train/clip_fraction`, `train/approx_kl` instead of the SAC-specific keys.)
3. **Decomposition** — drag in `reward/`* charts (`episode_total`, `base_sum`, `vel_penalty_sum`, `action_penalty_sum`, `pre_settle_penalty`, `terminal_base`, `terminal_shaping`) plus `eval/episode_reward_mean`/`eval/episode_reward_se`, `last_episode/dist`, `last_episode/episode_length`. Look at this only when something looks weird in **Headline**, or when you turn on a new shaping knob and want to confirm it's actually firing at the magnitude you expected.

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
3. `**--log_std_init` is skipped** (whatever value it had during the original run is in the saved weights; SAC has been moving it via gradient steps since then).
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

1. **Default exploration noise drowns out BC-trained mu (PPO and SAC).** Scripted hole-aware demos have action magnitude ≈ **0.03** in normalized [-1, 1] space (waypoint vels ~0.3 m/s ÷ `MAX_ACT_VEL=10`). Default initial std in *both* algos is ~1.0 in pre-tanh / pre-clip space, so noise is ~30× louder than the BC signal — sampled rollouts are essentially uniform on [-1, 1]^6 and `rollout/success_rate_100` stays at 0 even when `bc/mse` looks great and `eval/success_rate` (deterministic mu) is nonzero. Two distinct mechanisms:
  - **PPO** (`DiagGaussianDistribution`): `log_std = nn.Parameter` initialized from `policy_kwargs['log_std_init']`, default 0.0 → std=1.0. Action = `mu + N(0, std)` then clipped to [-1, 1]. Fix in `train_privileged.py` / `train_pixels.py`: `--log_std_init -3.5` (std≈0.030, matches demo magnitude).
  - **SAC** (`SquashedDiagGaussianDistribution`): `log_std = nn.Linear(...)` head with default torch init when `use_sde=False`; SB3 silently ignores `policy_kwargs['log_std_init']` on this code path. Output std ≈ 1 in pre-tanh, so `tanh(mu + N(0, ~1))` is heavily biased toward ±1. Fix in `train_privileged_sac.py` / `train_pixels_sac.py`: `--log_std_init -3.5` patches `actor.log_std.bias` to a constant after construction (and zeros the weight), giving std ≈ 0.030. BC bias is then immediately visible.
   Diagnostic: if `bc/mse` is small (≈10⁻⁴) but `rollout/success_rate_100` is flat at 0, check `eval/success_rate` next — if it's nonzero, this is the bug. The 77% you might be quoting from `bc/n_success_demos / bc/n_demos` is the **scripted controller**'s success rate during demo collection, not the BC policy's success rate.
2. **Auto-entropy collapses under large terminal rewards.** With `success_bonus=200` and `FINAL_REWARD_MULT=400` your terminal reward is ±300+, while non-terminal rewards live in ±0.5 — a 1000× scale gap. SAC's auto-tuned `ent_coef` interprets this as "already very confident, drive entropy down" and decays to <0.01 within 100–200k steps, killing exploration. PPO doesn't have this knob and is unaffected. **Fix**: `--ent_coef 0.2` (fixed, no auto) in any run with `success_bonus > 50`.
3. `**ep_len_mean` is *not* always `max_episode_len`.** Out-of-workspace early termination clips episodes (often to 60–80 steps) when the policy fails badly. This shifts the meaning of "100-ep moving average" charts (e.g. `rollout/success_rate_100`) — at step 8k with `ep_len_mean ≈ 80`, the 100-ep window has saturated; with `ep_len_mean ≈ 200`, only 40 episodes have happened. Translation between Step and "episodes seen" requires knowing the live mean. The `train/episodes_total` x-axis sidesteps this entirely.

