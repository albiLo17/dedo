# HangProcCloth — privileged-obs PPO+BC runs

Reference for the privileged hole_centroid PPO+BC sweep. The series tests
two related questions:

1. **Can we prevent BC erasure by PPO?** BC pretrain consistently lands the
   actor at 0.5 to 0.77 eval success rate on hole_centroid; PPO then
   walks the policy off that level over the first 100k to 300k env steps.
   Runs A through I explore lr / warmup / reward-shape / capacity /
   drift caps / actor-anchoring / critic-warmup variants, looking for a
   configuration that keeps post-BC eval success near or above the BC
   starting level.
2. **Do alternate reward shapes do better?** Different combinations of
   `success_factor`, `success_bonus`, and `pre_settle_coef` are sampled
   across runs to give independent reads on each.

All runs use:

- `--obs_mode hole_centroid` (18-dim privileged obs on `HangProcCloth-v1`)
- `--seed 42`
- `--max_act_vel 4.1`
- `--log_std_init -2.7`
- `--bc_episodes 300 --bc_demos_only_success --bc_epochs 60`
- `--total_env_steps 3000000`

Variations are noted per run.

---

## Quick summary

| Run | Hypothesis                                   | Net     | lr   | Critic warmup (rollouts / demo epochs) | sf  | sb  | psc | Other knobs                                              | Status / outcome                                       |
| --- | -------------------------------------------- | ------- | ---- | -------------------------------------- | --- | --- | --- | -------------------------------------------------------- | ------------------------------------------------------ |
| A   | BC-preserve via low lr + long warmup         | 256x256 | 5e-6 | 8 / 0                                  | 1.2 | 100 | 20  | (none)                                                   | Killed. Eval 0.56 to 0 by 280k. Worst collapse.        |
| B   | Tight success threshold forces threading     | 256x256 | 2e-5 | 4 / 0                                  | 0.8 | 100 | 20  | (none)                                                   | Killed. Flat near 0. Bug-contaminated (M4).            |
| C   | Smooth-cliff reward (low sb, high psc)       | 256x256 | 2e-5 | 4 / 0                                  | 1.2 | 25  | 80  | (none)                                                   | Killed. 0.67 to 0.13. Bug-contaminated (M4).           |
| D   | Bigger MLP for BC capacity                   | 512x512 | 5e-5 | 2 / 0                                  | 1.2 | 100 | 20  | (none)                                                   | Killed. 0.77 to 0.10–0.30. Highest BC retention. M4.   |
| E   | Kitchen-sink combined config                 | 512x512 | 1e-5 | 6 / 0                                  | 0.8 | 50  | 40  | (none)                                                   | Killed. Flat at 0.10.                                  |
| F   | Slow-drift / tight-PPO on D-base             | 512x512 | 5e-5 | 2 / 0                                  | 1.2 | 100 | 20  | clip_range=0.05, ppo_epochs=3, target_kl=0.02            | Running. Tracks G; slow-drift not the lever so far.    |
| G   | Clean L4 rerun of D                          | 512x512 | 5e-5 | 2 / 0                                  | 1.2 | 100 | 20  | (none)                                                   | Running. Baseline for F / H / I comparison.            |
| H   | BC anchor on D-base                          | 512x512 | 5e-5 | 2 / 0                                  | 1.2 | 100 | 20  | bc_anchor_batches=4, bc_anchor_lr=1e-4                   | Running.                                               |
| I   | Demo-based critic warmup on D-base           | 512x512 | 5e-5 | 0 / 50                                 | 1.2 | 100 | 20  | critic_warmup_demo_lr=3e-4                               | Running (rerun, after SB3 API fix). Real recovery: 0.03 plateau through 500k → 0.23 by 750k. Kept alive parallel to K. |
| J   | vp=0 ablation on D-base                      | 512x512 | 5e-5 | 2 / 0                                  | 1.2 | 100 | 20  | **vel_penalty=0** (rest = D-base)                        | **REJECTED.** vp=0 did not prevent BC erasure; same 0.7 → 0.05 collapse pattern as D/G by 250k. Diagnosis pivoted to reward-shape (terminal-step dominance), not vel_penalty. |
| K   | Dense reward redesign on D-base              | 512x512 | 5e-5 | 2 / 0                                  | 1.2 | 100 | 20  | **vel_penalty=0, dist_reward_coef=1.0, final_reward_mult=50** | TBD. First test of the rebalanced per-step reward (see "Reward redesign" section). |

Critic-warmup column reads as `<rollouts> / <demo_epochs>`:

- `--critic_warmup_rollouts N` (left side): freeze the actor for N PPO
  rollouts after BC pretrain, training only the critic on early-PPO
  rollout data.
- `--critic_warmup_demo_epochs N` (right side): pretrain V(s) on Monte
  Carlo discounted returns from the BC demos for N epochs **before** PPO
  starts (introduced for Run I).

Run F was originally launched with A-base (256x256, lr=5e-6, warmup=8),
then rebuilt on D-base after D was identified as the strongest baseline.
Both versions exist in wandb history under the same letter; the active
F is D-base.

---

## Wandb naming convention

Each run has **two** name forms in wandb:

1. **Manual descriptive name** — the prefix you set in the wandb UI (or
   via `wandb.run.name = ...` in code). This is the human-readable
   label used throughout this doc.
2. **Auto-generated suffix** — appended by the script at
   [train_privileged.py:529-530](scripts/train_privileged.py#L529-L530)
   to encode reward and arch params. Format:

   ```
   PPO_<YYMMDD>_<HHMMSS>_HangProcCloth-v1_<arch>_sf<sf>_sb<sb>[_fp<fp>]_vp<vp>[_ap<ap>][_psc<psc>]
   ```

   Where:

   - `arch` — `<h1>x<h2>...` from `--net_arch` (e.g. 256x256, 512x512)
   - `sf` — `--success_factor` (e.g. 1.2, 0.8, or `_sf_default` if disabled)
   - `sb` — `--success_bonus` (omitted if 0)
   - `fp` — `--fail_penalty` (omitted if 0)
   - `vp` — `--vel_penalty` (omitted if 0)
   - `ap` — `--action_penalty` (omitted if 0)
   - `psc` — `--pre_settle_coef` (omitted if 0)

   Example from a D-base run:

   ```
   PPO_260506_124353_HangProcCloth-v1_512x512_sf1.2_sb100_vp8_psc20
   ```

   → 512x512 net, success_factor=1.2, success_bonus=100, vel_penalty=8,
   pre_settle_coef=20.

The auto-suffix uniquely identifies the *config family* (reward + arch)
but **not** the run identity. F, G, H, and I all share the same
auto-suffix because they share D's reward and arch base:

```
_512x512_sf1.2_sb100_vp8_psc20
```

Use the manual prefix to disambiguate.

---

## Per-run details

### Run A — BC-preserve

**Hypothesis.** With BC pretrain bringing the actor close to the demo
manifold, low PPO lr (5e-6) plus extended critic warmup (8 rollouts,
approximately 131k frozen-actor steps) should let the critic catch up
before the actor is unfrozen, preventing the early-PPO update from
pushing the actor in noise directions.

**Wandb manual name.**

```
A: BC-preserve - low PPO lr (5e-6); 8 critic warmup - 256x256_sf1.2_sb100_vp8_psc20
```

**Auto-suffix.**

```
_256x256_sf1.2_sb100_vp8_psc20
```

**Command.** Launched on the L4 via [scripts/launch_l4.sh](scripts/launch_l4.sh).

```bash
python experiments/hang_obs_exp/scripts/train_privileged.py \
    --obs_mode hole_centroid \
    --max_act_vel 4.1 --log_std_init -2.7 \
    --lr 5e-6 --critic_warmup_rollouts 8 \
    --success_factor 1.2 --success_bonus 100 \
    --pre_settle_coef 20 --vel_penalty 8 \
    --bc_episodes 300 --bc_demos_only_success --bc_epochs 60 \
    --net_arch 256,256 \
    --total_env_steps 3000000 --seed 42 --cpu --use_wandb
```

**Outcome.** Eval success_rate started at 0.57 post-BC, fell to 0.10 by
80k steps and to 0.0 by 280k steps. Worst collapse of the sweep,
contradicting the BC-preservation premise. Conservative lr/warmup did
not save BC; the gradient *direction* (not magnitude) was the issue.

**Status.** Killed early to free cores.

---

### Run B — Tight-thresh

**Hypothesis.** Looser default success threshold (sf=1.2 means
the policy must end within 1.2 × hole_radius of the goal — eval videos
show this is roughly 1.27 m in dedo's sim units, so hole_radius is
about 1.06 m) lets the policy succeed without actually threading.
Tightening to sf=0.8 (about 0.85 m) forces real threading, giving the
policy a crisper signal about what "success" means.

**Wandb manual name.**

```
B: Tight-thresh - sf=0.8 forces actual threading - 256x256_sf0.8_sb100_vp8_psc20
```

**Auto-suffix.**

```
_256x256_sf0.8_sb100_vp8_psc20
```

**Command.** Launched on the M4 via [scripts/launch_m4.sh](scripts/launch_m4.sh).

```bash
python experiments/hang_obs_exp/scripts/train_privileged.py \
    --obs_mode hole_centroid \
    --max_act_vel 4.1 --log_std_init -2.7 \
    --lr 2e-5 --critic_warmup_rollouts 4 \
    --success_factor 0.8 --success_bonus 100 \
    --pre_settle_coef 20 --vel_penalty 8 \
    --bc_episodes 300 --bc_demos_only_success --bc_epochs 60 \
    --net_arch 256,256 \
    --total_env_steps 3000000 --seed 42 --cpu --use_wandb
```

**Outcome.** Eval flat near 0 throughout. Bug-contaminated: ran on M4
with the sim early-termination bug, so signal is noisy. Pattern still
matched the broader sweep (post-BC collapse).

**Status.** Killed.

---

### Run C — Smooth-cliff

**Hypothesis.** Standard config concentrates reward at the success
event (sb=100 at terminal). This creates a "cliff" in the value
landscape: high gradient at success, near zero outside. A smoother
distribution — lower terminal bonus (sb=25) but steeper pre-settle
gradient (psc=80, four times the default 20) — should give the policy
denser signal away from the threshold.

**Wandb manual name.**

```
C: Smooth-cliff - sb=25; psc=80 (steeper dist gradient) - 256x256_sf1.2_sb25_vp8_psc80
```

**Auto-suffix.**

```
_256x256_sf1.2_sb25_vp8_psc80
```

**Command.** Launched on the M4 via [scripts/launch_m4.sh](scripts/launch_m4.sh).

```bash
python experiments/hang_obs_exp/scripts/train_privileged.py \
    --obs_mode hole_centroid \
    --max_act_vel 4.1 --log_std_init -2.7 \
    --lr 2e-5 --critic_warmup_rollouts 4 \
    --success_factor 1.2 --success_bonus 25 \
    --pre_settle_coef 80 --vel_penalty 8 \
    --bc_episodes 300 --bc_demos_only_success --bc_epochs 60 \
    --net_arch 256,256 \
    --total_env_steps 3000000 --seed 42 --cpu --use_wandb
```

**Outcome.** Eval 0.67 to 0.13 over training. Stable mid-band retention
(better than A) but no climbing. Bug-contaminated (M4).

**Status.** Killed.

---

### Run D — Bigger-net

**Hypothesis.** With 300 demos and a 256x256 MLP, BC may be
capacity-limited (approximately 1.2 param/datum ratio). Bumping to
512x512 (approximately 5 param/datum) should give BC enough capacity
to fit demos cleanly, raising the post-BC starting point.

**Wandb manual name.**

```
D: Bigger-net - 512x512 MLP for BC capacity - 512x512_sf1.2_sb100_vp8_psc20
```

**Auto-suffix.**

```
_512x512_sf1.2_sb100_vp8_psc20
```

**Command.** Launched on the M4 via [scripts/launch_m4.sh](scripts/launch_m4.sh).

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

**Outcome.** Highest post-BC eval of the A–E sweep (0.77). Mid-training
band of 0.10 to 0.30, occasional spikes to 0.30. Bug-contaminated (M4)
but the hypothesis was still validated: bigger net retains more BC.

**Status.** Killed in favor of clean L4 rerun (G).

**Why it became the baseline for F/G/H/I.** Best BC retention of A–E.
All subsequent fix-tests use D's reward + arch + lr + warmup as the
control, varying only the new mechanism under test.

---

### Run E — Combined

**Hypothesis.** Mix the most-promising knobs from B / C / D: bigger net
(D), smoother reward shape (closer to C — sb=50, psc=40), lower sf (B),
moderate lr/warmup. Best-guess kitchen sink given uncertainty about
which knob matters most.

**Wandb manual name.**

```
E: Combined - lr=1e-5; 6 warmup; 512x512 - 512x512_sf0.8_sb50_vp8_psc40
```

**Auto-suffix.**

```
_512x512_sf0.8_sb50_vp8_psc40
```

**Command.** Launched on the L4 via [scripts/launch_l4.sh](scripts/launch_l4.sh).

```bash
python experiments/hang_obs_exp/scripts/train_privileged.py \
    --obs_mode hole_centroid \
    --max_act_vel 4.1 --log_std_init -2.7 \
    --lr 1e-5 --critic_warmup_rollouts 6 \
    --success_factor 0.8 --success_bonus 50 \
    --pre_settle_coef 40 --vel_penalty 8 \
    --bc_episodes 300 --bc_demos_only_success --bc_epochs 60 \
    --net_arch 512,512 \
    --total_env_steps 3000000 --seed 42 --cpu --use_wandb
```

**Outcome.** Eval flat at 0.10 throughout, never reached the BC peak
that D did. Mix of sf=0.8 + bigger net didn't compound, suggesting the
two-knob interaction is non-monotonic.

**Status.** Killed.

---

### Run F — Slow-drift on D-base

**Hypothesis.** PPO erases BC because per-rollout actor drift is
unbounded. Tightening `clip_range` (0.20 to 0.05), reducing `n_epochs`
(10 to 3), and adding a `target_kl` early-stop (0.02) limits how far
the actor can move per rollout, giving the critic time to align with
BC-visited regions before the actor strays.

**Wandb manual name.**

```
F: Slow-drift on D-base - clip=0.05; epochs=3; target_kl=0.02 (throttle PPO drift) - 512x512_sf1.2_sb100_vp8_psc20
```

**Auto-suffix.** Same as D / G / H / I:

```
_512x512_sf1.2_sb100_vp8_psc20
```

**Command.** Launched manually via tmux on the L4 (also wired into
[scripts/launch_l4.sh](scripts/launch_l4.sh)).

```bash
python experiments/hang_obs_exp/scripts/train_privileged.py \
    --obs_mode hole_centroid \
    --max_act_vel 4.1 --log_std_init -2.7 \
    --lr 5e-5 --critic_warmup_rollouts 2 \
    --success_factor 1.2 --success_bonus 100 \
    --pre_settle_coef 20 --vel_penalty 8 \
    --bc_episodes 300 --bc_demos_only_success --bc_epochs 60 \
    --net_arch 512,512 \
    --ppo_clip_range 0.05 --ppo_epochs 3 --ppo_target_kl 0.02 \
    --total_env_steps 3000000 --seed 42 --cpu --use_wandb
```

**Code.** New PPO knobs exposed at
[train_privileged.py:150-176](scripts/train_privileged.py#L150-L176),
wired into `rl_kwargs` at
[train_privileged.py:564-566](scripts/train_privileged.py#L564-L566).

**Outcome.** Tracking G's trajectory closely through about 150k
steps — slow-drift caps don't appear to be the lever. Post-hoc
reasoning: with
`clip_range=0.05` already very tight, most ratios are clipped and the
gradient signal is sparse; `target_kl=0.02` may early-stop noisy
batches before they integrate, leaving log_std drifting upward
(slightly more exploration, slightly more degradation).

**Status.** Running.

**Note.** F was originally launched on A-base (256x256, lr=5e-6,
warmup=8) before the data made clear D was the strongest baseline.
The A-base F was killed; the D-base F is the current run.

---

### Run G — D-clean

**Hypothesis.** D was the strongest run of A–E but ran on M4 with the
sim early-termination bug. A clean L4 rerun of D's exact recipe (a)
disambiguates "the bug helped D" vs "D's config is good," and (b)
provides a contention-free baseline against which F / H / I's fix
mechanisms can be compared.

**Wandb manual name.**

```
G: D-clean - 512x512 lr=5e-5 warmup=2 (clean L4 rerun of D) - 512x512_sf1.2_sb100_vp8_psc20
```

**Auto-suffix.** Same as D / F / H / I:

```
_512x512_sf1.2_sb100_vp8_psc20
```

**Command.** Launched manually via tmux on the L4. Identical to D.

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

**Outcome.** Tracking D's trajectory (0.77 starting BC eval, then
descent toward the same plateau). Confirms: D's pattern is real, not a
bug artifact.

**Status.** Running. The cross-comparison baseline.

---

### Run H — BC-anchor on D-base

**Hypothesis.** PPO has no built-in pull toward BC; the policy gradient
is "increase advantage" and never references the BC actor again. A
periodic MSE replay against demo (obs, action) pairs after each PPO
rollout adds that pull explicitly — the same mechanism as DAPG. Demos
are reused from the in-memory dataset; no extra env interaction.

**Wandb manual name.**

```
H: BC-anchor on D-base - 4 batches/rollout @ lr=1e-4 - 512x512_sf1.2_sb100_vp8_psc20
```

**Auto-suffix.** Same as D / F / G / I:

```
_512x512_sf1.2_sb100_vp8_psc20
```

**Command.** Launched manually via tmux on the L4.

```bash
python experiments/hang_obs_exp/scripts/train_privileged.py \
    --obs_mode hole_centroid \
    --max_act_vel 4.1 --log_std_init -2.7 \
    --lr 5e-5 --critic_warmup_rollouts 2 \
    --success_factor 1.2 --success_bonus 100 \
    --pre_settle_coef 20 --vel_penalty 8 \
    --bc_episodes 300 --bc_demos_only_success --bc_epochs 60 \
    --net_arch 512,512 \
    --bc_anchor_batches 4 --bc_anchor_batch_size 256 --bc_anchor_lr 1e-4 \
    --total_env_steps 3000000 --seed 42 --cpu --use_wandb
```

**Code.**

- Callback at [scripts/_bc_anchor.py](scripts/_bc_anchor.py).
- CLI knobs at [train_privileged.py:177-192](scripts/train_privileged.py#L177-L192).
- Wired up at [train_privileged.py:1053-1066](scripts/train_privileged.py#L1053-L1066).

**Outcome.** TBD, recently launched. Watch the new
`train/bc_anchor_mse` panel: should rise from about 0.001 (post-BC
level) toward some bounded value as the actor drifts and the anchor
pulls back.

**Status.** Running.

---

### Run J — vp=0 ablation on D-base

**Hypothesis** (the empirical reframe). The dominant per-episode reward
component across A–I is `vel_penalty_sum` (~60–75), which is *larger
than the success bonus* (`terminal_shaping` ~50 at BC start) and
*rises* over training (60 → 72) rather than decreasing. That's
consistent with a competing attractor: PPO can lower episode return
more reliably by reducing velocity than by completing the task.

The M4 natural experiment makes this concrete. D's only "good" window
(success 0.10–0.30) ran from 700k to 1.7M env steps — exactly the
window where `eval/episode_length_mean` dipped from 201 to 192–200,
likely due to a macOS pybullet velocity-violation cleanup that
terminated episodes early. Early termination capped how much
`vel_penalty` could accumulate per episode, effectively shielding a
real, fast, BC-derived policy from the do-nothing trough. Once
episode lengths returned to 201 (1.7M+), success decayed back.

So before invoking the structural critic argument (Run I), test the
simpler reward-shape hypothesis: drop `vel_penalty` to 0 and see
whether D-base survives.

**Wandb manual name (suggested).**

```
J: vp=0 ablation on D-base - test if vel_penalty was the attractor - 512x512_sf1.2_sb100_psc20
```

**Auto-suffix** (with the new auto-naming helper from `_helpers.py`).

```
_hole_centroid_512x512_lr5e-5_cw2_lstd-2.7_bc300x60_sf1.2_sb100_psc20
```

(Note: no `_vp8` segment — that's the whole point of the run.)

**Command.** Launched on the L4.

```bash
python experiments/hang_obs_exp/scripts/train_privileged.py \
    --obs_mode hole_centroid \
    --max_act_vel 4.1 --log_std_init -2.7 \
    --lr 5e-5 --critic_warmup_rollouts 2 \
    --success_factor 1.2 --success_bonus 100 \
    --pre_settle_coef 20 --vel_penalty 0 \
    --bc_episodes 300 --bc_demos_only_success --bc_epochs 60 \
    --net_arch 512,512 \
    --total_env_steps 3000000 --seed 42 --cpu --use_wandb
```

**Decision tree after J lands.**

- J climbs to ≥0.5: reward shape was the issue. Stack with H (BC
  anchor) and/or I (critic warmup) for further gains; structural
  diagnosis was wrong (or at least, not the binding constraint).
- J plateaus at 0.10–0.30 like D/G: reward shape was a confound, not
  the cause. Run fixed Run I (critic warmup) on D-base with vp=8 to
  test the structural fix in isolation.
- J collapses below D/G: vp=0 makes things worse (e.g. unstable
  actions). Strong evidence for keeping shaping; pivot to H-stacking
  or AWAC.

**Outcome (2026-05-08, ~600k env steps before kill).**
**REJECTED.** `eval/success_rate` collapsed 0.62 → 0.05 by 250k and
flatlined at 0–0.05 through 600k+ — identical shape to D and G
(vp=8 baselines). `vel_penalty_sum` was correctly suppressed to ~0
throughout, but the policy still drifted away from the BC manifold.
Diagnostics:

- `train/explained_variance` reached 0.6–0.8 — the critic was
  *not* random, weakening the structural-critic argument that
  motivated Run I.
- `train/value_loss` peaked at 3.5+ during collapse window (3–10×
  other runs), indicating PPO updates with high-magnitude noisy
  advantages.
- `train/approx_kl` was 0.025–0.030 (3–5× D/H/I), `train/clip_fraction`
  ~0.30 — PPO was making aggressive updates per rollout.
- Reward decomposition (`eval_reward_decomp.py`) on J's checkpoint at
  600k showed the policy *did* approach the hole (per-step base rises
  from −0.25 to −0.11 by step 75) and then *drifted away*
  (back to −0.20 by step 200). Cloth ends episodes at adaptive_dist
  ~3.4 vs threshold ~0.7.
- The terminal-step `base = -FINAL_REWARD_MULT × dist` term reached
  −100 to −250 per failed episode. With per-step base contributing
  only ~−40 cumulatively, terminal step dominated ~85% of episode
  return — credit assignment was broken regardless of vel_penalty.

**Diagnosis (revised).** The dominant cause of BC erasure is
**reward concentration at the terminal step combined with
post-settle dependence**, not vel_penalty and not random V. PPO
sees a tiny per-step signal during the episode and a huge,
delayed, settling-physics-dependent terminal signal — credit
assignment fails, advantages are noisy in magnitude, and the
policy walks away from BC. This motivated the reward redesign
(see below) and Run K.

**Status.** Killed at ~600k. Wandb run kept for diagnostic
comparison.

---

### Run K — Dense reward redesign on D-base

**Hypothesis.** Replace the terminal-dominated reward with a
per-step dense distance signal so PPO sees a continuous "you're
getting closer" reward throughout the episode rather than waiting
for a delayed, settling-physics-dependent terminal spike. Two
changes from J (which is otherwise identical):

1. `--dist_reward_coef 1.0` adds `+1 / (1 + adaptive_dist)` per step.
   Cumulative ~+30 to +110 per episode, depending on closeness —
   comparable in magnitude to the +100 success_bonus.
2. `--final_reward_mult 50` (down from dedo default 400) compresses
   the terminal-base magnitude. Without this, drifted episodes pay
   −100 to −250 at terminal — far larger than the per-step signal —
   and PPO's gradient is still terminal-dominated.

`threading_bonus_coef = 0` (deferred until a non-centroid-distance
threading detector replaces the flaky `adaptive_dist < threshold`
check; see "Reward redesign" subsection).

`pre_settle_coef` is held at 20 (D-base default). Pre-settle only
fires at terminal, magnitude ~−10 to −20 for scripted demos at
psc=20 — modest backstop against ballistic-drop exploits, doesn't
crush per-step signal.

**Wandb manual name (suggested).**

```
K: dense reward redesign on D-base - dr=1.0, vp=0, frm=50 (rebalanced terminal)
```

**Auto-suffix.** Includes the new `_dr` and `_frm` tags from
`build_run_name_suffix`:

```
_hole_centroid_512x512_lr5e-5_cw2_lstd-2.7_bc300x60_sf1.2_sb100_psc20_dr1_frm50
```

**Command.** Launched on the L4 (per RUN-J protocol — M4 sim
early-termination bug is not confirmed fixed).

```bash
python experiments/hang_obs_exp/scripts/train_privileged.py \
    --wandb_run_name 'K: dense reward redesign on D-base - dr=1.0, vp=0, frm=50 (rebalanced terminal)' \
    --obs_mode hole_centroid \
    --max_act_vel 4.1 --log_std_init -2.7 \
    --lr 5e-5 --critic_warmup_rollouts 2 \
    --success_factor 1.2 --success_bonus 100 \
    --pre_settle_coef 20 --vel_penalty 0 \
    --dist_reward_coef 1.0 --threading_bonus_coef 0 \
    --final_reward_mult 50 \
    --bc_episodes 300 --bc_demos_only_success --bc_epochs 60 \
    --net_arch 512,512 \
    --total_env_steps 3000000 --seed 42 --cpu --use_wandb
```

**Pre-launch verification.** Reward shape was visualized via
`eval_reward_decomp.py --scripted` under the K config before launch
(2026-05-08). Scripted demos under K-shape:
- 8/10 episodes succeed (matching vp=0 distribution)
- Net per-episode total: +150–200 for successes, +60–90 for
  near-miss failures (clean ~+85 cardinal gap from the success_bonus)
- Per-step `dist_reward` cumulative: +90–110 per episode —
  comparable in magnitude to the +100 terminal_shaping bonus
- Per-step contribution rises from 0.16 → 0.83 (peak at step 150)
  then settles to ~0.65 — the cloth-on-hole "stay here" signal is
  visible and continuous

**Status.** TBD (about to launch).

---

### Run I — Demo-V-warmup on D-base

**Hypothesis** (the strongest theoretical claim of the series).
Vanilla PPO+BC fails because **the critic is wrong**. After BC
pretrain, V(s) is random. PPO's gradient `∇log π · A(s,a)` with
`A = Q - V` is therefore noise; the actor faithfully follows that
noise away from the BC manifold before V can catch up. Fix the critic
at the start: pretrain V on Monte Carlo discounted returns from the
demo trajectories themselves. After 50 epochs of MSE on `(s, G_t)`
where `G_t = sum_{i>=t} gamma^(i-t) * r_i` is the demo MC return, V
outputs sensible values on BC-visited states from PPO step 0.

This is the structural fix; the runs A–H were stacking patches on a
broken initial condition.

**Wandb manual name.**

```
I: Demo-V-warmup on D-base - 50 epochs (critic seeded on demo MC returns) - 512x512_sf1.2_sb100_vp8_psc20
```

**Auto-suffix.** Same as D / F / G / H:

```
_512x512_sf1.2_sb100_vp8_psc20
```

**Command.** Launched manually via tmux on the L4.
`--critic_warmup_rollouts 0` because rollout-based warmup is redundant
once V is fit on demos.

```bash
python experiments/hang_obs_exp/scripts/train_privileged.py \
    --obs_mode hole_centroid \
    --max_act_vel 4.1 --log_std_init -2.7 \
    --lr 5e-5 --critic_warmup_rollouts 0 \
    --success_factor 1.2 --success_bonus 100 \
    --pre_settle_coef 20 --vel_penalty 8 \
    --bc_episodes 300 --bc_demos_only_success --bc_epochs 60 \
    --net_arch 512,512 \
    --critic_warmup_demo_epochs 50 \
    --critic_warmup_demo_lr 3e-4 \
    --critic_warmup_demo_batch_size 256 \
    --total_env_steps 3000000 --seed 42 --cpu --use_wandb
```

**Code.**

- Pretrain function at [scripts/_critic_warmup_demos.py](scripts/_critic_warmup_demos.py).
- Per-step rewards persisted in demo pkls at
  [train_privileged.py:798-803](scripts/train_privileged.py#L798-L803).
- CLI knobs at [train_privileged.py:193-209](scripts/train_privileged.py#L193-L209).
- Wired up at [train_privileged.py:1024-1050](scripts/train_privileged.py#L1024-L1050).

**Implementation notes.**

- Updates only critic-side params: `policy.mlp_extractor.value_net`
  (critic trunk) and `policy.value_net` (head). For SB3 `MlpPolicy`
  with list `net_arch`, actor and critic are separate networks within
  `MlpExtractor`, so the actor is untouched by warmup.
- Bootstraps `vec_normalize.ret_rms` with demo running returns so PPO's
  reward normalization at startup uses demo-scale stats.
- Targets are normalized with the same per-step normalize+clip that
  VecNormalize applies during PPO, so V's outputs are in the right
  scale.

**Outcome.** TBD. Watch `critic_warmup/mse` (should converge during
the 50 epochs to roughly 0.01 to 0.1 in normalized return space) and
`train/explained_variance` early in PPO (should start meaningfully
positive instead of near 0).

**Status.** First launch (2026-05-06) crashed during the demo-V
warmup step: `policy.mlp_extractor.forward_critic(...)` doesn't
exist in the installed SB3 version. Fixed in
`_critic_warmup_demos.py` by switching to the version-agnostic
`_, latent_vf = policy.mlp_extractor(features)` tuple unpack.

Rerun (2026-05-07) is **running and showing real recovery**:
post-BC drop to ~0.03 by 200k as expected, holds through 500k,
then `eval/success_rate` climbs to 0.10–0.23 by 750k. This is
comparable to D's M4 peak of 0.30 at 1.5M — possibly the demo-V
warmup is pulling the recovery curve earlier. Worth letting run
to ~3M steps. Kept running in parallel with K.

---

## Cross-cutting findings

1. **The post-BC cliff is universal.** Every run A–E followed the same
   shape: BC pretrain → eval success at 0.5 to 0.77 → fast collapse to
   0.10 to 0.30 over the first 100k to 300k env steps → noisy plateau.
   Variations in lr / warmup / reward-shape / capacity move *where*
   the plateau sits, not *whether* it happens.

2. **Bigger net retains more.** D (512x512) > A / C (256x256) at the
   post-BC starting point. Capacity helps BC fit, which gives a higher
   ceiling.

3. **Conservative lr is not the answer.** A had the most-conservative
   lr/warmup combo and the worst collapse. Capping per-rollout drift
   (F's clip_range / n_epochs / target_kl) doesn't separate from the
   no-cap baseline (G) either. The lever isn't update *magnitude*.

4. **The remaining hypothesis is structural.** PPO with a random V at
   start emits noise advantages that mislead the actor. H tests "fix
   the actor with BC anchor"; I tests "fix the critic with demo MC
   warmup". These are the principled fixes the rest of the sweep was
   working around.

---

## Knobs added during this sweep

The following CLI flags were added to
`scripts/train_privileged.py` over the course of the work:

| Flag                                | Added in / for | Default | Purpose                                          |
| ----------------------------------- | -------------- | ------- | ------------------------------------------------ |
| `--ppo_clip_range`                  | F              | 0.2     | PPO ratio clip (smaller = less per-update drift) |
| `--ppo_epochs`                      | F              | 10      | PPO grad epochs per rollout                      |
| `--ppo_target_kl`                   | F              | None    | KL early-stop on PPO update                      |
| `--bc_anchor_batches`               | H              | 0       | BC MSE replay batches per rollout                |
| `--bc_anchor_batch_size`            | H              | 256     | Anchor batch size                                |
| `--bc_anchor_lr`                    | H              | 1e-4    | Anchor optimizer lr                              |
| `--critic_warmup_demo_epochs`       | I              | 0       | Pretrain V on demo MC returns (epochs)           |
| `--critic_warmup_demo_lr`           | I              | 3e-4    | Critic warmup lr                                 |
| `--critic_warmup_demo_batch_size`   | I              | 256     | Critic warmup batch size                         |

The pre-existing `--critic_warmup_rollouts` (rollout-based actor-freeze
warmup) is orthogonal to `--critic_warmup_demo_epochs` (offline V
pretrain on demos) and remains available.

---

## Demo pkl format change

As of the I-series, demo pkls written by `_collect_demo_rollouts`
include a new `rewards` field (per-step raw rewards as
`np.ndarray`). Required for `--critic_warmup_demo_epochs > 0`. Legacy
pkls (pre-update collections, including A's `scripted_demos/`) lack
this field; `_load_manual_demos` warns and disables critic warmup if
any loaded pkl is missing it. To use older demo dirs with critic
warmup, re-collect via `--bc_episodes 300` on a fresh run, or write a
small migration script.

---

## Reward redesign (2026-05-08)

Triggered by Run J's rejection. Rather than continue patching PPO's
machinery (lr, warmup, anchor, critic-warmup), we repaired the
reward shape directly. Three new `PrivilegedObsWrapper` constructor
parameters (and matching `train_privileged.py` flags):

- `--dist_reward_coef <coef>`: per-step dense distance reward.
  `reward += coef / (1 + adaptive_dist)` every step (including
  terminal). Smooth, monotonic, peaks at the hole. Doesn't depend
  on a discrete threading detector. Suggested 0.5–2.0; cumulative-
  over-200-steps is comparable to `success_bonus`.
- `--threading_bonus_coef <coef>`: per-step bonus when
  `adaptive_dist < threshold`. **Caveat**: detection inherits the
  same flakiness as the terminal success check (centroid distance,
  not topological). Left at 0 until a non-distance threading
  detector (peg-z containment or winding number) replaces the
  centroid check.
- `--final_reward_mult <coef>`: override `DeformEnv.FINAL_REWARD_MULT`
  (default 400). Patches the class attribute the same way
  `--max_act_vel` does. Reducing to 50–100 compresses the terminal
  failure-tail magnitude so per-step `dist_reward` isn't crushed
  whenever cloth ends far from hole.

`info` dict gained `dist_reward`, `threading_bonus`, and
`adaptive_dist_step` keys per step; episode-end diagnostics gained
`rwd_diag/reward/dist_reward_sum`,
`rwd_diag/reward/threading_bonus_sum`,
`rwd_diag/task/threading_steps`, and
`rwd_diag/task/threading_fraction`. `eval_reward_decomp.py` was
extended to plot `+dist_reward` (cyan) and `+threading_bonus`
(olive) alongside the existing components, and accepts
`--override_dist_reward_coef`, `--override_threading_bonus_coef`,
and `--override_final_reward_mult` so reward shapes can be
prototyped on scripted demos before launching training.

**Why this is a methodologically clean change.** The K reward shape
was visualized via `eval_reward_decomp.py --scripted` *before*
launching, and verified to:
1. Provide a per-step signal of magnitude comparable to terminal
   (per-step dist_reward sum ~+95 vs terminal_shaping +100).
2. Cleanly separate successes (~+155 net) from near-miss failures
   (~+75 net) by the +100 success_bonus gap.
3. Not blow up the terminal step magnitude on canonical-expert
   trajectories.

This directly addresses the J-era diagnosis: PPO erased BC because
the per-step gradient was an order of magnitude smaller than the
terminal gradient, AND the terminal gradient depended on settling
physics the policy couldn't directly affect. Both are now fixed.

**Threading-metric work (deferred).** A proper threading detector
(peg-z between top and bottom hole-loop vertices, or winding
number of cloth loop around peg axis) is the next planned
infrastructure improvement. It (a) lets us re-enable
`threading_bonus_coef`, and (b) replaces the flaky terminal
success criterion that has known false negatives (cloth threads
during episode, slips during settle, marked failure). Deferred
behind K because dist_reward alone doesn't need it.

---

## Possible directions if I doesn't separate

1. **AWAC / IQL.** Offline-then-online algorithms designed for the
   "I have demos and want to improve from them" regime. Replay buffer
   naturally mixes BC and policy data. Larger code change but
   structurally different.
2. **Stack H + I** (Run J?). If neither alone works but the
   combination does, the problem is multi-causal.
3. **Switch envs.** Test the same recipe on a simpler dedo task. If
   PPO+BC works there but fails on HangProcCloth, the issue is task
   complexity / sparse reward, not the algorithm.

---

## Pixel observation runs (separate series)

The runs above all use the 18-dim `hole_centroid` privileged obs via
`scripts/train_privileged.py`. The pixel series uses
`scripts/train_pixels.py` instead — RGB image obs from a fixed camera,
with optional gripper proprio concatenated as a Dict obs (SB3
`MultiInputPolicy`) or pure-pixels (`CnnPolicy`, via `--no_grip`). The
features extractor is SB3's default `NatureCNN`; `--net_arch` controls
the post-CNN MLP head only.

Auto-suffix format from
[train_pixels.py:447-452](scripts/train_pixels.py#L447-L452):

```
<wandb name>_pixels<res><_grip|_pix>_net<arch>_sf<sf>[_sb<sb>][_fp<fp>][_vp<vp>][_ap<ap>][_psc<psc>]
```

Where `_grip` = MultiInputPolicy with gripper proprio, `_pix` = pure
CnnPolicy. Logdirs are written under
`logs/hang_obs_exp/pixgrip_<res>/` or `pixonly_<res>/`.

### Run P1 — First-pixels-CNN

**Hypothesis.** First end-to-end pixel attempt: can a 64x64 RGB CNN
learn HangProcCloth from scratch with the same BC-then-PPO recipe that
worked at the privileged-obs level? Uses MultiInputPolicy so the
gripper proprio is still available alongside the image (keeps the
input strictly richer than `--no_grip` pure-pixels). Larger BC budget
(500 episodes / 100 epochs vs 300 / 60 in the privileged sweep) to
give the CNN enough signal to fit. Conservative lr (1e-5) and longer
critic warmup (4 rollouts) given the much larger feature stack.

**Wandb manual name (suggested).**

```
P1: First-pixels-CNN - 64x64 RGB MultiInput (NatureCNN+512x512 head); lr=1e-5; BC 500/100 - pixels64_grip_net512x512_sf1_sb50_vp8_psc40
```

**Auto-suffix.**

```
_pixels64_grip_net512x512_sf1_sb50_vp8_psc40
```

**Command.** Launched on the L4 (GPU; CNN forward/backward needs it).

```bash
python experiments/hang_obs_exp/scripts/train_pixels.py \
    --cam_resolution 64 \
    --max_act_vel 4.1 --log_std_init -2.7 \
    --lr 1e-5 --critic_warmup_rollouts 4 \
    --success_factor 1.0 --success_bonus 50 \
    --pre_settle_coef 40 --vel_penalty 8 \
    --bc_episodes 500 --bc_demos_only_success --bc_epochs 100 \
    --net_arch 512,512 \
    --total_env_steps 3000000 --seed 42 --use_wandb
```

**Outcome.** TBD (first pixel run). Things to watch:

- BC eval success post-pretrain — sets the ceiling the CNN can fit
  500 demos to.
- Whether the post-BC cliff seen on privileged obs (A–E) reproduces on
  pixels, or whether the CNN's representation makes PPO's gradient
  better-behaved.
- `train/value_loss` and `train/explained_variance` — the larger
  feature stack means V has more to learn from scratch.

**Status.** Running / first attempt.
