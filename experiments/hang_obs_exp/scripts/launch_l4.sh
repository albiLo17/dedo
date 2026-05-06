#!/usr/bin/env bash
# Launch overnight runs on the L4 GCP instance (12 vCPUs, 1× L4 GPU).
#
# Slots used:
#   - existing PPO pixels CNN run keeps running (assumed already in tmux)
#   - 1 NEW PPO pixels run with the consolidated recipe       (GPU)
#   - Run A privileged: BC-preservation hypothesis            (CPU)
#   - Run E privileged: kitchen-sink best-guess               (CPU)
#
# Total CPU pressure with OMP_NUM_THREADS=2:
#   ~3 cores × 4 runs (CNN+vision share GPU, MLPs are pure CPU) = ~12 cores
#   Fits g2-standard-12 with a small margin.
#
# Run from repo root:  bash experiments/hang_obs_exp/scripts/launch_l4.sh
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$REPO_ROOT"

# Cap PyTorch parallelism so 4 simultaneous runs don't trash each other.
export OMP_NUM_THREADS=2
export MKL_NUM_THREADS=2

start() {
  local name="$1"; shift
  if tmux has-session -t "$name" 2>/dev/null; then
    echo "[skip] tmux session '$name' already exists — kill it first if you want to restart"
    return 0
  fi
  tmux new-session -d -s "$name" "$*"
  echo "[ok] launched tmux session '$name'"
}

# === New PPO pixels run with consolidated recipe ====================
start ppo_pix_new \
"caffeinate -dimsu python experiments/hang_obs_exp/scripts/train_pixels.py \
    --max_act_vel 4.1 --log_std_init -2.7 \
    --lr 1e-5 \
    --critic_warmup_rollouts 4 \
    --success_factor 1.0 --success_bonus 50 \
    --pre_settle_coef 40 --vel_penalty 8 \
    --cam_resolution 64 \
    --net_arch 512,512 \
    --bc_episodes 500 --bc_demos_only_success --bc_epochs 100 \
    --total_env_steps 3000000 --seed 42 --use_wandb"

# === Run A privileged: lr/warmup hypothesis =========================
start ppo_hc_A \
"python experiments/hang_obs_exp/scripts/train_privileged.py \
    --obs_mode hole_centroid \
    --max_act_vel 4.1 --log_std_init -2.7 \
    --lr 5e-6 --critic_warmup_rollouts 8 \
    --success_factor 1.2 --success_bonus 100 \
    --pre_settle_coef 20 --vel_penalty 8 \
    --bc_episodes 300 --bc_demos_only_success --bc_epochs 60 \
    --net_arch 256,256 \
    --total_env_steps 3000000 --seed 42 --cpu --use_wandb"

# === Run E privileged: kitchen-sink best-guess =======================
start ppo_hc_E \
"python experiments/hang_obs_exp/scripts/train_privileged.py \
    --obs_mode hole_centroid \
    --max_act_vel 4.1 --log_std_init -2.7 \
    --lr 1e-5 --critic_warmup_rollouts 6 \
    --success_factor 0.8 --success_bonus 50 \
    --pre_settle_coef 40 --vel_penalty 8 \
    --bc_episodes 300 --bc_demos_only_success --bc_epochs 60 \
    --net_arch 512,512 \
    --total_env_steps 3000000 --seed 42 --cpu --use_wandb"

echo
echo "All L4 sessions launched. Inspect with:  tmux ls"
echo "Attach with:  tmux attach -t <name>     (Ctrl-b d to detach)"
