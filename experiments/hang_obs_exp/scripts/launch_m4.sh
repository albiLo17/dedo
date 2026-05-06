#!/usr/bin/env bash
# Launch overnight runs on the M4 Pro Macbook (~12 cores).
#
# Slots used:
#   - Run B privileged: success_factor hypothesis            (CPU)
#   - Run C privileged: reward-shape hypothesis              (CPU)
#   - Run D privileged: MLP-capacity hypothesis              (CPU)
#
# Total CPU pressure with OMP_NUM_THREADS=2: ~3 cores × 3 = ~9 cores.
# Comfortable on M4 Pro.
#
# Run from repo root:  bash experiments/hang_obs_exp/scripts/launch_m4.sh
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$REPO_ROOT"

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

# === Run B privileged: success_factor hypothesis =====================
start ppo_hc_B \
"caffeinate -dimsu python experiments/hang_obs_exp/scripts/train_privileged.py \
    --obs_mode hole_centroid \
    --max_act_vel 4.1 --log_std_init -2.7 \
    --lr 2e-5 --critic_warmup_rollouts 4 \
    --success_factor 0.8 --success_bonus 100 \
    --pre_settle_coef 20 --vel_penalty 8 \
    --bc_episodes 300 --bc_demos_only_success --bc_epochs 60 \
    --net_arch 256,256 \
    --total_env_steps 3000000 --seed 42 --cpu --use_wandb"

# === Run C privileged: reward-shape hypothesis =======================
start ppo_hc_C \
"caffeinate -dimsu python experiments/hang_obs_exp/scripts/train_privileged.py \
    --obs_mode hole_centroid \
    --max_act_vel 4.1 --log_std_init -2.7 \
    --lr 2e-5 --critic_warmup_rollouts 4 \
    --success_factor 1.2 --success_bonus 25 \
    --pre_settle_coef 80 --vel_penalty 8 \
    --bc_episodes 300 --bc_demos_only_success --bc_epochs 60 \
    --net_arch 256,256 \
    --total_env_steps 3000000 --seed 42 --cpu --use_wandb"

# === Run D privileged: MLP-capacity hypothesis =======================
start ppo_hc_D \
"caffeinate -dimsu python experiments/hang_obs_exp/scripts/train_privileged.py \
    --obs_mode hole_centroid \
    --max_act_vel 4.1 --log_std_init -2.7 \
    --lr 5e-5 --critic_warmup_rollouts 2 \
    --success_factor 1.2 --success_bonus 100 \
    --pre_settle_coef 20 --vel_penalty 8 \
    --bc_episodes 300 --bc_demos_only_success --bc_epochs 60 \
    --net_arch 512,512 \
    --total_env_steps 3000000 --seed 42 --cpu --use_wandb"

echo
echo "All M4 sessions launched. Inspect with:  tmux ls"
echo "Attach with:  tmux attach -t <name>     (Ctrl-b d to detach)"
