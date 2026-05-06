#!/usr/bin/env bash
# Launch overnight runs on the M4 Pro Macbook (~12 cores, macOS).
# Activates the `dedo` conda env in each tmux session and uses
# caffeinate to prevent the laptop from sleeping.
#
# Slots used:
#   - Run B privileged: success_factor hypothesis            (CPU)
#   - Run C privileged: reward-shape hypothesis              (CPU)
#   - Run D privileged: MLP-capacity hypothesis              (CPU)
#
# Total CPU pressure with OMP_NUM_THREADS=2: ~9 cores. Comfortable.
#
# Run from repo root:  bash experiments/hang_obs_exp/scripts/launch_m4.sh
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$REPO_ROOT"

CONDA_ENV="dedo"

# Locate conda's profile script. `bash -lc` runs a non-interactive login
# shell; conda's init in ~/.bashrc/.zshrc is typically guarded by
# `[ -z "$PS1" ] && return` so it never runs in non-interactive shells.
# We have to source conda.sh explicitly. Try a few common locations.
detect_conda_sh() {
  local p
  if command -v conda >/dev/null 2>&1; then
    p="$(conda info --base 2>/dev/null)/etc/profile.d/conda.sh"
    [ -f "$p" ] && { echo "$p"; return; }
  fi
  for p in "$HOME/miniconda3/etc/profile.d/conda.sh" \
           "$HOME/anaconda3/etc/profile.d/conda.sh" \
           "$HOME/miniforge3/etc/profile.d/conda.sh" \
           "/opt/miniconda3/etc/profile.d/conda.sh" \
           "/opt/homebrew/Caskroom/miniconda/base/etc/profile.d/conda.sh" \
           "/opt/anaconda3/etc/profile.d/conda.sh"; do
    [ -f "$p" ] && { echo "$p"; return; }
  done
  echo ""
}
CONDA_SH="$(detect_conda_sh)"
if [ -z "$CONDA_SH" ]; then
  echo "ERROR: could not locate conda.sh. Edit launch_m4.sh and set CONDA_SH manually."
  exit 1
fi
echo "[init] using conda profile: $CONDA_SH"

# tmux command wrapper: spawn a login shell, source conda.sh, activate
# the env, set thread caps, cd to repo root, wrap python in caffeinate
# (macOS-only; prevents sleep) and run.
start() {
  local name="$1"; shift
  local cmd="$*"
  if tmux has-session -t "$name" 2>/dev/null; then
    echo "[skip] tmux session '$name' already exists — kill with 'tmux kill-session -t $name' first if you want to restart"
    return 0
  fi
  tmux new-session -d -s "$name" "bash -lc 'source \"$CONDA_SH\" && conda activate $CONDA_ENV && export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 && cd \"$REPO_ROOT\" && caffeinate -dimsu $cmd; echo; echo \"[$name] python exited; press enter to close session\"; read'"
  echo "[ok] launched tmux session '$name'"
}

# === Run B privileged: success_factor hypothesis =====================
start ppo_hc_B \
"python experiments/hang_obs_exp/scripts/train_privileged.py \
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
"python experiments/hang_obs_exp/scripts/train_privileged.py \
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
"python experiments/hang_obs_exp/scripts/train_privileged.py \
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
