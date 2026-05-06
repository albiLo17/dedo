#!/usr/bin/env bash
# Launch overnight runs on the L4 GCP instance (12 vCPUs, 1× L4 GPU, Linux).
# Activates the `dedo38` conda env in each tmux session.
#
# Slots used:
#   - existing PPO pixels CNN run keeps running (assumed already in tmux)
#   - 1 NEW PPO pixels run with the consolidated recipe       (GPU)
#   - Run A privileged: BC-preservation hypothesis            (CPU)
#   - Run E privileged: kitchen-sink best-guess               (CPU)
#
# Total CPU pressure with OMP_NUM_THREADS=2: ~12 cores. Fits g2-standard-12.
#
# Run from repo root:  bash experiments/hang_obs_exp/scripts/launch_l4.sh
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$REPO_ROOT"

CONDA_ENV="dedo38"

# Locate conda's profile script. `bash -lc` runs a non-interactive login
# shell; conda's init in ~/.bashrc is usually guarded by `[ -z "$PS1" ]
# && return` so it never runs in non-interactive shells. We have to
# source conda.sh explicitly. Try a few common install locations.
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
           "/opt/conda/etc/profile.d/conda.sh"; do
    [ -f "$p" ] && { echo "$p"; return; }
  done
  echo ""
}
CONDA_SH="$(detect_conda_sh)"
if [ -z "$CONDA_SH" ]; then
  echo "ERROR: could not locate conda.sh. Edit launch_l4.sh and set CONDA_SH manually."
  exit 1
fi
echo "[init] using conda profile: $CONDA_SH"

# tmux command wrapper: spawn a login shell, source conda.sh, activate
# the env, set thread caps, cd to repo root, run the given python.
start() {
  local name="$1"; shift
  local cmd="$*"
  if tmux has-session -t "$name" 2>/dev/null; then
    echo "[skip] tmux session '$name' already exists — kill with 'tmux kill-session -t $name' first if you want to restart"
    return 0
  fi
  tmux new-session -d -s "$name" "bash -lc 'source \"$CONDA_SH\" && conda activate $CONDA_ENV && export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 && cd \"$REPO_ROOT\" && $cmd; echo; echo \"[$name] python exited; press enter to close session\"; read'"
  echo "[ok] launched tmux session '$name'"
}

# === New PPO pixels run with consolidated recipe ====================
start ppo_pix_new \
"python experiments/hang_obs_exp/scripts/train_pixels.py \
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
