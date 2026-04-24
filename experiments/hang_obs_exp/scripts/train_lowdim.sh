#!/usr/bin/env bash
# =============================================================================
# Train Condition C: LOW-DIM BASELINE (no camera)
# Observation: gripper positions only (cam_resolution=0), 12-dim vector.
# This is the "no visual info" baseline -- no hole or hanger visible at all.
# Uses PPO with MLP policy.
# =============================================================================
set -e

REPO_ROOT="$(cd "$(dirname "$0")/../../.." && pwd)"
PYTHON="${PYTHON:-python}"
LOGDIR="${LOGDIR:-${REPO_ROOT}/logs/hang_obs_exp/lowdim}"
mkdir -p "${LOGDIR}"

echo "=== Condition C: Low-dim (no camera) Baseline ==="
echo "Observation: gripper positions only (cam_resolution=0)"
echo "Logdir: ${LOGDIR}"

cd "${REPO_ROOT}"
${PYTHON} -m dedo.run_rl_sb3 \
    --env=HangProcCloth-v1 \
    --rl_algo=PPO \
    --logdir="${LOGDIR}" \
    --cam_resolution=0 \
    --num_envs=4 \
    --total_env_steps=2000000 \
    --log_save_interval=50 \
    --lr=3e-4 \
    --seed=42 \
    "$@"
