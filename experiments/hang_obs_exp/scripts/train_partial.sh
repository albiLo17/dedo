#!/usr/bin/env bash
# =============================================================================
# Train Condition B: PARTIAL OBSERVABILITY
# Camera: near-top-down (pitch=-82), hole-to-hanger relation is lost.
# Uses PPO with RGB pixel observations (64x64, flattened MLP).
# =============================================================================
set -e

REPO_ROOT="$(cd "$(dirname "$0")/../../.." && pwd)"
PYTHON="${PYTHON:-python}"
LOGDIR="${LOGDIR:-${REPO_ROOT}/logs/hang_obs_exp/partial}"
mkdir -p "${LOGDIR}"

echo "=== Condition B: Partial Observability Training ==="
echo "Camera: pitch=-82 (near top-down, hole-hanger depth relation lost)"
echo "Logdir: ${LOGDIR}"

cd "${REPO_ROOT}"
${PYTHON} -m dedo.run_rl_sb3 \
    --env=HangProcCloth-v1 \
    --rl_algo=PPO \
    --logdir="${LOGDIR}" \
    --cam_resolution=64 \
    --cam_viewmat 8.0 -82.0 45.0 0.0 0.5 8.0 \
    --flat_obs \
    --num_envs=4 \
    --total_env_steps=2000000 \
    --log_save_interval=50 \
    --lr=3e-4 \
    --seed=42 \
    --use_wandb \
    "$@"
