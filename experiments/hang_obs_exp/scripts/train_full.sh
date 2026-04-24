#!/usr/bin/env bash
# =============================================================================
# Train Condition A: FULL OBSERVABILITY
# Camera: front-angled (yaw=45, pitch=-25), hole + hanger both visible.
# Uses PPO with RGB pixel observations (64x64, flattened MLP).
# =============================================================================
set -e

REPO_ROOT="$(cd "$(dirname "$0")/../../.." && pwd)"
PYTHON="${PYTHON:-python}"
LOGDIR="${LOGDIR:-${REPO_ROOT}/logs/hang_obs_exp/full}"
mkdir -p "${LOGDIR}"

echo "=== Condition A: Full Observability Training ==="
echo "Camera: yaw=45, pitch=-25 (front-angled, hole+hanger visible)"
echo "Logdir: ${LOGDIR}"

cd "${REPO_ROOT}"
${PYTHON} -m dedo.run_rl_sb3 \
    --env=HangProcCloth-v1 \
    --rl_algo=PPO \
    --logdir="${LOGDIR}" \
    --cam_resolution=64 \
    --cam_viewmat 9.0 -25.0 45.0 0.0 0.5 6.5 \
    --flat_obs \
    --num_envs=4 \
    --total_env_steps=2000000 \
    --log_save_interval=50 \
    --lr=3e-4 \
    --seed=42 \
    "$@"
