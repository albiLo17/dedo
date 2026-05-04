#!/usr/bin/env bash
# Run the three privileged conditions sequentially with wandb logging.
set -eo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/../../.." && pwd)"
PYTHON="${PYTHON:-python}"
LOG_ROOT="${REPO_ROOT}/logs/hang_obs_exp"

cd "${REPO_ROOT}"

for cond in hole_centroid hole_vertices full_mesh; do
    mkdir -p "${LOG_ROOT}/${cond}"
    logfile="${LOG_ROOT}/${cond}/train.log"
    echo "=== Starting ${cond} → ${logfile} ==="
    "${PYTHON}" experiments/hang_obs_exp/scripts/train_privileged.py \
        --obs_mode "${cond}" --use_wandb "$@" 2>&1 | tee "${logfile}"
    echo "=== Finished ${cond} ==="
done

echo "All privileged conditions done."
