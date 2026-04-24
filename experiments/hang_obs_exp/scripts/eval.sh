#!/usr/bin/env bash
# =============================================================================
# Evaluate a saved checkpoint (camera conditions: full, partial, lowdim).
#
# Usage:
#   bash experiments/hang_obs_exp/scripts/eval.sh <checkpoint_dir> [<output_dir>]
#
# Example:
#   bash experiments/hang_obs_exp/scripts/eval.sh \
#       logs/hang_obs_exp/full/PPO_260412_231040_HangProcCloth-v1 \
#       logs/hang_obs_exp/full/eval_videos
#
# Note: for privileged conditions (hole_centroid, hole_vertices, full_mesh)
#       use eval_privileged.py instead (the wrapper must be re-applied).
# =============================================================================
set -e

REPO_ROOT="$(cd "$(dirname "$0")/../../.." && pwd)"
PYTHON="${PYTHON:-python}"
CHECKPT="${1:-}"
OUTDIR="${2:-/tmp/dedo_eval}"

if [[ -z "${CHECKPT}" ]]; then
    echo "Usage: bash experiments/hang_obs_exp/scripts/eval.sh <checkpoint_dir> [<output_dir>]"
    exit 1
fi

# Resolve relative checkpoint path against repo root
if [[ "${CHECKPT}" != /* ]]; then
    CHECKPT="${REPO_ROOT}/${CHECKPT}"
fi

echo "=== Evaluating checkpoint: ${CHECKPT} ==="
echo "Output dir: ${OUTDIR}"
mkdir -p "${OUTDIR}"

cd "${REPO_ROOT}"
${PYTHON} -m dedo.run_rl_sb3 \
    --env=HangProcCloth-v1 \
    --play \
    --load_checkpt="${CHECKPT}" \
    --logdir="${OUTDIR}" \
    --cam_resolution=200 \
    "${@:3}"
