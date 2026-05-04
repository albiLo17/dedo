#!/usr/bin/env bash
# Run all 6 conditions in parallel.
# Logs go to logs/hang_obs_exp/<condition>/  as usual.
#
# Usage:
#   bash experiments/hang_obs_exp/scripts/train_all.sh          # all 6
#   CONDS="full partial" bash experiments/hang_obs_exp/scripts/train_all.sh  # subset
#
# Each job's stdout/stderr is tee'd to logs/hang_obs_exp/<condition>/train.log
# so you can follow any run with:
#   tail -f logs/hang_obs_exp/full/train.log
set -e

REPO_ROOT="$(cd "$(dirname "$0")/../../.." && pwd)"
PYTHON="${PYTHON:-python}"
LOG_ROOT="${REPO_ROOT}/logs/hang_obs_exp"

CONDS="${CONDS:-full partial lowdim hole_centroid hole_vertices full_mesh}"

pids=()

for cond in $CONDS; do
    mkdir -p "${LOG_ROOT}/${cond}"
    logfile="${LOG_ROOT}/${cond}/train.log"
    case "$cond" in
        full)
            bash "${REPO_ROOT}/experiments/hang_obs_exp/scripts/train_full.sh" \
                2>&1 | tee "$logfile" &
            ;;
        partial)
            bash "${REPO_ROOT}/experiments/hang_obs_exp/scripts/train_partial.sh" \
                2>&1 | tee "$logfile" &
            ;;
        lowdim)
            bash "${REPO_ROOT}/experiments/hang_obs_exp/scripts/train_lowdim.sh" \
                2>&1 | tee "$logfile" &
            ;;
        hole_centroid|hole_vertices|full_mesh)
            "${PYTHON}" "${REPO_ROOT}/experiments/hang_obs_exp/scripts/train_privileged.py" \
                --obs_mode "$cond" --use_wandb \
                2>&1 | tee "$logfile" &
            ;;
        *)
            echo "Unknown condition: $cond" >&2
            exit 1
            ;;
    esac
    pids+=($!)
    echo "Started $cond  (pid ${pids[-1]})  → $logfile"
done

echo ""
echo "All ${#pids[@]} jobs running. Waiting..."
failed=0
for i in "${!pids[@]}"; do
    cond_arr=($CONDS)
    if wait "${pids[$i]}"; then
        echo "✓ ${cond_arr[$i]} finished"
    else
        echo "✗ ${cond_arr[$i]} FAILED (exit $?)"
        failed=$((failed + 1))
    fi
done

echo ""
[ "$failed" -eq 0 ] && echo "All conditions done." || echo "$failed condition(s) failed."
exit "$failed"
