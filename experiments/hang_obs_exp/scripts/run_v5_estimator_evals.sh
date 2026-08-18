#!/usr/bin/env bash
# Estimator evaluations, run after the four BC policies are trained.
#
# Two separate questions, deliberately not conflated:
#   1. How accurate is the estimator itself?  -> eval_state_est_quality.py, on
#      the estimator's OWN held-out-topology validation split, in the units it
#      trained in, scored on the mesh. No policy involved.
#   2. Does the policy survive on estimated input? -> the `state` policy,
#      eval-only, with --use_state_estimator replacing the GT hole centroid.
#      Compared against that same policy's own GT-centroid final eval, so the
#      ONLY difference between the two rows is where the centroid came from.
#
# Both go through the SAME websocket serving path, so a number here is
# reproducible by the policy loop rather than being an in-process shortcut.
#
#   run_v5_estimator_evals.sh <se_run_dir> <state_policy_ckpt> <out_dir> <h5>
set -u

SE_RUN=${1:?state-est run dir (the dir containing config.yml + checkpoint-best)}
POLICY_CKPT=${2:?trained state-policy .pt}
OUT=$(readlink -f "${3:?output dir}")   # MUST be absolute: this script cds
H5=$(readlink -f "${4:?state-est h5}")  # into UniClothDiff and then into dedo,
                                        # so a relative path silently resolves
                                        # against whichever repo is current —
                                        # the server log redirect failed that
                                        # way and the server never started.
PORT=${PORT:-8123}

UCD=/juno/u/alberta/code/GCE/UniClothDiff
DEDO=/juno/u/alberta/code/GCE/dedo
PY=/juno/u/alberta/miniconda3/envs/dedo/bin/python
mkdir -p "$OUT"
OUT=$(readlink -f "$OUT")

echo "=== [$(date +%H:%M:%S)] starting serve_predictor on :$PORT"
cd "$UCD" || exit 1
# serve_predictor.py wants the RUN DIR (it reads config.yml, then resolves
# checkpoint-best/model itself). serve_pf_tracker.py instead wants
# <run>/checkpoint-best. Same-looking argument, different expectation.
WANDB_MODE=disabled nohup .venv/bin/python -u scripts/serve_predictor.py \
    --task state_est --checkpoint "$SE_RUN" --host 127.0.0.1 --port "$PORT" \
    > "$OUT/serve_predictor.log" 2>&1 &
SERVE_PID=$!
echo "    serve pid $SERVE_PID"
trap 'kill $SERVE_PID 2>/dev/null' EXIT

# wait for it to actually bind before firing clients at it
for _ in $(seq 1 60); do
  grep -qiE "serving|listening|running on|started" "$OUT/serve_predictor.log" && break
  sleep 5
done
sleep 5
tail -3 "$OUT/serve_predictor.log"

echo "=== [$(date +%H:%M:%S)] 1/2 estimator accuracy (own val split, occlusion sweep)"
cd "$DEDO" || exit 1
DISPLAY=:1 $PY -u experiments/hang_obs_exp/scripts/eval_state_est_quality.py \
    --h5 "$H5" --split validation --host 127.0.0.1 --port "$PORT" \
    --n_episodes 40 --frames_per_episode 4 \
    --drop_fracs 0,0.25,0.5,0.75 \
    --out "$OUT/state_est_quality.json" \
    > "$OUT/state_est_quality.log" 2>&1
echo "    exit=$?"

echo "=== [$(date +%H:%M:%S)] 2/2 policy with estimator in the loop"
DISPLAY=:1 WANDB_MODE=disabled $PY -u \
    experiments/hang_obs_exp/scripts/train_diffusion_bc.py \
    --demo_path "${DEMOS:-logs/hang_obs_exp/bc_demos_v5}" \
    --obs_mode state --state_key hole_centroid \
    --eval_only --resume "$POLICY_CKPT" \
    --logdir "$OUT/state_gse" \
    --n_final_eval_episodes 50 --success_metric legacy \
    --use_state_estimator 127.0.0.1:"$PORT" \
    > "$OUT/state_gse.log" 2>&1
echo "    exit=$?"

kill $SERVE_PID 2>/dev/null
echo "=== [$(date +%H:%M:%S)] estimator evals done"
