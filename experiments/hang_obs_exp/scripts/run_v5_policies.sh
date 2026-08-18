#!/usr/bin/env bash
# Train the four BC obs modes sequentially on one GPU, then make sure each one
# ends up with a final_eval_metrics.json no matter how training ended.
#
# Sequential on purpose: one 4090, two diffusion trainings do not fit. Every
# mode reads the SAME demo dir so the only thing differing between rows is the
# observation — train_diffusion_bc.py enforces that by parity-checking
# cam_resolution, max_act_vel, ctrl_freq, sim_freq, sim_steps_per_action,
# randomize_goal_radius and demo_speed across the dir.
#
# Training is time-boxed. A box that fires leaves numbered checkpoints
# (--save_every_epochs) but NO final eval, which would silently cost a row in
# the report — so evaluation is a SEPARATE step that runs off whatever
# checkpoint exists. That also means the eval settings are identical for every
# mode regardless of how its training ended.
#
#   run_v5_policies.sh <demo_dir> <out_root> <e_state> <e_pcd> <e_rgb> <e_mesh>
set -u

DEMOS=${1:?demo dir}
OUT=${2:?out root}
E_STATE=${3:-40}
E_PCD=${4:-12}
E_RGB=${5:-10}
E_MESH=${6:-12}
TRAIN_BOX=${TRAIN_BOX:-4200}     # 70 min per mode
N_EVAL=${N_EVAL:-50}

PY=/juno/u/alberta/miniconda3/envs/dedo/bin/python
TRAIN=experiments/hang_obs_exp/scripts/train_diffusion_bc.py
cd /juno/u/alberta/code/GCE/dedo || exit 1
mkdir -p "$OUT"

newest_ckpt () {   # newest usable weights for a mode, final preferred
  ls -t "$OUT/$1"/*/policy.pt "$OUT/$1"/*/*/policy.pt \
        "$OUT/$1"/*/policy_ep*.pt "$OUT/$1"/*/*/policy_ep*.pt 2>/dev/null | head -1
}
has_metrics () {
  ls "$OUT/$1"/*/final_eval_metrics.json "$OUT/$1"/*/*/final_eval_metrics.json \
     2>/dev/null | head -1
}

run_mode () {
  local mode=$1 epochs=$2 extra=${3:-}
  echo "=== [$(date +%H:%M:%S)] $mode : up to $epochs epochs (box ${TRAIN_BOX}s)"
  mkdir -p "$OUT/$mode"
  # shellcheck disable=SC2086
  DISPLAY=:1 WANDB_MODE=disabled timeout "$TRAIN_BOX" $PY -u "$TRAIN" \
    --demo_path "$DEMOS" --obs_mode "$mode" --logdir "$OUT/$mode" \
    --num_epochs "$epochs" --batch_size 64 --num_workers 4 \
    --eval_every_epochs 1000 --n_final_eval_episodes "$N_EVAL" \
    --save_every_epochs 2 --success_metric legacy \
    $extra > "$OUT/${mode}.log" 2>&1
  echo "    [$(date +%H:%M:%S)] $mode train exit=$?"

  if [ -z "$(has_metrics "$mode")" ]; then
    local ck; ck=$(newest_ckpt "$mode")
    if [ -n "$ck" ]; then
      echo "    [$(date +%H:%M:%S)] $mode: no eval from training; "\
"evaluating $ck separately"
      # shellcheck disable=SC2086
      DISPLAY=:1 WANDB_MODE=disabled timeout 2400 $PY -u "$TRAIN" \
        --demo_path "$DEMOS" --obs_mode "$mode" --logdir "$OUT/$mode" \
        --eval_only --resume "$ck" \
        --n_final_eval_episodes "$N_EVAL" --success_metric legacy \
        $extra >> "$OUT/${mode}.log" 2>&1
      echo "    [$(date +%H:%M:%S)] $mode eval exit=$?"
    else
      echo "    [$(date +%H:%M:%S)] $mode: NO CHECKPOINT — nothing to evaluate"
    fi
  fi

  local mj; mj=$(has_metrics "$mode")
  if [ -n "$mj" ]; then
    $PY - "$mj" <<'PYEOF'
import json, sys
d = json.load(open(sys.argv[1]))
k = [x for x in d if x.startswith('final_eval/')]
print('   ', {x.split('/')[-1]: (round(d[x], 4) if isinstance(d[x], float) else d[x])
              for x in sorted(k) if 'success' in x or 'reward' in x})
PYEOF
  else
    echo "    NO METRICS for $mode — see $OUT/${mode}.log"
  fi
}

# state first: cheapest, and it is the row the estimator-in-the-loop
# comparison replaces, so a failure here invalidates that experiment too.
run_mode state "$E_STATE" "--state_key hole_centroid"
run_mode pcd   "$E_PCD"
run_mode rgb   "$E_RGB"
run_mode mesh  "$E_MESH"

echo "=== [$(date +%H:%M:%S)] all four modes done"
