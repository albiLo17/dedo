#!/usr/bin/env bash
# Second half of the expert diagnosis: the support-rod fault.
#
# Round 1 (_diag_expert.sh) showed the hardcoded hover offset is real but only
# part of it -- C, which moves the cloth with the peg, recovered 4.8% -> 16.7%
# against A's 42%. Measuring the scene explained the rest:
#
#   scene   hanger arms top   ROD top   goal z   rod above goal
#   v5      8.753             8.002     8.200    -0.198  (post ends under the peg)
#   v6      6.766             8.002     6.213    +1.789  (post spikes THROUGH it)
#
# tallrod.urdf is 0.8 m long at globalScaling=10 = exactly 8.0 sim units, which
# is why the shipped preset paired hanger z=8.0 with rod z=0. Moving the hanger
# to the measured tip while leaving the rod grounded left a bare post standing
# 80 mm above the target. Both B and C had it, and so does the shipped v6
# DATASET -- every v6 demo was collected against a spiked peg.
#
# collect_bc_demos.py --peg_nominal now translates the rod too. E and F are C
# and D re-run with that fix; everything else is identical to round 1, same
# --seed, so E is comparable to C episode-for-episode.
#
#   E  peg + cloth + rod moved            <- the true pure translation
#   F  peg + rod moved, cloth on the rig  <- the geometry we actually want
#
# E recovering to ~A confirms the two faults together are the whole story.
set -u
DEDO=/juno/u/alberta/code/GCE/dedo
PY=/juno/u/alberta/miniconda3/envs/dedo/bin/python
S=/tmp/user/25330/claude-25330/-juno-u-alberta-code-GCE/0621e330-355e-4efb-aa91-380ba1fc4e32/scratchpad
OUT=${1:-$S/diag_expert}
N=${2:-60}
NVIZ=${3:-20}
mkdir -p "$OUT"
cd "$DEDO" || exit 1

COMMON=(
  --n_demos "$N" --no_only_success --success_metric legacy --success_factor 1.2
  --chain_fraction 0.0
  --seed 2026
  --demo_speed 0.25 --ctrl_freq 15.0 --sim_freq 500 --max_act_vel 4.0
  --randomize_goal_radius 2.0 --randomize_goal_dz 1.0
  --start_jitter_xy 2.0 --start_jitter_z 1.0
  --cam_viewmat 14.0 -24.0 316.5 0.0 0.0 5.5
  --cam_roll_deg 6.0 --cam_jitter_roll_deg 4.0
  --cam_resolution 96 --pcd_n_points 512
  --debug_viz_first_n "$NVIZ" --debug_viz_first_n_failed 0
)
REAL_TIP=(1.366 -1.266 6.213)

run () {
  local id=$1; shift
  echo "[$(date +%H:%M:%S)] $id : $*"
  DISPLAY=:1 $PY -u experiments/hang_obs_exp/scripts/collect_bc_demos.py \
    --demos_dir "$OUT/$id" "${COMMON[@]}" "$@" > "$OUT/$id.log" 2>&1
  echo "[$(date +%H:%M:%S)] $id exit=$?"
}

run E --peg_nominal "${REAL_TIP[@]}" --deform_init_pos 1.366 1.934 6.013 &
run F --peg_nominal "${REAL_TIP[@]}" --deform_init_pos -1.66 3.10  6.88  &
wait
echo "=== round 2 done: $OUT ==="
