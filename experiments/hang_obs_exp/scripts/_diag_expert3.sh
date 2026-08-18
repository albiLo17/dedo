#!/usr/bin/env bash
# Round 3: the dz fix, removing the post, and wider mesh variation.
#
# Round 2 established that moving the post with the hanger recovers the expert
# (C 10% -> E 45%). Two things were still outstanding:
#
#  1. deform_env applied --randomize_goal_dz to the HANGER ONLY, by explicit
#     design ("the tallrod stands on the floor"). That is the same fault as the
#     static one, re-created per episode: with dz in [-1, +1] the post ends up
#     as much as 0.8 units ABOVE the goal on the negative-dz half of episodes.
#     So F's 35% was still handicapped on roughly half its episodes. Fixed --
#     dz now moves both -- and F2 re-measures F on that fix.
#  2. Whether the post is worth keeping at all. It is a collision body the
#     cloth can snag on, and it occupies a large part of the rgb/pcd
#     observation where the real rig has no equivalent.
#
# All four use geometry F (real tip + grippers on the rig start pose), so the
# only differences are the ones named. Same --seed as rounds 1-2 throughout.
#
#   F2  fixed dz, post kept, mesh as before   <- the corrected baseline
#   G   F2 + no post
#   H   F2 + wide mesh variation
#   I   F2 + no post + wide mesh variation    <- the candidate for re-collection
#
# Mesh variation is capped at node_density 15 deliberately: the full_mesh obs
# holds 250 vertices, and density 16+ exceeds that. Going higher means widening
# the obs dimension, which forces a retrain of the state estimator and dynamics
# model too, so it is not a change to make incidentally.
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
  --peg_nominal 1.366 -1.266 6.213
  --deform_init_pos -1.66 3.10 6.88
)
MESH=(--node_density_range 9 15 --proc_cloth_size_range 0.5 3.2
      --proc_hole_frac_range 0.06 0.38)

run () {
  local id=$1; shift
  echo "[$(date +%H:%M:%S)] $id : $*"
  DISPLAY=:1 $PY -u experiments/hang_obs_exp/scripts/collect_bc_demos.py \
    --demos_dir "$OUT/$id" "${COMMON[@]}" "$@" > "$OUT/$id.log" 2>&1
  echo "[$(date +%H:%M:%S)] $id exit=$?"
}

run F2                          &
run G  --no_post                &
run H  "${MESH[@]}"             &
run I  --no_post "${MESH[@]}"   &
wait
echo "=== round 3 done: $OUT ==="
