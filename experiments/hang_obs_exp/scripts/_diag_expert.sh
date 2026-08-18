#!/usr/bin/env bash
# Why the scripted expert's threading success fell 48% -> 11% when the peg moved
# to the measured real hanger tip.
#
# A peg move should be a workspace translation and change nothing. It did, so
# something in the expert is anchored to absolute coordinates. The suspect is
# build_hole_aware_waypoints (_helpers.py:69), whose hover/thread/hold targets
# are constant offsets from the hanger:
#
#   scene   peg z   hover target z   cloth must move
#   v5      8.20    10.00            +1.89  (lift)
#   v6      6.21     8.01            -0.10  (no lift)
#
# Four geometries separate "the peg moved" from "the peg moved and the cloth
# did not". Everything else is identical, including --seed, so the cloth mesh
# and every jitter draw are paired episode-for-episode across conditions and
# the only difference is where the peg and the cloth start.
#
#   A  v5 control             peg [0,0,8.2],  cloth [0, 3.2, 8.0]
#   B  v6 as shipped          peg real tip,   cloth [0, 3.2, 8.0]
#   C  pure translation       peg real tip,   cloth shifted by the same delta
#   D  real-aligned start     peg real tip,   cloth set so the anchors land on
#                                             the measured real start pose
#
# D's cloth start is [-1.66, 3.10, 6.88]: the real anchor midpoint
# [-1.70, 3.20, 9.62] minus the measured anchor-to-deform_init_pos offset
# [-0.04, 0.10, 2.74] (from 120 v6 demos).
#
# Reading it: C recovering to ~48% confirms the hardcoded hover offset. C NOT
# recovering means the offsets are not the story and the next suspect is phase
# duration -- the three phases are fixed at 1.4/1.0/0.6 s and the v6 descent is
# ~3x longer, which the hole-height trace will show as the cloth never reaching
# its waypoint before the phase ends.
#
# --no_only_success is deliberate: it makes the denominator exactly N attempts
# and writes a pkl for failures too, so the traces cover the failures -- which
# are the episodes we actually need to look at. It also means nothing is ever
# "dropped", so --debug_viz_first_n_failed can never fire; videos come from
# --debug_viz_first_n instead and are paired to outcomes via the success flags
# in the matching demo_NNN.pkl.
#
# Randomization is left at the v6 settings rather than switched off, so A and B
# reproduce the historical 48% / 11% rather than measuring a different quantity.
# All four share it, so the geometry contrast is still clean.
set -u
DEDO=/juno/u/alberta/code/GCE/dedo
PY=/juno/u/alberta/miniconda3/envs/dedo/bin/python
S=/tmp/user/25330/claude-25330/-juno-u-alberta-code-GCE/0621e330-355e-4efb-aa91-380ba1fc4e32/scratchpad
OUT=${1:-$S/diag_expert}
N=${2:-60}
NVIZ=${3:-20}
mkdir -p "$OUT"
cd "$DEDO" || exit 1

# Exactly the v6 collection settings, minus obs fidelity: success is decided by
# physics, so cam_resolution/pcd_n_points only cost time here.
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
    --demos_dir "$OUT/$id" "${COMMON[@]}" "$@" \
    > "$OUT/$id.log" 2>&1
  echo "[$(date +%H:%M:%S)] $id exit=$?"
}

run A --peg_nominal 0 0 8.2               --deform_init_pos 0     3.2   8.0   &
run B --peg_nominal "${REAL_TIP[@]}"      --deform_init_pos 0     3.2   8.0   &
run C --peg_nominal "${REAL_TIP[@]}"      --deform_init_pos 1.366 1.934 6.013 &
run D --peg_nominal "${REAL_TIP[@]}"      --deform_init_pos -1.66 3.10  6.88  &
wait

echo
echo "=== expert success by geometry (n=$N attempts each) ==="
$PY - "$OUT" <<'PYEOF'
import glob, os, pickle, sys, math
out = sys.argv[1]
print(f'{"id":3s} {"peg tip":>22s} {"cloth start":>22s} {"n":>4s} '
      f'{"legacy":>16s} {"hanging":>8s}')
for gid in 'ABCD':
    fs = sorted(glob.glob(os.path.join(out, gid, '*.pkl')))
    if not fs:
        print(f'{gid:3s}  NO PKLS'); continue
    leg = han = 0
    peg = start = None
    for f in fs:
        d = pickle.load(open(f, 'rb'))
        leg += int(d['success_legacy']); han += int(d['success_hanging'])
        peg, start = d['peg_nominal'], d['deform_init_pos_nominal']
    n = len(fs); p = leg / n
    ci = 1.96 * math.sqrt(max(p * (1 - p), 1e-9) / n)
    fmt = lambda v: '[' + ' '.join(f'{x:6.2f}' for x in v) + ']'
    print(f'{gid:3s} {fmt(peg):>22s} {fmt(start):>22s} {n:4d} '
          f'{p:6.3f} +/-{ci:.3f} {han/n:8.3f}')
PYEOF
echo "=== done: $OUT ==="
