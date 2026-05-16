"""
Shared, side-effect-free utilities for hang_obs_exp scripts.

This module exists so record_demo.py / view_demo.py / train_privileged.py
can share helpers WITHOUT importing each other (they are scripts that run
training at import time, which would re-trigger pipelines unintentionally).
"""
import gym
import numpy as np
import pybullet


# ---------------------------------------------------------------------------
# Retry wrapper: dedo's procedural cloth generator occasionally produces a
# mesh that pybullet's loadSoftBody refuses (raises pybullet.error). reset()
# starts with reset_bullet() which clears sim state, so a retry is safe and
# will resample a new cloth.
# ---------------------------------------------------------------------------
class RetryResetEnv(gym.Wrapper):
    def __init__(self, env, max_retries=20):
        super().__init__(env)
        self._max_retries = max_retries

    def reset(self, **kwargs):
        last_err = None
        for attempt in range(self._max_retries):
            try:
                return self.env.reset(**kwargs)
            except pybullet.error as e:
                last_err = e
                print(f'[RetryReset] attempt {attempt+1}/{self._max_retries}: '
                      f'{e!r} — resampling cloth')
        raise last_err

    def step(self, action):
        try:
            return self.env.step(action)
        except pybullet.error as e:
            print(f'[RetryReset] pybullet.error during step: {e!r} — '
                  f'forcing episode end')
            obs = self.reset()
            return obs, 0.0, True, {'pybullet_step_error': True}


# ---------------------------------------------------------------------------
# Per-episode hole-aware waypoint builder. Reads privileged sim state
# (hole centroid, hanger goal, gripper init) and returns a {'a': ..., 'b': ...}
# dict consumable by dedo.demo_preset.build_traj.
#
# Waypoints are reasoned about in *hole* space (not gripper space). Each
# gripper sits ~Δ above and to the side of the cloth's hole; we want the
# hole — not the gripper — to track a desired path over the hanger. So we
# compute the per-gripper offset Δ = grip - hole at episode start and shift
# the gripper waypoints by Δ. Translating both grippers by the same vector
# preserves the gripper baseline so the cloth stays taut.
#
# Phases:
#   1. LIFT_AND_ALIGN: translate xy so the hole sits directly over the
#      hanger apex AND lift z so the hole is well above the apex pin top
#      (which extends to ~hanger.z + 0.55). We aim for hole.z = hanger.z
#      + 1.2.
#   2. THREAD: lower the hole through the pin so it ends right at the
#      apex (hole.z ≈ hanger.z).
#   3. HOLD: keep the gripper waypoint constant for the last fraction of
#      the trajectory so cloth dynamics settle while the hole is parked
#      at the apex; the post-trajectory zero-velocity hold and
#      make_final_steps then let the hanger catch the hole.
# ---------------------------------------------------------------------------
def build_hole_aware_waypoints(underlying):
    from dedo.utils.mesh_utils import get_mesh_data

    if not hasattr(underlying.args, 'deform_true_loop_vertices'):
        return None
    loops = underlying.args.deform_true_loop_vertices
    idxs = [i for loop in loops for i in loop]
    if len(idxs) == 0:
        return None

    _, verts = get_mesh_data(underlying.sim, underlying.deform_id)
    verts = np.array(verts, dtype=np.float32)
    hole_verts = verts[idxs]
    hole_verts = hole_verts[~np.isnan(hole_verts).any(axis=1)]
    if len(hole_verts) == 0:
        return None
    hole_centroid = hole_verts.mean(axis=0)

    hanger = np.array(underlying.goal_pos[0], dtype=np.float32)

    anc_ids = list(underlying.anchors.keys())
    grip_a = np.array(underlying.anchors[anc_ids[0]]['pos'], dtype=np.float32)
    grip_b = np.array(underlying.anchors[anc_ids[1]]['pos'], dtype=np.float32)

    delta_a = grip_a - hole_centroid  # 3-vector, gripper sits at hole + delta
    delta_b = grip_b - hole_centroid

    # Hole targets in world coordinates. Three-phase trajectory inspired
    # by the original `cloth/apron_0.obj` preset in dedo.utils.preset_info
    # (which ends at gripper y = -1.2, well past the hanger):
    #
    #   1. HOVER  — lift the hole well above the apex pin top
    #               (apex + 2.0; pin spans apex+0.05..apex+0.55) and
    #               translate it directly over the apex. Aligns the hole
    #               for the descent.
    #   2. THREAD — descend so the hole sweeps DOWN through the pin
    #               region and at the same time begin a y-overshoot:
    #               (y = apex.y - 0.5, z = apex.z + 0.0). The cloth body
    #               starts to sweep past the hanger plane, dragging the
    #               hanger arms through the cloth.
    #   3. CATCH  — continue past in y and slightly below in z
    #               (y = apex.y - 1.1, z = apex.z - 0.4). This is the
    #               key step: as the hole boundary slides past the apex
    #               in -y, the apex catches on the trailing edge of the
    #               hole. The cloth weight then drapes around the hanger
    #               arms during the make_final_steps gravity settle.
    hole_hover = np.array([hanger[0], hanger[1] + 0.2, hanger[2] + 1.8])
    hole_thread = np.array([hanger[0], hanger[1] - 0.4, hanger[2] + 0.1])
    hole_hold = np.array([hanger[0], hanger[1] - 1.2, hanger[2] - 0.5])

    def grip_target(hole_target, delta):
        return [float(hole_target[0] + delta[0]),
                float(hole_target[1] + delta[1]),
                float(hole_target[2] + delta[2])]

    # Phase 1: 1.4 s gives cloth time to translate from y=5 even when
    # initial gripper xy is far from the apex. Phase 2: 1.0 s slow swing
    # past the apex with the hole right at the pin's z range. Phase 3:
    # 0.6 s final overshoot that drags the cloth past so the apex catches
    # on the cloth's hole boundary.
    wp_a = [
        [*grip_target(hole_hover, delta_a), 1.4],
        [*grip_target(hole_thread, delta_a), 1.0],
        [*grip_target(hole_hold, delta_a), 0.6],
    ]
    wp_b = [
        [*grip_target(hole_hover, delta_b), 1.4],
        [*grip_target(hole_thread, delta_b), 1.0],
        [*grip_target(hole_hold, delta_b), 0.6],
    ]
    return {'a': wp_a, 'b': wp_b}


# ---------------------------------------------------------------------------
# Per-episode bag-handle-aware waypoint builder for HangBag-v1.
#
# Mirrors build_hole_aware_waypoints, but for tote bags. HangBag bags
# have TWO handle loops (see DEFORM_INFO['bags/totes/bag*_*.obj']
# ['deform_true_loop_vertices']) and TWO anchors (one grabbing each
# handle). Per the user spec, only ONE handle ("primary loop") needs
# to land on the hook; planning is done in that handle's centroid
# space.
#
# Both anchors translate by the same vector — i.e. they preserve the
# initial inter-anchor geometry (delta_a, delta_b are offsets from the
# primary handle centroid at episode start, then re-added to each
# handle target waypoint). This keeps the bag from being torn while
# the primary handle is driven to the hook.
#
# Phases (in HOOK space):
#   1. APPROACH — drive the handle centroid to a position just above
#                 the hook (small z clearance, same xy). This is the
#                 only long phase — the bag has to traverse from its
#                 init pose at y≈8, z≈2 to directly above the hook.
#   2. HOOK     — descend straight down so the handle centroid lands
#                 on the hook position; the hook ends up inside the
#                 handle ring. No high hover, no overshoot.
# ---------------------------------------------------------------------------
def build_bag_handle_waypoints(underlying, primary_loop_idx=0):
    from dedo.utils.mesh_utils import get_mesh_data

    if not hasattr(underlying.args, 'deform_true_loop_vertices'):
        return None
    loops = underlying.args.deform_true_loop_vertices
    if len(loops) <= primary_loop_idx:
        return None

    _, verts = get_mesh_data(underlying.sim, underlying.deform_id)
    verts = np.array(verts, dtype=np.float32)

    primary_idxs = loops[primary_loop_idx]
    handle_verts = verts[primary_idxs]
    handle_verts = handle_verts[~np.isnan(handle_verts).any(axis=1)]
    if len(handle_verts) == 0:
        return None
    handle_centroid = handle_verts.mean(axis=0)

    hook = np.array(underlying.goal_pos[0], dtype=np.float32)

    anc_ids = list(underlying.anchors.keys())
    grip_a = np.array(underlying.anchors[anc_ids[0]]['pos'], dtype=np.float32)
    grip_b = np.array(underlying.anchors[anc_ids[1]]['pos'], dtype=np.float32)

    delta_a = grip_a - handle_centroid
    delta_b = grip_b - handle_centroid

    # Hook (goal) is at e.g. [0, 1.28, 9] in the hangbag scene. Bag
    # starts at deform_init_pos = [0, 8, 2], so phase 1 must traverse
    # ~6.7 m in y and lift ~7+ m in z; budget a long first phase.
    # Phase 2 is a short straight-down descent — small clearance
    # turning into a clean hook.
    handle_approach = np.array([hook[0], hook[1], hook[2] + 0.4])
    handle_hook     = np.array([hook[0], hook[1], hook[2] - 0.1])

    def grip_target(handle_target, delta):
        return [float(handle_target[0] + delta[0]),
                float(handle_target[1] + delta[1]),
                float(handle_target[2] + delta[2])]

    wp_a = [
        [*grip_target(handle_approach, delta_a), 3.0],
        [*grip_target(handle_hook,     delta_a), 1.0],
    ]
    wp_b = [
        [*grip_target(handle_approach, delta_b), 3.0],
        [*grip_target(handle_hook,     delta_b), 1.0],
    ]
    return {'a': wp_a, 'b': wp_b}


# ---------------------------------------------------------------------------
# Per-episode hole-aware waypoint builder for ButtonProc-v0/v1.
#
# ButtonProc differs from HangBag in two key ways:
#   1. There are TWO hole loops AND TWO button goals — and the success
#      metric (deform_env.py:get_reward) takes the mean over both, so
#      both holes must reach their assigned buttons.
#   2. The cloth has its FAR edge pinned to the world via
#      deform_fixed_anchor_vertex_ids. That kills the HangBag-style
#      "rigid-translate both anchors together" trick, since the
#      cloth body is constrained on one side. Each movable anchor
#      drives its own hole independently.
#
# Pairing rule: nearest-hole at reset. anchor 0 is paired with the hole
# whose centroid is closest to it at episode start; anchor 1 gets the
# other hole. Robust to procedural hole order being random.
#
# Phases (per anchor, planned in HOLE space then offset back via the
# anchor->hole delta at reset):
#   1. APPROACH (2.0 s): hole_target = goal + [0, +0.3, 0]
#      — bring the hole right in front of its button in +y.
#   2. THREAD   (1.5 s): hole_target = goal + [0, -0.3, 0]
#      — pull past so the button pops through the hole.
# ---------------------------------------------------------------------------
def build_button_proc_waypoints(underlying):
    from dedo.utils.mesh_utils import get_mesh_data

    if not hasattr(underlying.args, 'deform_true_loop_vertices'):
        return None
    loops = underlying.args.deform_true_loop_vertices
    if len(loops) < 2 or len(underlying.goal_pos) < 2:
        return None

    _, verts = get_mesh_data(underlying.sim, underlying.deform_id)
    verts = np.array(verts, dtype=np.float32)

    hole_centroids = []
    for loop_idxs in loops[:2]:
        loop_verts = verts[loop_idxs]
        loop_verts = loop_verts[~np.isnan(loop_verts).any(axis=1)]
        if len(loop_verts) == 0:
            return None
        hole_centroids.append(loop_verts.mean(axis=0))

    goals = [np.asarray(underlying.goal_pos[i], dtype=np.float32)
             for i in range(2)]

    anc_ids = list(underlying.anchors.keys())
    grip_a = np.array(underlying.anchors[anc_ids[0]]['pos'], dtype=np.float32)
    grip_b = np.array(underlying.anchors[anc_ids[1]]['pos'], dtype=np.float32)

    # Nearest-hole assignment: anchor 0 -> nearest hole, anchor 1 -> other.
    d_a0 = float(np.linalg.norm(grip_a - hole_centroids[0]))
    d_a1 = float(np.linalg.norm(grip_a - hole_centroids[1]))
    if d_a0 <= d_a1:
        hole_for_a, goal_for_a = hole_centroids[0], goals[0]
        hole_for_b, goal_for_b = hole_centroids[1], goals[1]
    else:
        hole_for_a, goal_for_a = hole_centroids[1], goals[1]
        hole_for_b, goal_for_b = hole_centroids[0], goals[0]

    delta_a = grip_a - hole_for_a
    delta_b = grip_b - hole_for_b

    # Offsets are in HOLE space (added to each goal).
    #
    # APPROACH:
    #   -x by 0.2  → hole is just before the button plane (cloth-side),
    #                so the button hasn't entered the hole yet but is
    #                lined up.
    #   +y by 0.3  → hole is behind the button in y (cloth approaches
    #                from +y, button protrudes in +y from the torso).
    #   +z by 0.4  → compensates for cloth sagging under gravity once
    #                the anchors stop moving.
    #
    # THREAD:
    #   +x by 0.4  → hole is now past the button plane: button clearly
    #                sits *inside* the handle ring instead of just at
    #                the cloth plane.
    #   -y by 0.6  → swept past the button in -y (doubled overshoot
    #                vs APPROACH).
    #   +z by 0.4  → maintain the sag correction.
    approach_off = np.array([-0.2, +0.3, +0.4], dtype=np.float32)
    thread_off   = np.array([+0.4, -0.6, +0.4], dtype=np.float32)
    # RELAX: drift back to ~30% of thread overshoot so the cloth
    # de-stretches with the button still inside the hole, BEFORE the
    # trajectory ends and the zero-velocity hold begins. Without this,
    # the highly-stretched cloth at end-of-THREAD recoils against the
    # velocity controller (deform_damping_stiffness=0.01 is too low to
    # damp on its own) → high-frequency jitter.
    relax_off    = thread_off * 0.3

    def waypoints_for(goal, delta):
        approach = goal + approach_off
        thread   = goal + thread_off
        relax    = goal + relax_off
        return [
            # Phase timing trades motion speed vs. solver stability:
            # the cloth is fragile (elastic_stiffness=10), so cutting
            # too aggressively will produce "spike" artifacts from the
            # softbody integrator. APPROACH 2.0 + THREAD 1.0 + RELAX 1.2
            # = 4.2 s total. RELAX uses the longest duration so the
            # de-stretching is gradual and doesn't slingshot the cloth.
            [float(approach[0] + delta[0]),
             float(approach[1] + delta[1]),
             float(approach[2] + delta[2]), 2.0],
            [float(thread[0] + delta[0]),
             float(thread[1] + delta[1]),
             float(thread[2] + delta[2]), 1.0],
            [float(relax[0] + delta[0]),
             float(relax[1] + delta[1]),
             float(relax[2] + delta[2]), 1.2],
        ]

    return {
        'a': waypoints_for(goal_for_a, delta_a),
        'b': waypoints_for(goal_for_b, delta_b),
    }


def probe_peak_demo_vel(dedo_args, n_probes=3, max_attempts=12):
    """Probe scripted-demo trajectories without stepping the env to find
    the peak |velocity| the waypoint controller demands. Used to size
    DeformEnv.MAX_ACT_VEL safely from above so the demo collector's
    `clip(act / MAX_ACT_VEL, -1, 1)` round-trip never saturates and
    silently breaks demos.

    Returns peak m/s across up to n_probes successful build_traj calls
    on freshly-reset cloths, or None if all probes failed.
    """
    from copy import deepcopy
    from dedo.envs.deform_env import DeformEnv
    from dedo.demo_preset import build_traj, merge_traj

    args = deepcopy(dedo_args)
    args.debug = False
    args.viz = False
    env = gym.make(args.env, args=args)
    env = RetryResetEnv(env)
    env.seed(args.seed + 7777)
    ctrl_freq = args.sim_freq / args.sim_steps_per_action

    peaks = []
    attempts = 0
    while len(peaks) < n_probes and attempts < max_attempts:
        attempts += 1
        env.reset()
        underlying = env
        while hasattr(underlying, 'env'):
            underlying = underlying.env
            if isinstance(underlying, DeformEnv):
                break
        wp = build_hole_aware_waypoints(underlying)
        if wp is None:
            continue
        try:
            _, va = build_traj(underlying, wp, 'a', anchor_idx=0,
                               ctrl_freq=ctrl_freq, robot=None)
            _, vb = build_traj(underlying, wp, 'b', anchor_idx=1,
                               ctrl_freq=ctrl_freq, robot=None)
            traj = merge_traj(va, vb)
        except Exception:
            continue
        peaks.append(float(np.abs(traj).max()))

    env.close()
    return max(peaks) if peaks else None


# ---------------------------------------------------------------------------
# Wandb run-name autonaming.
#
# The wandb run name is the only thing visible in the runs table, in
# screenshots, and in URLs — so it must be self-describing enough that a
# crashed run can be identified without opening config.yaml. This builds a
# suffix that encodes:
#   - obs kind / arch
#   - lr (always)
#   - knobs that diverge from defaults (critic warmup, BC anchor,
#     demo-V warmup, PPO clip / epochs / target_kl, log_std_init,
#     BC budget, entropy coef)
#   - reward shape (sf, sb, fp, vp, ap, psc)
# Defaults are omitted so the name stays short on baselines and grows
# only when an actual experimental dial is turned.
# ---------------------------------------------------------------------------
def _fmt_lr(lr):
    """Compact lr format: 5e-05 -> 5e-5, 0.0003 -> 3e-4, 0.001 -> 1e-3."""
    if lr is None:
        return 'none'
    s = f'{float(lr):.0e}'  # '5e-05'
    mantissa, _, exp = s.partition('e')
    sign = '-' if exp.startswith('-') else ''
    exp_num = exp.lstrip('+-').lstrip('0') or '0'
    return f'{mantissa}e{sign}{exp_num}'


def build_run_name_suffix(extra_args, *, algo='PPO', obs_kind=None,
                          net_arch=None, extra_tag=''):
    """Build a wandb run-name suffix encoding all experimental dials.

    Args:
        extra_args: argparse Namespace with the script's `--` flags
            (lr, success_factor, critic_warmup_rollouts, etc.). Looked
            up by getattr; missing attributes fall back to defaults.
        algo: 'PPO' or 'SAC'. Controls which algo-specific knobs are
            considered.
        obs_kind: short string like 'privileged', 'pixels64_grip',
            'pcd1024'. Becomes the first segment.
        net_arch: list[int] for the MLP head/trunk; emitted as
            '_<h1>x<h2>...'.
        extra_tag: appended at the very end (e.g. policy='pointnet2'
            for the pcd script).

    Format:
        [_<obs>][_<arch>]_lr<lr>[_cw<n>][_bca<n>x<bs>@<lr>]
        [_cwd<n>@<lr>][_clip<r>][_pe<n>][_tkl<kl>][_lstd<v>]
        [_bc<eps>x<ep>][_ent<v>]_sf<sf>[_sb<sb>][_fp<fp>][_vp<vp>]
        [_ap<ap>][_psc<psc>][_<extra_tag>]
    """
    a = extra_args
    parts = []

    if obs_kind:
        parts.append(f'_{obs_kind}')
    if net_arch:
        parts.append('_' + 'x'.join(str(s) for s in net_arch))

    # lr — always shown (top hunt-down knob)
    if hasattr(a, 'lr') and a.lr is not None:
        parts.append(f'_lr{_fmt_lr(a.lr)}')

    # Critic warmup (rollout-based actor-freeze)
    if getattr(a, 'critic_warmup_rollouts', 0):
        parts.append(f'_cw{a.critic_warmup_rollouts}')

    # BC anchor (DAPG-style replay) — only show if active
    if getattr(a, 'bc_anchor_batches', 0):
        bs = getattr(a, 'bc_anchor_batch_size', 256)
        anc_lr = getattr(a, 'bc_anchor_lr', 1e-4)
        parts.append(f'_bca{a.bc_anchor_batches}x{bs}@{_fmt_lr(anc_lr)}')

    # Demo-V critic warmup (offline V pretrain on demos)
    if getattr(a, 'critic_warmup_demo_epochs', 0):
        cwd_lr = getattr(a, 'critic_warmup_demo_lr', 3e-4)
        parts.append(f'_cwd{a.critic_warmup_demo_epochs}@{_fmt_lr(cwd_lr)}')

    # PPO drift knobs (only if non-default SB3 values)
    if algo == 'PPO':
        clip = getattr(a, 'ppo_clip_range', None)
        if clip is not None and abs(float(clip) - 0.2) > 1e-9:
            parts.append(f'_clip{float(clip):g}')
        pe = getattr(a, 'ppo_epochs', None)
        if pe is not None and int(pe) != 10:
            parts.append(f'_pe{int(pe)}')
        tkl = getattr(a, 'ppo_target_kl', None)
        if tkl is not None:
            parts.append(f'_tkl{float(tkl):g}')

    # Action distribution scale
    log_std = getattr(a, 'log_std_init', None)
    if log_std is not None:
        parts.append(f'_lstd{float(log_std):g}')

    # BC budget — show if BC pretrain is active
    bc_eps = getattr(a, 'bc_episodes', 0) or 0
    bc_demo_path = getattr(a, 'bc_demo_path', None)
    if bc_eps > 0:
        bc_ep = getattr(a, 'bc_epochs', 0)
        parts.append(f'_bc{bc_eps}x{bc_ep}')
    elif bc_demo_path:
        bc_ep = getattr(a, 'bc_epochs', 0)
        parts.append(f'_bcExtx{bc_ep}')

    # Entropy coef
    ent = getattr(a, 'ent_coef', None)
    if ent is not None:
        if isinstance(ent, str):
            if ent != '0' and ent != '0.0':
                parts.append(f'_ent{ent}')
        elif float(ent) != 0:
            parts.append(f'_ent{float(ent):g}')

    # Reward shape (always; sf is the keystone)
    sf = getattr(a, 'success_factor', None)
    parts.append(f'_sf{float(sf):g}' if sf is not None else '_sf_default')
    for tag, name in [('sb', 'success_bonus'), ('fp', 'fail_penalty'),
                      ('vp', 'vel_penalty'), ('ap', 'action_penalty'),
                      ('psc', 'pre_settle_coef'),
                      ('dr', 'dist_reward_coef'),
                      ('tb', 'threading_bonus_coef')]:
        v = getattr(a, name, 0)
        if v:
            parts.append(f'_{tag}{float(v):g}')
    # Terminal-magnitude scale (override of dedo's FINAL_REWARD_MULT=400).
    # Only emitted when explicitly set, since None = keep dedo default.
    frm = getattr(a, 'final_reward_mult', None)
    if frm is not None:
        parts.append(f'_frm{float(frm):g}')
    # Success metric (hanging / topological / legacy). Always emitted
    # so the run name distinguishes runs evaluated under different
    # metrics, except for the default ('hanging') which is implicit.
    sm = getattr(a, 'success_metric', None)
    if sm and sm != 'hanging':
        parts.append(f'_sm-{sm}')

    if extra_tag:
        parts.append(f'_{extra_tag}')

    return ''.join(parts)
