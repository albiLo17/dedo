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
def build_hole_aware_waypoints(underlying, waypoint_scale=None):
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
    #   2. THREAD — sweep the hole past the apex in -y while staying
    #               near apex height (z = apex.z + 0.1). The cloth body
    #               translates well past the hanger plane BEFORE the
    #               descent, so the apex sits cleanly inside the hole
    #               boundary when CATCH begins. Pushing further forward
    #               here (vs. the original -0.4) leaves a shorter,
    #               mostly-vertical descent in CATCH — more realistic
    #               for a real robot replay.
    #   3. CATCH  — drop down past the apex (y = apex.y - 1.2,
    #               z = apex.z - 0.5). With THREAD already well forward,
    #               this is mostly a downward motion that lets the apex
    #               catch on the trailing edge of the hole. The cloth
    #               weight then drapes around the hanger arms during
    #               the make_final_steps gravity settle.
    # The offsets below were authored for the `hangcloth` scene, where the cloth
    # starts at y=+5 and the peg sits at y=0, so "thread" means sweeping along
    # -y. Two things are therefore scene-specific and must NOT stay hardcoded:
    #
    #  1. THE AXIS. hangcloth_real puts the cloth in the XZ plane approaching
    #     along +x (init x=0.275 -> goal x=0.5, dy=0). Sweeping -y there is
    #     ORTHOGONAL to the direction the cloth has to travel, so the hole never
    #     reaches the peg. We recover the approach direction from the geometry
    #     instead: the horizontal vector from the hole to the goal.
    #  2. THE MAGNITUDE. The offsets are absolute scene units. hangcloth_real is
    #     ~22x smaller (deform_scale 0.135 vs 3.0), where +1.8 above the peg is
    #     13.3 cloth-lengths rather than 0.6 — the gripper flies 1.8 m up inside
    #     a 0.3 m workspace. Default the scale to the ratio of this scene's cloth
    #     to the reference one, which is exactly the 0.045 sim->real factor in
    #     frame_transforms.md.
    #
    # On the reference scene this reduces EXACTLY to the original numbers
    # (u = -y, s = 1.0), so the validated hangcloth behaviour is unchanged.
    s = waypoint_scale
    if s is None:
        ref_scale = 3.0  # deform_scale of the reference `hangcloth` scene
        s = float(getattr(underlying.args, 'deform_scale', ref_scale)) / ref_scale

    # The approach axis comes from SCENE geometry (where the cloth spawns vs
    # where the peg is), NOT from this episode's hole centroid. Using the hole
    # would tilt the sweep by the hole's random in-cloth offset and lose the
    # "drive the hole directly over the peg in the perpendicular axis" property
    # that the original hanger[0] pin provided — measured as a real regression
    # on the reference scene (8/11 kept vs 8/9 on the same seed).
    init = np.array(getattr(underlying.args, 'deform_init_pos',
                            [0.0, 5.0, 8.0]), dtype=np.float32)
    approach = hanger - init
    approach[2] = 0.0  # horizontal only; height is handled by the z terms
    norm = float(np.linalg.norm(approach))
    if norm > 1e-6:
        u = approach / norm
        # Snap to the dominant axis. Both shipped scenes travel along exactly
        # one horizontal axis (-y for hangcloth, +x for hangcloth_real);
        # snapping keeps the perpendicular coordinate pinned to the peg's,
        # which is what makes the hole line up for the descent.
        k = int(np.argmax(np.abs(u[:2])))
        u = np.array([0.0, 0.0, 0.0], np.float32)
        u[k] = np.sign(approach[k])
    else:
        u = np.array([0.0, -1.0, 0.0], np.float32)

    def _wp(along, up):
        """`along` is signed distance past the peg in the approach direction;
        `up` is height above it. Both in reference-scene units, scaled by s.
        The perpendicular horizontal coordinate stays at the peg's."""
        p = hanger + u * (along * s)
        p[2] = hanger[2] + up * s
        return p

    hole_hover = _wp(along=-0.2, up=+1.8)   # behind + above: line up the descent
    hole_thread = _wp(along=+1.0, up=+0.4)  # sweep the hole past the apex
    hole_hold = _wp(along=+1.2, up=-0.5)    # drop so the apex catches the rim

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
# Per-episode max_episode_len computation. Mirrors collect_bc_demos.py
# exactly: build the scripted hole-aware trajectory the demo controller
# WOULD have run for the current cloth, then set the env cap to
# `len(traj) + episode_tail_frames`. This makes eval episodes terminate at
# the same control-step distribution as training demos, avoiding the OOD
# tail where the diffusion policy drifts into pull-taut behavior.
#
# Returns None on any failure (no hole loop, NaN mesh, traj build error),
# so the caller can fall back to a global safety cap.
# ---------------------------------------------------------------------------
def compute_per_episode_max_len(deform, ctrl_freq, tail_frames, safety_cap):
    from dedo.demo_preset import build_traj, merge_traj
    wp = build_hole_aware_waypoints(deform)
    if wp is None:
        return None
    try:
        _, va = build_traj(deform, wp, 'a', anchor_idx=0,
                           ctrl_freq=ctrl_freq, robot=None)
        _, vb = build_traj(deform, wp, 'b', anchor_idx=1,
                           ctrl_freq=ctrl_freq, robot=None)
        traj = merge_traj(va, vb)
    except Exception:
        return None
    return min(int(len(traj)) + int(tail_frames), int(safety_cap))


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


# ---------------------------------------------------------------------------
# Closed-loop hole servo for goal-chained ("chain") demo episodes.
#
# The waypoint builder above plans ONCE, in hole space, using a Δ = grip - hole
# offset frozen at episode start, and both anchors then receive the same
# velocity. Two consequences: the plan drifts off the hole as the cloth
# deforms, and the cloth only ever TRANSLATES. This servo fixes both — it
# recomputes the hole frame every control step, and adds a differential term
# that varies the inter-anchor geometry, which is what actually folds the
# cloth.
#
# Velocity is capped to real-rig scale, not to what the sim can do. Measured
# on 0807_demos (8 demos, both arms): p50 0.60, p90 1.02, max 1.89 sim
# units/s. DEFAULT_V_MAX sits just above p90 so demos look like teleop rather
# than like a robot flinging cloth; MAX_ACT_VEL (4.0) is ~4x that and must
# NOT become the operative limit.
# ---------------------------------------------------------------------------
REAL_SPEED_P50 = 0.60      # sim units/s, measured on 0807_demos
REAL_SPEED_P90 = 1.02
REAL_SPEED_MAX = 1.89

DEFAULT_V_MAX = 1.2        # common-mode cap
DEFAULT_V_DIFF_MAX = 0.5   # differential cap, deliberately < common mode


def sample_chain_goal(rng, centroid, normal, r_min, r_max, in_view_fn,
                      z_min=1.0, box=None, max_tries=64,
                      elev_max_deg=35.0, azim_span_deg=60.0):
    """A hole-centroid goal `r ~ U(r_min, r_max)` away, in a *reachable* place.

    Direction is NOT uniform on the sphere, and that is deliberate. A hole in a
    hanging cloth faces horizontally — measured: the normal sits at
    |cos(normal, world z)| = 0.015 — so a goal with a large vertical component
    can never satisfy the hole-orientation arrival test no matter how well the
    servo tracks it. Elevation is therefore bounded, and azimuth is sampled
    within `azim_span_deg` of where the hole currently points so the required
    re-aiming stays inside one segment's time budget. Successive goals compound,
    so the cloth can still end up facing anywhere over a full chain.

    This also matches the real task, where the approach to the peg is
    essentially horizontal followed by a drop.

    Rejects goals that leave the workspace, sink to/below the table, or fall
    outside the camera frustum. The last matters more than it looks: the cloth
    already contributes few pixels, and a goal that drags it out of frame
    starves the point cloud without failing anything loudly.
    """
    base_azim = float(np.arctan2(normal[1], normal[0])) \
        if normal is not None else float(rng.uniform(-np.pi, np.pi))
    # The normal's sign is arbitrary, so either facing is equally valid.
    if rng.random() < 0.5:
        base_azim += np.pi
    for _ in range(max_tries):
        azim = base_azim + np.radians(rng.uniform(-azim_span_deg, azim_span_deg))
        elev = np.radians(rng.uniform(-elev_max_deg, elev_max_deg))
        d = np.array([np.cos(elev) * np.cos(azim),
                      np.cos(elev) * np.sin(azim),
                      np.sin(elev)], dtype=np.float64)
        g = centroid + d * rng.uniform(r_min, r_max)
        if g[2] < z_min:
            continue
        if box is not None and np.abs(g).max() > box:
            continue
        if in_view_fn is not None and not in_view_fn(g):
            continue
        return g.astype(np.float32)
    return None


class HoleServo:
    """Drives the cloth's hole to a sequence of sampled goals.

    Per control step returns a 6-dim UNSCALED velocity (anchor0 xyz, anchor1
    xyz) — the same layout dedo's action expects before the
    `/ MAX_ACT_VEL` normalization the collector applies.
    """

    def __init__(self, rng, cloth_width, r_min=1.5, r_max=4.0,
                 v_max=DEFAULT_V_MAX, v_diff_max=DEFAULT_V_DIFF_MAX,
                 k_p=1.0, k_d=0.8, dwell=3, timeout_steps=90,
                 pos_tol_radii=1.0, pos_tol_floor=0.5, orient_tol_deg=45.0,
                 sep_lo=0.70, sep_hi=1.05, rot_deg=30.0,
                 z_min=1.0, box=None, in_view_fn=None,
                 slew_max=0.35, brake_speed=4.0):
        self.rng = rng
        self.cloth_width = float(cloth_width)
        self.r_min, self.r_max = r_min, r_max
        self.v_max, self.v_diff_max = v_max, v_diff_max
        self.k_p, self.k_d = k_p, k_d
        self.dwell, self.timeout_steps = dwell, timeout_steps
        self.pos_tol_radii, self.pos_tol_floor = pos_tol_radii, pos_tol_floor
        self.cos_tol = float(np.cos(np.radians(orient_tol_deg)))
        self.sep_lo, self.sep_hi, self.rot_deg = sep_lo, sep_hi, rot_deg
        self.z_min, self.box, self.in_view_fn = z_min, box, in_view_fn
        self.slew_max = slew_max          # max change in commanded v per step
        self.brake_speed = brake_speed    # measured |v| that triggers braking
        self.n_brake_steps = 0
        self._v_prev = np.zeros((2, 3), dtype=np.float64)

        self.goal = None
        self.approach = None
        self.sep_target = None
        self.rot_target = None
        self._dwell_count = 0
        self._seg_steps = 0
        self.goals_reached = 0
        self.goals_attempted = 0
        self.segment_steps = []
        self.goal_positions = []
        self.n_unreliable_frames = 0

    # -- goal lifecycle -----------------------------------------------------
    def new_goal(self, centroid, anchor_a, anchor_b, normal=None):
        g = sample_chain_goal(self.rng, centroid, normal, self.r_min,
                              self.r_max, self.in_view_fn, self.z_min, self.box)
        if g is None:
            return False
        self.goal = g
        self.approach = g - centroid
        n = float(np.linalg.norm(self.approach))
        self.approach = (self.approach / n if n > 1e-9
                         else np.array([0.0, 0.0, 1.0], np.float32))
        # Per-segment inter-anchor target. Clamped to the rest width so the
        # spring mesh is never asked to stretch past it.
        self.sep_target = float(min(
            self.cloth_width * self.rng.uniform(self.sep_lo, self.sep_hi),
            self.cloth_width))
        # Aim the anchor baseline PERPENDICULAR to the approach. The cloth
        # hangs from the two anchors, so its plane contains the baseline and
        # the hole normal is perpendicular to it — putting the baseline across
        # the approach is what turns the hole to face where it is going, which
        # the orientation arrival test then requires. The random term on top
        # is the deformation knob, not the aiming mechanism.
        a_h = np.array([self.approach[0], self.approach[1], 0.0])
        if float(np.linalg.norm(a_h)) > 1e-6:
            perp = np.cross(np.array([0.0, 0.0, 1.0]), a_h)
            aim = float(np.arctan2(perp[1], perp[0]))
        else:
            aim = 0.0
        self.rot_target = float(
            aim + np.radians(self.rng.uniform(-self.rot_deg, self.rot_deg)))
        self._dwell_count = 0
        self._seg_steps = 0
        self.goals_attempted += 1
        self.goal_positions.append(np.asarray(g, dtype=np.float32))
        return True

    def reached(self, centroid, normal, radius, reliable):
        """Position AND hole-plane orientation, held for `dwell` steps.

        Position alone is satisfiable with the hole edge-on to the approach,
        which would never thread a real peg — the orientation term is what
        makes this transfer. |cos| because the PCA normal's sign is arbitrary.
        Orientation is skipped on unreliable (collapsed) hole frames.
        """
        tol = max(self.pos_tol_radii * radius, self.pos_tol_floor)
        ok = bool(np.linalg.norm(centroid - self.goal) < tol)
        if ok and reliable and normal is not None:
            ok = abs(float(normal @ self.approach)) > self.cos_tol
        elif ok and not reliable:
            self.n_unreliable_frames += 1
        self._dwell_count = self._dwell_count + 1 if ok else 0
        return self._dwell_count >= self.dwell

    def timed_out(self):
        return self._seg_steps >= self.timeout_steps

    def close_segment(self, reached):
        self.segment_steps.append(int(self._seg_steps))
        if reached:
            self.goals_reached += 1

    # -- control ------------------------------------------------------------
    def action(self, centroid, anchor_a, anchor_b, vel_a=None, vel_b=None):
        """6-dim unscaled velocity: common-mode goal tracking + differential.

        Two limits here are not cosmetic. dedo drives anchors with a
        force-limited velocity PD (`command_anchor_velocity`: force =
        clip(50*dv, +-10) on a 0.1 kg anchor), so a STEP change in the target
        saturates that force and a taut cloth then flings the anchor past the
        env's 20 units/s abort threshold -- measured: the episode died at step
        23 with anchor linvel ~15 units/s while the command never exceeded
        1.9. Hence the slew limit. And the cap is applied to the TOTAL
        (common + differential) per anchor, not to each term separately, so
        the realized speed actually matches the real-rig envelope the cap was
        sized from rather than their sum.
        """
        self._seg_steps += 1

        v = np.clip(self.k_p * (self.goal - centroid), -self.v_max, self.v_max)
        v_common = np.repeat(v[None, :], 2, axis=0)

        # Differential term: servo the anchor baseline toward this segment's
        # (separation, yaw) target. Applied antisymmetrically so it deforms
        # the cloth without moving its centre.
        base = anchor_b - anchor_a
        sep = float(np.linalg.norm(base))
        if sep < 1e-6:
            out = v_common
        else:
            u = base / sep
            yaw = float(np.arctan2(u[1], u[0]))
            # The baseline direction is defined only up to sign (b-a vs a-b),
            # so aiming at rot_target must not force a needless 180 deg swing:
            # take whichever of rot_target, rot_target+pi is the shorter turn,
            # then wrap the error into (-pi, pi] before rate-limiting it.
            def _wrap(x):
                return (x + np.pi) % (2 * np.pi) - np.pi
            cand = [_wrap(self.rot_target - yaw),
                    _wrap(self.rot_target + np.pi - yaw)]
            err = min(cand, key=abs)
            desired_yaw = yaw + float(np.clip(err, -0.35, 0.35))
            desired = np.array([np.cos(desired_yaw), np.sin(desired_yaw), u[2]],
                               dtype=np.float64)
            desired /= max(float(np.linalg.norm(desired)), 1e-9)
            target_base = desired * self.sep_target
            d_half = self.k_d * 0.5 * (target_base - base)
            d_half = np.clip(d_half, -self.v_diff_max, self.v_diff_max)
            out = np.stack([v_common[0] - d_half, v_common[1] + d_half])

        # Cap the TOTAL per-anchor speed.
        for i in range(2):
            n = float(np.linalg.norm(out[i]))
            if n > self.v_max:
                out[i] *= self.v_max / n

        # Active brake. dedo ends the episode the moment a measured anchor
        # velocity exceeds MAX_OBS_VEL (20 units/s), and a 1 kg cloth swinging
        # on 0.1 kg anchors gets there on its own -- measured: |linvel|/20 ran
        # 0.49 -> 1.01 over five steps while the command never exceeded 1.2.
        # When an anchor is already moving too fast, command a target OPPOSING
        # its motion so the velocity PD spends its force budget decelerating
        # instead of chasing the goal.
        meas = [vel_a, vel_b]
        for i in range(2):
            if meas[i] is None:
                continue
            sp = float(np.linalg.norm(meas[i]))
            if sp > self.brake_speed:
                self.n_brake_steps += 1
                out[i] = -(meas[i] / sp) * min(sp - self.brake_speed, self.v_max)
        delta = out - self._v_prev
        for i in range(2):
            n = float(np.linalg.norm(delta[i]))
            if n > self.slew_max:
                delta[i] *= self.slew_max / n
        out = self._v_prev + delta
        self._v_prev = out.copy()
        return out.reshape(-1).astype(np.float32)


def cloth_min_z(deform_env):
    """Height of the LOWEST cloth vertex above the ground plane."""
    from dedo.utils.mesh_utils import get_mesh_data
    _, verts = get_mesh_data(deform_env.sim, deform_env.deform_id)
    v = np.asarray(verts, dtype=np.float32)
    v = v[~np.isnan(v).any(axis=1)]
    return float(v[:, 2].min()) if len(v) else float('inf')


class SlackPhase:
    """Put real slack and folds into the cloth before the goal-chasing starts.

    A cloth held taut between two anchors is a flat sheet, and a flat sheet is
    the easy case. Two motions run together to break that, both of them the
    kind a real arm can execute:

      CONVERGE — bring the anchors toward each other, to `squeeze` of the
        cloth's rest width. This is the only direction we ever drive the
        separation: pulling the anchors APART past the rest width stretches
        the spring mesh, which is elastic deformation rather than folding and
        does not correspond to anything the real cloth does.

      LOWER — descend until the lowest cloth vertex is within `floor_clear`
        of the ground, so the fabric piles against the floor and folds instead
        of hanging flat.

    Then hold, so the folds are settled rather than mid-swing when the goal
    servo takes over. Speeds are capped by the same real-rig envelope the
    chain controller uses, so these frames stay replayable on the robot.
    """

    def __init__(self, rng, cloth_width, squeeze_range=(0.45, 0.80),
                 floor_clear=0.4, v_max=REAL_SPEED_P90, hold_steps=8,
                 max_steps=120, slew_max=0.30, brake_speed=4.0,
                 abort_guard=11.0):
        self.rng = rng
        self.cloth_width = float(cloth_width)
        self.squeeze = float(rng.uniform(*squeeze_range))
        self.sep_target = self.cloth_width * self.squeeze
        self.floor_clear = float(floor_clear)
        self.v_max = float(v_max)
        self.hold_steps, self.max_steps = int(hold_steps), int(max_steps)
        self.slew_max = float(slew_max)
        self.brake_speed = float(brake_speed)
        self.abort_guard = float(abort_guard)   # dedo aborts at 20 units/s
        self.n_brake_steps = 0
        self.n_guard_trips = 0
        self._v_prev = np.zeros((2, 3), dtype=np.float64)
        self._steps = 0
        self._held = 0
        self.reached_floor = False
        self.min_z_seen = float('inf')
        self.sep_start = None
        self._last_min_z = None
        self._stalled = 0

    def action(self, a_pos, b_pos, min_z, v_a=None, v_b=None):
        """6-dim unscaled velocity, or None when the phase is finished.

        `v_a`/`v_b` are the MEASURED anchor velocities and are not optional in
        practice: the anchors are 0.1 kg against a 1.0 kg cloth, so a swinging
        or floor-piling cloth flings them far faster than anything commanded,
        and dedo ends the episode above 20 units/s. Without braking on the
        measured value this phase silently truncated 5 episodes in 6.
        """
        self._steps += 1
        self.min_z_seen = min(self.min_z_seen, min_z)
        a = np.asarray(a_pos, dtype=np.float64)
        b = np.asarray(b_pos, dtype=np.float64)
        if self.sep_start is None:
            self.sep_start = float(np.linalg.norm(a - b))

        sep = float(np.linalg.norm(a - b))
        low_enough = min_z <= self.floor_clear
        # Contact detection. Once the fabric is piling on the floor its lowest
        # point stops descending however far the anchors keep going, and
        # driving into the ground spikes the anchor velocity — dedo aborts the
        # episode above 20 units/s, which silently truncated every slack
        # episode before this check existed. Treat "stopped falling" as
        # arrival, not just the height threshold.
        if self._last_min_z is not None and min_z > self._last_min_z - 0.01:
            self._stalled += 1
        else:
            self._stalled = 0
        self._last_min_z = min_z
        if self._stalled >= 4:
            low_enough = True
        self.reached_floor = self.reached_floor or low_enough
        close_enough = sep <= self.sep_target

        if (low_enough and close_enough) or self._steps > self.max_steps:
            self._held += 1
            if self._held >= self.hold_steps:
                return None
            return np.zeros(6, dtype=np.float32)   # settle in place

        v = np.zeros((2, 3), dtype=np.float64)
        if not close_enough:
            # Antisymmetric: each anchor moves toward the midpoint, so the
            # cloth's centre stays put and only the separation changes.
            axis = (a - b)
            n = float(np.linalg.norm(axis))
            if n > 1e-9:
                axis = axis / n
                gain = min(1.0, (sep - self.sep_target) / max(self.cloth_width, 1e-6))
                v[0] -= axis * self.v_max * 0.6 * max(gain, 0.25)
                v[1] += axis * self.v_max * 0.6 * max(gain, 0.25)
        if not low_enough:
            drop = min(1.0, (min_z - self.floor_clear) / max(self.cloth_width, 1e-6))
            v[:, 2] -= self.v_max * 0.30 * max(drop, 0.2)

        # Brake on MEASURED velocity, exactly as HoleServo does: when the
        # cloth has flung an anchor, the only useful command is one that
        # opposes it, whatever the phase would otherwise like to do.
        meas = np.zeros((2, 3), dtype=np.float64)
        if v_a is not None and v_b is not None:
            meas[0] = np.asarray(v_a, dtype=np.float64)
            meas[1] = np.asarray(v_b, dtype=np.float64)
        fastest = max(float(np.linalg.norm(meas[i])) for i in range(2))
        for i in range(2):
            sp = float(np.linalg.norm(meas[i]))
            if sp > self.brake_speed:
                self.n_brake_steps += 1
                v[i] = -(meas[i] / sp) * min(sp - self.brake_speed, self.v_max)

        # Bail out before dedo does. The env ends the episode once an anchor
        # exceeds MAX_OBS_VEL (20 units/s), and a 1 kg cloth piling against the
        # floor can fling a 0.1 kg anchor there faster than a 1 unit/s command
        # can pull it back -- braking is far too weak an authority to win that
        # race. Ending the phase hands control to the goal servo, which is
        # proven stable, instead of losing the whole episode.
        if fastest > self.abort_guard:
            self.n_guard_trips += 1
            if self.n_guard_trips >= 2:
                return None
            return np.zeros(6, dtype=np.float32)

        # Same slew limit as the chain controller: a step change in commanded
        # velocity flings the 0.1 kg anchors and trips the env's velocity abort.
        delta = np.clip(v - self._v_prev, -self.slew_max, self.slew_max)
        out = self._v_prev + delta
        self._v_prev = out.copy()
        return out.reshape(-1).astype(np.float32)


def anchor_velocities(deform_env):
    """LIVE anchor linear velocities from the sim.

    Needed because the anchors are 0.1 kg and the cloth is 1.0 kg: a swinging
    cloth flings them far faster than anything commanded. dedo aborts the
    episode when |linvel| exceeds MAX_OBS_VEL (20 units/s), so the controller
    has to watch the measured velocity, not just its own setpoint.
    """
    out = []
    for aid in deform_env.anchor_ids:
        _, linvel = deform_env.sim.getBaseVelocity(aid)
        out.append(np.asarray(linvel, dtype=np.float32))
    return out


def anchor_positions(deform_env):
    """LIVE anchor positions from the sim.

    Not `deform_env.anchors[id]['pos']` — that dict holds the pose the anchor
    was CREATED at and never updates, so a controller that closes the loop on
    it silently commands a constant correction and stretches the cloth until
    the env aborts the episode. That is a real bug this cost an hour to find.
    """
    out = []
    for aid in deform_env.anchor_ids:
        pos, _ = deform_env.sim.getBasePositionAndOrientation(aid)
        out.append(np.asarray(pos, dtype=np.float32))
    return out


def make_in_view_fn(view, proj, margin=0.12):
    """world point -> bool, is it comfortably inside the camera frustum.

    Same projection convention as overlay_pcd_on_rgb / _diag_pcd_framing
    (clip = proj @ view @ p, column-major reshape). `margin` shrinks the NDC
    box so a goal never lands right at the image edge — the cloth hangs BELOW
    the hole, so a hole at the border puts most of the cloth outside.
    """
    v = np.asarray(view, dtype=np.float64).reshape(4, 4, order='F')
    p = np.asarray(proj, dtype=np.float64).reshape(4, 4, order='F')
    m = p @ v
    lim = 1.0 - float(margin)

    def _in_view(pt):
        clip = m @ np.array([pt[0], pt[1], pt[2], 1.0], dtype=np.float64)
        if abs(clip[3]) < 1e-9:
            return False
        ndc = clip[:3] / clip[3]
        return bool(abs(ndc[0]) < lim and abs(ndc[1]) < lim
                    and -1.0 < ndc[2] < 1.0)

    return _in_view
