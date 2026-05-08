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
                      ('psc', 'pre_settle_coef')]:
        v = getattr(a, name, 0)
        if v:
            parts.append(f'_{tag}{float(v):g}')

    if extra_tag:
        parts.append(f'_{extra_tag}')

    return ''.join(parts)
