"""Does this demo dataset actually bracket the real rig?

Run over a collected demo dir BEFORE committing to a full-size run. The
headline output is a table, one row per randomized axis:

    axis | sim p1-p99 | real value | real's percentile in sim | verdict

The design rule for v5 is that every real value sits INSIDE the sim
distribution with margin — near the middle, not in a tail. A row where the
real value lands below p5 or above p95 (or outside entirely) is a coverage
failure to fix before collecting 1000 demos, because no amount of training
fixes a value the policy never saw.

The panels beside it are for eyeballing shape, not for the verdict.

    python experiments/hang_obs_exp/scripts/_diag_dataset_coverage.py \
        --demo_path logs/.../bc_demos_v5 --out coverage.png

Real reference values are measured, not assumed — see HANDOFF_real2sim.md in
the franka-deformables repo for how each was obtained.
"""
import argparse
import os
import pickle

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt                                  # noqa: E402

SCALE_MM = 45.0          # 1 sim unit = 45 mm
WBOX = 20.0

# --- measured on the real rig (0807 sessions), single source of truth -------
REAL = {
    'goal_x': 1.366, 'goal_y': -1.266, 'goal_z': 6.213,
    'anchor_sep': 4.64,
    'start_r_x': 0.597, 'start_r_y': 3.547, 'start_r_z': 9.562,
    'start_l_x': -3.987, 'start_l_y': 2.848, 'start_l_z': 9.673,
    'speed_p50': 0.60, 'speed_p90': 1.02, 'speed_max': 1.89,
    'cam_yaw': 316.5, 'cam_pitch': -24.0, 'cam_roll': 6.0,
    'hole_radius_mm': 25.0,
    'cloth_w_m': 0.344, 'cloth_h_m': 0.292,
    'calib_trans_mm': 38.6, 'calib_rot_deg': 2.17,
}


def pctl_of(samples, value):
    """Where `value` falls in `samples`, as a percentile."""
    s = np.asarray(samples, dtype=np.float64)
    s = s[np.isfinite(s)]
    if len(s) == 0:
        return float('nan')
    return float((s < value).mean() * 100.0)


def verdict(p):
    if not np.isfinite(p):
        return 'NO DATA'
    if p <= 0.0 or p >= 100.0:
        return 'FAIL outside'
    if p < 5.0 or p > 95.0:
        return 'FAIL tail'
    if p < 15.0 or p > 85.0:
        return 'warn'
    return 'ok'


def planarity_rms(pts):
    """RMS distance from the best-fit plane, a scale-free deformation proxy."""
    p = np.asarray(pts, dtype=np.float64).reshape(-1, 3)
    p = p[np.isfinite(p).all(axis=1)]
    if len(p) < 4:
        return np.nan
    c = p.mean(axis=0)
    n = np.linalg.svd(p - c, full_matrices=False)[2][2]
    return float(np.sqrt(np.mean(((p - c) @ n) ** 2)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--demo_path', required=True)
    ap.add_argument('--out', default='coverage.png')
    ap.add_argument('--max_demos', type=int, default=0,
                    help='0 = all. Cap it for a quick look on a big dir.')
    ap.add_argument('--real_cloth_npz', default=None,
                    help='Optional segmented REAL cloth cloud in sim units '
                         '(cloth_sim_*.npz), overlaid on the deformation panel.')
    a = ap.parse_args()

    files = sorted(f for f in os.listdir(a.demo_path)
                   if f.startswith('demo_') and f.endswith('.pkl'))
    if a.max_demos:
        files = files[:a.max_demos]
    if not files:
        raise SystemExit(f'no demo_*.pkl in {a.demo_path}')

    D = {k: [] for k in ('goal', 'speed', 'sep', 'start_r', 'start_l', 'kind',
                         'cam', 'roll', 'hole_r', 'cloth_w', 'rest', 'eplen',
                         'defo_thread', 'defo_chain', 'npts', 'goals_hit',
                         'goals_tot', 'early', 'brake', 'unreliable')}
    max_act_vel = None
    for fn in files:
        with open(os.path.join(a.demo_path, fn), 'rb') as f:
            d = pickle.load(f)
        kind = d.get('episode_kind', 'thread')
        D['kind'].append(kind)
        goal = np.asarray(d['obs']['goal'], np.float64) * WBOX
        D['goal'].append(goal)
        grip = np.asarray(d['obs']['grip'], np.float64) * WBOX
        # grip layout is [a_pos(3) a_vel(3) b_pos(3) b_vel(3)] per frame
        pa, pb = grip[:, 0:3], grip[:, 6:9]
        D['start_r'].append(pa[0])
        D['start_l'].append(pb[0])
        D['sep'].append(float(np.linalg.norm(pa[0] - pb[0])))
        mav = float(d.get('max_act_vel', 4.0))
        max_act_vel = mav if max_act_vel is None else max_act_vel
        acts = np.asarray(d['acts'], np.float64) * mav
        sp = np.linalg.norm(acts.reshape(len(acts), 2, 3), axis=2).reshape(-1)
        D['speed'].append(sp[sp > 1e-6])
        D['eplen'].append(int(d.get('len', len(acts))))
        vm = d.get('cam_viewmat')
        if vm is not None:
            D['cam'].append([float(x) for x in vm])
        D['roll'].append(float(d.get('cam_roll_deg', 0.0)))
        D['hole_r'].append(float(d.get('hole_radius', np.nan)))
        D['cloth_w'].append(float(d.get('cloth_width', np.nan)))
        D['rest'].append(float(d.get('rest_extent', np.nan)))
        if kind == 'chain':
            D['goals_hit'].append(int(d.get('chain_goals_reached', 0)))
            D['goals_tot'].append(int(d.get('chain_goals_total', 0)))
            D['early'].append(bool(d.get('chain_ended_early', False)))
            D['brake'].append(int(d.get('chain_brake_steps', 0)))
            D['unreliable'].append(int(d.get('chain_unreliable_hole_frames', 0)))
        pcd = d['obs'].get('pcd')
        if pcd is not None:
            pcd = np.asarray(pcd)
            ix = np.linspace(0, len(pcd) - 1, min(8, len(pcd))).astype(int)
            D['npts'] += [len(np.unique(np.round(pcd[i], 5), axis=0)) for i in ix]
            key = 'defo_chain' if kind == 'chain' else 'defo_thread'
            w = float(d.get('cloth_width', 1.0)) or 1.0
            D[key] += [planarity_rms(pcd[i]) / w for i in ix]

    goal_all = np.concatenate(D['goal'])
    speed_all = np.concatenate(D['speed']) if D['speed'] else np.zeros(1)
    cam = np.asarray(D['cam'], np.float64) if D['cam'] else np.zeros((1, 6))
    start_r = np.asarray(D['start_r']); start_l = np.asarray(D['start_l'])
    n_chain = sum(1 for k in D['kind'] if k == 'chain')

    print(f'\n=== dataset coverage: {a.demo_path} ===')
    print(f'{len(files)} demos ({n_chain} chain, {len(files)-n_chain} thread), '
          f'{int(np.sum(D["eplen"]))} frames total')
    if D['goals_hit']:
        print(f'chain goals reached {sum(D["goals_hit"])}/{sum(D["goals_tot"])}; '
              f'{sum(D["early"])}/{n_chain} episodes ended early on the env '
              f'velocity abort; {sum(D["unreliable"])} frames fell back to the '
              f'position-only arrival test (degenerate hole)')

    rows = [
        ('goal x', goal_all[:, 0], REAL['goal_x']),
        ('goal y', goal_all[:, 1], REAL['goal_y']),
        ('goal z', goal_all[:, 2], REAL['goal_z']),
        ('anchor separation', np.asarray(D['sep']), REAL['anchor_sep']),
        ('start anchor0 x', start_r[:, 0], REAL['start_r_x']),
        ('start anchor0 y', start_r[:, 1], REAL['start_r_y']),
        ('start anchor0 z', start_r[:, 2], REAL['start_r_z']),
        ('start anchor1 x', start_l[:, 0], REAL['start_l_x']),
        ('start anchor1 y', start_l[:, 1], REAL['start_l_y']),
        ('start anchor1 z', start_l[:, 2], REAL['start_l_z']),
        ('speed (p50 ref)', speed_all, REAL['speed_p50']),
        ('speed (p90 ref)', speed_all, REAL['speed_p90']),
        ('cam yaw', cam[:, 2], REAL['cam_yaw']),
        ('cam pitch', cam[:, 1], REAL['cam_pitch']),
        ('cam roll', np.asarray(D['roll']), REAL['cam_roll']),
        ('hole radius (mm)', np.asarray(D['hole_r']) * SCALE_MM,
         REAL['hole_radius_mm']),
        # rest_extent is the cloth's AABB DIAGONAL, so compare it with the
        # real cloth's diagonal, not its side. `cloth_width` is the anchor
        # separation and is already covered by its own row above.
        ('cloth diagonal (m)', np.asarray(D['rest']) * SCALE_MM / 1000.0,
         float(np.hypot(REAL['cloth_w_m'], REAL['cloth_h_m']))),
    ]
    print(f'\n{"axis":<20s} {"sim p1":>9s} {"sim p99":>9s} {"real":>9s} '
          f'{"real pctl":>10s}  verdict')
    fails = 0
    for name, samples, real in rows:
        s = np.asarray(samples, np.float64)
        s = s[np.isfinite(s)]
        if len(s) == 0:
            print(f'{name:<20s} {"-":>9s} {"-":>9s} {real:9.2f} '
                  f'{"-":>10s}  NO DATA')
            continue
        p = pctl_of(s, real)
        v = verdict(p)
        fails += v.startswith('FAIL')
        print(f'{name:<20s} {np.percentile(s,1):9.2f} {np.percentile(s,99):9.2f} '
              f'{real:9.2f} {p:9.1f}%  {v}')
    print(f'\n{fails} axis/axes fail the coverage rule '
          f'(real must sit inside p5-p95).')

    # ---- panels ------------------------------------------------------------
    fig, ax = plt.subplots(2, 4, figsize=(23, 10))
    ax = ax.ravel()

    for i, (j, k, lj, lk) in enumerate([(0, 2, 'x', 'z'), (1, 2, 'y', 'z')]):
        ax[i].scatter(goal_all[::13, j], goal_all[::13, k], s=3, alpha=0.3,
                      label='sim goals')
        ax[i].scatter([REAL[f'goal_{lj}']], [REAL[f'goal_{lk}']], marker='*',
                      s=380, c='#ff7f0e', edgecolors='k', zorder=5, label='real')
        ax[i].set(xlabel=f'goal {lj}', ylabel=f'goal {lk}',
                  title=f'goal coverage {lj}-{lk}')
        ax[i].legend(fontsize=8)

    ax[2].scatter(cam[:, 2], cam[:, 1], s=22, alpha=0.7, label='sim episodes')
    ax[2].scatter([REAL['cam_yaw']], [REAL['cam_pitch']], marker='*', s=380,
                  c='#ff7f0e', edgecolors='k', zorder=5, label='real ZED')
    ax[2].set(xlabel='cam yaw (deg)', ylabel='cam pitch (deg)',
              title='camera pose')
    ax[2].legend(fontsize=8)

    ax[3].hist(speed_all, bins=60, color='#4c78a8')
    for lbl, key, col in [('real p50', 'speed_p50', '#ff7f0e'),
                          ('real p90', 'speed_p90', '#d62728'),
                          ('real max', 'speed_max', '#7f7f7f')]:
        ax[3].axvline(REAL[key], color=col, ls='--', label=lbl)
    if max_act_vel:
        ax[3].axvline(max_act_vel, color='k', ls=':', label='MAX_ACT_VEL')
    ax[3].set(xlabel='commanded anchor speed (sim units/s)',
              title='speed vs real teleop')
    ax[3].legend(fontsize=8)

    ax[4].scatter(start_r[:, 1], start_r[:, 2], s=22, alpha=0.7, label='sim a0')
    ax[4].scatter(start_l[:, 1], start_l[:, 2], s=22, alpha=0.7, label='sim a1')
    ax[4].scatter([REAL['start_r_y']], [REAL['start_r_z']], marker='*', s=340,
                  c='#ff7f0e', edgecolors='k', zorder=5, label='real a0')
    ax[4].scatter([REAL['start_l_y']], [REAL['start_l_z']], marker='*', s=340,
                  c='#d62728', edgecolors='k', zorder=5, label='real a1')
    ax[4].set(xlabel='start y', ylabel='start z', title='start anchors')
    ax[4].legend(fontsize=8)

    hr = np.asarray(D['hole_r']) * SCALE_MM
    ax[5].hist(hr[np.isfinite(hr)], bins=25, color='#59a14f')
    ax[5].axvline(REAL['hole_radius_mm'], color='#ff7f0e', ls='--',
                  label='real 25.0 mm')
    ax[5].set(xlabel='hole radius (mm)', title='hole size')
    ax[5].legend(fontsize=8)

    dt = np.asarray([x for x in D['defo_thread'] if np.isfinite(x)])
    dc = np.asarray([x for x in D['defo_chain'] if np.isfinite(x)])
    bins = np.linspace(0, max(0.02, float(np.percentile(
        np.concatenate([dt, dc]) if len(dt) + len(dc) else [0.02], 99))), 40)
    if len(dt):
        ax[6].hist(dt, bins=bins, alpha=0.6, label=f'thread (n={len(dt)})',
                   color='#4c78a8', density=True)
    if len(dc):
        ax[6].hist(dc, bins=bins, alpha=0.6, label=f'chain (n={len(dc)})',
                   color='#e45756', density=True)
    if a.real_cloth_npz and os.path.exists(a.real_cloth_npz):
        z = np.load(a.real_cloth_npz)
        keys = [k for k in z.files if k != 'area']
        rv = [planarity_rms(z[k]) / (REAL['cloth_w_m'] * 1000 / SCALE_MM)
              for k in keys]
        for v in rv:
            ax[6].axvline(v, color='#ff7f0e', ls='--', alpha=0.8)
        ax[6].plot([], [], color='#ff7f0e', ls='--', label='real demo frames')
    ax[6].set(xlabel='out-of-plane RMS / cloth width',
              title='deformation: chain should exceed thread')
    ax[6].legend(fontsize=8)

    npts = np.asarray(D['npts'])
    if len(npts):
        ax[7].hist(npts, bins=30, color='#b07aa1')
        ax[7].axvline(2048, color='#d62728', ls='--',
                      label='2048 requested')
        ax[7].set(xlabel='distinct points per frame',
                  title='PCD health (below the line = duplicated points)')
        ax[7].legend(fontsize=8)
    for x in ax:
        x.grid(alpha=0.25)
    fig.suptitle(f'v5 dataset coverage — {a.demo_path} ({len(files)} demos)')
    fig.tight_layout()
    fig.savefig(a.out, dpi=110)
    print(f'\nwrote {a.out}')
    if len(dt) and len(dc):
        print(f'deformation median: thread {np.median(dt):.4f}  '
              f'chain {np.median(dc):.4f}  '
              f'ratio {np.median(dc)/max(np.median(dt),1e-9):.2f}x')


if __name__ == '__main__':
    main()
