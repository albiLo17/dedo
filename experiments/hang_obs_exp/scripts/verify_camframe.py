#!/Users/evanrobert/miniconda3/envs/dedo/bin/python
"""Overlay a sim-world trajectory and its camera-frame version in viser.

Verification tool: loads the ORIGINAL rollout (sim-world frame) and a
CAMFRAME rollout (the same data expressed in the sim camera's frame),
and renders both. The camframe data is added as a child of a "/cam"
frame placed at the camera's world pose, so if the camera transform was
done correctly the two overlays land exactly on top of each other.

Each trajectory has its own visibility checkbox. A live "alignment
residual" readout shows the mean per-point distance between the
camframe pcd projected back to world and the original world pcd at the
current frame (≈0 ⇒ correct).

Pure viser + numpy — no gym/pybullet, so it runs anywhere viser does.

    python verify_camframe.py \
        --orig experiments/hang_obs_exp/hemal_traj_ep001_seed12025_l1_h1_t0.pkl \
        --cam  experiments/hang_obs_exp/traj_ep001_seed12025_l1_h1_t0_camframe.pkl
"""
import argparse
import pickle
import threading
import time
from pathlib import Path

import numpy as np
import viser

# Default sim-camera pose in the sim-world frame (from the user).
CAM_POS = (9.8618, -9.8618, 6.7202)
CAM_WXYZ = (-0.6242, 0.6812, 0.2821, -0.2585)   # viser wxyz
R_CAM_TO_WORLD = np.array([[0.707107,  0.061628, -0.704416],
                           [0.707107, -0.061628,  0.704416],
                           [0.0,      -0.996195, -0.087156]], dtype=np.float64)
PROPRIO_SCALE = 20.0     # grip/goal were divided by this on export


def load(path):
    d = pickle.load(open(Path(path).expanduser(), 'rb'))
    obs = d['obs']
    grip = np.asarray(obs['grip'], np.float32)
    half = grip.shape[1] // 2
    return {
        'pcd':  np.asarray(obs['pcd'], np.float32),            # (T,N,3)
        'ee_l': grip[:, 0:3] * PROPRIO_SCALE,                  # (T,3)
        'ee_r': grip[:, half:half + 3] * PROPRIO_SCALE,        # (T,3)
        'goal': np.asarray(obs['goal'], np.float32) * PROPRIO_SCALE,
        'T':    int(grip.shape[0]),
        'succ': (d.get('success_hanging'), d.get('success_topological'),
                 d.get('success_legacy')),
        'cam_frame': bool(d.get('cam_frame', False)),
    }


def _poly(pts):
    p = np.asarray(pts, np.float32)
    return np.stack([p[:-1], p[1:]], axis=1)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--orig', required=True, help='Sim-world-frame pkl.')
    ap.add_argument('--cam', required=True, help='Camera-frame pkl.')
    ap.add_argument('--port', type=int, default=8080)
    ap.add_argument('--cam_pos', type=float, nargs=3, default=list(CAM_POS))
    ap.add_argument('--cam_wxyz', type=float, nargs=4, default=list(CAM_WXYZ))
    args = ap.parse_args()

    O = load(args.orig)
    C = load(args.cam)
    n = min(O['T'], C['T'])
    print(f'[verify] original T={O["T"]} succ(h/t/l)={O["succ"]}  '
          f'cam_frame={O["cam_frame"]}')
    print(f'[verify] camframe T={C["T"]} succ(h/t/l)={C["succ"]}  '
          f'cam_frame={C["cam_frame"]}')

    server = viser.ViserServer(port=args.port)
    server.scene.world_axes.visible = True

    # World-frame group for the ORIGINAL trajectory (identity pose).
    orig_root = server.scene.add_frame('/orig', show_axes=False)
    # Camera frame: camframe-trajectory children are interpreted in
    # camera coords and land at the right world location.
    cam_root = server.scene.add_frame(
        '/cam', show_axes=True, axes_length=0.5, axes_radius=0.01,
        position=np.asarray(args.cam_pos, np.float32),
        wxyz=np.asarray(args.cam_wxyz, np.float32))
    server.scene.add_label('/cam/label', text='sim camera',
                           position=(0.0, 0.0, 0.0))

    # Static EE polylines + goal (added once; ride their parent frame).
    def _add_static(root, traj, cL, cR, cG):
        server.scene.add_line_segments(
            f'{root}/ee_l_path', points=_poly(traj['ee_l']),
            colors=np.asarray(cL, np.uint8), line_width=2.0)
        server.scene.add_line_segments(
            f'{root}/ee_r_path', points=_poly(traj['ee_r']),
            colors=np.asarray(cR, np.uint8), line_width=2.0)
        server.scene.add_icosphere(
            f'{root}/goal', radius=0.4,
            position=tuple(float(x) for x in traj['goal'][0]),
            color=cG)

    _add_static('/orig', O, (60, 200, 110), (40, 160, 90), (255, 0, 200))
    _add_static('/cam',  C, (255, 150, 40), (235, 110, 20), (255, 255, 255))

    # GUI -------------------------------------------------------------
    chk_orig = server.gui.add_checkbox('Show original (world)',
                                       initial_value=True)
    chk_cam = server.gui.add_checkbox('Show camframe (via /cam)',
                                      initial_value=True)
    sld_frame = server.gui.add_slider('Frame', min=0, max=n - 1, step=1,
                                      initial_value=0)
    sld_size = server.gui.add_slider('Point size', min=0.005, max=0.2,
                                     step=0.005, initial_value=0.03)
    btn_play = server.gui.add_button('Play / Pause')
    sld_hz = server.gui.add_slider('Hz', min=1, max=30, step=1,
                                   initial_value=10)
    gui_stat = server.gui.add_text('Align residual', initial_value='—',
                                   disabled=True)

    @chk_orig.on_update
    def _(_e):
        orig_root.visible = chk_orig.value

    @chk_cam.on_update
    def _(_e):
        cam_root.visible = chk_cam.value

    state = {'play': False}

    @btn_play.on_click
    def _(_e):
        state['play'] = not state['play']

    def draw(f):
        ps = float(sld_size.value)
        # Original: world coords directly under /orig.
        server.scene.add_point_cloud(
            '/orig/pcd', points=O['pcd'][f],
            colors=np.broadcast_to(np.uint8([60, 200, 110]),
                                   O['pcd'][f].shape).copy(),
            point_size=ps)
        for nm, p, c in (('ee_l', O['ee_l'][f], (40, 120, 255)),
                         ('ee_r', O['ee_r'][f], (0, 220, 220))):
            server.scene.add_icosphere(f'/orig/{nm}', radius=ps * 4,
                                       position=tuple(map(float, p)),
                                       color=c)
        # Camframe: camera coords under /cam (rigidly placed in world).
        server.scene.add_point_cloud(
            '/cam/pcd', points=C['pcd'][f],
            colors=np.broadcast_to(np.uint8([255, 150, 40]),
                                   C['pcd'][f].shape).copy(),
            point_size=ps)
        for nm, p, c in (('ee_l', C['ee_l'][f], (255, 60, 200)),
                         ('ee_r', C['ee_r'][f], (255, 230, 0))):
            server.scene.add_icosphere(f'/cam/{nm}', radius=ps * 4,
                                       position=tuple(map(float, p)),
                                       color=c)
        # Quantitative check: project camframe pcd back to world and
        # compare to the original world pcd (same point order).
        back = C['pcd'][f].astype(np.float64) @ R_CAM_TO_WORLD.T \
            + np.asarray(args.cam_pos, np.float64)
        d = np.linalg.norm(back - O['pcd'][f].astype(np.float64), axis=1)
        gui_stat.value = (f'frame {f}/{n - 1}  mean={d.mean()*1000:.3f} mm  '
                          f'max={d.max()*1000:.3f} mm')

    draw(0)

    @sld_frame.on_update
    def _(_e):
        if not state['play']:
            draw(int(sld_frame.value))

    print(f'[verify] viser on http://localhost:{args.port}  '
          f'(Ctrl-C to quit)')
    try:
        while True:
            if state['play']:
                f = (int(sld_frame.value) + 1) % n
                sld_frame.value = f
                draw(f)
                time.sleep(1.0 / max(1, int(sld_hz.value)))
            else:
                time.sleep(0.05)
    except KeyboardInterrupt:
        print('\n[verify] bye.')


if __name__ == '__main__':
    main()
