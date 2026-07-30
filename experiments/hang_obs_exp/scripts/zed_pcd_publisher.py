#!/usr/bin/env python3
"""Standalone ZED point-cloud publisher.

Run this ON THE ZED HOST (the Linux/Windows machine with the ZED SDK +
CUDA installed, in that machine's ZED-SDK Python env). It captures the
ZED point cloud, downsamples it, and either:

  - streams it over a plain TCP socket (``--mode tcp``), and/or
  - atomically rewrites a single ``.npy`` file (``--mode file``).

The dedo viewer (``view_demo_inspect.py``) reads either source on a
background thread and renders the cloud under ``/real_workspace`` so you
can hand-align it against the sim cloud with the GUI pose sliders.

This file is intentionally dependency-light: ``pyzed`` + ``numpy`` +
the Python stdlib only. ``pyzed`` is imported lazily inside ``main()``
so the file stays importable / byte-compilable on machines without the
ZED SDK (e.g. the macOS viewer host).

Wire format (one frame):
    4 bytes   big-endian uint32 = payload length L
    L bytes   numpy .npy buffer of a float32 array, shape (N, 6):
              columns = x, y, z, r, g, b   (xyz metres; rgb in 0..1)

Coordinate frame: we ask the ZED SDK for RIGHT_HANDED_Z_UP metric
coords, so the cloud arrives in the ZED's own frame with +Z up. You
then place it into the Franka world frame interactively via the
viewer's "ZED real cloud" pose sliders (no extrinsics calibration
needed for visual alignment).

Examples
--------
TCP stream on all interfaces, port 5556, ~15k pts, HD720 @ ~15 fps::

    python zed_pcd_publisher.py --mode tcp --host 0.0.0.0 --port 5556 \
        --max-points 15000 --resolution HD720 --fps 15

Atomic file (e.g. into a directory synced/mounted to the viewer host)::

    python zed_pcd_publisher.py --mode file --out-file /tmp/zed_pcd.npy
"""
import argparse
import io
import os
import socket
import struct
import sys
import threading
import time

import numpy as np


def _encode_frame(xyzrgb: np.ndarray) -> bytes:
    """(N,6) float32 -> length-prefixed .npy bytes."""
    buf = io.BytesIO()
    np.save(buf, np.ascontiguousarray(xyzrgb, dtype=np.float32),
            allow_pickle=False)
    payload = buf.getvalue()
    return struct.pack('>I', len(payload)) + payload


def _decode_zed_cloud(raw: np.ndarray, max_points: int):
    """ZED XYZRGBA measure (H,W,4 float32) -> (M,6) float32 cloud.

    Column 3 is RGBA packed into a float32; we reinterpret its bytes as
    4x uint8. NaN/inf points (no depth) are dropped, then the cloud is
    uniformly random-subsampled to <= max_points.
    """
    flat = raw.reshape(-1, 4)
    xyz = flat[:, :3]
    finite = np.isfinite(xyz).all(axis=1)
    flat = flat[finite]
    if flat.shape[0] == 0:
        return np.empty((0, 6), dtype=np.float32)

    rgba_u8 = flat[:, 3].copy().view(np.uint8).reshape(-1, 4)
    rgb = rgba_u8[:, :3].astype(np.float32) / 255.0

    out = np.empty((flat.shape[0], 6), dtype=np.float32)
    out[:, :3] = flat[:, :3]
    out[:, 3:] = rgb

    if max_points > 0 and out.shape[0] > max_points:
        idx = np.random.choice(out.shape[0], max_points, replace=False)
        out = out[idx]
    return out


# --------------------------------------------------------------------------
# TCP server: one thread per client, each gets the latest frame as it is
# produced. Slow clients just see dropped frames (we only keep latest).
# --------------------------------------------------------------------------
class _FrameHub:
    def __init__(self):
        self._lock = threading.Lock()
        self._frame = None          # encoded bytes (length-prefixed)
        self._seq = 0
        self._cv = threading.Condition(self._lock)

    def publish(self, encoded: bytes):
        with self._cv:
            self._frame = encoded
            self._seq += 1
            self._cv.notify_all()

    def wait_next(self, last_seq, timeout=5.0):
        with self._cv:
            if self._seq == last_seq:
                self._cv.wait(timeout=timeout)
            return self._frame, self._seq


def _serve_tcp(hub: _FrameHub, host: str, port: int):
    srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    srv.bind((host, port))
    srv.listen(8)
    print(f'[zed-pub] TCP serving on {host}:{port}', flush=True)

    def _client(conn, addr):
        print(f'[zed-pub] client connected: {addr}', flush=True)
        conn.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        last = -1
        try:
            while True:
                frame, last = hub.wait_next(last)
                if frame is None:
                    continue
                conn.sendall(frame)
        except (BrokenPipeError, ConnectionResetError, OSError):
            pass
        finally:
            conn.close()
            print(f'[zed-pub] client gone: {addr}', flush=True)

    while True:
        conn, addr = srv.accept()
        threading.Thread(target=_client, args=(conn, addr),
                          daemon=True).start()


def _atomic_write_npy(path: str, xyzrgb: np.ndarray):
    """Write .npy so the viewer never reads a half-written file."""
    tmp = f'{path}.tmp.{os.getpid()}'
    np.save(tmp, np.ascontiguousarray(xyzrgb, dtype=np.float32),
            allow_pickle=False)
    # np.save appends .npy if missing; normalize.
    if not os.path.exists(tmp) and os.path.exists(tmp + '.npy'):
        tmp = tmp + '.npy'
    os.replace(tmp, path)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--mode', choices=['tcp', 'file', 'both'],
                    default='tcp')
    ap.add_argument('--host', default='0.0.0.0',
                    help='TCP bind host (default all interfaces).')
    ap.add_argument('--port', type=int, default=5556)
    ap.add_argument('--out-file', default='/tmp/zed_pcd.npy',
                    help='Atomic .npy path for --mode file/both.')
    ap.add_argument('--max-points', type=int, default=15000,
                    help='Random-subsample each frame to <= this many '
                         'points (bandwidth + viewer perf). 0 = keep all.')
    ap.add_argument('--resolution', default='HD720',
                    choices=['HD2K', 'HD1080', 'HD720', 'VGA'])
    ap.add_argument('--depth-mode', default='NEURAL',
                    choices=['NEURAL', 'ULTRA', 'QUALITY', 'PERFORMANCE'])
    ap.add_argument('--fps', type=int, default=15,
                    help='Target capture/publish rate cap.')
    ap.add_argument('--max-depth', type=float, default=3.0,
                    help='Clip depth beyond this many metres.')
    args = ap.parse_args()

    try:
        import pyzed.sl as sl
    except ImportError:
        sys.exit('[zed-pub] ERROR: pyzed not found. Run this on the ZED '
                 'host inside its ZED-SDK Python environment.')

    init = sl.InitParameters()
    init.coordinate_units = sl.UNIT.METER
    init.coordinate_system = sl.COORDINATE_SYSTEM.RIGHT_HANDED_Z_UP
    init.camera_resolution = getattr(sl.RESOLUTION, args.resolution)
    init.depth_mode = getattr(sl.DEPTH_MODE, args.depth_mode)
    init.depth_maximum_distance = float(args.max_depth)

    zed = sl.Camera()
    status = zed.open(init)
    if status != sl.ERROR_CODE.SUCCESS:
        sys.exit(f'[zed-pub] ERROR: zed.open() -> {status}')
    print(f'[zed-pub] ZED open: {args.resolution} {args.depth_mode}',
          flush=True)

    hub = _FrameHub()
    if args.mode in ('tcp', 'both'):
        threading.Thread(target=_serve_tcp,
                         args=(hub, args.host, args.port),
                         daemon=True).start()

    runtime = sl.RuntimeParameters()
    cloud = sl.Mat()
    period = 1.0 / max(1, args.fps)
    n_pub = 0
    try:
        while True:
            t0 = time.time()
            if zed.grab(runtime) != sl.ERROR_CODE.SUCCESS:
                continue
            zed.retrieve_measure(cloud, sl.MEASURE.XYZRGBA)
            xyzrgb = _decode_zed_cloud(np.asarray(cloud.get_data()),
                                       args.max_points)
            if xyzrgb.shape[0] == 0:
                continue
            if args.mode in ('tcp', 'both'):
                hub.publish(_encode_frame(xyzrgb))
            if args.mode in ('file', 'both'):
                _atomic_write_npy(args.out_file, xyzrgb)
            n_pub += 1
            if n_pub % 30 == 0:
                print(f'[zed-pub] {n_pub} frames, last '
                      f'{xyzrgb.shape[0]} pts', flush=True)
            dt = time.time() - t0
            if dt < period:
                time.sleep(period - dt)
    except KeyboardInterrupt:
        print('\n[zed-pub] stopping.', flush=True)
    finally:
        zed.close()


if __name__ == '__main__':
    main()
