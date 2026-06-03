"""
HoleEstimator — recover the privileged hole-location from a partial point cloud
using a UniClothDiff GPS state-estimation model.

The state-estimation model reconstructs the FULL cloth mesh from a partial cloud
(+ rest template + topology). We then index the known hole-loop vertices in the
predicted mesh and take their centroid — the same 3-d quantity the privileged
diffusion BC policy consumes.

Two backends:

  remote (default) — talks to a UniClothDiff websocket server
      (UniClothDiff/scripts/serve_predictor.py) over the openpi-style
      msgpack+websockets protocol. This is the cross-environment path: the model
      runs in the UniClothDiff env (py3.12 / torch 2.9) while this client runs
      in the dedo env (py3.8 / pybullet). Only ``websockets`` + ``msgpack`` are
      needed here — NO torch / UniClothDiff import.

  local — imports the UniClothDiff pipeline directly and runs inference in
      process. Only usable when called FROM the UniClothDiff env (e.g. offline
      hole-centroid regeneration for policy fine-tuning, or benchmarking). All
      heavy imports are lazy so this module stays importable under py3.8.

Wire protocol / preprocessing mirror (keep in sync with):
  UniClothDiff/src/serving/msgpack_numpy.py
  UniClothDiff/src/serving/preprocessing.py  (preprocess/postprocess_state_est)
  UniClothDiff/src/serving/websocket_server.py

Example (dedo env, server already running):
  est = HoleEstimator(host="localhost", port=8000)
  centroid = est.estimate(pcd_world, rest_positions, hole_vertex_indices, edges)
"""
from __future__ import annotations

import numpy as np


# ---------------------------------------------------------------------------
# msgpack-numpy codec (vendored from UniClothDiff/src/serving/msgpack_numpy.py).
# ---------------------------------------------------------------------------
def _pack_array(obj):
    import numpy as _np
    if isinstance(obj, (_np.ndarray, _np.generic)) and obj.dtype.kind in ("V", "O", "c"):
        raise ValueError(f"Unsupported dtype for wire encoding: {obj.dtype}")
    if isinstance(obj, _np.ndarray):
        return {b"__ndarray__": True, b"data": _np.ascontiguousarray(obj).tobytes(),
                b"dtype": obj.dtype.str, b"shape": obj.shape}
    if isinstance(obj, _np.generic):
        return {b"__npgeneric__": True, b"data": obj.item(), b"dtype": obj.dtype.str}
    return obj


def _unpack_array(obj):
    if b"__ndarray__" in obj:
        return np.ndarray(buffer=obj[b"data"], dtype=np.dtype(obj[b"dtype"]),
                          shape=obj[b"shape"])
    if b"__npgeneric__" in obj:
        return np.dtype(obj[b"dtype"]).type(obj[b"data"])
    return obj


def _packb(obj):
    import msgpack
    return msgpack.packb(obj, default=_pack_array)


def _unpackb(data):
    import msgpack
    return msgpack.unpackb(data, object_hook=_unpack_array, raw=False)


# ---------------------------------------------------------------------------
# Preprocessing (mirror of preprocess/postprocess_state_est, "self" centering).
# ---------------------------------------------------------------------------
def _sample_points(pcd, target, rng):
    n = pcd.shape[0]
    if n == 0:
        raise ValueError("point_cloud is empty")
    mask = np.zeros(target, dtype=np.bool_)
    if n >= target:
        idx = rng.choice(n, target, replace=False)
        mask[:] = True
        return pcd[idx].astype(np.float32), mask
    out = np.empty((target, 3), dtype=np.float32)
    out[:n] = pcd
    out[n:] = pcd[rng.integers(0, n, size=target - n)]
    mask[:n] = True
    return out, mask


def _center_rest(rest, centroid, mode, V):
    out = rest.copy()
    shift = out[:V].mean(axis=0) if mode == "self" else np.asarray(centroid, out.dtype)
    out[:V] = out[:V] - shift
    return out


def _rest_scale(rest, V, eps=1e-6):
    """Radius of gyration of the rest mesh's valid verts — must match
    src/utils/rest_pos.py::rest_pos_scale so train/inference normalization agree."""
    v = rest[:V]
    c = v.mean(axis=0)
    rg = float(np.sqrt(np.mean(np.sum((v - c) ** 2, axis=1))))
    return max(rg, eps)


def _preprocess_state_est(point_cloud, rest_positions, edges, metadata,
                          num_inference_steps=None, seed=0):
    max_V = int(metadata["max_num_nodes"])
    max_E = int(metadata["max_num_edges"])
    n_pts = int(metadata["num_sample_points"])
    mode = metadata.get("rest_pos_centering", "pcd")
    steps = int(num_inference_steps if num_inference_steps is not None
                else metadata["default_num_inference_steps"])

    pcd = np.asarray(point_cloud, dtype=np.float32).reshape(-1, 3)
    rest = np.asarray(rest_positions, dtype=np.float32).reshape(-1, 3)
    V = rest.shape[0]
    if max_V < V:
        raise ValueError(f"rest mesh has {V} verts > server max_num_nodes={max_V}")
    if edges is not None:
        edges = np.asarray(edges, dtype=np.int64).reshape(-1, 2)
        if len(edges) > max_E:
            raise ValueError(f"mesh has {len(edges)} edges > server max_num_edges={max_E}")

    rng = np.random.default_rng(seed)
    points, pcd_mask = _sample_points(pcd, n_pts, rng)
    centroid = points.mean(axis=0)
    points = points - centroid
    rest_centered = _center_rest(rest, centroid, mode, V)

    # Scale-normalize to ~unit if the checkpoint was trained that way (mirror of
    # the dataset/batched_inference normalization). The caller un-scales the
    # prediction by the same factor.
    scale = _rest_scale(rest_centered, V) if metadata.get("normalize_scale", False) else 1.0
    if scale != 1.0:
        points = points / scale
        rest_centered = rest_centered.copy()
        rest_centered[:V] = rest_centered[:V] / scale

    rest_pad = np.zeros((max_V, 3), dtype=np.float32)
    rest_pad[:V] = rest_centered
    edge_pad = np.zeros((max_E, 2), dtype=np.int64)
    if edges is not None:
        edge_pad[:len(edges)] = edges
    node_mask = np.zeros(max_V, dtype=np.bool_)
    node_mask[:V] = True

    obs = {
        "task": "state_estimation",
        "point_cloud": points,
        "pcd_mask": pcd_mask,
        "rest_positions": rest_pad,
        "edges": edge_pad,
        "node_mask": node_mask,
        "num_nodes": V,
        "num_inference_steps": steps,
        "seed": int(seed),
    }
    return obs, centroid.astype(np.float32), float(scale)


# ---------------------------------------------------------------------------
# Remote backend: thin websocket client.
# ---------------------------------------------------------------------------
class _RemoteStateEst:
    def __init__(self, host="localhost", port=8000, timeout=30.0):
        import websockets.sync.client as _wsc  # noqa: F401  (fail early if missing)
        self._wsc = _wsc
        self._uri = f"ws://{host}:{port}"
        self._timeout = timeout
        self._conn = None
        self.metadata = None
        self._connect()

    def _connect(self):
        self._conn = self._wsc.connect(self._uri, compression=None, max_size=None,
                                       open_timeout=self._timeout)
        self.metadata = _unpackb(self._conn.recv())
        if self.metadata.get("task") != "state_estimation":
            raise RuntimeError(
                f"server serves task {self.metadata.get('task')!r}, "
                f"expected 'state_estimation'")

    def infer(self, obs):
        self._conn.send(_packb(obs))
        resp = self._conn.recv()
        if isinstance(resp, str):  # server sends a traceback string on error
            raise RuntimeError(f"remote inference failed:\n{resp}")
        return _unpackb(resp)

    def close(self):
        if self._conn is not None:
            self._conn.close()
            self._conn = None


# ---------------------------------------------------------------------------
# Local backend: in-process UniClothDiff pipeline (UniClothDiff env only).
# ---------------------------------------------------------------------------
class _LocalStateEst:
    def __init__(self, checkpoint_dir, config_path, device=None,
                 num_inference_steps=50):
        import torch
        from omegaconf import OmegaConf
        from src.models.gps.state_est import GPSStateEstModel
        from src.pipelines.cloth_state_est_gps_pipeline import ClothStateEstGPSPipeline
        from src.registry import SCHEDULERS

        self._torch = torch
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        cfg = OmegaConf.load(config_path)
        model = GPSStateEstModel.from_pretrained(checkpoint_dir).to(self.device).eval()
        sched = SCHEDULERS.build(OmegaConf.to_container(cfg.diffusion_cfg, resolve=True))
        self.pipe = ClothStateEstGPSPipeline(model=model, scheduler=sched).to(self.device)
        self.metadata = {
            "task": "state_estimation",
            "max_num_nodes": int(cfg.dataset_cfg.max_num_nodes),
            "max_num_edges": int(cfg.dataset_cfg.max_num_edges),
            "num_sample_points": int(cfg.dataset_cfg.num_sample_points),
            "rest_pos_centering": getattr(model.config, "rest_pos_centering", "self"),
            "normalize_scale": bool(getattr(model.config, "normalize_scale", False)),
            "scale_gain": float(getattr(model.config, "scale_gain", 1.0)),
            "default_num_inference_steps": int(num_inference_steps),
        }

    def infer(self, obs):
        torch = self._torch
        dev = self.device
        max_V = int(np.asarray(obs["rest_positions"]).shape[0])
        V = int(obs["num_nodes"])

        def b(x, dt):
            return torch.from_numpy(np.asarray(x, dtype=dt)[None]).to(dev)

        out = self.pipe(
            encoder_hidden_states=b(obs["point_cloud"], np.float32),
            q_temp=b(obs["rest_positions"], np.float32),
            shape=(1, max_V, 3),
            edge_index=b(obs["edges"], np.int64),
            num_nodes=torch.tensor([V], dtype=torch.long, device=dev),
            node_mask=b(obs["node_mask"], np.bool_),
            pcd_mask=b(obs["pcd_mask"], np.bool_),
            num_inference_steps=int(obs["num_inference_steps"]),
            generator=torch.Generator(device=dev).manual_seed(int(obs["seed"])),
        )
        return {"vertices": out.result_tensor[0, :V].cpu().numpy().astype(np.float32)}


# ---------------------------------------------------------------------------
# Particle-filter tracker client (stateful: reset once, then step per frame).
# ---------------------------------------------------------------------------
class PFTrackerClient:
    """Client for the UniClothDiff particle-filter tracker server
    (UniClothDiff/scripts/serve_pf_tracker.py).

    The tracker is STATEFUL: call reset() at the start of each episode, then
    step() every frame. Pass point_cloud=None on occluded frames — the filter
    propagates with GNS dynamics only. All geometry is in the caller's WORLD
    frame (the filter centers/un-centers internally). grasped_velocity is the
    per-grasped-vertex per-step displacement in ASCENDING actuated-vertex order
    (matches the GNS dt=1.0 displacement convention)."""

    def __init__(self, host="localhost", port=8001, timeout=60.0):
        import websockets.sync.client as _wsc
        self._uri = f"ws://{host}:{port}"
        self._conn = _wsc.connect(self._uri, compression=None, max_size=None,
                                  open_timeout=timeout)
        self.metadata = _unpackb(self._conn.recv())
        if self.metadata.get("task") != "pf_tracker":
            raise RuntimeError(
                f"server serves task {self.metadata.get('task')!r}, "
                f"expected 'pf_tracker'")
        # Most-recent fused estimate, stashed for viz (world frame).
        self.last_mesh = None
        self.last_centroid = None

    def _rpc(self, msg):
        self._conn.send(_packb(msg))
        resp = self._conn.recv()
        if isinstance(resp, str):
            raise RuntimeError(f"tracker error:\n{resp}")
        out = _unpackb(resp)
        if isinstance(out, dict) and "mesh" in out:
            self.last_mesh = np.asarray(out["mesh"], np.float32)
            self.last_centroid = np.asarray(out["hole_centroid"], np.float32)
        return out

    def reset(self, rest_positions, edges, faces, actuated_vertices,
              hole_vertex_indices, point_cloud):
        """Seed the filter from the first point cloud. Returns hole centroid (3,)."""
        out = self._rpc({
            "cmd": "reset",
            "rest_positions": np.asarray(rest_positions, np.float32),
            "edges": np.asarray(edges, np.int64),
            "faces": np.asarray(faces, np.int64),
            "actuated_vertices": np.asarray(actuated_vertices, np.int64),
            "num_nodes": int(np.asarray(rest_positions).shape[0]),
            "hole_vertex_indices": np.asarray(hole_vertex_indices, np.int64),
            "point_cloud": np.asarray(point_cloud, np.float32),
        })
        return np.asarray(out["hole_centroid"], np.float32)

    def step(self, grasped_velocity, point_cloud=None):
        """Advance one frame. point_cloud=None => GNS-only (occluded). Returns
        hole centroid (3,) in the world frame."""
        msg = {"cmd": "step",
               "grasped_velocity": np.asarray(grasped_velocity, np.float32)}
        if point_cloud is not None:
            msg["point_cloud"] = np.asarray(point_cloud, np.float32)
        out = self._rpc(msg)
        return np.asarray(out["hole_centroid"], np.float32)

    def close(self):
        if self._conn is not None:
            self._conn.close()
            self._conn = None


# ---------------------------------------------------------------------------
# Public estimator.
# ---------------------------------------------------------------------------
class HoleEstimator:
    """Estimate the cloth hole centroid (and full mesh) from a partial cloud."""

    def __init__(self, backend="remote", *, host="localhost", port=8000,
                 timeout=30.0, checkpoint_dir=None, config_path=None,
                 device=None, num_inference_steps=50):
        self.backend = backend
        self.default_steps = num_inference_steps
        if backend == "remote":
            self._impl = _RemoteStateEst(host=host, port=port, timeout=timeout)
        elif backend == "local":
            if checkpoint_dir is None or config_path is None:
                raise ValueError("local backend needs checkpoint_dir and config_path")
            self._impl = _LocalStateEst(checkpoint_dir, config_path, device=device,
                                        num_inference_steps=num_inference_steps)
        else:
            raise ValueError(f"unknown backend {backend!r}")
        self.metadata = self._impl.metadata
        # Most-recent prediction, stashed for visualization (world frame).
        self.last_mesh = None
        self.last_centroid = None

    def predict_mesh(self, pcd_world, rest_positions, edges=None,
                     num_inference_steps=None, seed=0):
        """Reconstruct the full cloth mesh (V, 3) in the caller's WORLD frame."""
        obs, centroid, scale = _preprocess_state_est(
            pcd_world, rest_positions, edges, self.metadata,
            num_inference_steps=num_inference_steps, seed=seed)
        result = self._impl.infer(obs)
        gain = float(self.metadata.get("scale_gain", 1.0))
        return np.asarray(result["vertices"], dtype=np.float32) * (scale * gain) + centroid

    def estimate(self, pcd_world, rest_positions, hole_vertex_indices,
                 edges=None, num_inference_steps=None, seed=0):
        """Return the estimated hole centroid (3,) in the caller's WORLD frame."""
        mesh = self.predict_mesh(pcd_world, rest_positions, edges,
                                 num_inference_steps=num_inference_steps, seed=seed)
        idx = np.asarray(hole_vertex_indices, dtype=np.int64)
        centroid = mesh[idx].mean(axis=0).astype(np.float32)
        self.last_mesh = mesh           # stash for viz overlay
        self.last_centroid = centroid
        return centroid

    def close(self):
        if hasattr(self._impl, "close"):
            self._impl.close()
