"""Trajectory viewer for training runs.

Scans the repo for run directories that contain `eval_*_step<N>.mp4` videos
plus a `config.json`, then serves a single-page web UI to browse them, scrub
through training-step videos, and compare runs side-by-side.

Usage:
    python scripts/trajectory_viewer.py [--root .] [--port 8000] [--host 127.0.0.1]

Then open the printed URL in a browser.
"""

from __future__ import annotations

import argparse
import json
import mimetypes
import os
import re
import socketserver
import sys
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler
from pathlib import Path
from typing import Optional
from urllib.parse import parse_qs, quote, unquote, urlparse

STEP_RE = re.compile(r"step(\d+)", re.IGNORECASE)
EVAL_RE = re.compile(r"^eval_.*\.mp4$", re.IGNORECASE)
RUN_DIR_RE = re.compile(r"^(PPO|SAC|TD3|DDPG|A2C|DQN)_.*", re.IGNORECASE)


def find_runs(root: Path) -> list[dict]:
    """Walk `root` looking for run directories.

    A run directory is one whose name starts with an algorithm prefix and
    that contains either `config.json` or `agent.zip`. Run directories with
    no eval videos are skipped (nothing to view).
    """
    runs: list[dict] = []
    skip_dirs = {".git", "__pycache__", "node_modules", "wandb", "logs",
                 "rendered", "dedo.egg-info", ".claude"}

    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = [d for d in dirnames if d not in skip_dirs]
        name = os.path.basename(dirpath)
        if not RUN_DIR_RE.match(name):
            continue
        has_marker = "config.json" in filenames or "agent.zip" in filenames
        if not has_marker:
            continue

        videos: list[dict] = []
        for fn in filenames:
            if not EVAL_RE.match(fn):
                continue
            m = STEP_RE.search(fn)
            step = int(m.group(1)) if m else -1
            videos.append({"file": fn, "step": step})
        videos.sort(key=lambda v: v["step"])

        if not videos:
            continue

        rel = os.path.relpath(dirpath, root)
        group = rel.split(os.sep)[0] if os.sep in rel else "(root)"
        runs.append({
            "id": rel.replace(os.sep, "/"),
            "name": name,
            "group": group,
            "rel_path": rel.replace(os.sep, "/"),
            "videos": videos,
            "n_videos": len(videos),
            "min_step": videos[0]["step"],
            "max_step": videos[-1]["step"],
        })
        # Don't descend into a run dir we already cataloged.
        dirnames[:] = []

    runs.sort(key=lambda r: (r["group"], r["name"]))
    return runs


def load_run_detail(root: Path, rel_path: str) -> Optional[dict]:
    run_dir = (root / rel_path).resolve()
    if not str(run_dir).startswith(str(root.resolve())):
        return None
    if not run_dir.is_dir():
        return None

    config = None
    cfg_path = run_dir / "config.json"
    if cfg_path.exists():
        try:
            config = json.loads(cfg_path.read_text())
        except Exception as e:
            config = {"_error": f"failed to parse config.json: {e}"}

    videos = []
    for fn in sorted(os.listdir(run_dir)):
        if not EVAL_RE.match(fn):
            continue
        m = STEP_RE.search(fn)
        step = int(m.group(1)) if m else -1
        videos.append({"file": fn, "step": step})
    videos.sort(key=lambda v: v["step"])

    return {"rel_path": rel_path, "config": config, "videos": videos}


def safe_join(root: Path, rel: str) -> Optional[Path]:
    """Join `rel` onto `root`, refusing paths that escape `root`."""
    rel = unquote(rel).lstrip("/")
    candidate = (root / rel).resolve()
    try:
        candidate.relative_to(root.resolve())
    except ValueError:
        return None
    return candidate


# ---------------------------------------------------------------------------
# HTTP handler
# ---------------------------------------------------------------------------

INDEX_HTML = r"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>Trajectory Viewer</title>
<style>
  :root {
    --bg: #11151c;
    --panel: #1a2030;
    --panel-2: #232a3d;
    --border: #2c3550;
    --fg: #e6e8ee;
    --muted: #8a92a6;
    --accent: #6aa9ff;
    --accent-2: #f6c177;
    --danger: #e06c75;
  }
  * { box-sizing: border-box; }
  html, body { margin: 0; height: 100%; background: var(--bg); color: var(--fg);
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif; font-size: 13px; }
  #app { display: grid; grid-template-columns: 320px 1fr; height: 100vh; }
  #sidebar { background: var(--panel); border-right: 1px solid var(--border);
    overflow-y: auto; padding: 12px; }
  #sidebar h1 { font-size: 14px; margin: 0 0 8px; color: var(--accent); letter-spacing: 0.5px; }
  #sidebar h2 { font-size: 11px; margin: 12px 0 4px; color: var(--muted);
    text-transform: uppercase; letter-spacing: 0.8px; }
  .run-item { padding: 6px 8px; margin: 2px 0; border-radius: 4px; cursor: pointer;
    border: 1px solid transparent; user-select: none; }
  .run-item:hover { background: var(--panel-2); border-color: var(--border); }
  .run-item .nm { font-size: 12px; word-break: break-all; }
  .run-item .meta { color: var(--muted); font-size: 11px; margin-top: 2px; }
  .run-item.active { background: var(--panel-2); border-color: var(--accent); }
  #main { display: flex; flex-direction: column; overflow: hidden; }
  #toolbar { display: flex; align-items: center; gap: 10px; padding: 8px 14px;
    border-bottom: 1px solid var(--border); background: var(--panel); }
  #toolbar label { color: var(--muted); display: flex; align-items: center; gap: 4px; }
  button { background: var(--panel-2); border: 1px solid var(--border); color: var(--fg);
    padding: 5px 10px; border-radius: 4px; cursor: pointer; font-size: 12px; }
  button:hover { border-color: var(--accent); }
  button.danger:hover { border-color: var(--danger); color: var(--danger); }
  #cols { flex: 1; display: flex; overflow-x: auto; gap: 12px; padding: 12px; }
  .col { flex: 1 1 0; min-width: 380px; max-width: 800px; background: var(--panel);
    border: 1px solid var(--border); border-radius: 6px; padding: 10px;
    display: flex; flex-direction: column; gap: 8px; }
  .col-head { display: flex; align-items: center; justify-content: space-between; gap: 8px; }
  .col-title { font-size: 12px; word-break: break-all; }
  .col-title .group { color: var(--muted); }
  video { width: 100%; background: black; border-radius: 4px; max-height: 50vh; }
  .slider-row { display: flex; align-items: center; gap: 8px; }
  .slider-row input[type=range] { flex: 1; }
  .step-label { font-family: ui-monospace, monospace; color: var(--accent-2); min-width: 110px; text-align: right; }
  details { background: var(--panel-2); border: 1px solid var(--border); border-radius: 4px;
    padding: 6px 10px; }
  details summary { cursor: pointer; color: var(--muted); font-size: 11px;
    text-transform: uppercase; letter-spacing: 0.6px; }
  pre.config { white-space: pre-wrap; word-break: break-word; font-size: 11px;
    margin: 6px 0 0; max-height: 240px; overflow: auto; color: var(--fg); }
  .empty { color: var(--muted); padding: 24px; text-align: center; }
  .key-config { display: grid; grid-template-columns: max-content 1fr; gap: 2px 10px;
    font-size: 11px; margin-top: 4px; }
  .key-config .k { color: var(--muted); }
  .key-config .v { font-family: ui-monospace, monospace; word-break: break-all; }
</style>
</head>
<body>
<div id="app">
  <aside id="sidebar">
    <h1>Trajectory Viewer</h1>
    <div id="runs"></div>
  </aside>
  <main id="main">
    <div id="toolbar">
      <label><input type="checkbox" id="sync" checked> Sync sliders</label>
      <label><input type="checkbox" id="autoplay" checked> Autoplay</label>
      <label><input type="checkbox" id="loop" checked> Loop</label>
      <label><input type="checkbox" id="muted" checked> Muted</label>
      <span style="flex:1"></span>
      <span style="color: var(--muted)">Click a run to view. Shift+click to add as compare column.</span>
      <button id="clear">Clear all</button>
    </div>
    <div id="cols"><div class="empty">No run selected. Pick one from the left.</div></div>
  </main>
</div>

<script>
const KEY_CONFIG_FIELDS = [
  ["env", "dedo.env"],
  ["algo", "rl.algo"],
  ["lr", "dedo.lr"],
  ["seed", "dedo.seed"],
  ["num_envs", "dedo.num_envs"],
  ["max_episode_len", "dedo.max_episode_len"],
  ["total_env_steps", "dedo.total_env_steps"],
  ["cam_resolution", "dedo.cam_resolution"],
  ["deform_init_pos", "dedo.deform_init_pos"],
];

function get(obj, path) {
  return path.split(".").reduce((o, k) => (o == null ? o : o[k]), obj);
}

const state = {
  runs: [],            // [{id, name, group, rel_path, videos, ...}]
  columns: [],         // [{rel_path, name, group, videos, config, stepIdx}]
};

async function fetchJSON(url) {
  const r = await fetch(url);
  if (!r.ok) throw new Error(r.status + " " + r.statusText);
  return r.json();
}

async function loadRuns() {
  state.runs = await fetchJSON("/api/runs");
  renderSidebar();
}

function renderSidebar() {
  const el = document.getElementById("runs");
  const groups = {};
  for (const r of state.runs) {
    (groups[r.group] = groups[r.group] || []).push(r);
  }
  el.innerHTML = "";
  const groupNames = Object.keys(groups).sort();
  for (const g of groupNames) {
    const h = document.createElement("h2");
    h.textContent = g + "  (" + groups[g].length + ")";
    el.appendChild(h);
    for (const r of groups[g]) {
      const div = document.createElement("div");
      div.className = "run-item";
      div.dataset.relPath = r.rel_path;
      div.innerHTML =
        '<div class="nm">' + r.name + '</div>' +
        '<div class="meta">' + r.n_videos + ' videos · steps ' +
        r.min_step.toLocaleString() + '–' + r.max_step.toLocaleString() + '</div>';
      div.addEventListener("click", (ev) => {
        if (ev.shiftKey) addColumn(r.rel_path);
        else replaceColumns(r.rel_path);
      });
      el.appendChild(div);
    }
  }
  refreshActive();
}

function refreshActive() {
  const active = new Set(state.columns.map(c => c.rel_path));
  document.querySelectorAll(".run-item").forEach(el => {
    el.classList.toggle("active", active.has(el.dataset.relPath));
  });
}

async function replaceColumns(relPath) {
  state.columns = [];
  await addColumn(relPath);
}

async function addColumn(relPath) {
  if (state.columns.some(c => c.rel_path === relPath)) return;
  const detail = await fetchJSON("/api/run?path=" + encodeURIComponent(relPath));
  const meta = state.runs.find(r => r.rel_path === relPath) || { name: relPath, group: "" };
  state.columns.push({
    rel_path: relPath,
    name: meta.name,
    group: meta.group,
    videos: detail.videos,
    config: detail.config,
    stepIdx: detail.videos.length - 1,  // start at last (best) checkpoint
  });
  renderColumns();
  refreshActive();
}

function removeColumn(relPath) {
  state.columns = state.columns.filter(c => c.rel_path !== relPath);
  renderColumns();
  refreshActive();
}

function renderColumns() {
  const root = document.getElementById("cols");
  if (state.columns.length === 0) {
    root.innerHTML = '<div class="empty">No run selected. Pick one from the left.</div>';
    return;
  }
  root.innerHTML = "";
  for (const c of state.columns) root.appendChild(makeColumn(c));
  // Apply sync if enabled
  applySync();
}

function makeColumn(col) {
  const root = document.createElement("div");
  root.className = "col";
  root.dataset.relPath = col.rel_path;

  const head = document.createElement("div");
  head.className = "col-head";
  head.innerHTML =
    '<div class="col-title"><span class="group">' + col.group + ' / </span>' + col.name + '</div>';
  const closeBtn = document.createElement("button");
  closeBtn.className = "danger";
  closeBtn.textContent = "✕";
  closeBtn.title = "Remove column";
  closeBtn.onclick = () => removeColumn(col.rel_path);
  head.appendChild(closeBtn);
  root.appendChild(head);

  const video = document.createElement("video");
  video.controls = true;
  video.preload = "auto";
  video.muted = document.getElementById("muted").checked;
  video.loop = document.getElementById("loop").checked;
  video.dataset.role = "video";
  root.appendChild(video);

  const sliderRow = document.createElement("div");
  sliderRow.className = "slider-row";
  const slider = document.createElement("input");
  slider.type = "range";
  slider.min = 0;
  slider.max = col.videos.length - 1;
  slider.value = col.stepIdx;
  slider.dataset.role = "slider";
  const stepLabel = document.createElement("span");
  stepLabel.className = "step-label";
  stepLabel.dataset.role = "step-label";
  sliderRow.appendChild(slider);
  sliderRow.appendChild(stepLabel);
  root.appendChild(sliderRow);

  const navRow = document.createElement("div");
  navRow.className = "slider-row";
  const prev = document.createElement("button");
  prev.textContent = "◀ Prev";
  const next = document.createElement("button");
  next.textContent = "Next ▶";
  const first = document.createElement("button");
  first.textContent = "⏮ First";
  const last = document.createElement("button");
  last.textContent = "Last ⏭";
  prev.onclick = () => { setColStep(col, Math.max(0, col.stepIdx - 1)); };
  next.onclick = () => { setColStep(col, Math.min(col.videos.length - 1, col.stepIdx + 1)); };
  first.onclick = () => { setColStep(col, 0); };
  last.onclick = () => { setColStep(col, col.videos.length - 1); };
  navRow.appendChild(first); navRow.appendChild(prev);
  navRow.appendChild(next); navRow.appendChild(last);
  root.appendChild(navRow);

  const det = document.createElement("details");
  const sum = document.createElement("summary");
  sum.textContent = "Config";
  det.appendChild(sum);
  const kc = document.createElement("div");
  kc.className = "key-config";
  for (const [label, path] of KEY_CONFIG_FIELDS) {
    const v = get(col.config, path);
    if (v === undefined || v === null) continue;
    const k = document.createElement("div"); k.className = "k"; k.textContent = label;
    const vv = document.createElement("div"); vv.className = "v";
    vv.textContent = typeof v === "object" ? JSON.stringify(v) : String(v);
    kc.appendChild(k); kc.appendChild(vv);
  }
  det.appendChild(kc);
  const pre = document.createElement("pre");
  pre.className = "config";
  pre.textContent = JSON.stringify(col.config, null, 2);
  det.appendChild(pre);
  root.appendChild(det);

  slider.addEventListener("input", (e) => {
    const idx = Number(e.target.value);
    setColStep(col, idx);
    if (document.getElementById("sync").checked) syncOthers(col, idx);
  });

  // Initial load
  setColStep(col, col.stepIdx);
  return root;
}

function colEl(col) {
  return document.querySelector('.col[data-rel-path="' + cssEscape(col.rel_path) + '"]');
}
function cssEscape(s) { return s.replace(/[^a-zA-Z0-9_-]/g, c => "\\" + c); }

function setColStep(col, idx) {
  col.stepIdx = idx;
  const root = colEl(col);
  if (!root) return;
  const v = col.videos[idx];
  const video = root.querySelector('[data-role="video"]');
  const slider = root.querySelector('[data-role="slider"]');
  const label = root.querySelector('[data-role="step-label"]');
  slider.value = idx;
  label.textContent = "step " + v.step.toLocaleString();
  const wasPlaying = !video.paused;
  const url = "/file?path=" + encodeURIComponent(col.rel_path + "/" + v.file);
  if (video.src !== location.origin + url) {
    video.src = url;
    if (document.getElementById("autoplay").checked || wasPlaying) {
      video.play().catch(() => {});
    }
  }
}

function syncOthers(srcCol, idx) {
  // For each other column, jump to the equivalent fractional position.
  const srcFrac = idx / Math.max(1, srcCol.videos.length - 1);
  for (const c of state.columns) {
    if (c === srcCol) continue;
    const targetIdx = Math.round(srcFrac * (c.videos.length - 1));
    setColStep(c, targetIdx);
  }
}

function applySync() {
  // Re-apply current toolbar toggles to all videos.
  document.querySelectorAll('video').forEach(v => {
    v.muted = document.getElementById("muted").checked;
    v.loop = document.getElementById("loop").checked;
  });
}

document.getElementById("clear").addEventListener("click", () => {
  state.columns = [];
  renderColumns();
  refreshActive();
});
for (const id of ["muted", "loop"]) {
  document.getElementById(id).addEventListener("change", applySync);
}

loadRuns().catch(err => {
  document.getElementById("runs").textContent = "Error loading runs: " + err.message;
});
</script>
</body>
</html>
"""


class TrajectoryHandler(BaseHTTPRequestHandler):
    server_version = "TrajectoryViewer/1.0"
    root: Path = Path(".")  # set by server factory

    # Quieter logs
    def log_message(self, fmt, *args):
        sys.stderr.write("[%s] %s\n" % (self.log_date_time_string(), fmt % args))

    def do_GET(self):
        parsed = urlparse(self.path)
        path = parsed.path
        qs = parse_qs(parsed.query)
        try:
            if path == "/" or path == "/index.html":
                self._send_html(INDEX_HTML)
            elif path == "/api/runs":
                self._send_json(find_runs(self.root))
            elif path == "/api/run":
                rel = qs.get("path", [""])[0]
                detail = load_run_detail(self.root, rel)
                if detail is None:
                    self._send_error(HTTPStatus.NOT_FOUND, "run not found")
                else:
                    self._send_json(detail)
            elif path == "/file":
                rel = qs.get("path", [""])[0]
                self._serve_file(rel)
            else:
                self._send_error(HTTPStatus.NOT_FOUND, "not found")
        except BrokenPipeError:
            pass
        except Exception as e:
            self._send_error(HTTPStatus.INTERNAL_SERVER_ERROR, str(e))

    def do_HEAD(self):
        # Only used by `/file`; reuse range-aware path.
        parsed = urlparse(self.path)
        if parsed.path != "/file":
            self._send_error(HTTPStatus.NOT_FOUND, "not found")
            return
        rel = parse_qs(parsed.query).get("path", [""])[0]
        self._serve_file(rel, head=True)

    # -- helpers ----------------------------------------------------------

    def _send_html(self, body: str):
        data = body.encode("utf-8")
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def _send_json(self, obj):
        data = json.dumps(obj).encode("utf-8")
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(data)

    def _send_error(self, status, msg: str):
        data = msg.encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "text/plain; charset=utf-8")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def _serve_file(self, rel: str, head: bool = False):
        target = safe_join(self.root, rel)
        if target is None or not target.is_file():
            self._send_error(HTTPStatus.NOT_FOUND, "file not found")
            return
        ctype, _ = mimetypes.guess_type(str(target))
        if ctype is None:
            ctype = "application/octet-stream"
        size = target.stat().st_size

        range_header = self.headers.get("Range")
        if range_header and range_header.startswith("bytes="):
            try:
                spec = range_header.split("=", 1)[1].split(",", 1)[0].strip()
                start_s, end_s = spec.split("-", 1)
                if start_s == "":
                    # suffix range: bytes=-N
                    n = int(end_s)
                    start = max(0, size - n)
                    end = size - 1
                else:
                    start = int(start_s)
                    end = int(end_s) if end_s else size - 1
                if start > end or start >= size:
                    raise ValueError("invalid range")
                end = min(end, size - 1)
            except Exception:
                self.send_response(HTTPStatus.REQUESTED_RANGE_NOT_SATISFIABLE)
                self.send_header("Content-Range", f"bytes */{size}")
                self.end_headers()
                return
            length = end - start + 1
            self.send_response(HTTPStatus.PARTIAL_CONTENT)
            self.send_header("Content-Type", ctype)
            self.send_header("Accept-Ranges", "bytes")
            self.send_header("Content-Range", f"bytes {start}-{end}/{size}")
            self.send_header("Content-Length", str(length))
            self.end_headers()
            if head:
                return
            with open(target, "rb") as f:
                f.seek(start)
                remaining = length
                chunk = 64 * 1024
                while remaining > 0:
                    buf = f.read(min(chunk, remaining))
                    if not buf:
                        break
                    self.wfile.write(buf)
                    remaining -= len(buf)
        else:
            self.send_response(HTTPStatus.OK)
            self.send_header("Content-Type", ctype)
            self.send_header("Accept-Ranges", "bytes")
            self.send_header("Content-Length", str(size))
            self.end_headers()
            if head:
                return
            with open(target, "rb") as f:
                while True:
                    buf = f.read(64 * 1024)
                    if not buf:
                        break
                    self.wfile.write(buf)


class ThreadingServer(socketserver.ThreadingMixIn, socketserver.TCPServer):
    daemon_threads = True
    allow_reuse_address = True


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--root", default=".", help="Repo root to scan for runs")
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=8000)
    args = p.parse_args()

    root = Path(args.root).resolve()
    if not root.is_dir():
        sys.exit(f"--root {root} is not a directory")

    handler = type("BoundHandler", (TrajectoryHandler,), {"root": root})
    runs = find_runs(root)
    print(f"Scanned {root}: found {len(runs)} runs with eval videos.")
    for r in runs:
        print(f"  {r['rel_path']}  ({r['n_videos']} videos)")
    print(f"\nServing at http://{args.host}:{args.port}/  (Ctrl-C to stop)")
    with ThreadingServer((args.host, args.port), handler) as httpd:
        try:
            httpd.serve_forever()
        except KeyboardInterrupt:
            print("\nbye.")


if __name__ == "__main__":
    main()
