"""Find which procedurally-shaped cloths hang pybullet's soft-body loader.

The shaped-outline collector wedges mid-run, and SIGINT produces no Python
traceback — so it is blocked inside native pybullet, not in our code. This
generates meshes the same way the collector does and hands each to
loadSoftBody in a SUBPROCESS with a timeout, so a hang is an observation
rather than a wedged run. Then it diffs the geometry of the meshes that hung
against the ones that loaded.

    python _diag_softbody_hang.py --n 120          # driver
    python _diag_softbody_hang.py --worker SEED    # one mesh, used internally
"""
import argparse
import collections
import json
import os
import subprocess
import sys
import tempfile

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '..'))
from dedo.utils.procedural_utils import (          # noqa: E402
    _shape_cut_cells, _validated_cuts, _hole_extent_range, try_gen_holes,
    create_cloth_obj, CLOTH_SHAPES)


def build(seed):
    """Deterministically build one cloth. Returns (obj_path, properties)."""
    rng = np.random.default_rng(seed)
    np.random.seed(seed)
    nd = int(rng.integers(9, 16))
    cons = dict(x_range=(2, nd - 2), y_range=(2, nd - 2),
                width_range=_hole_extent_range(nd, (0.06, 0.38)),
                height_range=_hole_extent_range(nd, (0.06, 0.38)))
    holes = try_gen_holes(nd, 1, cons)
    shape = str(rng.choice(list(CLOTH_SHAPES)))
    hc = [dict(h) for h in holes]
    cuts = _validated_cuts(nd, hc, _shape_cut_cells(nd, shape, rng), shape)
    path = os.path.join(tempfile.gettempdir(), f'_hangprobe_{seed}.obj')
    _, _, loops = create_cloth_obj([0, -1, -1], [0, 1, 1], nd, hc, path,
                                   node_coords=[], cut_cells=cuts)
    return path, mesh_props(path, nd, shape, len(cuts), loops)


def mesh_props(path, nd, shape, n_cuts, loops):
    """Geometric properties that plausibly distinguish a loadable mesh."""
    verts, faces = [], []
    for line in open(path):
        if line.startswith('v '):
            verts.append(tuple(float(v) for v in line.split()[1:]))
        elif line.startswith('f '):
            faces.append(tuple(int(v) - 1 for v in line.split()[1:]))

    edge_faces = collections.Counter()
    for f in faces:
        for a, b in ((f[0], f[1]), (f[1], f[2]), (f[2], f[0])):
            edge_faces[(min(a, b), max(a, b))] += 1
    # Face connectivity, which is NOT the same as grid adjacency: two regions
    # meeting at a single vertex are grid-connected but form a pinch point,
    # and a pinch is the classic degenerate soft body.
    adj = collections.defaultdict(set)
    for f in faces:
        for a in f:
            for b in f:
                if a != b:
                    adj[a].add(b)
    seen, comps = set(), 0
    sizes = []
    for v in range(len(verts)):
        if v in seen:
            continue
        comps += 1
        stack, sz = [v], 0
        seen.add(v)
        while stack:
            u = stack.pop()
            sz += 1
            for w in adj[u]:
                if w not in seen:
                    seen.add(w)
                    stack.append(w)
        sizes.append(sz)

    # A vertex whose incident faces do not form ONE fan is a pinch point.
    vfaces = collections.defaultdict(list)
    for i, f in enumerate(faces):
        for a in f:
            vfaces[a].append(i)
    pinches = 0
    for v, fl in vfaces.items():
        ring = collections.defaultdict(set)
        for i in fl:
            others = [x for x in faces[i] if x != v]
            ring[others[0]].add(others[1])
            ring[others[1]].add(others[0])
        if not ring:
            continue
        start = next(iter(ring))
        seen2, stack = {start}, [start]
        while stack:
            u = stack.pop()
            for w in ring[u]:
                if w not in seen2:
                    seen2.add(w)
                    stack.append(w)
        if len(seen2) != len(ring):
            pinches += 1

    return dict(nd=nd, shape=shape, n_cuts=n_cuts, n_verts=len(verts),
                n_faces=len(faces),
                boundary_edges=sum(1 for c in edge_faces.values() if c == 1),
                nonmanifold_edges=sum(1 for c in edge_faces.values() if c > 2),
                components=comps, smallest_component=min(sizes) if sizes else 0,
                pinch_vertices=pinches,
                hole_loop=len(loops[0]) if loops and loops[0] else 0)


def worker(seed):
    path, props = build(seed)
    import pybullet as p
    c = p.connect(p.DIRECT)
    p.setAdditionalSearchPath(os.path.join(os.path.dirname(__file__),
                                           '..', '..', '..', 'dedo', 'data'))
    p.loadSoftBody(path, scale=3, mass=1.0, useNeoHookean=0,
                   useBendingSprings=1, useMassSpring=1, springElasticStiffness=50,
                   springDampingStiffness=1, springBendingStiffness=1,
                   useSelfCollision=0, frictionCoeff=0.5, useFaceContact=1)
    for _ in range(5):
        p.stepSimulation()
    p.disconnect(c)
    print('PROPS ' + json.dumps(props))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--n', type=int, default=120)
    ap.add_argument('--timeout', type=float, default=20.0)
    ap.add_argument('--worker', type=int, default=None)
    a = ap.parse_args()
    if a.worker is not None:
        worker(a.worker)
        return

    hung, okd = [], []
    for seed in range(a.n):
        try:
            r = subprocess.run([sys.executable, __file__, '--worker', str(seed)],
                               capture_output=True, text=True,
                               timeout=a.timeout)
            line = [l for l in r.stdout.splitlines() if l.startswith('PROPS ')]
            if not line:
                print(f'seed {seed}: worker error\n{r.stderr[-400:]}')
                continue
            okd.append(json.loads(line[0][6:]))
        except subprocess.TimeoutExpired:
            _, props = build(seed)
            hung.append(props)
            print(f'seed {seed}: HUNG — {props}')

    print(f'\n=== {len(hung)} hung / {len(hung) + len(okd)} ===')
    if not hung:
        print('no hangs reproduced')
        return
    keys = ['nd', 'n_cuts', 'n_verts', 'n_faces', 'boundary_edges',
            'nonmanifold_edges', 'components', 'smallest_component',
            'pinch_vertices', 'hole_loop']
    print(f'{"property":20s} {"HUNG mean":>12s} {"OK mean":>12s}')
    for k in keys:
        h = np.mean([d[k] for d in hung])
        o = np.mean([d[k] for d in okd]) if okd else float('nan')
        flag = '   <<<' if okd and abs(h - o) > 0.5 * (abs(o) + 1e-6) else ''
        print(f'{k:20s} {h:12.2f} {o:12.2f}{flag}')
    print('shape counts  HUNG:',
          dict(collections.Counter(d['shape'] for d in hung)))
    print('shape counts  OK  :',
          dict(collections.Counter(d['shape'] for d in okd)))


if __name__ == '__main__':
    main()
