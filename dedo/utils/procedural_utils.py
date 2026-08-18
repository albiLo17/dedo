"""
Utilities for generating procedural cloth.

Procedural generation works as follows:
1. Generate a mesh, randomly carves out a square hole.
2. If hole > 1, also checks for overlapping of two holes. If overlap, randomly choose a new hole position. Repeat until no overlap found.
3. Saves hollowed mesh into .obj file in the /tmp/ directory.



Note: this code is for research i.e. quick experimentation; it has minimal
comments for now, but if we see further interest from the community -- we will
add further comments, unify the style, improve efficiency and add unittests.

@yonkshi, @jackson

"""

import os
import tempfile
import numpy as np
from matplotlib import pyplot as plt

# Defaults for the procedural sampling ranges, used when the caller does not
# override them via args (--proc_cloth_size_range / --proc_hole_frac_range on
# collect_bc_demos.py). These are the historical values, so an unconfigured
# run generates exactly the distribution it always did.
CLOTH_SIZE_RANGE = (0.5, 2.8)   # full side length, sim units before deform_scale
HOLE_FRAC_RANGE = (0.06, 0.30)  # hole extent as a fraction of node_density



# Outline shapes. Until now every procedural cloth was an axis-aligned
# rectangle -- only its size, and the size/placement of one square hole, varied.
# These carve the OUTLINE so the silhouette itself differs cloth to cloth.
#
# Grid orientation: create_cloth_obj lays out a node_density x node_density
# grid, and the two anchors are nodes (0, 0) and (0, node_density-1). So grid
# row x=0 is the grasp edge and increasing x runs down the hanging cloth. Cuts
# therefore never touch x < _SHAPE_KEEP_ROWS: that would delete an anchor or
# narrow the edge the grippers hold, which changes the grasp rather than the
# shape.
CLOTH_SHAPES = ('rect', 'taper', 'flare', 'corner_cut', 'notch', 'round_corners')
_SHAPE_KEEP_ROWS = 2


def _shape_cut_cells(node_density, shape, rng):
    """Grid cells to delete to give the cloth a non-rectangular outline.

    Returns a set of (x, y). Cells are removed from the mesh exactly the way
    hole cells are, but they are kept OUT of the hole list so they never count
    as hole-boundary vertices -- the hole centroid, and therefore the success
    metric, is unaffected by the silhouette.
    """
    nd = node_density
    cut = set()
    lo = _SHAPE_KEEP_ROWS
    span = nd - lo
    if shape == 'rect' or span < 3:
        return cut

    if shape in ('taper', 'flare'):
        # Trapezoid: each side loses a linearly growing number of columns as
        # you go down (taper) or up (flare). Drawn per side, so asymmetric
        # trapezoids and near-triangles both occur.
        for side in (0, 1):
            depth = int(rng.integers(1, max(2, nd // 3)))
            for x in range(lo, nd):
                t = (x - lo) / max(1, span - 1)
                k = int(round(depth * (t if shape == 'taper' else 1.0 - t)))
                for j in range(k):
                    cut.add((x, j) if side == 0 else (x, nd - 1 - j))

    elif shape == 'corner_cut':
        # Diagonal off one or both bottom corners.
        size = int(rng.integers(2, max(3, nd // 2)))
        corners = [(0, 1), (1, 0), (1, 1)][int(rng.integers(0, 3))]
        for x in range(max(lo, nd - size), nd):
            reach = size - (nd - 1 - x)
            for j in range(reach):
                if corners[0]:
                    cut.add((x, j))
                if corners[1]:
                    cut.add((x, nd - 1 - j))

    elif shape == 'notch':
        # A rectangular bite out of the bottom edge, like a split skirt.
        w = int(rng.integers(1, max(2, nd // 3)))
        d = int(rng.integers(1, max(2, span // 2)))
        y0 = int(rng.integers(0, max(1, nd - w)))
        for x in range(nd - d, nd):
            for y in range(y0, min(nd, y0 + w)):
                cut.add((x, y))

    elif shape == 'round_corners':
        # Remove the cells OUTSIDE a quarter-circle tucked into each bottom
        # corner. The arc centre has to sit inside the cloth at
        # (nd-1-r, r) / (nd-1-r, nd-1-r); testing distance from the corner
        # itself instead carves a band and leaves the corner tip floating as a
        # separate island, which is what wedged pybullet's soft-body loader.
        r = int(rng.integers(2, max(3, nd // 2)))
        for x in range(max(lo, nd - r), nd):
            for y in range(nd):
                for cy in (r, nd - 1 - r):
                    if (y < r) != (cy == r):
                        continue
                    if y >= r and cy == r:
                        continue
                    if y <= nd - 1 - r and cy != r:
                        continue
                    dx, dy = x - (nd - 1 - r), y - cy
                    if dx > 0 and dx * dx + dy * dy > r * r:
                        cut.add((x, y))

    # Never touch the grasp edge, whatever the shape logic produced.
    return {(x, y) for (x, y) in cut if x >= lo}


def _validated_cuts(node_density, holes, cut_cells, shape):
    """Drop a silhouette cut that would break the cloth instead of shaping it.

    Three ways a cut goes wrong, all of which otherwise surface much later as a
    confusing physics failure or an empty hole loop:
      - it swallows the hole, so there is no target to thread;
      - it removes an anchor node, so a gripper has nothing to hold;
      - it eats so much of the cloth there is barely a sheet left.
    Returning an empty set means "this cloth is a plain rectangle", which is a
    valid cloth -- so the run continues rather than dying on a sampling fluke.
    """
    nd = node_density
    if not cut_cells:
        return set()
    for anchor in ((0, 0), (0, nd - 1)):
        if anchor in cut_cells:
            return set()
    for hole in holes or []:
        hx = range(int(hole.get('x0', 0)), int(hole.get('x1', 0)) + 1)
        hy = range(int(hole.get('y0', 0)), int(hole.get('y1', 0)) + 1)
        # A cut touching the hole merges the two into one open notch: the hole
        # stops being a closed loop and the task stops being a threading task.
        for x in hx:
            for y in hy:
                for nb in ((x + 1, y), (x - 1, y), (x, y + 1), (x, y - 1)):
                    if nb in cut_cells:
                        return set()
    if len(cut_cells) > 0.45 * nd * nd:
        return set()

    # The cloth must stay in ONE piece. A cut that leaves an island -- a corner
    # tip sheared off, a strip isolated by a notch -- produces a soft body that
    # pybullet's loader wedges on: no error, no Python traceback, the collector
    # simply stops mid-run. Cheaper to reject the sample than to debug the hang.
    hole_cells = {(x, y)
                  for h in (holes or [])
                  for x in range(int(h['x0']), int(h['x1']) + 1)
                  for y in range(int(h['y0']), int(h['y1']) + 1)}
    kept = {(x, y) for x in range(nd) for y in range(nd)} - cut_cells - hole_cells
    if not kept:
        return set()
    stack, seen = [next(iter(kept))], set()
    seen.add(stack[0])
    while stack:
        x, y = stack.pop()
        for nb in ((x + 1, y), (x - 1, y), (x, y + 1), (x, y - 1)):
            if nb in kept and nb not in seen:
                seen.add(nb)
                stack.append(nb)
    if len(seen) != len(kept):
        return set()

    # Pinch points. Grid adjacency says these two regions are connected, but
    # the MESH only touches at a single vertex, and pybullet wedges on that the
    # same way it wedges on a detached island. A node is a pinch when exactly
    # the two diagonally-opposite quads around it survive: the surface passes
    # through one point with no shared edge.
    def alive(qx, qy):
        return all((qx + i, qy + j) in kept
                   for i in (0, 1) for j in (0, 1))
    for x in range(1, nd - 1):
        for y in range(1, nd - 1):
            nw, ne = alive(x - 1, y), alive(x, y)
            sw, se = alive(x - 1, y - 1), alive(x, y - 1)
            if (nw and se and not ne and not sw) or \
               (ne and sw and not nw and not se):
                return set()
    return cut_cells


def _hole_extent_range(node_density, frac_range):
    """Hole side length in NODES, from a fraction-of-resolution range.

    Clamped to at least 1 node (a zero-node hole is not a hole) and to at most
    node_density - 4, since gen_random_hole places the hole's origin inside
    (2, node_density - 2) and anything larger cannot satisfy the boundary
    constraint -- try_gen_holes would burn its 1000 Monte-Carlo attempts and
    return None, which surfaces much later as a confusing crash.
    """
    lo = max(1, int(round(node_density * frac_range[0])))
    hi = min(int(round(node_density * frac_range[1])), max(2, node_density - 4))
    # gen_random_hole calls np.random.randint(lo, hi), which is half-open and
    # raises outright when lo >= hi -- so hi must stay strictly above lo.
    hi = max(lo + 1, hi)
    return (lo, hi)


def gen_procedural_hang_cloth(args, preset_obj_name, deform_info_dict):
    '''
    Hang cloth procedrual generator. Generates a cloth of random size and places random holes inside. Checks for overlap
    :param args:
    :param preset_obj_name:
    :param deform_info_dict:
    :return:
    '''
    num_holes = args.num_holes
    node_density = args.node_density

    # cloth dimensions. Upper bound raised 2.0 -> 2.8 so the real cloth is
    # INSIDE the training distribution: measured 0.344 x 0.292 m, which is
    # 2.55 x 2.16 after dividing by deform_scale 3 and the 0.045 m/sim-unit
    # factor. At 2.0 the real cloth was larger than anything ever trained on.
    # Width and height are drawn INDEPENDENTLY, so aspect ratio varies too.
    # `proc_cloth_wh` pins the cloth instead of sampling it. The anchors are
    # grid corners (0,0) and (0,nd-1), i.e. the two ends of one edge, so the
    # cloth's HEIGHT is exactly the gripper separation -- initialising the
    # grippers at a measured real pose means sizing the cloth to it here.
    wh = getattr(args, 'proc_cloth_wh', None)
    if wh is not None:
        w, h = float(wh[0]) / 2, float(wh[1]) / 2
    else:
        width_range = getattr(args, 'proc_cloth_size_range', None) or CLOTH_SIZE_RANGE
        height_range = width_range
        w = np.random.uniform(*width_range) / 2
        h = np.random.uniform(*height_range) / 2

    # Hole generation. Extents are a FRACTION of node_density, so widening the
    # fraction range and randomizing node_density together are what actually
    # produce differently-shaped holes rather than one hole at N resolutions.
    hole_frac = getattr(args, 'proc_hole_frac_range', None) or HOLE_FRAC_RANGE
    constraints = {}
    constraints['x_range'] = (2, args.node_density - 2)
    constraints['y_range'] = (2, args.node_density - 2)
    constraints['width_range'] = _hole_extent_range(node_density, hole_frac)
    constraints['height_range'] = _hole_extent_range(node_density, hole_frac)
    holes = try_gen_holes(args.node_density, num_holes, constraints)

    # Outline shape. Without this every cloth is an axis-aligned rectangle and
    # the only silhouette variation in the whole dataset is its width/height.
    shapes = getattr(args, 'proc_cloth_shapes', None) or ('rect',)
    shape = str(np.random.choice(list(shapes)))
    rng = np.random.default_rng(int(np.random.randint(0, 2 ** 31 - 1)))
    cut_cells = _shape_cut_cells(node_density, shape, rng)
    cut_cells = _validated_cuts(node_density, holes, cut_cells, shape)
    args.proc_cloth_shape = shape   # read back by the collector for the record

    # cloth save path
    rand_id = np.random.uniform(1e7)
    args.deform_obj = os.path.join(tempfile.gettempdir(), f'procedural_hang{rand_id}.obj')
    savepath = args.deform_obj

    cloth_obj_path, cloth_anchor_indices, gt_loop_vertices = create_cloth_obj(
        # min_point=[0.00, -0.3, -0.3], max_point=[0.00, 0.3, 0.3],
        min_point=[0.00, -w, -h], max_point=[0.00, w, h],
        # min_point=[0,0.42,0.48], max_point=[0.2,0.45,0.52],
        node_density=args.node_density,
        holes=holes,
        data_path=savepath,
        cut_cells=cut_cells,
    )
    if args.deform_obj not in deform_info_dict.keys():
        deform_info_dict[args.deform_obj] = deform_info_dict[preset_obj_name].copy()

    deform_info_dict[args.deform_obj]['deform_anchor_vertices'] = list(cloth_anchor_indices)
    deform_info_dict[args.deform_obj]['deform_true_loop_vertices'] = gt_loop_vertices

    return args.deform_obj


def gen_procedural_hang_cloth_real(args, preset_obj_name, deform_info_dict):
    '''Real-world variant of gen_procedural_hang_cloth.
    Cloth mesh lies in the XZ plane (y=0 fixed) instead of the YZ plane,
    matching the real Franka workspace orientation after baking in the
    sim-to-world transform (scale=0.045, Rz(90°), translation).
    '''
    num_holes = args.num_holes
    node_density = args.node_density

    # Kept in lockstep with gen_procedural_hang_cloth above: this preset uses
    # deform_scale 0.135 = 3 x 0.045, so the two describe the IDENTICAL
    # real-world size range and must be changed together or the scenes
    # silently diverge.
    #
    # `proc_cloth_wh` pins the cloth instead of sampling it. The two anchors
    # are the TOP corners (see the override below), so the cloth's WIDTH *is*
    # the gripper separation -- the only way to initialise the grippers at a
    # measured real pose is to set the width to that separation here. Sampling
    # a width and correcting the anchors afterwards would move the grippers
    # without moving the cloth they hold.
    wh = getattr(args, 'proc_cloth_wh', None)
    if wh is not None:
        w = float(wh[0]) / 2
        h = float(wh[1]) / 2
    else:
        width_range = [0.5, 2.8]
        height_range = [0.5, 2.8]
        w = np.random.uniform(*width_range) / 2
        h = np.random.uniform(*height_range) / 2

    constraints = {}
    constraints['x_range'] = (2, args.node_density - 2)
    constraints['y_range'] = (2, args.node_density - 2)
    constraints['width_range'] = (1, int(round(node_density * 0.3)))
    constraints['height_range'] = (1, int(round(node_density * 0.3)))
    holes = try_gen_holes(args.node_density, num_holes, constraints)

    rand_id = np.random.uniform(1e7)
    args.deform_obj = os.path.join(tempfile.gettempdir(), f'procedural_hang_real{rand_id}.obj')
    savepath = args.deform_obj

    cloth_obj_path, cloth_anchor_indices, gt_loop_vertices = create_cloth_obj(
        min_point=[-w, 0.00, -h], max_point=[w, 0.00, h],
        node_density=args.node_density,
        holes=holes,
        data_path=savepath,
    )

    # Override anchors: use TOP-LEFT and TOP-RIGHT corners instead of both-left.
    # Both-left creates an unconstrained y-pendulum when the cloth is pulled in x.
    # TOP-LEFT = vertex (0, nd-1) = first row, last column → always index nd-1
    # TOP-RIGHT = vertex (nd-1, nd-1) = last vertex ever added (corners are never
    #   inside a hole since holes are constrained to x∈[2,nd-3], y∈[2,nd-3]).
    nd = args.node_density
    top_left_idx = nd - 1
    with open(cloth_obj_path) as _f:
        n_verts = sum(1 for _line in _f if _line.startswith('v '))
    top_right_idx = n_verts - 1
    cloth_anchor_indices = ([top_left_idx], [top_right_idx])

    if args.deform_obj not in deform_info_dict.keys():
        deform_info_dict[args.deform_obj] = deform_info_dict[preset_obj_name].copy()

    deform_info_dict[args.deform_obj]['deform_anchor_vertices'] = list(cloth_anchor_indices)
    deform_info_dict[args.deform_obj]['deform_true_loop_vertices'] = gt_loop_vertices

    return args.deform_obj


def gen_procedural_button_cloth(args, preset_obj_name, deform_info_dict):
    '''
    Button cloth procedural generator, generates one or two holes for the button cloth.
    :param args: args object
    :param preset_obj_name:
    :param deform_info_dict:
    :return:
    '''
    num_holes = args.num_holes

    # These are fine tuned ranges.
    width_range = [2, 3]
    height_range = [2, 3.5]
    w = np.random.uniform(*width_range)
    h = np.random.uniform(*height_range)

    # Dynamic node density based on fabric size.
    node_density = int(round((w + h) / 2 * 25 / 3))

    # Hole generation.
    constraints = {}
    constraints['x_range'] = (2, 7)  # (2, args.node_density - 2)
    constraints['y_range'] = (2, node_density - 2)  # (2, args.node_density - 2)
    constraints['width_range'] = (1, 2)  # (1, int(round(node_density*0.3)))
    constraints['height_range'] = (1, 2)  # (1, int(round(node_density*0.3)))
    holes = try_gen_holes(node_density, num_holes, constraints)

    # Make temporary obj file path.
    rand_id = np.random.uniform(1e7)
    args.deform_obj = os.path.join(tempfile.gettempdir(), f'procedural_hang{rand_id}.obj')
    savepath = args.deform_obj

    node_coords = []
    cloth_obj_path, cloth_anchor_indices, gt_loop_vertices, fixed_anchors = create_cloth_obj(
        min_point=[0.00, -w, -h / 2], max_point=[0.00, 0, h / 2],
        node_density=node_density,
        holes=holes,
        data_path=savepath,
        gen_fixed_anchors=True,
        node_coords=node_coords,
    )
    if args.deform_obj not in deform_info_dict.keys():
        deform_info_dict[args.deform_obj] = deform_info_dict[preset_obj_name].copy()

    # Find center of holes (coords).
    node_coords = np.array(node_coords)
    hole_centers = [np.mean(node_coords[gt_loop], axis=0) for gt_loop in gt_loop_vertices]

    deform_info_dict[args.deform_obj]['deform_anchor_vertices'] = list(cloth_anchor_indices)
    deform_info_dict[args.deform_obj]['deform_fixed_anchor_vertex_ids'] = fixed_anchors
    deform_info_dict[args.deform_obj]['deform_true_loop_vertices'] = gt_loop_vertices
    # print('deform_true_loop_vertices', gt_loop_vertices)
    # print('deform_anchor_vertices', cloth_anchor_indices)
    # print('deform_fixed_anchor_vertex_ids', fixed_anchors)

    return args.deform_obj, hole_centers


def overlap_constraint(A, B):
    """ Make sure two holes are not overlapping and have enough vertices between
    to create faces in between."""
    mb = 3  # minimum boundary
    lr = A['x0'] < B['x1'] + mb
    rl = A['x1'] > B['x0'] - mb
    tb = A['y0'] < B['y1'] + mb
    bt = A['y1'] > B['y0'] - mb
    return not (lr and rl and tb and bt)


def boundary_constraint(node_density, hole):
    """ Each hole should be at least 2 vertices away from the edge,
    so the edge could form a face."""

    for key, val in hole.items():
        # Setting the minimum boundary between edge and hole.
        # Min two vertices away so edge could form face.
        upper_bound = node_density - 3  # 0-index node_density-1
        lower_bound = 2
        if hole[key] >= upper_bound or hole[key] <= lower_bound:
            return False

    return True


def gen_random_hole(node_density, dim_constraints):
    """Generates a hole, minding existing hole so they don't overlap."""
    hole = {}

    x_range = dim_constraints['x_range']
    y_range = dim_constraints['y_range']
    width_range = dim_constraints['width_range']
    height_range = dim_constraints['height_range']

    # Infer actual coordinates from constraints and dimensions
    hole['x0'] = np.random.randint(*x_range)
    hole['x1'] = hole['x0'] + np.random.randint(*width_range)

    hole['y0'] = np.random.randint(*y_range)
    hole['y1'] = hole['y0'] + np.random.randint(*height_range)

    return hole


def try_gen_holes(node_density, num_holes, constraints):
    '''
    Monte Carlo method for placing holes in a cloth, checks for overlap so they don't overlap
    :param node_density:
    :param num_holes:
    :param constraints:
    :return:
    '''
    for i in range(1000):  # 1000 MC
        if num_holes == 2:
            holeA = gen_random_hole(node_density, constraints)
            holeB = gen_random_hole(node_density, constraints)
            if boundary_constraint(node_density, holeA) and boundary_constraint(
                    node_density, holeB) and overlap_constraint(holeA, holeB):
                return [holeA, holeB]  # satisfies boundary constraints
        elif num_holes == 1:
            hole = gen_random_hole(node_density, constraints)
            if boundary_constraint(node_density, hole):
                return [hole]
        else:
            raise NotImplemented('num_holes > 2 is not implemented yet')
    print('Failed to generate hole according to constraint')


def create_cloth_obj(min_point, max_point, node_density,
                     holes, data_path,
                     gen_fixed_anchors=False,
                     node_coords=[],
                     cut_cells=None):
    '''
    Core procedural generator code
    :param min_point: bottom,left corner
    :param max_point: top, right corner
    :param node_density: density of the nodes (per cm)
    :param holes: list of holes to be generated
    :param data_path: temporary path for storing the cloth obj file
    :param gen_fixed_anchors: annotate fixed anchors if true (For buttoning
    :param node_coords:
    :return:
    '''

    def validate_and_integerize(hole):
        # Convert ratio to aboslute.
        for key, val in hole.items():
            if isinstance(val, float):
                hole[key] = int(round(val * node_density))
                # Setting the minimum boundary between edge and hole.
                # Min two vertices away so edge could form face
                if hole[key] >= node_density - 2:
                    hole[key] = node_density - 3
                elif hole[key] <= 1:
                    hole[key] = 2
            assert isinstance(hole[key], int), \
                f'{hole} {key} must be either an int or a float'
        assert len(min_point) == len(max_point) == 3, \
            'min_point and max_point must both have length 3'

    holes_range = []
    holes_fp = []

    for hole in holes:
        holes_fp.append(hole.copy())
        validate_and_integerize(hole)
        # Create a 2d range of hole coords.
        x_range = np.arange(hole['x0'], hole['x1'] + 1)
        y_range = np.arange(hole['y0'], hole['y1'] + 1, )
        xx, yy = np.meshgrid(x_range, y_range)
        # (x1, x1 ..., x2, x2 ...)
        xx = xx.flatten()
        # (y1, y2 ..., y1, y2 ...)
        yy = yy.flatten()
        # ((x1, y1), (x1, y2), ... (y1, y2), (x2, y2)
        r = list(zip(xx, yy))
        holes_range.append(r)
    if data_path.endswith('.obj'):  # If data_path is already a file path
        obj_path = data_path
    else:
        # Check if file already exists
        if not os.path.exists(os.path.join(data_path, "generated_cloth")):
            os.makedirs(os.path.join(data_path, "generated_cloth"))
        fnm = "cloth_" + str(node_density) + "_" + str(hole[0]['x']) + "_" + \
              str(hole[0]['y']) + "_" + str(hole[0]['x']) + "_" + \
              str(min_point[0]) + "_" + str(min_point[1]) + "_" + \
              str(min_point[2]) + "_" + str(max_point[0]) + "_" + \
              str(max_point[1]) + "_" + str(max_point[2]) + ".obj"
        obj_path = os.path.join(
            data_path, "generated_cloth", fnm)

        if os.path.isfile(obj_path):
            print("Cloth obj file already exists, skipping mesh creation.")
            anchor_indices = []
            with open(obj_path, 'r') as f:
                words = f.readline().split()
                anchor_indices = (int(words[1]), int(words[2]))
                f.close()
            return obj_path, anchor_indices

    # Cells removed to shape the OUTLINE. Deliberately separate from
    # holes_range: a node next to a silhouette cut is an outer edge, not a hole
    # boundary, and gt_loop_vertices (which defines the hole centroid and hence
    # the success metric) is built from holes_range alone.
    cut_cells = set() if cut_cells is None else set(cut_cells)

    def node_in_hole(x, y):
        # Check all holes
        for hole in holes_range:
            if (x, y) in hole: return True
        return False

    def node_removed(x, y):
        """Absent from the mesh for ANY reason — hole or silhouette cut."""
        return (x, y) in cut_cells or node_in_hole(x, y)

    def which_hole(x, y):
        # Check all holes
        for i, hole in enumerate(holes_range):
            if (x, y) in hole: return i

        return None

    # Quads first, THEN nodes. A node can survive both the hole test and the
    # cut test and still belong to no triangle -- it only takes one removed
    # corner to kill each of the four quads around it, which is exactly what
    # the stair-stepped edge of a diagonal or rounded cut produces. Emitting
    # such a vertex writes an unreferenced `v` line into the .obj, and pybullet
    # hangs on the resulting soft body rather than erroring. Building the quad
    # list first and keeping only the nodes it references makes orphans
    # impossible by construction.
    quads = [(x, y) for x in range(node_density - 1)
             for y in range(node_density - 1)
             if not (node_removed(x, y) or node_removed(x + 1, y) or
                     node_removed(x, y + 1) or node_removed(x + 1, y + 1))]
    used = set()
    for x, y in quads:
        used.update(((x, y), (x + 1, y), (x, y + 1), (x + 1, y + 1)))

    # Construct the list of nodes [(x1, y1), (x2, y2), ... , (xn, yn)]
    nodes = [(x, y) for x in range(node_density) for y in range(node_density)
             if not node_removed(x, y) and (x, y) in used]
    index_of = {n: i for i, n in enumerate(nodes)}

    gt_loop_vertices = [[] for _ in range(len(holes))]
    for (x, y) in nodes:
        # Get a hole's boundary
        # get boundary nodes, used for topo_latents
        for neighbour_pos in ((x + 1, y), (x, y + 1), (x - 1, y), (x, y - 1)):
            if node_in_hole(*neighbour_pos):
                gt_loop_vertices[which_hole(*neighbour_pos)].append(index_of[(x, y)])
                break

    # Construct the list of faces (clockwise around triangles) [(i, i2, i3), ...]
    faces = []
    for x, y in quads:
        faces.append((
            index_of[(x, y)] + 1,
            index_of[(x, y + 1)] + 1,
            index_of[(x + 1, y)] + 1))
        faces.append((
            index_of[(x, y + 1)] + 1,
            index_of[(x + 1, y + 1)] + 1,
            index_of[(x + 1, y)] + 1))

    # Helper to linearly interpolate between two 3d points using cloth coords
    def lerp(pt1, pt2, percents):
        return (pt1[0] + (pt2[0] - pt1[0]) * percents[0],
                pt1[1] + (pt2[1] - pt1[1]) * percents[0],
                pt1[2] + (pt2[2] - pt1[2]) * percents[1])

    def get_neighbour_indices(center, size):
        left = max(0, center[0] - size)
        right = min(node_density, center[0] + size)
        bottom = max(0, center[1] - size)
        top = min(node_density, center[1] + size)

        indices = []

        for i in range(left, right):
            for j in range(bottom, top):
                if (i, j) in nodes:  # could be that hole is too big
                    indices.append(nodes.index((i, j)))
        return indices

    idx_left = (0, 0)
    idx_right = (0, node_density - 1)
    anchor_index = nodes.index(idx_right)
    anchor_index2 = nodes.index(idx_left)

    with open(obj_path, 'w') as f:
        # f.write("# %d %d anchor index\n" % (anchor_index, anchor_index2))
        for n in nodes:
            coord = lerp(min_point, max_point, (n[0] / (node_density - 1),
                                                n[1] / (node_density - 1)))
            node_coords.append(coord)
            f.write("v %.4f %.4f %.4f\n" % coord)
        for tri in faces:
            f.write("f %d %d %d\n" % tri)
        f.close()

    if gen_fixed_anchors:
        # pinned nodes are from (1, 1) to (node_density-1, 1)
        node_y = np.arange(0, node_density)
        fixed_anchors = [nodes.index((node_density - 1, i)) for i in node_y]
        return obj_path, ([anchor_index], [anchor_index2]), \
               gt_loop_vertices, fixed_anchors

    return obj_path, ([anchor_index], [anchor_index2]), gt_loop_vertices


def plotter(hole1, hole2, type):
    plt.figure()

    def plot_one(h):
        pts = np.array([[h['x0'], h['y0']], [h['x0'], h['y1']],
                        [h['x1'], h['y0']], [h['x1'], h['y1']]])
        plt.scatter(pts[:, 0], pts[:, 1])

    plot_one(hole1)
    plot_one(hole2)
    plt.savefig(f'/tmp/debug_procedural_cloth_{type}.png')
