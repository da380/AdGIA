"""A box cut by a planar fault, under topography: data/faulted_box_{2d,3d}.msh.

The box of the coseismic examples, x in [-W/2, W/2] (3-D: also y) and
z from -D up to a free surface z = h(x, y) (z is the second coordinate in
2-D). A planar fault lies inside it. The fault plane is extended to the
boundary of the body and the body fragmented by it, so that the two sides
of the fault are separate volumes: the footwall, below the plane, is
domain attribute 1 and the hanging wall, above it, is 2. The cut consists
of the fault patch (boundary attribute 1), across which the examples
prescribe the slip, and its extension to the boundary (2), across which
the displacement stays continuous; the fault edge is where the two meet.
The outer boundary is the free surface (3) and the remaining box sides,
split by the plane into the footwall's (4) and the hanging wall's (5),
which the examples clamp.

The fault geometry follows examples/fault_box.hpp: the centre at
x = --fault-x (3-D: y = 0) and depth --fault-depth below z = 0, the dip
from horizontal towards +x (rotated by the strike in 3-D), the strike
clockwise from the y axis, the length along dip and (3-D) the width along
strike. The plane must leave the body away from its corners, and the
patch must lie strictly inside it, below the surface.

The surface (--topography range, the default) is a mountain range raised
over the hanging wall of the thrust, as long-term slip on such a fault
builds one: a steep front facing the footwall, winding along strike, a
crest and a long back slope towards +x, a shallow foreland basin in front,
the range dying out towards the ends of the box along strike, and gentle
rolling hills over everything; --relief scales all of it (default 1 in
2-D, 2 in 3-D, where the range is a wider feature of the view), and
--topography flat gives z = 0. In gmsh the surface is a spline through
the topography (3-D: a surface lofted through splines along x), the
body is closed under it, and the fault plane is clipped to the body
before the fragmenting, so the fault stays planar and the mesh conforms
to the surface.

Sizing: --fault-size on the patch, --tip-size at its edge (the two ends
in 2-D: where an abruptly ending slip concentrates the stress, and where
the slip of a discretisation that pins the edge to zero ramps down over
one element), --surface-size on the free surface (to resolve the
topography and the surface observations), each growing linearly at the
rate --gradation to --far-size; the defaults depend on the dimension.
--scale multiplies the whole size field (every size, with the growth
distances kept), so 0.5 is a uniform halving: about four times the
elements in 2-D, eight in 3-D. Linear triangles or tetrahedra.

Beside the mesh, a JSON file of the same name records the box, the fault
(with its frame: unit normal into the hanging wall, down-dip and strike
directions), the topography (its kind, relief, highest point, largest
curvature and its profile along y = 0), the sizes used and the attribute
dictionary.

    python faulted_box.py --dimension 3 --out DIR
    python faulted_box.py --all --scale 0.5 --name half --out DIR
    python faulted_box.py --all --topography flat --name flat --out DIR

Used by: coseismic_fault; tests.
"""
from __future__ import annotations

import json
import math

import gmsh
import numpy as np

from common import parser, sizes

DOMAIN = {"footwall": 1, "hanging_wall": 2}
BOUNDARY = {"fault": 1, "fault_extension": 2, "surface": 3,
            "clamped_footwall": 4, "clamped_hanging_wall": 5}

#: Per dimension: the factor on the height of the default topography.
RELIEF = {2: 1.0, 3: 2.0}

#: Per dimension: element size on the fault, at its edge, on the free
#: surface, far from all three, and the growth of the size per unit
#: distance.
SIZING = {
    2: {"fault": 0.05, "tip": 0.0125, "surface": 0.2, "far": 0.4,
        "gradation": 0.25},
    3: {"fault": 0.15, "tip": 0.075, "surface": 0.4, "far": 0.8,
        "gradation": 0.3},
}


def range_front(x, y, relief):
    """The surface height of the default topography (y = 0 in 2-D)."""
    front = -1.6 + 0.3 * np.sin(np.pi * y / 5.0)
    rise = 0.5 * (1.0 + np.tanh((x - front) / 0.35))
    crest = -0.7
    back = np.exp(-(np.maximum(x - crest, 0.0) / 2.6) ** 2)
    along = 0.65 + 0.35 * np.cos(np.pi * y / 5.0)
    ridge = 0.32 * rise * back * along
    foreland = -0.06 * np.exp(-((x - front + 0.9) / 0.8) ** 2) * along
    hills = (0.045 * np.sin(2 * np.pi * x / 3.3 + 0.6)
             * np.cos(2 * np.pi * y / 4.1 + 1.1)
             + 0.03 * np.sin(2 * np.pi * (x + 0.6 * y) / 2.7 + 2.0)
             + 0.02 * np.cos(2 * np.pi * (0.4 * x - y) / 3.1 + 0.3))
    return relief * (ridge + foreland + hills)


def fault_frame(dim: int, dip_deg: float, strike_deg: float):
    """Down-dip direction, strike direction and upward unit normal, as in
    fault_box.hpp (3-D: x east, y north, z up; normal = down-dip x strike)."""
    dip, strike = math.radians(dip_deg), math.radians(strike_deg)
    if dim == 2:
        return ((math.cos(dip), -math.sin(dip)), None,
                (math.sin(dip), math.cos(dip)))
    s = (math.sin(strike), math.cos(strike), 0.0)
    d = (math.cos(dip) * math.cos(strike), -math.cos(dip) * math.sin(strike),
         -math.sin(dip))
    n = (d[1] * s[2] - d[2] * s[1], d[2] * s[0] - d[0] * s[2],
         d[0] * s[1] - d[1] * s[0])
    return d, s, n


def dot(a, b) -> float:
    return sum(x * y for x, y in zip(a, b))


def combine(p, *terms):
    """p + sum of t * v over the (t, v) terms."""
    q = list(p)
    for t, v in terms:
        for i in range(len(q)):
            q[i] += t * v[i]
    return q


def build_body(occ, dim, W, D, height, z_top, spacing):
    """The body under the surface z = height(x, y): a face (2-D) or volume
    (3-D) as a list of (dim, tag)."""
    x0, x1 = -0.5 * W, 0.5 * W
    if dim == 2:
        xs = np.linspace(x0, x1, max(41, int(math.ceil(W / spacing)) + 1))
        top = [occ.addPoint(x, float(height(x, 0.0)), 0.0) for x in xs]
        bottom_left = occ.addPoint(x0, -D, 0.0)
        bottom_right = occ.addPoint(x1, -D, 0.0)
        loop = occ.addCurveLoop([occ.addLine(bottom_left, bottom_right),
                                 occ.addLine(bottom_right, top[-1]),
                                 occ.addSpline(top),
                                 occ.addLine(top[0], bottom_left)])
        return [(2, occ.addPlaneSurface([loop]))]
    overhang = 0.02 * W
    n = max(21, int(math.ceil((W + 2 * overhang) / spacing)) + 1)
    xs = np.linspace(x0 - overhang, x1 + overhang, n)
    wires = []
    for y in np.linspace(x0 - overhang, x1 + overhang, n):
        points = [occ.addPoint(x, y, float(height(x, y))) for x in xs]
        wires.append(occ.addWire([occ.addSpline(points)]))
    surface = occ.addThruSections(wires, makeSolid=False, makeRuled=False)
    depth = D + z_top + 1.0
    swept = occ.extrude(surface, 0.0, 0.0, -depth)
    volumes = [e for e in swept if e[0] == 3]
    box = occ.addBox(x0, x0, -D, W, W, D + z_top + 0.5)
    body, _ = occ.intersect(volumes, [(3, box)])
    return body


def cut_plane(occ, dim, centre, frame, extent):
    """The fault plane through the centre, as a segment (2-D) or rectangle
    (3-D) of half-size extent, unclipped."""
    down_dip, strike, _ = frame
    if dim == 2:
        a = combine(centre, (-extent, down_dip))
        b = combine(centre, (extent, down_dip))
        return (1, occ.addLine(occ.addPoint(a[0], a[1], 0.0),
                               occ.addPoint(b[0], b[1], 0.0)))
    return (2, rectangle(occ, centre, down_dip, strike, extent, extent))


def rectangle(occ, centre, down_dip, strike, half_dip, half_strike) -> int:
    corners = [combine(centre, (sl * half_dip, down_dip),
                       (sw * half_strike, strike))
               for sl, sw in ((-1, -1), (1, -1), (1, 1), (-1, 1))]
    p = [occ.addPoint(*c) for c in corners]
    lines = [occ.addLine(p[i], p[(i + 1) % 4]) for i in range(4)]
    return occ.addPlaneSurface([occ.addCurveLoop(lines)])


def check_geometry(args, dim, centre, frame, height, margin):
    """Refuse a patch outside the body or too near its surface, and a plane
    passing too near a corner of the body."""
    down_dip, strike, normal = frame
    W, D, L = args.width, args.depth, args.length
    half = [(-0.5, 0.5)] if dim == 2 else [(-0.5, 0.5), (-0.5, 0.5)]
    samples = np.linspace(-0.5, 0.5, 21)
    for sl in samples:
        for sw in (samples if dim == 3 else [0.0]):
            p = combine(centre, (sl * L, down_dip))
            if dim == 3:
                p = combine(p, (sw * args.fault_width, strike))
            horizontal = p[:dim - 1]
            z = p[dim - 1]
            y = horizontal[1] if dim == 3 else 0.0
            if (any(abs(c) >= 0.5 * W - margin for c in horizontal)
                    or z <= -D + margin
                    or z >= float(height(horizontal[0], y)) - margin):
                raise SystemExit("the fault patch must lie strictly inside the "
                                 "body, below its surface: shrink --length "
                                 "or --fault-width, or move the fault")
    del half
    corners = []
    xs = (-0.5 * W, 0.5 * W)
    for x in xs:
        for y in (xs if dim == 3 else [0.0]):
            top = float(height(x, y))
            for z in (-D, top):
                corners.append([x, y, z] if dim == 3 else [x, z])
    if min(abs(dot([c - o for c, o in zip(v, centre)], normal))
           for v in corners) < margin:
        raise SystemExit("the fault plane passes too close to a corner of the "
                         "body: change the dip or strike, or move the fault")


def build(args, dim: int, out_name: str) -> dict:
    W, D = args.width, args.depth
    s = {k: getattr(args, f"{k}_size") for k in ("fault", "tip", "surface",
                                                  "far")}
    s = {k: (SIZING[dim][k] if v is None else v) for k, v in s.items()}
    gradation = (SIZING[dim]["gradation"] if args.gradation is None
                 else args.gradation)
    fault_size, tip_size, surface_size, far_size = sizes(
        args, s["fault"], s["tip"], s["surface"], s["far"])
    frame = fault_frame(dim, args.dip, args.strike)
    down_dip, strike, normal = frame
    centre = ([args.fault_x, -args.fault_depth] if dim == 2
              else [args.fault_x, 0.0, -args.fault_depth])
    relief = (0.0 if args.topography == "flat"
              else RELIEF[dim] if args.relief is None else args.relief)

    def height(x, y):
        return range_front(x, y, relief)

    # The highest point and the largest curvature of the surface (the
    # largest eigenvalue of the Hessian in 3-D), on a fine grid.
    grid = np.linspace(-0.5 * W, 0.5 * W, 401)
    step = grid[1] - grid[0]
    if dim == 3:
        gx, gy = np.meshgrid(grid, grid, indexing="ij")
        hz = height(gx, gy)
        hxx = np.diff(hz, 2, axis=0)[:, 1:-1] / step ** 2
        hyy = np.diff(hz, 2, axis=1)[1:-1, :] / step ** 2
        hxy = (hz[2:, 2:] - hz[2:, :-2] - hz[:-2, 2:] + hz[:-2, :-2]) / (
            4.0 * step ** 2)
        curvature = float(np.max(np.abs(hxx + hyy) / 2.0 + np.sqrt(
            ((hxx - hyy) / 2.0) ** 2 + hxy ** 2)))
    else:
        hz = height(grid, 0.0)
        curvature = float(np.max(np.abs(np.diff(hz, 2)))) / step ** 2
    z_top = max(float(np.max(hz)), 0.0)
    margin = 0.02 * min(W, D) + 2.0 * fault_size
    check_geometry(args, dim, centre, frame, height, margin)

    gmsh.initialize()
    try:
        gmsh.option.setNumber("General.Terminal", 1 if args.verbose else 0)
        gmsh.model.add(out_name)
        occ = gmsh.model.occ
        body = build_body(occ, dim, W, D, height, z_top,
                          min(0.5 * surface_size, 0.1 * W))
        extent = 2.0 * math.sqrt(dim - 1) * max(W, D + z_top)
        plane = cut_plane(occ, dim, centre, frame, extent)
        clipped, _ = occ.intersect([plane], body, removeObject=True,
                                   removeTool=False)
        if dim == 2:
            a = combine(centre, (-0.5 * args.length, down_dip))
            b = combine(centre, (0.5 * args.length, down_dip))
            patch = (1, occ.addLine(occ.addPoint(a[0], a[1], 0.0),
                                    occ.addPoint(b[0], b[1], 0.0)))
        else:
            patch = (2, rectangle(occ, centre, down_dip, strike,
                                  0.5 * args.length, 0.5 * args.fault_width))
        _, pieces = occ.fragment(body, clipped + [patch])
        occ.synchronize()
        model = gmsh.model

        # The points and section curves the surface was built from survive
        # as free entities, which gmsh would mesh into nodes no element
        # uses: keep only the closure of the two sides.
        closure = {(dim, t) for i in range(len(body)) for _, t in pieces[i]}
        layer = list(closure)
        while layer and layer[0][0] > 0:
            layer = sorted(set(model.getBoundary(layer, combined=False,
                                                 oriented=False)))
            closure.update(layer)
        occ.remove([e for d in range(dim + 1) for e in model.getEntities(d)
                    if e not in closure])
        occ.synchronize()

        def side(d: int, tag: int) -> float:
            cm = occ.getCenterOfMass(d, tag)[:dim]
            return dot([x - c for x, c in zip(cm, centre)], normal)

        n_body = len(body)
        n_clipped = len(clipped)
        fault = sorted({t for _, t in pieces[n_body + n_clipped]})
        extension = sorted({t for i in range(n_body, n_body + n_clipped)
                            for _, t in pieces[i]} - set(fault))
        cut = set(fault) | set(extension)
        volumes = sorted({t for i in range(n_body) for _, t in pieces[i]})
        hanging = [t for t in volumes if side(dim, t) > 0.0]
        foot = [t for t in volumes if side(dim, t) < 0.0]
        if len(hanging) != 1 or len(foot) != 1 or not fault or not extension:
            raise SystemExit("fragmenting the body did not give two sides "
                             "separated by the fault and its extension")

        # A face of the box (a side or the bottom) has its centre of mass on
        # the box's plane; the bounding boxes of faces with intersection
        # curves for edges are too loose to tell.
        tol = 1e-6 * max(W, D)
        surface, clamped_foot, clamped_hanging = [], [], []
        for _, t in model.getEntities(dim - 1):
            if t in cut:
                continue
            cm = occ.getCenterOfMass(dim - 1, t)
            on_side = any(abs(abs(cm[i]) - 0.5 * W) < tol
                          for i in range(dim - 1))
            on_bottom = abs(cm[dim - 1] + D) < tol
            if not on_side and not on_bottom:
                surface.append(t)
            elif side(dim - 1, t) > 0.0:
                clamped_hanging.append(t)
            else:
                clamped_foot.append(t)

        model.addPhysicalGroup(dim, foot, DOMAIN["footwall"], "footwall")
        model.addPhysicalGroup(dim, hanging, DOMAIN["hanging_wall"],
                               "hanging_wall")
        for name, tags in (("fault", fault), ("fault_extension", extension),
                           ("surface", surface),
                           ("clamped_footwall", clamped_foot),
                           ("clamped_hanging_wall", clamped_hanging)):
            model.addPhysicalGroup(dim - 1, tags, BOUNDARY[name], name)

        edge = sorted({t for _, t in model.getBoundary(
            [(dim - 1, t) for t in fault], combined=True, oriented=False)})
        field = model.mesh.field

        def linear_growth(key: str, tags: list[int], size: float,
                          growth_distance: float) -> int:
            distance = field.add("Distance")
            field.setNumbers(distance, key, tags)
            field.setNumber(distance, "Sampling", 400 if dim == 2 else 60)
            threshold = field.add("Threshold")
            field.setNumber(threshold, "InField", distance)
            field.setNumber(threshold, "SizeMin", size)
            field.setNumber(threshold, "SizeMax", far_size)
            field.setNumber(threshold, "DistMin", 0.0)
            field.setNumber(threshold, "DistMax", growth_distance)
            return threshold

        entity_key = "CurvesList" if dim == 2 else "SurfacesList"
        edge_key = "PointsList" if dim == 2 else "CurvesList"
        smallest = field.add("Min")
        field.setNumbers(smallest, "FieldsList", [
            linear_growth(entity_key, fault, fault_size,
                          (s["far"] - s["fault"]) / gradation),
            linear_growth(edge_key, edge, tip_size,
                          (s["far"] - s["tip"]) / gradation),
            linear_growth(entity_key, surface, surface_size,
                          (s["far"] - s["surface"]) / gradation)])
        field.setAsBackgroundMesh(smallest)
        for option in ("MeshSizeExtendFromBoundary", "MeshSizeFromPoints",
                       "MeshSizeFromCurvature"):
            gmsh.option.setNumber(f"Mesh.{option}", 0)

        model.mesh.generate(dim)
        if dim == 3:
            model.mesh.optimize("Netgen")
        element_type = 2 if dim == 2 else 4
        element_tags = model.mesh.getElementsByType(element_type)[0]
        quality = min(model.mesh.getElementQualities(element_tags,
                                                     "minSICN"))
        if quality <= 0.0:
            raise SystemExit(f"inverted element (min SICN {quality:.3f}): "
                             "nudge a size by a few per cent")
        gmsh.option.setNumber("Mesh.MshFileVersion", 2.2)
        gmsh.option.setNumber("Mesh.SaveAll", 0)
        msh_path = args.out / f"{out_name}.msh"
        gmsh.write(str(msh_path))
        n_nodes = len(model.mesh.getNodes()[0])
    finally:
        gmsh.finalize()

    profile_x = np.linspace(-0.5 * W, 0.5 * W, 801)
    profile_z = height(profile_x, 0.0)
    fault_record = {"x": args.fault_x, "depth": args.fault_depth,
                    "dip": args.dip, "length": args.length,
                    "centre": centre, "normal": list(normal),
                    "down_dip": list(down_dip)}
    if dim == 3:
        fault_record.update({"strike": args.strike,
                             "width": args.fault_width,
                             "strike_direction": list(strike)})
    return {
        "schema": "adgia.faulted_box/1",
        "dimension": dim,
        "mesh": {"file": msh_path.name, "nodes": n_nodes,
                 "elements": len(element_tags), "min_sicn": quality},
        "box": {"width": W, "depth": D},
        "fault": fault_record,
        "topography": {"kind": args.topography, "relief": relief,
                       "z_max": z_top, "max_curvature": curvature,
                       "profile": {"x": profile_x.tolist(),
                                   "z": profile_z.tolist()}},
        "sizes": {"fault": fault_size, "tip": tip_size,
                  "surface": surface_size, "far": far_size,
                  "gradation": args.scale * gradation},
        "attributes": {"domain": DOMAIN, "boundary": BOUNDARY},
    }


def main() -> None:
    p = parser(__doc__)
    p.add_argument("--dimension", type=int, choices=(2, 3), default=2,
                   help="2 (a rectangle) or 3 (a box) (default %(default)s)")
    p.add_argument("--all", action="store_true",
                   help="write both dimensions")
    p.add_argument("--width", type=float, default=8.0,
                   help="box width W (default %(default)s)")
    p.add_argument("--depth", type=float, default=4.0,
                   help="box depth D below z = 0 (default %(default)s)")
    p.add_argument("--topography", choices=("range", "flat"), default="range",
                   help="the free surface: a mountain range over the hanging "
                        "wall, or flat (default %(default)s)")
    p.add_argument("--relief", type=float,
                   help="factor on the height of the topography (default: 1 "
                        "in 2-D, 2 in 3-D)")
    p.add_argument("--fault-x", type=float, default=0.0,
                   help="x of the fault centre (default %(default)s)")
    p.add_argument("--fault-depth", type=float, default=1.0,
                   help="depth of the fault centre below z = 0 "
                        "(default %(default)s)")
    p.add_argument("--dip", type=float, default=30.0,
                   help="dip in degrees from horizontal (default %(default)s)")
    p.add_argument("--strike", type=float, default=0.0,
                   help="3-D: strike in degrees clockwise from the y axis "
                        "(default %(default)s)")
    p.add_argument("--length", type=float, default=1.5,
                   help="along-dip length of the fault (default %(default)s)")
    p.add_argument("--fault-width", type=float, default=1.5,
                   help="3-D: along-strike width of the fault "
                        "(default %(default)s)")
    p.add_argument("--fault-size", type=float,
                   help="element size on the fault (default: 0.05 in 2-D, "
                        "0.15 in 3-D)")
    p.add_argument("--tip-size", type=float,
                   help="element size at the fault edge (default: 0.0125 in "
                        "2-D, 0.075 in 3-D)")
    p.add_argument("--surface-size", type=float,
                   help="element size on the free surface (default: 0.2 in "
                        "2-D, 0.4 in 3-D)")
    p.add_argument("--far-size", type=float,
                   help="element size far from the fault and the surface "
                        "(default: 0.4 in 2-D, 0.8 in 3-D)")
    p.add_argument("--gradation", type=float,
                   help="growth of the element size per unit distance "
                        "(default: 0.25 in 2-D, 0.3 in 3-D)")
    p.add_argument("--name", default="",
                   help="suffix on the file names, to keep variants side by "
                        "side (default: none)")
    args = p.parse_args()
    if not 0.0 < args.dip < 90.0:
        raise SystemExit("--dip must lie strictly between 0 and 90 degrees")

    args.out.mkdir(parents=True, exist_ok=True)
    for dim in ((2, 3) if args.all else (args.dimension,)):
        out_name = f"faulted_box_{dim}d" + (f"_{args.name}" if args.name
                                            else "")
        record = build(args, dim, out_name)
        json_path = args.out / f"{out_name}.json"
        json_path.write_text(json.dumps(record, indent=2) + "\n")
        m = record["mesh"]
        kind = "triangles" if dim == 2 else "tetrahedra"
        print(f"{m['file']}: {m['elements']} {kind}, {m['nodes']} nodes, "
              f"min SICN {m['min_sicn']:.3f}")
        print(f"  manifest: {json_path}")


if __name__ == "__main__":
    main()
