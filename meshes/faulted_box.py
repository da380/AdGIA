"""A box cut by a planar fault: data/faulted_box_{2d,3d}.msh.

The box of the coseismic examples, x in [-W/2, W/2] (3-D: also y) and
z in [-D, 0] (z is the second coordinate in 2-D), with the free surface at
z = 0. A planar fault lies inside it. The fault plane is extended to the
box boundary and the box fragmented by it, so that the two sides of the
fault are separate volumes: the footwall, below the plane, is domain
attribute 1 and the hanging wall, above it, is 2. The cut consists of the
fault patch (boundary attribute 1), across which the examples prescribe
the slip, and its extension to the box boundary (2), across which the
displacement stays continuous; the fault edge is where the two meet. The
outer boundary is the free surface (3) and the remaining box sides,
split by the plane into the footwall's (4) and the hanging wall's (5),
which the examples clamp.

The fault geometry follows examples/fault_box.hpp: the centre at
x = --fault-x (3-D: y = 0) and depth --fault-depth, the dip from
horizontal towards +x (rotated by the strike in 3-D), the strike
clockwise from the y axis, the length along dip and (3-D) the width along
strike. The plane must leave the box away from its corners, and the
patch must lie strictly inside the box.

Sizing: --fault-size on the patch, --tip-size at its edge (the two ends
in 2-D: where an abruptly ending slip concentrates the stress, and where
the slip of a discretisation that pins the edge to zero ramps down over
one element), each growing linearly at the rate --gradation to
--far-size; the defaults depend on the dimension. --scale multiplies the
whole size field (every size, --tip-size included, with the growth
distances kept), so 0.5 is a uniform halving: about four times the
elements in 2-D, eight in 3-D. Linear triangles or tetrahedra.

Beside the mesh, a JSON file of the same name records the box, the fault
(with its frame: unit normal into the hanging wall, down-dip and strike
directions), the sizes used and the attribute dictionary.

    python faulted_box.py --dimension 3 --out DIR
    python faulted_box.py --all --scale 0.5 --name half --out DIR

Used by: coseismic_fault; tests.
"""
from __future__ import annotations

import json
import math

import gmsh

from common import parser, sizes

DOMAIN = {"footwall": 1, "hanging_wall": 2}
BOUNDARY = {"fault": 1, "fault_extension": 2, "surface": 3,
            "clamped_footwall": 4, "clamped_hanging_wall": 5}

#: Per dimension: element size on the fault, at its edge, far from it, and
#: the growth of the size per unit distance.
SIZING = {
    2: {"fault": 0.05, "tip": 0.0125, "far": 0.4, "gradation": 0.25},
    3: {"fault": 0.15, "tip": 0.075, "far": 0.8, "gradation": 0.3},
}


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


def box_bounds(dim: int, width: float, depth: float):
    horizontal = [(-0.5 * width, 0.5 * width)] * (dim - 1)
    return horizontal + [(-depth, 0.0)]


def inside(point, bounds, margin: float) -> bool:
    return all(a + margin < x < b - margin for x, (a, b) in zip(point, bounds))


def chord(centre, direction, bounds):
    """Parameters t_lo < t_hi where centre + t direction meets the box boundary."""
    lo, hi = -math.inf, math.inf
    for c, d, (a, b) in zip(centre, direction, bounds):
        if abs(d) < 1e-14:
            continue
        t1, t2 = sorted(((a - c) / d, (b - c) / d))
        lo, hi = max(lo, t1), min(hi, t2)
    return lo, hi


def snap(point, bounds, scale: float):
    """The point with coordinates within round-off of a box side put on it."""
    p = list(point)
    for i, (a, b) in enumerate(bounds):
        for v in (a, b):
            if abs(p[i] - v) < 1e-9 * scale:
                p[i] = v
    return p


def geometry_2d(args, occ, centre, frame, bounds):
    """The rectangle fragmented by the three segments of the extended fault
    line; returns the fragment map and the tool indices of the patch and of
    the extension."""
    down_dip = frame[0]
    L = args.length
    t_lo, t_hi = chord(centre, down_dip, bounds)
    if not (t_lo < -0.5 * L and t_hi > 0.5 * L):
        raise SystemExit("the fault patch must lie strictly inside the box: "
                         "shrink --length or move the fault")
    ends = [snap(combine(centre, (t, down_dip)), bounds, args.depth)
            for t in (t_lo, t_hi)]
    corners = [(x, z) for x in bounds[0] for z in bounds[1]]
    margin = 0.05 * min(args.width, args.depth)
    for e in ends:
        if min(math.dist(e, c) for c in corners) < margin:
            raise SystemExit("the fault plane leaves the box too close to a "
                             "corner: change the dip or move the fault")
    (x0, x1), (z0, z1) = bounds
    box = occ.addRectangle(x0, z0, 0.0, x1 - x0, z1 - z0)
    a = combine(centre, (-0.5 * L, down_dip))
    b = combine(centre, (0.5 * L, down_dip))
    p = [occ.addPoint(x, z, 0.0) for x, z in (ends[0], a, b, ends[1])]
    upper = occ.addLine(p[0], p[1])
    patch = occ.addLine(p[1], p[2])
    lower = occ.addLine(p[2], p[3])
    _, pieces = occ.fragment([(2, box)], [(1, upper), (1, patch), (1, lower)])
    return pieces, [2], [1, 3]


def geometry_3d(args, occ, centre, frame, bounds):
    """The box fragmented by the fault plane clipped to it and by the fault
    patch; returns the fragment map and the tool indices of the patch and of
    the clipped plane."""
    down_dip, strike, normal = frame
    L, Wf = args.length, args.fault_width
    margin = 0.02 * min(args.width, args.depth)
    for sl in (-0.5, 0.5):
        for sw in (-0.5, 0.5):
            corner = combine(centre, (sl * L, down_dip), (sw * Wf, strike))
            if not inside(corner, bounds, margin):
                raise SystemExit("the fault patch must lie strictly inside "
                                 "the box: shrink --length/--fault-width or "
                                 "move the fault")
    vertices = [(x, y, z) for x in bounds[0] for y in bounds[1]
                for z in bounds[2]]
    if min(abs(dot([v - c for v, c in zip(vtx, centre)], normal))
           for vtx in vertices) < 2.0 * margin:
        raise SystemExit("the fault plane passes too close to a box corner: "
                         "change the dip or strike or move the fault")

    def rectangle(half_dip: float, half_strike: float) -> int:
        corners = [combine(centre, (sl * half_dip, down_dip),
                           (sw * half_strike, strike))
                   for sl, sw in ((-1, -1), (1, -1), (1, 1), (-1, 1))]
        p = [occ.addPoint(*c) for c in corners]
        lines = [occ.addLine(p[i], p[(i + 1) % 4]) for i in range(4)]
        return occ.addPlaneSurface([occ.addCurveLoop(lines)])

    (x0, x1), (y0, y1), (z0, z1) = bounds
    box = occ.addBox(x0, y0, z0, x1 - x0, y1 - y0, z1 - z0)
    span = 2.0 * math.dist((x0, y0, z0), (x1, y1, z1))
    plane = rectangle(span, span)
    clipped, _ = occ.intersect([(2, plane)], [(3, box)], removeObject=True,
                               removeTool=False)
    patch = rectangle(0.5 * L, 0.5 * Wf)
    _, pieces = occ.fragment([(3, box)], clipped + [(2, patch)])
    return pieces, [len(clipped) + 1], list(range(1, len(clipped) + 1))


def build(args, dim: int, out_name: str) -> dict:
    W, D = args.width, args.depth
    s = {k: getattr(args, f"{k}_size") for k in ("fault", "tip", "far")}
    s = {k: (SIZING[dim][k] if v is None else v) for k, v in s.items()}
    gradation = (SIZING[dim]["gradation"] if args.gradation is None
                 else args.gradation)
    fault_size, tip_size, far_size = sizes(args, s["fault"], s["tip"],
                                           s["far"])
    frame = fault_frame(dim, args.dip, args.strike)
    down_dip, strike, normal = frame
    centre = ([args.fault_x, -args.fault_depth] if dim == 2
              else [args.fault_x, 0.0, -args.fault_depth])
    bounds = box_bounds(dim, W, D)

    gmsh.initialize()
    try:
        gmsh.option.setNumber("General.Terminal", 1 if args.verbose else 0)
        gmsh.model.add(out_name)
        occ = gmsh.model.occ
        builder = geometry_2d if dim == 2 else geometry_3d
        pieces, patch_tools, extension_tools = builder(args, occ, centre,
                                                       frame, bounds)
        occ.synchronize()
        model = gmsh.model

        def side(d: int, tag: int) -> float:
            cm = occ.getCenterOfMass(d, tag)[:dim]
            return dot([x - c for x, c in zip(cm, centre)], normal)

        fault = sorted({t for i in patch_tools for _, t in pieces[i]})
        extension = sorted({t for i in extension_tools for _, t in pieces[i]}
                           - set(fault))
        cut = set(fault) | set(extension)
        volumes = [t for _, t in pieces[0]]
        hanging = [t for t in volumes if side(dim, t) > 0.0]
        foot = [t for t in volumes if side(dim, t) < 0.0]
        if len(hanging) != 1 or len(foot) != 1 or not fault or not extension:
            raise SystemExit("fragmenting the box did not give two sides "
                             "separated by the fault and its extension")

        surface, clamped_foot, clamped_hanging = [], [], []
        for _, t in model.getEntities(dim - 1):
            if t in cut:
                continue
            z = occ.getCenterOfMass(dim - 1, t)[dim - 1]
            if abs(z) < 1e-9 * D:
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

        fault_key = "CurvesList" if dim == 2 else "SurfacesList"
        edge_key = "PointsList" if dim == 2 else "CurvesList"
        smallest = field.add("Min")
        field.setNumbers(smallest, "FieldsList", [
            linear_growth(fault_key, fault, fault_size,
                          (s["far"] - s["fault"]) / gradation),
            linear_growth(edge_key, edge, tip_size,
                          (s["far"] - s["tip"]) / gradation)])
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
        "sizes": {"fault": fault_size, "tip": tip_size, "far": far_size,
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
                   help="box depth D (default %(default)s)")
    p.add_argument("--fault-x", type=float, default=0.0,
                   help="x of the fault centre (default %(default)s)")
    p.add_argument("--fault-depth", type=float, default=1.0,
                   help="depth of the fault centre (default %(default)s)")
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
    p.add_argument("--far-size", type=float,
                   help="element size far from the fault (default: 0.4 in "
                        "2-D, 0.8 in 3-D)")
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
