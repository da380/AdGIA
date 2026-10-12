"""The unit ball in a thin buffer shell, surface-refined along the real
coastlines: data/earth_coastlines.msh.

Geometry and attributes as in ball_with_buffer.py: the body is the unit
ball (domain attribute 1, surface boundary 1) and a buffer shell (domain
attribute 2, outer boundary 2) reaches to radius --buffer (1.5 by
default, so 0.5 thick; the far-field condition is exact at any radius
and a thick shell only costs elements). Order-2 tetrahedra, 3-D only.
Lengths are in units of the Earth's radius.

The mesh is refined through a planetmodel Refinement (common.SeaLevelBand)
on the grid's pre-processed shoreline field `coast_rad`: the great-circle
distance (radians) to the coast of the near-native-resolution topography
(where its sign changes, plus the shallow shelves). earth_data/
topography_grid.py builds it, and its docstring says why a band on the
topography's value cannot define the coast evenly. Elements have size
--coast-size within --coast-width radians of that coast AND within
--depth-width of the surface radially, growing with slope --gradation to
2 * --h outside; the sphere sizing is --h, growing over 10 * --h. No
coastline polylines are isolated: it is one sizing field. --scale
multiplies --h and --coast-size.

The default grid is committed beside this script,
ice7g_topography_lmax64.npz (ICE-7G at the present day: topography at
lmax 64, coastline at lmax 180, shelf band 400 m), so the build makes
this mesh like any other. For another date or band, export a new grid
with earth_data/topography_grid.py (which needs pyslfp; see
earth_data/README.md) and pass it with --topography (the chain in
examples/ice_age_loading.cpp and meshes/README.md, "The coastline
mesh"). What sharpens the picture is --coast-size (or --scale), at the
usual cubic cost. The manifest meta records the grid used (topography
file, ice model, date in ka BP, lmax, shelf band).

Options beyond the shared ones (--out, --verbose, --scale); lengths in
Earth radii unless stated, defaults in brackets:
  --topography FILE   grid from topography_grid.py [the committed
                      lmax-64 ICE-7G grid beside this script].
  --coast-width       angular half-width of the fine band about the
                      coastline, radians [0.04].
  --gradation         growth of the size beyond the band, size per radian
                      (and per radius radially); lower is safer and
                      costlier [1.5].
  --h                 element size on the spheres [0.4].
  --coast-size        element size inside the band [0.035].
  --depth-width       radial half-width of the refined shell about the
                      surface [0.05].
  --buffer            outer radius of the buffer shell [1.5].
"""
from pathlib import Path

import numpy as np
from planetmodel import Geometry, Skeleton
from planetmodel.mesh3d import MeshSpec, Shell, build_layered_mesh

from common import SeaLevelBand, ball_in_buffer_sizing, parser, report

DEFAULT_TOPOGRAPHY = Path(__file__).with_name("ice7g_topography_lmax64.npz")


def main() -> None:
    p = parser(__doc__)
    p.add_argument("--topography", type=Path, default=DEFAULT_TOPOGRAPHY,
                   metavar="FILE",
                   help="topography/coastline .npz written by "
                        "earth_data/topography_grid.py (default: the "
                        "committed lmax-64 ICE-7G grid beside this script)")
    p.add_argument("--coast-width", type=float, default=0.04,
                   help="angular half-width of the finely meshed band "
                        "about the coast, in radians (default "
                        "%(default)s, about 250 km)")
    p.add_argument("--gradation", type=float, default=1.5,
                   help="growth rate of the element size beyond the band, "
                        "size per radian of distance to the coast (and per "
                        "Earth radius radially); lower is safer and "
                        "costlier (default %(default)s)")
    p.add_argument("--h", type=float, default=0.4,
                   help="element size on the spheres, in Earth radii "
                        "(default %(default)s; multiplied by --scale)")
    p.add_argument("--coast-size", type=float, default=0.035,
                   help="element size inside the coastal band, in Earth "
                        "radii (default %(default)s, about 220 km; "
                        "multiplied by --scale)")
    p.add_argument("--depth-width", type=float, default=0.05,
                   help="radial half-width of the refined shell about the "
                        "surface, in Earth radii (default %(default)s)")
    p.add_argument("--buffer", type=float, default=1.5,
                   help="OUTER RADIUS of the buffer shell, in Earth radii "
                        "(default %(default)s)")
    args = p.parse_args()

    h = args.scale * args.h
    coast_size = args.scale * args.coast_size
    if not args.topography.is_file():
        raise SystemExit(f"{args.topography}: no such file; export a grid "
                         "with earth_data/topography_grid.py")
    data = np.load(args.topography, allow_pickle=False)
    missing = [k for k in ("coast_rad", "coast_lats", "coast_lons",
                           "band_m", "meta", "version") if k not in data]
    if missing:
        raise SystemExit(
            f"{args.topography}: missing {', '.join(missing)} — re-export "
            "the grid with the current earth_data/topography_grid.py")

    field = SeaLevelBand(data["coast_lats"], data["coast_lons"],
                         data["coast_rad"],
                         band=args.coast_width, size=coast_size,
                         far_size=2.0 * h,
                         depth_width=args.depth_width,
                         gradation=args.gradation)
    frac = float((data["coast_rad"] < args.coast_width).mean())
    print(f"{args.topography.name}: within {args.coast_width:g} rad of "
          f"the coast (shelf band {float(data['band_m']):g} m) on "
          f"{frac:.0%} of the surface; sizes {coast_size:g} -> "
          f"{2.0 * h:g}")

    body = Geometry(Skeleton([0.0, 1.0]),
                    layer_names=["body"], interface_names=["surface"])
    buffer = Shell(radius=args.buffer, name="buffer")
    sizing = ball_in_buffer_sizing(h, args.buffer)
    meta = [float(v) for v in data["meta"]]
    spec = MeshSpec(body, sizing, dimension=3, order=2, shells=[buffer],
                    refinements=[field],
                    meta={"topography": str(args.topography),
                          "ice_model": str(data["version"]),
                          "date_ka": meta[0], "lmax": int(meta[1]),
                          "band_m": float(data["band_m"])})
    report(build_layered_mesh(spec, args.out / "earth_coastlines",
                              verbose=args.verbose))


if __name__ == "__main__":
    main()
