"""The coarse ball, surface-refined along the real coastlines:
data/earth_coastlines.msh.

The geometry and attributes of ball_with_buffer.py (body = unit ball,
attribute 1, surface boundary 1; buffer shell, attribute 2, outer
boundary 2 — but thin, --buffer, since the far-field condition is
exact at any radius and a thick shell only costs elements), with the
mesh refined through a planetmodel Refinement (common.SeaLevelBand) on
the grid's pre-processed shoreline field `coast_rad`: the great-circle
distance to the coast of the near-native-resolution topography (its
sign changes, plus the shallow shelves — postprocess/topography_grid.py
builds it, and its docstring says why a band on the topography's value
cannot define the coast evenly). Elements are smallest within
--coast-width radians of that coast AND radially near the surface,
growing to the background size outside. No coastline polylines are
isolated; it is all one sizing field.

The default grid is committed beside this script,
ice7g_topography_lmax64.npz (ICE-7G at the present day: topography at
lmax 64, coastline at lmax 180, shelf band 400 m), so the build makes
this mesh like any other. For another date or band, export a new grid
with postprocess/topography_grid.py (which needs pyslfp; see
postprocess/README.md) and pass it with --topography — the chain in
examples/ice_age_loading.cpp and meshes/README.md ("The coastline
mesh"). What sharpens the picture is --coast-size (or --scale), at the
usual cubic cost.

Beyond the shared options (--out, --verbose, --scale — which multiplies
both sizes below):
  --topography FILE   grid from topography_grid.py [the committed
                      lmax-64 ICE-7G grid beside this script].
  --coast-width       angular half-width of the fine band about the
                      coastline, rad [0.04].
  --gradation         growth rate of the size beyond the band; lower
                      is safer and costlier [1.5].
  --h                 element size on the spheres [0.4].
  --coast-size        element size inside the band [0.035].
  --depth-width       radial half-width of the refined shell, radii [0.05].
  --buffer            outer radius of the buffer shell [1.5].
"""
from pathlib import Path

import numpy as np
from planetmodel import Geometry, Skeleton
from planetmodel.mesh3d import (MeshSpec, Shell, UniformInterfaces,
                                build_layered_mesh)

from common import SeaLevelBand, parser, report

DEFAULT_TOPOGRAPHY = Path(__file__).with_name("ice7g_topography_lmax64.npz")


def main() -> None:
    p = parser(__doc__)
    p.add_argument("--topography", type=Path, default=DEFAULT_TOPOGRAPHY)
    p.add_argument("--coast-width", type=float, default=0.04)
    p.add_argument("--gradation", type=float, default=1.5)
    p.add_argument("--h", type=float, default=0.4)
    p.add_argument("--coast-size", type=float, default=0.035)
    p.add_argument("--depth-width", type=float, default=0.05)
    p.add_argument("--buffer", type=float, default=1.5)
    args = p.parse_args()

    h = args.scale * args.h
    coast_size = args.scale * args.coast_size
    data = np.load(args.topography, allow_pickle=False)
    if "coast_rad" not in data:
        raise SystemExit(
            f"{args.topography}: no coast_rad field — re-export the grid "
            "with the current postprocess/topography_grid.py")

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
    # The size far from the interfaces, capped at the buffer thickness
    # so the thin shell is not filled with flattened elements.
    far = max(h, min(2.0 * h, args.buffer - 1.0))
    sizing = UniformInterfaces(h, far, 10.0 * h)
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
