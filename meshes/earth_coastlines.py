"""The coarse ball, surface-refined where the sea level is near zero:
data/earth_coastlines.msh.

The second step of the real-coastline mesh chain: the geometry and
attributes of ball_with_buffer.py (body = unit ball, attribute 1,
surface boundary 1; buffer to radius 2, attribute 2, outer boundary 2),
with the mesh refined through a planetmodel Refinement (v1.2.4's
protocol) built from the truncated ICE-NG topography that
postprocess/topography_grid.py exported: elements are smallest where
|topography| is within --band metres of sea level AND the point is
radially near the surface, growing to the background size outside — so
real shorelines and the shallow shelves (where the flotation criterion
is delicate) attract refinement together, with no coastline isolation
at all. The element budget is set
by the band and the sizes; the defaults build a laptop-sized ~5k-tet
mesh, and the server runs shrink them.

Not part of the default mesh generation: the topography file needs
pyslfp (see the chain in examples/ice_age_loading.cpp). Run it by hand
with a planetmodel Python, e.g.
  poetry -C meshes run python meshes/earth_coastlines.py \\
      --topography <build>/postprocess/topography.npz --out <build>/data

Beyond the shared options (--out, --verbose):
  --topography FILE   the grid from topography_grid.py [topography.npz].
  --band METRES       |topography| treated as "at sea level" [400].
  --h                 element size on the spheres [0.45, the canned ball's].
  --coast-size        element size inside the band [0.18].
  --depth-width       radial half-width of the refined shell, radii [0.1].
"""
import json
from pathlib import Path

import numpy as np
from scipy.interpolate import RegularGridInterpolator
from planetmodel import Geometry, Skeleton
from planetmodel.mesh3d import (MeshSpec, Shell, UniformInterfaces,
                                build_layered_mesh)

from common import parser, report


class SeaLevelBand:
    """planetmodel Refinement: fine where the sea level is near zero.

    The effective distance at a reference point is the larger of
    |topography(direction)| in band units and |r - R| in depth widths;
    inside one unit the size is `size` (the plateau that lets the mesher
    realise it), then it grows linearly to `far_size` over one more
    unit. The topography grid is wrapped (poles and the 360-degree
    column included), so plain bilinear interpolation covers the sphere.
    """

    def __init__(self, lats, lons, topo_m, *, band_m, size, far_size,
                 radius=1.0, depth_width=0.1):
        self.size = float(size)
        self.far_size = float(far_size)
        self._band = float(band_m)
        self._radius = float(radius)
        self._depth = float(depth_width)
        order = np.argsort(lats)
        self._interp = RegularGridInterpolator(
            (lats[order], lons), topo_m[order, :],
            bounds_error=False, fill_value=None)

    def __call__(self, points):
        pts = np.asarray(points, dtype=float)
        r = np.linalg.norm(pts, axis=1)
        safe = np.maximum(r, 1e-12)
        lat = np.degrees(np.arcsin(np.clip(pts[:, 2] / safe, -1.0, 1.0)))
        lon = np.degrees(np.arctan2(pts[:, 1], pts[:, 0])) % 360.0
        t = np.abs(self._interp(np.column_stack([lat, lon]))) / self._band
        d = np.maximum(t, np.abs(r - self._radius) / self._depth)
        return np.minimum(self.far_size,
                          self.size + (self.far_size - self.size) *
                          np.maximum(0.0, d - 1.0))


def main() -> None:
    p = parser(__doc__)
    p.add_argument("--topography", type=Path, default=Path("topography.npz"))
    p.add_argument("--band", type=float, default=400.0)
    p.add_argument("--h", type=float, default=0.45)
    p.add_argument("--coast-size", type=float, default=0.18)
    p.add_argument("--depth-width", type=float, default=0.1)
    args = p.parse_args()

    data = np.load(args.topography, allow_pickle=False)
    field = SeaLevelBand(data["lats"], data["lons"], data["topo_m"],
                         band_m=args.band, size=args.coast_size,
                         far_size=2.0 * args.h,
                         depth_width=args.depth_width)
    frac = float((np.abs(data["topo_m"]) < args.band).mean())
    print(f"{args.topography.name}: |topo| < {args.band:g} m on "
          f"{frac:.0%} of the surface; sizes {args.coast_size:g} -> "
          f"{2.0 * args.h:g}")

    body = Geometry(Skeleton([0.0, 1.0]),
                    layer_names=["body"], interface_names=["surface"])
    buffer = Shell(radius=2.0, name="buffer")
    sizing = UniformInterfaces(args.h, 2.0 * args.h, 10.0 * args.h)
    meta = [float(v) for v in data["meta"]]
    spec = MeshSpec(body, sizing, dimension=3, order=2, shells=[buffer],
                    refinements=[field],
                    meta={"topography": str(args.topography),
                          "ice_model": str(data["version"]),
                          "date_ka": meta[0], "lmax": int(meta[1]),
                          "band_m": args.band})
    report(build_layered_mesh(spec, args.out / "earth_coastlines",
                              verbose=args.verbose))


if __name__ == "__main__":
    main()
