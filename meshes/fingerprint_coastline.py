"""The coarse ball, surface-refined around the analytic shoreline of
the sea-level fingerprint example: data/sea_level_fingerprint.msh.

The geometry and attributes of ball_with_buffer.py (body = unit ball,
attribute 1, surface boundary 1; buffer shell, attribute 2, outer
boundary 2 — thin, --buffer, as in earth_coastlines.py). The state is
examples/sea_level_fingerprint.cpp's, with
the same defaults: a super-Gaussian continent at the pole (+z), a
Gaussian ice cap on it, and the shoreline where the flotation field
q = rho_w SL0 - rho_i I0 crosses zero. The mesh is refined through a
planetmodel Refinement (common.SeaLevelBand) on the angular distance
to that circle (not on q itself, which saturates at rho_w x depth over
the open ocean and would drag the whole ocean into the band): elements
are
smallest in a ring of angular half-width --ring-width about the
shoreline circle — where the ocean function varies and the fingerprint
has its structure — and keep the background size elsewhere. Change a
state option here and in the example together, or the ring sits on the
wrong circle (the example accepts any ball-in-buffer mesh, so a
mismatch is wasteful, not wrong).

Beyond the shared options (--out, --verbose, --scale — which multiplies
both sizes below):
  --rhow --rhoi --ocean-depth --cont-amp --cont-width --cap-amp
  --cap-width         the example's state [0.05, 0.045, 1.0, 3.0, 0.7,
                      1.0, 0.35].
  --ring-width        refined angular half-width about the shoreline,
                      rad [0.12].
  --h                 element size on the spheres [0.4].
  --coast-size        element size inside the ring [0.06].
  --depth-width       radial half-width of the refined shell, radii [0.06].
  --buffer            outer radius of the buffer shell [1.5].
"""
import numpy as np
from scipy.optimize import brentq
from planetmodel import Geometry, Skeleton
from planetmodel.mesh3d import (MeshSpec, Shell, UniformInterfaces,
                                build_layered_mesh)

from common import SeaLevelBand, parser, report


def main() -> None:
    p = parser(__doc__)
    p.add_argument("--rhow", type=float, default=0.05)
    p.add_argument("--rhoi", type=float, default=0.045)
    p.add_argument("--ocean-depth", type=float, default=1.0)
    p.add_argument("--cont-amp", type=float, default=3.0)
    p.add_argument("--cont-width", type=float, default=0.7)
    p.add_argument("--cap-amp", type=float, default=1.0)
    p.add_argument("--cap-width", type=float, default=0.35)
    p.add_argument("--ring-width", type=float, default=0.12)
    p.add_argument("--h", type=float, default=0.4)
    p.add_argument("--coast-size", type=float, default=0.06)
    p.add_argument("--depth-width", type=float, default=0.06)
    p.add_argument("--buffer", type=float, default=1.5)
    args = p.parse_args()

    def flotation(theta):
        sl0 = (args.ocean_depth -
               args.cont_amp * np.exp(-0.5 * (theta / args.cont_width) ** 6))
        i0 = args.cap_amp * np.exp(-0.5 * (theta / args.cap_width) ** 2)
        return args.rhow * sl0 - args.rhoi * i0

    theta_c = brentq(flotation, 1e-6, np.pi)

    # The refinement field is the angular distance to the shoreline
    # circle itself (as a zonal wrapped lat-lon grid for the shared
    # refinement), NOT the flotation value: that saturates at
    # rho_w x ocean_depth over the open ocean, which a slope-calibrated
    # band mistakes for "near the shoreline" everywhere.
    lats = np.linspace(90.0, -90.0, 91)
    lons = np.linspace(0.0, 360.0, 73)
    theta = np.radians(90.0 - lats)
    q = np.repeat((theta - theta_c)[:, None], lons.size, axis=1)

    h = args.scale * args.h
    coast_size = args.scale * args.coast_size
    field = SeaLevelBand(lats, lons, q, band=args.ring_width,
                         size=coast_size,
                         far_size=2.0 * h, depth_width=args.depth_width)
    print(f"shoreline at colatitude {theta_c:.4f} rad, refined ring "
          f"half-width {args.ring_width:g}; sizes {coast_size:g} -> "
          f"{2.0 * h:g}")

    body = Geometry(Skeleton([0.0, 1.0]),
                    layer_names=["body"], interface_names=["surface"])
    buffer = Shell(radius=args.buffer, name="buffer")
    # The size far from the interfaces, capped at the buffer thickness
    # so the thin shell is not filled with flattened elements.
    far = max(h, min(2.0 * h, args.buffer - 1.0))
    sizing = UniformInterfaces(h, far, 10.0 * h)
    spec = MeshSpec(body, sizing, dimension=3, order=2, shells=[buffer],
                    refinements=[field],
                    meta={"shoreline_colatitude": theta_c,
                          "ring_width": args.ring_width})
    report(build_layered_mesh(spec, args.out / "sea_level_fingerprint",
                              verbose=args.verbose))


if __name__ == "__main__":
    main()
