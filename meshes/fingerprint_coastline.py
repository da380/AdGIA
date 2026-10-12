"""The unit ball in a thin buffer shell, surface-refined around the
analytic shoreline of the sea-level fingerprint example:
data/fingerprint_ring_3d.msh.

Geometry and attributes as in ball_with_buffer.py: the body is the unit
ball (domain attribute 1, surface boundary 1) and a buffer shell (domain
attribute 2, outer boundary 2) reaches to radius --buffer (1.5 by
default). Order-2 tetrahedra, 3-D only. Lengths are in units of the
body's radius.

The state is that of examples/sea_level_fingerprint.cpp, with the same
defaults: a super-Gaussian continent at the pole (+z), a Gaussian ice
cap on it, and the shoreline where the flotation field
q = rho_w SL0 - rho_i I0 crosses zero (a circle of colatitude theta_c,
found here by root-finding; the manifest meta records it). The mesh is
refined through a planetmodel Refinement (common.SeaLevelBand) on the
angular distance to that circle (not on q itself, which saturates at
rho_w x depth over the open ocean and would drag the whole ocean into
the band): elements have size --coast-size in a ring of angular
half-width --ring-width about the shoreline circle, where the ocean
function varies and the fingerprint has its structure, growing to
2 * --h elsewhere; the sphere sizing is --h, growing over 10 * --h.
Change a state option here and in the example together, or the ring
sits on the wrong circle (the example accepts any ball-in-buffer mesh,
so a mismatch is wasteful, not wrong). --scale multiplies --h and
--coast-size.

Options beyond the shared ones (--out, --verbose, --scale); lengths in
body radii, defaults in brackets:
  --rhow --rhoi --ocean-depth --cont-amp --cont-width --cap-amp
  --cap-width         the example's state, in its non-dimensional units
                      (widths in radians) [0.05, 0.045, 1.0, 3.0, 0.7,
                      1.0, 0.35]; they must equal the example's
                      -rhow -rhoi -ocean-depth -cont-amp -cont-width
                      -cap-amp -cap-width.
  --ring-width        refined angular half-width about the shoreline,
                      radians [0.12].
  --h                 element size on the spheres [0.4].
  --coast-size        element size inside the ring [0.06].
  --depth-width       radial half-width of the refined shell about the
                      surface [0.06].
  --buffer            outer radius of the buffer shell [1.5].
"""
import numpy as np
from scipy.optimize import brentq
from planetmodel import Geometry, Skeleton
from planetmodel.mesh3d import MeshSpec, Shell, build_layered_mesh

from common import SeaLevelBand, ball_in_buffer_sizing, parser, report


def main() -> None:
    p = parser(__doc__)
    # Defaults of the first seven: examples/sea_level_fingerprint.cpp's
    # g_rho_w ... g_cap_width; keep the two in step.
    same = " Must equal the example's {} (default %(default)s)."
    p.add_argument("--rhow", type=float, default=0.05,
                   help="water density, non-dimensional." +
                        same.format("-rhow"))
    p.add_argument("--rhoi", type=float, default=0.045,
                   help="ice density, non-dimensional." +
                        same.format("-rhoi"))
    p.add_argument("--ocean-depth", type=float, default=1.0,
                   help="initial ocean depth, in the example's length "
                        "units." + same.format("-ocean-depth"))
    p.add_argument("--cont-amp", type=float, default=3.0,
                   help="continent height (super-Gaussian at the pole), "
                        "in the example's length units." +
                        same.format("-cont-amp"))
    p.add_argument("--cont-width", type=float, default=0.7,
                   help="continent angular width, in radians." +
                        same.format("-cont-width"))
    p.add_argument("--cap-amp", type=float, default=1.0,
                   help="ice-cap thickness (Gaussian at the pole), in the "
                        "example's length units." + same.format("-cap-amp"))
    p.add_argument("--cap-width", type=float, default=0.35,
                   help="ice-cap angular width, in radians." +
                        same.format("-cap-width"))
    p.add_argument("--ring-width", type=float, default=0.12,
                   help="angular half-width of the finely meshed ring "
                        "about the shoreline circle, in radians (default "
                        "%(default)s)")
    p.add_argument("--h", type=float, default=0.4,
                   help="element size on the spheres, in body radii "
                        "(default %(default)s; multiplied by --scale)")
    p.add_argument("--coast-size", type=float, default=0.06,
                   help="element size inside the ring, in body radii "
                        "(default %(default)s; multiplied by --scale)")
    p.add_argument("--depth-width", type=float, default=0.06,
                   help="radial half-width of the refined shell about the "
                        "surface, in body radii (default %(default)s)")
    p.add_argument("--buffer", type=float, default=1.5,
                   help="OUTER RADIUS of the buffer shell, in body radii "
                        "(default %(default)s)")
    args = p.parse_args()

    def flotation(theta):
        sl0 = (args.ocean_depth -
               args.cont_amp * np.exp(-0.5 * (theta / args.cont_width) ** 6))
        i0 = args.cap_amp * np.exp(-0.5 * (theta / args.cap_width) ** 2)
        return args.rhow * sl0 - args.rhoi * i0

    try:
        theta_c = brentq(flotation, 1e-6, np.pi)
    except ValueError:
        raise SystemExit("the state has no shoreline; adjust the state "
                         "options") from None

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
    sizing = ball_in_buffer_sizing(h, args.buffer)
    spec = MeshSpec(body, sizing, dimension=3, order=2, shells=[buffer],
                    refinements=[field],
                    meta={"shoreline_colatitude": theta_c,
                          "ring_width": args.ring_width})
    report(build_layered_mesh(spec, args.out / "fingerprint_ring_3d",
                              verbose=args.verbose))


if __name__ == "__main__":
    main()
