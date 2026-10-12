"""The unit ball in a thin buffer shell, surface-refined along a shoreline:
the earth_coastlines.msh and fingerprint_ring_3d.msh meshes, selected by
--field.

Geometry and attributes: the body is the unit ball (domain attribute 1,
surface boundary 1) and a buffer shell (domain attribute 2, outer
boundary 2) reaches to radius --buffer (1.5 by default, so 0.5 thick; the
far-field condition is exact at any radius and a thick shell only costs
elements). Order-2 tetrahedra, 3-D only. Lengths are in units of the
body's radius (the Earth's, for --field earth). The mesh is refined
through a planetmodel Refinement (common.SeaLevelBand) on a surface field
that is an ANGULAR distance to the shoreline; the sphere sizing is --h,
growing over 10 * --h, and elements have size --coast-size in the band,
growing with slope --gradation (--field earth) to 2 * --h outside. No
coastline polylines are isolated: it is one sizing field. --scale
multiplies --h and --coast-size.

--field earth   (default) data/earth_coastlines.msh. The shoreline is the
  real one. The refinement field is the grid's pre-processed shoreline
  field `coast_rad`: the great-circle distance (radians) to the coast of
  the near-native-resolution topography (where its sign changes, plus the
  shallow shelves). earth_data/topography_grid.py builds it, and its
  docstring says why a band on the topography's value cannot define the
  coast evenly. Elements have size --coast-size within --coast-width
  radians of that coast AND within --depth-width of the surface radially,
  growing with slope --gradation to 2 * --h outside.

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
  Used by: ice_age_loading and sea_level_fingerprint through -m.

--field analytic   data/fingerprint_ring_3d.msh. The shoreline is that of
  the state of examples/sea_level_fingerprint.cpp, with the same
  defaults: a super-Gaussian continent at the pole (+z), a Gaussian ice
  cap on it, and the shoreline where the flotation field
  q = rho_w SL0 - rho_i I0 crosses zero (a circle of colatitude theta_c,
  found here by root-finding; the manifest meta records it). The
  refinement field is the angular distance to that circle (not q itself,
  which saturates at rho_w x depth over the open ocean and would drag the
  whole ocean into the band): elements have size --coast-size in a ring
  of angular half-width --ring-width about the shoreline circle, where the
  ocean function varies and the fingerprint has its structure, growing to
  2 * --h elsewhere. Change a state option here and in the example
  together, or the ring sits on the wrong circle (the example accepts any
  ball-in-buffer mesh, so a mismatch is wasteful, not wrong).
  Used by: sea_level_fingerprint (its default mesh).

Options beyond the shared ones (--out, --verbose, --scale); lengths in
body radii unless stated, defaults [earth / analytic] where they differ:
  --field {earth,analytic}  which mesh [earth].
  --h                 element size on the spheres [0.4].
  --coast-size        element size inside the band or ring [0.035 / 0.06].
  --depth-width       radial half-width of the refined shell about the
                      surface [0.05 / 0.06].
  --buffer            outer radius of the buffer shell [1.5].
 earth only:
  --topography FILE   grid from topography_grid.py [the committed
                      lmax-64 ICE-7G grid beside this script].
  --coast-width       angular half-width of the fine band about the
                      coastline, radians [0.04].
  --gradation         growth of the size beyond the band, size per radian
                      (and per radius radially); lower is safer and
                      costlier [1.5].
 analytic only:
  --rhow --rhoi --ocean-depth --cont-amp --cont-width --cap-amp
  --cap-width         the example's state, in its non-dimensional units
                      (widths in radians) [0.05, 0.045, 1.0, 3.0, 0.7,
                      1.0, 0.35]; they must equal the example's
                      -rhow -rhoi -ocean-depth -cont-amp -cont-width
                      -cap-amp -cap-width.
  --ring-width        refined angular half-width about the shoreline,
                      radians [0.12].
"""
from pathlib import Path

import numpy as np
from scipy.optimize import brentq
from planetmodel import Geometry, Skeleton
from planetmodel.mesh3d import MeshSpec, Shell, build_layered_mesh

from common import SeaLevelBand, ball_in_buffer_sizing, parser, report

DEFAULT_TOPOGRAPHY = Path(__file__).with_name("ice7g_topography_lmax64.npz")

#: Defaults of the options whose value depends on --field.
FIELD_DEFAULTS = {
    "earth": {"coast_size": 0.035, "depth_width": 0.05},
    "analytic": {"coast_size": 0.06, "depth_width": 0.06},
}

#: Options that exist for one --field only (argparse dest names).
EARTH_ONLY = ("topography", "coast_width", "gradation")
ANALYTIC_ONLY = ("rhow", "rhoi", "ocean_depth", "cont_amp", "cont_width",
                 "cap_amp", "cap_width", "ring_width")


def earth_refinement(args, h):
    """The SeaLevelBand on the grid's `coast_rad` field and the manifest
    meta recording the grid; the output name is earth_coastlines."""
    topography = args.topography or DEFAULT_TOPOGRAPHY
    coast_size = args.scale * args.coast_size
    if not topography.is_file():
        raise SystemExit(f"{topography}: no such file; export a grid "
                         "with earth_data/topography_grid.py")
    data = np.load(topography, allow_pickle=False)
    missing = [k for k in ("coast_rad", "coast_lats", "coast_lons",
                           "band_m", "meta", "version") if k not in data]
    if missing:
        raise SystemExit(
            f"{topography}: missing {', '.join(missing)} — re-export "
            "the grid with the current earth_data/topography_grid.py")

    field = SeaLevelBand(data["coast_lats"], data["coast_lons"],
                         data["coast_rad"],
                         band=args.coast_width, size=coast_size,
                         far_size=2.0 * h,
                         depth_width=args.depth_width,
                         gradation=args.gradation)
    frac = float((data["coast_rad"] < args.coast_width).mean())
    print(f"{topography.name}: within {args.coast_width:g} rad of "
          f"the coast (shelf band {float(data['band_m']):g} m) on "
          f"{frac:.0%} of the surface; sizes {coast_size:g} -> "
          f"{2.0 * h:g}")
    meta = [float(v) for v in data["meta"]]
    return field, {"topography": str(topography),
                   "ice_model": str(data["version"]),
                   "date_ka": meta[0], "lmax": int(meta[1]),
                   "band_m": float(data["band_m"])}


def analytic_refinement(args, h):
    """The SeaLevelBand on the angular distance to the flotation zero of
    the fingerprint example's state, and the manifest meta recording the
    shoreline colatitude; the output name is fingerprint_ring_3d."""

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

    coast_size = args.scale * args.coast_size
    field = SeaLevelBand(lats, lons, q, band=args.ring_width,
                         size=coast_size,
                         far_size=2.0 * h, depth_width=args.depth_width)
    print(f"shoreline at colatitude {theta_c:.4f} rad, refined ring "
          f"half-width {args.ring_width:g}; sizes {coast_size:g} -> "
          f"{2.0 * h:g}")
    return field, {"shoreline_colatitude": theta_c,
                   "ring_width": args.ring_width}


def main() -> None:
    p = parser(__doc__)
    p.add_argument("--field", choices=("earth", "analytic"),
                   default="earth",
                   help="the shoreline to refine along: the real one from "
                        "an ICE-7G topography grid (earth, writes "
                        "earth_coastlines) or the analytic flotation ring "
                        "of the sea-level fingerprint example (analytic, "
                        "writes fingerprint_ring_3d) (default %(default)s)")
    p.add_argument("--h", type=float, default=0.4,
                   help="element size on the spheres, in body radii "
                        "(Earth radii for --field earth) (default "
                        "%(default)s; multiplied by --scale)")
    p.add_argument("--coast-size", type=float, default=None,
                   help="element size inside the coastal band or ring, in "
                        "body radii (default 0.035 for earth, about 220 "
                        "km, 0.06 for analytic; multiplied by --scale)")
    p.add_argument("--depth-width", type=float, default=None,
                   help="radial half-width of the refined shell about the "
                        "surface, in body radii (default 0.05 for earth, "
                        "0.06 for analytic)")
    p.add_argument("--buffer", type=float, default=1.5,
                   help="OUTER RADIUS of the buffer shell, in body radii "
                        "(default %(default)s)")

    # --field earth only.
    p.add_argument("--topography", type=Path, default=None, metavar="FILE",
                   help="[earth] topography/coastline .npz written by "
                        "earth_data/topography_grid.py (default: the "
                        "committed lmax-64 ICE-7G grid beside this script)")
    p.add_argument("--coast-width", type=float, default=None,
                   help="[earth] angular half-width of the finely meshed "
                        "band about the coast, in radians (default 0.04, "
                        "about 250 km)")
    p.add_argument("--gradation", type=float, default=None,
                   help="[earth] growth rate of the element size beyond the "
                        "band, size per radian of distance to the coast "
                        "(and per Earth radius radially); lower is safer "
                        "and costlier (default 1.5)")

    # --field analytic only. Defaults of the first seven:
    # examples/sea_level_fingerprint.cpp's g_rho_w ... g_cap_width; keep
    # the two in step.
    same = " Must equal the example's {} (default %(default)s)."
    p.add_argument("--rhow", type=float, default=0.05,
                   help="[analytic] water density, non-dimensional." +
                        same.format("-rhow"))
    p.add_argument("--rhoi", type=float, default=0.045,
                   help="[analytic] ice density, non-dimensional." +
                        same.format("-rhoi"))
    p.add_argument("--ocean-depth", type=float, default=1.0,
                   help="[analytic] initial ocean depth, in the example's "
                        "length units." + same.format("-ocean-depth"))
    p.add_argument("--cont-amp", type=float, default=3.0,
                   help="[analytic] continent height (super-Gaussian at the "
                        "pole), in the example's length units." +
                        same.format("-cont-amp"))
    p.add_argument("--cont-width", type=float, default=0.7,
                   help="[analytic] continent angular width, in radians." +
                        same.format("-cont-width"))
    p.add_argument("--cap-amp", type=float, default=1.0,
                   help="[analytic] ice-cap thickness (Gaussian at the "
                        "pole), in the example's length units." +
                        same.format("-cap-amp"))
    p.add_argument("--cap-width", type=float, default=0.35,
                   help="[analytic] ice-cap angular width, in radians." +
                        same.format("-cap-width"))
    p.add_argument("--ring-width", type=float, default=0.12,
                   help="[analytic] angular half-width of the finely meshed "
                        "ring about the shoreline circle, in radians "
                        "(default %(default)s)")
    args = p.parse_args()

    # Options of the other --field are an error, not silently ignored;
    # the ones with --field-dependent defaults are resolved here.
    other = ANALYTIC_ONLY if args.field == "earth" else EARTH_ONLY
    given = [o for o in other if getattr(args, o) not in
             (None, p.get_default(o))]
    if given:
        p.error("--" + ", --".join(o.replace("_", "-") for o in given) +
                f" not valid with --field {args.field}")
    for key, value in FIELD_DEFAULTS[args.field].items():
        if getattr(args, key) is None:
            setattr(args, key, value)
    if args.field == "earth":
        if args.coast_width is None:
            args.coast_width = 0.04
        if args.gradation is None:
            args.gradation = 1.5

    h = args.scale * args.h
    refine = earth_refinement if args.field == "earth" else \
        analytic_refinement
    name = "earth_coastlines" if args.field == "earth" else \
        "fingerprint_ring_3d"
    field, meta = refine(args, h)

    body = Geometry(Skeleton([0.0, 1.0]),
                    layer_names=["body"], interface_names=["surface"])
    buffer = Shell(radius=args.buffer, name="buffer")
    sizing = ball_in_buffer_sizing(h, args.buffer)
    spec = MeshSpec(body, sizing, dimension=3, order=2, shells=[buffer],
                    refinements=[field], meta=meta)
    report(build_layered_mesh(spec, args.out / name, verbose=args.verbose))


if __name__ == "__main__":
    main()
