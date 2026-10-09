"""Aspherical layered Earth models for the equilibrium-figures example:
the flattened_* and cmb_topo_* meshes, in MFEM's own format.

The layerings and radii are those of layered_earth.py (ICB 1230 km, CMB
3483 km, surface 6371 km, buffer shell 0.2), and two shapes make a
radial density non-barotropic without any hand-inserted lateral term —
the geometry itself denies the model a static state, and the density
restoration of examples/equilibrium_density.cpp has to generate the
non-spherical equilibrium density:

  flattened   every interface flattened proportionally: in 3-D the
              oblate h = -f r P2(cos theta), in 2-D (the theta = pi/2
              plane) the ellipse h = f r cos(2 phi); --flattening sets
              f. The displacement grows with r, so the only kink is at
              the surface, where the taper to the spherical outer
              boundary starts (the DtN needs it), declared as a knot.

  cmb_topo    oscillatory topography on the CMB alone: h = a w(r) X,
              with X = cos(k phi) in 2-D and P_k(cos theta) in 3-D
              (--amplitude, --degree), and w a hat that peaks at the
              CMB and vanishes at the ICB (the centre in the two-layer
              model) and at the surface — its kinks lie on interfaces
              the mesh honours, declared as knots. The inner core and
              the surface stay spherical; only the CMB is wavy.

  cmb_bump    a single Gaussian topographic bump on the CMB, centred on
              the +x axis: the same radial hat w, with the angular
              pattern exp(-2(1 - cos gamma)/width^2) (gamma the angular
              distance from +x; smooth and periodic), --amplitude and
              --width. Localised rather than oscillatory, so it is
              gentler on the order-2 geometry than cmb_topo.

Domain and boundary attributes as in layered_earth.py (layers from the
centre, buffer last; interfaces from the centre, outer last). The
manifest's meta records the shape parameters and marks the fluid layer
("fluid_layers", as in aspherical_body.py). Order-2 elements.

Files: {flattened,cmb_topo,cmb_bump}_{two,three}_layer_{2,3}d.mesh;
`--all` builds the twelve of them.

Used by: equilibrium_density (its -shape flat|cmb defaults).

Try, from a build's examples/ directory:
    ./equilibrium_density -dim 2
    ./equilibrium_density -dim 2 -shape cmb -no-ic
    poetry run python equilibrium_bodies.py --shape flat --dim 2 \
        --flattening 0.15 --name _f15 --out ../some/build/data
"""
import numpy as np

from planetmodel import CallableDisplacement, Geometry, Skeleton
from planetmodel.mesh3d import (MeshSpec, Shell, UniformInterfaces,
                                build_layered_mesh, export_mfem_mesh)

from common import parser, report

EARTH_RADIUS_KM = 6371.0
ICB = 1230.0 / EARTH_RADIUS_KM
CMB = 3483.0 / EARTH_RADIUS_KM
SURFACE = 1.0
BUFFER = 0.2  # default --buffer; the taper caps the safe flattening at
              # about buffer / 2 (the smoothstep's peak slope)

FLATTENING = 0.1   # default --flattening
AMPLITUDE = 0.05   # default --amplitude of the CMB topography
DEGREE = 4         # default --degree of its oscillation
WIDTH = 0.5        # default --width (radians) of the CMB bump

# Element sizes as in layered_earth.py, so the spherical and the
# aspherical runs of the example are comparable; --scale multiplies
# them (smaller is finer).
SIZING = {
    2: (0.085, 0.17, 0.85),
    3: (0.135, 0.27, 1.35),
}


def geometry(layers: int) -> Geometry:
    """The skeleton and names of the two- or three-layer Earth."""
    if layers == 2:
        return Geometry(Skeleton([0.0, CMB, SURFACE]),
                        layer_names=["fluid_core", "mantle"],
                        interface_names=["cmb", "surface"])
    if layers == 3:
        return Geometry(Skeleton([0.0, ICB, CMB, SURFACE]),
                        layer_names=["inner_core", "outer_core", "mantle"],
                        interface_names=["icb", "cmb", "surface"])
    raise ValueError(f"--layers must be 2 or 3, got {layers}")


def taper(r, buffer):
    """1 inside the body, to 0 across the buffer by a smoothstep: its
    slope vanishes at BOTH ends, so the radial compression peaks at
    1.5 f / buffer in mid-buffer (a quadratic taper puts 2 f / buffer
    right at the surface, which folds elements at f = 0.1)."""
    x = np.clip((SURFACE + buffer - r) / buffer, 0.0, 1.0)
    return x * x * (3.0 - 2.0 * x)


def flattened_shape(dim: int, f: float, buffer: float):
    """h = -f r P2(cos theta) (3-D, oblate) or f r cos(2 phi) (2-D),
    tapered off across the buffer; the kink at the surface is the
    displacement's only one."""

    def h(r, theta, phi):
        rb = np.minimum(r, SURFACE)
        if dim == 2:
            pattern = np.cos(2.0 * phi)
        else:
            pattern = -0.5 * (3.0 * np.cos(theta) ** 2 - 1.0)
        return f * rb * pattern * taper(r, buffer)

    return CallableDisplacement(h, knots=[SURFACE], name="flattening")


def cmb_shape(layers: int, dim: int, a: float, degree: int):
    """h = a w(r) X(angle): the hat w peaks at the CMB and vanishes at
    the ICB (the centre in the two-layer model) and the surface, so
    only the CMB is displaced; every kink of w is an interface."""
    inner = ICB if layers == 3 else 0.0

    def h(r, theta, phi):
        up = (r - inner) / (CMB - inner)
        down = (SURFACE - r) / (SURFACE - CMB)
        w = np.clip(np.minimum(up, down), 0.0, 1.0)
        if dim == 2:
            pattern = np.cos(degree * phi)
        else:
            pattern = np.polynomial.legendre.Legendre.basis(degree)(
                np.cos(theta))
        return a * w * pattern

    knots = [ICB, CMB, SURFACE] if layers == 3 else [CMB, SURFACE]
    return CallableDisplacement(h, knots=knots, name="cmb topography")


def bump_shape(layers: int, dim: int, a: float, width: float):
    """h = a w(r) exp(-2(1 - cos gamma)/width^2): a single Gaussian
    topographic bump on the CMB, centred on the +x axis (gamma the
    angular distance from it — the cosine form is smooth and periodic),
    with the same radial hat w as the oscillatory topography."""
    inner = ICB if layers == 3 else 0.0

    def h(r, theta, phi):
        up = (r - inner) / (CMB - inner)
        down = (SURFACE - r) / (SURFACE - CMB)
        w = np.clip(np.minimum(up, down), 0.0, 1.0)
        if dim == 2:
            cosg = np.cos(phi)
        else:
            cosg = np.sin(theta) * np.cos(phi)
        g = np.exp(-2.0 * (1.0 - cosg) / (width * width))
        return a * w * g

    knots = [ICB, CMB, SURFACE] if layers == 3 else [CMB, SURFACE]
    return CallableDisplacement(h, knots=knots, name="cmb bump")


def build(shape: str, layers: int, dim: int, args) -> None:
    words = {2: "two", 3: "three"}
    if shape == "flat":
        displacement = flattened_shape(dim, args.flattening, args.buffer)
        meta = {"flattening": args.flattening}
        stem = "flattened"
    elif shape == "cmb":
        displacement = cmb_shape(layers, dim, args.amplitude, args.degree)
        meta = {"amplitude": args.amplitude, "degree": args.degree}
        stem = "cmb_topo"
    else:
        displacement = bump_shape(layers, dim, args.amplitude, args.width)
        meta = {"amplitude": args.amplitude, "width": args.width}
        stem = "cmb_bump"
    meta["buffer"] = args.buffer
    meta["fluid_layers"] = [2] if layers == 3 else [1]

    name = f"{stem}_{words[layers]}_layer_{dim}d{args.name}"
    body = geometry(layers).stretched(displacement)
    sizing = UniformInterfaces(*(args.scale * v for v in SIZING[dim]))
    spec = MeshSpec(body, sizing, dimension=dim, order=2,
                    shells=[Shell(ratio=args.buffer, name="buffer")],
                    outer_boundary="spherical", meta=meta)

    # The reference (spherical) mesh, then the MFEM file with the nodes
    # moved, as in aspherical_body.py.
    reference = build_layered_mesh(spec, args.out / f"{name}_reference",
                                   verbose=args.verbose)
    exported = export_mfem_mesh(reference, args.out / name,
                                delivery="physical")
    report(reference)
    print(f"{exported.mesh_path.name}: nodes moved by the {stem} shape; "
          f"smallest Jacobian ratio "
          f"{exported.quality.get('min_ratio', float('nan')):.3f}")
    print(f"  manifest: {exported.manifest_path}")
    for p in (reference.msh_path, reference.manifest_path):
        p.unlink()


def main() -> None:
    p = parser(__doc__)
    p.add_argument("--shape", choices=("flat", "cmb", "bump"),
                   default="flat",
                   help="flattened interfaces, oscillatory CMB "
                        "topography, or a single Gaussian CMB bump")
    p.add_argument("--layers", type=int, default=3, choices=(2, 3),
                   help="2 for fluid core + mantle, 3 with a solid inner core")
    p.add_argument("--dim", type=int, default=2, choices=(2, 3),
                   help="2 for a disc, 3 for a ball")
    p.add_argument("--flattening", type=float, default=FLATTENING,
                   help="flattening of the flat shape (default %(default)s)")
    p.add_argument("--amplitude", type=float, default=AMPLITUDE,
                   help="amplitude of the CMB topography (default %(default)s)")
    p.add_argument("--degree", type=int, default=DEGREE,
                   help="angular degree of the CMB topography "
                        "(default %(default)s)")
    p.add_argument("--width", type=float, default=WIDTH,
                   help="angular width (radians) of the CMB bump "
                        "(default %(default)s)")
    p.add_argument("--buffer", type=float, default=BUFFER,
                   help="buffer-shell thickness; the flattening taper "
                        "needs about 2 f of room (default %(default)s)")
    p.add_argument("--scale", type=float, default=1.0,
                   help="scale factor on the element sizes (smaller is "
                        "finer — e.g. 0.5 for picture-quality maps)")
    p.add_argument("--name", default="",
                   help="suffix on the file names, for parameter sweeps")
    p.add_argument("--all", action="store_true",
                   help="build the twelve meshes the example uses")
    args = p.parse_args()

    if args.all:
        for shape in ("flat", "cmb", "bump"):
            for layers in (2, 3):
                for dim in (2, 3):
                    build(shape, layers, dim, args)
    else:
        build(args.shape, args.layers, args.dim, args)


if __name__ == "__main__":
    main()
