"""PREM-like layered Earth models with a buffer shell, plain or aspherical:
the earth_{fluidcore,innercore}[_<shape>]_{2,3}d meshes.

Radii are in units of the Earth's radius, 6371 km: the inner-core
boundary at 1230 km (0.1931), the core-mantle boundary at 3483 km
(0.5467) and the surface at 6371 km (1). Two layerings are built, as a
disc (2-D) or a ball (3-D):

  --layers 2   fluid core, mantle
  --layers 3   solid inner core, fluid outer core, mantle

A buffer shell of relative thickness --buffer (default 0.2, so reaching
to 1.2) sits outside the surface. Domain attributes count the layers from
the centre, with the buffer last; boundary attributes count the
interfaces from the centre (ICB, CMB, surface, outer). The examples
recognise the layering from the number of domain attributes. Order-2
elements. Element sizes (common.SIZING, in surface radii): 0.085 on the
interfaces growing to 0.17 over 0.85 in 2-D, 0.135 to 0.27 over 1.35 in
3-D; --scale multiplies them.

--shape selects the geometry:

  none      spherical layers (the default). Files earth_fluidcore_2d.msh,
            earth_innercore_2d.msh, earth_fluidcore_3d.msh,
            earth_innercore_3d.msh (gmsh format).
            Used by: elastogravity_layered, gauged_fluid_cavity,
            self_gravitating_relaxation, self_gravitating_solvers,
            sliding_fluid_ellipse, equilibrium_density, the tests.

The other shapes are aspherical, written in MFEM's own format as
earth_{fluidcore,innercore}_<stem>_{2,3}d.mesh. They make a radial
density non-barotropic without any hand-inserted lateral term: the
geometry itself denies the model a static state, and the density
restoration of examples/equilibrium_density.cpp has to generate the
non-spherical equilibrium density. The outer boundary stays spherical (the
DtN needs it). The manifest's meta records the shape parameters, the
buffer thickness and marks the fluid layer ("fluid_layers", as in
aspherical_body.py).

  flat      (stem flat) every interface flattened proportionally: in 3-D
            the oblate h = -f r P2(cos theta), in 2-D (the theta = pi/2
            plane) the ellipse h = f r cos(2 phi); --flattening sets f.
            The displacement grows with r, so the only kink is at the
            surface, where the taper to the spherical outer boundary
            starts, declared as a knot.

  cmbtopo   (stem cmbtopo) oscillatory topography on the CMB alone:
            h = a w(r) X, with X = cos(k phi) in 2-D and P_k(cos theta)
            in 3-D (--amplitude, --degree), and w a hat that peaks at the
            CMB and vanishes at the ICB (the centre in the two-layer
            model) and at the surface; its kinks lie on interfaces the
            mesh honours, declared as knots. The inner core and the
            surface stay spherical; only the CMB is wavy.

  cmbbump   (stem cmbbump) a single Gaussian topographic bump on the CMB,
            centred on the +x axis: the same radial hat w, with the
            angular pattern exp(-2(1 - cos gamma)/width^2) (gamma the
            angular distance from +x; smooth and periodic), --amplitude
            and --width. Localised rather than oscillatory, so it is
            gentler on the order-2 geometry than cmbtopo.

Used by the aspherical shapes: equilibrium_density (its
-shape flat|cmb|bump defaults).

--all builds every layering and dimension: with no --shape, all sixteen
files (4 spherical + 12 aspherical); with --shape S, the four of S;
--shape shaped (with --all only) is the twelve aspherical ones.

Try, from a build's examples/ directory:
    ./equilibrium_density -dim 2
    ./equilibrium_density -dim 2 -shape cmb -no-ic
    poetry run python layered_earth.py --shape flat --layers 3 --dim 2 \
        --flattening 0.15 --name _f15 --out ../some/build/data
"""
import numpy as np

from planetmodel import CallableDisplacement
from planetmodel.mesh3d import (MeshSpec, Shell, UniformInterfaces,
                                build_layered_mesh)

from common import (SIZING, SURFACE, build_mfem, geometry, hat_knots, parser,
                    radial_hat, report, sizes)

BUFFER = 0.2       # default --buffer; the taper caps the safe flattening at
                   # about buffer / 2 (the smoothstep's peak slope)

FLATTENING = 0.1   # default --flattening
AMPLITUDE = 0.05   # default --amplitude of the CMB topography
DEGREE = 4         # default --degree of its oscillation
WIDTH = 0.5        # default --width (radians) of the CMB bump


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

    def h(r, theta, phi):
        w = radial_hat(r, layers)
        if dim == 2:
            pattern = np.cos(degree * phi)
        else:
            pattern = np.polynomial.legendre.Legendre.basis(degree)(
                np.cos(theta))
        return a * w * pattern

    return CallableDisplacement(h, knots=hat_knots(layers),
                                name="cmb topography")


def bump_shape(layers: int, dim: int, a: float, width: float):
    """h = a w(r) exp(-2(1 - cos gamma)/width^2): a single Gaussian
    topographic bump on the CMB, centred on the +x axis (gamma the
    angular distance from it — the cosine form is smooth and periodic),
    with the same radial hat w as the oscillatory topography."""

    def h(r, theta, phi):
        w = radial_hat(r, layers)
        if dim == 2:
            cosg = np.cos(phi)
        else:
            cosg = np.sin(theta) * np.cos(phi)
        g = np.exp(-2.0 * (1.0 - cosg) / (width * width))
        return a * w * g

    return CallableDisplacement(h, knots=hat_knots(layers), name="cmb bump")


ASPHERICAL = ("flat", "cmbtopo", "cmbbump")


def build(shape: str, layers: int, dim: int, args) -> None:
    """Build one mesh: `shape` is "none" or one of ASPHERICAL."""
    words = {2: "fluidcore", 3: "innercore"}
    sizing = UniformInterfaces(*sizes(args, *SIZING[dim]))
    shell = Shell(ratio=args.buffer, name="buffer")
    if shape == "none":
        name = f"earth_{words[layers]}_{dim}d{args.name}"
        spec = MeshSpec(geometry(layers), sizing, dimension=dim, order=2,
                        shells=[shell])
        report(build_layered_mesh(spec, args.out / name,
                                  verbose=args.verbose))
        return

    if shape == "flat":
        displacement = flattened_shape(dim, args.flattening, args.buffer)
        meta = {"flattening": args.flattening}
    elif shape == "cmbtopo":
        displacement = cmb_shape(layers, dim, args.amplitude, args.degree)
        meta = {"amplitude": args.amplitude, "degree": args.degree}
    else:
        displacement = bump_shape(layers, dim, args.amplitude, args.width)
        meta = {"amplitude": args.amplitude, "width": args.width}
    meta["buffer"] = args.buffer
    meta["fluid_layers"] = [2] if layers == 3 else [1]

    name = f"earth_{words[layers]}_{shape}_{dim}d{args.name}"
    body = geometry(layers).stretched(displacement)
    spec = MeshSpec(body, sizing, dimension=dim, order=2, shells=[shell],
                    outer_boundary="spherical", meta=meta)

    # The reference (spherical) mesh, then the MFEM file with the nodes
    # moved.
    build_mfem(spec, name, args, f"nodes moved by the {shape} shape")


def main() -> None:
    p = parser(__doc__)
    p.add_argument("--shape", choices=("none",) + ASPHERICAL + ("shaped",),
                   default=None,
                   help="spherical layers (none, the default for a single "
                        "mesh), flattened interfaces (flat), oscillatory CMB "
                        "topography (cmbtopo) or a single Gaussian CMB bump "
                        "(cmbbump); 'shaped' (with --all) is the three "
                        "aspherical ones")
    p.add_argument("--layers", type=int, default=2, choices=(2, 3),
                   help="number of layers: 2 for fluid core + mantle, 3 "
                        "with a solid inner core (default %(default)s; "
                        "ignored with --all)")
    p.add_argument("--dim", type=int, default=2, choices=(2, 3),
                   help="2 for a disc, 3 for a ball (default %(default)s; "
                        "ignored with --all)")
    p.add_argument("--flattening", type=float, default=FLATTENING,
                   help="dimensionless flattening f of the flat shape, "
                        "the displacement being f times the radius "
                        "(default %(default)s)")
    p.add_argument("--amplitude", type=float, default=AMPLITUDE,
                   help="peak radial displacement of the CMB topography "
                        "or bump, in units of the surface radius "
                        "(default %(default)s)")
    p.add_argument("--degree", type=int, default=DEGREE,
                   help="angular degree (a number of waves) of the cmbtopo "
                        "shape (default %(default)s)")
    p.add_argument("--width", type=float, default=WIDTH,
                   help="angular width of the CMB bump, in radians "
                        "(default %(default)s)")
    p.add_argument("--buffer", type=float, default=BUFFER,
                   help="buffer-shell THICKNESS in units of the surface "
                        "radius (the shell reaches to 1 + this); the "
                        "flattening taper needs about 2 f of room "
                        "(default %(default)s). Unlike aspherical_body.py, "
                        "where --buffer is a yes/no switch")
    p.add_argument("--name", default="",
                   help="suffix appended to the file names, for parameter "
                        "sweeps (default: none)")
    p.add_argument("--all", action="store_true",
                   help="build every layering and dimension: all sixteen "
                        "meshes, or those of --shape if given; ignores "
                        "--layers and --dim")
    args = p.parse_args()

    if args.all:
        shapes = {None: ("none",) + ASPHERICAL, "shaped": ASPHERICAL}.get(
            args.shape, (args.shape,))
        for shape in shapes:
            for layers in (2, 3):
                for dim in (2, 3):
                    build(shape, layers, dim, args)
    elif args.shape == "shaped":
        p.error("--shape shaped needs --all")
    else:
        build(args.shape or "none", args.layers, args.dim, args)


if __name__ == "__main__":
    main()
