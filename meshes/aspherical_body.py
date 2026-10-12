"""A body with a non-spherical shape given by a formula: the aspherical
meshes, in MFEM's own format.

The body is a unit ball (disc in 2-D, radius 1) whose surface, and every
sphere inside it, is pushed along the radius by

    h(r, theta, phi) = eps * r * [ P2(cos theta) + beta * sin^2(theta) * cos(2 phi) ]

an oblate part and an elliptical equatorial part (eps and beta are
dimensionless; h is in units of the radius). In 2-D the mesh lies in the
plane theta = pi/2, so only the elliptical part is seen. gmsh meshes the
sphere; the shape is applied to the nodes when the mesh is written for
MFEM, and the export checks that no element folds. Order-2 elements.

Variants (--dim 2|3):

  default         the body alone. Domain attribute 1, boundary 1 (the
                  surface). Files aspherical_{2d,3d}.mesh.
  --buffer        a buffer shell appended out to radius 1.2, the
                  displacement tapered quadratically to zero across it so
                  the outer boundary stays a sphere (as the DtN and
                  multipole conditions need); the surface is declared a
                  knot of the displacement. Domain attributes 1 body, 2
                  buffer; boundary 1 surface, 2 outer sphere. Files
                  aspherical_buffer_{2d,3d}.mesh.
  --fluid-core    two layers, a core below radius --rc (default the
                  Earth's CMB, 3483/6371) and a mantle, always with the
                  buffer; the shape stretches both, so the core boundary
                  is aspherical and on element faces. Domain attributes 1
                  core, 2 mantle, 3 buffer; boundary 1 core boundary, 2
                  surface, 3 outer sphere. The manifest meta records eps,
                  beta, buffer and r_c and marks the core fluid through
                  "fluid_layers": [1]. Files
                  aspherical_fluid_core_buffer_{2d,3d}.mesh; with
                  --spherical the same meshes unstretched (eps = 0, same
                  topology), spherical_fluid_core_buffer_{2d,3d}.mesh.

`--all` builds all eight: the four single-layer meshes and the four
fluid-core ones.

Try them with the examples, e.g. from a build's examples/:
    ./anisotropic_elasticity -m ../data/aspherical_3d.mesh -o 1
    ./poisson_dtn -m ../data/aspherical_buffer_2d.mesh -o 2 -mth 1
    ./elastogravity_layered -m ../data/aspherical_buffer_2d.mesh -o 2 -s 2
    ./slipping_interface -m ../data/aspherical_fluid_core_buffer_2d.mesh
"""
import numpy as np

from planetmodel import CallableDisplacement, Geometry, Skeleton
from planetmodel.mesh3d import MeshSpec, Shell, UniformInterfaces

from common import CMB, build_mfem, parser, sizes

EPS = 0.05     # amplitude of the shape, relative to the radius
BETA = 0.5     # elliptical part relative to the oblate part
BUFFER = 0.2   # thickness of the buffer shell relative to the body's radius
RC = CMB       # radius of the fluid core (--fluid-core), in body radii

SIZING = {2: (0.1, 0.2, 1.0), 3: (0.25, 0.5, 2.5)}
# The two-layer body: in 3D a tenth finer, which keeps gmsh's slivers
# between the core boundary and the surface out (minSICN 0.34 against 0.26).
SIZING_FLUID_CORE = {2: (0.1, 0.2, 1.0), 3: (0.225, 0.45, 2.25)}


def shape(r, theta, phi, eps=None, beta=None):
    """The radial displacement of the body, growing with the radius."""
    eps = EPS if eps is None else eps
    beta = BETA if beta is None else beta
    p2 = 0.5 * (3.0 * np.cos(theta) ** 2 - 1.0)
    return eps * r * (p2 + beta * np.sin(theta) ** 2 * np.cos(2.0 * phi))


def tapered_shape(r, theta, phi, eps=None, beta=None):
    """The same shape inside the body, tapered to zero across the buffer."""
    taper = np.clip((1.0 + BUFFER - r) / BUFFER, 0.0, 1.0) ** 2
    return shape(np.minimum(r, 1.0), theta, phi, eps, beta) * taper


def build(dim: int, buffer: bool, args) -> None:
    eps, beta = args.eps, args.beta
    sizing = UniformInterfaces(*sizes(args, *SIZING[dim]))

    def my_shape(r, theta, phi):
        return shape(r, theta, phi, eps, beta)

    def my_tapered(r, theta, phi):
        return tapered_shape(r, theta, phi, eps, beta)

    body = Geometry(Skeleton([0.0, 1.0]),
                    layer_names=["body"], interface_names=["surface"])
    suffix = args.name
    if buffer:
        geometry = body.stretched(
            CallableDisplacement(my_tapered, knots=[1.0],
                                 name="tapered shape"))
        spec = MeshSpec(geometry, sizing, dimension=dim, order=2,
                        shells=[Shell(ratio=BUFFER, name="buffer")],
                        outer_boundary="spherical",
                        meta={"eps": eps, "beta": beta, "buffer": BUFFER})
        name = f"aspherical_buffer_{dim}d{suffix}"
    else:
        geometry = body.stretched(CallableDisplacement(my_shape,
                                                       name="shape"))
        spec = MeshSpec(geometry, sizing, dimension=dim, order=2,
                        meta={"eps": eps, "beta": beta})
        name = f"aspherical_{dim}d{suffix}"

    # The reference (spherical) mesh, then the MFEM file with the nodes moved.
    build_mfem(spec, name, args, "nodes moved by the shape")


def build_fluid_core(dim: int, spherical: bool, args) -> None:
    """The two-layer body (fluid core, mantle) with its buffer: stretched
    by the shape, or with --spherical left spherical (eps = 0)."""
    eps = 0.0 if spherical else args.eps
    beta = args.beta
    rc = args.rc
    sizing = UniformInterfaces(*sizes(args, *SIZING_FLUID_CORE[dim]))

    def my_tapered(r, theta, phi):
        return tapered_shape(r, theta, phi, eps, beta)

    body = Geometry(Skeleton([0.0, rc, 1.0]),
                    layer_names=["core", "mantle"],
                    interface_names=["core_boundary", "surface"])
    if not spherical:
        body = body.stretched(
            CallableDisplacement(my_tapered, knots=[1.0],
                                 name="tapered shape"))
    spec = MeshSpec(body, sizing, dimension=dim, order=2,
                    shells=[Shell(ratio=BUFFER, name="buffer")],
                    outer_boundary="spherical",
                    meta={"eps": eps, "beta": beta, "buffer": BUFFER,
                          "r_c": rc, "fluid_layers": [1]})
    kind = "spherical" if spherical else "aspherical"
    name = f"{kind}_fluid_core_buffer_{dim}d{args.name}"

    what = "nodes as meshed (eps = 0)" if spherical else "nodes moved by the shape"
    build_mfem(spec, name, args, what)


def main() -> None:
    p = parser(__doc__)
    p.add_argument("--dim", type=int, default=2, choices=(2, 3),
                   help="2 for a disc, 3 for a ball (default %(default)s)")
    p.add_argument("--buffer", action="store_true",
                   help="BOOLEAN switch (default off): add a buffer shell "
                        "reaching to radius 1.2, with the shape tapered "
                        "off across it. Unlike layered_earth.py, "
                        "where --buffer is the shell's thickness")
    p.add_argument("--fluid-core", action="store_true",
                   help="two layers, fluid core and mantle, with the "
                        "buffer (default off)")
    p.add_argument("--spherical", action="store_true",
                   help="with --fluid-core: leave the body spherical "
                        "(eps = 0), the counterpart of the aspherical "
                        "mesh (default off)")
    p.add_argument("--rc", type=float, default=RC,
                   help="core radius with --fluid-core, in units of the body's "
                        "radius (default %(default).6f, the Earth's CMB)")
    p.add_argument("--all", action="store_true",
                   help="build all eight meshes (single-layer and fluid-core, 2-D "
                        "and 3-D), ignoring --dim, --buffer, --fluid-core "
                        "and --spherical")
    p.add_argument("--eps", type=float, default=EPS,
                   help="dimensionless amplitude eps of the shape, the "
                        "displacement being eps times the radius "
                        "(default %(default)s)")
    p.add_argument("--beta", type=float, default=BETA,
                   help="dimensionless weight beta of the elliptical part relative "
                        "to the oblate part (default %(default)s)")
    p.add_argument("--name", default="",
                   help="suffix appended to the file names, for parameter sweeps "
                        "(default: none)")
    args = p.parse_args()

    if args.all:
        for dim in (2, 3):
            for buffer in (False, True):
                build(dim, buffer, args)
            for spherical in (False, True):
                build_fluid_core(dim, spherical, args)
    elif args.fluid_core:
        build_fluid_core(args.dim, args.spherical, args)
    else:
        build(args.dim, args.buffer, args)


if __name__ == "__main__":
    main()
