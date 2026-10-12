"""Homogeneous unit bodies, bare or inside a buffer shell: the disc and
ball meshes of the examples and tests.

The body is the unit disc (--dim 2) or unit ball (--dim 3), one layer
and one interface (manifest names: layer "disc"/"ball" and interface
"boundary" when bare, "body" and "surface" with a buffer); lengths and element
sizes are in units of the body's radius. Without a buffer, domain
attribute 1 is the body and boundary attribute 1 its surface. With
--buffer-radius R > 1 a buffer shell extends to radius R; the
gravitational potential is solved on the whole domain and the far-field
condition is applied on the buffer's outer boundary, while the elastic
problem lives on the body alone. Domain attribute 1 is then the body and
2 the buffer; boundary attribute 1 is the body's surface and 2 the outer
circle or sphere.

Each product is a --preset (explicit options override its values), built
with the sizing rule "size H on every interface, growing to FAR over a
distance GROW":

  preset              file                       dim  order  buffer     (H, FAR, GROW)
  disc                disc_2d.msh                2    3      none       0.045, 0.09, 0.2
  ball                ball_3d.msh                3    2      none       0.047, 0.094, 0.2
  disc-buffer         disc_buffer_2d.msh         2    2      to 1.2     0.08, 0.16, 0.8
  disc-buffer-wide    disc_buffer_wide_2d.msh    2    2      to 2.0     0.08, 0.2, 0.8
  ball-buffer-coarse  ball_buffer_coarse_3d.msh  3    2      to 2.0     0.45, 0.9, 4.5

disc: refined towards the boundary. Used by: transformed_diffusion.

ball: refined towards the boundary; about fifty thousand tetrahedra, the
build takes a minute or two. Used by: transformed_diffusion,
anisotropic_elasticity.

disc-buffer: buffer of relative thickness 0.2. Used by:
elastogravity_layered, equilibrium_stress, lateral_viscosity,
love_numbers, referential_elastogravity, the tests.

disc-buffer-wide: the buffer reaches to radius 2.0 instead of 1.2. It
exists for equilibrium mappings that are strongly non-trivial on the
body's surface (large-ellipticity studies): the taper of the mapping to
the identity at the DtN circle needs room to stay diffeomorphic, and the
0.2-wide standard buffer runs out near ellipticity 0.1; the far size is
0.2 in the large buffer interior. Used by: prestress_loading.

ball-buffer-coarse: deliberately coarse so that it builds quickly and the
runs on it are cheap; the examples that use it say so where it matters.
Used by: coupled_poisson, elastogravity_layered, love_numbers,
referential_elastogravity, viscoelastic_love_numbers, the tests.

--scale multiplies H, FAR and GROW.
"""
from planetmodel import Geometry, Skeleton
from planetmodel.mesh3d import (MeshSpec, Shell, UniformInterfaces,
                                build_layered_mesh)

from common import parser, report, sizes

#: Per preset: manifest names of the layer and the interface, output name,
#: dimension, element order, buffer outer radius (0 for none) and the sizing
#: (size on the interfaces, far size, growth distance), in units of the
#: body's radius.
PRESETS = {
    "disc": dict(layer="disc", interface="boundary", name="disc_2d",
                 dim=2, order=3, buffer_radius=0.0,
                 h=0.045, far=0.09, grow=0.2),
    "ball": dict(layer="ball", interface="boundary", name="ball_3d",
                 dim=3, order=2, buffer_radius=0.0,
                 h=0.047, far=0.094, grow=0.2),
    "disc-buffer": dict(layer="body", interface="surface",
                        name="disc_buffer_2d", dim=2, order=2,
                        buffer_radius=1.2, h=0.08, far=0.16, grow=0.8),
    "disc-buffer-wide": dict(layer="body", interface="surface",
                             name="disc_buffer_wide_2d", dim=2, order=2,
                             buffer_radius=2.0, h=0.08, far=0.2, grow=0.8),
    "ball-buffer-coarse": dict(layer="body", interface="surface",
                               name="ball_buffer_coarse_3d", dim=3, order=2,
                               buffer_radius=2.0, h=0.45, far=0.9, grow=4.5),
}


def main() -> None:
    p = parser(__doc__)
    p.add_argument("--preset", choices=sorted(PRESETS), required=True,
                   help="the product to build: sets the file name, "
                        "dimension, order, buffer and sizing below; any of "
                        "those given explicitly overrides it")
    p.add_argument("--dim", type=int, choices=(2, 3), default=None,
                   help="2 for a disc, 3 for a ball (default: the preset's)")
    p.add_argument("--order", type=int, default=None,
                   help="polynomial order of the elements (default: the "
                        "preset's)")
    p.add_argument("--buffer-radius", type=float, default=None,
                   help="OUTER RADIUS of the buffer shell, in body radii; "
                        "0 for no buffer (default: the preset's)")
    p.add_argument("--h", type=float, default=None,
                   help="element size on the interfaces, in body radii "
                        "(default: the preset's; multiplied by --scale)")
    p.add_argument("--far-size", type=float, default=None,
                   help="element size far from the interfaces, in body "
                        "radii (default: the preset's; multiplied by "
                        "--scale)")
    p.add_argument("--growth-distance", type=float, default=None,
                   help="distance over which the size grows from --h to "
                        "--far-size, in body radii (default: the preset's; "
                        "multiplied by --scale)")
    p.add_argument("--name", default=None,
                   help="output file name without extension (default: the "
                        "preset's, e.g. disc_2d)")
    args = p.parse_args()

    preset = PRESETS[args.preset]

    def pick(value, key):
        return preset[key] if value is None else value

    dim = pick(args.dim, "dim")
    order = pick(args.order, "order")
    buffer_radius = pick(args.buffer_radius, "buffer_radius")
    name = pick(args.name, "name")
    h = pick(args.h, "h")
    far = pick(args.far_size, "far")
    grow = pick(args.growth_distance, "grow")

    body = Geometry(Skeleton([0.0, 1.0]), layer_names=[preset["layer"]],
                    interface_names=[preset["interface"]])
    shells = [Shell(radius=buffer_radius, name="buffer")] \
        if buffer_radius > 0.0 else []
    sizing = UniformInterfaces(*sizes(args, h, far, grow))

    spec = MeshSpec(body, sizing, dimension=dim, order=order, shells=shells)
    report(build_layered_mesh(spec, args.out / name, verbose=args.verbose))


if __name__ == "__main__":
    main()
