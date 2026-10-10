"""PREM-like layered Earth models with a buffer shell: the
elastogravity_*_layer meshes.

Radii are in units of the Earth's radius, 6371 km: the inner-core
boundary at 1230 km, the core-mantle boundary at 3483 km and the surface
at 6371 km. Two layerings are built:

  --layers 2   fluid core, mantle
  --layers 3   solid inner core, fluid outer core, mantle

A buffer shell of relative thickness 0.2 sits outside the surface. Domain
attributes count the layers from the centre, with the buffer last;
boundary attributes count the interfaces from the centre (ICB, CMB,
surface, outer). The examples recognise the layering from the number of
domain attributes. Order-2 elements.

Files: elastogravity_two_layer_2d.msh, elastogravity_three_layer_2d.msh,
elastogravity_two_layer_3d.msh, elastogravity_three_layer_3d.msh; `--all`
builds the four of them.

Used by: elastogravity_layered, gauged_fluid_cavity,
self_gravitating_relaxation, self_gravitating_solvers,
sliding_fluid_ellipse, the tests.
"""
from planetmodel import Geometry, Skeleton
from planetmodel.mesh3d import (MeshSpec, Shell, UniformInterfaces,
                                build_layered_mesh)

from common import parser, report, sizes

EARTH_RADIUS_KM = 6371.0
ICB = 1230.0 / EARTH_RADIUS_KM
CMB = 3483.0 / EARTH_RADIUS_KM
SURFACE = 1.0

# Element size on every interface, far from them, and the distance over
# which it grows, by dimension.
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


def build(layers: int, dim: int, args) -> None:
    words = {2: "two", 3: "three"}
    name = f"elastogravity_{words[layers]}_layer_{dim}d"
    sizing = UniformInterfaces(*sizes(args, *SIZING[dim]))
    spec = MeshSpec(geometry(layers), sizing, dimension=dim, order=2,
                    shells=[Shell(ratio=0.2, name="buffer")])
    report(build_layered_mesh(spec, args.out / name, verbose=args.verbose))


def main() -> None:
    p = parser(__doc__)
    p.add_argument("--layers", type=int, default=2, choices=(2, 3),
                   help="2 for fluid core + mantle, 3 with a solid inner core")
    p.add_argument("--dim", type=int, default=2, choices=(2, 3),
                   help="2 for a disc, 3 for a ball")
    p.add_argument("--all", action="store_true",
                   help="build the three meshes the examples use")
    args = p.parse_args()

    if args.all:
        for layers, dim in [(2, 2), (3, 2), (2, 3), (3, 3)]:
            build(layers, dim, args)
    else:
        build(args.layers, args.dim, args)


if __name__ == "__main__":
    main()
