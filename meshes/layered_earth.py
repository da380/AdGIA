"""PREM-like layered Earth models with a buffer shell: the
earth_{fluidcore,innercore}_{2,3}d meshes.

Radii are in units of the Earth's radius, 6371 km: the inner-core
boundary at 1230 km (0.1931), the core-mantle boundary at 3483 km
(0.5467) and the surface at 6371 km (1); the buffer shell reaches to
1.2. Two layerings are built, as a disc (2-D) or a ball (3-D):

  --layers 2   fluid core, mantle
  --layers 3   solid inner core, fluid outer core, mantle

A buffer shell of relative thickness 0.2 sits outside the surface. Domain
attributes count the layers from the centre, with the buffer last;
boundary attributes count the interfaces from the centre (ICB, CMB,
surface, outer). The examples recognise the layering from the number of
domain attributes. Order-2 elements.

Element sizes (common.SIZING, in surface radii): 0.085 on the interfaces
growing to 0.17 over 0.85 in 2-D, 0.135 to 0.27 over 1.35 in 3-D; --scale
multiplies them.

Files: earth_fluidcore_2d.msh, earth_innercore_2d.msh,
earth_fluidcore_3d.msh, earth_innercore_3d.msh, each
with a .json manifest; `--all` builds all four.

Used by: elastogravity_layered, gauged_fluid_cavity,
self_gravitating_relaxation, self_gravitating_solvers,
sliding_fluid_ellipse, the tests.
"""
from planetmodel.mesh3d import (MeshSpec, Shell, UniformInterfaces,
                                build_layered_mesh)

from common import SIZING, geometry, parser, report, sizes


def build(layers: int, dim: int, args) -> None:
    words = {2: "fluidcore", 3: "innercore"}
    name = f"earth_{words[layers]}_{dim}d"
    sizing = UniformInterfaces(*sizes(args, *SIZING[dim]))
    spec = MeshSpec(geometry(layers), sizing, dimension=dim, order=2,
                    shells=[Shell(ratio=0.2, name="buffer")])
    report(build_layered_mesh(spec, args.out / name, verbose=args.verbose))


def main() -> None:
    p = parser(__doc__)
    p.add_argument("--layers", type=int, default=2, choices=(2, 3),
                   help="number of layers: 2 for fluid core + mantle, 3 "
                        "with a solid inner core (default %(default)s; "
                        "ignored with --all)")
    p.add_argument("--dim", type=int, default=2, choices=(2, 3),
                   help="2 for a disc, 3 for a ball (default %(default)s; "
                        "ignored with --all)")
    p.add_argument("--all", action="store_true",
                   help="build all four meshes (2 and 3 layers, 2-D and 3-D) and "
                        "ignore --layers and --dim")
    args = p.parse_args()

    if args.all:
        for layers, dim in [(2, 2), (3, 2), (2, 3), (3, 3)]:
            build(layers, dim, args)
    else:
        build(args.layers, args.dim, args)


if __name__ == "__main__":
    main()
