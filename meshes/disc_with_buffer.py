"""A unit disc with a buffer annulus around it: data/elastogravity_2d.msh.

The body is the unit disc. Outside it a shell, or buffer, extends to
radius 1.2; the gravitational potential is solved on the whole domain and
the far-field condition is applied on the buffer's outer circle, while
the elastic problem lives on the body alone.

Domain attribute 1 is the body and 2 the buffer. Boundary attribute 1 is
the body's surface and 2 the outer circle. Order-2 elements.

Used by: elastogravity_layered, equilibrium_stress, lateral_viscosity,
love_numbers, referential_elastogravity, the tests.
"""
from planetmodel import Geometry, Skeleton
from planetmodel.mesh3d import (MeshSpec, Shell, UniformInterfaces,
                                build_layered_mesh)

from common import parser, report, sizes


def main() -> None:
    args = parser(__doc__).parse_args()

    body = Geometry(Skeleton([0.0, 1.0]),
                    layer_names=["body"], interface_names=["surface"])

    # The buffer's outer radius as a fraction of the radius it sits on:
    # ratio 0.2 on the unit disc gives an outer radius of 1.2.
    buffer = Shell(ratio=0.2, name="buffer")

    # Elements of size 0.08 on both circles, growing to 0.16 away from them.
    sizing = UniformInterfaces(*sizes(args, 0.08, 0.16, 0.8))

    spec = MeshSpec(body, sizing, dimension=2, order=2, shells=[buffer])
    report(build_layered_mesh(spec, args.out / "elastogravity_2d",
                              verbose=args.verbose))


if __name__ == "__main__":
    main()
