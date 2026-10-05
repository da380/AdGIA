"""A unit ball with a buffer shell around it: data/coupled_poisson.msh.

The body is the unit ball and the buffer extends to radius 2. The mesh is
deliberately coarse so that it builds quickly and the runs on it are
cheap; the examples that use it say so where it matters.

Domain attribute 1 is the body and 2 the buffer. Boundary attribute 1 is
the body's surface and 2 the outer sphere. Order-2 elements.

Used by: coupled_poisson, elastogravity_layered, love_numbers,
referential_elastogravity, viscoelastic_love_numbers, the tests.
"""
from planetmodel import Geometry, Skeleton
from planetmodel.mesh3d import (MeshSpec, Shell, UniformInterfaces,
                                build_layered_mesh)

from common import parser, report


def main() -> None:
    args = parser(__doc__).parse_args()

    body = Geometry(Skeleton([0.0, 1.0]),
                    layer_names=["body"], interface_names=["surface"])

    # A shell can also be given its outer radius directly.
    buffer = Shell(radius=2.0, name="buffer")

    # Coarse: elements of size 0.45 on both spheres, growing to 0.9.
    sizing = UniformInterfaces(0.45, 0.9, 4.5)

    spec = MeshSpec(body, sizing, dimension=3, order=2, shells=[buffer])
    report(build_layered_mesh(spec, args.out / "coupled_poisson",
                              verbose=args.verbose))


if __name__ == "__main__":
    main()
