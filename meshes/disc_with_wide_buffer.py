"""A unit disc with a WIDE buffer annulus: data/elastogravity_2d_wide.msh.

As disc_with_buffer.py, but the buffer extends to radius 2.0 instead of
1.2. The wide buffer exists for equilibrium mappings that are strongly
non-trivial on the body's surface (large-ellipticity studies): the taper
of the mapping to the identity at the DtN circle needs room to stay
diffeomorphic, and the 0.2-wide standard buffer runs out near
ellipticity 0.1.

Domain attribute 1 is the body and 2 the buffer. Boundary attribute 1 is
the body's surface and 2 the outer circle. Order-2 elements.

Used by: prestress_loading.
"""
from planetmodel import Geometry, Skeleton
from planetmodel.mesh3d import (MeshSpec, Shell, UniformInterfaces,
                                build_layered_mesh)

from common import parser, report


def main() -> None:
    args = parser(__doc__).parse_args()

    body = Geometry(Skeleton([0.0, 1.0]),
                    layer_names=["body"], interface_names=["surface"])

    buffer = Shell(ratio=1.0, name="buffer")

    # Elements of size 0.08 on the circles, growing to 0.2 in the (large)
    # buffer interior.
    sizing = UniformInterfaces(0.08, 0.2, 0.8)

    spec = MeshSpec(body, sizing, dimension=2, order=2, shells=[buffer])
    report(build_layered_mesh(spec, args.out / "elastogravity_2d_wide",
                              verbose=args.verbose))


if __name__ == "__main__":
    main()
