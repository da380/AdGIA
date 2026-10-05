"""The unit disc: data/disk.msh.

One layer and one boundary. Domain attribute 1 is the disc and boundary
attribute 1 its circle. Order-3 elements, refined towards the boundary.

Used by: transformed_diffusion.
"""
from planetmodel import Geometry, Skeleton
from planetmodel.mesh3d import MeshSpec, UniformInterfaces, build_layered_mesh

from common import parser, report


def main() -> None:
    args = parser(__doc__).parse_args()

    disc = Geometry(Skeleton([0.0, 1.0]),
                    layer_names=["disc"], interface_names=["boundary"])

    # Elements of size 0.045 on the boundary, growing to 0.09 over a
    # distance of 0.2 into the disc.
    sizing = UniformInterfaces(0.045, 0.09, 0.2)

    spec = MeshSpec(disc, sizing, dimension=2, order=3)
    report(build_layered_mesh(spec, args.out / "disk", verbose=args.verbose))


if __name__ == "__main__":
    main()
