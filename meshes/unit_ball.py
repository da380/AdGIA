"""The unit ball: data/ball.msh.

One layer and one boundary. Domain attribute 1 is the ball and boundary
attribute 1 its sphere. Order-2 elements, refined towards the boundary.
About fifty thousand tetrahedra; the build takes a minute or two.

Used by: transformed_diffusion, anisotropic_elasticity.
"""
from planetmodel import Geometry, Skeleton
from planetmodel.mesh3d import MeshSpec, UniformInterfaces, build_layered_mesh

from common import parser, report


def main() -> None:
    args = parser(__doc__).parse_args()

    ball = Geometry(Skeleton([0.0, 1.0]),
                    layer_names=["ball"], interface_names=["boundary"])

    # Elements of size 0.047 on the boundary, growing to 0.094 over a
    # distance of 0.2 into the ball.
    sizing = UniformInterfaces(0.047, 0.094, 0.2)

    spec = MeshSpec(ball, sizing, dimension=3, order=2)
    report(build_layered_mesh(spec, args.out / "ball", verbose=args.verbose))


if __name__ == "__main__":
    main()
