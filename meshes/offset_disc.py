"""A disc offset inside a larger one: data/circular_offset.msh.

A disc of radius 1, its centre displaced along x, inside a disc of radius
1.75 centred on the origin. This is the two-body benchmark geometry: the
inner disc is a body whose exterior potential is known in closed form,
and the outer circle is where the far-field condition is applied.

Domain attribute 1 is the inner disc and 2 the region around it. Boundary
attribute 1 is the inner circle and 2 the outer one. Order-3 elements.
The consumers locate the inner circle's centre themselves.

Used by: poisson_dtn, submesh_injection.
"""
import math

from planetmodel.mesh3d import InterfaceSizing, PerInterface, build_offset_mesh

from common import parser, report


def main() -> None:
    p = parser(__doc__)
    p.add_argument("--offset", type=float, default=0.5 * math.sqrt(2.0),
                   help="displacement of the inner disc's centre along x")
    args = p.parse_args()

    # Finer elements on the inner circle than on the outer one; each
    # sizing is (size on the boundary, size far away, distance over
    # which it grows).
    sizing = PerInterface({
        "inner_circle": InterfaceSizing(0.01, 0.06, 0.3),
        "outer_circle": InterfaceSizing(0.0175, 0.105, 0.525),
    })

    result = build_offset_mesh(
        args.out / "circular_offset",
        inner_radius=1.0, outer_radius=1.75, offset=args.offset,
        sizing=sizing, dimension=2, order=3,
        layer_names=["inner_disc", "surrounding"],
        interface_names=["inner_circle", "outer_circle"],
        verbose=args.verbose)
    report(result)


if __name__ == "__main__":
    main()
