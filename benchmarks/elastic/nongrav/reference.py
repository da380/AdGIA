"""The analytic E1 reference, written as exchange files.

For each named ladder model (the solid, piecewise-homogeneous ones), the
exact non-gravitating traction numbers of `common/analytic.py` go to
`<out>/e1_<model>_analytic.txt` in the cross-code exchange format: one
table of l, h_u, l_u, h_v, l_v in SI (m per Pa), NaN at the excluded
degree 1.  Every code's E1 answer is compared against these files.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent.parent / "common"))

import analytic  # noqa: E402
import exchange  # noqa: E402
import models  # noqa: E402
import outputs  # noqa: E402

#: The ladder models the analytic reference covers.
SOLID_HOMOGENEOUS = ("homogeneous", "homogeneous_lithosphere", "two_solid")

COLUMNS = ("l", "h_u", "l_u", "h_v", "l_v")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="write the analytic E1 reference as exchange files")
    parser.add_argument("names", nargs="*", default=list(SOLID_HOMOGENEOUS),
                        help=f"ladder models (default: {SOLID_HOMOGENEOUS})")
    parser.add_argument("--lmax", type=int, default=32)
    parser.add_argument("--out", type=Path, required=True,
                        help="output directory, outside the source tree")
    args = parser.parse_args(argv)
    out = outputs.outside_source(args.out)
    out.mkdir(parents=True, exist_ok=True)
    for name in args.names:
        numbers = analytic.traction_numbers(models.model(name), args.lmax)
        table = np.column_stack(
            [np.arange(args.lmax + 1)] + [numbers[c] for c in COLUMNS[1:]])
        path = exchange.write_table(
            out / f"e1_{name}_analytic.txt",
            comments=[
                "E1 non-gravitating traction numbers: the exact "
                "piecewise-homogeneous reference (common/analytic.py)",
                exchange.provenance_line("analytic", HERE),
                f"model {name} SI (m per Pa); NaN = excluded degree",
            ],
            scalars={"radius": numbers["radius"][0], "gravitating": 0.0},
            columns=COLUMNS,
            table=table,
        )
        print(path)


if __name__ == "__main__":
    main()
