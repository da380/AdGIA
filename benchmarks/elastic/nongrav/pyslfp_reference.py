"""pyslfp's E1 traction numbers, written as exchange files.

pyslfp owns its own environment, so this script runs from the pyslfp
checkout rather than through this repository's launchers:

    cd ~/dev/pyslfp && poetry run python \
        ~/dev/AdGIA/benchmarks/elastic/nongrav/pyslfp_reference.py \
        --out ~/dev/AdGIA/<build>/benchmarks/elastic/nongrav/runs

It writes `<out>/e1_<model>_pyslfp.txt` (or `..._pyslfp_grav.txt` with
--gravitating) for each named ladder model: l, h_u, l_u, h_v, l_v in SI
(m per Pa), on pyslfp's graded mesh at the given lmax.  `compare.py`
sets these against the analytic reference and the other codes.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent.parent / "common"))

import exchange  # noqa: E402
import models  # noqa: E402
import outputs  # noqa: E402
from pyslfp.love_numbers import traction_numbers  # noqa: E402

from reference import COLUMNS, SOLID_HOMOGENEOUS  # noqa: E402


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="write pyslfp's E1 traction numbers as exchange files")
    parser.add_argument("names", nargs="*", default=list(SOLID_HOMOGENEOUS),
                        help=f"ladder models (default: {SOLID_HOMOGENEOUS})")
    parser.add_argument("--lmax", type=int, default=32)
    parser.add_argument("--gravitating", action="store_true",
                        help="the full problem under the same forcing")
    parser.add_argument("--out", type=Path, required=True,
                        help="output directory, outside the source tree")
    args = parser.parse_args(argv)
    out = outputs.outside_source(args.out)
    out.mkdir(parents=True, exist_ok=True)
    import pyslfp

    suffix = "_pyslfp_grav" if args.gravitating else "_pyslfp"
    for name in args.names:
        tn = traction_numbers(models.model(name), args.lmax,
                              gravitating=args.gravitating)
        table = np.column_stack(
            [np.arange(args.lmax + 1)] + [getattr(tn, c) for c in COLUMNS[1:]])
        path = exchange.write_table(
            out / f"e1_{name}{suffix}.txt",
            comments=[
                "E1 traction numbers from pyslfp "
                f"({'gravitating' if args.gravitating else 'non-gravitating'})",
                exchange.provenance_line("pyslfp", Path(pyslfp.__file__)),
                f"model {name} lmax {args.lmax} graded mesh ngll 5; SI "
                "(m per Pa); NaN = excluded degree",
            ],
            scalars={"radius": tn.radius,
                     "gravitating": float(args.gravitating)},
            columns=COLUMNS,
            table=table,
        )
        print(path)


if __name__ == "__main__":
    main()
