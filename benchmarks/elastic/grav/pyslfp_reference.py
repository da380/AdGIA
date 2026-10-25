"""pyslfp's E2 conventional Love numbers, written as exchange files.

The gravitating elastic tier: for each named ladder model (solid or
fluid-solid), the conventional load numbers h' l' k' and the geodetic
tidal numbers h l k (NaN below degree 2), dimensionless, on pyslfp's
graded mesh.  pyslfp is this tier's reference: the most thoroughly
tested of the codes here, itself checked against an independent
Love-number code.  Runs from the pyslfp checkout, which owns its
environment:

    cd ~/dev/pyslfp && poetry run python \
        ~/dev/AdGIA/benchmarks/elastic/grav/pyslfp_reference.py \
        --out <build>/benchmarks/elastic/grav/runs
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
from pyslfp.love_numbers import love_numbers  # noqa: E402

COLUMNS = ("l", "hp", "lp", "kp", "hT", "lT", "kT")

#: The ladder models of the tier by default: solid and fluid-solid.
DEFAULT_MODELS = ("homogeneous", "two_solid", "fluid_core")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="write pyslfp's E2 Love numbers as exchange files")
    parser.add_argument("names", nargs="*", default=list(DEFAULT_MODELS),
                        help=f"ladder models (default: {DEFAULT_MODELS})")
    parser.add_argument("--lmax", type=int, default=32)
    parser.add_argument("--out", type=Path, required=True,
                        help="output directory, outside the source tree")
    args = parser.parse_args(argv)
    out = outputs.outside_source(args.out)
    out.mkdir(parents=True, exist_ok=True)
    import pyslfp

    degrees = np.arange(args.lmax + 1)
    for name in args.names:
        ln = love_numbers(models.model(name), args.lmax)
        c, t = ln.conventional(), ln.tidal()
        tidal = {k: np.where(degrees >= 2, t[k], np.nan) for k in "hlk"}
        table = np.column_stack([degrees, c["h"], c["l"], c["k"],
                                 tidal["h"], tidal["l"], tidal["k"]])
        path = exchange.write_table(
            out / f"e2_{name}_pyslfp.txt",
            comments=[
                "E2 conventional Love numbers from pyslfp (gravitating)",
                exchange.provenance_line("pyslfp", Path(pyslfp.__file__)),
                f"model {name} lmax {args.lmax} graded mesh ngll 5; "
                "dimensionless (load primed, geodetic tidal; tidal NaN "
                "below degree 2)",
            ],
            scalars={"radius": ln.radius,
                     "surface_gravity": ln.surface_gravity,
                     "gravitating": 1.0},
            columns=COLUMNS,
            table=table,
        )
        print(path)


if __name__ == "__main__":
    main()
