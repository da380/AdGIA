"""AdGIA's love_benchmark results as an E2 exchange file.

`love_benchmark` writes its Love numbers as JSON (`results.json`); this
turns one such file into the cross-code exchange format of the E2 tier:
columns l hp lp kp hT lT kT, dimensionless, NaN where a degree or a
tidal solve is absent (degree 1 of the tide, say).  Degree 0 of a Dahlen
run on a case with a fluid layer is NaN as well, computed or not: there
the method is wrong by design (the exhibited values stay in the JSON),
and the exchange file carries what is benchmarked.  The numbers are
copied, not recomputed: love_benchmark's load numbers are the
conventional primed ones and its tidal numbers the geodetic ones already.

    ./from_results results.json --out runs/e2_two_solid_adgia.txt
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent.parent / "common"))

import exchange  # noqa: E402
import outputs  # noqa: E402

COLUMNS = ("l", "hp", "lp", "kp", "hT", "lT", "kT")


def has_fluid_layer(results_path: Path, results: dict) -> bool:
    """Whether the run's case has a fluid layer, from its manifest (the
    path the results record, else case.json beside the results)."""
    candidates = []
    if results.get("case"):
        candidates.append(Path(results["case"]))
    candidates.append(results_path.resolve().parent / "case.json")
    for case in candidates:
        if case.is_file():
            manifest = json.loads(case.read_text())
            return any(bool(layer.get("fluid"))
                       for layer in manifest.get("layers", []))
    raise SystemExit(f"no case manifest found for {results_path}")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="convert love_benchmark results.json to an E2 "
        "exchange file")
    parser.add_argument("results", type=Path)
    parser.add_argument("--out", type=Path, required=True,
                        help="exchange file to write, outside the source "
                        "tree")
    args = parser.parse_args(argv)
    out = outputs.outside_source(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)

    results = json.loads(args.results.read_text())
    entries = {int(d["degree"]): d for d in results["degrees"]}
    dahlen_fluid = (results.get("method") == "dahlen"
                    and has_fluid_layer(args.results, results))
    if dahlen_fluid:
        entries.pop(0, None)
    lmax = max(entries)
    table = np.full((lmax + 1, len(COLUMNS)), np.nan)
    table[:, 0] = np.arange(lmax + 1)
    for l, d in entries.items():
        load = d.get("load")
        if load is not None:
            table[l, 1:4] = [load["h"], load["l"], load["k"]]
        tide = d.get("tide")
        if tide is not None:
            table[l, 4:7] = [tide["h"], tide["l"], tide["k"]]

    comments = [
        "E2 conventional Love numbers from AdGIA (love_benchmark)",
        exchange.provenance_line("adgia", HERE),
        f"results {args.results}; method {results.get('method', '?')}, "
        f"order {results.get('order', '?')}; dimensionless (load primed, "
        "geodetic tidal; NaN = absent solve"
        + (", degree 0 of a Dahlen fluid run left out by design)"
           if dahlen_fluid else ")"),
    ]
    scalars = {"gravitating": 1.0}
    for key in ("radius", "surface_gravity"):
        if key in results:
            scalars[key] = float(results[key])
    path = exchange.write_table(out, comments=comments, scalars=scalars,
                                columns=COLUMNS, table=table)
    print(path)


if __name__ == "__main__":
    main()
