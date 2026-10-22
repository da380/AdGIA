"""A love-family JSON as a V1 exchange file.

`viscoelastic_love` results and `laplace_reference.py` outputs share one
schema; this turns either into the cross-code exchange format of the V1
tier: columns t (in units of the shortest Maxwell time) and, per degree,
hp<l> lp<l> kp<l> — the conventional load numbers (`--channel tide`
writes the tidal triple instead, named hT<l> lT<l> kT<l>).  Times the
reference marks invalid (beyond its Gaver–Stehfest horizon) are NaN.
The numbers are copied, not recomputed.

    ./from_results results.json --out runs/v1_homogeneous_adgia.txt
    ./from_results ../../viscoelastic/love/references/laplace_homogeneous_tau1.json \
        --out runs/v1_homogeneous_laplace.txt
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent.parent / "common"))

import exchange  # noqa: E402
import outputs  # noqa: E402


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="convert a love-family JSON to a V1 exchange file")
    parser.add_argument("results", type=Path)
    parser.add_argument("--channel", choices=("load", "tide"),
                        default="load")
    parser.add_argument("--out", type=Path, required=True,
                        help="exchange file to write, outside the source "
                        "tree")
    args = parser.parse_args(argv)
    out = outputs.outside_source(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)

    d = json.loads(args.results.read_text())
    degrees = [int(l) for l in d["degree"]]
    taus = [lay.get("tau") for lay in d["layers"]]
    finite = [t for t in taus
              if isinstance(t, (int, float)) and math.isfinite(t)]
    if not finite:
        raise SystemExit("no finite Maxwell time in the layers")
    tau_min = min(finite)

    prefix = {"load": ("hp", "lp", "kp"), "tide": ("hT", "lT", "kT")}[
        args.channel]
    keys = [f"{c}_{args.channel}" for c in ("h", "l", "k")]
    rows = []
    for h in d["histories"]:
        row = [float(h["time"]) / tau_min]
        for i, _ in enumerate(degrees):
            for key in keys:
                value = h[key][i] if h.get("valid", True) else None
                row.append(np.nan if value is None else float(value))
        rows.append(row)

    columns = ["t"]
    for l in degrees:
        columns += [f"{p}{l}" for p in prefix]
    code = "adgia" if "scheme" in d or "cost" in d else "laplace"
    comments = [
        f"V1 Maxwell {args.channel} Love-number histories from {code} "
        f"({args.results.name})",
        exchange.provenance_line(code, HERE),
        f"model {d.get('model', '?')} taus "
        + ",".join(str(t) for t in taus)
        + f" (run units, tau_min {tau_min:g}); t in units of tau_min; "
        "dimensionless conventional numbers; NaN = invalid time",
    ]
    path = exchange.write_table(out, comments=comments, scalars={},
                                columns=columns, table=np.array(rows))
    print(path)


if __name__ == "__main__":
    main()
