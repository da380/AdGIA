"""Cost against accuracy of the CMB treatments.

For every case of a runs tree that has Dahlen runs under more than one
fluid-interface treatment (`run.py --cmb full/nomass/uniform/winkler`),
one table per model, element size and order: what each approximation
gains in cost — setup and solve seconds, iterations, potential
unknowns — against what it loses in accuracy, both to the pyslfp
reference and to the full treatment on the same mesh. The deviation
from full is the approximation error alone, mesh bias cancelled; the
error to the reference says when that deviation starts to matter, i.e.
below which mesh error the approximation is the bottleneck.

    python cmb_report.py runs
    python cmb_report.py runs --order 3 --out runs/cmb_summary.md

The comparison is of the load Love numbers: h' and l' from degree one,
k' from degree two (frame-fixed at one). Degree zero is always left out:
with a fluid layer the Dahlen-family treatments are wrong there by
design (README.md), and the report is about those treatments.

Combined runs (`run.py --cmb ... --combined`, results suffixed
`_combined`) get tables of their own, their rows marked "(combined)" and
compared with the combined full treatment: their "s / solve" and
iterations are of the one load solve for all the degrees, not a mean by
degree, and their numbers carry the leakage between degrees that the
solves by degree discard (README.md, "Combined solves").
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "common"))

from costs import load_cost  # noqa: E402
from outputs import outside_source  # noqa: E402

TREATMENTS = ("full", "nomass", "uniform", "winkler")
QUANTITIES = (("h", 1), ("l", 1), ("k", 2))  # key, first degree


def love_from(results: Path) -> dict:
    r = json.loads(results.read_text())
    values = {d["degree"]: d["load"] for d in r["degrees"]}
    cost = load_cost(r)
    return {"values": values, "setup": r["setup_seconds"],
            "unknowns": r["potential_unknowns"],
            "seconds": cost.seconds, "iterations": cost.iterations,
            "combined": cost.combined}


def love_reference(case: Path) -> dict[int, dict[str, float]]:
    r = json.loads((case / "reference.json").read_text())
    return {l: {q: r[f"{q}_load"][i] for q, _ in QUANTITIES}
            for i, l in enumerate(r["degree"])}


def worst(a: dict, b: dict, degrees, quantities=QUANTITIES) -> float:
    out = 0.0
    for l in degrees:
        for q, l0 in quantities:
            if l < max(l0, 1):
                continue
            if abs(b[l][q]) > 1e-10:
                out = max(out, abs(a[l][q] - b[l][q]) / abs(b[l][q]))
    return out


def main() -> None:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("runs", type=Path, help="a runs tree, or one model's")
    p.add_argument("--order", type=int, nargs="+", default=None,
                   help="orders to report (default: all found)")
    p.add_argument("--out", type=Path, default=None,
                   help="summary file (default: <runs>/cmb_summary.md)")
    args = p.parse_args()

    lines: list[str] = ["# CMB treatments: cost against accuracy", ""]
    for case in sorted(args.runs.glob("**/h*")):
        if not (case / "reference.json").exists():
            continue
        refs = love_reference(case)
        # The runs by degree and the combined runs (suffix _combined)
        # make separate tables: the combined numbers carry the leakage
        # between degrees, which would pollute the deviation from full.
        fulls = [(f, "") for f in sorted(case.glob("results_o*.json"))]
        fulls += [(f, "_combined") for f in
                  sorted(case.glob("results_o*_combined.json"))]
        for full, mode in fulls:
            stem = full.stem.removesuffix(mode)  # results_o<p>
            if stem.count("_") != 1:
                continue  # a method, solver or cmb variant
            order = int(stem.split("_o")[1])
            if args.order and order not in args.order:
                continue
            runs = {"full": love_from(full)}
            for t in TREATMENTS[1:]:
                f = case / f"results_o{order}_{t}{mode}.json"
                if f.exists():
                    runs[t] = love_from(f)
            if len(runs) < 2:
                continue
            degrees = sorted(set.intersection(
                *(set(r["values"]) for r in runs.values())) & set(refs))
            degrees = [l for l in degrees if l >= 1]
            if not degrees:
                continue  # a degree-zero-only run: nothing to compare
            heading = (f"## {case} (order {order}, degrees "
                       f"{degrees[0]}-{degrees[-1]})")
            if mode:
                heading = (f"## {case} (order {order}, degrees "
                           f"{degrees[0]}-{degrees[-1]}, combined solves:"
                           " s / solve and iterations are of the one load"
                           " solve for all the degrees)")
            lines += [heading, "",
                      "| treatment | h' vs ref | l' vs ref | k' vs ref "
                      "| vs full | potential unknowns | setup s | "
                      "s / solve | iterations |",
                      "|---|---|---|---|---|---|---|---|---|"]
            for t, r in runs.items():
                per_q = [worst(r["values"], refs, degrees, [(q, l0)])
                         for q, l0 in QUANTITIES]
                vs_full = worst(r["values"], runs["full"]["values"],
                                degrees) if t != "full" else 0.0
                label = f"{t} (combined)" if r["combined"] else t
                lines.append(
                    f"| {label} | " +
                    "".join(f"{e:.2e} | " for e in per_q) +
                    f"{'-' if t == 'full' else f'{vs_full:.2e}'} | "
                    f"{r['unknowns']} | {r['setup']:.1f} | "
                    f"{r['seconds']:.2f} | {r['iterations']:.0f} |")
            lines.append("")
    if len(lines) < 3:
        raise SystemExit(f"no case under {args.runs} has more than one "
                         "CMB treatment (run.py --cmb ...)")
    text = "\n".join(lines) + "\n"
    print(text)
    out = outside_source(args.out or args.runs / "cmb_summary.md")
    out.write_text(text)
    print(f"written to {out}")


if __name__ == "__main__":
    main()
