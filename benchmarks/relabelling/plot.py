"""Figures for the relabelling family: the aspherical-reference runs.

Reads the JSON files aspherical_reference writes (`aspherical*.json` in
the directory given, or the files named) and draws the relative L2
field errors against degree, one series per run, into
`aspherical.png` beside them: how far the solution on the independently
meshed aspherical body sits from the pyslfp radial solution composed
through the map. Flat in the shape amplitude and falling with
refinement is the verified picture; a non-converged degree is drawn
hollow.

The identity driver (`relabelled_identity`) prints single numbers and
has nothing to draw.

    python plot.py <build>/benchmarks/runs_campaign/relabelling
    python plot.py aspherical_homogeneous_o2.json aspherical_homogeneous_o3.json
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

COLOURS = ("#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4",
           "#008300", "#4a3aa7", "#e34948")
MARKERS = ("o", "s", "^", "D", "v", "P", "X", "*")
INK, MUTED, GRID, SURFACE = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"


def label(path: Path, r: dict) -> str:
    """`aspherical_<model>_o<p>.json` -> `model, order p, eps e`."""
    parts = path.stem.split("_")
    model = "_".join(parts[1:-1]) if len(parts) > 2 else path.stem
    return f"{model}, order {r['order']}, eps {r['eps']:g}"


def main() -> None:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("runs", type=Path, nargs="+",
                   help="a directory holding aspherical*.json, or files")
    p.add_argument("--out", type=Path, default=None,
                   help="figure file (default: aspherical.png beside "
                        "the first input)")
    args = p.parse_args()

    files: list[Path] = []
    for given in args.runs:
        files += sorted(given.glob("aspherical*.json")) \
            if given.is_dir() else [given]
    if not files:
        raise SystemExit(f"no aspherical*.json under {args.runs}")

    fig, axes = plt.subplots(1, 2, figsize=(9.6, 4.0), sharey=True,
                             facecolor=SURFACE)
    for i, path in enumerate(files):
        r = json.loads(path.read_text())
        colour, marker = COLOURS[i % 8], MARKERS[i % 8]
        for axis, key in zip(axes, ("u_error", "phi_error")):
            ls = [d["degree"] for d in r["degrees"]]
            axis.semilogy(ls, [d[key] for d in r["degrees"]],
                          color=colour, marker=marker, markersize=4.5,
                          linewidth=1.2, label=label(path, r))
            bad = [(d["degree"], d[key]) for d in r["degrees"]
                   if not d.get("converged", True)]
            if bad:
                axis.semilogy(*zip(*bad), linestyle="none", marker=marker,
                              markersize=8, markerfacecolor="none",
                              markeredgecolor=colour)
    for axis, title in zip(axes, ("displacement  $|u - u_{ref}| / "
                                  "|u_{ref}|$",
                                  "potential  $|\\phi^1 - \\phi^1_{ref}|"
                                  " / |\\phi^1_{ref}|$")):
        axis.set_title(title, loc="left", fontsize=10, color=INK)
        axis.set_xlabel("degree $l$", color=INK)
        axis.grid(True, color=GRID, linewidth=0.7)
        axis.set_facecolor(SURFACE)
        axis.tick_params(colors=MUTED)
    axes[0].legend(fontsize=8, framealpha=0.9)
    fig.suptitle("aspherical reference body: field errors through the "
                 "map", x=0.01, ha="left", fontsize=11, color=INK)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    out = args.out or (files[0].parent / "aspherical.png")
    fig.savefig(out, dpi=180)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
