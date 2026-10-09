"""Figures for the perturbation family: the interface-shift check.

Reads what perturbation_check leaves behind — the shifted references
`shift_<+-eps>/reference.json` and the runs
`results_o<p>_<method>_shift<+-eps>.json`, with the base run beside the
case — and draws one figure per method into the results directory,
`perturbation_<method>_o<p>.png`:

  left    the absolute agreement with pyslfp by degree, one line per
          eps: flat across the ladder means the mapped solve tracks the
          perturbed models as well as the unmapped solve tracks the
          base one;
  middle  the derivative check as an identity plot, at the widest eps
          of the ladder — the fixed-mesh central difference of the 3-D
          runs against the same difference of the 1-D references, one
          point per degree and Love number, agreement meaning the point
          sits on the line;
  right   the relative discrepancy of each of those derivatives, which
          the identity plot cannot resolve for the small ones.

    python plot.py <case> --out <root> --method referential slip_broken
    python plot.py runs/fluid_core/h0.3     # results beside the case

The comparison rules are perturbation_check's: h' everywhere, l' and k'
from degree two, degree one skipped, near-zero references and
derivatives left out.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "common"))
sys.path.insert(0, str(HERE))

from outputs import outside_source  # noqa: E402
from perturbation_check import love_from, love_reference  # noqa: E402

COLOURS = ("#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4",
           "#008300", "#4a3aa7", "#e34948")
MARKERS = ("o", "s", "^", "D", "v", "P", "X", "*")
INK, MUTED, GRID, SURFACE = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"


def compared(l: int) -> list[str]:
    return [q for q in "hlk" if q == "h" or l >= 2]


def gather(case: Path, root: Path, method: str, order: int):
    """eps -> (run values, reference values), from the files present."""
    out = {}
    for f in sorted(root.glob(f"results_o{order}_{method}_shift*.json")):
        eps = float(f.stem.split("shift")[1])
        out[eps] = (love_from(f),
                    love_reference(root / f"shift_{eps:+g}" /
                                   "reference.json"))
    base = case / f"results_o{order}_{method}.json"
    if not base.exists():
        base = root / f"results_o{order}_{method}.json"
    if base.exists() and out:
        out[0.0] = (love_from(base), love_reference(case /
                                                    "reference.json"))
    return out


def main() -> None:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("case", type=Path,
                   help="the case directory, runs/<model>/h<h>")
    p.add_argument("--out", type=Path, default=None,
                   help="where perturbation_check wrote (default: the "
                        "case directory)")
    p.add_argument("--method", nargs="+",
                   default=["referential", "slip_broken"])
    p.add_argument("--order", type=int, default=2)
    args = p.parse_args()
    root = outside_source(args.out if args.out is not None else args.case)

    drawn = 0
    for method in args.method:
        runs = gather(args.case, root, method, args.order)
        if len(runs) < 2:
            continue
        fig, axes = plt.subplots(1, 3, figsize=(14, 4.2),
                                 facecolor=SURFACE,
                                 gridspec_kw={"width_ratios": [1, 1, 1.1]})

        # Absolute agreement by degree, one line per eps.
        for i, eps in enumerate(sorted(runs)):
            values, refs = runs[eps]
            ls, errs = [], []
            for l in sorted(set(values) & set(refs)):
                if l == 1:
                    continue
                worst = 0.0
                for q in compared(l):
                    if abs(refs[l][q]) > 1e-10:
                        worst = max(worst, abs(values[l][q] - refs[l][q])
                                    / abs(refs[l][q]))
                ls.append(l)
                errs.append(worst)
            axes[0].semilogy(ls, errs, color=COLOURS[i % 8],
                             marker=MARKERS[i % 8], markersize=4.5,
                             linewidth=1.2,
                             label="eps = 0 (unmapped)" if eps == 0.0
                             else f"eps = {eps:+g}")
        axes[0].xaxis.set_major_locator(plt.MaxNLocator(integer=True))

        # The derivative identity: 3-D central difference against 1-D,
        # the Love number by marker and the degree by colour; and the
        # relative discrepancy of each, which the identity plot cannot
        # resolve for the small derivatives.
        pairs = sorted({abs(e) for e in runs if e != 0.0
                        and -abs(e) in runs and abs(e) in runs})
        shape = {"h": "o", "l": "s", "k": "^"}
        lo = hi = None
        bars = []
        for a in pairs[-1:]:  # the widest pair: the identity plot
            plus, minus = runs[a], runs[-a]
            for l in sorted(set(plus[0]) & set(minus[0])):
                if l == 1:
                    continue
                for q in compared(l):
                    d3 = (plus[0][l][q] - minus[0][l][q]) / (2 * a)
                    d1 = (plus[1][l][q] - minus[1][l][q]) / (2 * a)
                    if abs(d1) < 1e-10:
                        continue
                    axes[1].plot(d1, d3, linestyle="none",
                                 marker=shape[q], markersize=7,
                                 color=COLOURS[l % 8],
                                 markeredgecolor=SURFACE)
                    bars.append((f"${q}'_{l}$", abs(d3 - d1) / abs(d1),
                                 COLOURS[l % 8]))
                    lo = d1 if lo is None else min(lo, d1, d3)
                    hi = d1 if hi is None else max(hi, d1, d3)
        if lo is not None:
            pad = 0.08 * (hi - lo or 1.0)
            axes[1].plot([lo - pad, hi + pad], [lo - pad, hi + pad],
                         color=MUTED, linewidth=0.8, zorder=0)
            handles = [plt.Line2D([], [], ls="none", marker=m,
                                  color=MUTED, label=f"${q}'$")
                       for q, m in shape.items()]
            degrees = sorted({int(b[0].split("_")[1].strip("$"))
                              for b in bars})
            handles += [plt.Line2D([], [], ls="none", marker="o",
                                   color=COLOURS[l % 8],
                                   label=f"degree {l}") for l in degrees]
            axes[1].legend(handles=handles, fontsize=7, framealpha=0.9)
        if bars:
            y = range(len(bars))[::-1]
            axes[2].barh(y, [b[1] for b in bars],
                         color=[b[2] for b in bars], height=0.6)
            axes[2].set_yticks(list(y), [b[0] for b in bars])
            axes[2].set_xscale("log")
            # bars on a log axis need an explicit floor to start from
            axes[2].set_xlim(left=min(b[1] for b in bars) / 4)
            axes[2].grid(axis="y", visible=False)

        axes[0].set_title("agreement with pyslfp by degree (worst of "
                          "h', l', k')", loc="left", fontsize=10, color=INK)
        axes[0].set_xlabel("degree $l$", color=INK)
        axes[1].set_title(f"d(Love)/d$r_k$, eps = {pairs[-1]:g}: "
                          "3-D against 1-D" if pairs else "",
                          loc="left", fontsize=10, color=INK)
        axes[1].set_xlabel("1-D (pyslfp) central difference", color=INK)
        axes[1].set_ylabel("3-D central difference, fixed mesh", color=INK)
        axes[2].set_title("relative discrepancy of the derivatives",
                          loc="left", fontsize=10, color=INK)
        axes[2].set_xlabel("|3-D - 1-D| / |1-D|", color=INK)
        for axis in axes:
            axis.grid(True, color=GRID, linewidth=0.7)
            axis.set_facecolor(SURFACE)
            axis.tick_params(colors=MUTED)
        axes[0].legend(fontsize=8, framealpha=0.9)
        axes[2].grid(axis="y", visible=False)
        fig.suptitle(f"interface shift, {method} (order {args.order}, "
                     f"{args.case.parent.name})", x=0.01, ha="left",
                     fontsize=11, color=INK)
        fig.tight_layout(rect=(0, 0, 1, 0.94))
        out = root / f"perturbation_{method}_o{args.order}.png"
        fig.savefig(out, dpi=180)
        print(f"wrote {out}")
        drawn += 1
    if not drawn:
        raise SystemExit(f"no shifted results for {args.method} under "
                         f"{root} (run perturbation_check first)")


if __name__ == "__main__":
    main()
