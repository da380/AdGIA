"""Finite-element Love-number histories against the Laplace reference.

Reads one or more results files of `viscoelastic_love` and the
reference of `laplace_reference.py` for the same model and taus, and
writes, per results file,

  <stem>_errors.md     the history-error table, per degree and number
  <stem>_errors.json   the same, for the campaign and other scripts
  <stem>_history.png   overlay: the reference histories and the FE ones
  <stem>_error.png     the error against time
  <stem>_series.png    the histories on a linear time axis, when the run
                       has 20 or more output times

into --out (default: beside the results file); --no-plots skips the
figures.

The metric is the stepper survey's HISTORY metric carried over: the
worst error over every output time (not the final one alone), relative
to the largest value of the reference history,

  history  = max_t |FE(t) - REF(t)| / max_t |REF(t)|,

taken over the output times the reference marks valid (inside its
instability horizon) together with t = 0+, so a step that lands well at
one time and badly at the others cannot hide. Two companions separate
the error sources:

  elastic  = |FE(0+) - REF(0)| / |REF(0)|     the spatial error alone;
  relax    = max_t |dFE(t) - dREF(t)| / max_t |dREF(t)|,  d = f(t) - f(0):
             the error of the relaxation itself, where the spatial
             error of the elastic response largely cancels — what the
             time stepping is answerable for.

    ./compare results.json --reference laplace_fluid_core_tauf_1.json
    ./compare a.json b.json --reference ref.json --out figures/
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

#: The numbers compared, and how they are labelled.
LABELS = {"h_load": "h'", "l_load": "l'", "k_load": "k'",
          "h_tide": "h", "l_tide": "l", "k_tide": "k"}

INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e4e3df"
#: Categorical order (validated: light surface; markers carry identity
#: as well as colour, the table is the text view).
SERIES = ["#2a78d6", "#eda100", "#1baf7a", "#e34948"]
MARKERS = ["o", "s", "^", "D"]

plt.rcParams.update({
    "font.size": 12, "axes.titlesize": 12, "axes.labelsize": 12,
    "xtick.labelsize": 10, "ytick.labelsize": 10, "legend.fontsize": 10,
    "axes.edgecolor": MUTED, "axes.labelcolor": INK,
    "xtick.color": MUTED, "ytick.color": MUTED, "text.color": INK,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8,
    "lines.linewidth": 2.0, "lines.markersize": 7,
    "figure.facecolor": "white", "savefig.dpi": 160,
})


def match(fe_times: list[float], ref: dict) -> list[tuple[int, int]]:
    """Pairs (FE index, reference index) of equal times (relative 1e-4:
    the reference rounds its default times to six decimals)."""
    pairs = []
    for i, t in enumerate(fe_times):
        for j, h in enumerate(ref["histories"]):
            if abs(h["time"] - t) <= 1e-4 * max(t, 1e-300):
                pairs.append((i, j))
                break
    return pairs


def errors(fe: dict, ref: dict) -> dict:
    """Per number and degree: the history, elastic and relaxation
    errors, over the valid matched times."""
    pairs = [(i, j) for i, j in match([h["time"] for h in fe["histories"]],
                                      ref)
             if ref["histories"][j].get("valid", True)]
    out = {"times": [fe["histories"][i]["time"] for i, _ in pairs],
           "numbers": {}}
    for q in LABELS:
        if q not in fe["elastic"] or q not in ref["elastic"]:
            continue
        rows = {}
        for d, l in enumerate(fe["degree"]):
            if l not in ref["degree"]:
                continue
            r = ref["degree"].index(l)
            f0, r0 = fe["elastic"][q][d], ref["elastic"][q][r]
            if f0 is None or r0 is None:
                continue
            ft = [fe["histories"][i][q][d] for i, _ in pairs]
            rt = [ref["histories"][j][q][r] for _, j in pairs]
            if any(v is None for v in ft + rt):
                continue
            scale = max(abs(v) for v in rt + [r0])
            hist = max(abs(a - b) for a, b in zip(ft + [f0], rt + [r0]))
            drift = max((abs(b - r0) for b in rt), default=0.0)
            relax = max((abs((a - f0) - (b - r0)) for a, b in zip(ft, rt)),
                        default=0.0)
            rows[l] = {
                "history": hist / scale,
                "elastic": abs(f0 - r0) / max(abs(r0), 1e-300),
                "relax": relax / drift if drift > 0.0 else None,
                "pointwise": [abs(a - b) / scale for a, b in zip(ft, rt)],
            }
        if rows:
            out["numbers"][q] = rows
    return out


def table(name: str, fe: dict, e: dict) -> str:
    cost = fe.get("cost", {}).get("load", {})
    lines = [f"## {name}", "",
             f"scheme {fe.get('scheme')}, "
             f"{len(e['times'])} valid output times to "
             f"{max(e['times'], default=0):g}; cost: "
             f"{cost.get('steps')} steps, {cost.get('stepping_solves')} "
             f"stepping solves (+{cost.get('observation_solves')} "
             f"observation), {cost.get('assemblies')} assemblies, "
             f"{cost.get('iterations')} iterations, "
             f"{cost.get('seconds', 0):.1f} s", "",
             "| number | degree | history | elastic (t=0+) | relaxation |",
             "|---|---|---|---|---|"]
    for q, rows in e["numbers"].items():
        for l, r in rows.items():
            rel = "-" if r["relax"] is None else f"{r['relax']:.2e}"
            lines.append(f"| {LABELS[q]} | {l} | {r['history']:.2e} | "
                         f"{r['elastic']:.2e} | {rel} |")
    return "\n".join(lines) + "\n"


def plot_history(fe: dict, ref: dict, out: Path, title: str) -> None:
    numbers = [q for q in ("h_load", "k_load", "l_load")
               if q in fe["elastic"]]
    degrees = [l for l in fe["degree"] if l in ref["degree"]]
    fig, axes = plt.subplots(len(numbers), len(degrees), squeeze=False,
                             figsize=(3.4 * len(degrees) + 1.0,
                                      2.8 * len(numbers) + 0.8),
                             sharex=True)
    rt = [h["time"] for h in ref["histories"]]
    ft = [h["time"] for h in fe["histories"]]
    horizon = ref.get("horizon")
    for a, q in enumerate(numbers):
        for b, l in enumerate(degrees):
            ax = axes[a][b]
            r = ref["degree"].index(l)
            d = fe["degree"].index(l)
            ax.semilogx(rt, [h[q][r] for h in ref["histories"]],
                        color=INK, lw=1.6, label="reference (Laplace)")
            ax.semilogx(ft, [h[q][d] for h in fe["histories"]],
                        ls="none", marker=MARKERS[0], color=SERIES[0],
                        mfc="white", mew=1.8, label="finite elements")
            if horizon and horizon < max(rt + ft):
                ax.axvline(horizon, color=MUTED, lw=1.0, ls=":")
            ax.set_title(f"{LABELS[q]}, l = {l}", loc="left")
            if b == 0:
                ax.set_ylabel(LABELS[q])
            if a == len(numbers) - 1:
                ax.set_xlabel("time after the load")
    axes[0][0].legend(framealpha=0.95)
    fig.suptitle(title, x=0.01, ha="left")
    fig.tight_layout()
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out}")


def plot_error(e: dict, out: Path, title: str) -> None:
    numbers = list(e["numbers"])
    fig, axes = plt.subplots(1, len(numbers), squeeze=False,
                             figsize=(4.2 * len(numbers), 3.6), sharey=True)
    for a, q in enumerate(numbers):
        ax = axes[0][a]
        for k, (l, r) in enumerate(e["numbers"][q].items()):
            ax.loglog(e["times"], [max(v, 1e-16) for v in r["pointwise"]],
                      marker=MARKERS[k % 4], color=SERIES[k % 4],
                      label=f"l = {l}")
        ax.set_title(f"{LABELS[q]}: |FE - ref| / max|ref|", loc="left")
        ax.set_xlabel("time after the load")
    axes[0][0].legend(framealpha=0.95)
    fig.suptitle(title, x=0.01, ha="left")
    fig.tight_layout()
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out}")


def plot_series(fe: dict, ref: dict, out: Path, title: str) -> None:
    """The histories as plain time series, time on a LINEAR axis from the
    load onwards: h', k' and l' side by side, the degrees as colours,
    the reference as lines and the finite elements as markers, the
    elastic (t = 0+) values at t = 0, and the reference's static-fluid
    relaxed value, where it has one, as a dotted level. For exposition:
    the shape of the relaxation, which the logarithmic history plot
    hides. Wants output times spread over the span (run with -times)."""
    numbers = [q for q in ("h_load", "k_load", "l_load")
               if q in fe["elastic"]]
    degrees = [l for l in fe["degree"] if l in ref["degree"]]
    fig, axes = plt.subplots(1, len(numbers), squeeze=False,
                             figsize=(4.3 * len(numbers), 3.9))
    horizon = ref.get("horizon")
    for a, q in enumerate(numbers):
        ax = axes[0][a]
        for k, l in enumerate(degrees):
            r = ref["degree"].index(l)
            d = fe["degree"].index(l)
            colour = SERIES[k % len(SERIES)]
            rt = [0.0] + [h["time"] for h in ref["histories"]]
            rv = [ref["elastic"][q][r]] + [h[q][r]
                                           for h in ref["histories"]]
            ax.plot(rt, rv, color=colour, lw=1.6, label=f"l = {l}")
            ft = [0.0] + [h["time"] for h in fe["histories"]]
            fv = [fe["elastic"][q][d]] + [h[q][d] for h in fe["histories"]]
            every = max(1, len(ft) // 25)  # markers that stay legible
            ax.plot(ft[::every], fv[::every], ls="none",
                    marker=MARKERS[k % len(MARKERS)],
                    color=colour, mfc="white", mew=1.2, ms=4.5)
            relaxed = (ref.get("relaxed") or {}).get(q)
            if relaxed and relaxed[r] is not None:
                ax.axhline(relaxed[r], color=colour, lw=0.9, ls=":")
        if horizon and horizon < max(ft):
            ax.axvline(horizon, color=MUTED, lw=1.0, ls="--")
        ax.set_title(f"{LABELS[q]}(t)", loc="left")
        ax.set_xlabel("time after the load / Maxwell time")
        ax.set_xlim(left=0.0)
    handles = [plt.Line2D([], [], color=INK, lw=1.6),
               plt.Line2D([], [], ls="none", marker="o", color=INK,
                          mfc="white", mew=1.2, ms=4.5)]
    names = ["reference (Laplace)", "finite elements"]
    if any((ref.get("relaxed") or {}).get(q) and
           any(v is not None for v in ref["relaxed"][q]) for q in numbers):
        handles.append(plt.Line2D([], [], color=INK, lw=0.9, ls=":"))
        names.append("static-fluid relaxed value")
    deg_handles, deg_names = axes[0][0].get_legend_handles_labels()
    fig.legend(handles + deg_handles, names + deg_names, loc="center left",
               bbox_to_anchor=(1.0, 0.5), frameon=False)
    fig.suptitle(title, x=0.01, ha="left")
    fig.tight_layout()
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out}")


def main() -> None:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("results", type=Path, nargs="+",
                   help="results files of viscoelastic_love")
    p.add_argument("--reference", type=Path, required=True,
                   help="the laplace_reference.py file of the same model "
                        "and taus")
    p.add_argument("--out", type=Path, default=None,
                   help="where the tables and figures go (default: beside "
                        "each results file)")
    p.add_argument("--no-plots", action="store_true")
    args = p.parse_args()
    ref = json.loads(args.reference.read_text())
    for path in args.results:
        fe = json.loads(path.read_text())
        e = errors(fe, ref)
        if not e["times"]:
            raise SystemExit(f"{path}: no output time matches a valid "
                             f"reference time of {args.reference}")
        out = args.out or path.parent
        out.mkdir(parents=True, exist_ok=True)
        stem = path.stem
        text = table(stem, fe, e)
        print(text)
        (out / f"{stem}_errors.md").write_text(text)
        (out / f"{stem}_errors.json").write_text(json.dumps(
            {"results": str(path), "reference": str(args.reference),
             "scheme": fe.get("scheme"), "cost": fe.get("cost"), **e},
            indent=1))
        if not args.no_plots:
            taus = ", ".join(
                # an elastic layer's tau is written as "inf"
                f"{la['name']} " + ("fluid" if la.get("fluid") else
                                    "elastic" if la["tau"] in (None, "inf")
                                    else f"$\\tau$ = {la['tau']:g}")
                for la in fe.get("layers", []))
            title = (f"{ref['model']} ({taus}); order {fe.get('order')}, "
                     f"{fe.get('scheme')}")
            plot_history(fe, ref, out / f"{stem}_history.png", title)
            plot_error(e, out / f"{stem}_error.png", title)
            if len(fe["histories"]) >= 20:
                # enough times to draw the relaxation as a curve
                plot_series(fe, ref, out / f"{stem}_series.png", title)


if __name__ == "__main__":
    main()
