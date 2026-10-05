"""Slide-grade figures for talks, from the benchmark trees already run.

Single-message figures with slide typography (large type, one point per
panel), regenerated at will from the raw results:

  methods.png     five formulations, one reference: agreement by degree
                  and the cost of each (fluid_core, h = 0.3, order 2)
  cmb.png         what the CMB approximations cost, loading vs tidal
                  (prem_4, h = 0.2, order 2), the rotational-feedback
                  column being the tidal one
  identity.png    the relabelled change-of-variables identity: every
                  welded case certified to solver precision, the
                  covariant gauge penalty included; the slipping
                  interface informational
  derivative.png  d(Love)/d(interface radius), 3-D against 1-D
                  (fluid_core, referential and slip_broken)
  aspherical.png  field errors on the independently meshed aspherical
                  body against the shape amplitude: flat in the
                  amplitude, falling with refinement
  ladder.png      h-refinement at orders 2 and 3 (fluid_core)
  models.png      every model of the sweep at orders 2 and 3: the
                  typical and the worst error of its Love numbers
  viscoelastic.png  Maxwell Love-number histories in time, finite
                  elements against the Laplace-domain reference
  fieldmap.png    the response to an off-axis cap load on the surface:
                  reference, finite elements, difference

    python talk_figures.py --out talk
    python talk_figures.py --out talk --methods-case <dir> ...

Every input has a flag; the defaults point at the build-tree runs this
repository's campaigns produce (run from <build>/benchmarks).
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent / "common"))

from costs import load_cost  # noqa: E402

COLOURS = {"dahlen": "#2a78d6", "gauged": "#eb6834",
           "referential": "#eda100", "slip": "#008300",
           "slip_broken": "#4a3aa7"}
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e4e3df"
GOOD, WARN = "#1baf7a", "#eda100"

plt.rcParams.update({
    "font.size": 15, "axes.titlesize": 16, "axes.labelsize": 15,
    "xtick.labelsize": 13, "ytick.labelsize": 13, "legend.fontsize": 13,
    "axes.edgecolor": MUTED, "axes.labelcolor": INK,
    "xtick.color": MUTED, "ytick.color": MUTED, "text.color": INK,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8,
    "lines.linewidth": 2.4, "lines.markersize": 9,
    "figure.facecolor": "white", "savefig.dpi": 200,
})


def love(path: Path) -> dict:
    r = json.loads(path.read_text())
    return {d["degree"]: d for d in r["degrees"]}


def reference(case: Path) -> dict:
    r = json.loads((case / "reference.json").read_text())
    return {l: {q: r[f"{q}_load"][i] for q in "hlk"}
            for i, l in enumerate(r["degree"])}


def save(fig, out: Path) -> None:
    fig.tight_layout()
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out}")


def methods_figure(case: Path, out: Path, combined: bool = False) -> None:
    """Agreement by degree and cost per solve, the five formulations.

    Each formulation's results by degree are drawn, or its combined
    results (suffix _combined) where those are all there is; with
    combined=True the combined results are preferred. A combined run's
    cost is that of its one load solve for all the degrees, not a mean
    by degree, and its bar is labelled so."""
    ref = reference(case)
    series = [("dahlen", ""), ("gauged", "_gauged"),
              ("referential", "_referential"), ("slip", "_slip"),
              ("slip_broken", "_slip_broken")]
    fig, (a0, a1) = plt.subplots(1, 2, figsize=(12.6, 4.8),
                                 gridspec_kw={"width_ratios": [3, 2]})
    names, labels, costs, its, alls = [], [], [], [], []
    for name, tag in series:
        tags = [tag + "_combined", tag] if combined else \
            [tag, tag + "_combined"]
        path = next((case / f"results_o2{t}.json" for t in tags
                     if (case / f"results_o2{t}.json").exists()),
                    case / f"results_o2{tag}.json")
        r = love(path)
        ls, errs = [], []
        for l in sorted(set(r) & set(ref)):
            if l == 0 and name == "dahlen":
                continue  # differs by design with a fluid
            worst = 0.0
            for q in ("h", "k"):
                if q == "k" and l < 2:
                    continue
                if abs(ref[l][q]) > 1e-10:
                    worst = max(worst, abs(r[l]["load"][q] - ref[l][q])
                                / abs(ref[l][q]))
            ls.append(l)
            errs.append(worst)
        a0.semilogy(ls, errs, marker="o", color=COLOURS[name], label=name)
        cost = load_cost(json.loads(path.read_text()))
        names.append(name)
        labels.append(f"{name} (combined)" if cost.combined else name)
        costs.append(cost.seconds)
        its.append(cost.iterations)
        alls.append(cost.combined)
    a0.set_xlabel("degree $l$")
    a0.set_ylabel("relative error vs pyslfp")
    a0.set_title("five formulations, one radial reference", loc="left")
    a0.xaxis.set_major_locator(plt.MaxNLocator(integer=True))
    a0.legend(framealpha=0.95)
    y = range(len(names))[::-1]
    a1.barh(y, costs, color=[COLOURS[n] for n in names], height=0.62)
    for yi, c, it, one in zip(y, costs, its, alls):
        a1.text(c * 1.15, yi, f"{c:.1f} s   ({it:.0f} its)"
                + (", all $l$" if one else ""),
                va="center", fontsize=12, color=MUTED)
    a1.set_yticks(y, labels)
    a1.set_xscale("log")
    a1.set_xlim(right=max(costs) * 8)
    a1.set_xlabel("seconds per solve" if not any(alls) else
                  "seconds per solve (combined: one solve, all $l$)")
    a1.set_title("what each one costs", loc="left")
    a1.grid(axis="y", visible=False)
    save(fig, out / "methods.png")


def cmb_figure(case: Path, out: Path) -> None:
    """Loading vs tidal cost of the CMB approximations, vs full."""
    full = love(case / "results_o2.json")
    treatments = ["nomass", "uniform", "winkler"]
    fig, ax = plt.subplots(figsize=(8.6, 4.8))
    width, l = 0.34, 2
    for j, (forcing, colour, label) in enumerate(
            [("load", "#2a78d6", "loading $k'_2$"),
             ("tide", "#e34948", "tidal $k_2$")]):
        vals = []
        for t in treatments:
            r = love(case / f"results_o2_{t}.json")
            vals.append(abs(r[l][forcing]["k"] - full[l][forcing]["k"])
                        / abs(full[l][forcing]["k"]))
        pos = [i + (j - 0.5) * width for i in range(len(treatments))]
        ax.bar(pos, vals, width, color=colour, label=label)
        for p, v in zip(pos, vals):
            ax.text(p, v * 1.25, f"{v * 100:.2g}%", ha="center",
                    fontsize=12, color=MUTED)
    ax.set_yscale("log")
    ax.set_ylim(1e-4, 3.0)
    ax.set_xticks(range(len(treatments)), treatments)
    ax.set_ylabel("deviation from the full treatment")
    ax.set_title("the CMB approximations: cheap for loading, "
                 "not for tides", loc="left")
    ax.legend(framealpha=0.95)
    ax.grid(axis="x", visible=False)
    ax.text(0.01, -0.16,
            "rotational feedbacks are driven by the degree-2 tidal "
            "response — the red column is their error budget",
            transform=ax.transAxes, fontsize=12, color=MUTED)
    save(fig, out / "cmb.png")


def identity_figure(campaign: Path, out: Path) -> None:
    """The relabelled change-of-variables identity: every welded case
    strict, the gauge penalty included; the slipping interface is the
    one informational case left."""
    def measured(log: Path) -> float:
        for line in log.read_text().splitlines():
            if "u_A - u_B" in line:
                return float(line.split()[-1])
        raise SystemExit(f"no identity number in {log}")

    bars = []
    for log, label in [
            ("identity_homogeneous_h0.3.txt", "homogeneous (welded solid)"),
            ("identity_two_solid_h0.3.txt", "two solids (welded)"),
            ("identity_fluid_core_h0.3.txt",
             "fluid core (welded, gauged)")]:
        path = campaign / log
        if path.exists():
            bars.append((label, measured(path), GOOD))
    # The slipping interface: read the campaign's slip identity log when
    # it exists; the fallback constant is the measurement of
    # doc/benchmarks.tex, "Leg B" (fluid_core, A = 0.02, h = 0.3). The
    # slipping-interface forms are not certified covariant, so this bar
    # is informational.
    slip_log = campaign / "identity_fluid_core_slip_h0.3.txt"
    bars.append(("fluid core (slip interface)",
                 measured(slip_log) if slip_log.exists() else 1.2e-2,
                 WARN))
    fig, ax = plt.subplots(figsize=(9.2, 4.2))
    y = range(len(bars))[::-1]
    ax.barh(y, [b[1] for b in bars], color=[b[2] for b in bars],
            height=0.6)
    for yi, (_, v, _c) in zip(y, bars):
        ax.text(v * 1.4, yi, f"{v:.0e}", va="center", fontsize=13,
                color=MUTED)
    ax.set_yticks(y, [b[0] for b in bars])
    ax.set_xscale("log")
    ax.set_xlim(1e-8, 3)
    ax.set_ylim(-0.9, len(bars) - 0.4)
    ax.set_xlabel("relative difference of the two discrete solutions")
    ax.set_title("mapped assembly vs remeshed assembly: "
                 "the same linear system", loc="left")
    ax.axvline(2e-6, color=MUTED, linewidth=1.2, linestyle="--")
    ax.text(2.6e-6, -0.62, "solver / DtN floor", fontsize=12,
            color=MUTED)
    ax.grid(axis="y", visible=False)
    fig.text(0.13, -0.04,
             "green: every mapped term certified at once — the gauge "
             "penalty included\n"
             "amber: the slipping-interface forms (not certified "
             "covariant; the physics is benchmarked separately)",
             fontsize=12, color=MUTED, va="top")
    save(fig, out / "identity.png")


def derivative_figure(case: Path, out: Path, eps: float = 0.02,
                      methods: tuple[str, ...] = ("referential",
                                                  "slip_broken")) -> None:
    """The 3-D vs 1-D derivative identity plot: each method's fixed-mesh
    central difference against that of the pyslfp references of the
    perturbed models, the Love number by marker."""
    refs = {}
    for s in (eps, -eps):
        r = json.loads((case / f"shift_{s:+g}" /
                        "reference.json").read_text())
        refs[s] = {l: {q: r[f"{q}_load"][i] for q in "hlk"}
                   for i, l in enumerate(r["degree"])}
    shape = {"h": "o", "l": "s", "k": "^"}
    fig, ax = plt.subplots(figsize=(6.6, 6.0))
    lo = hi = None
    drawn = []
    for m in methods:
        paths = [case / f"results_o2_{m}_shift{s:+g}.json"
                 for s in (eps, -eps)]
        if not all(p.exists() for p in paths):
            continue
        plus, minus = love(paths[0]), love(paths[1])
        drawn.append(m)
        for l in sorted(set(plus) & set(minus)):
            if l == 1:
                continue
            for q in "hlk":
                if q != "h" and l < 2:
                    continue
                d3 = (plus[l]["load"][q] - minus[l]["load"][q]) / (2 * eps)
                d1 = (refs[eps][l][q] - refs[-eps][l][q]) / (2 * eps)
                if abs(d1) < 1e-10:
                    continue
                ax.plot(d1, d3, shape[q], color=COLOURS[m], markersize=10,
                        markerfacecolor=COLOURS[m] if m == methods[0]
                        else "none", markeredgewidth=2.0)
                lo = d1 if lo is None else min(lo, d1, d3)
                hi = d1 if hi is None else max(hi, d1, d3)
    pad = 0.1 * (hi - lo)
    ax.plot([lo - pad, hi + pad], [lo - pad, hi + pad], color=MUTED,
            linewidth=1.0, zorder=0)
    handles = [plt.Line2D([], [], ls="none", marker="o", color=COLOURS[m],
                          markerfacecolor=COLOURS[m] if m == methods[0]
                          else "none", markeredgewidth=2.0, label=m)
               for m in drawn]
    handles += [plt.Line2D([], [], ls="none", marker=mk, color=MUTED,
                           label=f"${q}'_l$") for q, mk in shape.items()]
    ax.legend(handles=handles, framealpha=0.95, loc="upper left")
    ax.set_xlabel("1-D reference (pyslfp, perturbed models)")
    ax.set_ylabel("3-D solver, one fixed mesh")
    ax.set_title("d(Love number) / d(core radius)", loc="left")
    save(fig, out / "derivative.png")


def aspherical_figure(results: Path, out: Path) -> None:
    """Field errors on the independently meshed aspherical body, against
    the shape amplitude (one gmsh mesh, its nodes moved by the shape:
    amplitude zero is the sphere): flat in the amplitude means the
    mapped machinery adds nothing, the mesh with elements halved shows
    the error is discretisation."""
    runs = {}
    for path in results.glob("aspherical_e*_o2.json"):
        r = json.loads(path.read_text())
        runs[float(r["eps"])] = {d["degree"]: d for d in r["degrees"]}
    # the stock mesh (amplitude 0.05), when the sweep lacks it; the
    # refined mesh (half-h) is drawn apart, below
    for name in ("aspherical_eps0.05_o2.json",):
        if (results / name).exists() and 0.05 not in runs:
            r = json.loads((results / name).read_text())
            runs[0.05] = {d["degree"]: d for d in r["degrees"]}
    fine = None
    if (results / "aspherical_half-h_o2.json").exists():
        r = json.loads((results / "aspherical_half-h_o2.json").read_text())
        fine = (float(r["eps"]), {d["degree"]: d for d in r["degrees"]})
    if not runs:
        print("no aspherical runs: aspherical.png skipped")
        return
    amps = sorted(runs)
    degrees = sorted(set.intersection(*(set(v) for v in runs.values())))
    palette = ["#2a78d6", "#eb6834", "#1baf7a", "#4a3aa7", "#e34948"]
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.8), sharey=True)
    for ax, key, title in zip(axes, ("u_error", "phi_error"),
                              ("displacement", "potential")):
        for i, l in enumerate(degrees):
            c = palette[i % len(palette)]
            ax.semilogy(amps, [runs[a][l][key] for a in amps], marker="o",
                        color=c, label=f"$l$ = {l}")
            if fine and l in fine[1]:
                ax.semilogy([fine[0]], [fine[1][l][key]], marker="*",
                            markersize=15, color=c, linestyle="none")
        ax.set_xlabel("shape amplitude $\\varepsilon$ (0: sphere)")
        ax.set_title(f"{title}: relative $L^2$ error", loc="left")
        ax.set_xticks(amps)
    axes[0].set_ylabel("error against pyslfp, through the map")
    handles, labels = axes[0].get_legend_handles_labels()
    if fine:
        handles.append(plt.Line2D([], [], ls="none", marker="*",
                                  markersize=15, color=MUTED))
        labels.append("elements halved")
    fig.legend(handles, labels, loc="center left", bbox_to_anchor=(1.0, 0.5),
               frameon=False)
    fig.suptitle("aspherical body, own mesh, exact reference: the error "
                 "does not see the shape", x=0.01, ha="left")
    save(fig, out / "aspherical.png")


def ladder_figure(tree: Path, out: Path) -> None:
    """h-refinement at orders 2 and 3: per mesh, the median and the
    worst relative error over the load and tidal Love numbers of
    degrees 2 to 4."""
    sys.path.insert(0, str(Path(__file__).resolve().parent / "love_numbers"))
    import plot as love_plot  # noqa: E402
    by_order: dict[int, list[tuple[float, float, float]]] = {}
    for case in sorted(tree.glob("h*")):
        ref = love_plot.reference_values(
            json.loads((case / "reference.json").read_text()))
        for path in sorted(case.glob("results_o*.json")):
            if path.stem.count("_") > 1:
                continue  # Dahlen, full CMB treatment only
            run = love_plot.read_run(path)
            errs = [e for key, *_ in love_plot.QUANTITIES
                    for l, e in love_plot.relative_error(run, ref,
                                                         key).items()
                    if 2 <= l <= 4]
            by_order.setdefault(run.order, []).append(
                (run.h, float(sorted(errs)[len(errs) // 2]), max(errs)))
    if not by_order:
        print(f"no ladder under {tree}: ladder.png skipped")
        return
    palette = {2: "#2a78d6", 3: "#eb6834"}
    fig, ax = plt.subplots(figsize=(7.8, 5.2))
    sizes = sorted({h for v in by_order.values() for h, *_ in v})
    for order, pts in sorted(by_order.items()):
        pts.sort()
        h, med, worst = zip(*pts)
        c = palette.get(order, MUTED)
        ax.fill_between(h, med, worst, color=c, alpha=0.15, linewidth=0)
        ax.loglog(h, med, marker="o", color=c,
                  label=f"order {order}: median (band: worst)")
    ax.set_xscale("log")
    ticks = [x for i, x in enumerate(sorted(sizes, reverse=True))
             if i == 0 or sorted(sizes, reverse=True)[i - 1] / x > 1.12]
    ax.set_xticks(ticks, [f"{x:g}" for x in ticks])
    ax.minorticks_off()
    ax.set_xlabel("element size $h$ (each size its own mesh)")
    ax.set_ylabel("relative error, Love numbers $l$ = 2–4")
    ax.set_title(f"{tree.name}: one order up buys 10–100×", loc="left")
    ax.legend(framealpha=0.95)
    save(fig, out / "ladder.png")


def models_figure(tree: Path, out: Path, h: float = 0.2) -> None:
    """Every model of the sweep at one element size, orders 2 and 3:
    the median relative error of its Love numbers (degrees 1 to 5, load
    and tide) as the bar, the worst as the tick above it."""
    sys.path.insert(0, str(Path(__file__).resolve().parent / "love_numbers"))
    import plot as love_plot  # noqa: E402
    rows = []
    sys.path.insert(0, str(Path(__file__).resolve().parent / "common"))
    import models as model_table  # noqa: E402
    # the named models only (a tree may hold variant studies beside
    # them, prem_4_cmb or fluid_core_geom3, which repeat a model)
    for model in sorted(d for d in tree.iterdir() if d.is_dir()
                        and d.name in model_table.MODELS):
        case = model / f"h{h:g}"
        if not (case / "reference.json").exists():
            continue
        ref = love_plot.reference_values(
            json.loads((case / "reference.json").read_text()))
        entry = {}
        for order in (2, 3):
            path = case / f"results_o{order}.json"
            if not path.exists():
                continue
            run = love_plot.read_run(path)
            errs = sorted(e for key, *_ in love_plot.QUANTITIES
                          for l, e in love_plot.relative_error(
                              run, ref, key).items() if 1 <= l <= 5)
            entry[order] = (errs[len(errs) // 2], errs[-1])
        if entry:
            rows.append((model.name, entry))
    if not rows:
        print(f"no model sweep under {tree}: models.png skipped")
        return
    palette = {2: "#2a78d6", 3: "#eb6834"}
    fig, ax = plt.subplots(figsize=(11.5, 4.8))
    width = 0.38
    for j, order in enumerate((2, 3)):
        xs = [i + (j - 0.5) * width for i in range(len(rows))]
        med = [r[1].get(order, (float("nan"),) * 2)[0] for r in rows]
        worst = [r[1].get(order, (float("nan"),) * 2)[1] for r in rows]
        ax.bar(xs, med, width, color=palette[order],
               label=f"order {order}: median")
        ax.plot(xs, worst, linestyle="none", marker="_", markersize=18,
                markeredgewidth=2.4, color=palette[order])
    ax.plot([], [], linestyle="none", marker="_", markersize=18,
            markeredgewidth=2.4, color=MUTED, label="worst")
    ax.set_yscale("log")
    ax.set_xticks(range(len(rows)), [r[0] for r in rows], rotation=20)
    ax.set_ylabel("relative error vs pyslfp")
    ax.set_title(f"every model, h = {h:g}: Love numbers of degrees 1–5, "
                 "load and tide", loc="left")
    ax.legend(framealpha=0.95, ncols=3, loc="upper center",
              bbox_to_anchor=(0.5, -0.22), frameon=False)
    ax.grid(axis="x", visible=False)
    save(fig, out / "models.png")


def viscoelastic_figure(series: Path, out: Path) -> None:
    """Maxwell load Love numbers h'_l(t) in time, finite elements
    (markers) on the Laplace-domain reference (lines): an elastic
    lithosphere over a Maxwell interior, and a Maxwell mantle over a
    fluid core."""
    cases = [("fe_hl_series.json", "laplace_hl_series.json",
              "elastic lithosphere, Maxwell interior"),
             ("fe_fluid_core_series.json", "laplace_fluid_core_series.json",
              "Maxwell mantle, fluid core")]
    cases = [c for c in cases if (series / c[0]).exists()
             and (series / c[1]).exists()]
    if not cases:
        print(f"no viscoelastic series under {series}: skipped")
        return
    palette = ["#2a78d6", "#eb6834", "#1baf7a"]
    fig, axes = plt.subplots(1, len(cases), figsize=(6.2 * len(cases), 4.8),
                             squeeze=False)
    for ax, (fe_name, ref_name, title) in zip(axes[0], cases):
        fe = json.loads((series / fe_name).read_text())
        ref = json.loads((series / ref_name).read_text())
        for k, l in enumerate(d for d in fe["degree"] if d in ref["degree"]):
            c = palette[k % len(palette)]
            r, d = ref["degree"].index(l), fe["degree"].index(l)
            ax.plot([0.0] + [h["time"] for h in ref["histories"]],
                    [ref["elastic"]["h_load"][r]]
                    + [h["h_load"][r] for h in ref["histories"]],
                    color=c, label=f"$l$ = {l}")
            ft = [0.0] + [h["time"] for h in fe["histories"]]
            fv = [fe["elastic"]["h_load"][d]] + [h["h_load"][d]
                                                 for h in fe["histories"]]
            ax.plot(ft[::2], fv[::2], "o", color=c, markerfacecolor="white",
                    markeredgewidth=1.6, markersize=6)
            relaxed = (ref.get("relaxed") or {}).get("h_load")
            if relaxed and relaxed[r] is not None:
                ax.axhline(relaxed[r], color=c, linewidth=1.0, linestyle=":")
        ax.set_title(title, loc="left")
        ax.set_xlabel("time after the load / Maxwell time")
        ax.set_xlim(left=0.0)
    axes[0][0].set_ylabel("load Love number $h'_l(t)$")
    handles, labels = axes[0][0].get_legend_handles_labels()
    handles += [plt.Line2D([], [], color=INK),
                plt.Line2D([], [], ls="none", marker="o", color=INK,
                           markerfacecolor="white", markeredgewidth=1.6),
                plt.Line2D([], [], color=INK, linestyle=":", linewidth=1.0)]
    labels += ["reference (Laplace)", "finite elements",
               "static-fluid limit"]
    fig.legend(handles, labels, loc="center left", bbox_to_anchor=(1.0, 0.5),
               frameon=False)
    save(fig, out / "viscoelastic.png")


def fieldmap_figure(field_path: Path, out: Path) -> None:
    """The radial displacement on the surface under an off-axis cap
    load: reference, finite elements, and their difference on its own
    scale."""
    import numpy as np
    sys.path.insert(0, str(Path(__file__).resolve().parent / "love_numbers"))
    import plot as love_plot  # noqa: E402
    if not field_path.exists():
        print(f"no {field_path}: fieldmap.png skipped")
        return
    field = json.loads(field_path.read_text())
    lat = np.linspace(-90.0, 90.0, 181)
    lon = np.linspace(-180.0, 180.0, 361)
    LON, LAT = np.meshgrid(np.radians(lon), np.radians(lat))
    Y = love_plot.harmonics(field["lmax"], 0.5 * np.pi - LAT, LON)
    fe = np.tensordot(np.array(field["u"]), Y, axes=1)
    ref = np.tensordot(np.array(field["u_reference"]), Y, axes=1)
    cmap = love_plot.diverging()
    fig, axes = plt.subplots(1, 3, figsize=(15, 3.9),
                             subplot_kw={"projection": "mollweide"})
    scale = np.abs(ref).max()
    diff = fe - ref
    for ax, values, name, vmax in (
            (axes[0], ref, "reference (pyslfp)", scale),
            (axes[1], fe, "finite elements", scale),
            (axes[2], diff, f"difference (max {np.abs(diff).max() / scale:.1%}"
             " of the signal)", np.abs(diff).max())):
        mesh = ax.pcolormesh(LON, LAT, values, cmap=cmap, vmin=-vmax,
                             vmax=vmax, shading="auto", rasterized=True)
        ax.set_title(name, loc="left", fontsize=14)
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.grid(True, color=GRID, linewidth=0.5)
        bar = fig.colorbar(mesh, ax=ax, orientation="horizontal",
                           fraction=0.05, pad=0.05)
        bar.formatter.set_powerlimits((-2, 2))
        bar.outline.set_visible(False)
    fig.suptitle("radial displacement under a cap load, degrees "
                 f"{field['lmin']}–{field['lmax']} (fluid core, order "
                 f"{field['order']})", x=0.01, ha="left")
    save(fig, out / "fieldmap.png")


def main() -> None:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    # Defaults read the campaign tree (runs_campaign, campaign.py) and
    # the hand-made trees beside it (talk_data, viscoelastic_series,
    # runs_campaign/ladder), relative to the working directory.
    base = Path("runs_campaign/love_numbers")
    p.add_argument("--out", type=Path, default=Path("talk"))
    p.add_argument("--methods-case", type=Path,
                   default=base / "fluid_core/h0.3")
    p.add_argument("--cmb-case", type=Path,
                   default=base / "prem_4/h0.2")
    p.add_argument("--identity-logs", type=Path,
                   default=Path("runs_campaign/relabelling"))
    p.add_argument("--aspherical", type=Path, default=Path("talk_data"))
    p.add_argument("--derivative-case", type=Path,
                   default=Path("runs_campaign/perturbation/fluid_core"),
                   help="where perturbation_check wrote the shifted runs "
                        "and references")
    p.add_argument("--ladder", type=Path,
                   default=Path("runs_campaign/ladder/fluid_core"),
                   help="the UNCAPPED ladder (--angular 1.0: every rung "
                        "refines the CMB)")
    p.add_argument("--models", type=Path, default=base)
    p.add_argument("--viscoelastic", type=Path,
                   default=Path("viscoelastic_series"))
    p.add_argument("--field", type=Path,
                   default=Path("runs_campaign/love_numbers/fluid_core/h0.3/"
                                "field_o2.json"))
    p.add_argument("--combined", action="store_true",
                   help="methods.png from the combined results "
                        "(results_o2[_<method>]_combined.json, run.py "
                        "--combined) where they exist; without it they "
                        "serve only where the results by degree are "
                        "missing. A combined bar's cost is its one load "
                        "solve for all the degrees")
    args = p.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    methods_figure(args.methods_case, args.out, args.combined)
    cmb_figure(args.cmb_case, args.out)
    identity_figure(args.identity_logs, args.out)
    derivative_figure(args.derivative_case, args.out)
    aspherical_figure(args.aspherical, args.out)
    ladder_figure(args.ladder, args.out)
    models_figure(args.models, args.out)
    viscoelastic_figure(args.viscoelastic, args.out)
    fieldmap_figure(args.field, args.out)


if __name__ == "__main__":
    main()
