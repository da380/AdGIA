"""The time-stepping survey: which scheme is worth using, and where.

Drives `viscoelastic_schemes` (the beam problem, every integrator, the
cost-to-target search) over the two axes that decide the practical
question for a GIA code:

  stiffness   the relaxation-time CONTRAST tau_max / tau_min, standing
              in for laterally varying viscosity (eta ~ 1e18 Pa s next
              to 1e21 is a contrast of 1e3, tau_min of order a year
              against simulated spans of 1e5 years): the explicit
              stability limit dt < ~2.8 tau_min binds while the
              implicit and exponential schemes never notice;
  regime      the load period over the relaxation time (a Deborah
              number): fast loads control the step for every scheme
              (nothing is stiff, explicit is honest work), slow loads
              leave relaxation in control and reward A-stability.

For each sweep point the example finds, per scheme, the coarsest step
(or loosest adaptive tolerance) that reaches the target relative error
of the displacement history — the worst over the final time and four
interior checkpoints, against RK4 at a small step — and its cost in
elastic solves, the cost unit that transfers to the real 3-D problems,
where one solve is one self-gravitating quasi-static system. Figures,
at the smallest of --targets:

  ve_stiffness.png   solves to reach the target vs the contrast
  ve_regimes.png     solves to reach the target vs the load period

    ./survey                       # both sweeps, then the figures
    ./survey --targets 1e-2 --contrasts 1 10 100
    ./survey --figures-only        # re-draw from the JSON already run
"""
from __future__ import annotations

import argparse
import json
import shlex
import subprocess
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

HERE = Path(__file__).resolve().parent
REPOSITORY = HERE.parent.parent.parent

COLOURS = {"RK4": "#e34948", "ETD1": "#e87ba4", "BE": "#eda100",
           "SDIRK23": "#2a78d6", "ExpTrap": "#1baf7a",
           "Adaptive": "#4a3aa7"}
MARKERS = {"RK4": "v", "ETD1": "P", "BE": "D", "SDIRK23": "o",
           "ExpTrap": "^", "Adaptive": "s"}
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e4e3df"

plt.rcParams.update({
    "font.size": 15, "axes.titlesize": 16, "axes.labelsize": 15,
    "xtick.labelsize": 13, "ytick.labelsize": 13, "legend.fontsize": 12,
    "axes.edgecolor": MUTED, "axes.labelcolor": INK,
    "xtick.color": MUTED, "ytick.color": MUTED, "text.color": INK,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8,
    "lines.linewidth": 2.4, "lines.markersize": 9,
    "figure.facecolor": "white", "savefig.dpi": 200,
})


def find_program(given: Path | None) -> Path:
    if given is not None:
        return given.resolve()
    for build in sorted(REPOSITORY.glob("build*")):
        candidate = build / "examples" / "viscoelastic_schemes"
        if candidate.exists():
            return candidate
    raise SystemExit("viscoelastic_schemes not found in a build tree "
                     "(build the examples), and no --program given")


def run(command: list[str], *, dry: bool) -> None:
    print("$ " + shlex.join(command), flush=True)
    if not dry:
        subprocess.run(command, check=True,
                       stdout=subprocess.DEVNULL if "-out" in command
                       else None)


def solves(path: Path, target: float) -> dict[str, float | None]:
    """Scheme -> elastic solves to reach `target`, None if not reached;
    the adaptive rows collapse onto one 'Adaptive' entry."""
    rows = json.loads(path.read_text())["rows"]
    out: dict[str, float | None] = {}
    for r in rows:
        if abs(r["target"] - target) > 1e-12 * target:
            continue
        name = r["scheme"].split()[0]
        if name not in out or (r["reached"] and out[name] is None):
            out[name] = r["solves"] if r["reached"] else None
    return out


def draw(points: dict[float, dict], target: float, xlabel: str,
         title: str, out: Path, note: str) -> None:
    fig, ax = plt.subplots(figsize=(8.4, 5.4))
    xs = sorted(points)
    schemes = [s for s in COLOURS if any(s in points[x] for x in xs)]
    top = 1.0
    arrows = False
    for s in schemes:
        xy = [(x, points[x][s]) for x in xs
              if points[x].get(s) is not None]
        if xy:
            ax.loglog(*zip(*xy), marker=MARKERS[s], color=COLOURS[s],
                      label=s)
            top = max(top, max(v for _, v in xy))
        missing = [x for x in xs if s in points[x]
                   and points[x][s] is None]
        for x in missing:
            arrows = True
            ax.annotate("", (x, top * 2.4), (x, top * 1.2),
                        arrowprops=dict(arrowstyle="->",
                                        color=COLOURS[s], lw=2.2))
    ax.set_xlabel(xlabel)
    ax.set_ylabel(f"elastic solves to {target:g} accuracy")
    ax.set_title(title, loc="left")
    ax.legend(framealpha=0.95, ncols=2)
    if arrows or not note.startswith("arrows"):
        # the arrows' key only where there are arrows to explain
        fig.text(0.12, -0.02, note, fontsize=12, color=MUTED, va="top",
                 wrap=True)
    fig.tight_layout()
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out}")


def main() -> None:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--out", type=Path, default=Path("survey"))
    p.add_argument("--program", type=Path, default=None)
    p.add_argument("--targets", type=float, nargs="+", default=[1e-3])
    p.add_argument("--contrasts", type=float, nargs="+",
                   default=[1.0, 10.0, 100.0, 1000.0])
    p.add_argument("--periods", type=float, nargs="+",
                   default=[0.13, 0.42, 1.3, 4.2, 13.0, 42.0],
                   help="load periods, units of tau (regime sweep); "
                        "incommensurate with the checkpoint grid, and "
                        "the load runs with a phase offset, so no "
                        "scheme can score stroboscopically")
    p.add_argument("--t-final", type=float, default=4.0)
    p.add_argument("--figures-only", action="store_true")
    p.add_argument("--dry-run", action="store_true")
    args = p.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    program = find_program(args.program)
    mesh = REPOSITORY / "data" / "beam-quad.mesh"
    targets = ",".join(f"{t:g}" for t in args.targets)
    target = min(args.targets)

    if not args.figures_only:
        for r in args.contrasts:
            f = args.out / f"stiffness_r{r:g}.json"
            if f.exists():
                continue
            cmd = [str(program), "-m", str(mesh), "-tau-ratio",
                   f"{r:g}", "-targets", targets, "-tf",
                   f"{args.t_final:g}", "-steps", "1", "-out", str(f)]
            if r >= 500:  # keep the RK4 reference affordable: 25
                cmd += ["-nref",  # steps per tau_min still ~1e-6 exact
                        str(int(25 * args.t_final * r))]
            run(cmd, dry=args.dry_run)
        for tp in args.periods:
            f = args.out / f"regime_tp{tp:g}.json"
            if f.exists():
                continue
            run([str(program), "-m", str(mesh), "-tp", f"{tp:g}",
                 "-ph", "1.0", "-targets", targets, "-tf",
                 f"{args.t_final:g}", "-steps", "1", "-out", str(f)],
                dry=args.dry_run)

    if args.dry_run:
        return
    stiff = {r: solves(args.out / f"stiffness_r{r:g}.json", target)
             for r in args.contrasts
             if (args.out / f"stiffness_r{r:g}.json").exists()}
    if stiff:
        draw(stiff, target,
             "relaxation-time contrast  $\\tau_{max}/\\tau_{min}$",
             "laterally varying viscosity: the explicit penalty",
             args.out / "ve_stiffness.png",
             "arrows: target not reached within the step budget — the "
             "explicit stability limit $dt \\lesssim 2.8\\,\\tau_{min}$")
    regimes = {tp: solves(args.out / f"regime_tp{tp:g}.json", target)
               for tp in args.periods
               if (args.out / f"regime_tp{tp:g}.json").exists()}
    if regimes:
        draw(regimes, target,
             "load period / relaxation time",
             "who controls the step: the load or the viscosity",
             args.out / "ve_regimes.png",
             "left: load-controlled (every scheme resolves the "
             "forcing).\nright: relaxation-controlled (A-stability "
             "pays)")


if __name__ == "__main__":
    main()
