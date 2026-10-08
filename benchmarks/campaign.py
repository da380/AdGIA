"""The whole benchmark campaign as one script, with settable defaults.

Collates every family of benchmarks/ — the Love-number cross-method
sweep with its CMB conditions and field benchmark, the relabelling
family (mapped runs, the solver-level identity, the independently
meshed aspherical body), the perturbation family (the degree-0
interface-shift check) and a weak-scaling study — behind two profiles:

  --profile local    a rehearsal on this machine: small meshes, few
                     degrees, 8 ranks; the default stages run end to
                     end in around an hour (no scaling rungs; the
                     viscoelastic stages are opt-in, below).
  --profile server   the production shape: the full model set, an
                     h-ladder at orders 2 and 3, higher degrees, 100
                     ranks on a 128-core shared-memory machine, plus
                     the weak-scaling stage.

Every profile default is a flag; --stages picks a subset; everything
delegates to the family scripts and drivers, all of which skip work
whose results exist, so an interrupted campaign resumes where it
stopped. The tree mirrors the families:

  <out>/love_numbers/<model>/h<h>/   cases and Love/field results
                     <model>/*.png   the family's figures (plot.py)
  <out>/relabelling/                 identity logs, aspherical results
                                     and aspherical.png
  <out>/perturbation/<model>/        shifted profiles, references, runs
                                     and perturbation_*.png
  <out>/scaling/                     the weak-scaling rungs
  <out>/viscoelastic/<model>/        (opt-in stage) case, Laplace
                                     reference, FE histories, error
                                     tables and figures
  <out>/viscoelastic_box/<study>/    (opt-in stage) the box studies,
                                     summaries and figures
  <out>/viscoelastic_sphere/<study>/ (opt-in stage) the sphere studies
  <out>/campaign_log.md              every command and stage outcome

plot.py renders the Love tree at the end and the scaling rungs collate
into scaling_summary.md.

    ./campaign --profile local
    ./campaign --profile server --out /scratch/love
    ./campaign --profile server --stages methods field --lmax 20
    ./campaign --profile server --stages scaling --dry-run
    ./campaign --profile local --stages viscoelastic plot
    ./campaign --profile local --stages methods cmb plot --combined

--combined (opt-in) runs the Love-number solves of the methods and cmb
stages combined (run.py --combined: one load solve for all the degrees,
one tidal solve), results suffixed _combined beside, not over, those by
degree. It leaves alone the field stage (no Love solve), the mapped
stage (its agreement is of the order of the combined solves' leakage
between degrees), the perturbation stage (exact solves by degree for
its finite differences), the identity and aspherical stages (other
drivers) and the scaling stage (its timings are by degree). Where a
combined run's cost is shown, it is that of its one load solve,
labelled combined.

The viscoelastic stage is OPT-IN (never in the default stage list; name
it with --stages): Maxwell Love-number histories by viscoelastic_love
against the correspondence-principle reference of
viscoelastic/love/laplace_reference.py, compared by viscoelastic/love/compare.py
in the plot stage (viscoelastic/love/README.md). So is viscoelastic_box:
every study of viscoelastic/box/study.py at the campaign's profile (the
server profile adds the long 3-D runs; viscoelastic/box/README.md), and
viscoelastic_sphere, those of viscoelastic/sphere/study.py.

Method-model compatibility is encoded here: the slipping methods take a
single fluid core, the mapped stages the referential family, and the
solver-level identity the solid models and the gauged fluid core, where
it is strict. Mind doc/gauged_fluid.md, "Equivalence and the
Adams–Williamson condition", when reading converged cross-method
ladders on models whose core is not neutrally stratified.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE / "common"))

from costs import load_cost  # noqa: E402
from drivers import find_programs, run  # noqa: E402
from outputs import outside_source  # noqa: E402

#: Which methods and stages run on which models, beyond the Eulerian
#: pair, which run on everything.
REFERENTIAL_MODELS = ("homogeneous", "two_solid", "linear_solid",
                      "fluid_core", "inner_core", "stratified_core",
                      "earth_like", "prem_4")
SLIP_MODELS = ("fluid_core",)
SOLID_MODELS = ("homogeneous", "two_solid", "linear_solid")
MAPPED_MODELS = ("homogeneous", "two_solid", "linear_solid", "fluid_core")
# Strict for every welded case, solids and the gauged fluid core alike
# (the gauge penalty is covariant).
IDENTITY_MODELS = ("homogeneous", "two_solid", "linear_solid",
                   "fluid_core")
ASPHERICAL_MODELS = ("homogeneous", "linear_solid")
# slip_broken runs the shift legs: the driver forces interpolated-F
# shift maps, which keep its one-sided interface terms side-consistent
# (perturbation/README.md).
PERTURBATION = {"fluid_core": ("referential", "slip_broken"),
                "two_solid": ("referential",)}

PROFILES = {
    "local": dict(models=["homogeneous", "two_solid", "fluid_core"],
                  h=[0.3], orders=[2], lmax=4, dtn_degree=16, np=8,
                  solvers=[1], cmb=["full", "nomass", "uniform", "winkler"],
                  map_amplitude=0.02,
                  shift_eps=[0.02], pert_lmax=3, field=True, scaling=[],
                  aspherical_scale=1.0, aspherical_lmax=4,
                  # model -> Maxwell time per solid layer ("inf":
                  # elastic), centre outward
                  viscoelastic=dict(models={"fluid_core": ["1"]}, h=0.3,
                                    order=2, lmax=4, scheme="sdirk23")),
    "server": dict(models=["homogeneous", "two_solid", "linear_solid",
                           "fluid_core", "inner_core", "stratified_core",
                           "earth_like", "prem_4"],
                   h=[0.2, 0.15, 0.1], orders=[2, 3], lmax=16,
                   dtn_degree=24, np=100, solvers=[1, 0],
                   cmb=["full", "nomass", "uniform", "winkler"],
                   map_amplitude=0.02, shift_eps=[0.01, 0.02],
                   pert_lmax=4, field=True,
                   # weak scaling: h ~ np^(-1/3), roughly constant
                   # unknowns per rank up the ladder
                   scaling=[(12, 0.2), (25, 0.157), (50, 0.125),
                            (100, 0.099)],
                   aspherical_scale=0.5, aspherical_lmax=8,
                   viscoelastic=dict(
                       models={"fluid_core": ["1"],
                               "homogeneous_lithosphere": ["1", "inf"]},
                       h=0.2, order=2, lmax=8, scheme="sdirk23")),
}

STAGES = ("methods", "cmb", "field", "mapped", "identity", "aspherical",
          "perturbation", "scaling", "viscoelastic", "viscoelastic_box",
          "viscoelastic_sphere", "plot")
#: Stages run only when named in --stages.
OPT_IN = ("viscoelastic", "viscoelastic_box", "viscoelastic_sphere")


def main() -> None:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--profile", choices=sorted(PROFILES), default="local")
    p.add_argument("--stages", nargs="+", choices=STAGES, default=None,
                   help="stages to run (default: all of the profile's)")
    p.add_argument("--out", type=Path, default=Path("runs_campaign"),
                   metavar="DIR", help="root of the results tree")
    p.add_argument("--models", nargs="+", default=None)
    p.add_argument("--h", type=float, nargs="+", default=None)
    p.add_argument("--orders", type=int, nargs="+", default=None)
    p.add_argument("--lmax", type=int, default=None)
    p.add_argument("--dtn-degree", type=int, default=None)
    p.add_argument("--np", type=int, default=None)
    p.add_argument("--solvers", type=int, nargs="+", default=None)
    p.add_argument("--cmb", nargs="+", default=None)
    p.add_argument("--map-amplitude", type=float, default=None)
    p.add_argument("--shift-eps", type=float, nargs="+", default=None)
    p.add_argument("--partition", action="store_true",
                   help="partition each case for the ranks, so that no "
                        "rank reads the whole mesh (large runs)")
    p.add_argument("--combined", action="store_true",
                   help="run the Love numbers of the methods and cmb "
                        "stages with one solve per forcing for all the "
                        "degrees (run.py --combined; see above for the "
                        "stages it leaves alone)")
    p.add_argument("--programs", type=Path, default=None)
    p.add_argument("--mpiexec", default="mpiexec")
    p.add_argument("--dry-run", action="store_true",
                   help="print every command without running anything")
    args = p.parse_args()

    prof = dict(PROFILES[args.profile])
    overrides = {"models": args.models, "h": args.h, "orders": args.orders,
                 "lmax": args.lmax, "dtn_degree": args.dtn_degree,
                 "np": args.np, "solvers": args.solvers, "cmb": args.cmb,
                 "map_amplitude": args.map_amplitude,
                 "shift_eps": args.shift_eps}
    prof.update({k: v for k, v in overrides.items() if v is not None})
    prof["dtn_degree"] = max(prof["dtn_degree"], prof["lmax"])
    stages = args.stages or [s for s in STAGES if s not in OPT_IN]

    programs = Path(".") if args.dry_run and args.programs is None \
        else find_programs(args.programs)
    py, mpiexec = sys.executable, args.mpiexec
    out = args.out if args.dry_run else outside_source(args.out)
    loves = out / "love_numbers"
    if not args.dry_run:
        for d in (loves, out / "relabelling", out / "perturbation"):
            d.mkdir(parents=True, exist_ok=True)
    log: list[str] = [f"# campaign --profile {args.profile}"
                      f" --stages {' '.join(stages)}"
                      + (" --combined" if args.combined else "") +
                      f"  ({time.strftime('%Y-%m-%d %H:%M')})", ""]
    failures: list[str] = []
    dry = ["--dry-run"] if args.dry_run else []

    def stage(name: str, ok: bool) -> None:
        log.append(f"[{name}] {'ok' if ok else 'FAILED'}")
        if not ok:
            failures.append(name)

    def sh(command: list, *, log_file: Path | None = None) -> bool:
        return run([str(c) for c in command], log=log_file,
                   dry_run=args.dry_run)

    def run_py(family: str, script: str, *extra) -> bool:
        # The family scripts have --dry-run of their own, so under
        # --dry-run they still run and print their commands.
        return run([py, str(HERE / family / f"{script}.py"),
                    *(str(e) for e in extra), "--programs", str(programs),
                    "--mpiexec", mpiexec, *dry], dry_run=False)

    common = ["--h", *prof["h"], "--order", *prof["orders"],
              "--np", prof["np"], "--lmax", prof["lmax"],
              "--dtn-degree", prof["dtn_degree"], "--out", loves,
              *(["--partition"] if args.partition else [])]
    # The Love-number stages that may run combined (--combined).
    combined = ["--combined"] if args.combined else []

    # 1. The cross-method sweep, with the Eulerian solver axis.
    if "methods" in stages:
        for model in prof["models"]:
            methods = ["dahlen", "gauged"]
            if model in REFERENTIAL_MODELS:
                # maxwell = the referential problem with the
                # Maxwell-relaxed fluid (the secular route); it also
                # takes the inner-core models the slip pair cannot.
                methods += ["referential", "maxwell"]
            if model in SLIP_MODELS:
                methods += ["slip", "slip_broken"]
            stage(f"methods:{model}", run_py(
                "love_numbers", "run", model, *common,
                "--method", *methods, "--solver", *prof["solvers"],
                *combined))

    # 2. The approximate CMB conditions (Dahlen path, fluid models):
    # cost against accuracy, collated by cmb_report.py in the plot stage.
    if "cmb" in stages:
        for model in prof["models"]:
            if model in SOLID_MODELS:
                continue  # no fluid layer, nothing to approximate
            stage(f"cmb:{model}", run_py(
                "love_numbers", "run", model, *common,
                "--cmb", *prof["cmb"], *combined))

    # 3. The field benchmark (Eulerian pair); runs when the profile's
    # field flag is set (both profiles set it) or the stage is named
    # in --stages.
    if "field" in stages and (prof["field"] or args.stages):
        for model in prof["models"]:
            stage(f"field:{model}", run_py(
                "love_numbers", "run", model, *common, "--field-only",
                "--method", "dahlen", "gauged"))

    # 4. The relabelled mapped runs: the same spherical physics from
    # laterally mapped coordinates, against the same references.
    if "mapped" in stages:
        for model in prof["models"]:
            if model not in MAPPED_MODELS:
                continue
            methods = ["referential"]
            if model in SLIP_MODELS:
                methods.append("slip_broken")
            stage(f"mapped:{model}", run_py(
                "love_numbers", "run", model, *common,
                "--method", *methods, "--map", prof["map_amplitude"]))

    # 5. The solver-level change-of-variables identity, strict on the
    # welded models (solids and the gauged fluid core).
    if "identity" in stages:
        for model in prof["models"]:
            if model not in IDENTITY_MODELS:
                continue
            for h in prof["h"]:
                case = loves / model / f"h{h:g}" / "case.json"
                if not case.exists() and not args.dry_run:
                    stage(f"identity:{model}:h{h:g}", False)
                    continue
                stage(f"identity:{model}:h{h:g}", sh(
                    [mpiexec, "-np", prof["np"],
                     programs / "relabelled_identity", "-c", case,
                     "-o", max(prof["orders"]),
                     "-map", prof["map_amplitude"]],
                    log_file=out / "relabelling" /
                    f"identity_{model}_h{h:g}.txt"))
                # The slipping-interface leg (informational, with the
                # broken-block and kernel probes) on the slip models.
                if model in SLIP_MODELS:
                    stage(f"identity:{model}:h{h:g}:slip", sh(
                        [mpiexec, "-np", prof["np"],
                         programs / "relabelled_identity", "-c", case,
                         "-o", max(prof["orders"]),
                         "-map", prof["map_amplitude"],
                         "-method", "slip_broken"],
                        log_file=out / "relabelling" /
                        f"identity_{model}_slip_h{h:g}.txt"))

    # 6. The independently meshed aspherical body (single solid layer).
    # The stock mesh serves at scale one; a finer profile regenerates it.
    if "aspherical" in stages:
        scale = prof["aspherical_scale"]
        if scale == 1.0:
            mesh = programs.parent.parent / "data" / \
                "aspherical_buffer_3d.mesh"
        else:
            mesh = out / "relabelling" / \
                f"aspherical_buffer_3d_s{scale:g}.mesh"
            if not mesh.exists():
                stage("aspherical:mesh", sh(
                    [py, HERE.parent / "meshes" / "aspherical_body.py",
                     "--dim", 3, "--buffer", "--scale", scale,
                     "--name", f"_s{scale:g}",
                     "--out", out / "relabelling"]))
        for model in prof["models"]:
            if model not in ASPHERICAL_MODELS:
                continue
            h = min(prof["h"])
            case = loves / model / f"h{h:g}" / "case.json"
            if not case.exists() and not args.dry_run:
                stage(f"aspherical:{model}", False)
                continue
            stage(f"aspherical:{model}", sh(
                [mpiexec, "-np", prof["np"],
                 programs / "aspherical_reference", "-m", mesh,
                 "-c", case, "-o", max(prof["orders"]),
                 "-lmax", prof["aspherical_lmax"],
                 "-out", out / "relabelling" /
                 f"aspherical_{model}_o{max(prof['orders'])}.json"],
                log_file=out / "relabelling" /
                f"aspherical_{model}.txt"))

    # 7. The degree-0 interface-shift perturbation check. The base runs
    # first through run.py (a no-op when the methods stage covered it).
    if "perturbation" in stages:
        for model, methods in PERTURBATION.items():
            if model not in prof["models"]:
                continue
            h = max(prof["h"])
            ok = run_py("love_numbers", "run", model, "--h", h,
                        "--order", max(prof["orders"]),
                        "--np", prof["np"], "--lmax", prof["pert_lmax"],
                        "--dtn-degree", prof["dtn_degree"],
                        "--out", loves, "--method", *methods)
            stage(f"perturbation:{model}", ok and run_py(
                "perturbation", "perturbation_check",
                loves / model / f"h{h:g}",
                "--out", out / "perturbation" / model,
                "--eps", *prof["shift_eps"], "--method", *methods,
                "--order", max(prof["orders"]),
                "--lmax", prof["pert_lmax"], "--np", prof["np"]))

    # 8. Weak scaling: a couple of degrees per rung, the timings land in
    # the results files and the summary below.
    if "scaling" in stages and prof["scaling"]:
        for np_s, h_s in prof["scaling"]:
            stage(f"scaling:np{np_s}", run_py(
                "love_numbers", "run", "fluid_core", "--h", h_s,
                "--order", max(prof["orders"]), "--np", np_s,
                "--lmax", 2, "--dtn-degree", prof["dtn_degree"],
                "--out", out / "scaling",
                *(["--partition"] if args.partition else [])))

    # 9. The viscoelastic histories (opt-in): per model a case (the Love
    # tree's when it has one), the Laplace reference and one stepper's
    # run; compared in the plot stage.
    ve = prof["viscoelastic"]
    ve_root = out / "viscoelastic"

    def ve_paths(model: str) -> tuple[Path, Path, Path]:
        tag = "_".join(ve["models"][model])
        d = ve_root / model
        return (d / f"laplace_tau{tag}.json",
                d / f"results_o{ve['order']}_{ve['scheme']}_tau{tag}.json",
                d)

    if "viscoelastic" in stages:
        for model, taus in ve["models"].items():
            ref, res, d = ve_paths(model)
            if not args.dry_run:
                d.mkdir(parents=True, exist_ok=True)
            case = loves / model / f"h{ve['h']:g}" / "case.json"
            ok = True
            if not case.exists():
                case = d / f"h{ve['h']:g}" / "case.json"
                if not case.exists():
                    ok = sh([py, HERE / "common" / "make_case.py", model,
                             "--out", case.parent, "--h", ve["h"],
                             "--lmax", ve["lmax"]])
            if ok and not ref.exists():
                ok = sh([py, HERE / "viscoelastic" / "love" / "laplace_reference.py",
                         model, "--tau", *taus, "--lmax", ve["lmax"],
                         "--out", ref], log_file=d / "laplace_log.txt")
            if ok and not res.exists():
                ok = sh([mpiexec, "-np", prof["np"],
                         programs / "viscoelastic_love", "-c", case,
                         "-o", ve["order"], "-lmax", ve["lmax"],
                         "-deg", max(prof["dtn_degree"], ve["lmax"]),
                         "-tau", ",".join(taus), "-scheme", ve["scheme"],
                         "-out", res], log_file=res.with_suffix(".txt"))
            stage(f"viscoelastic:{model}", ok)

    # 9b. The box viscoelastic studies (opt-in): every study of
    # viscoelastic/box/study.py at the campaign's profile, figures
    # included (the profile's 3-D ladder; its own rank counts).
    if "viscoelastic_box" in stages:
        stage("viscoelastic_box", sh(
            [py, HERE / "viscoelastic" / "box" / "study.py", "all",
             "--profile", args.profile, "--out", out / "viscoelastic_box",
             "--programs", programs, "--mpiexec", mpiexec, *dry],
            log_file=None if args.dry_run
            else out / "viscoelastic_box_log.txt"))

    # 9c. The sphere viscoelastic studies (opt-in), likewise.
    if "viscoelastic_sphere" in stages:
        stage("viscoelastic_sphere", sh(
            [py, HERE / "viscoelastic" / "sphere" / "study.py", "all",
             "--profile", args.profile, "--out",
             out / "viscoelastic_sphere", "--programs", programs,
             "--mpiexec", mpiexec, *dry],
            log_file=None if args.dry_run
            else out / "viscoelastic_sphere_log.txt"))

    # 10. Plots and the summaries, each skipped quietly when its stage
    # left nothing to draw.
    if "plot" in stages and not args.dry_run:
        if any(loves.rglob("results_*.json")):
            stage("plot", sh([py, HERE / "love_numbers" / "plot.py",
                              loves]))
        if "cmb" in stages:
            stage("cmb_report", sh(
                [py, HERE / "love_numbers" / "cmb_report.py", loves]))
        for model in ve["models"]:
            ref, res, d = ve_paths(model)
            if ref.exists() and res.exists():
                stage(f"plot:viscoelastic:{model}", sh(
                    [py, HERE / "viscoelastic" / "love" / "compare.py", res,
                     "--reference", ref, "--out", d]))
        if any((out / "relabelling").glob("aspherical*.json")):
            stage("plot:relabelling", sh(
                [py, HERE / "relabelling" / "plot.py",
                 out / "relabelling"]))
        for model in PERTURBATION:
            root = out / "perturbation" / model
            if model in prof["models"] and any(root.glob("results_*")):
                stage(f"plot:perturbation:{model}", sh(
                    [py, HERE / "perturbation" / "plot.py",
                     loves / model / f"h{max(prof['h']):g}",
                     "--out", root, "--order", str(max(prof["orders"])),
                     "--method", *PERTURBATION[model]]))
        rows = ["| ranks | h | elements | unknowns | setup s | "
                "s / solve | iterations |",
                "|---|---|---|---|---|---|---|"]
        for np_s, h_s in prof.get("scaling", []):
            path = (out / "scaling" / "fluid_core" / f"h{h_s:g}" /
                    f"results_o{max(prof['orders'])}.json")
            if not path.exists():
                continue
            r = json.loads(path.read_text())
            # The scaling stage never runs combined, but a combined
            # file in its place reports its one load solve, so marked.
            cost = load_cost(r)
            rows.append(
                f"| {r['ranks']} | {h_s:g} | {r['elements']} | "
                f"{r['displacement_unknowns'] + r['potential_unknowns']} "
                f"| {r['setup_seconds']:.1f} | "
                f"{cost.seconds:.2f}"
                f"{' (combined)' if cost.combined else ''} | "
                f"{cost.iterations:.0f} |")
        if len(rows) > 2:
            (out / "scaling_summary.md").write_text("\n".join(rows) + "\n")
            log.append("[scaling] summary in scaling_summary.md")

    log += ["", ("FAILED: " + ", ".join(failures)) if failures
            else "all stages ok", ""]
    if not args.dry_run:
        # Appended, not overwritten: an interrupted campaign resumes,
        # and the log keeps every invocation's record.
        with (out / "campaign_log.md").open("a") as f:
            f.write("\n".join(log) + "\n")
        print(f"\n{log[-2]}\nlog: {out / 'campaign_log.md'}")
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
