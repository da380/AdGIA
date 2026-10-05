"""Run the Love-number benchmark over element sizes and orders.

For each model named (`all` for every one), each element size `--h` is a case (made with make_case.py
if its directory does not hold one) and each order `--order`, method
`--method`, Eulerian solver `--solver` and CMB treatment `--cmb` a run of
love_benchmark on it under MPI, and with `--field` a run of field_benchmark
as well (Eulerian methods only), which compares the response to a cap load
as fields:

  <out>/<model>/h<h>/                 the case: mesh, fields, reference
  <out>/<model>/h<h>/results_o<p><s>.json
  <out>/<model>/h<h>/log_o<p><s>.txt
  <out>/<model>/h<h>/field_o<p><s>.json   with --field
  <out>/<model>/h<h>/paraview_o<p><s>/    with --field --paraview
  <out>/<model>/h<h>/parts_<np>/          with --partition

The suffix <s> is empty for the Dahlen method with the full CMB
treatment and block MINRES; otherwise it names the method, the CMB
treatment, the map amplitude (_map<A>), the Schur solver (_schur) and,
for the Love-number results, --combined, in that order.

A run whose results file exists is skipped unless `--force` is given, and a
case that exists is kept unless `--remake` is, so a sweep can be extended or
resumed. plot.py reads the tree.

    python run.py homogeneous --h 0.3 0.2 0.15 --order 2 --np 8
    python run.py inner_core --h 0.1 0.07 --order 2 3 --np 128 --lmax 8
    python run.py earth_like --h 0.2 --order 3 --field --paraview
    python run.py all --h 0.2 --order 2 3 --field

The build makes a launcher of this script, <build>/benchmarks/love_numbers/
run, which passes the drivers and the MPI launcher of the build and is
meant to be started there, so that `runs` is in the build tree:

    cd <build>/benchmarks/love_numbers
    ./run all --h 0.2 --order 2 3 --field
    ./plot runs

Started directly, the script looks for the drivers in a build tree of the
repository; the launcher is then `--mpiexec`, else the environment's
MPIEXEC, else `mpiexec`, and must belong to the MPI of the drivers.
`--launcher-args` are passed to it before the program (binding, host
files). `--dry-run` prints the commands without running anything.
`--combined` runs love_benchmark with one solve per forcing for all the
degrees (README.md), its results carrying the suffix `_combined`.
"""
from __future__ import annotations

import argparse
import os
import shlex
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "common"))

import models  # noqa: E402
from drivers import find_programs, run  # noqa: E402


def main() -> None:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("model", nargs="+",
                   choices=sorted(models.MODELS) + ["all"],
                   help="the models, or `all`")
    p.add_argument("--h", type=float, nargs="+", default=[0.2],
                   help="element sizes on the interfaces, one case each")
    p.add_argument("--order", type=int, nargs="+", default=[2],
                   help="finite element orders, one run each per case")
    p.add_argument("--np", type=int, default=os.cpu_count() // 2 or 1,
                   help="number of MPI ranks")
    p.add_argument("--lmax", type=int, default=6, help="highest degree")
    p.add_argument("--lmin", type=int, default=0, help="lowest degree")
    p.add_argument("--dtn-degree", type=int, default=16)
    p.add_argument("--rel-tol", type=float, default=1e-10)
    p.add_argument("--method", nargs="+",
                   choices=("dahlen", "gauged", "referential", "slip",
                            "slip_broken"),
                   default=["dahlen"],
                   help="the formulations to run, one results file each: "
                        "dahlen (Eulerian, fluid eliminated), gauged "
                        "(Eulerian, gauged fluid), referential (welded "
                        "gauged referential), slip and slip_broken (the "
                        "slipping interface, single-valued or broken "
                        "zeta); non-dahlen results carry the method as a "
                        "suffix")
    p.add_argument("--fluid", choices=("dahlen", "gauged"), default=None,
                   help="deprecated alias: --fluid gauged adds gauged to "
                        "--method")
    p.add_argument("--map", type=float, default=0.0, metavar="A",
                   help="amplitude of the interior relabelling: the 3-D "
                        "relabelled benchmark (relabelling.hpp) — the same "
                        "spherical physics described from laterally mapped "
                        "coordinates, methods referential and slip_broken "
                        "only; results "
                        "carry a _map<A> suffix")
    p.add_argument("--cmb", nargs="+",
                   choices=("full", "nomass", "uniform", "winkler"),
                   default=["full"],
                   help="Dahlen-path fluid-interface treatments to run, "
                        "one results file each (doc/self_gravitation.md); "
                        "non-full results carry the choice as a suffix, "
                        "and cmb_report.py compares their cost and "
                        "accuracy")
    p.add_argument("--solver", type=int, nargs="+", default=[1],
                   choices=(0, 1),
                   help="Eulerian linear solvers to run, one results file "
                        "each: 0 Schur-complement CG (suffix _schur), 1 "
                        "block MINRES; the referential methods have their "
                        "own solver and ignore the choice")
    p.add_argument("--combined", action="store_true",
                   help="solve the loads of all the degrees together, and "
                        "the tides, one solve each (love_benchmark "
                        "-combined); the results carry the suffix "
                        "_combined")
    p.add_argument("--buffer", type=float, default=0.2,
                   help="thickness of the buffer shell over the radius")
    p.add_argument("--angular", type=float, default=0.3,
                   help="largest element size on an interface over its radius")
    p.add_argument("--thin", type=float, default=4.0,
                   help="largest element size on an interface over the "
                        "thickness of the layers it bounds")
    p.add_argument("--out", type=Path, default=Path("runs"), metavar="DIR",
                   help="root of the results tree")
    p.add_argument("--programs", type=Path, default=None, metavar="DIR",
                   help="directory of the drivers (default: found in a "
                        "build tree)")
    p.add_argument("--field", action="store_true",
                   help="run field_benchmark as well")
    p.add_argument("--field-only", action="store_true",
                   help="run field_benchmark alone")
    p.add_argument("--field-lmax", type=int, default=8,
                   help="highest degree of the cap load")
    p.add_argument("--partition", action="store_true",
                   help="partition each case for the ranks beforehand, so "
                        "that no rank reads the whole mesh")
    p.add_argument("--paraview", action="store_true",
                   help="with --field, write the fields for ParaView")
    p.add_argument("--mpiexec", default=os.environ.get("MPIEXEC", "mpiexec"),
                   help="the MPI launcher")
    p.add_argument("--launcher-args", default="",
                   help="further arguments of the launcher, as one string")
    p.add_argument("--program-args", default="",
                   help="further arguments of the drivers, as one string")
    p.add_argument("--force", action="store_true",
                   help="run again where results exist")
    p.add_argument("--remake", action="store_true",
                   help="make the cases again where they exist")
    p.add_argument("--dry-run", action="store_true",
                   help="print the commands without running them")
    args = p.parse_args()

    programs = Path(".") if args.dry_run and args.programs is None \
        else find_programs(args.programs)
    launch = [args.mpiexec, "-np", str(args.np),
              *shlex.split(args.launcher_args)]
    failures = []
    names = list(models.MODELS) if "all" in args.model else args.model
    for name, h in ((name, h) for name in names for h in args.h):
        case = args.out / name / f"h{h:g}"
        print(f"{name}, h = {h:g}: {case}", flush=True)
        if args.remake or not (case / "case.json").exists():
            ok = run([sys.executable,
                      str(HERE.parent / "common" / "make_case.py"), name,
                      "--h", f"{h:g}", "--buffer", f"{args.buffer:g}",
                      "--angular", f"{args.angular:g}",
                      "--thin", f"{args.thin:g}",
                      "--lmax", str(max(args.lmax, args.field_lmax, 10)),
                      "--out", str(case)],
                     log=None, dry_run=args.dry_run)
            if not ok:
                failures.append(f"{case}: make_case.py")
                continue
        parts = case / f"parts_{args.np}"
        if args.partition and (args.remake or not parts.exists()):
            ok = run([str(programs / "partition_case"),
                      "-c", str(case / "case.json"), "-np", str(args.np)],
                     log=None, dry_run=args.dry_run)
            if not ok:
                failures.append(f"{case}: partition_case")
                continue
        methods = list(args.method)
        if args.fluid == "gauged" and "gauged" not in methods:
            methods.append("gauged")
        for order, method, solver, cmb in ((o, m, s, c) for o in args.order
                                           for m in methods
                                           for s in args.solver
                                           for c in args.cmb):
            if method not in ("dahlen", "gauged") and solver != args.solver[0]:
                continue  # the referential solvers ignore the choice
            if method != "dahlen" and cmb != args.cmb[0]:
                continue  # the treatments are the Dahlen path's: one
                # run of the other methods per sweep, without them
            mapped = args.map != 0.0
            if mapped and method not in ("referential", "slip_broken"):
                continue  # the mapped benchmark runs the mapped methods
            suffix = "" if method == "dahlen" else f"_{method}"
            if cmb != "full" and method == "dahlen":
                suffix += f"_{cmb}"
            if mapped:
                suffix += f"_map{args.map:g}"
            if solver == 0:
                suffix += "_schur"
            love_suffix = suffix + ("_combined" if args.combined else "")
            common = ["-c", str(case / "case.json"), "-o", str(order),
                      "-rt", f"{args.rel_tol:g}", "-s", str(solver),
                      "-method", method,
                      *(["-map", f"{args.map:g}"] if mapped else []),
                      *(["-cmb", cmb]
                        if cmb != "full" and method == "dahlen"
                        else []),
                      *shlex.split(args.program_args)]
            jobs = []
            if not args.field_only:
                jobs.append((
                    case / f"results_o{order}{love_suffix}.json",
                    case / f"log_o{order}{love_suffix}.txt",
                    [str(programs / "love_benchmark"), *common,
                     "-lmin", str(args.lmin), "-lmax", str(args.lmax),
                     "-deg", str(max(args.dtn_degree, args.lmax)),
                     *(["-combined"] if args.combined else [])]))
            if (args.field or args.field_only) and method in ("dahlen",
                                                              "gauged"):
                extra = (["-pv", str(case / f"paraview_o{order}{suffix}")]
                         if args.paraview else [])
                jobs.append((
                    case / f"field_o{order}{suffix}.json",
                    case / f"field_log_o{order}{suffix}.txt",
                    [str(programs / "field_benchmark"), *common,
                     "-lmax", str(args.field_lmax),
                     "-deg", str(max(args.dtn_degree, args.field_lmax)),
                     *extra]))
            for results, log, command in jobs:
                if results.exists() and not args.force:
                    print(f"  order {order}: {results} exists, skipped")
                    continue
                ok = run([*launch, *command, "-out", str(results)], log=log,
                         dry_run=args.dry_run)
                if not ok:
                    results.unlink(missing_ok=True)
                    failures.append(f"{results}")
    if failures:
        raise SystemExit("failed: " + "; ".join(failures))


if __name__ == "__main__":
    main()
