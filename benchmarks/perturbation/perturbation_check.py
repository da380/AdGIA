"""The degree-0 interface-shift perturbation benchmark.

One fixed spherical mesh; a family of PHYSICAL models with one interior
interface at r_k + eps, described from it by the piecewise-linear radial
map of relabelling.hpp (`love_benchmark -map-shift`). Every member of
the family is still spherical, so pyslfp solves it exactly at every
finite eps; and because the mesh is fixed, the discretisation bias
largely cancels in the finite difference, so

    [R_3D(+eps) - R_3D(-eps)] / 2 eps   vs   the same of pyslfp

compares the DERIVATIVE of the response with respect to the interface
radius, through the mapped assembly. The 1-D side is a finite difference
of pyslfp solves, not an analytic sensitivity kernel
(doc/benchmarks.tex, "The perturbation family").

The case must exist (love_numbers/run.py); its unmapped run of each
method, beside the case, is the eps = 0 point, run here if missing.

For each eps of the ladder this script builds the perturbed model's
exact radial profiles and pyslfp reference beside the base case, runs
`love_benchmark -map-shift eps -profiles ...` for each method, and
prints, per degree and Love number: the absolute agreement at each eps
and the derivative comparison, with the eps-ladder showing the O(eps^2)
remainder.

    python perturbation_check.py runs/fluid_core/h0.3 \
        --eps 0.01 0.02 --method referential slip_broken --np 8
"""
from __future__ import annotations

import argparse
import json
import shlex
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "common"))

import make_case  # noqa: E402
import models  # noqa: E402
from drivers import find_programs  # noqa: E402

#: Love numbers compared, as (results key path, reference key).
QUANTITIES = (("h", "h_load"), ("l", "l_load"), ("k", "k_load"))


def perturbed(model_name: str, eps: float):
    """The model with its shiftable interface moved by eps (units of the
    outer radius), in the benchmark's units. The interface keyword takes
    SI metres; the outer radius is unchanged, so the scaling matches the
    base case's."""
    import inspect
    builder = models.MODELS[model_name]
    keyword = models.SHIFTABLE[model_name]
    default = inspect.signature(builder).parameters[keyword].default
    radius = float(builder().skeleton.boundaries[-1])
    return models.with_pressure(
        models.scaled(builder(**{keyword: default + eps * radius})))


def prepare(root: Path, model_name: str, eps: float, lmax: int) -> Path:
    """The perturbed model's profiles and reference under `root`."""
    out = root / f"shift_{eps:+g}"
    out.mkdir(parents=True, exist_ok=True)
    model = perturbed(model_name, eps)
    if not (out / "radial_profiles.txt").exists():
        make_case.write_radial_profiles(model, out / "radial_profiles.txt")
    if not (out / "reference.json").exists():
        ref = make_case.reference(model, lmax, per_layer=17)
        (out / "reference.json").write_text(json.dumps(ref, indent=1))
    return out


def run(command: list[str], *, dry: bool) -> None:
    print("  $ " + shlex.join(command), flush=True)
    if not dry:
        subprocess.run(command, check=True)


def love_from(results: Path) -> dict[int, dict[str, float]]:
    r = json.loads(results.read_text())
    out: dict[int, dict[str, float]] = {}
    for d in r["degrees"]:
        out[d["degree"]] = {q: d["load"][q] for q, _ in QUANTITIES}
    return out


def love_reference(path: Path) -> dict[int, dict[str, float]]:
    r = json.loads(path.read_text())
    out: dict[int, dict[str, float]] = {}
    for i, l in enumerate(r["degree"]):
        out[l] = {q: r[key][i] for q, key in QUANTITIES}
    return out


def main() -> None:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("case", type=Path, help="a case directory, runs/<model>/h<h>")
    p.add_argument("--eps", type=float, nargs="+", default=[0.01, 0.02],
                   help="shift ladder, units of the outer radius; each "
                        "runs at +eps and -eps")
    p.add_argument("--method", nargs="+",
                   choices=("referential", "slip_broken"),
                   default=["referential"],
                   help="mapped methods to run")
    p.add_argument("--order", type=int, default=2)
    p.add_argument("--lmax", type=int, default=4)
    p.add_argument("--lmin", type=int, default=0)
    p.add_argument("--dtn-degree", type=int, default=16)
    p.add_argument("--np", type=int, default=8)
    p.add_argument("--interface", type=int, default=1,
                   help="which interior interface shifts (1 = innermost)")
    p.add_argument("--rel-tol", type=float, default=1e-10)
    p.add_argument("--out", type=Path, default=None, metavar="DIR",
                   help="where the shifted profiles, references and "
                        "results go (default: beside the case)")
    p.add_argument("--programs", type=Path, default=None)
    p.add_argument("--mpiexec", default="mpiexec")
    p.add_argument("--program-args", default="",
                   help="further arguments of love_benchmark, as one "
                        "string (say \"-theta 1e3 -al 16\")")
    p.add_argument("--force", action="store_true")
    p.add_argument("--dry-run", action="store_true")
    args = p.parse_args()

    if args.dry_run and not (args.case / "case.json").exists():
        model_name = args.case.parent.name  # runs/<model>/h<h>
    else:
        manifest = json.loads((args.case / "case.json").read_text())
        model_name = manifest.get("meta", {}).get("model") or \
            json.loads((args.case / "reference.json").read_text())["model"]
    if model_name not in models.SHIFTABLE:
        raise SystemExit(f"{model_name}: no shiftable interface "
                         f"(shiftable: {sorted(models.SHIFTABLE)})")
    programs = find_programs(args.programs)
    root = args.out if args.out is not None else args.case
    if not args.dry_run:
        root.mkdir(parents=True, exist_ok=True)

    # The runs: base (eps = 0 uses the unmapped method result beside the
    # case if there, else runs it), then each +-eps under `root`.
    all_eps = sorted({e for a in args.eps for e in (a, -a)} | {0.0})
    love: dict[str, dict[float, dict]] = {m: {} for m in args.method}
    refs: dict[float, dict] = {}
    for eps in all_eps:
        if args.dry_run:
            pass  # the perturbed profiles and references are real work
        elif eps == 0.0:
            refs[eps] = love_reference(args.case / "reference.json")
        else:
            sub = prepare(root, model_name, eps, args.lmax)
            refs[eps] = love_reference(sub / "reference.json")
        for m in args.method:
            tag = f"{m}_shift{eps:+g}" if eps else m
            results = args.case / f"results_o{args.order}_{tag}.json"
            if eps or (not results.exists() and root != args.case):
                results = root / f"results_o{args.order}_{tag}.json"
            if not results.exists() or args.force:
                command = [args.mpiexec, "-np", str(args.np),
                           str(programs / "love_benchmark"),
                           "-c", str(args.case / "case.json"),
                           "-o", str(args.order), "-method", m,
                           "-lmin", str(args.lmin),
                           "-lmax", str(args.lmax),
                           "-deg", str(max(args.dtn_degree, args.lmax)),
                           "-rt", f"{args.rel_tol:g}",
                           "-out", str(results),
                           *shlex.split(args.program_args)]
                if eps:
                    command += ["-map-shift", f"{eps:g}",
                                "-map-shift-interface",
                                str(args.interface),
                                "-profiles",
                                str(root / f"shift_{eps:+g}" /
                                    "radial_profiles.txt")]
                run(command, dry=args.dry_run)
            if not args.dry_run:
                love[m][eps] = love_from(results)

    if args.dry_run:
        return

    # The report: absolute agreement per eps, then the derivatives. A
    # quantity is compared only where its reference is meaningful: h
    # everywhere, l and k at l >= 2 (they vanish identically at degree
    # zero, and k' is fixed by the frame at degree one).
    def compared(l: int):
        return [q for q, _ in QUANTITIES if q == "h" or l >= 2]

    for m in args.method:
        print(f"\n=== {m} ===")
        degrees = sorted(set.intersection(
            *(set(love[m][e]) for e in all_eps))
            & set(range(0, args.lmax + 1)))
        print("absolute agreement with pyslfp per eps "
              "(|3D - ref| / |ref|, worst of the compared Love numbers "
              "per degree):")
        header = "   l " + "".join(f"  eps={e:+g}   " for e in all_eps)
        print(header)
        for l in degrees:
            if l == 1:
                continue
            row = f"{l:4d} "
            for e in all_eps:
                worst = 0.0
                for q in compared(l):
                    r = refs[e][l][q]
                    if abs(r) > 1e-10:
                        worst = max(worst,
                                    abs(love[m][e][l][q] - r) / abs(r))
                row += f"  {worst:9.2e} "
            print(row)
        print("\nderivative d(Love)/d(r_interface), 3D vs pyslfp "
              "(central differences on the eps ladder):")
        for a in sorted(args.eps):
            print(f"  eps = {a:g}:")
            print("   l        quantity     3D          1D         "
                  "rel diff")
            for l in degrees:
                if l == 1:
                    continue
                for q in compared(l):
                    d3 = (love[m][a][l][q] - love[m][-a][l][q]) / (2 * a)
                    d1 = (refs[a][l][q] - refs[-a][l][q]) / (2 * a)
                    if abs(d1) < 1e-10:
                        continue
                    print(f"{l:4d} {q:>15} {d3:11.5f} {d1:11.5f} "
                          f"{abs(d3 - d1) / abs(d1):11.2e}")


if __name__ == "__main__":
    main()
