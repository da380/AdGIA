"""Rung 0 of the disc family: run the driver and compare with the exact
reference (README.md; doc/planning/no_gravity_analytics.md).

For each requested (order, refinements) the driver solves every degree
with the welded and the slipping method; this script compares

  - each method against `disc_reference` (absolute responses, and the
    fluid pressure at l = 0),
  - the two methods against each other (the machinery identity:
    without gravity every slip is removable, so welded == slip exactly
    and any gap is implementation error),

and prints one table per run. At l = 1 the FE solution and the
reference fix the translation differently, so the comparison uses the
gauge-invariant combinations U + V per circle. Exit status is nonzero
when a tolerance fails: reference agreement must improve down the
ladder (no fixed tolerance), while the welded-slip identity must hold
near solver tolerance on every rung.
"""
import argparse
import json
import math
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from disc_reference import DiscModel, solve  # noqa: E402


def run_driver(programs: str, mpiexec: str, np: int, out: Path,
               method: str, order: int, ref: int, lmax: int,
               extra: list[str]) -> dict:
    if not out.exists():
        cmd = []
        if np > 1:
            cmd += [mpiexec, "-np", str(np)]
        cmd += [str(Path(programs) / "disc_benchmark"), "-method", method,
                "-o", str(order), "-ref", str(ref), "-lmax", str(lmax),
                "-out", str(out)] + extra
        print("  $", " ".join(cmd), flush=True)
        subprocess.run(cmd, check=True)
    return json.loads(out.read_text())


def compare(d: dict, quiet: bool = False) -> dict[int, dict[str, float]]:
    """Relative errors against the exact reference, per degree."""
    m = DiscModel(lam=d["kappa_s"] - d["mu_s"], mu=d["mu_s"],
                  kappa_f=d["kappa_f"], c=d["r_cmb"], a=1.0,
                  P0=d.get("P0", 0.0))
    errors: dict[int, dict[str, float]] = {}
    for e in d["degrees"]:
        l = e["l"]
        s = solve(m, l)
        row: dict[str, float] = {}
        if l == 1:
            for where in ("a", "c"):
                fe = e[f"ur_{where}"] + e[f"ut_{where}"]
                ex = s.responses[f"ur_{where}"] + s.responses[f"ut_{where}"]
                row[f"U+V_{where}"] = abs(fe - ex) / max(abs(ex), 1e-300)
        else:
            for key in ("ur_a", "ur_c") + (() if l == 0 else ("ut_a",
                                                              "ut_c")):
                ex = s.responses[key]
                row[key] = abs(e[key] - ex) / max(abs(ex), 1e-300)
            if l == 0:
                row["p1"] = abs(e["p1"] - s.p1) / abs(s.p1)
        errors[l] = row
        if not quiet:
            worst = max(row.values())
            parts = "  ".join(f"{k} {v:.2e}" for k, v in row.items())
            print(f"    l={l}: {parts}   (worst {worst:.2e})")
    return errors


def identity(dw: dict, ds: dict) -> float:
    """Worst welded-vs-slip relative gap over degrees and responses
    (translation-invariant combinations at l = 1)."""
    worst = 0.0
    for ew, es in zip(dw["degrees"], ds["degrees"]):
        l = ew["l"]
        keys = (("ur_a", "ur_c") if l == 0 else
                () if l == 1 else ("ur_a", "ut_a", "ur_c", "ut_c"))
        for k in keys:
            scale = max(abs(ew[k]), abs(es[k]), 1e-12)
            worst = max(worst, abs(ew[k] - es[k]) / scale)
        if l == 1:
            for where in ("a", "c"):
                w = ew[f"ur_{where}"] + ew[f"ut_{where}"]
                s = es[f"ur_{where}"] + es[f"ut_{where}"]
                worst = max(worst, abs(w - s) / max(abs(w), abs(s), 1e-12))
    return worst


def shift_mode(args, extra) -> bool:
    """The shift leg: solve at +/- eps through the mapped assembly on
    the fixed mesh and compare the central difference of the surface
    responses with the exact interface derivative of the reference.
    This isolates the mapped volume operators (welded) and, on top of
    them, the mapped constraint row (slip) under a jumping F."""
    eps = args.shift
    ok = True
    for order in args.orders:
        for ref in args.refinements:
            print(f"-- shift {eps}, order {order}, refinements {ref}")
            for method in ("welded", "slip"):
                runs = {}
                for sgn, tag in ((+1, "+"), (-1, "-")):
                    out = (Path(args.out) /
                           f"results_{method}_o{order}r{ref}"
                           f"_shift{tag}{eps}.json")
                    runs[sgn] = run_driver(
                        args.programs, args.mpiexec, args.np, out, method,
                        order, ref, args.lmax,
                        extra + ["-shift", str(sgn * eps)])
                d = runs[+1]
                m = DiscModel(lam=d["kappa_s"] - d["mu_s"], mu=d["mu_s"],
                              kappa_f=d["kappa_f"], c=d["r_cmb"], a=1.0,
                              P0=d.get("P0", 0.0))
                print(f"  {method}: central difference vs exact d/dc")
                for ep, em in zip(runs[+1]["degrees"], runs[-1]["degrees"]):
                    l = ep["l"]
                    s = solve(m, l)
                    if l == 1:
                        fd = ((ep["ur_a"] + ep["ut_a"])
                              - (em["ur_a"] + em["ut_a"])) / (2 * eps)
                        ex = (s.derivatives["ur_a"]
                              + s.derivatives["ut_a"])
                        items = [("d(U+V)_a/dc", fd, ex)]
                    else:
                        items = [("dur_a/dc",
                                  (ep["ur_a"] - em["ur_a"]) / (2 * eps),
                                  s.derivatives["ur_a"])]
                        if l >= 2:
                            items.append(
                                ("dut_a/dc",
                                 (ep["ut_a"] - em["ut_a"]) / (2 * eps),
                                 s.derivatives["ut_a"]))
                    for name, fd, ex in items:
                        rel = abs(fd - ex) / max(abs(ex), 1e-12)
                        print(f"    l={l} {name:<12} FD {fd:>12.6f} "
                              f"exact {ex:>12.6f}  rel {rel:.2e}")
    return ok


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--orders", type=int, nargs="+", default=[2])
    p.add_argument("--refinements", type=int, nargs="+", default=[0, 1])
    p.add_argument("--lmax", type=int, default=4)
    p.add_argument("--np", type=int, default=1)
    p.add_argument("--out", default=".", help="where results land")
    p.add_argument("--programs", default="../bin")
    p.add_argument("--mpiexec", default="mpiexec")
    p.add_argument("--identity-tol", type=float, default=1e-4,
                   help="welded-vs-slip gap allowed on every rung")
    p.add_argument("--P0", type=float, default=0.0,
                   help="rung 1: run with this uniform surface pressure "
                        "(passed to the driver; the reference follows)")
    p.add_argument("--shift", type=float, default=0.0,
                   help="run the shift leg at +/- this eps instead of "
                        "the absolute ladder")
    p.add_argument("--program-args", default="",
                   help="extra driver options, one string")
    args = p.parse_args()
    extra = args.program_args.split()
    if args.P0:
        extra += ["-P0", str(args.P0)]
    if args.shift:
        return 0 if shift_mode(args, extra) else 1

    ok = True
    ladders: dict[tuple[int, str], list[float]] = {}
    for order in args.orders:
        for ref in args.refinements:
            print(f"-- order {order}, refinements {ref}")
            runs = {}
            for method in ("welded", "slip"):
                tag = f"_P{args.P0}" if args.P0 else ""
                out = (Path(args.out) /
                       f"results_{method}_o{order}r{ref}{tag}.json")
                runs[method] = run_driver(args.programs, args.mpiexec,
                                          args.np, out, method, order, ref,
                                          args.lmax, extra)
                print(f"  {method} vs exact:")
                errs = compare(runs[method])
                for l, row in errs.items():
                    ladders.setdefault((l, method), []).append(
                        max(row.values()))
            gap = identity(runs["welded"], runs["slip"])
            flag = "ok" if gap <= args.identity_tol else "FAIL"
            ok &= gap <= args.identity_tol
            print(f"  welded == slip identity: worst gap {gap:.2e}  {flag}")

    if min(len(v) for v in ladders.values()) >= 2:
        print("-- convergence down the ladder (worst error per degree)")
        for (l, method), v in sorted(ladders.items()):
            rate = (" rate " + "/".join(
                f"{math.log2(v[i] / v[i + 1]):.1f}"
                for i in range(len(v) - 1))) if v[-1] < v[0] else "  NOT CONVERGING"
            converged = v[-1] < v[0] or v[0] < 1e-5
            ok &= converged
            print(f"    l={l} {method:>6}: " +
                  "  ".join(f"{e:.2e}" for e in v) + rate)
    print("PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
