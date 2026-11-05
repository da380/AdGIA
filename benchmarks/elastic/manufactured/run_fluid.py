"""Run the fluid-core manufactured benchmark over element sizes, orders and
level counts.

    run_fluid [--h 0.3 0.2 ...] [--order 2 3] [--levels-at 6]
              [--ladder-h 0.3] [--ladder-levels 3 4 6 8 12] [--np 4]
              [--out runs_fluid]

For each element size h a two-layer ball (fluid core, solid shell, buffer
to 1.2) is built with meshes/layered_earth.py at the scale that puts the
element size h on the interfaces (the canned sizing, common.SIZING, has
0.135 there: scale h / 0.135), unless it exists; for each order the
driver manufactured_fluid_benchmark solves the problem of
fields_fluid.hpp with the level border at a level count proportional to
1/h (--levels-at at h = 0.3: 6, 9, 12, 18 at h = 0.3, 0.2, 0.15, 0.1 by
default) and writes <out>/<route>/o<p>/h<h>_L<levels>.json. The levels
ladder runs the same at a fixed size (--ladder-h) over --ladder-levels.
Two tables are printed and written beside the results: the h ladder
(table_h.txt) with the observed rates log(e1/e2)/log(h1/h2) between
consecutive sizes, and the levels ladder (table_levels_h<h>.txt) with
the rates log(e1/e2)/log(n2/n1) in the level count n. Existing results
are kept unless --force is given.

The class refuses a level count whose grid has an element without a
fluid quadrature point (its volume integrals are binned quadrature
points): on the h = 0.3 mesh beyond 6 levels at order 2 and beyond 12
at order 3. Such a run is recorded as refused and the ladder goes on.

Columns: relative L2 errors of u and of the stress over the shell, of phi
over the ball and over the core; the border's nodal U, s, nu (largest
error over the largest exact value) and the wall multiplier lambda; vol,
the largest relative deviation of the border's element volumes (binned
fluid quadrature points) from the analytic shells between its levels.
"""
from __future__ import annotations

import argparse
import json
import math
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parent.parent.parent
MESH_DIR = REPO / "meshes"
MESH_SCRIPT = MESH_DIR / "layered_earth.py"
H_REF = 0.3  # the size at which --levels-at applies


def interface_size() -> float:
    """The canned interface element size of the 3-D layered meshes."""
    sys.path.insert(0, str(MESH_DIR))
    try:
        from common import SIZING
    finally:
        sys.path.pop(0)
    return SIZING[3][0]


def mesh_for(h: float, out: Path, buffer: float, verbose: bool) -> Path:
    scale = h / interface_size()
    suffix = f"_h{h:g}"
    path = out / "meshes" / f"earth_fluidcore_3d{suffix}.msh"
    if path.exists():
        return path
    (out / "meshes").mkdir(parents=True, exist_ok=True)
    cmd = [sys.executable, str(MESH_SCRIPT), "--layers", "2", "--dim", "3",
           "--scale", f"{scale:.6g}", "--buffer", str(buffer),
           "--name", suffix, "--out", str(out / "meshes")]
    if verbose:
        cmd.append("--verbose")
    print("  $ " + " ".join(cmd))
    subprocess.run(cmd, check=True, cwd=MESH_DIR)
    return path


def levels_for(h: float, args) -> int:
    return max(3, int(round(args.levels_at * (H_REF / h) ** args.levels_power)))


REFUSAL = "holds no fluid quadrature point"


def run_one(args, mesh: Path, order: int, levels: int,
            result: Path) -> dict | None:
    """The driver's results, or None when the class refuses the level
    count (a level element without a fluid quadrature point)."""
    log = result.with_suffix(".log")
    if not args.force:
        if result.exists():
            print(f"  order {order}, {levels} levels: {result} exists, "
                  "skipped")
            return json.loads(result.read_text())
        if log.exists() and REFUSAL in log.read_text():
            print(f"  order {order}, {levels} levels: refused before "
                  f"({log})")
            return None
    result.parent.mkdir(parents=True, exist_ok=True)
    cmd = [args.mpiexec, "-np", str(args.np),
           str(Path(args.programs) / "manufactured_fluid_benchmark"),
           "-m", str(mesh), "-o", str(order), "-levels", str(levels),
           "-rt", str(args.rel_tol), "-deg", str(args.dtn_degree),
           "-out", str(result)]
    if args.projected_levels:
        cmd.append("-projected-levels")
    if args.exact_hessian:
        cmd.append("-exact-hessian")
    if args.interpolant:
        cmd.append("-interpolant")
    print("  $ " + " ".join(cmd))
    with open(log, "w") as f:
        done = subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT)
    if done.returncode != 0:
        if REFUSAL in log.read_text():
            print(f"  refused: a level element holds no fluid quadrature "
                  f"point ({log})")
            return None
        raise subprocess.CalledProcessError(done.returncode, cmd)
    return json.loads(result.read_text())


COLUMNS = [("u_l2", "u"), ("stress_l2", "stress"), ("phi_l2", "phi"),
           ("phi_core_l2", "phi core"), ("U_max", "U"), ("s_max", "s"),
           ("nu_max", "nu"), ("lambda", "lambda")]


def table(rows: list[tuple[int, float, int, dict | None]], x_name: str,
          rate_of) -> str:
    """rows: (order, x, levels, result or None if refused) in ladder
    order per order; rate_of(x1, x2) gives the denominator log of the
    rate between consecutive rows."""
    head = f"{'order':>5s} {x_name:>6s} {'levels':>6s} {'elements':>9s} " \
           f"{'dofs_u':>8s}"
    for _, name in COLUMNS:
        head += f" {name:>9s} {'rate':>5s}"
    head += f" {'vol':>8s} {'its':>4s} {'s':>7s}"
    lines = [head]
    prev = None
    for order, x, levels, r in rows:
        if prev is not None and prev[0] != order:
            prev = None
        if r is None:
            lines.append(f"{order:5d} {x:6g} {levels:6d}  refused: a level "
                         "element holds no fluid quadrature point")
            prev = None
            continue
        line = f"{order:5d} {x:6g} {r['levels']:6d} {r['elements']:9d} " \
               f"{r['dofs_u']:8d}"
        for key, _ in COLUMNS:
            v = r[key]
            if prev is None or v <= 0 or prev[2][key] <= 0:
                rate = "    -"
            else:
                rate = f"{math.log(prev[2][key] / v) / rate_of(prev[1], x):5.2f}"
            line += f" {v:9.2e} {rate}"
        line += f" {r['vol_max']:8.2e} {r['iterations']:4d} " \
                f"{r['solve_seconds']:7.1f}"
        lines.append(line)
        prev = (order, x, r)
    return "\n".join(lines)


def main(argv=None) -> None:
    p = argparse.ArgumentParser(
        description=__doc__.split("\n\n")[0],
        formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--h", type=float, nargs="*", default=[0.3, 0.2, 0.15, 0.1],
                   help="element sizes on the interfaces (none: skip the "
                        "h ladder)")
    p.add_argument("--order", type=int, nargs="+", default=[2, 3])
    p.add_argument("--levels-at", type=int, default=6,
                   help="level count at h = 0.3 of the h ladder (the "
                        "largest the order-2 border admits on that mesh)")
    p.add_argument("--levels-power", type=float, default=1.0,
                   help="the h ladder's level count scales as h^-power")
    p.add_argument("--ladder-h", type=float, default=0.3,
                   help="element size of the levels ladder")
    p.add_argument("--ladder-levels", type=int, nargs="*",
                   default=[3, 4, 6, 8, 12],
                   help="level counts of the levels ladder (none: skip it)")
    p.add_argument("--buffer", type=float, default=0.2,
                   help="buffer thickness in surface radii (to 1.2: the "
                        "fields' outer radius)")
    p.add_argument("--projected-levels", action="store_true",
                   help="the border reads the projected Phi0, not the "
                        "analytic one")
    p.add_argument("--exact-hessian", action="store_true",
                   help="gravity term with the exact Hessian of Phi0")
    p.add_argument("--interpolant", action="store_true",
                   help="gravity term by the nodal interpolant")
    p.add_argument("--np", type=int, default=4)
    p.add_argument("--rel-tol", type=float, default=1e-11)
    p.add_argument("--dtn-degree", type=int, default=12)
    p.add_argument("--out", type=Path, default=Path("runs_fluid"))
    p.add_argument("--programs", default=".",
                   help="directory of the drivers")
    p.add_argument("--mpiexec", default="mpiexec")
    p.add_argument("--force", action="store_true")
    p.add_argument("--verbose", action="store_true")
    args = p.parse_args(argv)
    args.out = args.out.resolve()

    tag = "projected" if args.projected_levels else "exact"
    if args.exact_hessian:
        tag += "_hess"
    if args.interpolant:
        tag += "_interp"
    out = args.out / tag
    out.mkdir(parents=True, exist_ok=True)

    if args.h:
        rows = []
        for order in args.order:
            for h in args.h:
                levels = levels_for(h, args)
                print(f"h = {h:g}, order {order}, {levels} levels")
                mesh = mesh_for(h, args.out, args.buffer, args.verbose)
                result = out / f"o{order}" / f"h{h:g}_L{levels}.json"
                rows.append((order, h, levels,
                             run_one(args, mesh, order, levels, result)))
        text = table(rows, "h", lambda h1, h2: math.log(h1 / h2))
        print(text)
        (out / "table_h.txt").write_text(text + "\n")

    if args.ladder_levels:
        rows = []
        mesh = mesh_for(args.ladder_h, args.out, args.buffer, args.verbose)
        for order in args.order:
            for levels in args.ladder_levels:
                print(f"h = {args.ladder_h:g}, order {order}, {levels} levels")
                result = out / f"o{order}" / f"h{args.ladder_h:g}_L{levels}.json"
                rows.append((order, levels, levels,
                             run_one(args, mesh, order, levels, result)))
        text = table(rows, "levels", lambda n1, n2: math.log(n2 / n1))
        print(text)
        (out / f"table_levels_h{args.ladder_h:g}.txt").write_text(text + "\n")


if __name__ == "__main__":
    main()
