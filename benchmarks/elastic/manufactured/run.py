"""Run the manufactured-solution benchmark over element sizes and orders.

    run [--dim 3] [--h 0.3 0.2 ...] [--order 1 2 3] [--np 4] [--out runs]

For each element size a mesh of the unit body in a buffer is built with
meshes/homogeneous_body.py (unless it exists), and for each order the
driver manufactured_benchmark solves the manufactured problem of
fields.hpp on it and writes <out>/d<dim>/h<h>/results_o<p>.json. The table
printed (and written to <out>/d<dim>/table.txt) gives, per order, the
relative L2 errors of the displacement, the stress and the potential with
their observed rates between consecutive sizes, log(e1/e2)/log(h1/h2) on
the requested sizes. Existing results are kept unless --force is given.

The expected rates on a smooth solution are p+1 for u and phi in L2 and p
for the stress, for elements of order p, down to the floor set by the
curved (order-2) geometry of the mesh and the solver tolerance.
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
MESH_SCRIPT = REPO / "meshes" / "homogeneous_body.py"


def mesh_for(dim: int, h: float, out: Path, buffer: float, gorder: int,
             verbose: bool) -> Path:
    name = f"body_d{dim}_g{gorder}_h{h:g}"
    path = out / "meshes" / f"{name}.msh"
    if path.exists():
        return path
    (out / "meshes").mkdir(parents=True, exist_ok=True)
    preset = "ball-buffer" if dim == 3 else "disc-buffer"
    cmd = [sys.executable, str(MESH_SCRIPT), "--preset", preset,
           "--dim", str(dim), "--order", str(gorder),
           "--buffer-radius", str(buffer),
           "--h", str(h), "--far-size", str(2 * h),
           "--growth-distance", "0.5",
           "--name", name, "--out", str(out / "meshes")]
    if verbose:
        cmd.append("--verbose")
    print("  $ " + " ".join(cmd))
    subprocess.run(cmd, check=True)
    return path


def run_one(args, mesh: Path, order: int, result: Path) -> dict:
    if result.exists() and not args.force:
        print(f"  order {order}: {result} exists, skipped")
        return json.loads(result.read_text())
    result.parent.mkdir(parents=True, exist_ok=True)
    cmd = [args.mpiexec, "-np", str(args.np),
           str(Path(args.programs) / "manufactured_benchmark"),
           "-m", str(mesh), "-o", str(order),
           "-ophi", str(order + args.phi_offset), "-rt", str(args.rel_tol),
           "-deg", str(args.dtn_degree), "-out", str(result)]
    if args.schur:
        cmd.append("-schur")
    if args.exact_hessian:
        cmd.append("-exact-hessian")
    if args.discrete_hessian:
        cmd.append("-discrete-hessian")
    print("  $ " + " ".join(cmd))
    log = result.with_suffix(".log")
    with open(log, "w") as f:
        subprocess.run(cmd, check=True, stdout=f, stderr=subprocess.STDOUT)
    return json.loads(result.read_text())


def table(results: dict[tuple[float, int], dict], hs: list[float],
          orders: list[int]) -> str:
    cols = ["u_l2", "stress_l2", "phi_l2"]
    lines = [f"{'order':>5s} {'h':>6s} {'elements':>9s} {'dofs_u':>8s} "
             f"{'u_l2':>9s} {'rate':>5s} {'stress':>9s} {'rate':>5s} "
             f"{'phi_l2':>9s} {'rate':>5s} {'its':>4s} {'s':>7s}"]
    for p in orders:
        prev = None
        for h in hs:
            r = results.get((h, p))
            if r is None:
                continue
            rates = []
            for c in cols:
                if prev is None or r[c] <= 0 or prev[1][c] <= 0:
                    rates.append("    -")
                else:
                    rates.append(f"{math.log(prev[1][c] / r[c]) / math.log(prev[0] / h):5.2f}")
            lines.append(f"{p:5d} {h:6g} {r['elements']:9d} {r['dofs_u']:8d} "
                         f"{r['u_l2']:9.2e} {rates[0]} {r['stress_l2']:9.2e} "
                         f"{rates[1]} {r['phi_l2']:9.2e} {rates[2]} "
                         f"{r['iterations']:4d} {r['solve_seconds']:7.1f}")
            prev = (h, r)
    return "\n".join(lines)


def main(argv=None) -> None:
    p = argparse.ArgumentParser(
        description=__doc__.split("\n\n")[1],
        formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dim", type=int, choices=(2, 3), default=3)
    p.add_argument("--h", type=float, nargs="+", default=[0.3, 0.2, 0.15],
                   help="element sizes on the surface, in body radii")
    p.add_argument("--order", type=int, nargs="+", default=[1, 2, 3])
    p.add_argument("--buffer", type=float, default=1.5,
                   help="outer radius of the buffer, in body radii")
    p.add_argument("--geometry-order", type=int, default=2,
                   help="polynomial order of the mesh geometry")
    p.add_argument("--phi-offset", type=int, default=0,
                   help="order of the potential space above the "
                        "displacement's (the background gravity is the "
                        "gradient of the projected Phi0)")
    p.add_argument("--np", type=int, default=4)
    p.add_argument("--rel-tol", type=float, default=1e-11)
    p.add_argument("--dtn-degree", type=int, default=12)
    p.add_argument("--schur", action="store_true",
                   help="Schur-complement CG instead of block MINRES")
    p.add_argument("--exact-hessian", action="store_true",
                   help="gravity term by the product rule with the exact "
                        "Hessian of Phi0 instead of the nodal interpolant")
    p.add_argument("--discrete-hessian", action="store_true",
                   help="gravity term by the product rule with the Hessian "
                        "of the discrete Phi0 (the production route)")
    p.add_argument("--out", type=Path, default=Path("runs"))
    p.add_argument("--programs", default=".",
                   help="directory of the drivers")
    p.add_argument("--mpiexec", default="mpiexec")
    p.add_argument("--force", action="store_true")
    p.add_argument("--verbose", action="store_true")
    args = p.parse_args(argv)

    tag = f"d{args.dim}_g{args.geometry_order}"
    if args.phi_offset:
        tag += f"_phi+{args.phi_offset}"
    if args.exact_hessian:
        tag += "_hess"
    if args.discrete_hessian:
        tag += "_dhess"
    out = args.out / tag
    results: dict[tuple[float, int], dict] = {}
    for h in args.h:
        print(f"dim {args.dim}, h = {h:g}")
        mesh = mesh_for(args.dim, h, args.out, args.buffer,
                        args.geometry_order, args.verbose)
        for order in args.order:
            result = out / f"h{h:g}" / f"results_o{order}.json"
            results[(h, order)] = run_one(args, mesh, order, result)
    text = table(results, args.h, args.order)
    print(text)
    out.mkdir(parents=True, exist_ok=True)
    (out / "table.txt").write_text(text + "\n")


if __name__ == "__main__":
    main()
