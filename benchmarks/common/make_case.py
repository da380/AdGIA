"""Build one case of the Love-number benchmark: the mesh, the model on it
and the reference solution.

A case is a directory holding

  case.mesh          the MFEM mesh of the body and its buffer shell
  case.json          the manifest: layers (with their fluidity), interfaces
                     (with the one-sided field values), fields, scales and G
  case.<name>.gf     rho, kappa, mu and the hydrostatic pressure p0 as L2
                     GridFunctions on the mesh
  mesh_summary.json  the mesh parameters and the mesher's summary
  gmsh/              the mesher's own files
  reference.json     the Love numbers and radial solutions of pyslfp
  reference_fields.txt  the radial solutions of the load problem on
                     Chebyshev nodes of each layer, for the driver that
                     compares fields
  radial_profiles.txt   the model's fields on Chebyshev nodes of each
                     layer, for the relabelled (mapped) runs

all from one planetmodel model in the benchmark's units (see models.py),
so that the finite-element solver and the radial solver are given the
same body. The mesh has the element size `--h` on every interface, or
`--angular` times the interface's radius or `--thin` times the thickness of
a layer it bounds where those are smaller, growing to
`--h-max` over the distance `--decay`, in units of the outer radius.

    python make_case.py homogeneous --h 0.15 --out cases/homogeneous_h0.15
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from planetmodel import Model, gravity, is_fluid
from planetmodel.mesh3d import (CappedInterfaces, MeshSpec, Shell,
                                build_layered_mesh, export_mfem)
from pyslfp.love_numbers import LoveNumbers, love_numbers, solve_degree

import models

#: The fields the finite-element solver reads: the material, and the
#: hydrostatic pressure p0 that the referential methods need (bare
#: moduli, S_e = -p0 I, the interface pressure of the slipping methods).
FIELDS = ("rho", "kappa", "mu", "p0")

#: The basename of the files of a case.
BASENAME = "case"

#: The file of radial solutions the field driver reads.
FIELDS_FILE = "reference_fields.txt"

#: The exact radial profiles of the model's fields, for the relabelled
#: (mapped) benchmark: composing a radial coefficient with a mapping
#: needs its value at |xi(x)|, which the L2 fields cannot give.
PROFILES_FILE = "radial_profiles.txt"

#: The forcings whose radial solutions are kept, for plotting.
PROFILE_FORCINGS = ("load", "tide")


def fluid_attributes(model: Model) -> list[int]:
    """The element attributes of the model's fluid layers."""
    return [i + 1 for i, layer in enumerate(model.layers) if is_fluid(layer)]


def build_mesh(model: Model, out: Path, *, h: float, h_max: float, decay: float,
               angular: float, thin: float, buffer: float, order: int,
               optimise: bool = True, verbose: bool = False) -> dict:
    """The mesh, the fields and the manifest; returns a summary."""
    spec = MeshSpec(model.geometry,
                    CappedInterfaces(h, h_max, decay, angular=angular,
                                     thin=thin),
                    dimension=3, order=order,
                    shells=[Shell(ratio=buffer, name="buffer")],
                    optimise="Netgen" if optimise else None,
                    meta={"model": model.name})
    scratch = out / "gmsh"
    scratch.mkdir(parents=True, exist_ok=True)
    built = build_layered_mesh(spec, scratch / BASENAME, verbose=verbose)
    export = export_mfem(built, out / BASENAME, model=model, fields=FIELDS)
    return {"h": h, "h_max": h_max, "decay": decay, "angular": angular,
            "thin": thin,
            "buffer": buffer, "optimised": optimise,
            "order": order, "counts": dict(export.counts),
            "summary": built.summary(),
            "validation": str(built.validation)}


def _clean(a: np.ndarray) -> list:
    """An array as a list for JSON, NaN as null."""
    return [None if not np.isfinite(x) else float(x)
            for x in np.asarray(a, dtype=float)]


def profiles(model: Model, lmax: int, *, per_layer: int) -> dict:
    """U, V and phi by radius for each degree and forcing, sampled within
    each layer; a radius on a boundary appears once for each side."""
    b = np.asarray(model.skeleton.boundaries, dtype=float)
    nudge = 1e-9 * b[-1]
    radii, layer = [], []
    for i, (lo, hi) in enumerate(zip(b[:-1], b[1:])):
        radii.append(np.linspace(lo + nudge, hi - nudge, per_layer))
        layer.append(np.full(per_layer, i + 1))
    radii, layer = np.concatenate(radii), np.concatenate(layer)
    out = {"radius": _clean(radii), "layer": [int(i) for i in layer],
           "solutions": []}
    for l in range(lmax + 1):
        for forcing in PROFILE_FORCINGS:
            if forcing == "tide" and l < 2:
                continue
            U, V, phi = solve_degree(model, l, forcing=forcing).evaluate(radii)
            out["solutions"].append({"degree": l, "forcing": forcing,
                                     "U": _clean(U), "V": _clean(V),
                                     "phi": _clean(phi)})
    return out


def write_reference_fields(model: Model, lmax: int, path: Path, *,
                           nodes: int) -> None:
    """U, V and phi of the load problem by degree, per unit coefficient of
    the surface density, on `nodes` Chebyshev points of the second kind in
    each layer, from which a polynomial interpolant recovers them within
    the layer. The end points sit just within the layer, so that each side
    of an interface has its own values. In a fluid layer U and V are not
    defined above degree zero and are written as zero.

    The file is text: `lmax`, `layers`, then for each layer a line
    `layer <attribute> <r_inner> <r_outer> <fluid> <nodes>`, the radii of
    its nodes, the gravity of the model at them, and for each degree from
    zero three lines, U, V and phi at the nodes.
    """
    b = np.asarray(model.skeleton.boundaries, dtype=float)
    nudge = 1e-9 * b[-1]
    x = np.cos(np.pi * np.arange(nodes) / (nodes - 1))[::-1]     # -1 .. 1
    radii = [0.5 * (lo + hi) + 0.5 * (hi - lo - 2.0 * nudge) * x
             for lo, hi in zip(b[:-1], b[1:])]
    solutions = [solve_degree(model, l, forcing="load").evaluate(
        np.concatenate(radii)) for l in range(lmax + 1)]
    fluid = fluid_attributes(model)

    def line(values: np.ndarray) -> str:
        return " ".join(f"{v:.17e}" for v in np.nan_to_num(values, nan=0.0))

    with path.open("w") as f:
        f.write(f"lmax {lmax}\nlayers {len(radii)}\n")
        for i, r in enumerate(radii):
            f.write(f"layer {i + 1} {b[i]:.17e} {b[i + 1]:.17e} "
                    f"{int(i + 1 in fluid)} {nodes}\n{line(r)}\n"
                    f"{line(gravity(model, r))}\n")
            part = slice(i * nodes, (i + 1) * nodes)
            for U, V, phi in solutions:
                f.write(f"{line(U[part])}\n{line(V[part])}\n"
                        f"{line(phi[part])}\n")


def write_radial_profiles(model: Model, path: Path, *,
                          nodes: int = 33) -> None:
    """The model's fields (and p0) at Chebyshev points of the second kind
    in each layer, exactly, for the relabelled benchmark
    (relabelling.hpp): per layer a line `layer <attribute> <r_lo> <r_hi>
    <nodes>`, the radii, then one line per field of FIELDS. The end
    points sit just inside the layer."""
    b = np.asarray(model.skeleton.boundaries, dtype=float)
    nudge = 1e-12 * b[-1]
    x = np.cos(np.pi * np.arange(nodes) / (nodes - 1))[::-1]  # -1 .. 1

    def line(values: np.ndarray) -> str:
        return " ".join(f"{v:.17e}" for v in np.asarray(values, dtype=float))

    with path.open("w") as f:
        f.write(f"layers {len(model.layers)} nodes {nodes} fields "
                + " ".join(FIELDS) + "\n")
        for i, layer in enumerate(model.layers):
            lo, hi = b[i], b[i + 1]
            r = 0.5 * (lo + hi) + 0.5 * (hi - lo - 2.0 * nudge) * x
            f.write(f"layer {i + 1} {lo:.17e} {hi:.17e} {nodes}\n"
                    f"{line(r)}\n")
            for name in FIELDS:
                f.write(f"{line(layer.fields[name](r))}\n")


def reference(model: Model, lmax: int, *, per_layer: int) -> dict:
    """What pyslfp gives for the model, in the model's units."""
    love: LoveNumbers = love_numbers(model, lmax)
    load, tide = love.conventional(), love.tidal()
    s = model.scales
    return {
        "model": model.name,
        "scales": {"length": s.length, "mass": s.mass, "time": s.time},
        "G": float(love.G),
        "radius": float(love.radius),
        "surface_gravity": float(love.surface_gravity),
        "degree": [int(l) for l in love.degree],
        # dimensionless: load numbers h', l', k' and tidal numbers h, l, k
        "h_load": _clean(load["h"]), "l_load": _clean(load["l"]),
        "k_load": _clean(load["k"]),
        "h_tide": _clean(tide["h"]), "l_tide": _clean(tide["l"]),
        "k_tide": _clean(tide["k"]),
        # per unit surface density or potential, in the model's units
        "generalised": {name: _clean(getattr(love, name))
                        for name in ("h_u", "l_u", "k_u", "h_phi", "l_phi",
                                     "k_phi", "h_t", "l_t", "k_t")},
        "reciprocity_residual": _clean(love.reciprocity_residual()),
        "profiles": profiles(model, lmax, per_layer=per_layer),
    }


def main() -> None:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("model", choices=sorted(models.MODELS))
    p.add_argument("--out", type=Path, required=True, metavar="DIR",
                   help="the case directory")
    p.add_argument("--h", type=float, default=0.15,
                   help="element size on every interface")
    p.add_argument("--h-max", type=float, default=None,
                   help="element size far from the interfaces (default 2 h)")
    p.add_argument("--decay", type=float, default=None,
                   help="distance over which the size grows (default 10 h)")
    p.add_argument("--no-optimise", action="store_true",
                   help="leave the tetrahedra as the mesher made them")
    p.add_argument("--angular", type=float, default=0.3,
                   help="largest element size on an interface over its radius")
    p.add_argument("--thin", type=float, default=4.0,
                   help="largest element size on an interface over the "
                        "thickness of the layers it bounds")
    p.add_argument("--buffer", type=float, default=0.2,
                   help="thickness of the buffer shell over the radius")
    p.add_argument("--order", type=int, default=2, help="geometry order")
    p.add_argument("--lmax", type=int, default=10,
                   help="highest degree of the reference")
    p.add_argument("--time-scale", type=float, default=None,
                   help="time scale in seconds (default: G equal to one)")
    p.add_argument("--profile-points", type=int, default=41,
                   help="radii per layer in the reference's radial solutions")
    p.add_argument("--field-nodes", type=int, default=33,
                   help="Chebyshev nodes per layer in reference_fields.txt")
    p.add_argument("--reference-only", action="store_true",
                   help="write reference.json alone, leaving the mesh")
    p.add_argument("--verbose", action="store_true",
                   help="let gmsh print as it works")
    args = p.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    model = models.with_pressure(
        models.scaled(models.model(args.model), time_scale=args.time_scale))

    ref = reference(model, args.lmax, per_layer=args.profile_points)
    (args.out / "reference.json").write_text(json.dumps(ref, indent=1))
    print(f"reference.json: degrees 0..{args.lmax}, G = {ref['G']:.6g}, "
          f"g = {ref['surface_gravity']:.6g}")
    write_reference_fields(model, args.lmax, args.out / FIELDS_FILE,
                           nodes=args.field_nodes)
    write_radial_profiles(model, args.out / PROFILES_FILE)
    if args.reference_only:
        return

    h_max = 2.0 * args.h if args.h_max is None else args.h_max
    decay = 10.0 * args.h if args.decay is None else args.decay
    summary = build_mesh(model, args.out, h=args.h, h_max=h_max, decay=decay,
                         angular=args.angular, thin=args.thin,
                         buffer=args.buffer,
                         optimise=not args.no_optimise, order=args.order,
                         verbose=args.verbose)
    (args.out / "mesh_summary.json").write_text(json.dumps(summary, indent=1))
    print(f"{BASENAME}.mesh: {summary['summary']}")


if __name__ == "__main__":
    main()
