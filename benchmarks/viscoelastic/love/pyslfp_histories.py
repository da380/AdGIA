"""pyslfp's Maxwell load Love-number histories, as a V1 exchange file.

The time-domain twin of `laplace_reference.py`: the same Heaviside unit
load, stepped with pyslfp's own viscoelastic operator rather than
inverted from the Laplace domain.  The stepping rule and the exchange
layout match sl3dpp's `love_histories` driver exactly — exponential
trapezoid, equal steps between the love family's 13 geometric output
times, at least `--steps` of them and none longer than `--dtmax` tau_min
— and the mesh is the uniform lmax rule both spectral codes share, so
their differences isolate the radial discretisation.  Columns: t (in
units of tau_min) then hp<l> lp<l> kp<l> per degree, the conventional
dimensionless load numbers.

Runs from the pyslfp checkout, which owns its environment:

    cd ~/dev/pyslfp && poetry run python \
        ~/dev/AdGIA/benchmarks/viscoelastic/love/pyslfp_histories.py \
        homogeneous --tau 1e11 --out <runs dir>
"""
from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent.parent / "common"))

import exchange  # noqa: E402
import models  # noqa: E402
import outputs  # noqa: E402
from planetmodel import RadialMesh  # noqa: E402
from pyslfp.love_numbers import Material  # noqa: E402
from pyslfp.viscoelastic import Rheology, ViscoelasticOperator  # noqa: E402

#: The love family's 13 geometric output times, in units of tau_min,
#: rounded to six decimals exactly as laplace_reference.py writes them.
TIMES = [round(0.011 * 2.0 ** (5.0 * k / 6.0), 6) for k in range(13)]


def rheology(material: Material, taus: list[float]) -> tuple[Rheology, float]:
    """A Maxwell body with the given relaxation time per layer (inf
    elastic; a fluid layer must be inf), and tau_min."""
    mesh = material.mesh
    mu = np.asarray(material.moduli.kappa_mu()[1]).real
    itau = np.zeros_like(mu)
    finite = [t for t in taus if math.isfinite(t)]
    if not finite:
        raise SystemExit("every layer is elastic: nothing relaxes")
    for e in range(mesh.nspec):
        layer = int(mesh.layer[e])
        if not math.isfinite(taus[layer]):
            continue
        if material.fluid[e]:
            raise SystemExit(f"layer {layer} is fluid: its tau must be inf")
        itau[e, :] = 1.0 / taus[layer]
    return Rheology(material, [(mu, itau)]), min(finite)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="pyslfp Maxwell Love-number histories as an exchange "
        "file")
    parser.add_argument("name", help="a ladder model")
    parser.add_argument("--tau", required=True,
                        help="Maxwell times in seconds, comma list per "
                        "layer from the centre out, inf = elastic")
    parser.add_argument("--lmax", type=int, default=4)
    parser.add_argument("--steps", type=int, default=4,
                        help="least steps between output times")
    parser.add_argument("--dtmax", type=float, default=0.25,
                        help="longest step, in units of tau_min")
    parser.add_argument("--out", type=Path, required=True,
                        help="output directory, outside the source tree")
    args = parser.parse_args(argv)
    out = outputs.outside_source(args.out)
    out.mkdir(parents=True, exist_ok=True)
    import pyslfp

    model = models.model(args.name)
    material = Material(RadialMesh(model, lmax=args.lmax, ngll=5), model)
    taus = [math.inf if t.strip() == "inf" else float(t)
            for t in args.tau.split(",")]
    if len(taus) != len(model.layers):
        raise SystemExit(f"{len(taus)} taus for {len(model.layers)} layers")
    rheo, tau_min = rheology(material, taus)
    op = ViscoelasticOperator(rheo, args.lmax)

    g, a, G = (material.surface_gravity, material.radius, material.G)
    factors = (2.0 * np.arange(args.lmax + 1) + 1.0) / (4.0 * np.pi * G * a)

    def load(l, t):
        return np.array([1.0])

    rows = []
    state = op.zero_state()
    previous = 0.0
    for t_tau in TIMES:
        t = t_tau * tau_min
        interval = t - previous
        n = max(args.steps, math.ceil(interval / (args.dtmax * tau_min)))
        for _ in range(n):
            state, solutions = op.step(state, interval / n, load=load)
        previous = t
        U = op.surface(solutions, "U")
        V = op.surface(solutions, "V")
        P = op.surface(solutions, "phi")
        row = [t_tau]
        for l in range(args.lmax + 1):
            f = factors[l]
            row += [f * g * float(U[l][0].real), f * g * float(V[l][0].real),
                    -f * float(P[l][0].real) - 1.0]
        rows.append(row)

    columns = ["t"]
    for l in range(args.lmax + 1):
        columns += [f"hp{l}", f"lp{l}", f"kp{l}"]
    path = exchange.write_table(
        out / f"v1_{args.name}_pyslfp.txt",
        comments=[
            "V1 Maxwell load Love-number histories from pyslfp (Heaviside "
            "unit load)",
            exchange.provenance_line("pyslfp", Path(pyslfp.__file__)),
            f"model {args.name} lmax {args.lmax} taus {args.tau} s; "
            f"exponential trapezoid, >= {args.steps} steps per interval, "
            f"dt <= {args.dtmax} tau_min; uniform lmax mesh rule; t in "
            "units of tau_min; dimensionless conventional load numbers",
        ],
        scalars={"radius": material.radius},
        columns=columns,
        table=np.array(rows),
    )
    print(path)


if __name__ == "__main__":
    main()
