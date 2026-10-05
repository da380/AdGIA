"""Exact references for the sphere benchmarks (viscoelastic_sphere.cpp).

A spherically layered, non-gravitating body under the surface pressure
p(t) Y_l on r = a. Write u = U(r) Y_l rhat + V(r) grad_1 Y_l and the
tractions sigma_rr = R Y_l, sigma_rtheta-part = S grad_1 Y_l. The static
equilibrium equations (no gravity, no inertia) for constant moduli are of
Euler type in the scaled variables Y = (U, V, rR, rS):

    r dY/dr = M(lambda, mu, l) Y,

M independent of r (derive_euler.py derives it with sympy from Navier's
equations in spherical or polar coordinates and checks it). Its
eigenvalues are l - 1, l + 1, -l, -l - 2 in 3-D and l - 1, l + 1,
-l + 1, -l - 1 in 2-D, whatever the moduli, so

  * the solutions regular at the centre are the eigenvectors of the two
    largest eigenvalues, times r^m;
  * across a layer [r1, r2] the propagator is (r2/r1)^M = expm(M ln r2/r1);
  * U(a), V(a) per unit pressure (R(a) = -1, S(a) = 0) are rational
    functions of the moduli.

By the correspondence principle (mu -> mu(s), lambda = kappa - 2 mu(s)/d
in every Maxwell layer) the transfer functions U(a; s), V(a; s) are
rational in s, and the modes, checks and modal histories are those of the
box slabs (box/reference.py: AAA poles and residues, fixed-Talbot check of
the Heaviside response, exact convolutions with the load history). The
checks are stored and printed only: unlike the box slabs, there is no
fallback to the Talbot inversion when they fail.

Radial models only; a laterally varying one has no reference here (the
studies compare it with a much smaller step on the same mesh).

    ./reference case.json --out reference.json
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import math
import sys
from pathlib import Path

import numpy as np
from scipy.linalg import expm

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "box"))
sys.path.insert(0, str(HERE))

import histories as hist  # noqa: E402
from meshes import outside_source  # noqa: E402

_spec = importlib.util.spec_from_file_location(
    "box_reference", HERE.parent / "box" / "reference.py")
box = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(box)


def euler_matrix(dim: int, l: int, lam: complex, mu: complex) -> np.ndarray:
    """M of r dY/dr = M Y, Y = (U, V, rR, rS) (derive_euler.py)."""
    F = lam + 2 * mu
    if dim == 3:
        L = l * (l + 1)
        return np.array([
            [-2 * lam / F, L * lam / F, 1 / F, 0],
            [-1, 1, 0, 1 / mu],
            [4 * mu * (3 * lam + 2 * mu) / F,
             -2 * L * mu * (3 * lam + 2 * mu) / F, (lam - 2 * mu) / F, L],
            [-2 * mu * (3 * lam + 2 * mu) / F,
             2 * mu * (2 * L * (lam + mu) - lam - 2 * mu) / F, -lam / F, -2],
        ], dtype=complex)
    L = l * l
    g = 4 * mu * (lam + mu) / F
    return np.array([
        [-lam / F, L * lam / F, 1 / F, 0],
        [-1, 1, 0, 1 / mu],
        [g, -L * g, lam / F, L],
        [-g, L * g, -lam / F, -1],
    ], dtype=complex)


class SphereCase:
    def __init__(self, path: Path):
        self.path = Path(path)
        d = json.loads(self.path.read_text())
        self.raw = d
        self.name, self.dim = d["name"], int(d["dim"])
        self.radii = [float(r) for r in d["radii"]]
        self.layers = [box.Layer(ld, r0) for ld, r0
                       in zip(self.layers_with_tops(d), self.radii[:-1])]
        load = d["load"]
        self.degree = int(load["degree"])
        self.amplitude = float(load["amplitude"])
        self.history = hist.History.from_json(load["history"])
        self.times = [float(t) for t in d["times"]]
        self.lmax = int(d.get("lmax", self.degree))
        if not all(layer.constant for layer in self.layers):
            raise SystemExit("the sphere reference is for radial models "
                             "with constant layers")

    def layers_with_tops(self, d):
        return [dict(ld, top=r1) for ld, r1 in zip(d["layers"],
                                                    d["radii"][1:])]

    def rates(self) -> list[float]:
        return [1 / float(tk(layer.bottom)) for layer in self.layers
                for m, tk in layer.branches if float(m(layer.bottom)) > 0]


def transfer(case: SphereCase, s: complex) -> tuple[complex, complex]:
    """(U(a), V(a)) per unit surface pressure at Laplace variable s."""
    d, l = case.dim, case.degree
    B = None
    for layer, r0, r1 in zip(case.layers, case.radii[:-1], case.radii[1:]):
        lam, mu = box.complex_moduli(layer, s, d)
        M = euler_matrix(d, l, lam, mu)
        if B is None:
            # The regular solutions: the two largest exponents, l - 1 and
            # l + 1, evaluated at the top of the central layer.
            w, V = np.linalg.eig(M)
            idx = np.argsort(-w.real)[:2]
            B = V[:, idx] * (r1 ** w[idx].real)[None, :]
        else:
            B = expm(M * math.log(r1 / r0)) @ B
    a = case.radii[-1]
    c = np.linalg.solve(B[2:4], np.array([-a, 0.0], dtype=complex))
    Y = B @ c
    return Y[0], Y[1]


def modes(case: SphereCase, which: int) -> dict:
    rates = case.rates()
    smin, smax = min(rates), max(rates)
    radii = np.geomspace(1e-3 * smin, 1e3 * smax, 90)
    angles = np.array([0.0, 0.25, -0.25, 0.5, -0.5, 0.7, -0.7]) * math.pi
    Z = np.concatenate([r * np.exp(1j * angles) for r in radii])
    F = np.array([transfer(case, z)[which] for z in Z])
    z, f, w = box.aaa(Z, F)
    poles, res, r_inf = box.aaa_poles_residues(z, f, w)
    test = np.geomspace(2e-3 * smin, 5e2 * smax, 37) * np.exp(0.33j * math.pi)
    Ft = np.array([transfer(case, t)[which] for t in test])
    fit = float(np.abs(box.aaa_eval(z, f, w, test) - Ft).max()
                / np.abs(Ft).max())
    finite = np.isfinite(res)
    poles, res = poles[finite], res[finite]
    big = np.abs(res) > 1e-12 * max(1.0, np.abs(res).max())
    poles, res = poles[big], res[big]
    return {"poles": poles, "residues": res, "r_inf": complex(r_inf),
            "fit_error": fit,
            "pole_imag": float(np.max(np.abs(poles.imag) / np.maximum(
                np.abs(poles), smin), initial=0.0)),
            "pole_max_real": float(np.max(poles.real, initial=-np.inf))}


def reference(case: SphereCase) -> dict:
    t_all = np.asarray([0.0] + case.times)
    mU, mV = modes(case, 0), modes(case, 1)
    U = box.modal_response(mU, case.history, case.amplitude, t_all)
    V = box.modal_response(mV, case.history, case.amplitude, t_all)
    t = np.asarray(case.times)
    step = hist.heaviside()
    sU = box.modal_response(mU, step, 1.0, t)
    sV = box.modal_response(mV, step, 1.0, t)
    tU = np.array([box.talbot(lambda s: transfer(case, s)[0] / s, tj)
                   for tj in t])
    tV = np.array([box.talbot(lambda s: transfer(case, s)[1] / s, tj)
                   for tj in t])
    checks = {"fit_error_U": mU["fit_error"], "fit_error_V": mV["fit_error"],
              "pole_imag_U": mU["pole_imag"],
              "pole_max_real_U": mU["pole_max_real"],
              "talbot_vs_modal_U": float(np.abs(tU - sU).max()
                                         / np.abs(sU).max()),
              "talbot_vs_modal_V": float(np.abs(tV - sV).max()
                                         / np.abs(sV).max())}
    n = case.lmax + 1

    def per_degree(x):
        out = [0.0] * n
        out[case.degree] = float(x)
        return out

    series = [{"U": per_degree(U[j]), "V": per_degree(V[j])}
              for j in range(len(t_all))]
    return {"case": case.name, "case_file": str(case.path),
            "dim": case.dim, "kind": "sphere", "load": "surface",
            "degree": case.degree, "lmax": case.lmax, "times": case.times,
            "elastic": series[0],
            "histories": [dict(time=tj, **sj)
                          for tj, sj in zip(case.times, series[1:])],
            "modes": [{"pole": float(p.real), "residue_U": float(r.real)}
                      for p, r in sorted(zip(mU["poles"], mU["residues"]),
                                         key=lambda pr: -pr[0].real)],
            "checks": checks}


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("case", type=Path)
    p.add_argument("--out", type=Path, default=Path("reference.json"))
    a = p.parse_args()
    ref = reference(SphereCase(a.case))
    out = outside_source(a.out)
    out.write_text(json.dumps(ref, indent=1))
    for k, v in ref["checks"].items():
        print(f"  {k:22s} {v:.3e}")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
