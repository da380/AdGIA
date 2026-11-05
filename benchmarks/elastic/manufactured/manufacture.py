"""Manufactured solutions of the static self-gravitating elastic problem.

    manufacture [--out DIR]

Writes `fields.hpp`: the exact fields (u*, phi*) of a solid, self-gravitating
body and the loads that make them the solution of
LinearQuasiStaticMixedSelfGravitatingProblem's weak form, as C++ functions,
for two and three dimensions. The derivation is symbolic (sympy); every
source term is differentiated from the chosen fields, so the fields can be
changed here and the header regenerated. The driver manufactured_benchmark
reads the header; nothing of the physics is typed by hand on the C++ side.

The weak form (doc/methods/self_gravitation.md, a solid body M inside the
computational ball B, test functions v, chi):

  A(u,phi | v,chi) = int_M kappa div u div v + 2 mu d(u):d(v)
                   + 1/2 rho [grad(u.g).v + grad(v.g).u]
                   - 1/2 rho [(u.g) div v + (v.g) div u]
                   + rho (grad phi . v + grad chi . u)
                   + 1/(4 pi G) int_B grad phi . grad chi  (+ DtN at dB)

with g = grad Phi0 and d the deviatoric strain, eps - (div u / dim) I.
Integrating by parts, the exact pair (u*, phi*) satisfies A = l_u(v) +
l_phi(chi) with

  l_u(v)     = int_M f.v + int_dM T.v,
  f  = -div sigma + 1/2 rho grad(u.g) - 1/2 div(rho u) g
       + 1/2 grad(rho u.g) - 1/2 rho g div u + rho grad phi,
  T  = sigma n + 1/2 rho (u.n) g - 1/2 rho (u.g) n,
  sigma = kappa (div u) I + 2 mu d(u),

  l_phi(chi) = int_B q chi + int_dM s chi,
  q  = -lap phi / (4 pi G) - div(rho u)          in M,
       -lap phi / (4 pi G)                        in the buffer,
  s  = (dphi_in/dn - dphi_out/dn) / (4 pi G) + rho u.n   on dM,

n the outward normal of M. The potential is continuous across dM and may
have a kink there (s carries it); it is taken EXACTLY harmonic in the
buffer, as a sum of exterior multipoles about the centre, so that q
vanishes there and the DtN condition at the outer sphere holds exactly
whatever the buffer radius. Inside M the potential is the interior
continuation of the multipoles' traces plus (a^2 - r^2) w(x), w smooth.

The background is the self-consistent one of a homogeneous ball of
radius a (Phi0 quadratic inside, the exterior multipole outside) so that
the rigid modes are exact null pairs of the continuous problem, as the
solver assumes: the body is in hydrostatic equilibrium. The moduli vary
smoothly, to exercise the variable-coefficient path.

Units are the non-dimensional ones of the tests (a = 1, G = 0.05,
rho = 1, kappa ~ 1, mu ~ 0.5: rho g a / mu ~ 0.4 at the surface).
"""
from __future__ import annotations

import argparse
import textwrap
from pathlib import Path

import sympy as sp

A = sp.Integer(1)           # body radius
G = sp.Rational(1, 20)      # gravitational constant, 0.05
RHO0 = sp.Integer(1)        # density


def fields(dim: int):
    """The chosen exact fields and the material, as sympy expressions in
    the symbols x[0..dim-1] (named so that ccode prints C array access)."""
    x = sp.Matrix([sp.Symbol(f"x[{i}]") for i in range(dim)])
    r2 = (x.T * x)[0]
    r = sp.sqrt(r2)
    amp = sp.Rational(1, 10)
    if dim == 3:
        u = amp * sp.Matrix([
            sp.sin(x[1]) * sp.cos(x[2]) + x[0] ** 2 / 2,
            sp.exp(sp.Rational(3, 10) * x[0]) * sp.sin(x[2]),
            sp.cos(x[0] * x[1]) + x[1] * x[2],
        ])
        kappa = 1 + sp.Rational(1, 5) * x[0] * x[1]
        mu = sp.Rational(1, 2) + sp.Rational(1, 10) * x[2] ** 2
        # Exterior multipoles (harmonic, decaying): monopole, dipole,
        # quadrupole; the interior continuation of their traces on r = a.
        m = sp.Rational(1, 10)
        d = sp.Matrix([sp.Rational(3, 10), -sp.Rational(1, 5),
                       sp.Rational(1, 10)])
        Q = sp.Matrix([[sp.Rational(1, 5), sp.Rational(1, 10), 0],
                       [sp.Rational(1, 10), -sp.Rational(1, 10),
                        sp.Rational(1, 20)],
                       [0, sp.Rational(1, 20), -sp.Rational(1, 10)]])
        xQx = (x.T * Q * x)[0]
        phi_ext = m / r + (d.T * x)[0] / r ** 3 + xQx / r ** 5
        trace = m / A + (d.T * x)[0] / A ** 3 + xQx / A ** 5
        w = sp.Rational(1, 2) * sp.cos(x[0]) * sp.sin(x[1]) \
            + sp.Rational(3, 10) * x[2] + sp.Rational(1, 5)
        phi0_in = 2 * sp.pi * G * RHO0 / 3 * (r2 - 3 * A ** 2)
        phi0_out = -(4 * sp.pi * G * RHO0 * A ** 3 / 3) / r
    else:
        u = amp * sp.Matrix([
            sp.sin(x[1]) + x[0] ** 2 / 2,
            sp.exp(sp.Rational(3, 10) * x[0]) * sp.cos(x[1]),
        ])
        kappa = 1 + sp.Rational(1, 5) * x[0] * x[1]
        mu = sp.Rational(1, 2) + sp.Rational(1, 10) * x[1] ** 2
        # No monopole in 2-D (log r is not in the DtN's range; the
        # constant is projected out).
        d = sp.Matrix([sp.Rational(3, 10), -sp.Rational(1, 5)])
        Q = sp.Matrix([[sp.Rational(1, 5), sp.Rational(1, 10)],
                       [sp.Rational(1, 10), -sp.Rational(1, 5)]])
        xQx = (x.T * Q * x)[0]
        phi_ext = (d.T * x)[0] / r ** 2 + xQx / r ** 4
        trace = (d.T * x)[0] / A ** 2 + xQx / A ** 4
        w = sp.Rational(1, 2) * sp.cos(x[0]) * sp.sin(x[1]) \
            + sp.Rational(1, 5)
        phi0_in = sp.pi * G * RHO0 * (r2 - A ** 2)
        phi0_out = 2 * sp.pi * G * RHO0 * A ** 2 * sp.log(r / A)
    phi_in = trace + (A ** 2 - r2) * w
    rho = RHO0 + 0 * x[0]
    return dict(x=x, r2=r2, u=u, kappa=kappa, mu=mu, rho=rho,
                phi_in=phi_in, phi_ext=phi_ext, phi0_in=phi0_in,
                phi0_out=phi0_out)


def grad(f, x):
    return sp.Matrix([sp.diff(f, xi) for xi in x])


def div(v, x):
    return sum(sp.diff(v[i], x[i]) for i in range(len(x)))


def laplacian(f, x):
    return sum(sp.diff(f, xi, 2) for xi in x)


def derive(dim: int):
    F = fields(dim)
    x, u, kappa, mu, rho = F["x"], F["u"], F["kappa"], F["mu"], F["rho"]
    n = sp.Matrix([sp.Symbol(f"n[{i}]") for i in range(dim)])
    I = sp.eye(dim)
    gradu = sp.Matrix(dim, dim, lambda i, j: sp.diff(u[i], x[j]))
    eps = (gradu + gradu.T) / 2
    divu = div(u, x)
    dev = eps - divu / dim * I
    sigma = kappa * divu * I + 2 * mu * dev
    div_sigma = sp.Matrix([sum(sp.diff(sigma[i, j], x[j]) for j in range(dim))
                           for i in range(dim)])
    g = grad(F["phi0_in"], x)
    ug = (u.T * g)[0]
    rho_u = rho * u
    f = (-div_sigma + rho / 2 * grad(ug, x) - div(rho_u, x) / 2 * g
         + grad(rho * ug, x) / 2 - rho / 2 * g * divu
         + rho * grad(F["phi_in"], x))
    T = sigma * n + rho / 2 * (u.T * n)[0] * g - rho / 2 * ug * n
    fourpiG = 4 * sp.pi * G
    q_in = -laplacian(F["phi_in"], x) / fourpiG - div(rho_u, x)
    q_out = sp.simplify(-laplacian(F["phi_ext"], x) / fourpiG)
    assert q_out == 0, f"the exterior potential is not harmonic: {q_out}"
    s = ((n.T * grad(F["phi_in"], x))[0] - (n.T * grad(F["phi_ext"], x))[0]) \
        / fourpiG + (rho_u.T * n)[0]
    # Background consistency: lap Phi0 = 4 pi G rho inside, 0 outside,
    # continuous with continuous gradient at r = a.
    assert sp.simplify(laplacian(F["phi0_in"], x) - fourpiG * rho) == 0
    assert sp.simplify(laplacian(F["phi0_out"], x)) == 0
    return dict(F=F, n=n, f=f, T=T, q_in=q_in, s=s, g=g,
                grad_phi0_out=grad(F["phi0_out"], x), sigma=sigma,
                gradu=gradu,
                hess_phi0_in=sp.Matrix(dim, dim, lambda i, j: sp.diff(
                    F["phi0_in"], x[i], x[j])))


def ccode(e):
    return sp.ccode(e, standard="C99")


def emit_scalar(name: str, expr, args: str) -> str:
    reps, (red,) = sp.cse([expr], symbols=sp.numbered_symbols("t"))
    body = [f"  const double {s} = {ccode(e)};" for s, e in reps]
    body.append(f"  return {ccode(red)};")
    return f"inline double {name}({args}) {{\n" + "\n".join(body) + "\n}\n"


def emit_vector(name: str, exprs, args: str) -> str:
    reps, red = sp.cse(list(exprs), symbols=sp.numbered_symbols("t"))
    body = [f"  const double {s} = {ccode(e)};" for s, e in reps]
    body += [f"  out[{i}] = {ccode(e)};" for i, e in enumerate(red)]
    return f"inline void {name}({args}) {{\n" + "\n".join(body) + "\n}\n"


def emit_namespace(dim: int) -> str:
    D = derive(dim)
    F = D["F"]
    X = "const double* x"
    XN = "const double* x, const double* n"
    XO = "const double* x, double* out"
    XNO = "const double* x, const double* n, double* out"
    parts = [
        f"namespace d{dim} {{\n",
        f"constexpr int kDim = {dim};\n",
        "/// The exact displacement u*.\n",
        emit_vector("u", F["u"], XO),
        "/// The exact potential inside the body (r <= a).\n",
        emit_scalar("phi_in", F["phi_in"], X),
        "/// The exact potential in the buffer (r >= a): exterior multipoles.\n",
        emit_scalar("phi_out", F["phi_ext"], X),
        "/// The background potential inside the body.\n",
        emit_scalar("phi0_in", F["phi0_in"], X),
        "/// The background potential outside the body.\n",
        emit_scalar("phi0_out", F["phi0_out"], X),
        "/// grad Phi0 inside the body.\n",
        emit_vector("grad_phi0_in", D["g"], XO),
        "/// grad Phi0 outside the body.\n",
        emit_vector("grad_phi0_out", D["grad_phi0_out"], XO),
        "/// The Hessian of Phi0 inside the body (row-major).\n",
        emit_vector("hess_phi0_in", list(D["hess_phi0_in"]), XO),
        "/// The density, bulk and shear moduli.\n",
        emit_scalar("rho", F["rho"], X),
        emit_scalar("kappa", F["kappa"], X),
        emit_scalar("mu", F["mu"], X),
        "/// The body force f of the displacement row (in the body).\n",
        emit_vector("body_force", D["f"], XO),
        "/// The traction T of the displacement row on the surface, n the\n"
        "/// outward normal.\n",
        emit_vector("traction", D["T"], XNO),
        "/// The volume source q of the potential row inside the body (it\n"
        "/// vanishes in the buffer by construction).\n",
        emit_scalar("potential_source", D["q_in"], X),
        "/// The surface source s of the potential row on the surface.\n",
        emit_scalar("surface_source", D["s"], XN),
        "/// The exact stress sigma(u*) (row-major), for traces.\n",
        emit_vector("stress", list(D["sigma"]), XO),
        "/// The exact displacement gradient du*_i/dx_j (row-major).\n",
        emit_vector("grad_u", list(D["gradu"]), XO),
        f"}}  // namespace d{dim}\n",
    ]
    return "".join(parts)


def describe(dim: int) -> str:
    F = fields(dim)
    D = derive(dim)
    lines = [f"dimension {dim}:",
             f"  u*     = {sp.simplify(F['u'].T)}",
             f"  phi*   = {F['phi_in']}  (r <= a)",
             f"         = {F['phi_ext']}  (r >= a)",
             f"  Phi0   = {F['phi0_in']}  (r <= a)",
             f"         = {F['phi0_out']}  (r >= a)",
             f"  rho    = {F['rho']}",
             f"  kappa  = {F['kappa']}",
             f"  mu     = {F['mu']}"]
    return "\n".join(lines)


def header() -> str:
    doc = textwrap.indent(__doc__.strip(), " * ", lambda _: True)
    desc = "\n".join(describe(d) for d in (2, 3))
    desc = textwrap.indent(desc, " * ", lambda _: True)
    out = [
        "// GENERATED by manufacture.py -- do not edit; change the fields there\n",
        "// and regenerate.\n",
        "/*\n", doc, "\n *\n * The fields of this header:\n", desc, "\n */\n",
        "#pragma once\n#include <cmath>\n#ifndef M_PI\n"
        "#define M_PI 3.14159265358979323846\n#endif\n\n",
        "namespace manufactured {\n\n",
        f"constexpr double kRadius = {float(A)!r};\n",
        f"constexpr double kG = {float(G)!r};\n",
        f"constexpr double kRho = {float(RHO0)!r};\n\n",
        emit_namespace(2), "\n", emit_namespace(3),
        "\n}  // namespace manufactured\n",
    ]
    return "".join(out)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--out", type=Path, default=Path(__file__).resolve().parent,
                   help="directory of fields.hpp (default: beside this script)")
    args = p.parse_args()
    path = args.out / "fields.hpp"
    path.write_text(header())
    print(f"wrote {path}")
    for d in (2, 3):
        print(describe(d))


if __name__ == "__main__":
    main()
