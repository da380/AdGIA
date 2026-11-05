"""Manufactured solutions of the self-gravitating problem with a closed fluid core.

    manufacture_fluid [--out DIR] [--check]

Writes `fields_fluid.hpp`: the exact fields of a two-layer ball whose
core is a stably stratified fluid closed by the level border (the
Dahlen closure, doc/theory/dahlen_closure.tex, "Implementation"), and the loads
that make them the solution of the bordered weak form of
LinearQuasiStaticMixedSelfGravitatingProblem with SetLevelBorder, as C++
functions, in three dimensions. The derivation is symbolic (sympy), as
in manufacture.py (the solid family, whose helpers and displacement it
reuses); `--check` verifies it numerically (below).

Geometry and background. Fluid core F: r < b; solid shell S: b < r < 1;
buffer 1 < r < R (b = 0.5467 and R = 1.2 are those of the canned mesh
data/earth_fluidcore_3d.msh: attributes 1 core, 2 shell, 3 buffer;
boundary attributes 1 CMB, 2 surface, 3 outer sphere). The density is
rho_f(r) = rho_c (1 - beta r^2) in F, rho_s (constant) in S, 0 in the
buffer, and Phi0 is its exact potential, lap Phi0 = 4 pi G rho, -G M / r
outside, Phi0 and g = grad Phi0 continuous (Phi0 increases outwards; g
points outwards). The levels are the spheres
r = const; c = Phi0(r) runs over [c_min, c_w] = [Phi0(0), Phi0(b)] in F,
and c -> r is closed form (Phi0 is a quadratic in r^2 in F). The level
integrals (doc/theory/dahlen_closure.tex, eq. coarea) are

  A(c) = oint_{r(c)} dS / g = 4 pi r^2 / g(r),     h_bar(c) = oint h dS / g,

A the volume per unit potential (A = dVol/dc), so h_bar = A <h>, <h> the
spherical mean. rho' = d rho / d Phi0 = g^-1 d_r rho (the FluidRegion's
density_gradient), D = rho' + rho^2 / kappa_f, N^2 = -g^2 D / rho =
-g (d_r rho / rho + rho g / kappa_f); kappa_f is chosen so that N^2 > 0
for r > 0 (it vanishes like r^2 at the centre, where D < 0 still).

Unknowns and weak form. The displacement u on S, the potential phi on
the ball B, the border functions U, s, nu of the level on [c_min, c_w]
(U(c_min) = 0 essential at the centre) and one wall multiplier lambda.
With test functions v, chi, W (W(c_min) = 0), sigma, eta, theta, and m
the outward normal of the solid (m = -x/r at the CMB, +x/r at r = 1):

  (u)   a_S(u,v) + F2(u,v) + c(phi,v) + lambda w(v)        = l_u(v)
  (phi) 1/(4 pi G) [int_B grad phi.grad chi + DtN(phi,chi)]
        + int_F rho' phi chi + c(chi,u) + int_F nu(Phi0) chi = l_phi(chi)
  (U)   int kappa_f U' W' / A dc + int rho s W' dc + int nu W dc
        + lambda W(c_w)                                     = l_U(W)
  (s)   int (rho U' - rho' A s + A nu) sigma dc              = l_s(sigma)
  (nu)  int (U + s A) eta dc + int_F eta(Phi0) phi dV        = l_nu(eta)
  (lam) theta [U(c_w) + w(u)]                               = l_lam(theta)

primes d/dc, rho, rho', kappa_f, A level functions of F, and

  a_S(u,v) = int_S kappa div u div v + 2 mu d(u):d(v)
             + 1/2 rho_s [grad(u.g).v + grad(v.g).u]
             - 1/2 rho_s [(u.g) div v + (v.g) div u]          (d deviatoric)
  F2(u,v)  = -oint_CMB rho_F (m.g)(m.u)(m.v) dS
  c(chi,u) = int_S rho_s grad chi.u - oint_CMB rho_F chi (m.u) dS
  w(v)     = -oint_CMB (m.g/|g|)(m.v) dS = -S[v],  S[v] = oint_CMB v.x/r dS,

rho_F the fluid-side density at the wall. These are, term by term, the
blocks the class assembles: the mixed operator with (F1)-(F3)
(doc/methods/self_gravitation.md, "Bilinear form") plus the level border of
src/level_border.cpp (eq. L-built): K_UU = int kappa N_i' N_j' / A dc,
K_Us = int rho N_i' N_j dc, K_Unu = int N_i N_j dc, K_ss = -int_F rho'
N_i(Phi0) N_j(Phi0) dV = -int rho' A N_i N_j dc, K_snu = int_F N_i(Phi0)
N_j(Phi0) dV = int A N_i N_j dc, the nu columns int_F N_k(Phi0) chi dV =
int N_k X_bar dc on the potential row, and the wall column w(v) on the
displacement row, its transpose the lam row: U(c_w) - S[u] = l_lam.

Sources (every load functional, applied to the exact fields with
arbitrary tests and integrated by parts):

  l_u(v)     = int_S f.v + oint_{r=1} T.v dS + oint_CMB T_cmb.v dS,
    f      = -div sigma + 1/2 rho_s grad(u.g) - 1/2 div(rho_s u) g
             + 1/2 grad(rho_s u.g) - 1/2 rho_s g div u + rho_s grad phi,
    T      = sigma m + 1/2 rho_s (u.m) g - 1/2 rho_s (u.g) m      (r = 1),
    T_cmb  = T - rho_F (m.g)(m.u) m - rho_F phi m - lambda (m.g/|g|) m,
  l_phi(chi) = int_F q_F chi + int_S q_S chi + oint_{r=1} s_1 chi dS
               + oint_CMB s_b chi dS,
    q_F    = -lap phi / (4 pi G) + rho' phi + nu(Phi0),
    q_S    = -lap phi / (4 pi G) - div(rho_s u),   q = 0 in the buffer,
    s_1    = (m.grad phi_S - m.grad phi_out) / (4 pi G) + rho_s u.m,
    s_b    = (m.grad phi_S - m.grad phi_F) / (4 pi G) + (rho_s - rho_F) u.m,
  l_U(W)     = int r_U W dc + R_w W(c_w),
    r_U    = -(kappa_f U'/A + rho s)' + nu,
    R_w    = kappa_f U'(c_w)/A_w + rho_F s(c_w) + lambda,
  l_s(sigma) = int r_s sigma dc,   r_s = rho U' - rho' A s + A nu,
  l_nu(eta)  = int r_nu eta dc,    r_nu = U + s A + Phi_bar,
  l_lam      = U(c_w) - S[u].

The border sources are densities against dc: on the class's hat basis
N_k(c) the row data are int N_k r dc (plus R_w on the wall node of the
U row). Equivalently the U row's data are the weak residual itself,
int N_k' (kappa_f U'/A + rho s) dc + int N_k nu dc + lambda N_k(c_w),
which needs no second derivative. In three dimensions A, s A and Phi_bar
behave like sqrt(c - c_min) at the centre: integrate in r (dc = g dr)
rather than in c on the first element.

Choices.
- u*: the solid family's trigonometric/exponential field plus alpha x,
  whose flux through the CMB is the only nonzero one: S[u*] = 4 pi alpha
  b^3 (the rest has divergence (x + y)/10, odd).
- phi*: in the buffer the solid family's exterior multipoles (harmonic,
  so q = 0 there and the DtN condition is exact at any R); in the shell
  their interior trace plus (1 - r^2)(w0 + (r^2 - b^2) w1), w0 a
  harmonic polynomial and w1 the solid family's w; in the core
  a0(r) + H(x) + (b^2 - r^2) p(x), H the harmonic polynomial of degree
  >= 1 continuing the shell's trace at r = b and p odd, so that the
  level integral Phi_bar(c) = A(c) a0(r(c)) is closed form (spherical
  means of H and p vanish); a0(b) is the degree-0 part of the trace.
  phi* is continuous at r = b and r = 1, with kinks carried by s_b, s_1.
- U*, s*, nu*: polynomials in r^2 (smooth in c, U* ~ r^3 at the centre,
  U*(c_min) = 0), chosen freely: they satisfy none of the fluid's
  equations, so every border row carries a source, and U*(c_w) differs
  from S[u*] (l_lam != 0). lambda* is a free constant.
- No two-dimensional family: the class regularises the 2-D potential by
  removing the constant (a Euclidean projection on the true dofs, and a
  uniform flux added to the potential load), which with a fluid is not
  a null-space fact (doc/methods/self_gravitation.md, section 5); the 2-D
  bordered solve then differs from the continuum at the level of a
  constant, which no field can be manufactured to match.

Units are those of the solid family (G = 0.05, unit surface radius, rho_s
= 1, kappa ~ 1, mu ~ 0.5).

--galerkin solves the border rows alone by P1 Galerkin on uniform grids
in c (4 to 64 nodes) with exact level integrals, the exact potential's
level integral and wall flux as data, and prints the errors of U, s, nu
and lambda: the one-dimensional discretisation's own accuracy, below
which the class's border (whose level integrals are binned fluid
quadrature points) cannot go.

--check evaluates both sides of every row's identity by quadrature on
the ball (Gauss-Legendre in r and cos(theta), trapezoid in longitude)
and on the level interval (Gauss-Legendre in r per level element), for
random polynomial tests v, chi and random hat-function tests on random
level grids, the left side computed from the forms above (the potential
coupling of the nu row and the A-weighted terms of the s row as fluid
volume integrals, as the class assembles them) and the right side from
the emitted sources, and prints the relative residuals.
"""
from __future__ import annotations

import argparse
import textwrap
from pathlib import Path

import sympy as sp

import manufacture as solid
from manufacture import div, grad, laplacian

G = sp.Rational(1, 20)            # gravitational constant
RS = sp.Integer(1)                # surface radius
B = sp.Rational(5467, 10000)      # core-mantle boundary (the canned meshes)
ROUT = sp.Rational(6, 5)          # outer radius of the buffer
RHO_S = sp.Integer(1)             # shell density
RHO_C = sp.Rational(6, 5)         # fluid density at the centre
BETA = sp.Rational(3, 10)         # rho_f = rho_c (1 - beta r^2)
KAPPA_F = sp.Integer(10)          # fluid bulk modulus
ALPHA = sp.Rational(1, 20)        # degree-0 part of u*: alpha x
LAMBDA = sp.Rational(1, 20)       # the wall multiplier lambda*

r = sp.Symbol("r", positive=True)   # radius, the level coordinate
c = sp.Symbol("c", real=True)       # level value


def background():
    """The layered background as functions of r: density, mass, gravity
    and potential per region, the continuity constants solved."""
    s = sp.Symbol("s", positive=True)
    rho_f = RHO_C * (1 - BETA * r ** 2)
    mass_core = sp.integrate((rho_f * r ** 2).subs(r, s), (s, 0, r))
    mb = mass_core.subs(r, B)
    mass_shell = mb + RHO_S * (r ** 3 - B ** 3) / 3
    m_tot = 4 * sp.pi * mass_shell.subs(r, RS)
    g_core = sp.expand(4 * sp.pi * G * mass_core / r ** 2)
    g_shell = 4 * sp.pi * G * mass_shell / r ** 2
    g_out = G * m_tot / r ** 2
    C1, C2 = sp.symbols("C1 C2")
    phi_out = -G * m_tot / r
    phi_shell = sp.integrate(g_shell, r) + C2
    C2v = sp.solve(sp.Eq(phi_shell.subs(r, RS), phi_out.subs(r, RS)), C2)[0]
    phi_shell = sp.expand(phi_shell.subs(C2, C2v))
    phi_core = sp.integrate(g_core, r) + C1
    C1v = sp.solve(sp.Eq(phi_core.subs(r, B), phi_shell.subs(r, B)), C1)[0]
    phi_core = sp.expand(phi_core.subs(C1, C1v))
    # Continuity of g at both interfaces, and Poisson's equation (radial
    # Laplacian) in each region.
    for a, b_, at in ((g_core, g_shell, B), (g_shell, g_out, RS)):
        assert sp.simplify((a - b_).subs(r, at)) == 0
    for p, rho in ((phi_core, rho_f), (phi_shell, RHO_S), (phi_out, 0)):
        lap = sp.diff(p, r, 2) + 2 * sp.diff(p, r) / r
        assert sp.simplify(lap - 4 * sp.pi * G * rho) == 0
    for p, g in ((phi_core, g_core), (phi_shell, g_shell), (phi_out, g_out)):
        assert sp.simplify(sp.diff(p, r) - g) == 0
    return dict(rho_f=rho_f, g_core=g_core, g_shell=g_shell, g_out=g_out,
                phi_core=phi_core, phi_shell=phi_shell, phi_out=phi_out,
                c_min=phi_core.subs(r, 0), c_w=phi_core.subs(r, B))


def level_radius_expr(bg):
    """r(c) in the core: Phi0 = c_min + a1 r^2 - a2 r^4 is a quadratic in
    r^2; the root that vanishes at c_min, in the cancellation-free form."""
    t = sp.Symbol("t", positive=True)
    poly = sp.Poly(sp.expand(bg["phi_core"].subs(r, sp.sqrt(t)) - bg["c_min"]), t)
    a1 = poly.coeff_monomial(t)
    a2 = -poly.coeff_monomial(t ** 2)
    assert poly.degree() == 2 and poly.coeff_monomial(1) == 0
    dc = c - bg["c_min"]
    t_of_c = 2 * dc / (a1 + sp.sqrt(a1 ** 2 - 4 * a2 * dc))
    assert sp.simplify(bg["phi_core"].subs(r, sp.sqrt(t_of_c)) - c) == 0
    # The map is monotone on the core: d Phi0 / d(r^2) > 0 up to r = b.
    assert a1 - 2 * a2 * B ** 2 > 0
    return sp.sqrt(t_of_c), a1, a2


def d_dc(expr, bg):
    """d/dc of a level function of r in the core: (1/g) d/dr."""
    return sp.cancel(sp.diff(expr, r) / bg["g_core"])


def fields():
    """The exact fields and the material, as sympy expressions in the
    symbols x[0..2] (named so that ccode prints C array access), and the
    level functions of the core as expressions in r."""
    bg = background()
    dim = 3
    x = sp.Matrix([sp.Symbol(f"x[{i}]") for i in range(dim)])
    r2 = (x.T * x)[0]
    rx = sp.sqrt(r2)
    S0 = solid.fields(dim)            # the solid family's choices
    amp = sp.Rational(1, 10)
    u_solid = amp * sp.Matrix([
        sp.sin(x[1]) * sp.cos(x[2]) + x[0] ** 2 / 2,
        sp.exp(sp.Rational(3, 10) * x[0]) * sp.sin(x[2]),
        sp.cos(x[0] * x[1]) + x[1] * x[2],
    ])
    assert sp.simplify(u_solid - S0["u"]) == sp.zeros(dim, 1)
    u = u_solid + ALPHA * x
    kappa, mu = S0["kappa"], S0["mu"]
    # Exterior multipoles of the solid family (unit body radius) and the
    # interior harmonic continuation of their trace on r = 1.
    m = sp.Rational(1, 10)
    d = sp.Matrix([sp.Rational(3, 10), -sp.Rational(1, 5), sp.Rational(1, 10)])
    Q = sp.Matrix([[sp.Rational(1, 5), sp.Rational(1, 10), 0],
                   [sp.Rational(1, 10), -sp.Rational(1, 10), sp.Rational(1, 20)],
                   [0, sp.Rational(1, 20), -sp.Rational(1, 10)]])
    xQx = (x.T * Q * x)[0]
    phi_out = m / rx + (d.T * x)[0] / rx ** 3 + xQx / rx ** 5
    assert sp.simplify(phi_out - S0["phi_ext"]) == 0
    trace = m + (d.T * x)[0] + xQx
    w1 = sp.Rational(1, 2) * sp.cos(x[0]) * sp.sin(x[1]) \
        + sp.Rational(3, 10) * x[2] + sp.Rational(1, 5)
    assert sp.simplify(S0["phi_in"] - (trace + (1 - r2) * w1)) == 0
    w0 = sp.Rational(1, 5) + x[0] / 10 - x[2] / 10      # harmonic
    # The fields below are written with the symbol b (printed so) and
    # evaluated at b = B.
    bs = sp.Symbol("b", positive=True)
    pretty = {}
    pretty["phi_shell"] = trace + (1 - r2) * (w0 + (r2 - bs ** 2) * w1)
    # Core: the shell's trace at r = b is trace + (1 - b^2) w0, harmonic;
    # its degree-0 part K0 becomes a0(b), the rest continues as H.
    w00 = w0.subs({xi: 0 for xi in x})
    K0 = m + (1 - bs ** 2) * w00
    H = (d.T * x)[0] + xQx + (1 - bs ** 2) * (w0 - w00)
    assert sp.expand(laplacian(H, x)) == 0
    assert H.subs({xi: 0 for xi in x}) == 0
    p = sp.Rational(1, 2) * sp.sin(x[0]) * sp.cos(x[1]) \
        + sp.Rational(1, 5) * x[2] * sp.cos(x[0] * x[1])
    assert sp.simplify(p.subs({xi: -xi for xi in x}, simultaneous=True) + p) == 0
    pretty["a0"] = K0 + sp.Rational(1, 5) * (bs ** 2 - r ** 2) \
        - sp.Rational(1, 10) * (bs ** 2 - r ** 2) ** 2
    pretty["phi_core"] = pretty["a0"].subs(r, rx) + H + (bs ** 2 - r2) * p
    # Border functions (polynomials in r^2; U ~ r^3 at the centre).
    pretty["U"] = sp.Rational(1, 10) * (r / bs) ** 3 * (1 + (r / bs) ** 2 / 2)
    pretty["s"] = sp.Rational(1, 20) - sp.Rational(1, 10) * (r / bs) ** 2
    pretty["nu"] = sp.Rational(1, 10) + sp.Rational(1, 5) * (r / bs) ** 2
    vals = {k: v.subs(bs, B) for k, v in pretty.items()}
    phi_shell, phi_core, a0 = vals["phi_shell"], vals["phi_core"], vals["a0"]
    U, s, nu = vals["U"], vals["s"], vals["nu"]
    H, K0 = H.subs(bs, B), K0.subs(bs, B)
    to_x = lambda e: sp.simplify(e.subs(r, rx))  # noqa: E731
    phi0 = {k: bg[f"phi_{k}"].subs(r, rx) for k in ("core", "shell", "out")}
    return dict(bg=bg, x=x, r2=r2, u=u, kappa=kappa, mu=mu, w0=w0, w1=w1,
                p=p, a0=a0, K0=K0, H=H, phi_core=phi_core,
                phi_shell=phi_shell, phi_out=phi_out, U=U, s=s, nu=nu,
                phi0=phi0, to_x=to_x, pretty=pretty)


def derive():
    F = fields()
    bg, x, u = F["bg"], F["x"], F["u"]
    dim = 3
    n = sp.Matrix([sp.Symbol(f"n[{i}]") for i in range(dim)])
    I = sp.eye(dim)
    fourpiG = 4 * sp.pi * G
    # --- the shell: stress, body force, tractions ---------------------------
    gradu = sp.Matrix(dim, dim, lambda i, j: sp.diff(u[i], x[j]))
    eps = (gradu + gradu.T) / 2
    divu = div(u, x)
    sigma = F["kappa"] * divu * I + 2 * F["mu"] * (eps - divu / dim * I)
    div_sigma = sp.Matrix([sum(sp.diff(sigma[i, j], x[j]) for j in range(dim))
                           for i in range(dim)])
    g = grad(F["phi0"]["shell"], x)
    ug = (u.T * g)[0]
    rho_u = RHO_S * u
    f = (-div_sigma + RHO_S / 2 * grad(ug, x) - div(rho_u, x) / 2 * g
         + grad(RHO_S * ug, x) / 2 - RHO_S / 2 * g * divu
         + RHO_S * grad(F["phi_shell"], x))
    un = (u.T * n)[0]
    T = sigma * n + RHO_S / 2 * un * g - RHO_S / 2 * ug * n
    rho_F = F["to_x"](bg["rho_f"])      # fluid-side density at the wall
    gn = (n.T * g)[0]
    g_abs = sp.sqrt((g.T * g)[0])
    T_cmb = T - rho_F * gn * un * n - rho_F * F["phi_shell"] * n \
        - LAMBDA * gn / g_abs * n
    # --- level functions of the core (expressions in r) ---------------------
    rho = bg["rho_f"]
    gc = bg["g_core"]
    A = sp.cancel(4 * sp.pi * r ** 2 / gc)
    rho_p = sp.cancel(sp.diff(rho, r) / gc)          # d rho / d Phi0
    D = rho_p + rho ** 2 / KAPPA_F
    N2 = sp.cancel(-gc ** 2 * D / rho)
    U, s, nu = F["U"], F["s"], F["nu"]
    Up = d_dc(U, bg)
    phibar = sp.expand(A * F["a0"])                  # level integral of phi*
    flux_U = sp.cancel(KAPPA_F * Up / A + rho * s)
    r_U = sp.cancel(-d_dc(flux_U, bg) + nu)
    R_w = flux_U.subs(r, B) + LAMBDA
    r_s = sp.cancel(rho * Up - rho_p * A * s + A * nu)
    r_nu = sp.cancel(U + s * A + phibar)
    # S[u*] = int_{r<b} div u* dV (div u* is a polynomial here).
    th, ph, rr = sp.symbols("theta varphi rr", positive=True)
    sph = {x[0]: rr * sp.sin(th) * sp.cos(ph), x[1]: rr * sp.sin(th) * sp.sin(ph),
           x[2]: rr * sp.cos(th)}
    divu_poly = sp.expand(divu)
    assert divu_poly.is_polynomial(*x), divu_poly
    S_flux = sp.integrate(divu_poly.subs(sph) * rr ** 2 * sp.sin(th),
                          (ph, 0, 2 * sp.pi), (th, 0, sp.pi), (rr, 0, B))
    assert sp.simplify(S_flux - 4 * sp.pi * ALPHA * B ** 3) == 0
    r_lam = U.subs(r, B) - S_flux
    # --- potential sources --------------------------------------------------
    q_F = -laplacian(F["phi_core"], x) / fourpiG \
        + F["to_x"](rho_p) * F["phi_core"] + F["to_x"](nu)
    q_S = -laplacian(F["phi_shell"], x) / fourpiG - div(rho_u, x)
    q_out = sp.simplify(-laplacian(F["phi_out"], x) / fourpiG)
    assert q_out == 0, f"the exterior potential is not harmonic: {q_out}"
    dn = lambda e: (n.T * grad(e, x))[0]  # noqa: E731
    s_1 = (dn(F["phi_shell"]) - dn(F["phi_out"])) / fourpiG + RHO_S * un
    s_b = (dn(F["phi_shell"]) - dn(F["phi_core"])) / fourpiG \
        + (RHO_S - rho_F) * un
    hess = {k: sp.Matrix(dim, dim, lambda i, j: sp.diff(F["phi0"][k], x[i], x[j]))
            for k in F["phi0"]}
    gradp0 = {k: grad(F["phi0"][k], x) for k in F["phi0"]}
    return dict(F=F, n=n, sigma=sigma, gradu=gradu, f=f, T=T, T_cmb=T_cmb,
                g=g, A=A, rho_p=rho_p, D=D, N2=N2, Up=Up, phibar=phibar,
                flux_U=flux_U, r_U=r_U, R_w=R_w, r_s=r_s, r_nu=r_nu,
                S_flux=S_flux, r_lam=r_lam, q_F=q_F, q_S=q_S, s_1=s_1, s_b=s_b,
                hess=hess, gradp0=gradp0, rho_F=rho_F)


# --- emission -------------------------------------------------------------


def num(e) -> str:
    return repr(float(e))


def floatify(e):
    """Replace the rationals whose numerator or denominator does not fit a
    32-bit integer by 17-digit floats: the exact constants (b = 0.5467)
    otherwise print as integer literals too large for any C++ type."""
    e = sp.sympify(e)
    big = 2 ** 31
    reps = {a: sp.Float(a, 17) for a in e.atoms(sp.Rational)
            if abs(a.p) >= big or abs(a.q) >= big}
    return e.xreplace(reps)


def ccode(e):
    return solid.ccode(floatify(e))


def emit_scalar(name: str, expr, args: str) -> str:
    expr = floatify(expr)
    if not expr.free_symbols:   # a constant: no named argument (unused)
        unnamed = ", ".join(a.rsplit(" ", 1)[0] for a in args.split(", "))
        return f"inline double {name}({unnamed}) {{\n  return {ccode(expr)};\n}}\n"
    return solid.emit_scalar(name, expr, args)


def emit_vector(name: str, exprs, args: str) -> str:
    return solid.emit_vector(name, [floatify(e) for e in exprs], args)


def emit_level(name: str, expr, doc: str) -> str:
    """A level function of the core: `name_r(r)` of the radius and
    `name(c)` of the level value."""
    reps, (red,) = sp.cse([floatify(expr)], symbols=sp.numbered_symbols("t"))
    body = [f"  const double {s_} = {ccode(e)};" for s_, e in reps]
    body.append(f"  return {ccode(red)};")
    return (f"/// {doc}\n"
            f"inline double {name}_r(double r) {{\n" + "\n".join(body) + "\n}\n"
            f"inline double {name}(double c) {{ return {name}_r(level_radius(c)); }}\n")


def emit_dispatch_scalar(name: str) -> str:
    return (f"inline double {name}(const double* x) {{\n"
            "  const double r2 = x[0] * x[0] + x[1] * x[1] + x[2] * x[2];\n"
            f"  if (r2 < kCmbRadius * kCmbRadius) return {name}_core(x);\n"
            f"  if (r2 < kSurfaceRadius * kSurfaceRadius) return {name}_shell(x);\n"
            f"  return {name}_out(x);\n}}\n")


def emit_dispatch_vector(name: str) -> str:
    return (f"inline void {name}(const double* x, double* out) {{\n"
            "  const double r2 = x[0] * x[0] + x[1] * x[1] + x[2] * x[2];\n"
            f"  if (r2 < kCmbRadius * kCmbRadius) return {name}_core(x, out);\n"
            f"  if (r2 < kSurfaceRadius * kSurfaceRadius) return {name}_shell(x, out);\n"
            f"  {name}_out(x, out);\n}}\n")


def emit_namespace(D) -> str:
    F = D["F"]
    bg = F["bg"]
    X = "const double* x"
    XN = "const double* x, const double* n"
    XO = "const double* x, double* out"
    XNO = "const double* x, const double* n, double* out"
    r_of_c, a1, a2 = level_radius_expr(bg)
    level_radius = (
        "/// The radius of the level c in the core (Phi0 is a quadratic in r^2\n"
        "/// there): r^2 = 2 (c - c_min) / (a1 + sqrt(a1^2 - 4 a2 (c - c_min))),\n"
        "/// clamped to [c_min, c_w].\n"
        "inline double level_radius(double c) {\n"
        f"  constexpr double a1 = {num(a1)};\n"
        f"  constexpr double a2 = {num(a2)};\n"
        "  const double dc = std::fmin(std::fmax(c - kCMin, 0.0), kCWall - kCMin);\n"
        "  return std::sqrt(2.0 * dc / (a1 + std::sqrt(a1 * a1 - 4.0 * a2 * dc)));\n"
        "}\n")
    rho_x = F["to_x"](bg["rho_f"])
    parts = [
        "namespace d3 {\n",
        "constexpr int kDim = 3;\n",
        "\n// --- background ---------------------------------------------------\n",
        "/// The background potential Phi0 per region and by radius.\n",
        emit_scalar("phi0_core", F["phi0"]["core"], X),
        emit_scalar("phi0_shell", F["phi0"]["shell"], X),
        emit_scalar("phi0_out", F["phi0"]["out"], X),
        emit_dispatch_scalar("phi0"),
        "/// grad Phi0 (outwards) per region and by radius.\n",
        emit_vector("grad_phi0_core", D["gradp0"]["core"], XO),
        emit_vector("grad_phi0_shell", D["gradp0"]["shell"], XO),
        emit_vector("grad_phi0_out", D["gradp0"]["out"], XO),
        emit_dispatch_vector("grad_phi0"),
        "/// The Hessian of Phi0 (row-major) per region and by radius.\n",
        emit_vector("hess_phi0_core", list(D["hess"]["core"]), XO),
        emit_vector("hess_phi0_shell", list(D["hess"]["shell"]), XO),
        emit_vector("hess_phi0_out", list(D["hess"]["out"]), XO),
        emit_dispatch_vector("hess_phi0"),
        "/// The background density per region and by radius (0 in the buffer).\n",
        emit_scalar("rho_core", rho_x, X),
        emit_scalar("rho_shell", RHO_S + 0 * F["x"][0], X),
        emit_scalar("rho_out", sp.Integer(0), X),
        emit_dispatch_scalar("rho"),
        "/// The fluid's density gradient d rho / d Phi0 = g^-1 d_r rho\n"
        "/// (FluidRegion::density_gradient) and bulk modulus.\n",
        emit_scalar("rho_prime_fluid", F["to_x"](D["rho_p"]), X),
        emit_scalar("kappa_fluid", KAPPA_F + 0 * F["x"][0], X),
        "/// The shell's bulk and shear moduli.\n",
        emit_scalar("kappa", F["kappa"], X),
        emit_scalar("mu", F["mu"], X),
        "\n// --- level functions of the core ----------------------------------\n",
        level_radius,
        "/// The level value of the radius r in the core, Phi0(r).\n",
        "inline double level_value(double r) {\n"
        f"  return {ccode(bg['phi_core'])};\n}}\n",
        "/// |grad Phi0| in the core at radius r: dc = g dr.\n",
        "inline double level_gravity(double r) {\n"
        f"  return {ccode(bg['g_core'])};\n}}\n",
        emit_level("level_area", D["A"],
                   "A = oint dS / g = 4 pi r^2 / g, the volume per unit potential."),
        emit_level("level_density", bg["rho_f"], "The fluid density rho(c)."),
        emit_level("level_density_gradient", D["rho_p"],
                   "rho' = d rho / d Phi0."),
        emit_level("stratification", D["D"],
                   "D = rho' + rho^2 / kappa_f (< 0: stable)."),
        emit_level("buoyancy_frequency_squared", D["N2"],
                   "N^2 = -g^2 D / rho (> 0 for r > 0)."),
        "\n// --- the exact solution --------------------------------------------\n",
        "/// The exact displacement u* (shell).\n",
        emit_vector("u", F["u"], XO),
        "/// The exact displacement gradient du*_i/dx_j (row-major).\n",
        emit_vector("grad_u", list(D["gradu"]), XO),
        "/// The exact stress sigma(u*) (row-major).\n",
        emit_vector("stress", list(D["sigma"]), XO),
        "/// The exact potential per region and by radius.\n",
        emit_scalar("phi_core", F["phi_core"], X),
        emit_scalar("phi_shell", F["phi_shell"], X),
        emit_scalar("phi_out", F["phi_out"], X),
        emit_dispatch_scalar("phi"),
        emit_level("border_U", F["U"], "The exact border volume flux U*(c)."),
        emit_level("border_s", F["s"], "The exact lift s*(c)."),
        emit_level("border_nu", F["nu"], "The exact multiplier nu*(c)."),
        emit_level("level_integral_phi", D["phibar"],
                   "Phi_bar*(c) = oint phi* dS / g = A(c) a0(r(c))."),
        "\n// --- loads ---------------------------------------------------------\n",
        "/// The body force f of the displacement row (shell).\n",
        emit_vector("body_force", D["f"], XO),
        "/// The traction T on the surface r = 1, n the solid's outward normal.\n",
        emit_vector("traction_surface", D["T"], XNO),
        "/// The traction T_cmb on the CMB, n the SOLID's outward normal (into\n"
        "/// the core, -x/r): T plus the F2 and F3 terms and the wall pressure\n"
        "/// -lambda* (n.g/|g|) n.\n",
        emit_vector("traction_cmb", D["T_cmb"], XNO),
        "/// The volume source of the potential row in the core (with F1 and\n"
        "/// the border's +nu*(Phi0)) and in the shell; zero in the buffer.\n",
        emit_scalar("potential_source_core", D["q_F"], X),
        emit_scalar("potential_source_shell", D["q_S"], X),
        "/// The surface source of the potential row on r = 1, n = x/r.\n",
        emit_scalar("surface_source", D["s_1"], XN),
        "/// The surface source of the potential row on the CMB, n the SOLID's\n"
        "/// outward normal (-x/r).\n",
        emit_scalar("cmb_source", D["s_b"], XN),
        emit_level("border_U_flux", D["flux_U"],
                   "The U row's flux kappa_f U*'/A + rho s*: its data are\n"
                   "/// int N_k' flux dc + int N_k nu* dc + lambda* N_k(c_w)."),
        emit_level("border_U_source", D["r_U"],
                   "r_U = -(flux)' + nu*, the U row's source density against\n"
                   "/// dc; plus kBorderUWall on the wall node."),
        emit_level("border_s_source", D["r_s"],
                   "r_s = rho U*' - rho' A s* + A nu*, against dc."),
        emit_level("border_nu_source", D["r_nu"],
                   "r_nu = U* + s* A + Phi_bar*, against dc."),
        "}  // namespace d3\n",
    ]
    return "".join(parts)


def describe(D) -> str:
    F = D["F"]
    bg = F["bg"]
    P = F["pretty"]
    lines = [
        f"dimension 3 (b = {B}, buffer to {ROUT}):",
        f"  u*      = {F['u'].T}  (shell)",
        f"  phi*    = {F['phi_out']}  (r >= 1)",
        f"          = {P['phi_shell']}  (b <= r <= 1)",
        f"          = {P['phi_core']}  (r <= b)",
        f"  a0(r)   = {P['a0']}",
        f"  rho     = {bg['rho_f']} (core), {RHO_S} (shell)",
        f"  kappa_f = {KAPPA_F}; kappa = {F['kappa']}, mu = {F['mu']} (shell)",
        f"  U*(r)   = {P['U']}",
        f"  s*(r)   = {P['s']}",
        f"  nu*(r)  = {P['nu']}",
        f"  lambda* = {LAMBDA}",
        f"  S[u*]   = 4 pi alpha b^3 = {float(D['S_flux']):.10g} (alpha = {ALPHA})",
        f"  c_min   = {float(bg['c_min']):.12g}, c_w = {float(bg['c_w']):.12g}",
    ]
    return "\n".join(lines)


def header(D) -> str:
    F = D["F"]
    bg = F["bg"]
    doc = textwrap.indent(__doc__.strip(), " * ", lambda _: True)
    desc = textwrap.indent(describe(D), " * ", lambda _: True)
    consts = [
        ("kG", G, "gravitational constant"),
        ("kSurfaceRadius", RS, "the body's surface"),
        ("kCmbRadius", B, "the core-mantle boundary"),
        ("kOuterRadius", ROUT, "the buffer's outer sphere (DtN)"),
        ("kRhoSolid", RHO_S, "shell density"),
        ("kRhoCentre", RHO_C, "fluid density at the centre"),
        ("kBeta", BETA, "rho_f = rho_c (1 - beta r^2)"),
        ("kRhoCmb", bg["rho_f"].subs(r, B), "fluid-side density at the CMB"),
        ("kKappaFluid", KAPPA_F, "fluid bulk modulus"),
        ("kCMin", bg["c_min"], "Phi0 at the centre, the lowest level"),
        ("kCWall", bg["c_w"], "Phi0 on the CMB, the wall's level"),
        ("kLambda", LAMBDA, "the exact wall multiplier lambda*"),
        ("kWallFlux", D["S_flux"], "S[u*] = oint_CMB u*.x/r dS"),
        ("kBorderUWall", D["R_w"],
         "R_w: the U row's point source at the wall node"),
        ("kBorderLambdaSource", D["r_lam"], "l_lam = U*(c_w) - S[u*]"),
    ]
    out = [
        "// GENERATED by manufacture_fluid.py -- do not edit; change the fields\n",
        "// there and regenerate.\n",
        "/*\n", doc, "\n *\n * The fields of this header:\n", desc, "\n */\n",
        "#pragma once\n#include <cmath>\n#ifndef M_PI\n"
        "#define M_PI 3.14159265358979323846\n#endif\n\n",
        "namespace manufactured_fluid {\n\n",
    ]
    out += [f"constexpr double {k} = {num(v)};  ///< {d}\n" for k, v, d in consts]
    out += ["\n", emit_namespace(D), "\n}  // namespace manufactured_fluid\n"]
    return "".join(out)


def print_stratification(D) -> None:
    F = D["F"]
    bg = F["bg"]
    print("stratification of the core (stable where N^2 > 0, D < 0):")
    print("     r        c          rho        rho'        D           N^2")
    for rv in (0.0, 0.1, 0.2, 0.3, 0.4, 0.5, float(B)):
        vals = [float(e.subs(r, rv)) for e in
                (bg["phi_core"], bg["rho_f"], D["rho_p"], D["D"], D["N2"])]
        print(f"  {rv:6.4f}  " + "  ".join(f"{v: .4e}" for v in vals))


# --- the numerical check ---------------------------------------------------


def check(D, seed: int = 20261105) -> float:
    """Evaluate both sides of every row's weak identity by quadrature with
    random tests; print the relative residuals; return the largest."""
    import numpy as np

    rng = np.random.default_rng(seed)
    F = D["F"]
    bg = F["bg"]
    x = list(F["x"])
    fourpiG = float(4 * sp.pi * G)
    b, rs, rout = float(B), float(RS), float(ROUT)

    def lam_x(e):
        f = sp.lambdify(x, e, "numpy")
        return lambda P: np.broadcast_to(np.asarray(f(*P.T), float),
                                         P.shape[:1]).copy()

    def lam_vec(es):
        fs = [lam_x(e) for e in es]
        return lambda P: np.stack([fi(P) for fi in fs], axis=-1)

    def lam_r(e):
        f = sp.lambdify([r], e, "numpy")
        return lambda R: np.broadcast_to(np.asarray(f(R), float), R.shape).copy()

    # Quadrature: Gauss in r and mu = cos(theta), trapezoid in longitude.
    nmu, naz = 40, 80
    mu_, wmu = np.polynomial.legendre.leggauss(nmu)
    az = 2 * np.pi * np.arange(naz) / naz
    waz = np.full(naz, 2 * np.pi / naz)
    MU, AZ = np.meshgrid(mu_, az, indexing="ij")
    st = np.sqrt(1 - MU ** 2)
    dirs = np.stack([st * np.cos(AZ), st * np.sin(AZ), MU], -1).reshape(-1, 3)
    wdir = np.outer(wmu, waz).reshape(-1)

    def sphere(R):
        return R * dirs, dirs, R * R * wdir

    def shell(r0, r1, nr=28):
        t, wt = np.polynomial.legendre.leggauss(nr)
        R = 0.5 * (r1 - r0) * t + 0.5 * (r1 + r0)
        WR = 0.5 * (r1 - r0) * wt
        P = (R[:, None, None] * dirs[None]).reshape(-1, 3)
        W = (WR[:, None] * R[:, None] ** 2 * wdir[None]).reshape(-1)
        return P, W, np.repeat(R, dirs.shape[0])

    results = []

    def report(name, lhs_terms, rhs):
        lhs = sum(lhs_terms)
        scale = max(max(abs(t) for t in lhs_terms), abs(rhs))
        rel = abs(lhs - rhs) / scale
        results.append(rel)
        print(f"  {name:34s} lhs {lhs: .10e}  rhs {rhs: .10e}  rel {rel:.2e}")

    # Exact fields and coefficients.
    u = lam_vec(F["u"])
    gu = lam_vec(list(D["gradu"]))
    kap, mu_s = lam_x(F["kappa"]), lam_x(F["mu"])
    g_sh = lam_vec(D["gradp0"]["shell"])
    H_sh = lam_vec(list(D["hess"]["shell"]))
    phi = {k: lam_x(F[f"phi_{k}"]) for k in ("core", "shell", "out")}
    gphi = {k: lam_vec(grad(F[f"phi_{k}"], x)) for k in ("core", "shell", "out")}
    rho_F = lam_x(D["rho_F"])
    rho_p_x = lam_x(F["to_x"](D["rho_p"]))
    nu_x = lam_x(F["to_x"](F["nu"]))
    lam = float(LAMBDA)
    rho_s = float(RHO_S)

    def sub_n(e, nvec):
        return e.subs({D["n"][i]: nvec[i] for i in range(3)})

    rhat = [xi / sp.sqrt(F["r2"]) for xi in x]
    f_body = lam_vec(D["f"])
    T_surf = lam_vec(sub_n(D["T"], rhat))
    T_cmb = lam_vec(sub_n(D["T_cmb"], [-e for e in rhat]))
    q_F, q_S = lam_x(D["q_F"]), lam_x(D["q_S"])
    s_1 = lam_x(sub_n(D["s_1"], rhat))
    s_b = lam_x(sub_n(D["s_b"], [-e for e in rhat]))

    # Continuity of the background and of phi* at the interfaces.
    print("interfaces (largest mismatch over a sphere):")
    for R, ka, kb in ((b, "core", "shell"), (rs, "shell", "out")):
        P, _, _ = sphere(R)
        p0a, p0b = lam_x(F["phi0"][ka]), lam_x(F["phi0"][kb])
        print(f"  r = {R}: Phi0 {np.abs(p0a(P) - p0b(P)).max():.2e}, "
              f"phi* {np.abs(phi[ka](P) - phi[kb](P)).max():.2e}")

    def rand_poly(deg=3):
        mons = [x[0] ** i * x[1] ** j * x[2] ** k for i in range(deg + 1)
                for j in range(deg + 1 - i) for k in range(deg + 1 - i - j)]
        return sum(sp.Float(rng.normal()) * mm for mm in mons)

    # --- (u) row ------------------------------------------------------------
    print("displacement row, random cubic v:")
    Ps, Ws, _ = shell(b, rs)
    Pc_, Nc_, Wcb = sphere(b)
    m_cmb = -Nc_
    P1, N1, W1 = sphere(rs)
    for trial in range(4):
        vs = sp.Matrix([rand_poly() for _ in range(3)])
        v = lam_vec(vs)
        gv = lam_vec([sp.diff(vs[i], x[j]) for i in range(3) for j in range(3)])
        # a_S(u*, v) + rho grad phi*.v, by the product rule.
        U_, GU, V, GV = u(Ps), gu(Ps).reshape(-1, 3, 3), v(Ps), gv(Ps).reshape(-1, 3, 3)
        g_, H_ = g_sh(Ps), H_sh(Ps).reshape(-1, 3, 3)
        divu, divv = np.trace(GU, axis1=1, axis2=2), np.trace(GV, axis1=1, axis2=2)
        eu, ev = 0.5 * (GU + GU.transpose(0, 2, 1)), 0.5 * (GV + GV.transpose(0, 2, 1))
        devdev = np.einsum("pij,pij->p", eu, ev) - divu * divv / 3
        grad_ug = np.einsum("pik,pi->pk", GU, g_) + np.einsum("pik,pi->pk", H_, U_)
        grad_vg = np.einsum("pik,pi->pk", GV, g_) + np.einsum("pik,pi->pk", H_, V)
        ug, vg = np.einsum("pi,pi->p", U_, g_), np.einsum("pi,pi->p", V, g_)
        elastic = kap(Ps) * divu * divv + 2 * mu_s(Ps) * devdev
        gravity = 0.5 * rho_s * (np.einsum("pi,pi->p", grad_ug, V)
                                 + np.einsum("pi,pi->p", grad_vg, U_)) \
            - 0.5 * rho_s * (ug * divv + vg * divu)
        coupling = rho_s * np.einsum("pi,pi->p", gphi["shell"](Ps), V)
        Uc, Vc, gc = u(Pc_), v(Pc_), g_sh(Pc_)
        mg = np.einsum("pi,pi->p", m_cmb, gc)
        mu_c, mv_c = np.einsum("pi,pi->p", m_cmb, Uc), np.einsum("pi,pi->p", m_cmb, Vc)
        F2 = -np.sum(Wcb * rho_F(Pc_) * mg * mu_c * mv_c)
        F3 = -np.sum(Wcb * rho_F(Pc_) * phi["shell"](Pc_) * mv_c)
        wall = lam * np.sum(Wcb * (-mg / np.linalg.norm(gc, axis=1)) * mv_c)
        terms = [np.sum(Ws * elastic), np.sum(Ws * gravity),
                 np.sum(Ws * coupling), F2, F3, wall]
        rhs = (np.sum(Ws * np.einsum("pi,pi->p", f_body(Ps), V))
               + np.sum(W1 * np.einsum("pi,pi->p", T_surf(P1), v(P1)))
               + np.sum(Wcb * np.einsum("pi,pi->p", T_cmb(Pc_), Vc)))
        report(f"u row, test {trial}", terms, rhs)
        if trial == 0:
            print("    terms: elastic {:.3e} gravity {:.3e} coupling {:.3e} "
                  "F2 {:.3e} F3 {:.3e} wall {:.3e}".format(*terms))

    # --- (phi) row ------------------------------------------------------------
    print("potential row, random cubic chi:")
    Pc, Wc, _ = shell(0.0, b, 36)
    Pb, Wb, _ = shell(rs, rout)
    PR, NR, WR = sphere(rout)
    for trial in range(4):
        cs = rand_poly()
        chi = lam_x(cs)
        gchi = lam_vec(grad(cs, x))
        lap_terms = [np.sum(Wq * np.einsum("pi,pi->p", gphi[k](Pq), gchi(Pq)))
                     / fourpiG for k, Pq, Wq in (("core", Pc, Wc), ("shell", Ps, Ws),
                                                  ("out", Pb, Wb))]
        dtn = -np.sum(WR * chi(PR) * np.einsum("pi,pi->p", NR, gphi["out"](PR))) \
            / fourpiG
        F1 = np.sum(Wc * rho_p_x(Pc) * phi["core"](Pc) * chi(Pc))
        nu_t = np.sum(Wc * nu_x(Pc) * chi(Pc))
        c_vol = np.sum(Ws * rho_s * np.einsum("pi,pi->p", gchi(Ps), u(Ps)))
        c_cmb = -np.sum(Wcb * rho_F(Pc_) * chi(Pc_)
                        * np.einsum("pi,pi->p", m_cmb, u(Pc_)))
        terms = lap_terms + [dtn, F1, nu_t, c_vol, c_cmb]
        rhs = (np.sum(Wc * q_F(Pc) * chi(Pc)) + np.sum(Ws * q_S(Ps) * chi(Ps))
               + np.sum(W1 * s_1(P1) * chi(P1)) + np.sum(Wcb * s_b(Pc_) * chi(Pc_)))
        report(f"phi row, test {trial}", terms, rhs)
        if trial == 0:
            print("    terms: lap core {:.3e} shell {:.3e} buffer {:.3e} DtN {:.3e} "
                  "F1 {:.3e} nu {:.3e} c vol {:.3e} c cmb {:.3e}".format(*terms))

    # --- border rows ----------------------------------------------------------
    print("border rows, random hat tests on random level grids:")
    c_min, c_w = float(bg["c_min"]), float(bg["c_w"])
    c_of_r = lam_r(bg["phi_core"])
    g_r = lam_r(bg["g_core"])
    r_of_c = sp.lambdify([c], level_radius_expr(bg)[0], "numpy")
    Lr = {k: lam_r(D[k]) for k in ("A", "rho_p", "Up", "phibar", "flux_U",
                                   "r_U", "r_s", "r_nu")}
    Lr.update({k: lam_r(F[k]) for k in ("U", "s", "nu")})
    rho_r = lam_r(bg["rho_f"])
    a0_r = lam_r(F["a0"])
    kf = float(KAPPA_F)
    R_w = float(D["R_w"])
    cr = np.linspace(c_min, c_w, 7)
    print(f"  r(c) inverse, largest error {np.abs(c_of_r(r_of_c(cr)) - cr).max():.2e}")

    def hats(coef, cgrid):
        """W(c) = sum coef_k N_k(c) and W'(c), at given c values."""
        def W(cv):
            return np.interp(cv, cgrid, coef)

        def Wp(cv):
            e = np.clip(np.searchsorted(cgrid, cv) - 1, 0, len(cgrid) - 2)
            return (coef[e + 1] - coef[e]) / (cgrid[e + 1] - cgrid[e])
        return W, Wp

    def level_quad(cgrid, nq=24):
        """Gauss in r per level element: the c values, radii and dc weights."""
        t, wt = np.polynomial.legendre.leggauss(nq)
        rg = r_of_c(cgrid)
        Rs, Ws_ = [], []
        for e in range(len(cgrid) - 1):
            R = 0.5 * (rg[e + 1] - rg[e]) * t + 0.5 * (rg[e + 1] + rg[e])
            Rs.append(R)
            Ws_.append(0.5 * (rg[e + 1] - rg[e]) * wt * g_r(R))
        R = np.concatenate(Rs)
        return c_of_r(R), R, np.concatenate(Ws_)

    def fluid_quad(cgrid, nq=24):
        """3-D quadrature over the core, split at the grid's radii."""
        rg = r_of_c(cgrid)
        out = [shell(rg[e], rg[e + 1], nq) for e in range(len(cgrid) - 1)]
        P = np.concatenate([o[0] for o in out])
        W = np.concatenate([o[1] for o in out])
        R = np.concatenate([o[2] for o in out])
        return P, W, R

    for trial in range(3):
        n_el = int(rng.integers(4, 10))
        interior = np.sort(rng.uniform(c_min, c_w, n_el - 1))
        cgrid = np.concatenate([[c_min], interior, [c_w]])
        cq, Rq, Wq = level_quad(cgrid)
        Pf, Wf, Rf = fluid_quad(cgrid)
        cf = c_of_r(Rf)
        # U row (W(c_min) = 0).
        coef = rng.normal(size=n_el + 1)
        coef[0] = 0.0
        Wf_, Wpf = hats(coef, cgrid)
        terms = [np.sum(Wq * kf * Lr["Up"](Rq) / Lr["A"](Rq) * Wpf(cq)),
                 np.sum(Wq * rho_r(Rq) * Lr["s"](Rq) * Wpf(cq)),
                 np.sum(Wq * Lr["nu"](Rq) * Wf_(cq)),
                 lam * coef[-1]]
        rhs = np.sum(Wq * Lr["r_U"](Rq) * Wf_(cq)) + R_w * coef[-1]
        report(f"U row, grid {trial} ({n_el} el.)", terms, rhs)
        # s row: the A-weighted terms as fluid volume integrals.
        coef = rng.normal(size=n_el + 1)
        Sg, _ = hats(coef, cgrid)
        terms = [np.sum(Wq * rho_r(Rq) * Lr["Up"](Rq) * Sg(cq)),
                 -np.sum(Wf * Lr["rho_p"](Rf) * Lr["s"](Rf) * Sg(cf)),
                 np.sum(Wf * Lr["nu"](Rf) * Sg(cf))]
        rhs = np.sum(Wq * Lr["r_s"](Rq) * Sg(cq))
        report(f"s row, grid {trial}", terms, rhs)
        # nu row: int eta U dc + int_F eta(Phi0) (s(Phi0) + phi*) dV.
        coef = rng.normal(size=n_el + 1)
        Eg, _ = hats(coef, cgrid)
        terms = [np.sum(Wq * Lr["U"](Rq) * Eg(cq)),
                 np.sum(Wf * Lr["s"](Rf) * Eg(cf)),
                 np.sum(Wf * phi["core"](Pf) * Eg(cf))]
        rhs = np.sum(Wq * Lr["r_nu"](Rq) * Eg(cq))
        report(f"nu row, grid {trial}", terms, rhs)
        # The level integral alone: int_F eta(Phi0) phi* dV = int eta A a0 dc.
        report(f"  level integral Phi_bar, grid {trial}",
               [np.sum(Wf * phi["core"](Pf) * Eg(cf))],
               np.sum(Wq * Lr["A"](Rq) * a0_r(Rq) * Eg(cq)))
    # lambda row: S[u*] by quadrature on the CMB.
    S_num = np.sum(Wcb * np.einsum("pi,pi->p", Nc_, u(Pc_)))
    report("lambda row: U*(c_w) - S[u*]",
           [float(F["U"].subs(r, B)), -S_num], float(D["r_lam"]))
    worst = max(results)
    print(f"largest relative residual {worst:.2e}")
    return worst


def galerkin(D, counts=(4, 8, 16, 32, 64)) -> None:
    """The border rows alone, by P1 Galerkin on a uniform grid in c of
    `counts` nodes (the class's level count: SetLevelBorder(levels)) with
    EXACT level integrals, the potential's level integral Phi_bar* and
    the wall flux S[u*] as data (the lambda row then fixes U(c_w)): the
    one-dimensional discretisation's own error, against which the
    class's binned blocks are measured. Prints, per level count, the
    largest nodal error over the largest exact value and the
    volume-weighted L2 error (int A (v_h - v*)^2 dc, relative) of U, s
    and nu, and the wall multiplier's relative error."""
    import numpy as np

    F = D["F"]
    bg = F["bg"]

    def lam_r(e):
        f = sp.lambdify([r], e, "numpy")
        return lambda R: np.broadcast_to(np.asarray(f(R), float), R.shape).copy()

    A, rho, rp, g = (lam_r(e) for e in (D["A"], bg["rho_f"], D["rho_p"],
                                         bg["g_core"]))
    c_of_r, phibar = lam_r(bg["phi_core"]), lam_r(D["phibar"])
    Ue, se, nue = lam_r(F["U"]), lam_r(F["s"]), lam_r(F["nu"])
    flux, r_s, r_nu = lam_r(D["flux_U"]), lam_r(D["r_s"]), lam_r(D["r_nu"])
    r_of_c = sp.lambdify([c], level_radius_expr(bg)[0], "numpy")
    kf, lam_ex = float(KAPPA_F), float(LAMBDA)
    c_min, c_w = float(bg["c_min"]), float(bg["c_w"])
    U_wall = float(D["r_lam"] + D["S_flux"])
    t, wt = np.polynomial.legendre.leggauss(30)
    print("the border rows by P1 Galerkin with exact level integrals:")
    print(" levels U max     s max     nu max    U vol     s vol     nu vol"
          "    lambda")
    for levels in counts:
        n = levels - 1
        cg = np.linspace(c_min, c_w, n + 1)
        rg = np.concatenate([[0.0], r_of_c(cg[1:])])
        m = n + 1
        KUU, KUs, KUn, Kss, Ksn = (np.zeros((m, m)) for _ in range(5))
        rU, rs, rn, pb = (np.zeros(m) for _ in range(4))
        quad = []
        for e in range(n):
            R = 0.5 * (rg[e + 1] - rg[e]) * t + 0.5 * (rg[e + 1] + rg[e])
            W = 0.5 * (rg[e + 1] - rg[e]) * wt * g(R)       # dc
            h = cg[e + 1] - cg[e]
            C = c_of_r(R)
            N = [(cg[e + 1] - C) / h, (C - cg[e]) / h]
            d = [-1.0 / h, 1.0 / h]
            quad.append((R, W, N))
            for i in range(2):
                I = e + i
                rU[I] += np.sum(W * (d[i] * flux(R) + N[i] * nue(R)))
                rs[I] += np.sum(W * N[i] * r_s(R))
                rn[I] += np.sum(W * N[i] * r_nu(R))
                pb[I] += np.sum(W * N[i] * phibar(R))
                for j in range(2):
                    J = e + j
                    KUU[I, J] += np.sum(W * kf / A(R)) * d[i] * d[j]
                    KUs[I, J] += np.sum(W * rho(R) * N[j]) * d[i]
                    KUn[I, J] += np.sum(W * N[i] * N[j])
                    Kss[I, J] -= np.sum(W * rp(R) * A(R) * N[i] * N[j])
                    Ksn[I, J] += np.sum(W * A(R) * N[i] * N[j])
        rU[n] += lam_ex
        # Unknowns U_1..U_n (U_0 = 0 at the centre), s_0..s_n, nu_0..nu_n,
        # lambda; the potential's columns moved to the data.
        oS, oN = n, n + m
        size = n + 2 * m + 1
        K, b = np.zeros((size, size)), np.zeros(size)
        for i in range(1, m):
            K[i - 1, 0:n] = KUU[i, 1:]
            K[i - 1, oS:oS + m] = KUs[i]
            K[oS:oS + m, i - 1] = KUs[i]
            K[i - 1, oN:oN + m] = KUn[i]
            K[oN:oN + m, i - 1] = KUn[i]
            b[i - 1] = rU[i]
        K[oS:oS + m, oS:oS + m] = Kss
        K[oS:oS + m, oN:oN + m] = Ksn
        K[oN:oN + m, oS:oS + m] = Ksn
        b[oS:oS + m] = rs
        b[oN:oN + m] = rn - pb
        K[n - 1, -1] = K[-1, n - 1] = 1.0
        b[-1] = U_wall
        x = np.linalg.solve(K, b)
        nodes = r_of_c(cg)
        vals = {"U": (np.concatenate([[0.0], x[:n]]), Ue),
                "s": (x[oS:oS + m], se), "nu": (x[oN:oN + m], nue)}

        def errors(vh, f):
            mx = np.abs(vh - f(nodes)).max() / np.abs(f(nodes)).max()
            num = den = 0.0
            for e, (R, W, N) in enumerate(quad):
                v = vh[e] * N[0] + vh[e + 1] * N[1]
                num += np.sum(W * A(R) * (v - f(R)) ** 2)
                den += np.sum(W * A(R) * f(R) ** 2)
            return mx, np.sqrt(num / den)

        E = {k: errors(*v) for k, v in vals.items()}
        print(f"  {levels:3d}  " + "  ".join(f"{E[k][0]:.2e}" for k in E) + "  "
              + "  ".join(f"{E[k][1]:.2e}" for k in E)
              + f"  {abs(x[-1] - lam_ex) / lam_ex:.2e}")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--out", type=Path, default=Path(__file__).resolve().parent,
                   help="directory of fields_fluid.hpp (default: beside this script)")
    p.add_argument("--check", action="store_true",
                   help="verify the derivation numerically (no header written)")
    p.add_argument("--galerkin", action="store_true",
                   help="solve the border rows alone by P1 Galerkin with "
                        "exact level integrals and print their errors "
                        "(no header written)")
    args = p.parse_args()
    D = derive()
    print(describe(D))
    print_stratification(D)
    if args.galerkin:
        galerkin(D)
        return
    if args.check:
        worst = check(D)
        raise SystemExit(0 if worst < 1e-8 else 1)
    path = args.out / "fields_fluid.hpp"
    path.write_text(header(D))
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
