"""Exact reference for the no-gravity disc family (rung 0).

A fluid disc 0 <= r <= c (bulk modulus kappa_f, no shear, no
pre-stress) inside a solid annulus c <= r <= a (plane strain, Lame
lam, mu), no gravity. Load: boundary traction on r = a of one Fourier
degree, sigma.rhat = -t cos(l theta) rhat (shear-free), except at
l = 1 where the self-equilibrated variant sigma_rt = -t sin(theta) is
added so the net force vanishes (doc/planning/no_gravity_analytics.md).

Without gravity the fluid's pressure perturbation is constant, so for
l >= 1 the interface is exactly traction-free and the solid solves a
free-inner-boundary annulus problem; at l = 0 the fluid is an area
spring p1 = -2 kappa_f u_r(c) / c. Welded and slipping formulations
share these solutions exactly.

The solution basis is built from the Navier power-law ansatz
u_r = u r^k cos(l theta), u_theta = v r^k sin(l theta) rather than
from the Michell tables: with d = u(k+1) + l v and w = v(k+1) + l u
(the div and curl amplitudes), Navier reduces to

    (lam + 2 mu)(k - 1) d = mu l w,
    (lam + 2 mu) l d    = mu (k - 1) w,

whose solvability gives the exponents k = 1 +/- l (d, w modes) and,
with d = w = 0, k = -1 +/- l. The same construction generalises to
the pre-stressed rung, where only the symbol changes. At l = 1 the
exponents k = 1 - l and k = l - 1 collide at zero (the translation /
logarithmic pair); the translation is gauge and the logarithmic
solution carries exactly the net force, so the self-equilibrated load
is solved in the two-mode basis k = +/- 2 and the overdetermined
boundary system must be consistent -- its residual is asserted.

Everything is a single power of r per basis column, so the boundary
system is M(c) x = b with each entry coef * c^p, and the interface
derivative is exact:

    dx/dc = M^{-1} (-dM/dc x),      dM/dc entry = p / c * entry.

Responses and their exact c-derivatives are what the FE disc runs
compare against, absolute and through the shift maps.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class DiscModel:
    """Plane-strain annulus (lam, mu) on [c, a] over a fluid disc of
    bulk modulus kappa_f; t is the load amplitude."""
    lam: float
    mu: float
    kappa_f: float
    c: float
    a: float
    t: float = 1.0


@dataclass(frozen=True)
class Mode:
    """One power-law solution: u_r = u r^k cos, u_th = v r^k sin."""
    k: float
    u: float
    v: float
    d: float  # div amplitude u(k+1) + l v
    w: float  # curl amplitude v(k+1) + l u


def modes(lam: float, mu: float, l: int) -> list[Mode]:
    """The solution basis of degree l (2 modes at l in {0, 1}, else 4)."""
    if l == 0:
        # u_r = A r + B / r, u_theta = 0.
        return [Mode(1.0, 1.0, 0.0, 2.0, 0.0),
                Mode(-1.0, 1.0, 0.0, 0.0, 0.0)]
    out: list[Mode] = []
    # d, w modes, k = 1 +/- l (k = 1 - l dropped at l = 1: the
    # logarithmic pair carries net force and is excluded by the
    # self-equilibrated load).
    for k in ([1.0 + l] if l == 1 else [1.0 + l, 1.0 - l]):
        d = 1.0
        w = (lam + 2.0 * mu) * l * d / (mu * (k - 1.0))
        A = np.array([[k + 1.0, float(l)], [float(l), k + 1.0]])
        u, v = np.linalg.solve(A, np.array([d, w]))
        out.append(Mode(k, float(u), float(v), d, w))
    # d = w = 0 modes, k = -1 +/- l (k = l - 1 is the rigid
    # translation at l = 1: zero stress, gauge, excluded).
    for k in ([-1.0 - l] if l == 1 else [l - 1.0, -1.0 - l]):
        u = 1.0
        v = -u * (k + 1.0) / l
        out.append(Mode(k, u, v, 0.0, 0.0))
    return out


def _rows(model: DiscModel, l: int, m: Mode):
    """Per-mode amplitudes (coef, power) of the fields entering the
    boundary conditions and the responses: all pure powers of r."""
    lam, mu = model.lam, model.mu
    srr = (lam * m.d + 2.0 * mu * m.u * m.k, m.k - 1.0)
    srt = (mu * (m.v * (m.k - 1.0) - l * m.u), m.k - 1.0)
    ur = (m.u, m.k)
    ut = (m.v, m.k)
    return srr, srt, ur, ut


def _system(model: DiscModel, l: int):
    """The boundary system M(c) x = b, its exact c-derivative dM, and
    the response rows (value rows R and their explicit c-derivative
    rows dR for the interface responses)."""
    ms = modes(model.lam, model.mu, l)
    c, a, t = model.c, model.a, model.t
    n = len(ms)

    def at(pair, r):
        coef, p = pair
        return coef * r ** p

    def dat(pair, r):
        coef, p = pair
        return coef * p * r ** (p - 1.0)

    rows_a, rows_c, drows_c = [], [], []
    resp, dresp = {}, {}
    SRR, SRT, UR, UT = range(4)
    per_mode = [_rows(model, l, m) for m in ms]

    def row(which, r, deriv=False):
        f = dat if deriv else at
        return np.array([f(pm[which], r) for pm in per_mode])

    if l == 0:
        # sigma_rr(a) = -t ; sigma_rr(c) = -p1 = +(2 kappa_f / c) u_r(c),
        # i.e. sigma_rr(c) - (2 kappa_f / c) u_r(c) = 0.
        M = np.vstack([row(SRR, a),
                       row(SRR, c) - 2.0 * model.kappa_f / c * row(UR, c)])
        b = np.array([-t, 0.0])
        # the spring row is a pure power of c too: d/dc applies to both
        # pieces, and d/dc[(2k/c) u r^k at r=c] = 2k u (k-1) c^{k-2}.
        spring = np.array([2.0 * model.kappa_f * pm[UR][0]
                           * (pm[UR][1] - 1.0) * c ** (pm[UR][1] - 2.0)
                           for pm in per_mode])
        dM = np.vstack([np.zeros(n), row(SRR, c, True) - spring])
    else:
        srt_a = -t if l == 1 else 0.0  # self-equilibrated at l = 1
        M = np.vstack([row(SRR, a), row(SRT, a),
                       row(SRR, c), row(SRT, c)])
        b = np.array([-t, srt_a, 0.0, 0.0])
        dM = np.vstack([np.zeros(n), np.zeros(n),
                        row(SRR, c, True), row(SRT, c, True)])

    resp["ur_a"], dresp["ur_a"] = row(UR, a), np.zeros(n)
    resp["ut_a"], dresp["ut_a"] = row(UT, a), np.zeros(n)
    resp["ur_c"], dresp["ur_c"] = row(UR, c), row(UR, c, True)
    resp["ut_c"], dresp["ut_c"] = row(UT, c), row(UT, c, True)
    return ms, M, b, dM, resp, dresp


@dataclass(frozen=True)
class DiscSolution:
    """Response coefficients (of cos l theta for u_r, sin l theta for
    u_theta) and their exact d/dc, at the surface and the interface;
    p1 is the fluid pressure (l = 0 only, else 0)."""
    l: int
    x: np.ndarray
    responses: dict[str, float]
    derivatives: dict[str, float]
    p1: float
    residual: float


def solve(model: DiscModel, l: int) -> DiscSolution:
    """The exact solution and its exact interface derivative."""
    ms, M, b, dM, resp, dresp = _system(model, l)
    if M.shape[0] == M.shape[1]:
        x = np.linalg.solve(M, b)
        residual = float(np.linalg.norm(M @ x - b))
        dx = np.linalg.solve(M, -dM @ x)
    else:
        # l = 1: four conditions, two admissible modes; the
        # self-equilibrated load makes the system consistent.
        x, *_ = np.linalg.lstsq(M, b, rcond=None)
        residual = float(np.linalg.norm(M @ x - b))
        dx, *_ = np.linalg.lstsq(M, -dM @ x, rcond=None)
    responses = {k: float(r @ x) for k, r in resp.items()}
    derivatives = {k: float(resp[k] @ dx + dresp[k] @ x) for k in resp}
    p1 = (-2.0 * model.kappa_f * responses["ur_c"] / model.c
          if l == 0 else 0.0)
    return DiscSolution(l, x, responses, derivatives, p1, residual)


def displacement(model: DiscModel, sol: DiscSolution):
    """(U(r), V(r)) radial functions of the solved annulus field."""
    ms = modes(model.lam, model.mu, sol.l)

    def U(r):
        r = np.asarray(r, dtype=float)
        return sum(a * m.u * r ** m.k for a, m in zip(sol.x, ms))

    def V(r):
        r = np.asarray(r, dtype=float)
        return sum(a * m.v * r ** m.k for a, m in zip(sol.x, ms))

    return U, V


def navier_residual(model: DiscModel, sol: DiscSolution,
                    n: int = 4000) -> float:
    """Finite-difference residual of the two radial Navier equations on
    the solved field, relative to the term scale: an independent check
    of the basis construction (the only non-exact step is the FD)."""
    lam2mu, mu, l = model.lam + 2.0 * model.mu, model.mu, sol.l
    r = np.linspace(model.c, model.a, n)
    U, V = displacement(model, sol)
    u, v = U(r), V(r)
    du, dv = np.gradient(u, r), np.gradient(v, r)
    D = du + u / r + l * v / r
    W = dv + v / r + l * u / r
    R1 = lam2mu * np.gradient(D, r) - mu * l * W / r
    R2 = lam2mu * l * D / r - mu * np.gradient(W, r)
    scale = np.max(np.abs(lam2mu * np.gradient(D, r))) + np.max(
        np.abs(mu * np.gradient(W, r))) + 1e-300
    inner = slice(2, -2)  # one-sided-difference ends excluded
    return float(max(np.max(np.abs(R1[inner])), np.max(np.abs(R2[inner])))
                 / scale)


def _check(label: str, value: float, tol: float) -> str:
    flag = "ok" if value <= tol else "FAIL"
    return f"  {label:<52s} {value:10.3e}  {flag}"


def validate(lmax: int = 6, verbose: bool = True) -> bool:
    """Internal checks; returns True when everything passes."""
    model = DiscModel(lam=2.0, mu=1.0, kappa_f=1.5, c=0.55, a=1.0)
    lines, ok = [], True

    def record(label, value, tol):
        nonlocal ok
        ok &= value <= tol
        lines.append(_check(label, value, tol))

    # l = 0 limits: kappa_f -> 0 is the open hole, -> infinity pins
    # u_r(c); both against independently coded Lame formulas.
    lam, mu, c, a = model.lam, model.mu, model.c, model.a
    for kf, label in ((0.0, "open hole"), (1e12, "rigid-volume core")):
        s = solve(DiscModel(lam, mu, kf, c, a), 0)
        A_, B_ = np.linalg.solve(
            np.array([[2 * (lam + mu), -2 * mu / a ** 2],
                      [2 * (lam + mu), -2 * mu / c ** 2]]),
            np.array([-1.0, 0.0]))
        if kf == 0.0:
            record(f"l=0 {label} vs Lame", abs(s.responses["ur_a"]
                   - (A_ * a + B_ / a)), 1e-12)
        else:
            record(f"l=0 {label}: |u_r(c)|", abs(s.responses["ur_c"]),
                   1e-10)

    # Navier residual and boundary-system residual per degree.
    for l in range(0, lmax + 1):
        s = solve(model, l)
        record(f"l={l} boundary-system residual", s.residual, 1e-9)
        if l >= 1:
            record(f"l={l} Navier FD residual", navier_residual(model, s),
                   1e-5)

    # Exact derivative vs central finite difference in c.
    eps = 1e-6
    for l in range(0, lmax + 1):
        s = solve(model, l)
        sp = solve(DiscModel(lam, mu, model.kappa_f, c + eps, a), l)
        sm = solve(DiscModel(lam, mu, model.kappa_f, c - eps, a), l)
        for key in ("ur_a", "ut_a"):
            if l == 0 and key == "ut_a":
                continue
            fd = (sp.responses[key] - sm.responses[key]) / (2 * eps)
            scale = max(abs(fd), 1e-12)
            record(f"l={l} d{key}/dc exact vs FD",
                   abs(s.derivatives[key] - fd) / scale, 1e-7)

    if verbose:
        print("disc_reference validation "
              f"({'PASS' if ok else 'FAIL'}):")
        print("\n".join(lines))
    return ok


if __name__ == "__main__":
    raise SystemExit(0 if validate() else 1)
