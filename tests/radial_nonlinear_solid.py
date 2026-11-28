"""Radial reference for the self-gravitating solid at finite strain: the
numbers of TestNonlinearSelfGravitating.UniformLoadOnABall.

A homogeneous ball of radius a and referential density rho, in hydrostatic
equilibrium (p0 = (2 pi / 3) G rho^2 (a^2 - R^2)), with the quadratic law
about that state, S = -p0 1 + lam (tr E) 1 + 2 mu E (lam, mu the bare
moduli, constant), under a uniform surface mass sigma per unit reference
area carried by the material surface.

In Lagrangian form the unknown is the deformed radius r(R). The mass
inside a material sphere does not change, so gravity at a particle is
G m(R) / r^2 and no potential is solved for, which makes this independent
of the referential potential of the finite-element problem:

    d P_rr / dR + 2 (P_rr - P_tt) / R = rho G m(R) / r^2,
    P_rr = r' S_rr,  P_tt = (r / R) S_tt,
    E_rr = (r'^2 - 1) / 2,  E_tt = ((r / R)^2 - 1) / 2,
    P_rr(a) = -sigma G (M + M_load / 2) / r(a)^2,  M_load = 4 pi a^2 sigma

(the load is pulled by the body and by half of itself). Solved by shooting
on the stretch at the centre. `linear=True` solves the problem linearised
about the reference.

    python radial_nonlinear_solid.py

prints u_r(a) for a ladder of loads; the difference between the two falls
as the load squared. Needs numpy and scipy.
"""
import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import brentq

G = 1.0


def solve(sigma, *, a=1.0, rho=1.0, lam=5.0, mu=5.0, linear=False, eps=1e-6):
    K = 4.0 * np.pi * G * rho / 3.0          # g(R) = K R in the reference

    def p0(R):
        return 0.5 * rho * K * (a * a - R * R)

    def stresses(R, r, q):
        """P_rr, P_tt and dP_rr/dq at (R, r, q = r')."""
        s = r / R
        if linear:
            # first order in (q - 1, s - 1) about the reference
            err, ett = q - 1.0, s - 1.0
            S_rr = -p0(R) + (lam + 2 * mu) * err + 2 * lam * ett
            S_tt = -p0(R) + lam * err + 2 * (lam + mu) * ett
            P_rr = S_rr - p0(R) * err
            P_tt = S_tt - p0(R) * ett
            dP = lam + 2 * mu - p0(R)
            return P_rr, P_tt, dP
        err, ett = 0.5 * (q * q - 1.0), 0.5 * (s * s - 1.0)
        S_rr = -p0(R) + (lam + 2 * mu) * err + 2 * lam * ett
        S_tt = -p0(R) + lam * err + 2 * (lam + mu) * ett
        dS_rr = (lam + 2 * mu) * q
        return q * S_rr, s * S_tt, S_rr + q * dS_rr

    def rhs(R, y):
        r, q = y
        s = r / R
        P_rr, P_tt, dP = stresses(R, r, q)
        m = rho * 4.0 * np.pi * R ** 3 / 3.0
        if linear:
            grav = rho * K * R * (1.0 - 2.0 * (s - 1.0))
            # d P_rr / dR at fixed q: the explicit R and r dependence
            dp0 = -rho * K * R
            ds = (q - s) / R
            dPdR = -dp0 - dp0 * (q - 1.0) + 2 * lam * ds
        else:
            grav = rho * G * m / (r * r)
            dp0 = -rho * K * R
            ds = (q - s) / R
            dPdR = q * (-dp0 + 2 * lam * s * ds)
        dq = (grav - 2.0 * (P_rr - P_tt) / R - dPdR) / dP
        return [q, dq]

    M = rho * 4.0 * np.pi * a ** 3 / 3.0
    M_load = 4.0 * np.pi * a * a * sigma

    def shoot(c, dense=False):
        sol = solve_ivp(rhs, [eps * a, a], [c * eps * a, c], rtol=1e-11, atol=1e-13,
                        dense_output=dense)
        r, q = sol.y[0, -1], sol.y[1, -1]
        P_rr, _, _ = stresses(a, r, q)
        if linear:
            target = -sigma * G * M / (a * a)
            # the reference traction is zero; the load at first order
            return P_rr - target, sol
        return P_rr + sigma * G * (M + 0.5 * M_load) / (r * r), sol

    c = brentq(lambda x: shoot(x)[0], 0.7, 1.3, xtol=1e-14, rtol=1e-14)
    _, sol = shoot(c, dense=True)
    return sol.y[0, -1] - a, c, sol


if __name__ == "__main__":
    print(" sigma        u_r(a) nonlinear   u_r(a) linear     difference / sigma^2")
    for sigma in (0.0, 0.0125, 0.025, 0.05, 0.1):
        u, _, _ = solve(sigma)
        ul, _, _ = solve(sigma, linear=True)
        print(f"{sigma:7.4f}  {u: .12e}  {ul: .12e}  {(u - ul) / sigma ** 2 if sigma else 0.0: .6f}")
