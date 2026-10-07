"""The gravitating disc models of rung 2 (disc_gravity_reference.md).

Two-layer discs in the codebase's 2-D convention (grad^2 Phi = 4 pi G
rho, g = (4 pi G / r) int rho s ds), in benchmark-style units G = 1,
outer radius 1, densities in units of 5000 kg/m^3 — the same scaling
rules as the 3-D benchmark models, applied to the disc:

  fluid_core_2d   uniform fluid core (the 3-D fluid_core values) under
                  the uniform solid mantle; N^2 < 0 in the core.
  aw_core_2d      the closed-form neutrally stratified twin:
                  rho = rho0 (1 - alpha x^2), kappa = -rho^2 g / rho',
                  mass-matched to fluid_core_2d's core so the mantle
                  state is identical; N^2 = 0 exactly. A shifted cmb
                  re-derives kappa with (rho0, alpha) frozen, so every
                  member of the shift family is neutral.

Each model carries callables rho, kappa, mu per layer and the
hydrostatic state (g, p0) by cumulative quadrature of its own density
(p0(outer radius) = 0). `validate()` checks the AW identity, the mass
match and the hydrostatic residual.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Callable

import numpy as np

#: Benchmark-style units: G = 1, radius 1, density scale 5000 kg/m^3.
G = 1.0
RADIUS = 1.0
CMB = 3483.0 / 6371.0

#: fluid_core's SI values in the density scale.
RHO_CORE = 11000.0 / 5000.0
RHO_MANTLE = 4500.0 / 5000.0
#: Moduli: rho vp^2-style values scaled as in the 3-D benchmark's
#: units (kappa / (rho_scale G_SI-free combination)); what matters for
#: the physics ratios is kappa / (G rho^2 R^2), kept at the 3-D
#: benchmark's order one. Chosen to mirror fluid_core: kappa of the
#: core from vp 9000 against rho 11000, mantle kappa/mu from vp 11000,
#: vs 6000 against rho 4500, in the unit where 4 pi G rho_scale R^2 /3
#: ~ g-scale. The absolute scale is free in a linear problem; ratios
#: follow the 3-D model.
KAPPA_CORE = RHO_CORE * (9000.0 / 11000.0) ** 2 * 10.0
KAPPA_MANTLE = RHO_MANTLE * ((11000.0 / 11000.0) ** 2
                             - 4.0 / 3.0 * (6000.0 / 11000.0) ** 2) * 10.0
MU_MANTLE = RHO_MANTLE * (6000.0 / 11000.0) ** 2 * 10.0

#: aw_core_2d: mass-matched central density, rho0 (1 - alpha/2) =
#: RHO_CORE, alpha = 0.2.
AW_ALPHA = 0.2
AW_RHO0 = RHO_CORE / (1.0 - AW_ALPHA / 2.0)


@dataclass(frozen=True)
class DiscGravityModel:
    """Two-layer gravitating disc: fields as callables of r (vectorised),
    layer 0 = fluid core [0, c], layer 1 = solid mantle [c, a]."""
    name: str
    c: float
    a: float
    rho: tuple[Callable, Callable]
    kappa: tuple[Callable, Callable]
    mu: tuple[Callable, Callable]
    #: gravitational constant used for this model's state (G -> 0
    #: reduces the assembler to the rung-0 problem exactly: p0 and g
    #: both scale with it)
    G: float = G
    #: core density derivative d rho / dr (analytic; the Dahlen F1
    #: term and N^2 use it)
    rho_prime_core: Callable = field(default=None, compare=False)
    #: filled by _attach_state
    g: Callable = field(default=None, compare=False)
    p0: Callable = field(default=None, compare=False)
    total_mass: float = field(default=0.0, compare=False)
    core_mass: float = field(default=0.0, compare=False)


def _attach_state(m: DiscGravityModel, n: int = 120000) -> DiscGravityModel:
    """g and p0 by cumulative trapezoid of the model's own density
    (2-D: g = (4 pi G / r) int rho s ds; p0' = -rho g, p0(a) = 0).
    Per-layer grids with the interface as an exact node, so no
    quadrature interval straddles the density jump."""
    Gm = m.G
    r1 = np.linspace(0.0, m.c, n // 2)
    r2 = np.linspace(m.c, m.a, n - n // 2)
    r = np.concatenate([r1, r2[1:]])
    rho_q = np.concatenate([m.rho[0](r1), m.rho[1](r2[1:])])
    # one-sided densities per interval: both endpoints of an interval
    # lie in one layer by construction, except the pair straddling the
    # duplicated node, which uses the matching one-sided values.
    rho_lo = np.concatenate([m.rho[0](r1[:-1]), m.rho[1](r2[:-1])])
    rho_hi = np.concatenate([m.rho[0](r1[1:]), m.rho[1](r2[1:])])
    r_lo = np.concatenate([r1[:-1], r2[:-1]])
    r_hi = np.concatenate([r1[1:], r2[1:]])
    dm = 0.5 * (rho_lo * r_lo + rho_hi * r_hi) * (r_hi - r_lo)
    integ = np.concatenate(([0.0], np.cumsum(dm)))
    with np.errstate(invalid="ignore", divide="ignore"):
        g = np.where(r > 0,
                     4.0 * math.pi * Gm * integ / np.maximum(r, 1e-300),
                     0.0)
    rg_lo = rho_lo * np.interp(r_lo, r, g)
    rg_hi = rho_hi * np.interp(r_hi, r, g)
    dp = 0.5 * (rg_lo + rg_hi) * (r_hi - r_lo)
    p_cum = np.concatenate(([0.0], np.cumsum(dp)))
    p0 = p_cum[-1] - p_cum
    core_mass = 2.0 * math.pi * float(integ[len(r1) - 1])
    total_mass = 2.0 * math.pi * float(integ[-1])

    def g_of(x):
        # exterior continuation: g = 2 G M_tot / r beyond the surface
        x = np.asarray(x, dtype=float)
        inside = np.interp(x, r, g)
        outside = 2.0 * Gm * total_mass / np.maximum(x, 1e-300)
        return np.where(x <= m.a, inside, outside)

    def p0_of(x):
        return np.interp(np.asarray(x, dtype=float), r, p0)

    object.__setattr__(m, "g", g_of)
    object.__setattr__(m, "p0", p0_of)
    object.__setattr__(m, "core_mass", core_mass)
    object.__setattr__(m, "total_mass", total_mass)
    return m


def fluid_core_2d(cmb: float = CMB, G_newton: float = G) -> DiscGravityModel:
    return _attach_state(DiscGravityModel(
        "fluid_core_2d", cmb, RADIUS, G=G_newton,
        rho=(np.vectorize(lambda r: RHO_CORE),
             np.vectorize(lambda r: RHO_MANTLE)),
        rho_prime_core=lambda r: np.zeros_like(np.asarray(r, dtype=float)),
        kappa=(np.vectorize(lambda r: KAPPA_CORE),
               np.vectorize(lambda r: KAPPA_MANTLE)),
        mu=(np.vectorize(lambda r: 0.0),
            np.vectorize(lambda r: MU_MANTLE))))


def aw_core_2d(cmb: float = CMB, G_newton: float = G) -> DiscGravityModel:
    """The neutral twin: closed-form AW core under fluid_core_2d's
    mantle; kappa re-derived at the given cmb with (rho0, alpha)
    frozen (disc_gravity_reference.md)."""
    rho0, alpha = AW_RHO0, AW_ALPHA
    kappa0 = math.pi * G_newton * rho0 ** 2 * cmb ** 2 / alpha

    def rho_core(r):
        x2 = (np.asarray(r, dtype=float) / cmb) ** 2
        return rho0 * (1.0 - alpha * x2)

    def kappa_core(r):
        x2 = (np.asarray(r, dtype=float) / cmb) ** 2
        return kappa0 * (1.0 - alpha * x2) ** 2 * (1.0 - 0.5 * alpha * x2)

    def rho_prime(r):
        return -2.0 * rho0 * alpha * np.asarray(r, dtype=float) / cmb ** 2

    return _attach_state(DiscGravityModel(
        "aw_core_2d", cmb, RADIUS, G=G_newton,
        rho_prime_core=rho_prime,
        rho=(rho_core, np.vectorize(lambda r: RHO_MANTLE)),
        kappa=(kappa_core, np.vectorize(lambda r: KAPPA_MANTLE)),
        mu=(np.vectorize(lambda r: 0.0),
            np.vectorize(lambda r: MU_MANTLE))))


def n_squared(m: DiscGravityModel, r: np.ndarray) -> np.ndarray:
    """Core N^2 = -g (rho'/rho + g rho / kappa), analytic rho' where
    available (aw) else finite differences."""
    r = np.asarray(r, dtype=float)
    rho = m.rho[0](r)
    if m.name == "aw_core_2d":
        rho0, alpha = AW_RHO0, AW_ALPHA
        drho = -2.0 * rho0 * alpha * r / m.c ** 2
    elif m.name == "fluid_core_2d":
        drho = np.zeros_like(r)  # uniform core
    else:
        drho = np.gradient(rho, r)
    g = m.g(r)
    return -g * (drho / rho + g * rho / m.kappa[0](r))


def validate(verbose: bool = True) -> bool:
    fc, aw = fluid_core_2d(), aw_core_2d()
    r = np.linspace(1e-6, CMB - 1e-9, 4000)
    checks = []
    # mass match: identical mantle state between the twins
    checks.append(("core mass aw/fc - 1",
                   abs(aw.core_mass / fc.core_mass - 1.0), 1e-9))
    # neutrality of aw (analytic derivative), scaled as in 3-D
    n2 = n_squared(aw, r)
    scale = np.max(np.abs(aw.g(r) ** 2 * aw.rho[0](r) / aw.kappa[0](r)))
    checks.append(("aw max |N^2| / scale", float(np.max(np.abs(n2)) / scale),
                   1e-9))
    # fluid_core is genuinely non-neutral: N^2 = -g^2 rho / kappa < 0
    # at mid-core, order of the scale
    n2_mid = float(n_squared(fc, np.array([0.5 * CMB]))[0])
    scale_fc = float((fc.g(0.5 * CMB) ** 2 * fc.rho[0](0.5 * CMB)
                      / fc.kappa[0](0.5 * CMB)))
    checks.append(("fc N^2(c/2)/scale + 1 (expect -1)",
                   abs(n2_mid / scale_fc + 1.0), 1e-6))
    # hydrostatic residual of the attached state: p0' + rho g = 0
    for m in (fc, aw):
        rr = np.linspace(m.c + 1e-6, m.a - 1e-6, 2000)
        res = np.gradient(m.p0(rr), rr) + m.rho[1](rr) * m.g(rr)
        checks.append((f"{m.name} hydrostatic residual (mantle)",
                       float(np.max(np.abs(res))
                             / np.max(m.rho[1](rr) * m.g(rr))), 5e-4))
        checks.append((f"{m.name} p0(a)", abs(float(m.p0(m.a))), 1e-12))
    ok = True
    for label, value, tol in checks:
        good = value <= tol
        ok &= good
        if verbose:
            print(f"  {label:<44s} {value:10.3e}  {'ok' if good else 'FAIL'}")
    if verbose:
        print("disc_models validation", "PASS" if ok else "FAIL")
    return ok


if __name__ == "__main__":
    raise SystemExit(0 if validate() else 1)
