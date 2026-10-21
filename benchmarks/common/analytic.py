"""Exact non-gravitating elastic solutions of piecewise-homogeneous spheres.

The reference of the E1 (non-gravitating elastic) tier.  In a homogeneous
isotropic layer the static Navier equation has, per spherical-harmonic
degree l >= 2, four spheroidal solutions built from the harmonic
potentials chi = r^l Y and chi = r^-(l+1) Y: the gradient grad(chi) and
the Kelvin combination c1 r^2 grad(chi) + c2 x chi, with

    c1 = (3 + n) (lambda + mu) + 2 mu,
    c2 = -(2 n (lambda + mu) + (4 n + 2) mu),

n the potential's degree (l or -(l + 1)), which satisfies Navier
identically -- no division, and when c1 vanishes (possible only on the
singular side) x chi alone is the solution.  Displacements and tractions
on spheres are power laws, so a stack of homogeneous layers welded at its
interfaces is solved exactly by a small linear system: regularity keeps
the two regular solutions in the innermost layer, four unknowns per layer
above, with (U, V, T_U, T_V) continuous at the interfaces and the
tractions prescribed at the surface.  At degree 0 the two solutions are
U = r and U = r^-2 and the system halves.  Degree 1 is excluded: a
surface traction there has a net force against the rigid translations.

The forcing convention is the codes': a unit normal traction pressing on
the surface (T_U(a) = -1, T_V(a) = 0) for the channels (h_u, l_u), and a
unit tangential traction (T_U(a) = 0, T_V(a) = -1) for (h_v, l_v), each
per unit coefficient of the fully normalised Y_lm, so the numbers are
metres per pascal and compare directly with pyslfp's `traction_numbers`
and sl3dpp's `ComputeTractionNumbers` at `gravitating=False`, and with
AdGIA's traction problem.  Self-adjointness gives h_v = l(l+1) l_u
exactly.

`traction_numbers(model, lmax)` takes a ladder model (`models.py`) whose
layers are solid with constant density and velocities, and returns the
numbers in SI.  The layer systems are solved in layer-scaled radii, so
high degrees stay well conditioned.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class HomogeneousLayer:
    """One solid homogeneous layer, SI: [bottom, top] and Lame moduli."""

    bottom: float
    top: float
    lam: float
    mu: float


def homogeneous_layers(model) -> list[HomogeneousLayer]:
    """The model's layers as homogeneous Lame moduli, refused unless every
    layer is solid with constant rho, vp and vs (sampling each on a grid
    and requiring it constant to rounding)."""
    out = []
    for layer in model.layers:
        lo, hi = layer.interval
        r = np.linspace(lo, hi, 9)
        values = {}
        for name in ("rho", "vp", "vs"):
            key = next((k for k in (name, name + "v") if k in layer), None)
            if key is None:
                raise ValueError(f"layer {layer.index} lacks {name}")
            v = np.asarray(layer[key](r), dtype=float)
            if np.ptp(v) > 1e-12 * max(abs(v[0]), 1.0):
                raise ValueError(
                    f"layer {layer.index} is not homogeneous in {name}: the "
                    "analytic reference takes piecewise-constant models")
            values[name] = float(v[0])
        if values["vs"] <= 0.0:
            raise ValueError(
                f"layer {layer.index} is fluid: the non-gravitating problem "
                "takes a solid model")
        mu = values["rho"] * values["vs"] ** 2
        lam = values["rho"] * values["vp"] ** 2 - 2.0 * mu
        out.append(HomogeneousLayer(float(lo), float(hi), lam, mu))
    return out


def _kelvin(n: float, lam: float, mu: float) -> tuple[float, float]:
    """(c1, c2) of the Kelvin solution c1 r^2 grad(chi) + c2 x chi for a
    harmonic chi of degree n; satisfies Navier identically."""
    return (3.0 + n) * (lam + mu) + 2.0 * mu, -(
        2.0 * n * (lam + mu) + (4.0 * n + 2.0) * mu
    )


def _basis(l: int, lam: float, mu: float, x: float,
           singular: bool) -> np.ndarray:
    """The state (U, V, T_U, T_V) at radius x of the two regular (or two
    singular) solutions of degree l >= 2, columns per solution.  Tractions
    carry the moduli's units; x is a layer-scaled radius of order one."""
    n = -(l + 1.0) if singular else float(l)
    k2 = l * (l + 1.0)
    out = np.empty((4, 2))
    # grad(r^n Y): U = n x^(n-1), V = x^(n-1), divergence-free.
    out[:, 0] = [n * x ** (n - 1.0), x ** (n - 1.0),
                 2.0 * mu * n * (n - 1.0) * x ** (n - 2.0),
                 2.0 * mu * (n - 1.0) * x ** (n - 2.0)]
    # c1 r^2 grad(chi) + c2 x chi: U = (c1 n + c2) x^(n+1), V = c1 x^(n+1),
    # div = (c1 2n + c2 (3 + n)) x^n Y.
    c1, c2 = _kelvin(n, lam, mu)
    div = c1 * 2.0 * n + c2 * (3.0 + n)
    U, V = c1 * n + c2, c1
    out[:, 1] = [U * x ** (n + 1.0), V * x ** (n + 1.0),
                 (lam * div + 2.0 * mu * U * (n + 1.0)) * x ** n,
                 mu * (V * (n + 1.0) - V + U) * x ** n]
    return out


def _basis0(lam: float, mu: float, x: float, singular: bool) -> np.ndarray:
    """The state (U, T_U) at radius x of the degree-0 solution U = x
    (regular) or U = x^-2 (singular), one column."""
    if singular:
        return np.array([[x ** -2.0], [-4.0 * mu * x ** -3.0]])
    return np.array([[x], [3.0 * lam + 2.0 * mu]])


def _solve_degree(layers: list[HomogeneousLayer], l: int) -> np.ndarray:
    """The surface displacement rows for the unit pressing tractions, as
    the array [[h_u, h_v], [l_u, l_v]] in SI (degree 0: [[h_u]]).

    The basis of each layer is evaluated in radii scaled by the layer's
    top, with the traction rows divided by that scale so they are
    physical; the innermost layer, which must reach the centre, keeps the
    regular solutions only.
    """
    if layers[0].bottom != 0.0:
        raise ValueError("the model must reach the centre")
    nlay = len(layers)
    half = 1 if l == 0 else 2  # (U) or (U, V); tractions likewise
    cols = [half] + [2 * half] * (nlay - 1)
    offsets = np.concatenate([[0], np.cumsum(cols)])
    ndof = int(offsets[-1])

    def states(j: int, r: float) -> np.ndarray:
        """(displacements; physical tractions) columns of layer j at r."""
        scale = layers[j].top
        x = r / scale
        lam, mu = layers[j].lam, layers[j].mu
        basis = _basis0 if l == 0 else lambda *a: _basis(l, *a)
        block = basis(lam, mu, x, False)
        if cols[j] == 2 * half:
            block = np.hstack([block, basis(lam, mu, x, True)])
        block = block.copy()
        block[half:] /= scale
        return block

    A = np.zeros((ndof, ndof))
    b = np.zeros((ndof, half))
    row = 0
    for j in range(nlay - 1):
        rb = layers[j].top
        A[row:row + 2 * half, offsets[j]:offsets[j + 1]] = states(j, rb)
        A[row:row + 2 * half, offsets[j + 1]:offsets[j + 2]] = -states(
            j + 1, rb)
        row += 2 * half
    surface = states(nlay - 1, layers[-1].top)
    A[row:row + half, offsets[-2]:offsets[-1]] = surface[half:]
    for channel in range(half):
        b[row + channel, channel] = -1.0  # T_U(a) = -1, then T_V(a) = -1
    x = np.linalg.solve(A, b)
    return surface[:half] @ x[offsets[-2]:offsets[-1]]


def traction_numbers(model, lmax: int) -> dict[str, np.ndarray]:
    """The exact non-gravitating traction numbers of a piecewise
    homogeneous solid ladder model, SI: arrays h_u, l_u, h_v, l_v over
    degrees 0..lmax, NaN at the excluded degree 1, plus the radius."""
    layers = homogeneous_layers(model)
    out = {name: np.full(lmax + 1, np.nan) for name in
           ("h_u", "l_u", "h_v", "l_v")}
    for l in range(lmax + 1):
        if l == 1:
            continue
        s = _solve_degree(layers, l)
        out["h_u"][l] = s[0, 0]
        if l > 0:
            out["l_u"][l] = s[1, 0]
            out["h_v"][l] = s[0, 1]
            out["l_v"][l] = s[1, 1]
        else:
            out["l_u"][l] = out["h_v"][l] = out["l_v"][l] = 0.0
    out["radius"] = np.array([layers[-1].top])
    return out
