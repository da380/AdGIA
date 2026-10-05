"""The spherically symmetric models of the Love-number benchmark.

A ladder of models of increasing complexity on an Earth-sized body, each
a planetmodel `Model` in SI:

  homogeneous      one solid layer of constant density and moduli
  homogeneous_lithosphere
                   the same body cut into an interior and an outer shell
                   (the viscoelastic benchmark's elastic lithosphere)
  two_solid        two solid layers with a jump in every parameter
  fluid_core       a uniform fluid core under a uniform solid mantle
  inner_core       a solid inner core, a fluid outer core and a mantle,
                   each uniform
  linear_solid     one solid layer, density and velocities linear in radius
  stratified_core  a fluid core whose density falls with radius, under a
                   mantle, every parameter linear in radius
  earth_like       inner core, fluid outer core and mantle, every parameter
                   linear in radius within each
  prem_4           PREM without its ocean, isotropic, on four layers: inner
                   core, outer core, lower mantle, and the upper mantle
                   with the crust
  prem_6           the same on six: the transition zone and the crust are
                   layers of their own

The PREM models keep the named boundaries and merge what lies between
them (`Model.coarsened`): within a merged layer each parameter is the
cubic closest to PREM's, so that the model is smooth within its layers,
as the mesh takes it to be.

Every model is a planetmodel `LayeredIsotropicElastic`, whose layers take
a constant or the coefficients of a polynomial in r / RADIUS for each
parameter. The values are round numbers of the Earth's order, so that the
ratio rho g a / mu of gravitational to elastic forces is of order one, as
in the Earth. `model(name)` returns the model in SI and `scaled(model)`
the same model in the benchmark's units, in which the mesh is built and
the finite-element problem solved: lengths in units of the model's outer
radius, densities in units of `DENSITY_SCALE`, and times in units of
`time_scale`, by default the one that makes G equal to one.
"""
from __future__ import annotations

import math
from collections.abc import Callable, Sequence

import numpy as np
from planetmodel import (PREM, LayeredIsotropicElastic, Model, RadialField,
                         gravity, kappa_mu, polynomial_fit)
from planetmodel.units import G_SI, Scales

#: The outer radius of the constructed models, in metres.
RADIUS = 6371.0e3

#: The density scale of the benchmark's units, in kg m^-3.
DENSITY_SCALE = 5000.0

#: Radii of the internal boundaries, in metres.
ICB = 1200.0e3
CMB = 3500.0e3
MID_MANTLE = 5000.0e3
#: The base of the elastic lithosphere of the viscoelastic benchmark's
#: homogeneous_lithosphere: thick (~670 km) so that coarse meshes resolve
#: it, the point being an elastic layer that survives the relaxed limit.
LITHOSPHERE_BASE = 5700.0e3


def homogeneous() -> Model:
    return LayeredIsotropicElastic(
        [0.0, RADIUS], rho=[5500.0], vp=[10000.0], vs=[5500.0],
        layer_names=["body"], interface_names=["surface"], name="homogeneous")


def homogeneous_lithosphere(base: float = LITHOSPHERE_BASE) -> Model:
    """The homogeneous body cut in two at `base`: elastically the same
    body, but the outer shell is a layer of its own, so that the
    viscoelastic benchmark can keep it elastic (an elastic lithosphere
    over a Maxwell interior; viscoelastic/README.md)."""
    return LayeredIsotropicElastic(
        [0.0, base, RADIUS], rho=[5500.0, 5500.0], vp=[10000.0, 10000.0],
        vs=[5500.0, 5500.0], layer_names=["interior", "lithosphere"],
        interface_names=["lithosphere_base", "surface"],
        name="homogeneous_lithosphere")


def two_solid(discontinuity: float = MID_MANTLE) -> Model:
    """The two-solid model; `discontinuity` moves the interface (the
    degree-0 perturbation benchmark shifts it)."""
    return LayeredIsotropicElastic(
        [0.0, discontinuity, RADIUS], rho=[6500.0, 3500.0],
        vp=[12000.0, 8500.0], vs=[6500.0, 4500.0],
        layer_names=["lower", "upper"],
        interface_names=["discontinuity", "surface"], name="two_solid")


def fluid_core(cmb: float = CMB) -> Model:
    """The two-layer model; `cmb` moves the interface (the degree-0
    perturbation benchmark shifts it by a small amount)."""
    return LayeredIsotropicElastic(
        [0.0, cmb, RADIUS], rho=[11000.0, 4500.0], vp=[9000.0, 11000.0],
        vs=[0.0, 6000.0], layer_names=["core", "mantle"],
        interface_names=["cmb", "surface"], name="fluid_core")


def inner_core() -> Model:
    return LayeredIsotropicElastic(
        [0.0, ICB, CMB, RADIUS], rho=[13000.0, 11000.0, 4500.0],
        vp=[11000.0, 9000.0, 11000.0], vs=[3500.0, 0.0, 6000.0],
        layer_names=["inner_core", "outer_core", "mantle"],
        interface_names=["icb", "cmb", "surface"], name="inner_core")


def linear_solid() -> Model:
    return LayeredIsotropicElastic(
        [0.0, RADIUS], rho=[(8000.0, -4500.0)], vp=[(12500.0, -5000.0)],
        vs=[(7000.0, -3000.0)], scale=RADIUS, layer_names=["body"],
        interface_names=["surface"], name="linear_solid")


def stratified_core() -> Model:
    return LayeredIsotropicElastic(
        [0.0, CMB, RADIUS],
        rho=[(12500.0, -4500.0), (7000.0, -3000.0)],
        vp=[(10500.0, -4500.0), (15500.0, -6500.0)],
        vs=[0.0, (8000.0, -3000.0)], scale=RADIUS,
        layer_names=["core", "mantle"], interface_names=["cmb", "surface"],
        name="stratified_core")


def earth_like() -> Model:
    return LayeredIsotropicElastic(
        [0.0, ICB, CMB, RADIUS],
        rho=[(13100.0, -1500.0), (12800.0, -5000.0), (7000.0, -3000.0)],
        vp=[(11300.0, -1000.0), (10800.0, -5000.0), (15500.0, -6500.0)],
        vs=[(3700.0, -500.0), 0.0, (8000.0, -3000.0)], scale=RADIUS,
        layer_names=["inner_core", "outer_core", "mantle"],
        interface_names=["icb", "cmb", "surface"], name="earth_like")


def prem_like(keep: Sequence[str], layers: Sequence[str], *,
              name: str, degree: int = 3) -> Model:
    """PREM without its ocean, isotropic, with every interior interface
    but `keep` (PREM's own names) merged away: within a merged layer each
    parameter is the polynomial of `degree` closest to PREM's, refit
    through the velocities so the elastic relations hold exactly, and a
    fluid layer never merges with a solid one. `layers` names the coarse
    layers, centre outward."""
    coarse, _ = PREM(ocean=False).isotropic().coarsened(keep=list(keep),
                                                        degree=degree)
    return coarse.renamed(layers=list(layers)).replaced(name=name,
                                                        check=False)


def prem_4() -> Model:
    return prem_like(
        ["icb", "cmb", "d670"],
        ["inner_core", "outer_core", "lower_mantle", "upper_mantle"],
        name="prem_4")


def prem_6() -> Model:
    return prem_like(
        ["icb", "cmb", "d670", "d400", "moho"],
        ["inner_core", "outer_core", "lower_mantle", "transition_zone",
         "upper_mantle", "crust"], name="prem_6")


#: Models whose single interior interface the perturbation benchmark can
#: shift, with the keyword that moves it (SI metres).
SHIFTABLE: dict[str, str] = {
    "fluid_core": "cmb",
    "two_solid": "discontinuity",
}

MODELS: dict[str, Callable[[], Model]] = {
    "homogeneous": homogeneous,
    "homogeneous_lithosphere": homogeneous_lithosphere,
    "two_solid": two_solid,
    "fluid_core": fluid_core,
    "inner_core": inner_core,
    "linear_solid": linear_solid,
    "stratified_core": stratified_core,
    "earth_like": earth_like,
    "prem_4": prem_4,
    "prem_6": prem_6,
}


def model(name: str) -> Model:
    """The named model, in SI."""
    if name not in MODELS:
        raise KeyError(f"no model named {name!r}; models are {sorted(MODELS)}")
    return MODELS[name]()


def scales(radius: float, *, time_scale: float | None = None) -> Scales:
    """The benchmark's units: the model's outer `radius`, `DENSITY_SCALE`
    and a time scale in seconds, by default 1 / sqrt(G DENSITY_SCALE), for
    which G is one."""
    if time_scale is None:
        time_scale = 1.0 / math.sqrt(G_SI * DENSITY_SCALE)
    return Scales(length=radius, mass=DENSITY_SCALE * radius ** 3,
                  time=float(time_scale))


def scaled(si: Model, *, time_scale: float | None = None) -> Model:
    """`si` in the benchmark's units, in which its outer radius is one,
    with the bulk and shear moduli attached to every layer as the fields
    `kappa` and `mu`."""
    radius = float(si.skeleton.boundaries[-1])
    out = si.converted(scales(radius, time_scale=time_scale))
    for i, layer in enumerate(out.layers):
        kappa, mu = kappa_mu(layer)
        out = (out.with_field(i, "kappa", kappa, replace=True)
                  .with_field(i, "mu", mu, replace=True))
    return out


def with_pressure(model: Model, *, degree: int = 12,
                  quadrature: int = 64) -> Model:
    """The model with its hydrostatic pressure attached to every layer as
    the field `p0`: p(r) = int_r^a rho g dr, zero at the surface, in the
    model's own units. The equilibrium pressure the referential solvers
    need (bare moduli, S_e = -p0 I, and the interface pressure of the
    slipping methods).

    Per layer the integrand rho g is smooth (the enclosed-mass 1/r^2 of g
    stays off the centre wherever it has a coefficient), so a Gauss-
    Legendre rule of `quadrature` points per segment and a polynomial fit
    of `degree` per layer hold it to close to machine accuracy for the
    polynomial models here."""
    b = np.asarray(model.skeleton.boundaries, dtype=float)
    x_gl, w_gl = np.polynomial.legendre.leggauss(quadrature)

    def integrand(i: int, r: np.ndarray) -> np.ndarray:
        return np.asarray(model.layers[i].fields["rho"](r),
                          dtype=float) * gravity(model, r)

    def segment(i: int, lo: np.ndarray | float, hi: float) -> np.ndarray:
        lo = np.atleast_1d(np.asarray(lo, dtype=float))
        mid, half = 0.5 * (lo + hi), 0.5 * (hi - lo)
        # nodes[q, j] over the segments [lo_j, hi]
        nodes = mid[None, :] + half[None, :] * x_gl[:, None]
        values = integrand(i, nodes.ravel()).reshape(nodes.shape)
        return half * np.einsum("q,qj->j", w_gl, values)

    # The pressure at the top of each layer, surface down.
    n = len(model.layers)
    p_top = np.zeros(n)
    for i in range(n - 2, -1, -1):
        p_top[i] = p_top[i + 1] + float(segment(i + 1, b[i + 1],
                                                b[i + 2])[0])

    out = model
    for i in range(n):
        lo, hi = float(b[i]), float(b[i + 1])
        top = p_top[i]

        def p(r: np.ndarray, i=i, hi=hi, top=top) -> np.ndarray:
            return top + segment(i, r, hi)

        fit = polynomial_fit((lo, hi), p, degree=degree)
        out = out.with_field(i, "p0", RadialField((lo, hi), fit, name="p0"),
                             replace=True)
    return out
