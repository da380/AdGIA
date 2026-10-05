"""Exact references for the box benchmarks (viscoelastic_box.cpp).

Every case of this sub-family is linear, non-gravitating and laterally
simple, so its response is a finite sum of relaxation modes,

    y(t) = R_inf S(t) + sum_i r_i int_0^t e^{p_i (t-u)} S(u) du,

with real poles p_i <= 0, and the convolutions are evaluated in closed form
for the piecewise load histories (histories.py). What differs between the
cases is how the modes are found.

  box, uniaxial_stress / uniaxial_strain (a homogeneous state)
      The mean strain and internal variables of one generalised Maxwell
      body. Under stress control the deviatoric amplitude delta and the
      branch variables q_k satisfy

          delta = (s + sum_k a_k q_k) / M,   dq_k/dt = (delta - q_k)/tau_k,

      a = 2 mu_k, M = 2 mu_U; the matrix of this linear system is similar
      to a symmetric one, whose eigen-decomposition (mpmath, 40 digits)
      gives the modes. Under strain control each q_k relaxes on its own.

  slab, mode 0 (the laterally uniform column)
      Uniaxial strain with sigma_zz = -p S(t) at every depth: the same
      0-D system per point with M = kappa + 2 c mu_U, a_k = 2 c mu_k,
      c = 1 - 1/d; the surface displacement W = int_0^H eps_zz dz,
      by layer (constant moduli) or Gauss-Legendre quadrature in z (moduli
      varying in z, 48 points per layer).

  slab, mode n > 0 (Fourier-mode loading, the "Cartesian Love numbers")
      By the correspondence principle the transform of the response is
      the elastic response with mu -> mu(s) = mu_inf + sum_k mu_k s tau_k/
      (1 + s tau_k), lambda = kappa - 2 mu / d (d the space dimension: the
      library's 2-D continuum in 2-D, plane strain of the 3-D body in 3-D,
      where horizontal isotropy makes cos(kx x) cos(ky y) behave as one
      wave of |k|). For u_x = U(z) sin kx, u_z = W(z) cos kx and the
      tractions sigma_xz = T sin kx, sigma_zz = N cos kx,

          U' = T/mu + k W,
          W' = (N - lambda k U)/(lambda + 2 mu),
          T' = 4 mu (lambda + mu)/(lambda + 2 mu) k^2 U
               + lambda k N/(lambda + 2 mu),
          N' = -k T,

      propagated through the layers by matrix exponentials (U = W = 0 at
      the base; T = 0, N = -1 at the top). The transfer functions
      W(H; s), U(H; s) are RATIONAL in s (the eigenvalues of the system
      are +-k whatever the moduli, so the propagators are polynomials in
      the moduli times functions of kH), and the AAA algorithm recovers
      their poles and residues from samples off the negative real axis.
      Two independent checks are stored with the reference: the fit error
      at fresh points, and the Heaviside response by fixed-Talbot
      inversion (complex s, ~1e-10) against the modal sum. A third
      inversion, Gaver-Stehfest at real s (the method of the gravitating
      Love-number references), is stored for the inversion study. When
      the modal form fails its checks (relaxation rates spanning many
      decades, viscosity contrasts of 1e4 and more), a Heaviside history
      is taken from the Talbot inversion instead, and other histories are
      flagged unreliable.

    ./reference case.json --out reference.json
    ./reference case.json --stehfest 12 16 --out reference.json
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import mpmath as mp
import numpy as np
from scipy.linalg import expm

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent.parent / "common"))
sys.path.insert(0, str(HERE))

import histories as hist  # noqa: E402
from outputs import outside_source  # noqa: E402

mp.mp.dps = 40


# --- the case ----------------------------------------------------------------


class Field:
    """A layer field: constant, or geometric in z between the layer's
    bottom and top (as the driver)."""

    def __init__(self, spec, z0: float, z1: float):
        self.z0, self.z1 = z0, z1
        if isinstance(spec, (int, float)):
            self.kind, self.v0, self.v1 = "constant", float(spec), float(spec)
        else:
            if spec.get("kind") != "geometric_z":
                raise SystemExit(f"unknown field kind {spec.get('kind')!r}")
            self.kind = "geometric_z"
            self.v0, self.v1 = float(spec["bottom"]), float(spec["top"])

    @property
    def constant(self) -> bool:
        return self.kind == "constant"

    def __call__(self, z):
        if self.constant:
            return self.v0 + 0.0 * np.asarray(z, dtype=float)
        s = np.clip((np.asarray(z) - self.z0) / (self.z1 - self.z0), 0, 1)
        return self.v0 * (self.v1 / self.v0) ** s


class Layer:
    def __init__(self, d: dict, bottom: float):
        self.name = d["name"]
        self.bottom, self.top = bottom, float(d["top"])
        self.kappa = Field(d["kappa"], bottom, self.top)
        self.mu_inf = Field(d["mu_inf"], bottom, self.top)
        self.branches = [(Field(b["mu"], bottom, self.top),
                          Field(b["tau"], bottom, self.top))
                         for b in d["branches"]]

    @property
    def constant(self) -> bool:
        return all(f.constant for f in self.fields())

    def fields(self):
        yield self.kappa
        yield self.mu_inf
        for mu, tau in self.branches:
            yield mu
            yield tau

    def at(self, z: float):
        """(kappa, mu_inf, [(mu_k, tau_k)]) at height z."""
        return (float(self.kappa(z)), float(self.mu_inf(z)),
                [(float(m(z)), float(t(z))) for m, t in self.branches])


class Case:
    def __init__(self, path: Path):
        self.path = Path(path)
        d = json.loads(self.path.read_text())
        self.raw = d
        self.name = d["name"]
        self.dim = int(d["dim"])
        g = d["geometry"]
        self.kind = g["kind"]
        self.lengths = [float(x) for x in g["length"]]
        self.H = float(g["height"])
        self.layers, bottom = [], 0.0
        for ld in d["layers"]:
            self.layers.append(Layer(ld, bottom))
            bottom = self.layers[-1].top
        load = d["load"]
        self.load = load["kind"]
        self.amplitude = float(load["amplitude"])
        self.history = hist.History.from_json(load["history"])
        self.mode = [float(n) for n in load.get("mode", [0] * (self.dim - 1))]
        self.k = [2 * math.pi * n / L for n, L in zip(self.mode, self.lengths)]
        self.kmod = math.sqrt(sum(k * k for k in self.k))
        self.times = [float(t) for t in d["times"]]


# --- the 0-D engine ------------------------------------------------------------


def zero_d_modes(M: float, a: list[float], tau: list[float]):
    """Modes of  x = (s + a.q)/M,  dq_k/dt = (x - q_k)/tau_k  driven by s.

    Returns (lam, G, H): eigenvalues lam_i and the matrices with
        q(t)  = G @ [I(lam_i, t) * s-amplitude]_i,
        x(t)  = (s(t) + H @ [I(lam_i, t)]_i) / M,
    every branch with a_k > 0. The system matrix A = T^{-1}(1 a^T/M - I) is
    D^{-1} B D with B symmetric, D = diag(sqrt(a_k tau_k)):
        B_kj = sqrt(a_k/tau_k) sqrt(a_j/tau_j)/M - delta_kj/tau_k.
    """
    n = len(a)
    if n == 0:
        return np.zeros(0), np.zeros((0, 0)), np.zeros(0)
    am = [mp.mpf(x) for x in a]
    tm = [mp.mpf(x) for x in tau]
    Mm = mp.mpf(M)
    B = mp.matrix(n, n)
    for k in range(n):
        for j in range(n):
            B[k, j] = mp.sqrt(am[k] / tm[k]) * mp.sqrt(am[j] / tm[j]) / Mm
        B[k, k] -= 1 / tm[k]
    lam, V = mp.eigsy(B)
    # q(t) = D^{-1} V diag(I) V^T D b,  b_k = 1/(M tau_k)
    D = [mp.sqrt(am[k] * tm[k]) for k in range(n)]
    Db = [D[k] / (Mm * tm[k]) for k in range(n)]
    coef = [sum(V[j, i] * Db[j] for j in range(n)) for i in range(n)]
    G = np.array([[float(V[k, i] * coef[i] / D[k]) for i in range(n)]
                  for k in range(n)])
    Hrow = np.array([float(sum(am[k] * V[k, i] * coef[i] / D[k]
                               for k in range(n))) for i in range(n)])
    return np.array([float(x) for x in lam]), G, Hrow


def zero_d_response(M, a, tau, history, scale, times):
    """x(t) and q_k(t) of the 0-D system driven by s = scale * S(t)."""
    t = np.asarray(times, dtype=float)
    S = scale * history(t)
    lam, G, Hrow = zero_d_modes(M, a, tau)
    if len(lam) == 0:
        return S / M, np.zeros((0, len(t)))
    I = np.array([scale * history.convolve(l, t) for l in lam])
    q = G @ I
    x = (S + Hrow @ I) / M
    return x, q


# --- box cases -----------------------------------------------------------------


def box_reference(case: Case, times: list[float]) -> dict:
    if len(case.layers) != 1 or not case.layers[0].constant:
        raise SystemExit("box cases are homogeneous: one constant layer")
    d = case.dim
    kappa, mu_inf, br = case.layers[0].at(0.0)
    mu = [m for m, _ in br]
    tau = [t for _, t in br]
    t = np.asarray(times, dtype=float)
    Px = -np.eye(d) / d
    Px[0, 0] += 1.0
    S = case.amplitude * case.history(t)
    if case.load == "uniaxial_stress":
        mu_u = mu_inf + sum(mu)
        delta, q = zero_d_response(2 * mu_u, [2 * m for m in mu], tau,
                                   case.history, case.amplitude, t)
        tr = S / (d * kappa)
        E = [tr[j] / d * np.eye(d) + delta[j] * Px for j in range(len(t))]
    elif case.load == "uniaxial_strain":
        q = np.array([case.amplitude / tk * case.history.convolve(-1 / tk, t)
                      for tk in tau])
        ex = np.zeros((d, d))
        ex[0, 0] = 1.0
        E = [S[j] * ex for j in range(len(t))]
    else:
        raise SystemExit(f"box load {case.load!r}?")
    out = []
    for j in range(len(t)):
        out.append({"strain": E[j].ravel().tolist(),
                    "internal": [(q[k, j] * Px).ravel().tolist()
                                 for k in range(len(tau))]})
    return {"series": out}


# --- the column ----------------------------------------------------------------


def column_reference(case: Case, times: list[float]) -> dict:
    """Laterally uniform slab: W(t) = int_0^H eps_zz(z, t) dz."""
    d = case.dim
    c = 1.0 - 1.0 / d
    t = np.asarray(times, dtype=float)
    W = np.zeros(len(t))
    for layer in case.layers:
        if layer.constant:
            nodes, weights = [0.5 * (layer.bottom + layer.top)], \
                [layer.top - layer.bottom]
        else:
            x, w = np.polynomial.legendre.leggauss(48)
            h = layer.top - layer.bottom
            nodes = layer.bottom + 0.5 * h * (x + 1)
            weights = 0.5 * h * w
        for z, w in zip(nodes, weights):
            kappa, mu_inf, br = layer.at(z)
            mu_u = mu_inf + sum(m for m, _ in br)
            live = [(m, tk) for m, tk in br if m > 0.0]
            x, _ = zero_d_response(kappa + 2 * c * mu_u,
                                   [2 * c * m for m, _ in live],
                                   [tk for _, tk in live],
                                   case.history, -case.amplitude, t)
            W += w * x
    return {"series": [{"W": float(W[j]), "U": 0.0}
                       for j in range(len(t))]}


# --- the Fourier slab ----------------------------------------------------------


def complex_moduli(layer: Layer, s: complex, d: int):
    kappa, mu_inf, br = layer.at(layer.bottom)
    mu = mu_inf + sum(m * s * tk / (1 + s * tk) for m, tk in br)
    return kappa - 2 * mu / d, mu


def slab_transfer(case: Case, s: complex) -> tuple[complex, complex]:
    """(W(H), U(H)) per unit pressure amplitude at Laplace variable s."""
    k, d = case.kmod, case.dim
    P = np.eye(4, dtype=complex)
    for layer in case.layers:
        lam, mu = complex_moduli(layer, s, d)
        m = lam + 2 * mu
        A = np.array([[0, k, 1 / mu, 0],
                      [-lam * k / m, 0, 0, 1 / m],
                      [4 * mu * (lam + mu) / m * k * k, 0, 0, lam * k / m],
                      [0, 0, -k, 0]], dtype=complex)
        P = expm(A * (layer.top - layer.bottom)) @ P
    # y(0) = (0, 0, T0, N0); T(H) = 0, N(H) = -1.
    Q = P[2:4, 2:4]
    T0N0 = np.linalg.solve(Q, np.array([0.0, -1.0], dtype=complex))
    yH = P[:, 2:4] @ T0N0
    return yH[1], yH[0]


def aaa(Z: np.ndarray, F: np.ndarray, tol: float = 1e-13, mmax: int = 80):
    """The AAA rational approximation (Nakatsukasa, Sete & Trefethen 2018):
    support points, values and weights of the barycentric form."""
    M = len(Z)
    J = np.ones(M, dtype=bool)
    z, f, C = [], [], np.zeros((M, 0), dtype=complex)
    R = np.full(M, F.mean())
    scale = np.abs(F).max()
    w = np.zeros(0)
    for _ in range(mmax):
        j = np.argmax(np.where(J, np.abs(F - R), -1.0))
        z.append(Z[j])
        f.append(F[j])
        J[j] = False
        with np.errstate(divide="ignore", invalid="ignore"):
            col = 1.0 / (Z - Z[j])
        col[j] = 0.0
        C = np.column_stack([C, col])
        fa = np.array(f)
        A = F[:, None] * C - C * fa[None, :]
        _, _, Vh = np.linalg.svd(A[J], full_matrices=False)
        w = Vh[-1].conj()
        N, D = C @ (w * fa), C @ w
        R = F.copy()
        R[J] = N[J] / D[J]
        if np.abs(F - R).max() <= tol * scale:
            break
    return np.array(z), np.array(f), w


def aaa_eval(z, f, w, s):
    s = np.atleast_1d(s)
    out = np.empty(len(s), dtype=complex)
    for i, si in enumerate(s):
        diff = si - z
        hit = np.abs(diff) == 0
        if hit.any():
            out[i] = f[hit][0]
            continue
        out[i] = np.sum(w * f / diff) / np.sum(w / diff)
    return out


def aaa_poles_residues(z, f, w):
    m = len(z)
    E = np.zeros((m + 1, m + 1), dtype=complex)
    E[0, 1:] = w
    E[1:, 0] = 1.0
    E[1:, 1:] = np.diag(z)
    B = np.eye(m + 1, dtype=complex)
    B[0, 0] = 0.0
    from scipy.linalg import eig
    ev = eig(E, B, right=False)
    poles = ev[np.isfinite(ev)]
    res = []
    # A pole on a support point is spurious (the function is finite
    # there); its residue comes out nan and is dropped with the others.
    with np.errstate(divide="ignore", invalid="ignore"):
        for p in poles:
            diff = p - z
            N = np.sum(w * f / diff)
            dD = -np.sum(w / diff ** 2)
            res.append(N / dD)
    r_inf = np.sum(w * f) / np.sum(w)
    return poles, np.array(res), r_inf


def slab_modes(case: Case, which: int) -> dict:
    """Poles/residues of the transfer function W (which=0) or U (1)."""
    rates = [1 / float(tk(layer.bottom)) for layer in case.layers
             for m, tk in layer.branches if float(m(layer.bottom)) > 0]
    smin, smax = min(rates), max(rates)
    radii = np.geomspace(1e-3 * smin, 1e3 * smax, 90)
    angles = np.array([0.0, 0.25, -0.25, 0.5, -0.5, 0.7, -0.7]) * math.pi
    Z = np.concatenate([r * np.exp(1j * angles) for r in radii])
    F = np.array([slab_transfer(case, s)[which] for s in Z])
    z, f, w = aaa(Z, F)
    poles, res, r_inf = aaa_poles_residues(z, f, w)
    # Fit error at fresh points.
    test = np.geomspace(2e-3 * smin, 5e2 * smax, 37) * np.exp(0.33j * math.pi)
    Ft = np.array([slab_transfer(case, s)[which] for s in test])
    fit = float(np.abs(aaa_eval(z, f, w, test) - Ft).max()
                / np.abs(Ft).max())
    # Physical poles are real and <= 0; drop negligible (Froissart) ones.
    finite = np.isfinite(res)
    poles, res = poles[finite], res[finite]
    big = np.abs(res) > 1e-12 * max(1.0, np.abs(res).max())
    poles, res = poles[big], res[big]
    worst_imag = float(np.max(np.abs(poles.imag) / np.maximum(
        np.abs(poles), smin), initial=0.0))
    return {"poles": poles, "residues": res, "r_inf": complex(r_inf),
            "fit_error": fit, "pole_imag": worst_imag,
            "pole_max_real": float(np.max(poles.real, initial=-np.inf))}


def modal_response(modes: dict, history: hist.History, scale: float,
                   times) -> np.ndarray:
    t = np.asarray(times, dtype=float)
    y = modes["r_inf"] * scale * history(t)
    for p, r in zip(modes["poles"], modes["residues"]):
        pr = p.real if abs(p.imag) < 1e-10 * max(1.0, abs(p)) else p
        y = y + r * scale * history.convolve(pr, t)
    return np.real(y)


def talbot(F, t: float, M: int = 32) -> float:
    """Fixed-Talbot inversion (Abate & Valko 2004) of F at time t."""
    r = 2.0 * M / (5.0 * t)
    total = 0.5 * F(r) * math.exp(r * t)
    for k in range(1, M):
        th = k * math.pi / M
        cot = 1.0 / math.tan(th)
        s = r * th * (cot + 1j)
        sig = th + (th * cot - 1.0) * cot
        total += (np.exp(t * s) * F(s) * (1 + 1j * sig)).real
    return float((r / M) * total.real)


def stehfest_weights(n: int) -> list[float]:
    half = n // 2
    V = []
    for k in range(1, n + 1):
        acc = 0.0
        for j in range((k + 1) // 2, min(k, half) + 1):
            acc += (j ** half * math.factorial(2 * j)
                    / (math.factorial(half - j) * math.factorial(j)
                       * math.factorial(j - 1) * math.factorial(k - j)
                       * math.factorial(2 * j - k)))
        V.append((-1) ** (k + half) * acc)
    return V


def stehfest(F, t: float, n: int = 12) -> float:
    V = stehfest_weights(n)
    a = math.log(2.0) / t
    return a * sum(V[k - 1] * F(k * a).real for k in range(1, n + 1))


def slab_reference(case: Case, times: list[float],
                   stehfest_orders=()) -> dict:
    if not all(layer.constant for layer in case.layers):
        raise SystemExit("Fourier slabs need constant layers "
                         "(z-varying moduli: use the column, mode 0)")
    mW, mU = slab_modes(case, 0), slab_modes(case, 1)
    t = np.asarray(times, dtype=float)
    W = modal_response(mW, case.history, case.amplitude, t)
    U = modal_response(mU, case.history, case.amplitude, t)
    # The Heaviside response three ways: modal, Talbot, Stehfest.
    step = hist.heaviside()
    t = t[t > 0.0]  # the inversions need t > 0
    stepW = modal_response(mW, step, 1.0, t)
    stepU = modal_response(mU, step, 1.0, t)
    tal_W = np.array([talbot(lambda s: slab_transfer(case, s)[0] / s, tj)
                      for tj in t])
    tal_U = np.array([talbot(lambda s: slab_transfer(case, s)[1] / s, tj)
                      for tj in t])
    scaleW = np.abs(stepW).max()
    scaleU = max(np.abs(stepU).max(), 1e-300)
    checks = {
        "fit_error_W": mW["fit_error"], "fit_error_U": mU["fit_error"],
        "pole_imag_W": mW["pole_imag"], "pole_imag_U": mU["pole_imag"],
        "pole_max_real_W": mW["pole_max_real"],
        "talbot_vs_modal_W": float(np.abs(tal_W - stepW).max() / scaleW),
        "talbot_vs_modal_U": float(np.abs(tal_U - stepU).max() / scaleU),
    }
    inversion = {"times": t.tolist(), "modal_W": stepW.tolist(),
                 "modal_U": stepU.tolist(), "talbot_W": tal_W.tolist(),
                 "talbot_U": tal_U.tolist()}
    for n in stehfest_orders:
        inversion[f"stehfest{n}_W"] = [
            stehfest(lambda s: slab_transfer(case, s)[0] / s, tj, n)
            for tj in t]
        inversion[f"stehfest{n}_U"] = [
            stehfest(lambda s: slab_transfer(case, s)[1] / s, tj, n)
            for tj in t]
    # The modal form is trusted only when its checks pass. Over relaxation
    # rates spanning many decades (viscosity contrasts of 1e4 and more)
    # AAA cannot resolve the transfer function from its samples (fit
    # errors, spurious complex or positive poles); a Heaviside history is
    # then taken from the fixed-Talbot inversion instead, which evaluates
    # the transfer function exactly and never needs the extreme rates
    # resolved. Other histories are flagged unreliable.
    rate_scale = max(1 / float(tk(l.bottom)) for l in case.layers
                     for m, tk in l.branches if float(m(l.bottom)) > 0)
    modal_ok = (max(mW["fit_error"], mU["fit_error"]) < 1e-9
                and mW["pole_max_real"] <= 1e-9 * rate_scale
                and mU["pole_max_real"] <= 1e-9 * rate_scale
                and checks["talbot_vs_modal_W"] < 1e-7
                and checks["talbot_vs_modal_U"] < 1e-7)
    method, reliable = "modal", True
    if not modal_ok:
        pieces = case.history.pieces
        heaviside = (len(pieces) == 1 and pieces[0].start == 0.0
                     and math.isinf(pieces[0].end) and pieces[0].c1 == 0.0
                     and pieces[0].a == 0.0 and pieces[0].b == 0.0)
        if heaviside:
            method = "talbot"
            c0 = case.amplitude * pieces[0].c0
            big = 1e8 * rate_scale  # s -> infinity: the elastic response
            W0, U0 = (float(np.real(x)) for x in slab_transfer(case, big))
            W = np.concatenate([[c0 * W0], c0 * tal_W])
            U = np.concatenate([[c0 * U0], c0 * tal_U])
        else:
            reliable = False
            print(f"warning: {case.name}: the modal reference failed its "
                  "checks and the history is not a Heaviside; the "
                  "reference is unreliable", flush=True)
    modes = [{"pole": float(p.real), "residue_W": float(r.real)}
             for p, r in sorted(zip(mW["poles"], mW["residues"]),
                                key=lambda pr: -pr[0].real)]
    relaxed = None
    if mW["pole_max_real"] < -1e-12 * max(1.0, abs(mW["pole_max_real"])):
        relaxed = [float(np.real(slab_transfer(case, 1e-12)[i]))
                   for i in (0, 1)]
    return {"series": [{"W": float(W[j]), "U": float(U[j])}
                       for j in range(len(W))],
            "elastic_transfer": [float(np.real(mW["r_inf"])),
                                 float(np.real(mU["r_inf"]))],
            "relaxed_transfer": relaxed,
            "method": method, "reliable": reliable,
            "modes": modes, "checks": checks, "inversion": inversion}


# --- driver ----------------------------------------------------------------------


def reference(case: Case, stehfest_orders=()) -> dict:
    times = case.times
    # t = 0+: the elastic response to S(0).
    if case.kind == "box":
        body = box_reference(case, [0.0] + times)
    elif case.kmod == 0.0:
        body = column_reference(case, [0.0] + times)
    else:
        body = slab_reference(case, [0.0] + times, stehfest_orders)
    series = body.pop("series")
    out = {"case": case.name, "case_file": str(case.path),
           "dim": case.dim, "kind": case.kind, "load": case.load,
           "times": times, "elastic": series[0],
           "histories": [dict(time=t, **s)
                         for t, s in zip(times, series[1:])]}
    out.update(body)
    return out


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("case", type=Path)
    p.add_argument("--out", type=Path, default=Path("reference.json"))
    p.add_argument("--stehfest", type=int, nargs="*", default=[],
                   help="Stehfest orders for the inversion study (slab)")
    args = p.parse_args()
    case = Case(args.case)
    ref = reference(case, args.stehfest)
    outside_source(args.out).write_text(json.dumps(ref, indent=1))
    if "checks" in ref:
        for k, v in ref["checks"].items():
            print(f"  {k:22s} {v:.3e}")
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
