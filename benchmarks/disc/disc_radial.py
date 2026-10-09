"""The per-degree radial reference of the gravitating disc (rung 2).

The theta-reduction of the welded gauged referential weak form at
phi_e = id with the hydrostatic natural state
(doc/gravitating_elasticity.md, "the linearised system in (u, zeta1)";
doc/planning/disc_gravity_reference.md), solved by a 1-D hp-FEM:

  u-row:    material (bare C) + geometric (S_e = -p0 I)
            + (1/8 pi G) <a''(u,v) g0, g0>
            + (1/4 pi G) <a'(v) g0, grad zeta1>
  zeta-row: (1/4 pi G) <grad zeta1, grad chi>  (+ DtN)
            + (1/4 pi G) <a'(u) g0, grad chi>

with, at the identity, a'(u) = (div u) 1 - Du - Du^T and the a''
contraction of the reference doc. g0 = g(r) rhat lives everywhere the
background field does — buffer included — through the vacuum-extended
displacement, here the code's taper rule (linear taper of the surface
trace; the extension is gauge). The referential organisation conserves
mass exactly, so the l = 0 exterior monopole vanishes and the 2-D
logarithmic exterior never arises; for l >= 1 the DtN at the outer
radius R is the harmonic-decay energy l Z(R) W(R).

Fields per degree (u_r = U cos l theta, u_theta = V sin l theta,
zeta1 = Z cos l theta): U, V continuous on the body (welded), Z on
[0, R]. Every theta-integral pairs like channels, so the a''/a'
contractions are plain 2x2 amplitude-matrix algebra per dof pair with
one common angular factor that divides out.

Degrees: l = 0 (V absent) and l >= 2 in this version; l = 1 (the
translation null pair) is pending. Methods: welded, slip and dahlen
(see solve_degree), plus the Maxwell relaxation route
(solve_degree_maxwell): the core as an artificial Maxwell solid,
time-stepped under the Heaviside load until the solid stops moving —
the secular limit computed with welded solid-elastic solves only.

Validation (run as a script): the G -> 0 limit against the rung-0
closed forms (disc_reference), and gravity smoke tests.
"""
from __future__ import annotations

import dataclasses
import math
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from disc_models import DiscGravityModel  # noqa: E402


# ---------------------------------------------------------------- mesh

def _gll_nodes(p: int) -> np.ndarray:
    """Gauss-Legendre-Lobatto nodes on [-1, 1] (Lagrange basis)."""
    if p == 1:
        return np.array([-1.0, 1.0])
    # roots of (1 - x^2) P'_p(x): eigenvalue-free via Chebyshev init
    x = np.cos(np.pi * np.arange(p + 1) / p)[::-1].copy()
    P = np.zeros((p + 1, p + 1))
    for _ in range(100):
        P[:, 0] = 1.0
        P[:, 1] = x
        for k in range(2, p + 1):
            P[:, k] = ((2 * k - 1) * x * P[:, k - 1]
                       - (k - 1) * P[:, k - 2]) / k
        dx = (x * P[:, p] - P[:, p - 1]) / ((p + 1) * P[:, p])
        x -= dx
        if np.max(np.abs(dx)) < 1e-15:
            break
    return x


def _lagrange_eval(nodes: np.ndarray, x: np.ndarray):
    """Values and derivatives of the Lagrange basis at points x."""
    n = len(nodes)
    V = np.ones((len(x), n))
    D = np.zeros((len(x), n))
    for i in range(n):
        for k in range(n):
            if k == i:
                continue
            V[:, i] *= (x - nodes[k]) / (nodes[i] - nodes[k])
        for j in range(n):
            if j == i:
                continue
            term = np.ones(len(x)) / (nodes[i] - nodes[j])
            for k in range(n):
                if k in (i, j):
                    continue
                term *= (x - nodes[k]) / (nodes[i] - nodes[k])
            D[:, i] += term
    return V, D


@dataclass
class _Segment:
    lo: float
    hi: float
    nel: int


class RadialSolution:
    def __init__(self, l, r_nodes_body, U, V, r_nodes_pot, Z, model):
        self.l = l
        self._rb, self._U, self._V = r_nodes_body, U, V
        self._rp, self._Z = r_nodes_pot, Z
        self.model = model

    def _interp(self, r, xs, ys):
        return float(np.interp(r, xs, ys))

    def responses(self) -> dict:
        m = self.model
        out = {
            "ur_a": self._interp(m.a, self._rb, self._U),
            "ur_c": self._interp(m.c, self._rb, self._U),
            "z_a": self._interp(m.a, self._rp, self._Z),
        }
        if self._V is not None:
            out["ut_a"] = self._interp(m.a, self._rb, self._V)
            out["ut_c"] = self._interp(m.c, self._rb, self._V)
        return out


def solve_degree(model: DiscGravityModel, l: int, *, p: int = 8,
                 nel: tuple[int, int, int] = (10, 10, 6),
                 R: float = 1.5, t: float = 1.0,
                 eps_reg: float = 1e-10,
                 method: str = "welded",
                 _return_system: bool = False) -> RadialSolution:
    """Degree-l solve under the dead surface traction
    -t cos(l theta) rhat (shear-free). `method`:

      welded  full fluid, continuous displacement (the default);
      slip    full fluid, free-slip CMB: U(c) shared (exact normal
              continuity), V broken at c (one extra dof), the zeta
              jump collapses at the identity (b radial), and the
              interface forms B_Sigma (pi = p0(c)) and G_Sigma in
              their circle amplitudes; l = 0 is welded identically;
      dahlen  the Eulerian mixed-class system transcribed as its own
              assembly path (doc/quasi_static_models.tex eq. (mixedA)
              + (F1)-(F3)): effective moduli with the deviatoric
              split, the symmetrised rho-advective terms, the
              volume + interface coupling c, Eulerian phi
              (continuous — no zeta glue), no buffer u-extension,
              fluid displacement eliminated. At l = 0 the reduction
              misses the fluid's bulk modulus (M4), so this method
              dispatches to the welded full-fluid solve — exactly
              pyslfp's l = 0 treatment in this organisation.

    Every interface-form sign is pinned by the aw_core_2d null test
    (welded == slip == dahlen on the neutral core)."""
    assert l == 0 or l >= 2, "l = 1 not implemented yet"
    assert method in ("welded", "slip", "dahlen")
    if l == 0 and method == "slip":
        method = "welded"  # no tangential channel at l = 0
    if l == 0 and method == "dahlen":
        method = "welded"  # the l = 0 tweak: full fluid (M4 hole)
    slip = method == "slip"
    dahlen = method == "dahlen"
    G = model.G
    c, a = model.c, model.a
    has_V = l >= 1

    # nodes: body segments [0,c],[c,a]; potential adds [a,R]
    ref = _gll_nodes(p)
    segs = [_Segment(0.0, c, nel[0]), _Segment(c, a, nel[1]),
            _Segment(a, R, nel[2])]

    def seg_nodes(seg):
        edges = np.linspace(seg.lo, seg.hi, seg.nel + 1)
        els = []
        for e in range(seg.nel):
            els.append(0.5 * (edges[e] + edges[e + 1])
                       + 0.5 * (edges[e + 1] - edges[e]) * ref)
        return els

    body_els = seg_nodes(segs[0]) + seg_nodes(segs[1])
    buffer_els = seg_nodes(segs[2])
    pot_els = body_els + buffer_els

    def global_nodes(els):
        xs = [els[0]]
        idx = [np.arange(len(els[0]))]
        start = len(els[0]) - 1
        for e in els[1:]:
            xs.append(e[1:])
            idx.append(start + np.arange(len(e)))
            start += len(e) - 1
        return np.concatenate(xs), idx

    rb, body_idx = global_nodes(body_els)   # body nodes (U, V)
    rp, pot_idx = global_nodes(pot_els)     # potential nodes

    nb, npnt = len(rb), len(rp)
    ic = int(np.argmin(np.abs(rb - c)))      # interface node in rb
    nfl = len(body_els) // 2                  # fluid elements count
    if dahlen:
        # u lives on the mantle only: map body node k -> mantle dof
        # (k - ic), fluid nodes carry no u.
        nU = nb - ic
        nV = nU if has_V else 0
    else:
        nU = nb
        nV = nb if has_V else 0
    n_extra = 1 if (slip and has_V) else 0    # broken V_f(c)
    ndof = nU + nV + n_extra + npnt
    if dahlen:
        iU = lambda k: k - ic
        iV = lambda k: nU + (k - ic)
    else:
        iU = lambda k: k
        iV = lambda k: nU + k
    iVfc = nU + nV                            # slip: fluid V at c
    iZ = lambda k: nU + nV + n_extra + k

    K = np.zeros((ndof, ndof))
    F = np.zeros(ndof)

    # quadrature on the reference element
    gq, gw = np.polynomial.legendre.leggauss(p + 6)
    Vq, Dq = _lagrange_eval(ref, gq)

    def el_map(el):
        lo, hi = el[0], el[-1]
        J = 0.5 * (hi - lo)
        r = 0.5 * (lo + hi) + J * gq
        return r, J

    four_pi_G = 4.0 * math.pi * max(G, 1e-300)
    gc = float(model.g(c))
    p0c = float(model.p0(c))

    # ------------------------------------------------ element kernels
    def add_element(el, bi, pi_, in_fluid, in_buffer):
        r, J = el_map(el)
        w = gw * J * r                      # r dr measure
        nloc = len(el)
        zmap = [[(iZ(pi_[jj]), 1.0)] for jj in range(nloc)]
        dahlen_fluid = dahlen and in_fluid and not in_buffer

        # per-dof amplitude tables at the quadrature points:
        # u-dofs: columns [U, U', V, V'] per local dof (U-dofs then
        # V-dofs); buffer: the taper pair (U_a, V_a slots).
        if in_buffer and dahlen:
            uU = np.zeros((len(r), 0, 2))
            n_u = 0
        elif in_buffer:
            tau = (R - r) / (R - a)
            dtau = -np.ones_like(r) / (R - a)
            uU = np.stack([tau, dtau], axis=-1)[:, None, :]   # 1 dof
            n_u = 1
        elif dahlen_fluid:
            uU = np.zeros((len(r), 0, 2))                     # no u
            n_u = 0
        else:
            uU = np.stack([Vq, Dq / J], axis=-1)              # nloc dofs
            n_u = nloc

        rho0 = None  # density never appears in the perturbation rows
        p0 = np.zeros_like(r) if in_buffer else model.p0(r)
        g0 = model.g(r)
        if in_fluid:
            kap, mu_eff = model.kappa[0](r), model.mu[0](r)
        elif in_buffer:
            kap = mu_eff = np.zeros_like(r)
        else:
            kap, mu_eff = model.kappa[1](r), model.mu[1](r)
        lam_b = (kap - mu_eff) - p0          # 2-D dictionary
        mu_b = mu_eff + p0
        if in_fluid:
            # negligible deviatoric pin of the fluid's near-null
            # directions (the 1-D counterpart of the gauge penalty;
            # bias O(eps_reg) of the response)
            mu_b = mu_b + eps_reg * kap
            lam_b = lam_b - eps_reg * kap

        # amplitude builders for one scalar shape (N, N') in the U or
        # V slot: H = [[U', -(lU+V)/r], [V', (U+lV)/r]]
        def amps(N, dN, slot):
            zero = np.zeros_like(N)
            U = N if slot == 0 else zero
            dU = dN if slot == 0 else zero
            Vv = N if slot == 1 else zero
            dV = dN if slot == 1 else zero
            H00 = dU
            H01 = -(l * U + Vv) / r
            H10 = dV
            H11 = (U + l * Vv) / r
            err = dU
            ett = (U + l * Vv) / r
            two_ert = dV - (Vv + l * U) / r
            d = H00 + H11
            return np.stack([H00, H01, H10, H11, err, ett, two_ert, d,
                             U, Vv], axis=-1)

        slots = [0] + ([1] if has_V else [])
        A = []     # per u-dof amplitude rows
        dofmap = []  # per local u-dof: list of (global dof, weight)
        el_in_fluid_region = (not in_buffer) and in_fluid
        for slot in slots:
            for i in range(n_u):
                A.append(amps(uU[:, i, 0], uU[:, i, 1], slot))
                if in_buffer:
                    g_ = iU(nb - 1) if slot == 0 else iV(nb - 1)
                    dofmap.append([(g_, 1.0)])
                elif slip and slot == 1 and el_in_fluid_region \
                        and bi[i] == ic:
                    dofmap.append([(iVfc, 1.0)])   # broken V_f(c)
                else:
                    dofmap.append([(iU(bi[i]), 1.0) if slot == 0
                                   else (iV(bi[i]), 1.0)])
        if A:
            A = np.stack(A, axis=0)  # (ndof_u, nq, 10)
        else:
            A = np.zeros((0, len(r), 10))
        H = A[..., 0:4]
        err, ett, two_ert, d = (A[..., 4], A[..., 5], A[..., 6], A[..., 7])
        UVvals = A[..., 8:10]

        nu = len(dofmap)

        if dahlen and not in_buffer and not in_fluid:
            # ---- the Eulerian mixed solid (eq. mixedA, transcribed):
            # effective moduli with the deviatoric split, the
            # symmetrised rho-advective terms, and the volume part of
            # the coupling c. g' = 4 pi G rho - g / r analytically.
            rho_s = model.rho[1](r)
            dg = 4.0 * math.pi * G * rho_s - g0 / r
            # kappa d d' + 2 mu (eps:eps' - d d' / 2)
            mat = (kap * d[:, None, :] * d[None, :, :]
                   + 2.0 * mu_eff * (err[:, None, :] * err[None, :, :]
                                     + ett[:, None, :] * ett[None, :, :]
                                     + 0.5 * two_ert[:, None, :]
                                     * two_ert[None, :, :]
                                     - 0.5 * d[:, None, :]
                                     * d[None, :, :]))
            # amplitudes: U_i = H11*r - l V... recover U, V values from
            # the stored amplitudes: U = r*H11 - l*V; easier: rebuild
            # from err/ett is ambiguous — store U,V directly instead.
            Uv = UVvals[..., 0]
            Vv = UVvals[..., 1]
            # grad(u . grad Phi0) . u' : ((gU)' Ubar - (l g U / r) Vbar)
            gU = g0 * Uv
            dgU = dg * Uv + g0 * H[..., 0]      # (gU)' = g'U + gU'
            adv = 0.5 * rho_s * (
                dgU[:, None, :] * Uv[None, :, :]
                + dgU[None, :, :] * Uv[:, None, :]
                - (l * gU[:, None, :] / r) * Vv[None, :, :]
                - (l * gU[None, :, :] / r) * Vv[:, None, :])
            # - 1/2 rho [ (u.gradPhi0) d' + sym ]
            adv2 = -0.5 * rho_s * (gU[:, None, :] * d[None, :, :]
                                   + gU[None, :, :] * d[:, None, :])
            Kel = ((mat + adv + adv2) * w).sum(axis=-1)
            for ii in range(nu):
                for gi, wi in dofmap[ii]:
                    for jj in range(nu):
                        for gj, wj in dofmap[jj]:
                            K[gi, gj] += wi * wj * Kel[ii, jj]
            # coupling c, volume part: rho (grad phi . u' + sym):
            # rho [ Z' Ubar - (l/r) Z Vbar ] symmetrised
            Zv0, Zd0 = Vq, Dq / J
            for ii in range(nu):
                for jj in range(nloc):
                    val = np.sum(rho_s * (Zd0[:, jj] * Uv[ii]
                                          - (l / r) * Zv0[:, jj] * Vv[ii])
                                 * w)
                    for gi, wi in dofmap[ii]:
                        K[gi, iZ(pi_[jj])] += wi * val
                        K[iZ(pi_[jj]), gi] += wi * val

        # material + geometric (skip in buffer: vacuum; none in the
        # dahlen fluid; the dahlen solid used the mixed kernels above)
        if not in_buffer and nu > 0 and not dahlen:
            mat = (lam_b * d[:, None, :] * d[None, :, :]
                   + 2.0 * mu_b * (err[:, None, :] * err[None, :, :]
                                   + ett[:, None, :] * ett[None, :, :])
                   + mu_b * two_ert[:, None, :] * two_ert[None, :, :])
            geo = -p0 * np.einsum('inq,jnq->ijq',
                                  A[:, :, 0:4].transpose(0, 2, 1),
                                  A[:, :, 0:4].transpose(0, 2, 1))
            Kel = ((mat + geo) * w).sum(axis=-1)
            for ii in range(nu):
                for gi, wi in dofmap[ii]:
                    for jj in range(nu):
                        for gj, wj in dofmap[jj]:
                            K[gi, gj] += wi * wj * Kel[ii, jj]

        # F1 of the Dahlen fluid: + rho'_F zeta zeta' (negative mass
        # term for stable-or-neutral stratification)
        if dahlen_fluid:
            rp_f = model.rho_prime_core(r)
            with np.errstate(invalid="ignore", divide="ignore"):
                rhoF = np.where(g0 > 0, rp_f / np.maximum(g0, 1e-300), 0.0)
            Zv0 = Vq
            for ii in range(nloc):
                for jj in range(nloc):
                    val = np.sum(rhoF * Zv0[:, ii] * Zv0[:, jj] * w)
                    for za, wa in zmap[ii]:
                        for zb, wb in zmap[jj]:
                            K[za, zb] += wa * wb * val

        # gravity-gravity: (1/8 pi G) g^2 a''_rr(u, v) — the
        # referential organisation only
        if G > 0 and nu > 0 and not dahlen:
            H00, H01, H10, H11 = (H[..., 0], H[..., 1], H[..., 2],
                                  H[..., 3])

            def pair(i, j):
                # like-channel contractions of the 2x2 amplitude
                # matrices (doc/gravitating_elasticity.md a'' at id)
                a00, a01, a10, a11 = H00[i], H01[i], H10[i], H11[i]
                b00, b01, b10, b11 = H00[j], H01[j], H10[j], H11[j]
                di, dj = d[i], d[j]
                trAB = a00 * b00 + a01 * b10 + a10 * b01 + a11 * b11
                AB00 = a00 * b00 + a01 * b10
                BA00 = b00 * a00 + b01 * a10
                ABt00 = a00 * b00 + a01 * b01
                BAt00 = b00 * a00 + b01 * a01
                return (di * dj - trAB - 2.0 * di * b00 - 2.0 * dj * a00
                        + 2.0 * (AB00 + BA00) + (ABt00 + BAt00))

            coef = g0 ** 2 / (2.0 * four_pi_G)
            for ii in range(nu):
                for jj in range(nu):
                    val = np.sum(coef * pair(ii, jj) * w)
                    for gi, wi in dofmap[ii]:
                        for gj, wj in dofmap[jj]:
                            K[gi, gj] += wi * wj * val

            # coupling (1/4 pi G) <a'(u) g0, grad chi>, both slots
            Zv, Zd = Vq, Dq / J
            cr = (d - 2.0 * H00)            # cos channel with Z'
            ct = (H10 + H01)                # sin channel with (l/r) Z
            for ii in range(nu):
                for jj in range(nloc):
                    val = np.sum(g0 / four_pi_G
                                 * (cr[ii] * Zd[:, jj]
                                    + ct[ii] * (l / r) * Zv[:, jj]) * w)
                    for zg, zw in zmap[jj]:
                        for gi, wi in dofmap[ii]:
                            K[gi, zg] += wi * zw * val
                            K[zg, gi] += wi * zw * val

        # Laplace block
        Zv, Zd = Vq, Dq / J
        lap = (np.einsum('qi,qj->ijq', Zd, Zd)
               + (l ** 2 / r ** 2) * np.einsum('qi,qj->ijq', Zv, Zv))
        Kel = (lap / four_pi_G * w).sum(axis=-1)
        for ii in range(nloc):
            for jj in range(nloc):
                for za, wa in zmap[ii]:
                    for zb, wb in zmap[jj]:
                        K[za, zb] += wa * wb * Kel[ii, jj]

    for k, el in enumerate(body_els):
        add_element(el, body_idx[k], pot_idx[k],
                    in_fluid=el[-1] <= c + 1e-14, in_buffer=False)
    for k, el in enumerate(buffer_els):
        add_element(el, None, pot_idx[len(body_els) + k],
                    in_fluid=False, in_buffer=True)

    # ---------------- interface terms at r = c ----------------
    if slip and has_V:
        # B_Sigma + G_Sigma in circle amplitudes (/pi normalisation,
        # x c measure already folded):
        #   B = + p0(c) S (l SumU + SumV),
        #   G = + (g^2/8 pi G) S (l SumU + SumV) - (g l / 2 pi G) S Z,
        # with S = V_f - V_s, SumU = 2 U(c), SumV = V_s + V_f; applied
        # in symmetrised polarisation. Signs pinned by the aw null
        # test.
        coef_BG = p0c + gc * gc / (2.0 * four_pi_G)
        dU, dVs, dVf, dZ = iU(ic), iV(ic), iVfc, iZ(ic)
        # S-slot vector and (l SumU + SumV)-slot vector over the dofs
        pairs = []  # (dof, S-weight, T-weight) with T = l SumU + SumV
        pairs.append((dU, 0.0, 2.0 * l))
        pairs.append((dVs, -1.0, 1.0))
        pairs.append((dVf, 1.0, 1.0))
        for da, Sa, Ta in pairs:
            for db, Sb, Tb in pairs:
                K[da, db] += 0.5 * coef_BG * (Sa * Tb + Sb * Ta)
        qlc = gc * l / (2.0 * math.pi * max(G, 1e-300))
        for da, Sa, Ta in pairs:
            K[da, dZ] += -0.5 * qlc * Sa
            K[dZ, da] += -0.5 * qlc * Sa

    if dahlen:
        # (F2) and the interface part of the coupling c ((F3)), with
        # m = -rhat (fluid below), m . grad Phi0 = -g(c),
        # m . u = -U(c), phi = Z(c) (Eulerian, continuous); x c, /pi:
        rhoF_c = float(model.rho[0](np.array([c]))[0])
        dU, dZ = iU(ic), iZ(ic)
        # F2 = -rho_F (m.gradPhi0)(m.u)(m.u') -> + rho_F g(c) c U U'
        K[dU, dU] += rhoF_c * gc * c
        # F3 = -rho_F [phi (m.u') + phi' (m.u)] -> + rho_F c (Z U)
        K[dU, dZ] += rhoF_c * c
        K[dZ, dU] += rhoF_c * c

    # DtN at R (l >= 1): harmonic-decay energy l Z(R)^2 / (4 pi G)
    if l >= 1:
        K[iZ(npnt - 1), iZ(npnt - 1)] += l / four_pi_G
    else:
        # l = 0: referential mass conservation kills the monopole;
        # fix the exterior constant
        pass

    # load: dead traction -t cos(l theta) rhat at r = a
    F[iU(nb - 1)] += -t * a  # iU maps nb-1 correctly in every method

    # essential conditions
    fixed = []
    if l >= 2:
        fixed += [iZ(0)]
        if not dahlen:
            fixed += [iU(0), iV(0)]
    else:
        if not dahlen:
            fixed += [iU(0)]
        fixed += [iZ(npnt - 1)]  # l = 0 exterior constant gauge
    if _return_system:
        # private: the assembled (K, F) and dof maps, for reuse by
        # solve_degree_maxwell (operator-level Maxwell iteration)
        return {"K": K, "F": F, "fixed": fixed, "ndof": ndof,
                "nU": nU, "nV": nV, "n_extra": n_extra,
                "nb": nb, "ic": ic, "npnt": npnt, "rb": rb, "rp": rp,
                "iU": iU, "iV": iV, "iZ": iZ}
    keep = np.setdiff1d(np.arange(ndof), fixed)
    Kr = K[np.ix_(keep, keep)]
    Fr = F[keep]
    # Jacobi equilibration: the Laplace block scales as 1/4 pi G and
    # at small G swamps the elastic scales; the scaled system is
    # O(1) throughout and a direct solve is accurate.
    dscale = 1.0 / np.sqrt(np.maximum(np.abs(np.diag(Kr)), 1e-300))
    Ks = dscale[:, None] * Kr * dscale[None, :]
    xr = dscale * np.linalg.solve(Ks, dscale * Fr)
    x = np.zeros(ndof)
    x[keep] = xr

    r_sol = rb[ic:] if dahlen else rb
    U = x[:nU]
    Vsol = x[nU:nU + nV] if has_V else None
    Z = x[nU + nV + n_extra:]
    out = RadialSolution(l, r_sol, U, Vsol, rp, Z, model)
    if slip and has_V:
        out.ut_c_fluid = float(x[iVfc])
        out.slip_amplitude = float(x[iVfc] - x[iV(ic)])
    return out


def solve_degree_maxwell(model: DiscGravityModel, l: int, *, p: int = 8,
                         nel: tuple[int, int, int] = (10, 10, 6),
                         R: float = 1.5, t: float = 1.0,
                         mu_core: float | None = None,
                         dt_over_tau: float = 3.0,
                         tol: float = 1e-10,
                         stag_ratio: float = 0.9,
                         min_steps: int = 4,
                         max_steps: int = 400,
                         escalate: float = 10.0,
                         beta_max: float = 1e12,
                         mode: str = "converge") -> RadialSolution:
    """Degree-l static response by Maxwell relaxation of the core.

    The core is made an artificial Maxwell solid (shear modulus
    mu_core, Maxwell time tau), the dead load is applied as a step,
    and the quasi-static system is relaxed by backward Euler until the
    SOLID region stops moving — the secular route to the static
    response (doc/static_fluid_core.tex), computed with nothing but
    welded solid-elastic solves: no fluid region, no slip machinery,
    no kernel handling beyond the usual, any geometry.

    The problem is linear, so the Maxwell internal variable is the
    deviatoric core strain of a memory displacement w and one step is

        (K0 + gamma B) u_{n+1} = F + gamma B w_n,
        w_{n+1} = (w_n + beta u_{n+1}) / (1 + beta),

    with beta = dt/tau, gamma = mu_core / (1 + beta), K0 the welded
    operator at core mu = 0 (no eps_reg pin — the Maxwell term is the
    regulariser) and B the unit-shear deviatoric core stiffness
    (exact, since K is linear in the core mu). The fixed point is the
    K0 solution, independent of mu_core, beta and the schedule.

    MEASURED trajectory structure (fc/aw/stable twins, l = 2): the
    elastic phase contracts at 1/(1 + beta) per step and is done in
    ~10 steps; what follows is a CONFIGURATIONAL cascade of viscous
    gravitational relaxation modes with rates ~ theta/mu_core,
    theta = lambda/b down to ~1e-4 — physical times 1e4+ tau, so a
    "smallish multiple of tau" only ever completes the elastic phase.
    At N^2 < 0 the cascade additionally contains genuine
    Rayleigh-Taylor GROWTH (on fc the growing rates are ~300x the
    slowest stable ones: no plateau window, the static question's
    ambiguity band shown in time). Backward Euler is L-stable, so
    escalating beta past ~2 mu b/|lambda| stabilises every growing
    mode: the default schedule steps physically at beta = dt_over_tau
    until the solid increment stagnates, then multiplies beta by
    `escalate` per step (continuation gamma -> 0, i.e. toward the
    static welded solve with a vanishing deviatoric pin) until the
    increment is below tol. Cost: ~25-35 solves over ~5 distinct
    operators. mode="plateau" instead stops at stagnation and returns
    the physical plateau (the honest finite-time answer; on fc it
    differs from the fixed point by the band, percent-level).

    Stopping monitors the relative per-step SOLID displacement
    increment only — the fields the secular limit defines; the fluid
    displacement does not converge and must not be monitored.

    mu_core defaults to the mantle shear at the CMB. Diagnostics in
    .maxwell: the converged value's step count and operator count,
    the stagnation step and plateau value (their gap is a band
    estimate at N^2 != 0), and the full step history."""
    assert l == 0 or l >= 2, "l = 1 not implemented yet"
    assert mode in ("converge", "plateau")
    if mu_core is None:
        mu_core = float(np.asarray(model.mu[1](np.array([model.c])),
                                   dtype=float).ravel()[0])

    def const_mu(v):
        return lambda r: np.full_like(np.asarray(r, dtype=float), v)

    kw = dict(p=p, nel=nel, R=R, t=t, eps_reg=0.0, _return_system=True)
    s0 = solve_degree(dataclasses.replace(
        model, mu=(const_mu(0.0), model.mu[1])), l, **kw)
    s1 = solve_degree(dataclasses.replace(
        model, mu=(const_mu(1.0), model.mu[1])), l, **kw)
    K0, F = s0["K"], s0["F"]
    B = s1["K"] - K0                 # exact: K is linear in core mu

    ndof, nb, ic = s0["ndof"], s0["nb"], s0["ic"]
    nU, nV, n_extra = s0["nU"], s0["nV"], s0["n_extra"]
    iU, iV = s0["iU"], s0["iV"]
    keep = np.setdiff1d(np.arange(ndof), s0["fixed"])
    K0r = K0[np.ix_(keep, keep)]
    Br = B[np.ix_(keep, keep)]
    Fr = F[keep]

    # the stopping metric's dofs: mantle U, V (U(c) shared, solid-owned)
    solid = [iU(k) for k in range(ic, nb)]
    if nV:
        solid += [iV(k) for k in range(ic, nb)]
    solid = np.array(solid)

    rb = s0["rb"]
    beta = float(dt_over_tau)
    last_beta = None
    Ks = dscale = None
    w = np.zeros(len(keep))
    x = np.zeros(len(keep))
    u = np.zeros(ndof)
    hist = []
    ref = None
    prev_solid = None
    prev_delta = None
    t_phys = 0.0
    stag_step = None
    plateau = None
    stop = "max_steps"
    n_ops = 0
    for n in range(1, max_steps + 1):
        if beta != last_beta:
            gamma = mu_core / (1.0 + beta)
            Kr = K0r + gamma * Br
            dscale = 1.0 / np.sqrt(np.maximum(np.abs(np.diag(Kr)),
                                              1e-300))
            Ks = dscale[:, None] * Kr * dscale[None, :]
            last_beta = beta
            n_ops += 1
        x = dscale * np.linalg.solve(Ks, dscale
                                     * (Fr + gamma * (Br @ w)))
        w = (w + beta * x) / (1.0 + beta)
        t_phys += beta
        u = np.zeros(ndof)
        u[keep] = x
        us = x[np.searchsorted(keep, solid)]
        if ref is None:
            ref = float(np.max(np.abs(us))) or 1.0
            delta = math.inf
        else:
            delta = float(np.max(np.abs(us - prev_solid))) / ref
        hist.append({"t_over_tau": t_phys, "beta": beta,
                     "ur_a": float(np.interp(model.a, rb, u[:nU])),
                     "delta_solid": delta})
        prev_solid = us.copy()
        if delta <= tol:
            stop = "converged"
            break
        if (stag_step is None and n > min_steps
                and prev_delta is not None
                and delta > stag_ratio * prev_delta):
            stag_step = n
            plateau = hist[-1]["ur_a"]
            if mode == "plateau":
                stop = "stagnation"
                break
        if stag_step is not None:
            beta = min(beta * escalate, beta_max)
        if np.isfinite(delta):
            prev_delta = delta

    U = u[:nU]
    Vsol = u[nU:nU + nV] if nV else None
    Z = u[nU + nV + n_extra:]
    out = RadialSolution(l, rb, U, Vsol, s0["rp"], Z, model)
    out.maxwell = {"mu_core": mu_core, "dt_over_tau": float(dt_over_tau),
                   "mode": mode, "steps": len(hist), "operators": n_ops,
                   "stop": stop, "stag_step": stag_step,
                   "plateau_ur_a": plateau,
                   "delta_floor": hist[-1]["delta_solid"],
                   "history": hist}
    return out


# ------------------------------------------------------------ validate

def validate(verbose: bool = True) -> bool:
    import disc_models
    from disc_reference import DiscModel, solve as solve0

    ok = True
    lines = []

    def record(label, value, tol):
        nonlocal ok
        good = value <= tol
        ok &= good
        lines.append(f"  {label:<52s} {value:10.3e}  "
                     f"{'ok' if good else 'FAIL'}")

    # G -> 0 limit vs rung 0 on fluid_core_2d's elastic data
    tiny = disc_models.fluid_core_2d(G_newton=1e-12)
    m0 = DiscModel(lam=disc_models.KAPPA_MANTLE - disc_models.MU_MANTLE,
                   mu=disc_models.MU_MANTLE,
                   kappa_f=disc_models.KAPPA_CORE,
                   c=disc_models.CMB, a=1.0)
    for l in (0, 2, 3, 4):
        s = solve_degree(tiny, l)
        ex = solve0(m0, l)
        r = s.responses()
        e = abs(r["ur_a"] - ex.responses["ur_a"]) / abs(
            ex.responses["ur_a"])
        record(f"G->0 l={l} ur_a vs rung 0", e, 1e-5)
        if l >= 2:
            e = abs(r["ut_a"] - ex.responses["ut_a"]) / abs(
                ex.responses["ut_a"])
            record(f"G->0 l={l} ut_a vs rung 0", e, 1e-5)

    # resolution independence: STRICT on the neutral core (well-posed);
    # the non-neutral core's drift is physics (the static operator
    # inverts at the edge of its essential spectrum for N^2 != 0) and
    # is RECORDED, bounded loosely — the floor all fluid_core_2d
    # comparisons inherit.
    fc = disc_models.fluid_core_2d()
    aw = disc_models.aw_core_2d()
    s1 = solve_degree(fc, 2, p=8).responses()
    s2 = solve_degree(fc, 2, p=11, nel=(14, 14, 8)).responses()
    a1 = solve_degree(aw, 2, p=8).responses()
    a2 = solve_degree(aw, 2, p=11, nel=(14, 14, 8)).responses()
    record("aw_core_2d l=2 p-ladder drift (strict)",
           abs(a1["ur_a"] - a2["ur_a"]) / abs(a2["ur_a"]), 1e-7)
    fc_drift = abs(s1["ur_a"] - s2["ur_a"]) / abs(s2["ur_a"])
    record("fluid_core_2d l=2 p-ladder drift (physics floor, < 2%)",
           fc_drift, 2e-2)

    # the twins differ (N^2 physics present), but modestly
    sa = a1
    split = abs(sa["ur_a"] - s1["ur_a"]) / abs(s1["ur_a"])
    record("aw vs fc l=2 split in (0.1%, 50%)",
           0.0 if 1e-3 < split < 0.5 else 1.0, 0.5)
    lines.append(f"  [info] fc l=2 non-neutral drift {fc_drift:.2e} "
                 "(regulariser- and resolution-sensitive by physics)")

    # ---- the discriminator stack ----
    # aw null tests: welded == slip == dahlen on the neutral core
    for l in (2, 3):
        w_ = solve_degree(aw, l).responses()["ur_a"]
        s_ = solve_degree(aw, l, method="slip").responses()["ur_a"]
        d_ = solve_degree(aw, l, method="dahlen").responses()["ur_a"]
        record(f"aw null l={l}: |slip/welded - 1|",
               abs(s_ / w_ - 1.0), 1e-8)
        record(f"aw null l={l}: |dahlen/welded - 1|",
               abs(d_ / w_ - 1.0), 1e-6)
    # G -> 0: slip == welded == rung 0
    sl = solve_degree(tiny, 2, method="slip").responses()["ur_a"]
    record("G->0 l=2 slip vs rung 0",
           abs(sl - solve0(m0, 2).responses["ur_a"])
           / abs(solve0(m0, 2).responses["ur_a"]), 1e-5)
    # fc splits (physics): measured, loosely bounded
    for l in (2,):
        w_ = solve_degree(fc, l).responses()["ur_a"]
        s_ = solve_degree(fc, l, method="slip").responses()["ur_a"]
        d_ = solve_degree(fc, l, method="dahlen").responses()["ur_a"]
        lines.append(f"  [info] fc l={l}: welded {w_:.6f}, slip {s_:.6f}"
                     f" (FRL/weld-suppression split {abs(s_/w_-1):.3%}),"
                     f" dahlen {d_:.6f} (vs slip — the genuine"
                     f" fluid-treatment gap — {abs(d_/s_-1):.3%})")

    # ---- Maxwell relaxation route (the secular time-stepping) ----
    # G -> 0: the relaxed core is the rung-0 fluid exactly
    for l in (0, 2):
        sm = solve_degree_maxwell(tiny, l)
        ex = solve0(m0, l).responses["ur_a"]
        record(f"maxwell G->0 l={l} ur_a vs rung 0 "
               f"({sm.maxwell['steps']} steps, {sm.maxwell['stop']})",
               abs(sm.responses()["ur_a"] - ex) / abs(ex), 1e-5)
    # aw null: the plateau joins welded == slip == dahlen
    for l in (2, 3):
        wv = solve_degree(aw, l).responses()["ur_a"]
        sm = solve_degree_maxwell(aw, l)
        record(f"maxwell aw null l={l}: |maxwell/welded - 1| "
               f"({sm.maxwell['steps']} steps, {sm.maxwell['stop']})",
               abs(sm.responses()["ur_a"] / wv - 1.0), 1e-6)
    # fc (N^2 < 0, RT growth in the cascade): the escalated iteration
    # must land within the band of dahlen; the physical plateau and
    # its gap to the limit are the band display
    d_fc = solve_degree(fc, 2, method="dahlen").responses()["ur_a"]
    w_fc = solve_degree(fc, 2).responses()["ur_a"]
    mm = solve_degree_maxwell(fc, 2)
    m_fc = mm.responses()["ur_a"]
    record("maxwell fc l=2 converged vs dahlen (within band, < 1%)",
           abs(m_fc / d_fc - 1.0), 1e-2)
    record("maxwell fc l=2 converged vs welded eps->0 (same limit)",
           abs(m_fc / w_fc - 1.0), 1e-3)
    lines.append(f"  [info] fc l=2 maxwell {m_fc:.6f} "
                 f"({mm.maxwell['steps']} steps, "
                 f"{mm.maxwell['operators']} operators, "
                 f"{mm.maxwell['stop']}); physical plateau "
                 f"{mm.maxwell['plateau_ur_a']:.6f} at step "
                 f"{mm.maxwell['stag_step']} — plateau-to-limit gap "
                 f"{abs(mm.maxwell['plateau_ur_a']/m_fc-1):.2%} = the "
                 f"RT-smeared band; dahlen {d_fc:.6f}, "
                 f"welded {w_fc:.6f}")
    # certificates: the limit is independent of the artificial clock
    # (exactly — the fixed point is the K0 solution)
    m_mu = solve_degree_maxwell(
        fc, 2, mu_core=4.0 * disc_models.MU_MANTLE).responses()["ur_a"]
    m_dt = solve_degree_maxwell(fc, 2,
                                dt_over_tau=1.5).responses()["ur_a"]
    record("maxwell fc l=2 limit shift under mu_core x 4 (strict)",
           abs(m_mu / m_fc - 1.0), 1e-7)
    record("maxwell fc l=2 limit shift under dt/tau 3 -> 1.5 (strict)",
           abs(m_dt / m_fc - 1.0), 1e-7)
    # stable twin (N^2 > 0: AW density, kappa x 2): no RT, converges;
    # dahlen is kappa-blind (identical to aw) while maxwell sees the
    # stratification — the band made visible by a method pair
    stable = disc_models._attach_state(dataclasses.replace(
        aw, kappa=(lambda r: 2.0 * aw.kappa[0](np.asarray(r)),
                   aw.kappa[1])))
    ms = solve_degree_maxwell(stable, 2)
    d_st = solve_degree(stable, 2, method="dahlen").responses()["ur_a"]
    record(f"maxwell stable (N^2>0) l=2 vs dahlen (within band, < 1%) "
           f"({ms.maxwell['steps']} steps, {ms.maxwell['stop']})",
           abs(ms.responses()["ur_a"] / d_st - 1.0), 1e-2)

    if verbose:
        print("disc_radial validation "
              f"({'PASS' if ok else 'FAIL'}):")
        print("\n".join(lines))
        print(f"  [info] fc l=2 ur_a = {s1['ur_a']:.6f}, "
              f"aw l=2 ur_a = {sa['ur_a']:.6f}, split {split:.3%}")
    return ok


if __name__ == "__main__":
    raise SystemExit(0 if validate() else 1)
