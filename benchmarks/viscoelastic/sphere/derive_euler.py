"""Derive the Euler matrix M of the static, non-gravitating spheroidal
system (reference.py) from Navier's equations, and check reference.py's
hand-coded matrix against it.

u_r = U(r) P_l(cos theta), u_theta = V(r) dP_l/dtheta in 3-D
(axisymmetric, order 0; P'' eliminated by Legendre's equation), u_r =
U(r) cos(l theta), u_theta = V(r) d/dtheta cos(l theta) in 2-D; Hooke's
law with lambda, mu; the two equilibrium equations; then the first-order
system for (U, V, R, S), sigma_rr = R Y, sigma_rtheta = S dY/dtheta, and
the Euler form r dY/dr = M Y for Y = (U, V, rR, rS).

    ./derive_euler.py      # prints M, checks the eigenvalues and
                           # reference.euler_matrix at sample moduli
"""
import sys
from pathlib import Path

import numpy as np
import sympy as sp
r, th, lam, mu, l = sp.symbols('r theta lambda mu l', positive=True)
U, V = sp.Function('U')(r), sp.Function('V')(r)

def system(dim):
    if dim == 3:
        P = sp.Function('P')(th)
        L = l*(l+1)
        legendre = {P.diff(th, 2): -sp.cot(th)*P.diff(th) - L*P}
        ur, ut = U*P, V*P.diff(th)
        err = ur.diff(r)
        ett = ut.diff(th)/r + ur/r
        epp = ur/r + ut*sp.cot(th)/r
        ert = sp.Rational(1, 2)*(ut.diff(r) - ut/r + ur.diff(th)/r)
        div = err + ett + epp
        srr, stt, spp, srt = lam*div+2*mu*err, lam*div+2*mu*ett, lam*div+2*mu*epp, 2*mu*ert
        eq_r = srr.diff(r) + srt.diff(th)/r + (2*srr - stt - spp + srt*sp.cot(th))/r
        eq_t = srt.diff(r) + stt.diff(th)/r + ((stt - spp)*sp.cot(th) + 3*srt)/r
        d2 = legendre
        d3 = {P.diff(th, 3): sp.diff(-sp.cot(th)*P.diff(th) - L*P, th)}
        def red(e):
            e = sp.expand(e.subs(d3).subs(d2).subs(d2))
            return sp.simplify(e)
        Rc = sp.simplify(red(srr)/P)              # sigma_rr = R P
        Sc = sp.simplify(red(srt)/P.diff(th))     # sigma_rt = S dP/dtheta
        er = sp.simplify(red(eq_r)/P)
        et = sp.simplify(red(eq_t)/P.diff(th))
        return Rc, Sc, er, et
    else:
        k = l
        ur, ut = U*sp.cos(k*th), V*(-k*sp.sin(k*th))  # u_t = V dY/dtheta
        err = ur.diff(r)
        ett = ut.diff(th)/r + ur/r
        ert = sp.Rational(1, 2)*(ut.diff(r) - ut/r + ur.diff(th)/r)
        div = err + ett
        srr, stt, srt = lam*div+2*mu*err, lam*div+2*mu*ett, 2*mu*ert
        eq_r = srr.diff(r) + srt.diff(th)/r + (srr - stt)/r
        eq_t = srt.diff(r) + stt.diff(th)/r + 2*srt/r
        Rc = sp.simplify(srr/sp.cos(k*th))
        Sc = sp.simplify(srt/(-k*sp.sin(k*th)))
        er = sp.simplify(sp.expand(eq_r)/sp.cos(k*th))
        et = sp.simplify(sp.expand(eq_t)/(-k*sp.sin(k*th)))
        return Rc, Sc, er, et

def derived():
  for dim in (2, 3):
      Rc, Sc, er, et = system(dim)
      # variables y = (U, V, R, S); unknown derivatives U', V', R', S'
      Rs, Ss = sp.symbols('R S')
      Up, Vp, Rp, Sp_ = sp.symbols('Up Vp Rp Sp')
      sub = {U.diff(r, 2): sp.Symbol('U2'), V.diff(r, 2): sp.Symbol('V2'),
             U.diff(r): Up, V.diff(r): Vp}
      # R = Rc(U,U',V), S = Sc(...) -> solve for U', V'
      sol = sp.solve([sp.Eq(Rs, Rc.subs(sub)), sp.Eq(Ss, Sc.subs(sub))], [Up, Vp], dict=True)[0]
      # R' = d/dr Rc ; equilibrium er=0, et=0 contain U'', V''
      Rd = sp.diff(Rc, r).subs(sub); Sd = sp.diff(Sc, r).subs(sub)
      e1, e2 = er.subs(sub), et.subs(sub)
      s2 = sp.solve([e1, e2], [sp.Symbol('U2'), sp.Symbol('V2')], dict=True)[0]
      Rprime = sp.simplify(Rd.subs(s2).subs(sol))
      Sprime = sp.simplify(Sd.subs(s2).subs(sol))
      Uprime, Vprime = sp.simplify(sol[Up]), sp.simplify(sol[Vp])
      Usym, Vsym = sp.symbols('Us Vs')
      rep = {U: Usym, V: Vsym}
      rhs = [e.subs(rep) for e in (Uprime, Vprime, Rprime, Sprime)]
      # Euler form with Y = (U, V, rR, rS): r Y' = M Y
      y = [Usym, Vsym, Rs, Ss]
      M = sp.zeros(4, 4)
      exprs = [r*rhs[0], r*rhs[1], r*(Rs + r*rhs[2]), r*(Ss + r*rhs[3])]  # r d(rR)/dr = r(R + r R')
      Ysub = {Rs: sp.Symbol('Y3')/r, Ss: sp.Symbol('Y4')/r}
      Yv = [Usym, Vsym, sp.Symbol('Y3'), sp.Symbol('Y4')]
      for i, e in enumerate(exprs):
          e = sp.expand(sp.simplify(e.subs(Ysub)))
          for j, v in enumerate(Yv):
              M[i, j] = sp.simplify(e.coeff(v))
      yield dim, M


def main() -> None:
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    import reference
    for dim, M in derived():
        print(f"dim {dim}:")
        sp.pprint(M)
        for lv, lamv, muv in ((2, 2.0, 1.0), (3, 0.7, 1.3), (5, 1.1, 0.4)):
            Mn = np.array(M.subs({lam: lamv, mu: muv, l: lv}),
                          dtype=complex)
            ref = reference.euler_matrix(dim, lv, lamv, muv)
            err = np.abs(Mn - ref).max()
            ev = sorted(np.linalg.eigvals(Mn).real)
            print(f"  l={lv}: |M - reference| = {err:.1e}, "
                  f"eigenvalues {np.round(ev, 12)}")
            assert err < 1e-12


if __name__ == "__main__":
    main()
