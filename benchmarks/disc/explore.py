"""Interactive exploration of the 2-D disc problems (rung 2).

Solves the per-degree radial problems of `disc_radial.py` for any
combination of model, method and solver knobs, prints the observables
and the method splits, and plots the radial profiles — the quickest
way to *see* the physics of this family: the welded/slip/dahlen
comparison, the Adams-Williamson null, the non-neutral floor (rerun
with --p or --eps-reg and watch the full-elastic fluid wander while
dahlen stands still), and the Eulerian potential staying smooth while
the referential fluid fields carry the near-null noise.

Examples (run from <build>/benchmarks/disc as ./explore, or directly
with the benchmarks Python):

  ./explore                                   # fc, l=2, all methods
  ./explore --model aw                        # the neutral twin: all
                                              # methods coincide
  ./explore --kappa-scale 10                  # stiffer core: N^2 / 10
  ./explore --l 3 --methods welded slip
  ./explore --p 12 --nel 20 20 10             # finer; fc wanders,
                                              # dahlen does not
  ./explore --eps-reg 1e-6                    # regulariser knob
  ./explore --G 1e-12 --rung0                 # gravity off, against
                                              # the rung-0 closed form
  ./explore --out profiles.png                # save instead of show

Outputs land where you run it (the build tree by convention).
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "common"))
import disc_models  # noqa: E402
import disc_radial  # noqa: E402
from outputs import outside_source  # noqa: E402


def build_model(args):
    if args.model == "aw":
        m = disc_models.aw_core_2d(G_newton=args.G)
        if args.kappa_scale != 1.0:
            raise SystemExit("--kappa-scale breaks the AW identity; "
                             "use --model fc")
        return m
    base = disc_models.fluid_core_2d(G_newton=args.G)
    if args.kappa_scale == 1.0:
        return base
    f = args.kappa_scale
    return disc_models._attach_state(disc_models.DiscGravityModel(
        "fluid_core_2d", base.c, base.a, G=base.G, rho=base.rho,
        rho_prime_core=base.rho_prime_core,
        kappa=(np.vectorize(lambda r: disc_models.KAPPA_CORE * f),
               base.kappa[1]),
        mu=base.mu))


def main() -> int:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", choices=["fc", "aw"], default="fc",
                   help="fluid_core_2d (N^2 < 0) or aw_core_2d (N^2 = 0)")
    p.add_argument("--kappa-scale", type=float, default=1.0,
                   help="scale the fc core bulk modulus (N^2 ~ 1/scale)")
    p.add_argument("--l", type=int, default=2, help="degree (0 or >= 2)")
    p.add_argument("--methods", nargs="+", default=["welded", "slip",
                                                    "dahlen", "maxwell"],
                   choices=["welded", "slip", "dahlen", "maxwell"])
    p.add_argument("--maxwell-mode", choices=["converge", "plateau"],
                   default="converge",
                   help="maxwell: escalate to the secular limit "
                        "(default) or stop at the physical plateau")
    p.add_argument("--p", type=int, default=8, help="element order")
    p.add_argument("--nel", type=int, nargs=3, default=[10, 10, 6],
                   metavar=("FLUID", "MANTLE", "BUFFER"))
    p.add_argument("--eps-reg", type=float, default=1e-10,
                   help="fluid deviatoric regulariser")
    p.add_argument("--G", type=float, default=disc_models.G,
                   help="gravitational constant (1e-12 ~ gravity off)")
    p.add_argument("--rung0", action="store_true",
                   help="also print the rung-0 closed form (meaningful "
                        "with --G tiny on --model fc)")
    p.add_argument("--out", default=None,
                   help="save the figure here instead of showing it")
    args = p.parse_args()
    if args.out:
        args.out = str(outside_source(Path(args.out)))

    model = build_model(args)
    sols = {}
    print(f"{args.model} (G = {args.G:g}, kappa x {args.kappa_scale:g}), "
          f"l = {args.l}, p = {args.p}, nel = {tuple(args.nel)}, "
          f"eps_reg = {args.eps_reg:g}")
    for m in args.methods:
        if m == "maxwell":
            s = disc_radial.solve_degree_maxwell(model, args.l,
                                                 p=args.p,
                                                 nel=tuple(args.nel),
                                                 mode=args.maxwell_mode)
        else:
            s = disc_radial.solve_degree(model, args.l, p=args.p,
                                         nel=tuple(args.nel),
                                         eps_reg=args.eps_reg, method=m)
        sols[m] = s
        r = s.responses()
        # the welded/slip potential variable is zeta = phi + u.gradPhi0;
        # dahlen's is the Eulerian phi: report phi uniformly
        ga = float(model.g(model.a))
        phi_a = r["z_a"] - (ga * r["ur_a"] if m != "dahlen" else 0.0)
        extra = ""
        if hasattr(s, "slip_amplitude"):
            extra = f"  slip amplitude {s.slip_amplitude:+.3e}"
        if hasattr(s, "maxwell"):
            mx = s.maxwell
            extra = (f"  [{mx['steps']} steps, {mx['operators']} ops, "
                     f"{mx['stop']}")
            if mx["plateau_ur_a"] is not None:
                extra += (f"; physical plateau {mx['plateau_ur_a']:+.7f}"
                          f" at step {mx['stag_step']}")
            extra += "]"
        print(f"  {m:>7}: ur_a {r['ur_a']:+.7f}"
              + (f"  ut_a {r['ut_a']:+.7f}" if "ut_a" in r else "")
              + f"  phi_a {phi_a:+.7f}" + extra)
    if len(sols) > 1:
        ms = list(sols)
        for i in range(len(ms)):
            for j in range(i + 1, len(ms)):
                a = sols[ms[i]].responses()["ur_a"]
                b = sols[ms[j]].responses()["ur_a"]
                print(f"  split {ms[i]}/{ms[j]}: "
                      f"{abs(a / b - 1.0):.3e}")
    if args.rung0:
        from disc_reference import DiscModel, solve as solve0
        m0 = DiscModel(lam=disc_models.KAPPA_MANTLE - disc_models.MU_MANTLE,
                       mu=disc_models.MU_MANTLE,
                       kappa_f=disc_models.KAPPA_CORE * args.kappa_scale,
                       c=model.c, a=1.0)
        ex = solve0(m0, args.l)
        print(f"  rung-0 exact: ur_a {ex.responses['ur_a']:+.7f}"
              + (f"  ut_a {ex.responses['ut_a']:+.7f}"
                 if args.l >= 1 else ""))

    import matplotlib
    if args.out:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    npanel = 3 if args.l >= 1 else 2
    fig, axes = plt.subplots(1, npanel, figsize=(5 * npanel, 4))
    axU, axZ = axes[0], axes[-1]
    for m, s in sols.items():
        axU.plot(s._rb, s._U, label=m)
        if args.l >= 1 and s._V is not None:
            axes[1].plot(s._rb, s._V, label=m)
        if m == "dahlen":
            axZ.plot(s._rp, s._Z, "--", label=f"{m}: phi", alpha=0.8)
        else:
            gvals = model.g(s._rp)
            Ur = np.interp(s._rp, s._rb, s._U,
                           left=np.nan, right=np.nan)
            phi = s._Z - gvals * np.where(np.isnan(Ur), 0.0, Ur)
            axZ.plot(s._rp, s._Z, label=f"{m}: zeta")
            axZ.plot(s._rp, phi, "--", label=f"{m}: phi", alpha=0.6)
    for ax, title in zip(axes, ["U(r)", "V(r)", "potential"][:npanel]):
        ax.axvline(model.c, color="k", lw=0.5, ls=":")
        ax.axvline(model.a, color="k", lw=0.5, ls=":")
        ax.set_title(title)
        ax.set_xlabel("r")
        ax.legend(fontsize=8)
    fig.suptitle(f"{args.model}, l = {args.l}; dotted lines: CMB, surface")
    fig.tight_layout()
    if args.out:
        fig.savefig(args.out, dpi=150)
        print(f"wrote {args.out}")

    if "maxwell" in sols:
        # the relaxation trajectory: elastic phase, configurational
        # cascade / RT growth, escalation endgame
        h = sols["maxwell"].maxwell["history"]
        tt = [e["t_over_tau"] for e in h]
        fig2, (axr, axd) = plt.subplots(1, 2, figsize=(10, 4))
        axr.semilogx(tt, [e["ur_a"] for e in h], ".-", label="maxwell")
        for m, col in (("dahlen", "C1"), ("welded", "C2")):
            if m in sols:
                axr.axhline(sols[m].responses()["ur_a"], ls="--",
                            lw=0.8, color=col, label=m)
        axr.set_xlabel("t / tau"), axr.set_ylabel("ur_a")
        axr.legend(fontsize=8)
        axd.loglog(tt, [e["delta_solid"] for e in h], ".-")
        axd.set_xlabel("t / tau")
        axd.set_ylabel("solid increment (relative)")
        st = sols["maxwell"].maxwell["stag_step"]
        if st is not None:
            for ax in (axr, axd):
                ax.axvline(tt[st - 1], color="k", lw=0.5, ls=":")
        fig2.suptitle(f"maxwell relaxation, {args.model}, l = {args.l}"
                      + ("; dotted: stagnation -> escalation"
                         if st is not None else ""))
        fig2.tight_layout()
        if args.out:
            relax_out = Path(args.out)
            relax_out = relax_out.with_name(relax_out.stem + "_relax"
                                            + relax_out.suffix)
            fig2.savefig(relax_out, dpi=150)
            print(f"wrote {relax_out}")

    if not args.out:
        plt.show()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
