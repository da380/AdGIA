"""Gravitating viscoelastic Love-number histories through the
correspondence principle, with pyslfp as the elastic engine.

For a layered MAXWELL body under a Heaviside load, the Laplace transform
of the response is the elastic response with the s-dependent shear
modulus, divided by s:

    mu_j(s) = mu_j s tau_j / (1 + s tau_j),   tau_j = eta_j / mu_j,

with the bulk modulus and the density untouched (elastic in bulk) and
fluid layers unchanged. Gaver-Stehfest inversion needs the transform at
REAL positive s only, where mu_j(s) is real — so every evaluation is an
ordinary pyslfp elastic solve on a modified model, and the machinery
that benchmarks the elastic solvers benchmarks the viscoelastic ones
too: spherical, layered, self-gravitating, fluid cores included.

Every SOLID layer has its own Maxwell time; `inf` keeps a layer elastic
(an elastic lithosphere then survives the relaxed limit):

    ./laplace_reference fluid_core --tau 1 --lmax 4
    ./laplace_reference homogeneous_lithosphere --tau 1 inf
    ./laplace_reference homogeneous --tau 1 --lmax 2 --times 0.1 1 10

`--tau` takes one value per solid layer, centre outward (one value for
all of them); times are in the same unit. The output JSON holds, per
time and degree, the load and tidal Love numbers on their way from the
elastic (t -> 0) limit, with the limits computed directly as elastic
solves for a cross-check, and the sanity checks below.

THE INSTABILITY HORIZON. A compressible layer of UNIFORM density is
convectively unstable (N^2 = -rho g^2 / kappa < 0), and once its shear
relaxes the buoyancy is unopposed: the transform then has poles at
real POSITIVE s, the growth rates of Rayleigh-Taylor-like modes, and
the "relaxed limit" of such a layer is not an equilibrium at all. Every
uniform-layer model here has such layers. Stehfest's samples
s_k = k ln2 / t are honest Laplace integrals (of the true, growing,
history) only while ln2 / t exceeds the largest pole, so the script
scans the real axis for poles (--scan, then --fine-scan where the
horizon is decided: close pairs of poles hide from a coarse grid),
reports them, and flags every
output time beyond the horizon t_h = ln2 / s_pole as invalid. The
direct relaxed solve (mu = 0 in the Maxwell layers) is pyslfp's static
fluid treatment, which is neutral by construction: it is NOT the
t -> infinity limit of a body with poles, and is kept as a label only.

Sanity checks (stored under "sanity", printed):
  t -> 0     the inversion at t0 = 1e-6 min(tau) against the elastic
             numbers (Stehfest reproduces constants exactly, and the
             physical drift by t0 is ~1e-6, so this measures the
             inversion's accuracy);
  monotone   h' and k' (and h, k) histories monotone in time over the
             valid times (wiggles below 10x the t -> 0 error, or 1e-5
             of the largest number, forgiven); where a static-fluid
             relaxed value exists, the fraction of the way to it
             covered by the last valid time, and whether the history
             heads towards it, overshoots it, or moves away
             (informational: see the horizon above, and the
             compressible-layer caveat in README.md).

Restricted to models whose layers have CONSTANT moduli (the velocities
are rebuilt from the modified moduli, which needs no refitting there).
The Stehfest order trades accuracy against cancellation; n = 12 in
double precision gives ~4-5 significant figures on smooth transforms,
plenty below the 3-D discretisation error it referees.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent.parent / "common"))

import models  # noqa: E402
from planetmodel import LayeredIsotropicElastic  # noqa: E402

#: The uniform-layer models this reference supports.
SUPPORTED = ("homogeneous", "homogeneous_lithosphere", "two_solid",
             "fluid_core", "inner_core")

#: The Love numbers carried.
QUANTITIES = ("h_load", "l_load", "k_load", "h_tide", "l_tide", "k_tide")

#: The quantities whose histories must be monotone.
MONOTONE = ("h_load", "k_load", "h_tide", "k_tide")


def stehfest_weights(n: int) -> list[float]:
    """The Gaver-Stehfest weights V_k, k = 1..n (n even)."""
    V = []
    for k in range(1, n + 1):
        s = 0.0
        for j in range((k + 1) // 2, min(k, n // 2) + 1):
            s += (j ** (n // 2) * math.factorial(2 * j) /
                  (math.factorial(n // 2 - j) * math.factorial(j) *
                   math.factorial(j - 1) * math.factorial(k - j) *
                   math.factorial(2 * j - k)))
        V.append((-1) ** (k + n // 2) * s)
    return V


def layer_moduli(name: str) -> list[dict]:
    """Per layer of the SI model: name, rho, kappa, mu (uniform)."""
    base = models.model(name)
    sk = base.skeleton
    out = []
    for i, layer in enumerate(base.layers):
        r0 = float(sk.boundaries[i])
        f = layer.fields
        d = float(f["rho"](r0 + 1.0))
        p_wave = float(f["vp"](r0 + 1.0))
        s_wave = float(f["vs"](r0 + 1.0))
        mu = d * s_wave**2
        out.append({"name": layer.name, "rho": d, "mu": mu,
                    "kappa": d * (p_wave**2 - 4.0 * s_wave**2 / 3.0),
                    "fluid": mu == 0.0})
    return out


def layer_taus(name: str, taus: list[float]) -> list[float | None]:
    """The Maxwell time of every layer of the model, centre outward:
    `taus` per SOLID layer (one value broadcasts), None for a fluid
    layer, inf for an elastic one."""
    layers = layer_moduli(name)
    solids = [i for i, la in enumerate(layers) if not la["fluid"]]
    if len(taus) == 1:
        taus = taus * len(solids)
    if len(taus) != len(solids):
        raise SystemExit(f"{name} has {len(solids)} solid layers "
                         f"({', '.join(layers[i]['name'] for i in solids)}"
                         f"); --tau gave {len(taus)} values")
    out: list[float | None] = [None] * len(layers)
    for i, t in zip(solids, taus):
        if not t > 0.0:
            raise SystemExit("--tau values must be positive (inf: elastic)")
        out[i] = t
    return out


def maxwell_model(name: str, taus: list[float | None], s: float | None):
    """The model with mu -> mu s tau / (1 + s tau) in every layer of
    finite tau (s = None: the elastic model; s = 0: relaxed), in the
    benchmark's units with pressure attached; tau and 1/s in the same
    (arbitrary) time unit."""
    base = models.model(name)
    sk = base.skeleton
    rho, vp, vs = [], [], []
    for la, tau in zip(layer_moduli(name), taus):
        mu, kappa, d = la["mu"], la["kappa"], la["rho"]
        if s is not None and tau is not None and math.isfinite(tau):
            mu = mu * (s * tau) / (1.0 + s * tau) if s > 0.0 else 0.0
        vs.append(math.sqrt(mu / d))
        vp.append(math.sqrt((kappa + 4.0 * mu / 3.0) / d))
        rho.append(d)
    rebuilt = LayeredIsotropicElastic(
        [float(b) for b in sk.boundaries], rho=rho, vp=vp, vs=vs,
        layer_names=[la.name for la in base.layers],
        interface_names=list(base.skeleton.interface_names)
        if hasattr(base.skeleton, "interface_names") else None,
        name=base.name)
    return models.with_pressure(models.scaled(rebuilt))


def love(model, lmax: int) -> dict:
    import make_case
    r = make_case.reference(model, lmax, per_layer=3)
    return {q: r[q] for q in ("degree",) + QUANTITIES}


class Transform:
    """R(s): the Love numbers of the s-modified model, cached."""

    def __init__(self, name: str, taus: list[float | None], lmax: int):
        self.name, self.taus, self.lmax = name, taus, lmax
        self.cache: dict[float, dict] = {}

    def __call__(self, s: float) -> dict:
        if s not in self.cache:
            self.cache[s] = love(maxwell_model(self.name, self.taus, s),
                                 self.lmax)
        return self.cache[s]


def find_poles(R: Transform, s_min: float, s_max: float,
               per_decade: int, degrees: list[int]) -> list[dict]:
    """Real positive poles of the transform: brackets of the log grid
    across which h' AND l' (degrees >= 1) both change sign through
    large values — a pole is common to every Love number of its degree,
    a zero is not — refined by bisection on h'. Returns, per pole, the
    degree, the growth rate s and the e-folding time 1/s."""
    n = max(2, int(round(per_decade * math.log10(s_max / s_min))))
    grid = [s_min * (s_max / s_min) ** (i / n) for i in range(n + 1)]
    values = [R(s) for s in grid]
    poles = []
    for i, l in enumerate(degrees):
        for a in range(n):
            ha, hb = values[a]["h_load"][i], values[a + 1]["h_load"][i]
            if ha is None or hb is None or ha * hb >= 0.0:
                continue
            if l >= 1:
                la, lb = values[a]["l_load"][i], values[a + 1]["l_load"][i]
                if la is None or lb is None or la * lb >= 0.0:
                    continue
            lo, hi = grid[a], grid[a + 1]
            sign_lo = math.copysign(1.0, ha)
            for _ in range(30):
                mid = math.sqrt(lo * hi)
                hm = R(mid)["h_load"][i]
                if math.copysign(1.0, hm) == sign_lo:
                    lo = mid
                else:
                    hi = mid
            # A pole grows without bound at the bracket; a zero does not.
            peak = max(abs(R(lo)["h_load"][i]), abs(R(hi)["h_load"][i]))
            scale = max(abs(values[0]["h_load"][i]),
                        abs(values[-1]["h_load"][i]))
            if peak > 20.0 * scale:
                s = math.sqrt(lo * hi)
                poles.append({"degree": l, "s": s, "efold_time": 1.0 / s})
    return poles


def stehfest(R: Transform, t: float, V: list[float]) -> dict:
    """f(t) ~ ln2/t sum_k V_k F(k ln2/t), F(s) = R(s)/s for the
    Heaviside load: the two factors of s cancel to sum_k V_k R(s_k)/k."""
    ln2 = math.log(2.0)
    n_deg = len(R(ln2 / t)["degree"])
    acc = {q: [0.0] * n_deg for q in QUANTITIES}
    for k in range(1, len(V) + 1):
        r = R(k * ln2 / t)
        for q in QUANTITIES:
            for i in range(n_deg):
                v = r[q][i]
                if v is None:
                    acc[q][i] = None
                elif acc[q][i] is not None:
                    acc[q][i] += V[k - 1] * v / k
    return acc


def monotone(series: list[float], start: float,
             tol: float) -> tuple[bool, str]:
    """Is the history monotone from `start`? Wiggles below `tol` (the
    inversion's accuracy) are forgiven."""
    full = [start] + series
    steps = [b - a for a, b in zip(full, full[1:])]
    drift = full[-1] - full[0]
    if abs(drift) <= tol:
        return True, "flat"
    sign = math.copysign(1.0, drift)
    ok = all(sign * d >= -tol for d in steps)
    return ok, "monotone" if ok else "NOT monotone"


def direction(last: float, start: float, relaxed: float | None,
              tol: float) -> tuple[float | None, str]:
    """Where the last valid value sits on the way from `start` to the
    static-fluid `relaxed` value: the fraction covered and a verdict
    (informational: that value is not the t -> inf limit of a body
    whose transform has positive poles, nor of a compressible Maxwell
    layer that is not neutrally stratified)."""
    if relaxed is None or abs(relaxed - start) <= tol:
        return None, "no relaxed value"
    frac = (last - start) / (relaxed - start)
    if frac < -tol:
        return frac, "AWAY from the static-fluid value"
    if frac > 1.0 + 1e-3:
        return frac, "OVERSHOOTS the static-fluid value"
    return frac, "towards the static-fluid value"


def main() -> None:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("model", choices=SUPPORTED)
    p.add_argument("--tau", type=float, nargs="+", default=[1.0],
                   help="Maxwell time of each solid layer, centre "
                        "outward ('inf': elastic); one value for all")
    p.add_argument("--times", type=float, nargs="+", default=None,
                   help="times after the Heaviside load, same unit "
                        "(default: 13 log-spaced from 0.011 to 11.3 of "
                        "the smallest finite tau)")
    p.add_argument("--lmax", type=int, default=4)
    p.add_argument("--stehfest", type=int, default=12,
                   help="Stehfest order (even)")
    p.add_argument("--scan", type=float, nargs=2, default=[1e-4, 1e2],
                   metavar=("S_MIN", "S_MAX"),
                   help="range of the pole scan, in units of 1/min(tau)")
    p.add_argument("--scan-density", type=int, default=24,
                   help="scan points per decade (0: no scan)")
    p.add_argument("--fine-scan", type=float, nargs=3,
                   default=[1e-3, 1.0, 120.0],
                   metavar=("S_MIN", "S_MAX", "DENSITY"),
                   help="a second, dense scan where the horizon is "
                        "decided (units of 1/min(tau); DENSITY 0: none): "
                        "the coarse grid misses close PAIRS of poles")
    p.add_argument("--out", type=Path, default=None,
                   help="results file (default: laplace_<model>_tau<taus>"
                        ".json)")
    args = p.parse_args()
    if args.stehfest % 2:
        raise SystemExit("--stehfest must be even")
    V = stehfest_weights(args.stehfest)
    ln2 = math.log(2.0)

    taus = layer_taus(args.model, args.tau)
    finite = [t for t in taus if t is not None and math.isfinite(t)]
    if not finite:
        raise SystemExit("every solid layer is elastic: nothing relaxes")
    tau_min = min(finite)
    times = sorted(args.times) if args.times else \
        [round(0.011 * tau_min * 2.0 ** (k * 10 / 12), 6)
         for k in range(13)]
    R = Transform(args.model, taus, args.lmax)
    layers = layer_moduli(args.model)

    print(f"{args.model}: layers " + ", ".join(
        f"{la['name']} ({'fluid' if t is None else f'tau {t:g}'})"
        for la, t in zip(layers, taus)), flush=True)
    elastic = love(maxwell_model(args.model, taus, None), args.lmax)
    degrees = elastic["degree"]
    if all(t is None or math.isfinite(t) for t in taus):
        # With no elastic layer the body relaxes to a fluid ball (fluid
        # layers included), which pyslfp (rightly) refuses: no solid
        # surface to hang the load on. The degenerate limit; an elastic
        # layer (--tau ... inf) removes it.
        relaxed = {q: [None] * len(degrees) for q in QUANTITIES}
        relaxed_note = "unavailable: the whole body relaxes to a fluid"
    else:
        try:
            relaxed = love(maxwell_model(args.model, taus, 0.0), args.lmax)
            relaxed = {q: relaxed[q] for q in QUANTITIES}
            relaxed_note = ("pyslfp static fluid in the Maxwell layers "
                            "(neutral; the t -> inf limit only if the "
                            "transform has no positive poles)")
        except Exception as e:  # noqa: BLE001
            relaxed = {q: [None] * len(degrees) for q in QUANTITIES}
            relaxed_note = f"unavailable ({type(e).__name__}: {e})"
    print(f"  relaxed limit: {relaxed_note}")

    # Poles on the positive real axis: the instability horizon.
    poles: list[dict] = []
    if args.scan_density > 0:
        print("  scanning the real axis for poles...", flush=True)
        poles = find_poles(R, args.scan[0] / tau_min,
                           args.scan[1] / tau_min, args.scan_density,
                           degrees)
        lo, hi, dens = args.fine_scan
        if dens > 0:
            for q in find_poles(R, lo / tau_min, hi / tau_min, int(dens),
                                degrees):
                if not any(o["degree"] == q["degree"] and
                           abs(o["s"] - q["s"]) < 1e-2 * q["s"]
                           for o in poles):
                    poles.append(q)
    s_pole = max((q["s"] for q in poles), default=0.0)
    horizon = ln2 / s_pole if s_pole > 0.0 else math.inf
    for q in sorted(poles, key=lambda q: (q["degree"], -q["s"])):
        print(f"    pole: degree {q['degree']}, s = {q['s']:.4g} "
              f"(e-folding {q['efold_time']:.4g})")
    print(f"  Stehfest horizon ln2 / s_pole = {horizon:.4g}"
          + ("" if poles else " (no poles found)"))

    histories = []
    for t in times:
        acc = stehfest(R, t, V)
        histories.append({"time": t, "valid": t < horizon, **acc})
        print(f"  t = {t:g}{'' if t < horizon else '  (beyond horizon)'}",
              flush=True)

    # Sanity: t -> 0 against the elastic numbers, and monotonicity over
    # the valid times.
    t0 = 1e-6 * tau_min
    early = stehfest(R, t0, V)
    zero_err = {}
    for q in QUANTITIES:
        errs = [abs(a - b) for a, b in zip(early[q], elastic[q])
                if a is not None and b is not None]
        zero_err[q] = max(errs, default=None)
    scale = max(abs(v) for q in QUANTITIES for v in elastic[q]
                if v is not None)
    tol = max(10.0 * max(v for v in zero_err.values() if v is not None),
              1e-5 * scale)
    valid = [h for h in histories if h["valid"]]
    checks = []
    for q in MONOTONE:
        for i, l in enumerate(degrees):
            if elastic[q][i] is None or (q.endswith("tide") and l < 2):
                continue
            if q == "k_load" and l == 1:
                continue  # -1 by the frame, at every time
            series = [h[q][i] for h in valid]
            if not series or any(v is None for v in series):
                continue
            ok, why = monotone(series, elastic[q][i], tol)
            frac, where = direction(series[-1], elastic[q][i],
                                    relaxed[q][i], tol)
            checks.append({"quantity": q, "degree": l, "ok": ok,
                           "verdict": why, "direction": where,
                           "fraction_relaxed_at_last_valid": frac})
    print(f"  t -> 0 (t0 = {t0:g}): max |f(t0) - elastic| " + ", ".join(
        f"{q} {v:.2e}" for q, v in zero_err.items() if v is not None))
    bad = [c for c in checks if not c["ok"]]
    print(f"  monotone: {len(checks) - len(bad)} of {len(checks)} "
          "histories pass (valid times, tolerance "
          f"{tol:.1e})")
    for c in checks:
        if not c["ok"] or c["direction"] not in ("no relaxed value",
                                                 "towards the static-"
                                                 "fluid value"):
            frac = c["fraction_relaxed_at_last_valid"]
            print(f"    {c['quantity']} l={c['degree']}: {c['verdict']}, "
                  f"{c['direction']}"
                  + ("" if frac is None else f" (fraction {frac:.3f})"))

    tag = "_".join("f" if t is None else ("inf" if not math.isfinite(t)
                                          else f"{t:g}") for t in taus)
    out = args.out or Path(f"laplace_{args.model}_tau{tag}.json")

    def js(x):
        return None if x is None or not math.isfinite(x) else x

    out.write_text(json.dumps({
        "model": args.model,
        "layers": [{"name": la["name"], "fluid": la["fluid"],
                    "tau": None if t is None else
                    ("inf" if not math.isfinite(t) else t)}
                   for la, t in zip(layers, taus)],
        # the shortest Maxwell time, for readers that take a single tau
        # (the per-layer values are under "layers")
        "tau": tau_min, "lmax": args.lmax, "stehfest": args.stehfest,
        "degree": degrees, "times": times,
        "elastic": {q: elastic[q] for q in QUANTITIES},
        "relaxed": relaxed, "relaxed_note": relaxed_note,
        "poles": poles, "horizon": js(horizon),
        "histories": histories,
        "sanity": {"t0": t0, "t0_error": zero_err, "tolerance": tol,
                   "monotone": checks}}, indent=1))
    print(f"wrote {out}")

    if 2 in degrees:
        i = degrees.index(2)
        e, r0 = elastic["h_load"][i], relaxed["h_load"][i]
        print(f"h'_2: elastic {e:.5f} .. relaxed "
              f"{'n/a' if r0 is None else f'{r0:.5f}'}; " + ", ".join(
                  f"t = {h['time']:g}: {h['h_load'][i]:.5f}"
                  for h in histories))


if __name__ == "__main__":
    main()
