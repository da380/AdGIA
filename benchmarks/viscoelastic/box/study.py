"""The box studies: sweeps of viscoelastic_box against exact references.

  steppers   the homogeneous box (FE exact in space, so every error is the
             integrator's): rheologies maxwell / sls / burgers /
             prony_wide x loads (stress, strain control) x every history x
             every scheme x a ladder of steps (and adaptive tolerances),
             with the breakpoints aligned and not. Figures: error against
             dt (the orders), error against elastic solves (the cost),
             aligned against unaligned (the order lost to a jump or kink),
             and the periodic transient.
  slab       Fourier-mode slabs (2-D, and a 3-D subset): models x modes x
             mesh ladder x element order, time error kept negligible: the
             spatial convergence of the relaxation, and the staircase
             penalty of layer interfaces that cut elements (composite vs
             pointwise rheology).
  stehfest   the Gaver-Stehfest inversion (the method behind the
             gravitating Love-number references) against the exact modal
             histories of the slabs: W (vertical) and U (horizontal) at
             orders 8..18, and its amplification of noise in the
             transform values.
  contrast   the viscosity-contrast ladder: the 'contrast' model as a
             column (mode 0) and a slab (mode 1), contrast 1..1e8, every
             scheme on a step ladder, plus the graded column: accuracy,
             solves AND Krylov iterations per solve (the conditioning of
             the effective operator where dt >> tau).

    ./study steppers [--quick] [--jobs 10]
    ./study slab                      # the local profile
    ./study all --profile server      # + the long 3-D runs, more ranks
    ./study steppers --figures-only

Results: <out>/<study>/summary.json (one record per run) and figures
beside it; per-run cases, references and results under runs/.
"""
from __future__ import annotations

import argparse
import concurrent.futures as cf
import json
import math
import shlex
import subprocess
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent.parent / "common"))
sys.path.insert(0, str(HERE))

import cases  # noqa: E402
import compare  # noqa: E402
import histories as hist  # noqa: E402
from outputs import outside_source  # noqa: E402
import reference as refmod  # noqa: E402

SCHEMES = ("etd1", "be", "exptrap", "sdirk23", "rk4", "adaptive")
COLOURS = {"rk4": "#e34948", "etd1": "#e87ba4", "be": "#eda100",
           "sdirk23": "#2a78d6", "exptrap": "#1baf7a",
           "adaptive": "#4a3aa7"}
MARKERS = {"rk4": "v", "etd1": "P", "be": "D", "sdirk23": "o",
           "exptrap": "^", "adaptive": "s"}
LABELS = {"rk4": "RK4", "etd1": "ETD1", "be": "BE", "sdirk23": "SDIRK23",
          "exptrap": "ExpTrap", "adaptive": "Adaptive"}


def find_program(given: Path | None) -> Path:
    if given is not None:
        p = given / "viscoelastic_box" if given.is_dir() else given
        return p.resolve()
    for build in sorted(HERE.parent.parent.parent.glob("build*")):
        c = build / "benchmarks" / "bin" / "viscoelastic_box"
        if c.exists():
            return c
    raise SystemExit("viscoelastic_box not found; pass --programs")


class Runner:
    def __init__(self, args):
        self.program = find_program(args.programs)
        self.mpiexec = args.mpiexec
        self.out = args.out
        self.jobs = args.jobs
        self.dry = args.dry_run

    def reference(self, case: dict, stehfest=()) -> Path:
        d = self.out / "runs" / case["name"]
        d.mkdir(parents=True, exist_ok=True)
        cpath, rpath = d / "case.json", d / "reference.json"
        text = json.dumps(case, indent=1)
        if not cpath.exists() or cpath.read_text() != text:
            # A changed case invalidates its reference and every result.
            cpath.write_text(text)
            for stale in [rpath, *d.glob("res_*.json"), *d.glob("log_*")]:
                if stale.exists():
                    stale.unlink()
        if not rpath.exists():
            ref = refmod.reference(refmod.Case(cpath), stehfest)
            rpath.write_text(json.dumps(ref, indent=1))
        return cpath

    def run(self, case: dict, opts: dict, np_: int = 1) -> dict:
        cpath = self.reference(case)
        tag = "_".join(f"{k}{v}" for k, v in opts.items())
        res = cpath.parent / f"res_{tag}.json"
        cmd = [self.mpiexec, "-np", str(np_), str(self.program),
               "-c", str(cpath), "-out", str(res)]
        for k, v in opts.items():
            if isinstance(v, bool):
                cmd.append(f"-{k}" if v else f"-no-{k}")
            else:
                cmd += [f"-{k}", str(v)]
        rec = {"case": case["name"], **opts, "np": np_, "ok": False}
        if self.dry:
            print("$ " + shlex.join(cmd))
            return rec
        result = None
        if res.exists():
            try:
                result = json.loads(res.read_text())
            except json.JSONDecodeError:
                # a run killed mid-write leaves a truncated file: remake
                # it rather than crash the study on it
                print(f"  {res}: unreadable (interrupted run?), remade",
                      flush=True)
                res.unlink()
        if result is None:
            p = subprocess.run(cmd, capture_output=True, text=True)
            if p.returncode != 0 or not res.exists():
                (cpath.parent / f"log_{tag}.txt").write_text(
                    p.stdout + p.stderr)
                return rec
            result = json.loads(res.read_text())
        ref = json.loads((cpath.parent / "reference.json").read_text())
        m = compare.metrics(result, ref)
        if not ref.get("reliable", True):
            rec["unreliable_reference"] = True
        c = result["cost"]
        solves = max(c["stepping_solves"] + c["observation_solves"], 1)
        # The cost is every solve the run made. Stepping and observation
        # solves are not separable: ETD1 reuses an observation solve as
        # the start of its next step, and SDIRK23 pays an extra one per
        # output (its last stage is not the new state).
        rec.update(ok=c["converged"], error=compare.worst(m),
                   errors=m,
                   solves=c["stepping_solves"] + c["observation_solves"],
                   stepping_solves=c["stepping_solves"],
                   observation_solves=c["observation_solves"],
                   assemblies=c["assemblies"],
                   setups=c["preconditioner_setups"],
                   iterations=c["iterations"],
                   its_per_solve=c["iterations"] / solves,
                   steps=c["steps"], rejected=c["rejected_steps"],
                   seconds=c["seconds"], dofs=result["dofs"])
        if not all(math.isfinite(v["history"]) for v in m.values()):
            rec["ok"] = False
        return rec

    def sweep(self, jobs: list[tuple[dict, dict, int]]) -> list[dict]:
        # Reference first (in this process, once per case), then the runs.
        seen = set()
        for case, _, _ in jobs:
            if case["name"] not in seen:
                seen.add(case["name"])
                self.reference(case)
        out = []
        width = max(1, self.jobs // max(j[2] for j in jobs))
        with cf.ThreadPoolExecutor(width) as ex:
            futures = [ex.submit(self.run, *j) for j in jobs]
            for n, f in enumerate(cf.as_completed(futures), 1):
                out.append(f.result())
                if n % 50 == 0 or n == len(futures):
                    print(f"  {n}/{len(futures)} runs", flush=True)
        return out


def save(records: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(records, indent=1))
    print(f"wrote {path} ({len(records)} runs)")


# --- the studies ----------------------------------------------------------------

# Profiles, as campaign.py's: "local" fits a laptop (the longest run a few
# minutes on 4 ranks), "server" adds the long 3-D runs (p = 2 at nz = 10,
# ~106k unknowns, ~1.5 h on 4 ranks; p = 1 at nz = 20) and more ranks.
PROFILES = {
    "local": dict(np=2, jobs=10,
                  slab_3d=[(1, 5), (1, 10), (2, 5)]),
    "server": dict(np=8, jobs=96,
                   slab_3d=[(1, 5), (1, 10), (1, 20), (2, 5), (2, 10)]),
}

# The stepper ladder: at most the output spacing (0.5), down to 1/1024.
DTS = [2.0 ** -j for j in range(1, 11)]
RTOLS = [10.0 ** -j for j in range(1, 8)]          # 1e-1 .. 1e-7


def stepper_cases(quick: bool) -> list[dict]:
    models = ("maxwell", "burgers") if quick else \
        ("maxwell", "sls", "burgers", "prony_wide")
    names = ("heaviside", "ramp", "load_unload", "periodic") if quick else \
        tuple(hist.PRESETS)
    out = []
    for model in models:
        for load in ("uniaxial_stress", "uniaxial_strain"):
            for h in names:
                out.append(cases.make_case(
                    model, dim=2, load=load, history=hist.make(h),
                    t_final=8.0, n_times=16))
    return out


def study_steppers(R: Runner, quick: bool) -> list[dict]:
    jobs = []
    for case in stepper_cases(quick):
        has_bp = bool(case["load"]["history"]["breakpoints"])
        for align in ((True, False) if has_bp else (True,)):
            for scheme in SCHEMES:
                if scheme == "adaptive":
                    for rtol in RTOLS:
                        jobs.append((case, {"scheme": scheme, "rtol": rtol,
                                            "atol": 1e-12, "dt": 0.25,
                                            "align": align,
                                            "nx": 1, "nz": 1}, 1))
                else:
                    for dt in DTS:
                        jobs.append((case, {"scheme": scheme, "dt": dt,
                                            "align": align,
                                            "nx": 1, "nz": 1}, 1))
    recs = R.sweep(jobs)
    meta = {c["name"]: c for c, _, _ in jobs}
    for r in recs:
        c = meta[r["case"]]
        r.update(model=c["model"], load=c["load"]["kind"],
                 history=c["load"]["history"]["name"])
    return recs


def slab_cases(quick: bool, profile: dict) -> list[tuple[dict, list]]:
    """(case, meshes) with meshes [(order, nz, nx, rheology)]."""
    out = []
    models = ("lid", "channel") if quick else \
        ("lid", "channel", "sls_stack", "burgers_lid", "prony_lid",
         "maxwell_slab")
    ladder = [(o, nz) for o in (1, 2) for nz in (5, 10, 20, 40)
              if not (o == 2 and nz == 40)]
    for model in models:
        for mode in ((1, 4) if quick else (1, 2, 4)):
            c = cases.make_case(model, dim=2, mode=(mode,),
                                t_final=8.0, n_times=16)
            out.append((c, [(o, nz, 2 * nz, "composite")
                            for o, nz in ladder]))
    # The deliberate misfit (-no-conform; every other run builds the mesh
    # around the layer interfaces): uniform nz = 8, 16, 32 against the lid
    # at 0.8 (and the channel's 0.6/0.8), staircased (composite) or
    # resolved at the points (pointwise).
    for model in ("lid", "channel"):
        c = cases.make_case(model, dim=2, mode=(1,), t_final=8.0,
                            n_times=16)
        out.append((c, [(2, nz, 2 * nz, rh) for nz in (8, 16, 32)
                        for rh in ("composite", "pointwise")]))
    # A periodic load with its initial transient, and the 3-D subset.
    c = cases.make_case("lid", dim=2, mode=(1,),
                        history=hist.periodic(omega=2.0), t_final=8.0,
                        n_times=32)
    out.append((c, [(2, nz, 2 * nz, "composite") for nz in (5, 10, 20)]))
    for mode in ((1, 0), (1, 1)):
        c = cases.make_case("lid", dim=3, mode=mode, t_final=8.0, n_times=8)
        out.append((c, [(o, nz, 2 * nz, "composite")
                        for o, nz in profile["slab_3d"]]))
    return out


def study_slab(R: Runner, quick: bool, np_: int,
               profile: dict) -> list[dict]:
    jobs = []
    for case, meshes in slab_cases(quick, profile):
        for o, nz, nx, rh in meshes:
            opts = {"scheme": "exptrap", "dt": 0.025, "o": o, "nz": nz,
                    "nx": nx}
            if nz in (8, 16, 32):  # the interface study: uniform spacing
                opts["conform"] = False
            opts["rheology"] = rh
            jobs.append((case, opts, np_ * (2 if case["dim"] == 3 else 1)))
    recs = R.sweep(jobs)
    meta = {c["name"]: c for c, _, _ in jobs}
    for r in recs:
        c = meta[r["case"]]
        r.update(model=c["model"], dim=c["dim"], mode=c["load"]["mode"],
                 history=c["load"]["history"]["name"])
    return recs


def study_stehfest(R: Runner, quick: bool) -> list[dict]:
    recs = []
    orders = (8, 10, 12, 14, 16, 18)
    models = ("lid", "channel", "burgers_lid", "prony_lid", "maxwell_slab",
              "sls_stack")
    for model in models[:2] if quick else models:
        for mode in (1, 2, 4):
            c = cases.make_case(model, dim=2, mode=(mode,), t_final=8.0,
                                n_times=40)
            case = refmod.Case(R.reference(c))
            t = np.asarray(case.times)
            mW, mU = refmod.slab_modes(case, 0), refmod.slab_modes(case, 1)
            step = hist.heaviside()
            exact = {"W": refmod.modal_response(mW, step, 1.0, t),
                     "U": refmod.modal_response(mU, step, 1.0, t)}
            for q, which in (("W", 0), ("U", 1)):
                F = (lambda s, w=which:
                     refmod.slab_transfer(case, s)[w] / s)
                tal = np.array([refmod.talbot(F, tj) for tj in t])
                scale = np.abs(exact[q]).max()
                rec = {"model": model, "mode": mode, "quantity": q,
                       "times": t.tolist(), "exact": exact[q].tolist(),
                       "talbot": float(np.abs(tal - exact[q]).max()
                                       / scale),
                       "sign_change": bool(np.any(np.diff(
                           np.sign(exact[q])) != 0))}
                # slab_reference's certification of the modal curve
                # (Talbot agreement < 1e-7) applies here too: without it
                # a bad AAA fit would pass as the "exact" curve
                rec["exact_certified"] = rec["talbot"] < 1e-7
                if not rec["exact_certified"]:
                    print(f"  WARNING: {model} n={mode} {q}: Talbot "
                          f"disagrees with the modal curve at "
                          f"{rec['talbot']:.1e}; the Stehfest errors are "
                          "against an uncertified curve", flush=True)
                for n in orders:
                    st = np.array([refmod.stehfest(F, tj, n) for tj in t])
                    rec[f"stehfest{n}"] = float(np.abs(st - exact[q]).max()
                                                / scale)
                    rec[f"stehfest{n}_series"] = st.tolist()
                recs.append(rec)
                print(f"  {model:13s} n={mode} {q}: talbot "
                      f"{rec['talbot']:.1e}, stehfest12 "
                      f"{rec['stehfest12']:.1e}, 16 {rec['stehfest16']:.1e}")
    # Noise amplification: the transform values of a numerical reference
    # (pyslfp's elastic solves) carry errors; Stehfest multiplies them by
    # about sum_k |V_k| / k. Random relative noise eps on the exact
    # transform of the channel model's U at n = 4 (the smallest amplitude).
    c = cases.make_case("channel", dim=2, mode=(4,), t_final=8.0,
                        n_times=40)
    case = refmod.Case(R.reference(c))
    t = np.asarray(case.times)
    exact = refmod.modal_response(refmod.slab_modes(case, 1),
                                  hist.heaviside(), 1.0, t)
    rng = np.random.default_rng(1)
    for n in (8, 10, 12, 14, 16):
        V = refmod.stehfest_weights(n)
        gain = sum(abs(v) / k for k, v in enumerate(V, 1))
        for eps in (1e-12, 1e-10, 1e-8, 1e-6):
            F = (lambda s, e=eps:
                 refmod.slab_transfer(case, s)[1] / s
                 * (1 + e * rng.standard_normal()))
            st = np.array([refmod.stehfest(F, tj, n) for tj in t])
            err = float(np.abs(st - exact).max() / np.abs(exact).max())
            recs.append({"noise": eps, "order": n, "gain": gain,
                         "model": "channel", "mode": 4, "quantity": "U",
                         "error": err})
            print(f"  noise {eps:.0e}, order {n}: U error {err:.1e}")
    return recs


CONTRASTS = (1.0, 1e2, 1e4, 1e6, 1e8)


def study_contrast(R: Runner, quick: bool, np_: int) -> list[dict]:
    jobs = []
    contrasts = CONTRASTS[::2] if quick else CONTRASTS
    dts = [2.0 ** -j for j in range(0, 8)]
    for contrast in contrasts:
        for mode, nx in ((0, 3), (1, 20)):
            c = cases.make_case("contrast", dim=2, mode=(mode,),
                                contrast=contrast, t_final=8.0, n_times=16,
                                name=f"contrast{contrast:g}_n{mode}")
            for scheme in SCHEMES:
                if scheme == "adaptive":
                    # Not tighter than 1e-4: the conservative estimate makes
                    # tight tolerances cost tens of thousands of solves.
                    for rtol in RTOLS[:4]:
                        jobs.append((c, {"scheme": scheme, "rtol": rtol,
                                         "atol": 1e-12, "dt": 0.25,
                                         "o": 2, "nz": 10, "nx": nx}, np_))
                elif scheme == "rk4" and contrast > 1e2:
                    continue  # stability-bound: dt < 2.8/contrast
                else:
                    for dt in dts:
                        jobs.append((c, {"scheme": scheme, "dt": dt,
                                         "o": 2, "nz": 10, "nx": nx}, np_))
    for contrast in contrasts[1:]:
        c = cases.make_case("gradient", dim=2, mode=(0,), contrast=contrast,
                            t_final=8.0, n_times=16,
                            name=f"gradient{contrast:g}")
        for scheme in ("exptrap", "sdirk23", "be"):
            for dt in dts:
                jobs.append((c, {"scheme": scheme, "dt": dt, "o": 2,
                                 "nz": 20, "nx": 3}, np_))
    recs = R.sweep(jobs)
    meta = {c["name"]: c for c, _, _ in jobs}
    for r in recs:
        c = meta[r["case"]]
        r.update(contrast=c["contrast"], mode=c["load"]["mode"][0],
                 model=c["model"])
    return recs


# --- figures ---------------------------------------------------------------------


def _plt():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size": 11, "axes.grid": True,
                         "grid.color": "#e4e3df", "lines.linewidth": 1.8,
                         "lines.markersize": 5, "savefig.dpi": 160})
    return plt


def _ok(recs):
    """Runs that finished with a finite error below 10 (an unstable
    explicit run can finish with errors of 1e100 and more)."""
    return [r for r in recs if r.get("ok") and math.isfinite(
        r.get("error", float("nan"))) and r["error"] < 10.0]


def order_table(recs: list[dict]) -> list[dict]:
    """Observed orders: the least-squares slope of log error against log dt
    over the steps whose errors lie in (1e-11, 1e-2), i.e. above the
    round-off floor and in the asymptotic range (nan with fewer than
    three such points; an exact scheme has none)."""
    rows = []
    keys = sorted({(r["model"], r["load"], r["history"], r["align"],
                    r["scheme"]) for r in recs if r["scheme"] != "adaptive"})
    for key in keys:
        pts = sorted((r["dt"], r["error"]) for r in _ok(recs)
                     if (r["model"], r["load"], r["history"], r["align"],
                         r["scheme"]) == key)
        pts = [(d, e) for d, e in pts if 1e-11 < e < 1e-2]
        rate = float("nan")
        if len(pts) >= 3:
            x, y = np.log([d for d, _ in pts]), np.log([e for _, e in pts])
            rate = float(np.polyfit(x, y, 1)[0])
        floor = min((r["error"] for r in _ok(recs)
                     if (r["model"], r["load"], r["history"], r["align"],
                         r["scheme"]) == key), default=float("nan"))
        rows.append(dict(zip(("model", "load", "history", "align",
                              "scheme"), key), order=rate, best=floor))
    return rows


def figures_steppers(recs: list[dict], out: Path) -> None:
    plt = _plt()
    ok = _ok(recs)
    names = list(hist.PRESETS)
    # 1. error vs dt, per model and load, one panel per history
    # (breakpoints aligned).
    for model in ("maxwell", "sls", "burgers", "prony_wide"):
        for load in ("uniaxial_stress", "uniaxial_strain"):
            sel = [r for r in ok if r["model"] == model
                   and r["load"] == load and r["align"]]
            hs = [h for h in names if any(r["history"] == h for r in sel)]
            if not hs:
                continue
            cols = 4
            rows = math.ceil(len(hs) / cols)
            fig, axes = plt.subplots(rows, cols, figsize=(3.6 * cols,
                                                          3.0 * rows),
                                     squeeze=False, sharex=True, sharey=True)
            for ax, h in zip(axes.flat, hs):
                for s in SCHEMES:
                    if s == "adaptive":
                        continue
                    pts = sorted((r["dt"], r["error"]) for r in sel
                                 if r["history"] == h and r["scheme"] == s)
                    if pts:
                        x, y = zip(*pts)
                        ax.loglog(x, np.maximum(y, 1e-16), "-",
                                  marker=MARKERS[s], color=COLOURS[s],
                                  label=LABELS[s])
                ax.set_title(h)
                ax.set_ylim(1e-14, 10)
            for ax in axes.flat[len(hs):]:
                ax.set_visible(False)
            for ax in axes[-1]:
                ax.set_xlabel("dt / tau")
            for ax in axes[:, 0]:
                ax.set_ylabel("history error")
            axes.flat[0].legend(fontsize=8)
            fig.suptitle(f"{model}, {load.replace('_', ' ')}: error "
                         "against step (breakpoints aligned)")
            fig.tight_layout()
            fig.savefig(out / f"steppers_dt_{model}_{load}.png")
            plt.close(fig)
    # 2. error vs solves, per model, for a few histories (stress control).
    for h in ("heaviside", "periodic", "sawtooth", "load_unload"):
        models = [m for m in ("maxwell", "sls", "burgers", "prony_wide")
                  if any(r["model"] == m for r in ok)]
        fig, axes = plt.subplots(1, len(models), figsize=(3.8 * len(models),
                                                          3.3),
                                 squeeze=False, sharey=True)
        for ax, model in zip(axes[0], models):
            sel = [r for r in ok if r["model"] == model and r["history"] == h
                   and r["load"] == "uniaxial_stress" and r["align"]]
            for s in SCHEMES:
                pts = sorted((r["solves"], r["error"]) for r in sel
                             if r["scheme"] == s)
                if pts:
                    x, y = zip(*pts)
                    ax.loglog(x, np.maximum(y, 1e-16), "-",
                              marker=MARKERS[s], color=COLOURS[s],
                              label=LABELS[s])
            ax.set_title(model)
            ax.set_xlabel("elastic solves")
        axes[0, 0].set_ylabel("history error")
        axes[0, 0].set_ylim(1e-15, 10)
        axes[0, 0].legend(fontsize=8)
        fig.suptitle(f"cost against accuracy: {h}, stress control")
        fig.tight_layout()
        fig.savefig(out / f"steppers_cost_{h}.png")
        plt.close(fig)
    # 3. aligned vs unaligned breakpoints.
    hs = [h for h in names if any(not r["align"] and r["history"] == h
                                  for r in ok)]
    if hs:
        fig, axes = plt.subplots(1, len(hs), figsize=(3.6 * len(hs), 3.3),
                                 squeeze=False, sharey=True)
        for ax, h in zip(axes[0], hs):
            for s in ("exptrap", "sdirk23", "be"):
                for align, ls in ((True, "-"), (False, ":")):
                    pts = sorted((r["dt"], r["error"]) for r in ok
                                 if r["history"] == h and r["scheme"] == s
                                 and r["align"] == align
                                 and r["model"] == "burgers"
                                 and r["load"] == "uniaxial_stress")
                    if pts:
                        x, y = zip(*pts)
                        ax.loglog(x, np.maximum(y, 1e-16), ls,
                                  marker=MARKERS[s], color=COLOURS[s],
                                  label=f"{LABELS[s]}"
                                  + ("" if align else " (unaligned)"))
            ax.set_title(h)
            ax.set_xlabel("dt / tau")
        axes[0, 0].set_ylabel("history error")
        axes[0, 0].legend(fontsize=7)
        fig.suptitle("burgers, stress control: breakpoints on the step grid "
                     "(solid) or not (dotted)")
        fig.tight_layout()
        fig.savefig(out / "steppers_alignment.png")
        plt.close(fig)
    # 4. the periodic transient: the standard linear solid under a
    # periodic stress, and the pointwise error over time for a few
    # schemes at dt = 1/8.
    _transient_figure(recs, out, plt)


TRANSIENT_DT = 0.125


def transient_case() -> dict:
    """A standard linear solid under a periodic stress switched on at t = 0,
    densely observed: the transient decays on the retardation time
    tau mu_U / mu_inf = 2 into the periodic state."""
    return cases.make_case("sls", dim=2, load="uniaxial_stress",
                           history=hist.periodic(omega=2.0, mean=0.5),
                           t_final=12.0, n_times=240,
                           name="sls_transient")


def study_transient(R: Runner) -> list[dict]:
    case = transient_case()
    jobs = [(case, {"scheme": s, "dt": TRANSIENT_DT, "align": True,
                    "nx": 1, "nz": 1}, 1)
            for s in ("etd1", "be", "exptrap", "sdirk23", "rk4")]
    recs = R.sweep(jobs)
    for r in recs:
        r.update(model="sls_transient", load="uniaxial_stress",
                 history="periodic_mean")
    return recs


def _transient_figure(recs, out, plt) -> None:
    d = out / "runs" / "sls_transient"
    if not (d / "reference.json").exists():
        return
    ref = json.loads((d / "reference.json").read_text())
    obs = compare.observables(ref)
    t = np.asarray([0.0] + [h["time"] for h in ref["histories"]])
    fig, (a1, a2) = plt.subplots(2, 1, figsize=(8, 6), sharex=True)
    a1.plot(t, obs["e_xx"], "-", color="#0b0b0b", label="exact")
    # The periodic state alone: the late-time solution continued back.
    a1.set_ylabel("e_xx")
    for s in ("etd1", "be", "exptrap", "sdirk23", "rk4"):
        f = d / f"res_scheme{s}_dt{TRANSIENT_DT}_alignTrue_nx1_nz1.json"
        if not f.exists():
            continue
        fe = compare.observables(json.loads(f.read_text()))["e_xx"]
        a1.plot(t[::6], fe[::6], MARKERS[s], color=COLOURS[s], ms=3.5,
                label=LABELS[s])
        a2.semilogy(t, np.maximum(np.abs(fe - obs["e_xx"])
                                  / np.abs(obs["e_xx"]).max(), 1e-17),
                    "-", lw=1.2, color=COLOURS[s], label=LABELS[s])
    a1.legend(fontsize=8, ncol=3)
    a2.set_ylabel("|error| / max")
    a2.set_xlabel("t / tau")
    a2.legend(fontsize=8, ncol=5)
    fig.suptitle("SLS, stress 0.5 + sin(2t + 0.3) from t = 0, dt = 1/8:\n"
                 "the transient (retardation time 2) into the periodic state")
    fig.tight_layout()
    fig.savefig(out / "steppers_transient.png")
    plt.close(fig)


def figures_slab(recs: list[dict], out: Path) -> None:
    plt = _plt()
    ok = _ok(recs)
    sel = [r for r in ok if r["dim"] == 2 and r["rheology"] == "composite"
           and r["history"] == "heaviside" and r["nz"] in (5, 10, 20, 40)]
    models = sorted({r["model"] for r in sel})
    cols = 3
    rows = math.ceil(len(models) / cols)
    fig, axes = plt.subplots(rows, cols, figsize=(4.2 * cols, 3.4 * rows),
                             squeeze=False, sharey=True)
    for ax in axes.flat[len(models):]:
        ax.set_visible(False)
    for ax, model in zip(axes.flat, models):
        for mode, colour in ((1, "#2a78d6"), (2, "#1baf7a"), (4, "#e34948")):
            for o, ls in ((1, ":"), (2, "-")):
                pts = sorted((1.0 / r["nz"], r["error"]) for r in sel
                             if r["model"] == model and r["mode"][0] == mode
                             and r["o"] == o)
                if pts:
                    x, y = zip(*pts)
                    ax.loglog(x, y, ls, marker="o", color=colour,
                              label=f"n={mode}, p={o}")
        ax.set_title(model)
        ax.set_xlabel("h")
        hs = [0.025, 0.05, 0.1, 0.2]
        ax.set_xticks(hs, [f"{h:g}" for h in hs])
        ax.minorticks_off()
    for ax in axes[:, 0]:
        ax.set_ylabel("history error (W, U)")
    axes[0, 0].legend(fontsize=7)
    fig.suptitle("Fourier slabs (2-D, Heaviside): spatial convergence")
    fig.tight_layout()
    fig.savefig(out / "slab_convergence.png")
    plt.close(fig)
    # Staircase vs pointwise.
    st = [r for r in ok if r["nz"] in (8, 16, 32)]
    if st:
        fig, ax = plt.subplots(figsize=(5, 3.6))
        for model, colour in (("lid", "#2a78d6"), ("channel", "#e34948")):
            for rh, ls in (("composite", "-"), ("pointwise", "--")):
                pts = sorted((1.0 / r["nz"], r["error"]) for r in st
                             if r["model"] == model and r["rheology"] == rh)
                if pts:
                    x, y = zip(*pts)
                    ax.loglog(x, y, ls, marker="o", color=colour,
                              label=f"{model}, {rh}")
        # The same models on conforming meshes, for comparison.
        for model, colour in (("lid", "#2a78d6"), ("channel", "#e34948")):
            pts = sorted((1.0 / r["nz"], r["error"]) for r in ok
                         if r["model"] == model and r["dim"] == 2
                         and r["o"] == 2 and r["nz"] in (5, 10, 20)
                         and r["history"] == "heaviside"
                         and r["mode"][0] == 1
                         and r["rheology"] == "composite")
            if pts:
                x, y = zip(*pts)
                ax.loglog(x, y, ":", marker="s", color=colour, alpha=0.7,
                          label=f"{model}, conforming")
        ax.set_xlabel("h")
        ax.set_ylabel("history error")
        ax.legend(fontsize=7)
        ax.set_title("layer interfaces cutting elements (p = 2)")
        fig.tight_layout()
        fig.savefig(out / "slab_interfaces.png")
        plt.close(fig)


def figures_stehfest(recs: list[dict], out: Path) -> None:
    plt = _plt()
    noise = [r for r in recs if "noise" in r]
    recs = [r for r in recs if "noise" not in r]
    if noise:
        fig, ax = plt.subplots(figsize=(5.2, 3.8))
        for n, c in zip((8, 10, 12, 14, 16), ("#e87ba4", "#eda100",
                                               "#e34948", "#1baf7a",
                                               "#2a78d6")):
            pts = sorted((r["noise"], r["error"]) for r in noise
                         if r["order"] == n)
            if pts:
                x, y = zip(*pts)
                ax.loglog(x, y, "-o", color=c, label=f"order {n}")
        ax.set_xlabel("relative noise in the transform")
        ax.set_ylabel("history error of U")
        ax.set_title("Stehfest: amplification of transform errors")
        ax.legend(fontsize=8)
        fig.tight_layout()
        fig.savefig(out / "stehfest_noise.png")
        plt.close(fig)
    orders = sorted(int(k[8:]) for k in recs[0]
                    if k.startswith("stehfest") and k[8:].isdigit())
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.8), sharey=True)
    for ax, q in zip(axes, ("W", "U")):
        for r in recs:
            if r["quantity"] != q:
                continue
            ax.semilogy(orders, [r[f"stehfest{n}"] for n in orders], "-o",
                        label=f"{r['model']} n={r['mode']}")
        ax.set_title(f"{q}: Stehfest error (max over t / max |exact|)")
        ax.set_xlabel("Stehfest order")
    axes[0].set_ylabel("history error")
    axes[1].legend(fontsize=6, ncol=2)
    fig.tight_layout()
    fig.savefig(out / "stehfest_orders.png")
    plt.close(fig)
    # One example history: the U where Stehfest 12 does worst.
    ex = sorted((r for r in recs if r["quantity"] == "U"),
                key=lambda r: -r["stehfest12"])
    if ex:
        r = ex[0]
        fig, (a1, a2) = plt.subplots(2, 1, figsize=(7, 5.5), sharex=True)
        t = np.asarray(r["times"])
        e = np.asarray(r["exact"])
        a1.plot(t, e, "-", color="#0b0b0b", label="exact (modal)")
        for n, c in ((12, "#e34948"), (16, "#2a78d6")):
            s = np.asarray(r[f"stehfest{n}_series"])
            a1.plot(t, s, "o", ms=3.5, color=c, label=f"Stehfest {n}")
            a2.semilogy(t, np.abs(s - e) / np.abs(e).max(), "-", color=c,
                        label=f"Stehfest {n}")
        a1.set_ylabel("U (Heaviside)")
        a1.legend(fontsize=8)
        a2.set_ylabel("|error| / max")
        a2.set_xlabel("t")
        fig.suptitle(f"{r['model']} model, n = {r['mode']}: the horizontal "
                     "amplitude (Stehfest's worst case here)")
        fig.tight_layout()
        fig.savefig(out / "stehfest_example.png")
        plt.close(fig)


def figures_contrast(recs: list[dict], out: Path) -> None:
    plt = _plt()
    ok = _ok(recs)
    for mode in (0, 1):
        sel = [r for r in ok if r["model"] == "contrast" and r["mode"] == mode]
        if not sel:
            continue
        cs = sorted({r["contrast"] for r in sel})
        fig, axes = plt.subplots(2, len(cs), figsize=(3.2 * len(cs), 6),
                                 squeeze=False, sharey="row")
        for j, c in enumerate(cs):
            for s in SCHEMES:
                pts = sorted((r["solves"], r["error"], r["its_per_solve"])
                             for r in sel if r["contrast"] == c
                             and r["scheme"] == s)
                if pts:
                    x, y, it = zip(*pts)
                    axes[0, j].loglog(x, np.maximum(y, 1e-16), "-",
                                      marker=MARKERS[s], color=COLOURS[s],
                                      label=LABELS[s])
                    axes[1, j].semilogx(x, it, "-", marker=MARKERS[s],
                                        color=COLOURS[s])
            axes[0, j].set_title(f"contrast {c:g}")
            axes[1, j].set_xlabel("elastic solves")
        axes[0, 0].set_ylabel("history error")
        axes[1, 0].set_ylabel("CG iterations / solve")
        axes[0, 0].legend(fontsize=7)
        fig.suptitle("viscosity-contrast ladder, "
                     + ("column (n = 0)" if mode == 0 else "slab (n = 1)"))
        fig.tight_layout()
        fig.savefig(out / f"contrast_n{mode}.png")
        plt.close(fig)


# --- main --------------------------------------------------------------------------


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("study", choices=("steppers", "slab", "stehfest",
                                     "contrast", "all"))
    p.add_argument("--out", type=Path, default=Path("."))
    p.add_argument("--programs", type=Path)
    p.add_argument("--mpiexec", default="mpiexec")
    p.add_argument("--profile", choices=sorted(PROFILES), default="local")
    p.add_argument("--jobs", type=int, default=None,
                   help="cores to use at once, runs x ranks (profile)")
    p.add_argument("--np", type=int, default=None,
                   help="ranks per slab/contrast run (profile)")
    p.add_argument("--quick", action="store_true")
    p.add_argument("--figures-only", action="store_true")
    p.add_argument("--dry-run", action="store_true")
    a = p.parse_args()
    profile = PROFILES[a.profile]
    a.jobs = a.jobs or profile["jobs"]
    a.np = a.np or profile["np"]
    a.out = outside_source(a.out)
    studies = (("steppers", "slab", "stehfest", "contrast")
               if a.study == "all" else (a.study,))
    for study in studies:
        d = a.out / study
        d.mkdir(parents=True, exist_ok=True)
        summary = d / "summary.json"
        if not a.figures_only:
            a_out = a.out
            a.out = d
            R = Runner(a)
            a.out = a_out
            print(f"== {study}")
            if study == "steppers":
                recs = study_steppers(R, a.quick) + study_transient(R)
            elif study == "slab":
                recs = study_slab(R, a.quick, a.np, profile)
            elif study == "stehfest":
                recs = study_stehfest(R, a.quick)
            else:
                recs = study_contrast(R, a.quick, a.np)
            if a.dry_run:
                continue
            save(recs, summary)
            if study == "steppers":
                (d / "orders.json").write_text(
                    json.dumps(order_table(recs), indent=1))
        recs = json.loads(summary.read_text())
        {"steppers": figures_steppers, "slab": figures_slab,
         "stehfest": figures_stehfest,
         "contrast": figures_contrast}[study](recs, d)
        print(f"figures in {d}")


if __name__ == "__main__":
    main()
