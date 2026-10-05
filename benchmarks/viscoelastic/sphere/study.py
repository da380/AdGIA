"""The sphere studies: viscoelastic_sphere on layered balls and discs.

  radial   radial models against their exact references (reference.py):
           every model, degrees 2 and 4 (2-D), a mesh ladder at orders 1
           and 2, the time error kept small (ExpTrap, dt = 1/40): the
           spatial convergence of the relaxation on curved, spherically
           layered meshes, 2-D discs and (fewer, coarser) 3-D balls.
  lateral  the time integrators with a laterally varying viscosity: the
           'lateral' model (mantle tau from 1 to C across the polar axis,
           smooth: a slow region), C = 1, 1e2, 1e4, under a Heaviside, a
           load-and-remove and a periodic load, and 'lateral_weak' (tau
           from 1/C to 1: a weak region, stiff), C = 1e2, 1e4, Heaviside
           and periodic; every scheme on a step ladder. There is no
           exact reference, so each run is compared with SDIRK23 at
           dt = 1/256 on the same mesh (the spatial error cancels); ExpTrap
           at the same small step measures that reference's own error. The
           lateral variation couples the load's degree to the others:
           the observables are the surface coefficients of every degree to
           l + 4.

Profiles: "local" (default) keeps every run to minutes on a laptop (3-D
only on the coarsest balls); "server" adds finer 3-D meshes and the 3-D
lateral study.

    ./study radial
    ./study lateral
    ./study all --profile server
    ./study radial --figures-only

Outputs (never in the source tree): <out>/<study>/summary.json, figures,
and per run its case, reference and results under runs/; meshes under
<out>/meshes.
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
sys.path.insert(0, str(HERE.parent / "box"))
sys.path.insert(0, str(HERE))

import cases  # noqa: E402
import compare  # noqa: E402
import histories as hist  # noqa: E402
import reference as sref  # noqa: E402
from meshes import mesh_path, outside_source  # noqa: E402


# The box studies' scheme styles.
COLOURS = {"rk4": "#e34948", "etd1": "#e87ba4", "be": "#eda100",
           "sdirk23": "#2a78d6", "exptrap": "#1baf7a",
           "adaptive": "#4a3aa7"}
MARKERS = {"rk4": "v", "etd1": "P", "be": "D", "sdirk23": "o",
           "exptrap": "^", "adaptive": "s"}
LABELS = {"rk4": "RK4", "etd1": "ETD1", "be": "BE", "sdirk23": "SDIRK23",
          "exptrap": "ExpTrap", "adaptive": "Adaptive"}


def _plt():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size": 11, "axes.grid": True,
                         "grid.color": "#e4e3df", "lines.linewidth": 1.8,
                         "lines.markersize": 5, "savefig.dpi": 160})
    return plt

PROFILES = {
    "local": dict(np=2, jobs=10,
                  radial_2d=[(1, 0.1), (1, 0.05), (2, 0.2), (2, 0.1),
                             (2, 0.05)],
                  radial_3d=[(2, 0.25), (2, 0.18)],
                  lateral_dims=(2,), lateral_h={2: 0.1}),
    "server": dict(np=8, jobs=96,
                   radial_2d=[(1, 0.1), (1, 0.05), (1, 0.025), (2, 0.2),
                              (2, 0.1), (2, 0.05), (2, 0.025)],
                   radial_3d=[(2, 0.25), (2, 0.18), (2, 0.12), (2, 0.09)],
                   lateral_dims=(2, 3), lateral_h={2: 0.05, 3: 0.18}),
}
SCHEMES = ("etd1", "be", "exptrap", "sdirk23", "rk4", "adaptive")
LATERAL_DTS = [2.0 ** -j for j in range(1, 7)]        # 1/2 .. 1/64
LATERAL_RTOLS = [1e-2, 1e-3, 1e-4]  # tighter: tens of thousands of solves
REFERENCE_DT = 1.0 / 256


def find_program(given: Path | None) -> Path:
    if given is not None:
        p = given / "viscoelastic_sphere" if given.is_dir() else given
        return p.resolve()
    for build in sorted(HERE.parent.parent.parent.glob("build*")):
        c = build / "benchmarks" / "bin" / "viscoelastic_sphere"
        if c.exists():
            return c
    raise SystemExit("viscoelastic_sphere not found; pass --programs")


class Runner:
    def __init__(self, a, out: Path):
        self.program = find_program(a.programs)
        self.mpiexec = a.mpiexec
        self.out = out
        self.meshes = a.out / "meshes"
        self.jobs = a.jobs
        self.dry = a.dry_run

    def case_dir(self, case: dict, mesh: tuple) -> tuple[Path, Path]:
        """Write the case (with its mesh) once; a changed case clears its
        results."""
        o, h = mesh
        case = dict(case)
        case["mesh"] = str(mesh_path(self.meshes, case["dim"],
                                     case["radii"], h, 2))
        d = self.out / "runs" / f"{case['name']}_h{h:g}"
        d.mkdir(parents=True, exist_ok=True)
        cpath = d / "case.json"
        text = json.dumps(case, indent=1)
        if not cpath.exists() or cpath.read_text() != text:
            cpath.write_text(text)
            for stale in [d / "reference.json", *d.glob("res_*.json")]:
                if stale.exists():
                    stale.unlink()
        return d, cpath

    def exact(self, d: Path, cpath: Path) -> Path:
        rpath = d / "reference.json"
        if not rpath.exists():
            rpath.write_text(json.dumps(
                sref.reference(sref.SphereCase(cpath)), indent=1))
        return rpath

    def run(self, cpath: Path, opts: dict, np_: int) -> Path | None:
        tag = "_".join(f"{k}{v}" for k, v in opts.items())
        res = cpath.parent / f"res_{tag}.json"
        cmd = [self.mpiexec, "-np", str(np_), str(self.program),
               "-c", str(cpath), "-out", str(res)]
        for k, v in opts.items():
            if isinstance(v, bool):
                cmd.append(f"-{k}" if v else f"-no-{k}")
            else:
                cmd += [f"-{k}", str(v)]
        if self.dry:
            print("$ " + shlex.join(cmd))
            return None
        if not res.exists():
            p = subprocess.run(cmd, capture_output=True, text=True)
            if p.returncode != 0 or not res.exists():
                (cpath.parent / f"log_{tag}.txt").write_text(
                    p.stdout + p.stderr)
                return None
        return res

    def parallel(self, jobs, np_):
        out = []
        with cf.ThreadPoolExecutor(max(1, self.jobs // np_)) as ex:
            futures = [ex.submit(self.run, *j) for j in jobs]
            for n, f in enumerate(cf.as_completed(futures), 1):
                out.append(f.result())
                if n % 20 == 0 or n == len(futures):
                    print(f"  {n}/{len(futures)} runs", flush=True)
        return out


def family_metrics(result: dict, reference: dict) -> dict:
    """Errors of the surface coefficients of all degrees together, per
    family (U radial, V tangential), relative to the largest reference
    coefficient of the family over degrees and times. Used against a
    finite-element reference (the lateral study): per-degree scales would
    divide the numerical leakage into the uncoupled degrees by itself."""
    out = {}
    for fam in ("U", "V"):
        f = np.array([s[fam] for s in [result["elastic"]]
                      + result["histories"]], dtype=float)
        g = np.array([s[fam] for s in [reference["elastic"]]
                      + reference["histories"]], dtype=float)
        scale = np.abs(g).max()
        df, dg = f - f[0], g - g[0]
        out[fam] = {"history": float(np.abs(f - g).max() / scale),
                    "elastic": float(np.abs(f[0] - g[0]).max() / scale),
                    "relax": float(np.abs(df - dg).max()
                                   / max(np.abs(dg).max(), 1e-300))}
    return out


def record(res: Path | None, ref: Path, family: bool = False,
           **meta) -> dict:
    rec = dict(meta, ok=False)
    if res is None:
        return rec
    r = json.loads(res.read_text())
    reference = json.loads(ref.read_text())
    m = (family_metrics(r, reference) if family
         else compare.metrics(r, reference))
    c = r["cost"]
    solves = c["stepping_solves"] + c["observation_solves"]
    rec.update(ok=c["converged"], error=compare.worst(m), errors=m,
               solves=solves, assemblies=c["assemblies"],
               setups=c["preconditioner_setups"],
               iterations=c["iterations"],
               its_per_solve=c["iterations"] / max(solves, 1),
               steps=c["steps"], seconds=c["seconds"], dofs=r["dofs"])
    if not all(math.isfinite(v["history"]) for v in m.values()):
        rec["ok"] = False
    return rec


# --- the studies ------------------------------------------------------------------


def study_radial(R: Runner, prof: dict, np_: int) -> list[dict]:
    jobs, meta = [], []
    specs = [(m, 2, l) for m in cases.RADIAL for l in (2, 4)]
    specs += [(m, 3, 2) for m in ("core_lid", "burgers_lid")]
    for model, dim, degree in specs:
        c = cases.make_case(model, dim=dim, degree=degree)
        ladder = prof["radial_2d"] if dim == 2 else prof["radial_3d"]
        for o, h in ladder:
            d, cpath = R.case_dir(c, (o, h))
            ref = R.exact(d, cpath)
            opts = {"scheme": "exptrap", "dt": 0.025, "o": o}
            n = np_ * (2 if dim == 3 else 1)
            jobs.append((cpath, opts, n))
            meta.append((ref, dict(model=model, dim=dim, degree=degree,
                                   order=o, h=h, case=c["name"])))
    R.parallel(jobs, max(j[2] for j in jobs))
    return [record(res, ref, **m) for res, (ref, m) in
            zip(_in_order(R, jobs), meta)]


def _in_order(R: Runner, jobs):
    """The result paths of jobs, in job order (all are cached by now)."""
    return [R.run(*j) for j in jobs]


def study_lateral(R: Runner, prof: dict, np_: int) -> list[dict]:
    recs = []
    for dim in prof["lateral_dims"]:
        h = prof["lateral_h"][dim]
        n = np_ * (2 if dim == 3 else 1)
        runs = []
        specs = [("lateral", c, name) for c in (1.0, 1e2, 1e4)
                 for name in ("heaviside", "load_unload", "periodic")]
        specs += [("lateral_weak", c, name) for c in (1e2, 1e4)
                  for name in ("heaviside", "periodic")]
        for model, contrast, hname in specs:
            c = cases.make_case(model, dim=dim, degree=2,
                                history=hist.make(hname),
                                contrast=contrast)
            d, cpath = R.case_dir(c, (2, h))
            base = {"o": 2}
            ref_opts = dict(base, scheme="sdirk23", dt=REFERENCE_DT)
            chk_opts = dict(base, scheme="exptrap", dt=REFERENCE_DT)
            trial = []
            for s in SCHEMES:
                if s == "adaptive" and model == "lateral_weak":
                    # Its first-order estimate follows the weak region's
                    # tau = 1/C transient, which the L-stable schemes step
                    # over: runs of over an hour on a disc (as the box
                    # contrast study's tau = 1e-8 column).
                    continue
                if s == "adaptive":
                    for rtol in LATERAL_RTOLS:
                        trial.append(dict(base, scheme=s, rtol=rtol,
                                          atol=1e-12, dt=0.25))
                else:
                    for dt in LATERAL_DTS:
                        trial.append(dict(base, scheme=s, dt=dt))
            runs.append((cpath, ref_opts, chk_opts, trial,
                         dict(model=model, dim=dim, h=h,
                              contrast=contrast, history=hname,
                              case=c["name"])))
        # The references first (the longest runs), then everything.
        R.parallel([(cp, ro, n) for cp, ro, _, _, _ in runs]
                   + [(cp, co, n) for cp, _, co, _, _ in runs], n)
        R.parallel([(cp, t, n) for cp, _, _, tr, _ in runs for t in tr], n)
        for cp, ro, co, tr, meta in runs:
            ref = R.run(cp, ro, n)
            if ref is None:
                continue
            chk = record(R.run(cp, co, n), ref, family=True, **meta,
                         scheme="check", dt=REFERENCE_DT)
            meta = dict(meta, reference_error=chk.get("error"))
            for t in tr:
                recs.append(record(R.run(cp, t, n), ref, family=True,
                                   **meta, scheme=t["scheme"], dt=t["dt"],
                                   rtol=t.get("rtol")))
            # the coupling: the largest coefficient of another degree,
            # relative to the load's, over the reference history
            r = json.loads(ref.read_text())
            U = np.array([s["U"] for s in [r["elastic"]] + r["histories"]])
            lead = np.abs(U[:, r["degree"]]).max()
            others = np.delete(np.abs(U), r["degree"], axis=1).max()
            recs.append(dict(meta, scheme="coupling",
                             coupling=float(others / lead), ok=True))
    return recs


# --- figures -------------------------------------------------------------------------


def figures_radial(recs, out: Path) -> None:
    plt = _plt()
    ok = [r for r in recs if r.get("ok")]
    for dim in (2, 3):
        sel = [r for r in ok if r["dim"] == dim]
        models = [m for m in cases.RADIAL if any(r["model"] == m
                                                 for r in sel)]
        if not models:
            continue
        cols = min(3, len(models))
        rows = math.ceil(len(models) / cols)
        fig, axes = plt.subplots(rows, cols, figsize=(4.2 * cols, 3.4 * rows),
                                 squeeze=False, sharey=True)
        for ax in axes.flat[len(models):]:
            ax.set_visible(False)
        for ax, m in zip(axes.flat, models):
            for degree, colour in ((2, "#2a78d6"), (4, "#e34948")):
                for o, ls in ((1, ":"), (2, "-")):
                    pts = sorted((r["h"], r["error"]) for r in sel
                                 if r["model"] == m and r["degree"] == degree
                                 and r["order"] == o)
                    if pts:
                        x, y = zip(*pts)
                        ax.loglog(x, y, ls, marker="o", color=colour,
                                  label=f"l={degree}, p={o}")
            ax.set_title(m)
            ax.set_xlabel("h")
            hs = sorted({r["h"] for r in sel})
            ax.set_xticks(hs, [f"{h:g}" for h in hs])
            ax.minorticks_off()
        for ax in axes[:, 0]:
            ax.set_ylabel("history error (U, V)")
        axes[0, 0].legend(fontsize=7)
        fig.suptitle(f"{'discs' if dim == 2 else 'balls'}: spatial "
                     "convergence against the exact histories")
        fig.tight_layout()
        fig.savefig(out / f"radial_{dim}d.png")
        plt.close(fig)


def figures_lateral(recs, out: Path) -> None:
    plt = _plt()
    allok = [r for r in recs if r.get("ok") and r["scheme"] not in
             ("coupling", "check") and r.get("error", 1e9) < 10]
    for model, dim in sorted({(r.get("model", "lateral"), r["dim"])
                              for r in allok}):
        ok = [r for r in allok if r.get("model", "lateral") == model
              and r["dim"] == dim]
        _lateral_figure(ok, model, dim, out)


def _lateral_figure(ok, model, dim, out: Path) -> None:
    plt = _plt()
    for xkey, xlabel, fname in (("dt", "dt / tau", "dt"),
                                ("solves", "elastic solves", "cost")):
        hs = [h for h in ("heaviside", "load_unload", "periodic")
              if any(r["history"] == h for r in ok)]
        cs = sorted({r["contrast"] for r in ok})
        fig, axes = plt.subplots(len(hs), len(cs),
                                 figsize=(3.6 * len(cs), 3.0 * len(hs)),
                                 squeeze=False, sharex=True, sharey=True)
        for i, h in enumerate(hs):
            for j, c in enumerate(cs):
                ax = axes[i, j]
                for s in SCHEMES:
                    if xkey == "dt" and s == "adaptive":
                        continue
                    pts = sorted((r[xkey], r["error"]) for r in ok
                                 if r["dim"] == dim and r["history"] == h
                                 and r["contrast"] == c
                                 and r["scheme"] == s)
                    if pts:
                        x, y = zip(*pts)
                        ax.loglog(x, np.maximum(y, 1e-14), "-",
                                  marker=MARKERS[s], color=COLOURS[s],
                                  label=LABELS[s])
                ax.set_title(f"{h}, C = {c:g}", fontsize=10)
                if xkey == "dt":
                    ax.set_xticks(LATERAL_DTS,
                                  [f"1/{round(1 / d)}" for d in LATERAL_DTS])
                    ax.minorticks_off()
                if i == len(hs) - 1:
                    ax.set_xlabel(xlabel)
            axes[i, 0].set_ylabel("error vs dt = 1/256")
        axes[0, 0].legend(fontsize=7)
        fig.suptitle(f"{model} ({dim}-D): "
                     + ("error against step" if xkey == "dt"
                        else "error against cost"))
        fig.tight_layout()
        fig.savefig(out / f"{model}_{fname}_{dim}d.png")
        plt.close(fig)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("study", choices=("radial", "lateral", "all"))
    p.add_argument("--profile", choices=sorted(PROFILES), default="local")
    p.add_argument("--out", type=Path, default=Path("."))
    p.add_argument("--programs", type=Path)
    p.add_argument("--mpiexec", default="mpiexec")
    p.add_argument("--jobs", type=int, default=None)
    p.add_argument("--np", type=int, default=None)
    p.add_argument("--figures-only", action="store_true")
    p.add_argument("--dry-run", action="store_true")
    a = p.parse_args()
    prof = PROFILES[a.profile]
    a.jobs = a.jobs or prof["jobs"]
    a.np = a.np or prof["np"]
    a.out = outside_source(a.out)
    for study in (("radial", "lateral") if a.study == "all"
                  else (a.study,)):
        d = a.out / study
        d.mkdir(parents=True, exist_ok=True)
        summary = d / "summary.json"
        if not a.figures_only:
            print(f"== {study}", flush=True)
            R = Runner(a, d)
            recs = (study_radial(R, prof, a.np) if study == "radial"
                    else study_lateral(R, prof, a.np))
            if a.dry_run:
                continue
            summary.write_text(json.dumps(recs, indent=1))
            print(f"wrote {summary} ({len(recs)} records)")
        recs = json.loads(summary.read_text())
        (figures_radial if study == "radial" else figures_lateral)(recs, d)
        print(f"figures in {d}")


if __name__ == "__main__":
    main()
