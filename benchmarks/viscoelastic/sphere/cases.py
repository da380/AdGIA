"""Cases of the sphere benchmarks: models on layered balls and discs.

Non-gravitating, solid-only, surface radius 1, kappa = 5/3 and shear
moduli of order 1 as in the box sub-family, time in units of the mantle's
Maxwell time. The mesh comes first: every interface radius is a multiple
of 0.2, and meshes.py makes each one a surface of the mesh.

MODELS (layers centre outward; a layer with no branches is elastic):

  maxwell_ball  one Maxwell layer [0, 1]: no elastic support, so under a
                degree-l load it flows (a pole at s = 0)
  lid           Maxwell [0, 0.8], elastic lid [0.8, 1]
  core_lid      elastic core [0, 0.4], Maxwell mantle [0.4, 0.8], elastic
                lid [0.8, 1]: a bounded relaxation
  channel       elastic core [0, 0.4], Maxwell [0.4, 0.6] tau 1, channel
                [0.6, 0.8] tau 1/contrast, elastic lid [0.8, 1]
  burgers_lid   a Burgers mantle (0.6, tau 1), (0.4, tau 0.05) [0, 0.8]
                under an elastic lid
  prony_lid     mu_inf 0.1 and branches 0.3 at tau 1e-2, 1, 1e2, [0, 0.8],
                under an elastic lid
  lateral       core_lid whose mantle tau varies smoothly sideways,
                tau = C^((1 + tanh(x_a / w)) / 2) with C = contrast,
                w = 0.15, x_a the polar axis (z in 3-D, x in 2-D, so that
                the m = 0 / cosine loads stay symmetric): the slow side
                tau = C, the fast side tau = 1. No exact reference: the
                studies compare with a much smaller step on the same mesh.
  lateral_weak  tau from 1/C (the weak side) to 1: a low-viscosity
                region, hence a stiff problem (tau_min = 1/C); the mantle
                is a standard linear solid (mu_inf 0.1, branch 0.9), since
                a fully relaxing one leaves the elastic core free (no
                gravity) and the response diverges

Schema:

  {"schema": "viscoelastic_sphere/1", "name", "model", "dim",
   "radii": [0, ..., 1], "mesh": "<manifest path>",
   "layers": [{"name", "kappa", "mu_inf", "branches": [{"mu", "tau"}]}],
   "load": {"degree", "amplitude", "history"}, "lmax", "times"}

The mesh path is filled in by the caller (study.py), which builds it.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "box"))
sys.path.insert(0, str(HERE))

import histories as hist  # noqa: E402
from meshes import outside_source  # noqa: E402

KAPPA = 5.0 / 3.0


def maxwell(name, tau, mu=1.0):
    return {"name": name, "kappa": KAPPA, "mu_inf": 0.0,
            "branches": [{"mu": mu, "tau": tau}]}


def elastic(name, mu=1.0):
    return {"name": name, "kappa": KAPPA, "mu_inf": mu, "branches": []}


def model(name: str, dim: int, contrast: float = 100.0):
    """(radii, layers)."""
    if name == "maxwell_ball":
        return [0.0, 1.0], [maxwell("mantle", 1.0)]
    if name == "lid":
        return [0.0, 0.8, 1.0], [maxwell("mantle", 1.0), elastic("lid")]
    if name == "core_lid":
        return [0.0, 0.4, 0.8, 1.0], [elastic("core"),
                                      maxwell("mantle", 1.0),
                                      elastic("lid")]
    if name == "channel":
        return [0.0, 0.4, 0.6, 0.8, 1.0], [
            elastic("core"), maxwell("mantle", 1.0),
            maxwell("channel", 1.0 / contrast), elastic("lid")]
    if name == "burgers_lid":
        return [0.0, 0.8, 1.0], [
            {"name": "mantle", "kappa": KAPPA, "mu_inf": 0.0,
             "branches": [{"mu": 0.6, "tau": 1.0},
                          {"mu": 0.4, "tau": 0.05}]},
            elastic("lid")]
    if name == "prony_lid":
        return [0.0, 0.8, 1.0], [
            {"name": "mantle", "kappa": KAPPA, "mu_inf": 0.1,
             "branches": [{"mu": 0.3, "tau": t} for t in (1e-2, 1.0, 1e2)]},
            elastic("lid")]
    if name in ("lateral", "lateral_weak"):
        # lateral: tau from 1 to C (a slow region; tau_min = 1 whatever C);
        # lateral_weak: tau from 1/C to 1 (a weak region; stiff).
        tau = {"kind": "lateral_tanh",
               "value": 1.0 if name == "lateral" else 1.0 / contrast,
               "contrast": contrast, "width": 0.15,
               "axis": dim - 1 if dim == 3 else 0}
        # lateral_weak's mantle keeps mu_inf = 0.1 (a standard linear
        # solid, unrelaxed modulus 1 as the others): with a Maxwell mantle
        # that relaxes fully, the elastic core of this free, non-gravitating
        # body has nothing to hold it, its motion relative to the lid is a
        # zero-stiffness mode the whole-body rigid projection does not
        # remove, and the odd lateral variation forces it (at C = 1e4 the
        # response diverges).
        mu_inf = 0.0 if name == "lateral" else 0.1
        return [0.0, 0.4, 0.8, 1.0], [
            elastic("core"),
            {"name": "mantle", "kappa": KAPPA, "mu_inf": mu_inf,
             "branches": [{"mu": 1.0 - mu_inf, "tau": tau}]},
            elastic("lid")]
    raise SystemExit(f"unknown model {name!r}")


MODELS = ("maxwell_ball", "lid", "core_lid", "channel", "burgers_lid",
          "prony_lid", "lateral", "lateral_weak")
RADIAL = MODELS[:-2]


def output_times(t_final: float, n: int, history) -> list[float]:
    times = [t_final * (j + 1) / n for j in range(n)]
    bps = [b["time"] for b in history.breakpoints()]
    for j, t in enumerate(times):
        for b in bps:
            if abs(t - b) < 1e-6 * t_final:
                times[j] = t + 1e-3 * t_final / n
    return times


def make_case(model_name: str, *, dim: int = 2, degree: int = 2,
              history=None, amplitude: float = 1.0,
              contrast: float = 100.0, t_final: float = 8.0,
              n_times: int = 16, lmax: int | None = None,
              name: str | None = None) -> dict:
    radii, layers = model(model_name, dim, contrast)
    for r in radii:
        if abs(r * 5 - round(r * 5)) > 1e-12:
            raise SystemExit(f"{model_name}: radius {r} off the 0.2 grid")
    history = history or hist.heaviside()
    lmax = lmax if lmax is not None else (
        degree + 4 if model_name.startswith("lateral") else degree)
    tag = name or "_".join([model_name, f"{dim}d", f"l{degree}",
                            history.name]
                           + ([f"C{contrast:g}"]
                              if model_name in ("lateral", "lateral_weak",
                                                "channel")
                              else []))
    return {"schema": "viscoelastic_sphere/1", "name": tag,
            "model": model_name, "dim": dim, "radii": radii, "mesh": None,
            "layers": layers,
            "load": {"degree": degree, "amplitude": amplitude,
                     "history": history.to_json()},
            "lmax": lmax, "contrast": contrast,
            "times": output_times(t_final, n_times, history)}


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("model", choices=MODELS)
    p.add_argument("--dim", type=int, default=2)
    p.add_argument("--degree", type=int, default=2)
    p.add_argument("--history", default="heaviside",
                   choices=sorted(hist.PRESETS))
    p.add_argument("--contrast", type=float, default=100.0)
    p.add_argument("--h", type=float, default=0.1,
                   help="element size of the mesh to build")
    p.add_argument("--order", type=int, default=2, help="mesh order")
    p.add_argument("--meshes", type=Path, default=Path.cwd() / "meshes")
    p.add_argument("--out", type=Path, default=Path("case.json"))
    a = p.parse_args()
    from meshes import mesh_path
    case = make_case(a.model, dim=a.dim, degree=a.degree,
                     history=hist.make(a.history), contrast=a.contrast)
    case["mesh"] = str(mesh_path(a.meshes, a.dim, case["radii"], a.h,
                                 a.order))
    out = outside_source(a.out)
    out.write_text(json.dumps(case, indent=1))
    print(f"wrote {out} ({case['name']})")


if __name__ == "__main__":
    main()
