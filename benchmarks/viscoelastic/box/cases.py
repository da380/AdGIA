"""Cases of the box benchmarks: models, loads, histories, output times.

A case is one JSON file read by both viscoelastic_box.cpp and
reference.py (schema below). Everything is nondimensional: shear moduli
of order 1, kappa = 5/3 (Poisson's ratio 1/4 for the 3-D body), and time
in units of the shortest relaxation time of interest (1 unless a model
says otherwise).

MODELS (layers bottom-up; a layer with no branches is elastic):

  homogeneous bodies, for the box loads (uniaxial_stress/_strain):
    maxwell       mu_inf = 0, one branch (1, tau 1)
    sls           standard linear solid: mu_inf 0.5, one branch (0.5, 1)
    burgers       mu_inf = 0, branches (0.6, 1), (0.4, 0.05)
    prony_wide    mu_inf 0.1, four branches with tau 1e-3, 1e-1, 1e1, 1e3:
                  a relaxation spectrum spanning six decades (stiff)

  layered slabs (height 1, width --width, periodic), surface loads:
    maxwell_slab  one Maxwell layer: no elastic support, so under a
                  non-uniform load it FLOWS (a pole at s = 0); bounded
                  under a uniform load (it relaxes to its kappa response)
    lid           Maxwell layer [0, 0.8], elastic lid [0.8, 1]
    channel       Maxwell mantle [0, 0.6] tau 1, low-viscosity channel
                  [0.6, 0.8] tau 1/contrast, elastic lid [0.8, 1]
    sls_stack     two standard linear solids, tau 1 below 0.6 and 10 above
    burgers_lid   a Burgers mantle (branches (0.6, tau 1), (0.4, tau 0.05))
                  [0, 0.8] under an elastic lid
    prony_lid     a generalised Maxwell mantle, mu_inf 0.1 and three
                  branches of 0.3 at tau 1e-2, 1, 1e2, under an elastic lid
    contrast      Maxwell [0, 0.4] tau 1, [0.4, 0.6] tau 1/contrast,
                  [0.6, 0.8] tau contrast (stiff, nearly elastic), elastic
                  lid [0.8, 1]: the viscosity-contrast ladder
    gradient      one Maxwell layer [0, 0.8] whose tau varies geometrically
                  from 1/contrast at the base to 1 at its top, elastic lid
                  [0.8, 1] (column, mode 0, only)

The mesh comes first: every layer interface is a multiple of 0.2, the
cell size of the coarsest mesh of the ladders (nz = 5), so that every
uniform mesh with nz a multiple of 5 has the material discontinuities on
element faces. The driver refuses a mesh that cuts a layer unless asked
for the deliberate misfit (-no-conform).

Schema:

  {"schema": "viscoelastic_box/1", "name", "dim",
   "geometry": {"kind": "box" | "slab", "length": [Lx(, Ly)], "height"},
   "layers": [{"name", "top", "kappa", "mu_inf",
               "branches": [{"mu", "tau"}]}],      # fields: number or
                                                    # {"kind": "geometric_z",
                                                    #  "bottom", "top"}
   "load": {"kind", "amplitude", "mode": [n...], "history": {...}},
   "times": [...]}

    ./cases.py lid --dim 2 --mode 1 --history periodic --param omega=3 \\
        --out lid.json
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent.parent / "common"))
sys.path.insert(0, str(HERE))

import histories as hist  # noqa: E402
from outputs import outside_source  # noqa: E402

KAPPA = 5.0 / 3.0


def maxwell_layer(name, top, tau, mu=1.0, kappa=KAPPA):
    return {"name": name, "top": top, "kappa": kappa, "mu_inf": 0.0,
            "branches": [{"mu": mu, "tau": tau}]}


def elastic_layer(name, top, mu=1.0, kappa=KAPPA):
    return {"name": name, "top": top, "kappa": kappa, "mu_inf": mu,
            "branches": []}


def model(name: str, contrast: float = 100.0) -> tuple[str, list[dict]]:
    """(geometry kind, layers) of a model, height 1."""
    if name == "maxwell":
        return "box", [maxwell_layer("body", 1.0, 1.0)]
    if name == "sls":
        return "box", [{"name": "body", "top": 1.0, "kappa": KAPPA,
                        "mu_inf": 0.5,
                        "branches": [{"mu": 0.5, "tau": 1.0}]}]
    if name == "burgers":
        return "box", [{"name": "body", "top": 1.0, "kappa": KAPPA,
                        "mu_inf": 0.0,
                        "branches": [{"mu": 0.6, "tau": 1.0},
                                     {"mu": 0.4, "tau": 0.05}]}]
    if name == "prony_wide":
        return "box", [{"name": "body", "top": 1.0, "kappa": KAPPA,
                        "mu_inf": 0.1,
                        "branches": [{"mu": 0.225, "tau": t}
                                     for t in (1e-3, 1e-1, 1e1, 1e3)]}]
    if name == "maxwell_slab":
        return "slab", [maxwell_layer("mantle", 1.0, 1.0)]
    if name == "lid":
        return "slab", [maxwell_layer("mantle", 0.8, 1.0),
                        elastic_layer("lid", 1.0)]
    if name == "channel":
        return "slab", [maxwell_layer("mantle", 0.6, 1.0),
                        maxwell_layer("channel", 0.8, 1.0 / contrast),
                        elastic_layer("lid", 1.0)]
    if name == "sls_stack":
        return "slab", [
            {"name": "lower", "top": 0.6, "kappa": KAPPA, "mu_inf": 0.5,
             "branches": [{"mu": 0.5, "tau": 1.0}]},
            {"name": "upper", "top": 1.0, "kappa": KAPPA, "mu_inf": 0.3,
             "branches": [{"mu": 0.7, "tau": 10.0}]}]
    if name == "burgers_lid":
        return "slab", [{"name": "mantle", "top": 0.8, "kappa": KAPPA,
                         "mu_inf": 0.0,
                         "branches": [{"mu": 0.6, "tau": 1.0},
                                      {"mu": 0.4, "tau": 0.05}]},
                        elastic_layer("lid", 1.0)]
    if name == "prony_lid":
        return "slab", [{"name": "mantle", "top": 0.8, "kappa": KAPPA,
                         "mu_inf": 0.1,
                         "branches": [{"mu": 0.3, "tau": t}
                                      for t in (1e-2, 1.0, 1e2)]},
                        elastic_layer("lid", 1.0)]
    if name == "contrast":
        return "slab", [maxwell_layer("lower", 0.4, 1.0),
                        maxwell_layer("weak", 0.6, 1.0 / contrast),
                        maxwell_layer("stiff", 0.8, contrast),
                        elastic_layer("lid", 1.0)]
    if name == "gradient":
        return "slab", [
            {"name": "mantle", "top": 0.8, "kappa": KAPPA, "mu_inf": 0.0,
             "branches": [{"mu": 1.0,
                           "tau": {"kind": "geometric_z",
                                   "bottom": 1.0 / contrast, "top": 1.0}}]},
            elastic_layer("lid", 1.0)]
    raise SystemExit(f"unknown model {name!r}")


MODELS = ("maxwell", "sls", "burgers", "prony_wide", "maxwell_slab", "lid",
          "channel", "sls_stack", "burgers_lid", "prony_lid", "contrast",
          "gradient")


def output_times(t_final: float, n: int, history: hist.History) -> list[float]:
    """n equally spaced times ending at t_final. A step that divides the
    spacing then keeps the step grid uniform (one effective operator for
    the whole run); the histories' breakpoints sit at irrational times, so
    they fall between grid points unless aligned. A time that would land
    on a breakpoint (the load is ambiguous there) is nudged off it."""
    times = [t_final * (j + 1) / n for j in range(n)]
    bps = [b["time"] for b in history.breakpoints()]
    for j, t in enumerate(times):
        for b in bps:
            if abs(t - b) < 1e-6 * t_final:
                times[j] = t + 1e-3 * t_final / n
    return times


def make_case(model_name: str, *, dim: int = 2, mode=(1,),
              history: hist.History | None = None, load: str | None = None,
              amplitude: float = 1.0, width: float = 2.0,
              contrast: float = 100.0, t_final: float = 8.0,
              n_times: int = 40, times: list[float] | None = None,
              name: str | None = None) -> dict:
    kind, layers = model(model_name, contrast)
    for layer in layers[:-1]:
        if abs(layer["top"] * 5 - round(layer["top"] * 5)) > 1e-12:
            raise SystemExit(f"{model_name}: interface {layer['top']} is "
                             "not on the nz = 5 grid")
    history = history or hist.heaviside()
    if kind == "box":
        load = load or "uniaxial_stress"
        geometry = {"kind": "box", "length": [1.0] * (dim - 1),
                    "height": 1.0}
        load_spec = {"kind": load, "amplitude": amplitude}
    else:
        load = "surface"
        mode = list(mode) + [0] * (dim - 1 - len(mode))
        geometry = {"kind": "slab", "length": [width] * (dim - 1),
                    "height": 1.0}
        load_spec = {"kind": "surface", "amplitude": amplitude,
                     "mode": mode[: dim - 1]}
    load_spec["history"] = history.to_json()
    if times is None:
        times = output_times(t_final, n_times, history)
    tag = name or "_".join(
        [model_name, f"{dim}d", load]
        + ([f"n{'x'.join(str(m) for m in load_spec['mode'])}"]
           if kind == "slab" else [])
        + [history.name])
    return {"schema": "viscoelastic_box/1", "name": tag, "model": model_name,
            "dim": dim,
            "geometry": geometry, "layers": layers, "load": load_spec,
            "contrast": contrast, "times": times}


def parse_params(items: list[str]) -> dict:
    out = {}
    for item in items:
        k, v = item.split("=", 1)
        out[k] = int(v) if v.isdigit() else float(v)
    return out


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("model", choices=MODELS)
    p.add_argument("--dim", type=int, default=2)
    p.add_argument("--mode", type=int, nargs="*", default=[1])
    p.add_argument("--load", choices=("uniaxial_stress", "uniaxial_strain"))
    p.add_argument("--history", default="heaviside",
                   choices=sorted(hist.PRESETS))
    p.add_argument("--param", nargs="*", default=[],
                   help="history parameters, key=value")
    p.add_argument("--contrast", type=float, default=100.0)
    p.add_argument("--width", type=float, default=2.0)
    p.add_argument("--t-final", type=float, default=8.0)
    p.add_argument("--n-times", type=int, default=40)
    p.add_argument("--out", type=Path, default=Path("case.json"))
    a = p.parse_args()
    h = hist.make(a.history, **parse_params(a.param))
    case = make_case(a.model, dim=a.dim, mode=a.mode, history=h, load=a.load,
                     width=a.width, contrast=a.contrast, t_final=a.t_final,
                     n_times=a.n_times)
    out = outside_source(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(case, indent=1))
    print(f"wrote {a.out} ({case['name']})")


if __name__ == "__main__":
    main()
