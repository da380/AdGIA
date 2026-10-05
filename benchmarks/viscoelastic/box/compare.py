"""Compare viscoelastic_box results with their exact reference.

The metrics follow those of the Love-number histories (../love/compare.py),
per observable f over t = 0+ and every output time, all relative to the
largest reference value (the Love-number elastic error is relative to
f_ref(0+) instead):

  history   max_t |f_FE - f_ref| / max_t |f_ref|      (the headline)
  elastic   |f_FE(0+) - f_ref(0+)| / max_t |f_ref|    (spatial error)
  relax     max_t |Df_FE - Df_ref| / max_t |Df_ref|,  Df = f - f(0+)

The observables: for the homogeneous box the loaded strain components
(e_xx, e_yy) under stress control and the branch variables m_k,xx;
for a slab the surface amplitudes W (vertical) and, for n > 0, U
(horizontal); for a sphere the surface coefficients U<d> (radial) and
V<d> (tangential) per degree d (an observable whose reference is zero,
e.g. the other degrees of a radial model, is skipped).

    ./compare results.json --reference reference.json [--plot fig.png]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "common"))
from outputs import outside_source  # noqa: E402


def observables(run: dict) -> dict[str, np.ndarray]:
    """name -> values at (0+, t_1, ...) of a results or reference file."""
    snaps = [run["elastic"]] + run["histories"]
    out: dict[str, list[float]] = {}
    if isinstance(snaps[0].get("U"), list):
        # Sphere: the surface coefficients per degree d, U (radial) and V
        # (tangential).
        for d in range(len(snaps[0]["U"])):
            out[f"U{d}"] = [s["U"][d] for s in snaps]
            out[f"V{d}"] = [s["V"][d] for s in snaps]
    elif "W" in snaps[0]:
        out["W"] = [s["W"] for s in snaps]
        if any(abs(s.get("U", 0.0)) > 0.0 for s in snaps):
            out["U"] = [s["U"] for s in snaps]
    else:
        d = run["dim"]
        if run["load"] == "uniaxial_stress":
            out["e_xx"] = [s["strain"][0] for s in snaps]
            out["e_yy"] = [s["strain"][d + 1] for s in snaps]
        for k in range(len(snaps[0]["internal"])):
            out[f"m{k}_xx"] = [s["internal"][k][0] for s in snaps]
    # Non-finite values are written as null (an unstable run): nan here.
    return {k: np.asarray([np.nan if x is None else x for x in v],
                          dtype=float) for k, v in out.items()}


def metrics(result: dict, reference: dict) -> dict[str, dict[str, float]]:
    fe, ref = observables(result), observables(reference)
    tf = np.asarray([0.0] + [h["time"] for h in result["histories"]])
    tr = np.asarray([0.0] + [h["time"] for h in reference["histories"]])
    if len(tf) != len(tr) or np.abs(tf - tr).max() > 1e-9 * tr.max():
        raise SystemExit("results and reference have different times")
    out = {}
    for name, r in ref.items():
        if name not in fe:
            continue
        f = fe[name]
        scale = np.abs(r).max()
        if scale == 0.0:
            continue
        dr, df = r - r[0], f - f[0]
        dscale = np.abs(dr).max()
        out[name] = {
            "history": float(np.abs(f - r).max() / scale),
            "elastic": float(abs(f[0] - r[0]) / scale),
            "relax": float(np.abs(df - dr).max() / dscale)
            if dscale > 0 else float("nan"),
        }
    return out


def worst(m: dict[str, dict[str, float]], key: str = "history") -> float:
    return max((v[key] for v in m.values()), default=float("nan"))


def plot(result: dict, reference: dict, path: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fe, ref = observables(result), observables(reference)
    t = np.asarray([0.0] + [h["time"] for h in reference["histories"]])
    names = [n for n in ref if n in fe]
    fig, axes = plt.subplots(2, len(names), figsize=(4.2 * len(names), 6),
                             squeeze=False, sharex=True)
    for j, n in enumerate(names):
        axes[0, j].plot(t, ref[n], "-", color="#2a78d6", label="exact")
        axes[0, j].plot(t, fe[n], "o", ms=3.5, color="#e34948", label="FE")
        axes[0, j].set_title(n)
        err = np.abs(fe[n] - ref[n]) / np.abs(ref[n]).max()
        axes[1, j].semilogy(t, np.maximum(err, 1e-17), "-o", ms=3,
                            color="#52514e")
        axes[1, j].set_xlabel("t")
    axes[0, 0].legend()
    axes[1, 0].set_ylabel("|error| / max|ref|")
    fig.suptitle(f"{result['case']}: {result['scheme']}, "
                 f"dt {result['dt']:g}, order {result['order']}")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("results", type=Path)
    p.add_argument("--reference", type=Path, required=True)
    p.add_argument("--plot", type=Path)
    a = p.parse_args()
    res = json.loads(a.results.read_text())
    ref = json.loads(a.reference.read_text())
    m = metrics(res, ref)
    print(f"{res['case']}: {res['scheme']} dt {res['dt']:g}, "
          f"{res['cost']['stepping_solves']} solves")
    print(f"  {'':8s} {'history':>10s} {'elastic':>10s} {'relax':>10s}")
    for name, v in m.items():
        print(f"  {name:8s} {v['history']:10.3e} {v['elastic']:10.3e} "
              f"{v['relax']:10.3e}")
    if a.plot:
        plot(res, ref, outside_source(a.plot))
        print(f"wrote {a.plot}")


if __name__ == "__main__":
    main()
