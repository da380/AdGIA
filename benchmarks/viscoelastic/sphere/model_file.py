"""The viscosity of a sphere case as a netCDF model file.

The sphere cases (cases.py) give each Maxwell layer one branch {mu, tau};
its viscosity is eta = mu tau, laterally varying for the lateral cases.
This writes that viscosity as a planetmodel model file, which the
driver's -model flag reads back (tau = eta / mu, through planetmodel's
interpolants) in place of the analytic fields — the delivery route of
real, gridded viscosity models. 3-D only, as the flag is (the lateral
cases' 2-D axis is not the equatorial plane a disc reads).

    ./model_file lateral --out <dir> [--contrast 100] [--lmax 64]

writes <out>/<model>_viscosity.nc and prints the path.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent.parent / "common"))
sys.path.insert(0, str(HERE))

import numpy as np

from outputs import outside_source  # noqa: E402
import cases  # noqa: E402


def viscosity(layer: dict):
    """eta(r, theta, phi) of a one-branch Maxwell layer, in the case's
    conventions (viscoelastic_common.hpp: lateral_tanh on the polar
    axis)."""
    [branch] = layer["branches"]
    mu, tau = branch["mu"], branch["tau"]
    if isinstance(tau, (int, float)):
        return lambda r, theta, phi: mu * tau * np.ones_like(r)
    assert tau["kind"] == "lateral_tanh" and tau["axis"] == 2
    value, contrast, width = tau["value"], tau["contrast"], tau["width"]

    def eta(r, theta, phi):
        z = r * np.cos(theta)
        return mu * value * contrast ** (0.5 * (1.0 + np.tanh(z / width)))

    return eta


def main() -> None:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("model", choices=cases.MODELS, help="the model")
    p.add_argument("--out", type=Path, required=True, metavar="DIR",
                   help="output directory (outside the source tree)")
    p.add_argument("--contrast", type=float, default=100.0,
                   help="viscosity contrast of the channel/lateral models")
    p.add_argument("--lmax", type=int, default=64,
                   help="angular grid of the file")
    args = p.parse_args()

    from planetmodel import (AnalyticField, Geometry, Model, Skeleton,
                             gauss_legendre, modelfile)

    radii, layers = cases.model(args.model, dim=3, contrast=args.contrast)
    geometry = Geometry(Skeleton(list(radii)),
                        layer_names=[layer["name"] for layer in layers],
                        interface_names=[f"r{r:g}" for r in radii[1:-1]]
                        + ["surface"])
    fields = []
    for j, layer in enumerate(layers):
        if not layer["branches"]:
            fields.append({})
            continue
        if len(layer["branches"]) != 1:
            raise SystemExit(
                f"{args.model}: layer {layer['name']} has "
                f"{len(layer['branches'])} branches; the file's viscosity "
                "defines one (tau = eta / mu)")
        interval = (radii[j], radii[j + 1])
        fields.append(
            {"viscosity": AnalyticField(interval, viscosity(layer))})
    model = Model(geometry, fields, name=f"{args.model} viscosity")

    out = outside_source(args.out)
    out.mkdir(parents=True, exist_ok=True)
    path = modelfile.write_model(out / f"{args.model}_viscosity.nc", model,
                                 grid=gauss_legendre(args.lmax))
    print(path)


if __name__ == "__main__":
    main()
