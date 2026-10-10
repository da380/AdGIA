#!/usr/bin/env python3
"""Sample an ice-ng loading history at exported surface nodes.

The ingest leg of the surface exchange format (WP6 of
doc/planning/sea_level_plan.md): reads the CSV that
SeaLevelOperator::WriteSurfaceField writes (the node set; the value
column is ignored), samples an ICE-5G/6G/7G field at those nodes for a
list of dates, and writes a time-stack CSV that IceHistory reads —
header `x,y,z,<t0>,<t1>,...` with one value column per time, the
columns named by ascending model time.

The data files are pyslfp's: its Zenodo downloader fetches and caches
them in its own data directory on first use (nothing is copied here),
and its IceNG loader does the per-date file resolution and the
interpolation between the model's time slices. Each date's field comes
back on a wrapped DH grid (poles and the 360-degree column included),
so sampling at the nodes is plain bilinear interpolation in latitude
and longitude with no seam or pole casing.

Times and units: dates are in ka BP, oldest first; the column times are
model times t_k = (date_0 - date_k) / time-scale, ascending from zero,
so a run steps forward from the oldest date. Lengths are divided by
--length-scale (pyslfp's own non-dimensionalisation), so pass the case
length scale in metres for a non-dimensional run.

Options (defaults in brackets):
  csv              the exported surface CSV, the node set (positional).
  -o/--out         output CSV [<csv stem>_<field>.csv].
  --version        ICE7G, ICE6G or ICE5G [ICE7G].
  --field          ice, topography or sea_level [ice].
  --dates          comma list, ka BP, oldest first [21,15,10,5,0].
  --lmax           grid truncation of the ice model's fields [180].
  --length-scale   metres per length unit of the output [1.0].
  --time-scale     ka per time unit of the column times [1.0].

Sample run (after the C++ side exported surface nodes):
  ./ice_ng_to_surface surface_nodes.csv --dates 21,15,10,5,0 \\
      --length-scale 6.371e6
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from scipy.interpolate import RegularGridInterpolator


def read_nodes(path: Path) -> np.ndarray:
    data = np.genfromtxt(path, delimiter=",", names=True)
    names = data.dtype.names
    if names is None or "z" not in names:
        raise SystemExit(f"{path}: expected the x,y,z,... header of "
                         "WriteSurfaceField on a 3-D surface (ice-ng "
                         "data is a map; a 2-D export is a curve)")
    return np.column_stack([data["x"], data["y"], data["z"]])


def sample(grid, lats_deg: np.ndarray, lons_deg: np.ndarray) -> np.ndarray:
    """Bilinear sample of a wrapped pyshtools DH grid at points."""
    lats = grid.lats()   # descending from +90, poles included
    lons = grid.lons()   # 0 .. 360 inclusive (extend=True)
    interp = RegularGridInterpolator(
        (lats[::-1], lons), grid.data[::-1, :],
        bounds_error=False, fill_value=None)
    return interp(np.column_stack([lats_deg, lons_deg]))


def main() -> None:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("csv", type=Path)
    p.add_argument("-o", "--out", type=Path, default=None)
    p.add_argument("--version", default="ICE7G",
                   choices=["ICE5G", "ICE6G", "ICE7G"])
    p.add_argument("--field", default="ice",
                   choices=["ice", "topography", "sea_level"])
    p.add_argument("--dates", default="21,15,10,5,0")
    p.add_argument("--lmax", type=int, default=180)
    p.add_argument("--length-scale", type=float, default=1.0)
    p.add_argument("--time-scale", type=float, default=1.0)
    args = p.parse_args()

    dates = [float(tok) for tok in args.dates.split(",") if tok.strip()]
    if not dates or any(b >= a for a, b in zip(dates, dates[1:])):
        raise SystemExit("--dates must be a descending ka-BP list, "
                         "oldest first")
    times = [(dates[0] - d) / args.time_scale for d in dates]

    xyz = read_nodes(args.csv)
    r = np.linalg.norm(xyz, axis=1)
    lats_deg = np.degrees(np.arcsin(np.clip(xyz[:, 2] / r, -1.0, 1.0)))
    lons_deg = np.degrees(np.arctan2(xyz[:, 1], xyz[:, 0])) % 360.0

    from pyslfp.ice import IceNG
    model = IceNG(version=args.version, length_scale=args.length_scale)

    columns = []
    for date in dates:
        if args.field == "sea_level":
            _, field = model.get_ice_thickness_and_sea_level(
                date, args.lmax)
        else:
            ice, topo = model.get_ice_thickness_and_topography(
                date, args.lmax)
            field = ice if args.field == "ice" else topo
        columns.append(sample(field, lats_deg, lons_deg))

    out = args.out or args.csv.with_name(
        f"{args.csv.stem}_{args.field}.csv")
    with open(out, "w") as f:
        f.write("x,y,z," + ",".join(f"{t:.10g}" for t in times) + "\n")
        for i in range(xyz.shape[0]):
            row = [f"{v:.16g}" for v in xyz[i]]
            row += [f"{c[i]:.16g}" for c in columns]
            f.write(",".join(row) + "\n")
    print(f"{out}: {xyz.shape[0]} nodes, {len(dates)} times "
          f"({args.version} {args.field}; model times "
          f"{times[0]:g} .. {times[-1]:g})")


if __name__ == "__main__":
    main()
