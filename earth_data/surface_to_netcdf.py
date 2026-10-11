#!/usr/bin/env python3
"""Grid an exported surface field onto a regular lon-lat array.

Reads the CSV that SeaLevelOperator::WriteSurfaceField writes (header
`x,y,z,value`; one row per surface node, exact nodal data) and
interpolates it onto a regular longitude-latitude grid:

  - latitudes from +90 to -90 (nlat rows, endpoints included),
  - longitudes from 0 to 360 (nlon columns, 360 excluded),

which is the pyshtools DH convention up to its row count, so
`pyshtools.SHGrid.from_array(grid[:-1, :], grid="DH")` (dropping the
south-pole row for an even row count) and cartopy's regular-grid
plotting both take it directly.

The interpolation is a local thin-plate RBF in the unit-direction
coordinates (scipy RBFInterpolator on the 3-D node directions), so the
poles and the date line need no special casing and the grid points —
which lie outside the convex hull of the node directions — pose no
extrapolation problem.

Output: a NetCDF file (netCDF4; falls back to a .npz beside it with a
warning when netCDF4 is missing) with variables lat, lon, value.
With --plot, a quick-look map: cartopy (Robinson) when available,
plain matplotlib otherwise.

Options (defaults in brackets):
  csv          the exported surface CSV (positional).
  -o/--out     output file [the CSV with .nc extension].
  --nlat       latitude rows [181].
  --nlon       longitude columns [360].
  --name       variable name in the file [the CSV stem].
  --plot       also write a quick-look PNG beside the output.

Sample run:
  python surface_to_netcdf.py sea_level.csv --plot
"""
from __future__ import annotations

import argparse
import sys
import warnings
from pathlib import Path

import numpy as np
from scipy.interpolate import RBFInterpolator


def read_nodes(path: Path):
    data = np.genfromtxt(path, delimiter=",", names=True)
    names = data.dtype.names
    if names is None or "value" not in names or "z" not in names:
        raise SystemExit(f"{path}: expected the x,y,z,value header of "
                         "WriteSurfaceField on a 3-D surface (a 2-D "
                         "x,y,value export is a curve, not a map)")
    xyz = np.column_stack([data["x"], data["y"], data["z"]])
    return xyz, np.asarray(data["value"])


def grid(xyz: np.ndarray, values: np.ndarray, nlat: int, nlon: int):
    # Unit directions of the nodes and of the target grid; linear
    # interpolation in R^3 restricted to the sphere handles the poles
    # and the date line without special cases.
    d = xyz / np.linalg.norm(xyz, axis=1, keepdims=True)
    lats = np.linspace(90.0, -90.0, nlat)
    lons = np.arange(nlon) * (360.0 / nlon)
    glat, glon = np.meshgrid(np.deg2rad(lats), np.deg2rad(lons),
                             indexing="ij")
    target = np.column_stack([
        (np.cos(glat) * np.cos(glon)).ravel(),
        (np.cos(glat) * np.sin(glon)).ravel(),
        np.sin(glat).ravel(),
    ])
    neighbors = min(64, values.size)
    rbf = RBFInterpolator(d, values, neighbors=neighbors,
                          kernel="thin_plate_spline")
    out = rbf(target)
    return lats, lons, out.reshape(nlat, nlon)


def write(out_path: Path, name: str, lats, lons, field) -> Path:
    try:
        import netCDF4
    except ImportError:
        npz = out_path.with_suffix(".npz")
        warnings.warn(f"netCDF4 not installed; writing {npz} instead")
        np.savez(npz, lat=lats, lon=lons, **{name: field})
        return npz
    with netCDF4.Dataset(out_path, "w") as nc:
        nc.createDimension("lat", lats.size)
        nc.createDimension("lon", lons.size)
        vlat = nc.createVariable("lat", "f8", ("lat",))
        vlon = nc.createVariable("lon", "f8", ("lon",))
        v = nc.createVariable(name, "f8", ("lat", "lon"))
        vlat[:] = lats
        vlon[:] = lons
        v[:, :] = field
        vlat.units = "degrees_north"
        vlon.units = "degrees_east"
        nc.source = "AdGIA SeaLevelOperator::WriteSurfaceField"
    return out_path


def quick_look(png: Path, name: str, lats, lons, field) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    try:
        import cartopy.crs as ccrs
        fig = plt.figure(figsize=(8, 4.5))
        ax = fig.add_subplot(projection=ccrs.Robinson())
        m = ax.pcolormesh(lons, lats, field,
                          transform=ccrs.PlateCarree(), cmap="RdBu_r")
        ax.coastlines(linewidth=0.5)
    except ImportError:
        fig, ax = plt.subplots(figsize=(8, 4.5))
        m = ax.pcolormesh(lons, lats, field, cmap="RdBu_r")
        ax.set_xlabel("longitude")
        ax.set_ylabel("latitude")
    fig.colorbar(m, ax=ax, shrink=0.7, label=name)
    fig.tight_layout()
    fig.savefig(png, dpi=150)
    print(f"wrote {png}")


def main() -> None:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("csv", type=Path)
    p.add_argument("-o", "--out", type=Path, default=None)
    p.add_argument("--nlat", type=int, default=181)
    p.add_argument("--nlon", type=int, default=360)
    p.add_argument("--name", default=None)
    p.add_argument("--plot", action="store_true")
    args = p.parse_args()

    name = args.name or args.csv.stem
    out_path = args.out or args.csv.with_suffix(".nc")
    xyz, values = read_nodes(args.csv)
    lats, lons, field = grid(xyz, values, args.nlat, args.nlon)
    written = write(out_path, name, lats, lons, field)
    print(f"wrote {written} ({args.nlat} x {args.nlon}, "
          f"{values.size} nodes)")
    if args.plot:
        quick_look(written.with_suffix(".png"), name, lats, lons, field)


if __name__ == "__main__":
    sys.exit(main())
