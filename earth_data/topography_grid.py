#!/usr/bin/env python3
"""Export an ICE-NG topography grid and its pre-processed shoreline
for mesh refinement.

The first step of the real-coastline mesh chain
(doc/planning/planetmodel_coastline_sizing_plan.md): rather than
isolating coastline polylines, hand the mesher a smooth field to size
elements by. Two fields go in the .npz, each a wrapped lat-lon grid
(poles and the 360-degree column included, so sampling needs no seam
handling):

  topo_m      the topography truncated at --lmax, for maps and for any
              use that wants the (smoothed) field itself.
  coast_rad   the shoreline, pre-processed: the great-circle distance
              (radians) to the coast set of the --shore-lmax
              topography — the cells where it changes sign, plus the
              ocean within --band metres of the surface, the shallow
              shelves where the flotation criterion is delicate. The
              raw field's zero contour has wildly varying gradients
              (cliff coasts vs shelf seas), so no band on its value
              picks the coast out evenly; the distance field has unit
              gradient everywhere, and meshes/earth_coastlines.py
              refines on it directly.

Reads the data through pyslfp (its Zenodo downloader caches it on
first use).

Options (defaults in brackets):
  -o/--out       output file [topography.npz].
  --version      ICE7G, ICE6G or ICE5G [ICE7G].
  --date         ka BP [0: the present day].
  --lmax         truncation degree of topo_m [24].
  --shore-lmax   resolution the coastline is defined at [180, about
                 the data's native degree].
  --band         ocean within this depth counts as coast [400 m].
  --plot         quick-look map PNG beside the output.

Sample run:
  ./topography_grid --lmax 64 --plot
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np


def main() -> None:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("-o", "--out", type=Path, default=Path("topography.npz"))
    p.add_argument("--version", default="ICE7G",
                   choices=["ICE5G", "ICE6G", "ICE7G"])
    p.add_argument("--date", type=float, default=0.0)
    p.add_argument("--lmax", type=int, default=24)
    p.add_argument("--shore-lmax", type=int, default=180)
    p.add_argument("--band", type=float, default=400.0)
    p.add_argument("--plot", action="store_true")
    args = p.parse_args()

    from pyslfp.ice import IceNG
    model = IceNG(version=args.version)
    _, topo = model.get_ice_thickness_and_topography(args.date, args.lmax)

    # The shoreline, pre-processed at (near-)native resolution so that
    # it is uniformly defined everywhere — the lmax-truncated field's
    # zero contour has wildly varying gradients (sharp at steep
    # margins, smeared over shelves), so neither a band on its value
    # nor one on a first-order distance picks out the coast evenly.
    # The coast set is where the full-resolution topography changes
    # sign, plus the ocean within --band metres of the surface (the
    # shallow shelves, where the flotation criterion is delicate);
    # coast_rad is each grid point's great-circle distance to it, a
    # field with unit gradient that meshes/earth_coastlines.py refines
    # on directly.
    from scipy.spatial import cKDTree
    _, hi = model.get_ice_thickness_and_topography(args.date,
                                                   args.shore_lmax)
    t = hi.data
    land = t > 0.0
    coast = (~land) & (t >= -args.band)
    coast[:-1, :] |= land[:-1, :] != land[1:, :]
    coast[1:, :] |= land[1:, :] != land[:-1, :]
    coast[:, :-1] |= land[:, :-1] != land[:, 1:]
    coast[:, 1:] |= land[:, 1:] != land[:, :-1]
    lat_r = np.radians(hi.lats())[:, None]
    lon_r = np.radians(hi.lons())[None, :]
    xyz = np.stack(np.broadcast_arrays(np.cos(lat_r) * np.cos(lon_r),
                                       np.cos(lat_r) * np.sin(lon_r),
                                       np.sin(lat_r) + 0.0 * lon_r),
                   axis=-1)
    chord, _ = cKDTree(xyz[coast]).query(xyz.reshape(-1, 3))
    coast_rad = (2.0 * np.arcsin(np.clip(0.5 * chord, 0.0, 1.0))
                 ).reshape(t.shape).astype(np.float32)

    np.savez_compressed(args.out, lats=topo.lats(), lons=topo.lons(),
                        topo_m=topo.data,
                        coast_lats=hi.lats().astype(np.float32),
                        coast_lons=hi.lons().astype(np.float32),
                        coast_rad=coast_rad,
                        meta=np.array([args.date, args.lmax],
                                      dtype=float),
                        band_m=np.array(args.band, dtype=float),
                        version=np.array(args.version))
    land_frac = float((topo.data > 0).mean())
    print(f"{args.out}: {topo.data.shape[0]} x {topo.data.shape[1]} grid, "
          f"{args.version} at {args.date:g} ka BP, lmax {args.lmax}, "
          f"land fraction {land_frac:.2f}; coastline at lmax "
          f"{args.shore_lmax} ({t.shape[0]} x {t.shape[1]}, shelf band "
          f"{args.band:g} m)")

    if args.plot:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(9, 4.5))
        lim = np.max(np.abs(topo.data))
        m = ax.pcolormesh(topo.lons(), topo.lats(), topo.data,
                          cmap="BrBG_r", vmin=-lim, vmax=lim,
                          rasterized=True)
        ax.contour(topo.lons(), topo.lats(), topo.data, levels=[0.0],
                   colors="#0b0b0b", linewidths=0.6)
        fig.colorbar(m, ax=ax, label="topography (m)")
        ax.set_xlabel("longitude")
        ax.set_ylabel("latitude")
        ax.set_title(f"{args.version}, {args.date:g} ka BP, "
                     f"lmax {args.lmax}")
        png = args.out.with_suffix(".png")
        fig.tight_layout()
        fig.savefig(png, dpi=150)
        plt.close(fig)
        print(f"{png}: quick-look map")


if __name__ == "__main__":
    main()
