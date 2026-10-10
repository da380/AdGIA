#!/usr/bin/env python3
"""Export a truncated ICE-NG topography grid for mesh refinement.

The first step of the real-coastline mesh chain
(doc/planning/planetmodel_coastline_sizing_plan.md): rather than
isolating coastline polylines, hand the mesher the topography itself
and let it size elements by how close the sea level is to zero —
shorelines AND shallow shelves, where the flotation criterion is
delicate, attract refinement together.

Reads the ICE-NG topography at a date through pyslfp (its Zenodo
downloader caches the data on first use), truncates it at --lmax —
which smooths the field to a resolution a coarse mesh can follow — and
writes the wrapped lat-lon grid (poles and the 360-degree column
included, so sampling needs no seam handling) to an .npz that
meshes/earth_coastlines.py turns into a planetmodel Refinement.

Options (defaults in brackets):
  -o/--out       output file [topography.npz].
  --version      ICE7G, ICE6G or ICE5G [ICE7G].
  --date         ka BP [0: the present day].
  --lmax         truncation degree; lower = smoother shorelines [24].
  --plot         quick-look map PNG beside the output.

Sample run:
  ./topography_grid --lmax 24 --plot
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
    p.add_argument("--plot", action="store_true")
    args = p.parse_args()

    from pyslfp.ice import IceNG
    model = IceNG(version=args.version)
    _, topo = model.get_ice_thickness_and_topography(args.date, args.lmax)

    np.savez_compressed(args.out, lats=topo.lats(), lons=topo.lons(),
                        topo_m=topo.data,
                        meta=np.array([args.date, args.lmax],
                                      dtype=float),
                        version=np.array(args.version))
    land = float((topo.data > 0).mean())
    print(f"{args.out}: {topo.data.shape[0]} x {topo.data.shape[1]} grid, "
          f"{args.version} at {args.date:g} ka BP, lmax {args.lmax}, "
          f"land fraction {land:.2f}")

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
