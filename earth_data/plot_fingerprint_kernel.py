#!/usr/bin/env python3
"""The reciprocity-kernel figure of the sea-level fingerprint example.

Takes one or two kernel CSVs exported by
`sea_level_fingerprint -kernel` (typically the 1-D model and a -lateral
3-D run of the same observation site) and draws the kernel MASKED TO
THE PRESENT-DAY ICE SHEETS — where it reads as the sensitivity of the
sea-level observation to ice melt (scaled by rho_i) — as a global map
plus zoom panels over Greenland and Antarctica, with the observation
site starred. With two inputs the second column is, by default, the
DIFFERENCE (second minus first) on its own scale: the 3-D-structure
signature is a small fraction of the kernel's spread (sub-percent at
30 % lateral moduli) and is invisible on a shared scale;
--side-by-side draws the two kernels instead. The kernels are the raw
L2 fields; the Gaussian observation cap is the only smoothing.

The gridding is surface_to_netcdf's (RBF on the node directions); the
ice mask is pyslfp's present-day ICE-7G thickness (its Zenodo
downloader caches the data); the maps are pyslfp's cartopy wrapper.

Usage:
  plot_fingerprint_kernel kernel.csv [kernel_3d.csv]
      [--titles "1-D model" "3-D model"] [--out fingerprint_kernel.png]
      [--obs-lat 42.36 --obs-lon -71.06]
      [--nlat 181 --nlon 360] [--ice-lmax 180] [--version ICE7G]
      [--ice-threshold 1.0]  (metres of present-day ice)
      [--no-mask]            (toy geographies: draw unmasked maps only)

Run from the build tree as <build>/earth_data/plot_fingerprint_kernel;
the figure lands there.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import surface_to_netcdf as s2n  # noqa: E402  (sibling script as module)


def ice_mask(lats: np.ndarray, lons: np.ndarray, version: str, lmax: int,
             threshold: float) -> np.ndarray:
    """Present-day ice thickness > threshold on the (lat, lon) grid."""
    from scipy.interpolate import RegularGridInterpolator

    from pyslfp.ice import IceNG

    model = IceNG(version=version)
    ice, _ = model.get_ice_thickness_and_topography(0.0, lmax)
    glats = np.asarray(ice.lats())
    glons = np.asarray(ice.lons())
    data = np.asarray(ice.data)
    order = np.argsort(glats)
    interp = RegularGridInterpolator(
        (glats[order], glons), data[order, :],
        bounds_error=False, fill_value=0.0)
    glat, glon = np.meshgrid(lats, lons % 360.0, indexing="ij")
    thickness = interp(np.column_stack([glat.ravel(), glon.ravel()]))
    return thickness.reshape(len(lats), len(lons)) > threshold


def to_shgrid(lats: np.ndarray, field: np.ndarray):
    """Wrap the gridded field as a pyshtools SHGrid for pyslfp.plot
    (DH sampling: drop the south-pole row of the inclusive grid)."""
    import pyshtools

    assert lats[0] == 90.0 and lats[-1] == -90.0
    return pyshtools.SHGrid.from_array(field[:-1, :], grid="DH")


def main() -> None:
    p = argparse.ArgumentParser(
        description="Reciprocity-kernel maps, ice-masked, via pyslfp.")
    p.add_argument("csv", nargs="+", type=Path,
                   help="kernel CSVs (1-D model first, 3-D second)")
    p.add_argument("--titles", nargs="+", default=None)
    p.add_argument("--out", type=Path, default=Path("fingerprint_kernel.png"))
    p.add_argument("--obs-lat", type=float, default=42.36)
    p.add_argument("--obs-lon", type=float, default=-71.06)
    p.add_argument("--nlat", type=int, default=361)
    p.add_argument("--nlon", type=int, default=720)
    p.add_argument("--version", default="ICE7G",
                   choices=["ICE5G", "ICE6G", "ICE7G"])
    p.add_argument("--ice-lmax", type=int, default=180)
    p.add_argument("--ice-threshold", type=float, default=1.0)
    p.add_argument("--no-mask", action="store_true",
                   help="skip the ice mask (toy geographies)")
    p.add_argument("--side-by-side", action="store_true",
                   help="two kernels on one scale instead of kernel + "
                        "difference")
    args = p.parse_args()
    if len(args.csv) > 2:
        raise SystemExit("at most two kernels (1-D and 3-D)")
    difference = len(args.csv) == 2 and not args.side_by_side
    if args.titles:
        titles = args.titles
    elif difference:
        titles = ["1-D model", "3-D $-$ 1-D"]
    else:
        titles = ["1-D model", "3-D model"][: len(args.csv)]

    import matplotlib

    matplotlib.use("Agg")
    import cartopy.crs as ccrs
    import matplotlib.pyplot as plt
    import pyslfp

    fields = []
    lats = lons = None
    for path in args.csv:
        xyz, values = s2n.read_nodes(path)
        lats, lons, field = s2n.grid(xyz, values, args.nlat, args.nlon)
        fields.append(field)

    if args.no_mask:
        mask = np.ones((args.nlat, args.nlon), dtype=bool)
    else:
        mask = ice_mask(lats, lons, args.version, args.ice_lmax,
                        args.ice_threshold)

    if difference:
        fields = [fields[0], fields[1] - fields[0]]
    masked = [np.where(mask, f, np.nan) for f in fields]

    # Per-column colour scales, robust over the masked values. The
    # kernel sits on the mass-conservation plateau -1/(rho_w A), so
    # its limits are percentiles (symmetric-about-zero would drown the
    # structure); a difference IS zero-centred, so its limits are
    # symmetric.
    def limits(m, symmetric):
        finite = m[np.isfinite(m)]
        if not finite.size:
            return -1.0, 1.0
        if symmetric:
            v = np.percentile(np.abs(finite), 99.0)
            return (-v, v) if v > 0 else (-1.0, 1.0)
        lo, hi = np.percentile(finite, [1.0, 99.0])
        return (lo, hi) if lo < hi else (lo - 1.0, hi + 1.0)

    if difference:
        scales = [limits(masked[0], False), limits(masked[1], True)]
    else:
        joint = np.concatenate([m[np.isfinite(m)] for m in masked])
        scales = [limits(joint[None, :], False)] * len(masked)

    # Panels: per model a global map plus Greenland and Antarctica zooms.
    ncols = len(fields)
    fig = plt.figure(figsize=(7.2 * ncols, 12.0), layout="constrained")
    gs = fig.add_gridspec(3, ncols, height_ratios=[1.6, 1.0, 1.0])
    pc = ccrs.PlateCarree()

    def panel(ax, field, vmin, vmax, zoom=None):
        g = to_shgrid(lats, field)
        pyslfp.plot(g, ax=ax, cmap="RdBu_r", coasts=True, gridlines=False,
                    colorbar=False, vmin=vmin, vmax=vmax)
        if zoom is not None:
            ax.set_extent(zoom, crs=pc)
        ax.plot(args.obs_lon, args.obs_lat, marker="*", markersize=10,
                color="k", transform=pc, zorder=5)
        return ax

    col_axes = [[] for _ in masked]
    for c, (field, title, (vmin, vmax)) in enumerate(
            zip(masked, titles, scales)):
        ax = fig.add_subplot(gs[0, c], projection=ccrs.Robinson())
        panel(ax, field, vmin, vmax)
        ax.set_global()
        ax.set_title(title)
        axg = fig.add_subplot(gs[1, c], projection=pc)
        panel(axg, field, vmin, vmax, zoom=[-75.0, -10.0, 58.0, 85.0])
        axg.set_title("Greenland")
        axa = fig.add_subplot(
            gs[2, c], projection=ccrs.SouthPolarStereo())
        panel(axa, field, vmin, vmax, zoom=[-180.0, 180.0, -90.0, -63.0])
        axa.set_title("Antarctica")
        col_axes[c] = [ax, axg, axa]

    labels = (["dSL(obs) / dzeta  (L2 kernel, ice-masked)",
               "kernel difference"] if difference else
              ["dSL(obs) / dzeta  (L2 kernel, ice-masked)"] * ncols)
    done = set()
    for c, ((vmin, vmax), label) in enumerate(zip(scales, labels)):
        key = (vmin, vmax)
        if not difference and key in done:
            continue
        done.add(key)
        sm = plt.cm.ScalarMappable(
            cmap="RdBu_r", norm=plt.Normalize(vmin=vmin, vmax=vmax))
        axes = (col_axes[c] if difference else
                [a for col in col_axes for a in col])
        fig.colorbar(sm, ax=axes, orientation="horizontal",
                     shrink=0.7 if difference else 0.5, pad=0.02,
                     label=label)
    fig.savefig(args.out, dpi=200)
    print(f"{args.out}: {ncols} column(s)"
          f"{' (kernel + difference)' if difference else ''}, "
          f"{'un' if args.no_mask else 'ice-'}masked; scales " +
          "; ".join(f"[{a:.3g}, {b:.3g}]" for a, b in scales))


if __name__ == "__main__":
    main()
