#!/usr/bin/env python3
"""Draw the map A from fault slip to surface displacement and its adjoint
A*, from the map tables of coseismic_fault.cpp.

Run from the build's examples/ directory, where the example writes its
tables and the build copies this script:

    python3 plot_fault_map.py coseismic_fault        # writes coseismic_fault_map.png
    python3 plot_fault_map.py coseismic_fault --show

It reads <prefix>_map_fault.csv and <prefix>_map_surface.csv and draws one
row for the map and one for each observable, the fault on the left and the
surface on the right:

    fault slip s                --A-->    surface displacement A s
    sensitivity kernel A* t_k  <--A*--    point traction t_k = d_k delta(x - x_k)

The observable J_k = d_k . u(x_k) is the displacement at the station x_k
along d_k; its traction is drawn as an arrow along d_k at the station. In
2-D the fields are curves: along the fault against the distance down dip
from its centre (the slip and the kernels along the slip), along the
surface against x (the fault's horizontal extent shaded, the stations
dotted), and each traction on a section through the surface and the
fault. In 3-D they are maps: the fault in its own plane seen from the
hanging wall (along strike, up dip), colour the component along the slip
and arrows the vector in the plane; the surface from above, colour the
vertical component and arrows the horizontal, the fault's outline dashed
and its up-dip edge solid; a traction's horizontal part is an arrow, its
vertical part a circled dot (up) or cross (down), over contours of the
surface height. In 3-D the surface panels are cut to where the fields
and stations are (--extent sets their half-width in either dimension, 0
the whole surface). Needs numpy and matplotlib.
"""
import argparse
from pathlib import Path

# The categorical slots of the house palette (the first two), and the ink
# of axes, outlines and arrows.
BLUE, ORANGE = "#2a78d6", "#eb6834"
INK, MUTED = "#0b0b0b", "#8a8984"

# The length of a traction arrow, in points.
ARROW = 42


def read(path):
    """The "# key: value" lines and the columns of a table, as arrays."""
    import numpy as np

    meta, lines = {}, []
    with open(path) as f:
        for line in f:
            if line.startswith("#"):
                key, _, value = line[1:].partition(":")
                meta[key.strip()] = value.strip()
            elif line.strip():
                lines.append(line)
    header = lines[0].strip().split(",")
    body = np.array([[float(v) for v in l.split(",")] for l in lines[1:]])
    return meta, {name: body[:, i] for i, name in enumerate(header)}


def numbers(text):
    return [float(v) for v in text.split()]


def outline(meta):
    """The fault seen from above: its corners (3-D) or ends (2-D)."""
    return [numbers(p) for p in meta["fault_outline"].split(";")]


def observables(meta):
    """The stations, directions and J (forward, adjoint) of the header."""
    found, k = [], 1
    while f"station_{k}" in meta:
        found.append({"station": numbers(meta[f"station_{k}"]),
                      "direction": numbers(meta[f"direction_{k}"]),
                      "J": numbers(meta[f"J_{k}"])})
        k += 1
    return found


def surface_extent(columns, corners, stations, half_width):
    """The half-width of the surface panels about the origin (None: the
    whole surface, the default in 2-D); in 3-D by default where the
    vertical displacement exceeds 10% of its peak, with the fault, the
    stations and a margin."""
    import numpy as np

    if half_width is not None:
        return half_width if half_width > 0 else None
    if "y" not in columns:
        return None
    planar = ("x", "y")
    u = np.abs(columns["u_z"])
    near = u > 0.1 * u.max()
    reach = max(np.abs(columns[c][near]).max() for c in planar)
    reach = max([reach] + [abs(v) for p in corners for v in p]
                + [abs(v) for s in stations for v in s[:2]])
    whole = max(np.abs(columns[c]).max() for c in planar)
    return min(1.2 * reach, whole)


def triangulation(a, b, extent=None):
    """The Delaunay triangulation of the distinct points (a, b) (the element
    samples repeat on element boundaries), masked to the panel; and the
    indices of those points."""
    import matplotlib.tri as mtri
    import numpy as np

    scale = max(np.ptp(a), np.ptp(b), 1.0)
    keys = np.round(np.column_stack([a, b]) / scale, 9)
    _, keep = np.unique(keys, axis=0, return_index=True)
    keep = np.sort(keep)
    tri = mtri.Triangulation(a[keep], b[keep])
    if extent is not None:
        inside = ((np.abs(a[keep]) <= 1.05 * extent)
                  & (np.abs(b[keep]) <= 1.05 * extent))
        tri.set_mask(~inside[tri.triangles].all(axis=1))
    return tri, keep


def curve(ax, s, values, label, colour):
    import numpy as np

    order = np.argsort(s, kind="stable")
    ax.plot(s[order], values[order], color=colour, lw=2, label=label)


def map_panel(ax, a, b, colour, arrows, cmap, symmetric, label, extent=None):
    """A field on a planar domain: colour filled contours of one component,
    arrows of the vector (u, v) on a regular grid."""
    import matplotlib.tri as mtri
    import numpy as np
    from matplotlib.ticker import MaxNLocator

    tri, keep = triangulation(a, b, extent)
    a, b, colour = a[keep], b[keep], colour[keep]
    peak = np.abs(colour).max() or 1.0
    levels = (np.linspace(-peak, peak, 21) if symmetric
              else np.linspace(min(colour.min(), 0.0), colour.max() or 1.0, 21))
    filled = ax.tricontourf(tri, colour, levels=levels, cmap=cmap)
    if arrows is not None:
        u, v = arrows[0][keep], arrows[1][keep]
        lo_a, hi_a = (-extent, extent) if extent else (a.min(), a.max())
        lo_b, hi_b = (-extent, extent) if extent else (b.min(), b.max())
        ga, gb = np.meshgrid(np.linspace(lo_a, hi_a, 15)[1:-1],
                             np.linspace(lo_b, hi_b, 15)[1:-1])
        gu = mtri.LinearTriInterpolator(tri, u)(ga, gb)
        gv = mtri.LinearTriInterpolator(tri, v)(ga, gb)
        ax.quiver(ga, gb, gu, gv, color=INK, alpha=0.8, width=0.004,
                  pivot="mid")
    if extent is not None:
        ax.set_xlim(-extent, extent)
        ax.set_ylim(-extent, extent)
    ax.set_aspect("equal")
    bar = ax.figure.colorbar(filled, ax=ax, orientation="horizontal",
                             fraction=0.05, pad=0.14, label=label)
    bar.locator = MaxNLocator(5, symmetric=symmetric)
    bar.formatter.set_powerlimits((-2, 3))
    bar.update_ticks()


def traction_arrow(ax, point, direction):
    """An arrow of fixed length on the page from the point along the
    direction (its components in the panel's axes), whatever the aspect."""
    ax.annotate("", xy=point, xytext=(ARROW * direction[0], ARROW * direction[1]),
                textcoords="offset points", annotation_clip=False,
                arrowprops=dict(arrowstyle="<|-", lw=2, color=BLUE,
                                mutation_scale=16, shrinkA=0, shrinkB=0))


def station_marker(ax, point, label):
    ax.plot(*point, "o", ms=7, mfc="white", mec=INK, mew=1.5, zorder=4)
    ax.annotate(label, point, xytext=(6, 6), textcoords="offset points",
                fontsize=10, color=INK)


def scientific(v):
    """v as mathtext, a x 10^n."""
    mantissa, _, exponent = f"{v:.3e}".partition("e")
    return f"{mantissa} \\times 10^{{{int(exponent)}}}"


def j_label(ax, k, J):
    ax.text(0.02, 0.96, f"$J_{k} = d_{k}\\cdot u(x_{k}) = {scientific(J[0])}$",
            transform=ax.transAxes, va="top", fontsize=9,
            bbox=dict(boxstyle="round", fc="white", ec=MUTED, alpha=0.9))


def arrow(ax, text, leftwards):
    ax.set_axis_off()
    start, end = ((0.9, 0.5), (0.1, 0.5)) if leftwards else ((0.1, 0.5), (0.9, 0.5))
    ax.annotate("", xy=end, xytext=start, xycoords="axes fraction",
                arrowprops=dict(arrowstyle="-|>", lw=1.8, color=INK,
                                mutation_scale=18))
    ax.text(0.5, 0.58, text, ha="center", va="bottom", fontsize=14,
            transform=ax.transAxes)


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("prefix", help="the example's -csv prefix (coseismic_fault)")
    parser.add_argument("--extent", type=float,
                        help="half-width of the surface panels (0: the whole "
                             "surface; default: in 3-D where the fields are, "
                             "in 2-D the whole surface)")
    parser.add_argument("--out", type=Path,
                        help="figure file (default: <prefix>_map.png)")
    parser.add_argument("--show", action="store_true", help="open a window as well")
    args = parser.parse_args()

    import matplotlib
    if not args.show:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    fault_meta, fault = read(Path(args.prefix + "_map_fault.csv"))
    surface_meta, surface = read(Path(args.prefix + "_map_surface.csv"))
    three_d = "y" in surface
    corners = outline(surface_meta)
    obs = observables(surface_meta)
    extent = surface_extent(surface, corners, [o["station"] for o in obs],
                            args.extent)

    rows = 1 + len(obs)
    fig = plt.figure(figsize=(12, (4.7 if three_d else 3.7) * rows + 0.4))
    grid = fig.add_gridspec(rows, 3, width_ratios=[1, 0.16, 1])
    ax = {(r, c): fig.add_subplot(grid[r, c]) for r in range(rows) for c in (0, 2)}
    arrow(fig.add_subplot(grid[0, 1]), "$A$", leftwards=False)
    ax[0, 0].set_title("fault slip $s$", fontsize=11)
    ax[0, 2].set_title("surface displacement $A s$", fontsize=11)
    for k in range(1, rows):
        arrow(fig.add_subplot(grid[k, 1]), "$A^*$", leftwards=True)
        ax[k, 2].set_title(f"point traction $t_{k} = d_{k}\\,\\delta(x - x_{k})$",
                           fontsize=11)
        ax[k, 0].set_title(f"sensitivity kernel $A^* t_{k}$", fontsize=11)

    if not three_d:
        d = fault["down_dip"]
        curve(ax[0, 0], d, fault["slip"], None, BLUE)
        ax[0, 0].set_ylabel("slip along the slip direction")
        x = surface["x"]
        curve(ax[0, 2], x, surface["u_z"], "$u_z$", BLUE)
        curve(ax[0, 2], x, surface["u_x"], "$u_x$", ORANGE)
        ends = [p[0] for p in corners]
        ax[0, 2].axvspan(min(ends), max(ends), color=MUTED, alpha=0.18, lw=0,
                         label="fault, from above")
        for k, o in enumerate(obs, start=1):
            ax[0, 2].axvline(o["station"][0], color=INK, ls=":", lw=1)
            ax[0, 2].annotate(f"$x_{k}$", (o["station"][0], 1.0),
                              xycoords=("data", "axes fraction"),
                              xytext=(3, -12), textcoords="offset points",
                              fontsize=10)
            curve(ax[k, 0], d, fault[f"kernel_{k}"], None, BLUE)
            ax[k, 0].set_ylabel("kernel along the slip direction")
            # The traction on a section through the surface and the fault.
            section = ax[k, 2]
            curve(section, x, surface["z"], "surface", MUTED)
            curve(section, fault["x"], fault["z"], "fault", INK)
            station_marker(section, o["station"], f"$x_{k}$")
            traction_arrow(section, o["station"], o["direction"])
            j_label(section, k, o["J"])
            section.set_ylabel("z")
            section.set_ylim(fault["z"].min() - 0.4,
                             surface["z"].max() + 0.9)
            section.legend(fontsize=8, frameon=False, loc="lower right")
        for r in range(rows):
            ax[r, 2].set_xlabel("x")
            if extent is not None:
                ax[r, 2].set_xlim(-extent, extent)
            else:
                ax[r, 2].set_xlim(x.min(), x.max())
            ax[r, 0].set_xlabel("distance down dip from the fault centre")
        for r in range(rows):
            for a in (ax[r, 0], ax[r, 2] if r == 0 else None):
                if a is None:
                    continue
                a.axhline(0.0, color=MUTED, lw=0.8)
                a.ticklabel_format(axis="y", style="sci", scilimits=(-2, 3))
            for a in (ax[r, 0], ax[r, 2]):
                a.grid(True, alpha=0.3)
        ax[0, 2].legend(fontsize=8, frameon=False, loc="lower left")
    else:
        a, b = fault["along_strike"], fault["up_dip"]
        map_panel(ax[0, 0], a, b, fault["slip"],
                  (fault["slip_strike"], fault["slip_up_dip"]), "Blues", False,
                  "slip along the slip direction")
        x, y = surface["x"], surface["y"]
        map_panel(ax[0, 2], x, y, surface["u_z"], (surface["u_x"], surface["u_y"]),
                  "RdBu_r", True, "$u_z$; arrows $(u_x, u_y)$", extent)
        tri, keep = triangulation(x, y, extent)
        # The corners run from the up-dip edge round; the last side closes
        # the ring along that edge, drawn solid.
        ring = np.array(corners + corners[:1])
        for r in range(rows):
            ax[r, 2].plot(ring[:-1, 0], ring[:-1, 1], "--", color=INK, lw=1,
                          label="fault, from above")
            ax[r, 2].plot(ring[-2:, 0], ring[-2:, 1], "-", color=INK, lw=2,
                          label="its up-dip edge")
            ax[r, 2].set_xlabel("x")
            ax[r, 2].set_ylabel("y")
            ax[r, 0].set_xlabel("along strike")
            ax[r, 0].set_ylabel("up dip")
        ax[0, 2].legend(fontsize=8, loc="upper right", framealpha=0.85)
        for k, o in enumerate(obs, start=1):
            station_marker(ax[0, 2], o["station"][:2], f"$x_{k}$")
            name = f"kernel_{k}"
            map_panel(ax[k, 0], a, b, fault[name],
                      (fault[name + "_strike"], fault[name + "_up_dip"]),
                      "RdBu_r", True, "kernel along the slip direction")
            # The traction from above, over contours of the surface height:
            # its horizontal part an arrow, its vertical part a circled dot
            # (up) or cross (down).
            plan = ax[k, 2]
            plan.tricontour(tri, surface["z"][keep], levels=8, colors=MUTED,
                            linewidths=0.6, linestyles="solid")
            point, (dx, dy, dz) = o["station"][:2], o["direction"]
            if abs(dz) > 1e-9:
                plan.plot(*point, "o", ms=16, mfc="white", mec=BLUE, mew=2,
                          zorder=4)
                plan.plot(*point, "." if dz > 0 else "x", ms=8 if dz > 0 else 9,
                          color=BLUE, mew=2, zorder=5)
            else:
                station_marker(plan, point, "")
            if np.hypot(dx, dy) > 1e-9:
                traction_arrow(plan, point, (dx, dy))
            plan.annotate(f"$x_{k}$", point, xytext=(10, -16),
                          textcoords="offset points", fontsize=10)
            j_label(plan, k, o["J"])
            plan.text(0.02, 0.03, "grey: contours of the surface height",
                      transform=plan.transAxes, fontsize=8, color=INK,
                      bbox=dict(boxstyle="round", fc="white", ec="none",
                                alpha=0.85))
            if extent is not None:
                plan.set_xlim(-extent, extent)
                plan.set_ylim(-extent, extent)
            plan.set_aspect("equal")
            # A blank bar keeps the panel the size of its neighbours.
            spacer = fig.colorbar(plt.cm.ScalarMappable(), ax=plan,
                                  orientation="horizontal", fraction=0.05,
                                  pad=0.14)
            spacer.ax.set_visible(False)

    title = fault_meta.get("title", args.prefix)
    note = [fault_meta.get("note", "")]
    note += [f"$J_{k}$ forward {o['J'][0]:.6g}, adjoint {o['J'][1]:.6g}"
             for k, o in enumerate(obs, start=1)]
    fig.suptitle(title + "\n" + "; ".join(n for n in note if n), fontsize=10)
    fig.tight_layout()

    out = args.out or Path(args.prefix + "_map.png")
    fig.savefig(out, dpi=150)
    print(f"Wrote {out}")
    if args.show:
        plt.show()


if __name__ == "__main__":
    main()
