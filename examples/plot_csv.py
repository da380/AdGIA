#!/usr/bin/env python3
"""Plot the CSV tables the examples write (examples/visualisation.hpp).

Run from the build's examples/ directory, where the examples write their
tables and the build copies this script:

    python3 plot_csv.py love_numbers.csv            # writes love_numbers.png
    python3 plot_csv.py viscoelastic_loading.csv --show

The figure is written beside the table, with the extension .png, or to
--out. Needs numpy and matplotlib.

A table describes its own plot in "# key: value" lines above the header:

    title   figure title
    x       the abscissa column (default: the first column)
    y       the ordinate columns, comma-separated; "|" starts a new panel
            (default: every numeric column other than x, group and the
            "_exact" columns, one panel)
    group   a column whose distinct values split the rows into curves
            (e.g. a scheme name); y then names one column per panel
    xlabel  axis label (default: the x column)
    ylabel  axis labels, "|"-separated per panel
    logx    "true" for a logarithmic abscissa
    logy    "true", or "|"-separated per panel; plots |y|
    note    a line of text under the title

A column "<name>_exact" is the reference for column "<name>": it is drawn
dashed in the same colour and not listed on its own.
"""
import argparse
import csv
from pathlib import Path


def read(path):
    meta, lines = {}, []
    with open(path) as f:
        for line in f:
            if line.startswith("#"):
                key, _, value = line[1:].partition(":")
                meta[key.strip()] = value.strip()
            elif line.strip():
                lines.append(line)
    rows = list(csv.reader(lines))
    header, body = rows[0], rows[1:]
    columns = {name: [r[i] for r in body] for i, name in enumerate(header)}
    return meta, header, columns


def numbers(values):
    return [float(v) for v in values]


def is_number(values):
    try:
        numbers(values)
        return True
    except ValueError:
        return False


def per_panel(value, n, default):
    parts = value.split("|") if value else []
    return [(parts[i].strip() if i < len(parts) else default) for i in range(n)]


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("csv", type=Path)
    parser.add_argument("--out", type=Path, help="figure file (default: beside the table)")
    parser.add_argument("--show", action="store_true", help="open a window as well")
    args = parser.parse_args()

    import matplotlib
    if not args.show:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    meta, header, columns = read(args.csv)
    x_name = meta.get("x", header[0])
    group = meta.get("group")
    if "y" in meta:
        panels = [[c.strip() for c in p.split(",") if c.strip()]
                  for p in meta["y"].split("|")]
    else:
        panels = [[c for c in header if c not in (x_name, group)
                   and not c.endswith("_exact") and is_number(columns[c])]]
    n = len(panels)
    ylabels = per_panel(meta.get("ylabel", ""), n, "")
    logy = [v.lower() == "true" for v in per_panel(meta.get("logy", ""), n, "false")]
    logx = meta.get("logx", "false").lower() == "true"

    fig, axes = plt.subplots(n, 1, figsize=(7, 3.2 * n + 0.8), squeeze=False,
                             sharex=True)
    x_all = np.array(numbers(columns[x_name]))
    for k, (ax, names) in enumerate(zip(axes[:, 0], panels)):
        lift = np.abs if logy[k] else (lambda v: v)
        if group:
            labels = columns[group]
            for g in dict.fromkeys(labels):  # first-seen order
                rows = [i for i, v in enumerate(labels) if v == g]
                for name in names:
                    y = np.array(numbers(columns[name]))[rows]
                    ax.plot(x_all[rows], lift(y), "o-", ms=3,
                            label=g if len(names) == 1 else f"{g} {name}")
        else:
            for name in names:
                line, = ax.plot(x_all, lift(np.array(numbers(columns[name]))),
                                "o-", ms=3, label=name)
                exact = name + "_exact"
                if exact in columns:
                    ax.plot(x_all, lift(np.array(numbers(columns[exact]))), "--",
                            color=line.get_color(), lw=1)
        if logy[k]:
            ax.set_yscale("log")
        if logx:
            ax.set_xscale("log")
        ax.set_ylabel(ylabels[k])
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8, ncol=2 if len(ax.get_lines()) > 6 else 1)
    if any(name + "_exact" in columns for names in panels for name in names):
        axes[0, 0].plot([], [], "k--", lw=1, label="exact")
        axes[0, 0].legend(fontsize=8)
    axes[-1, 0].set_xlabel(meta.get("xlabel", x_name))
    title = meta.get("title", args.csv.stem)
    if "note" in meta:
        title += "\n" + meta["note"]
    fig.suptitle(title, fontsize=10)
    fig.tight_layout()

    out = args.out or args.csv.with_suffix(".png")
    fig.savefig(out, dpi=150)
    print(f"Wrote {out}")
    if args.show:
        plt.show()


if __name__ == "__main__":
    main()
