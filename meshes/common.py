"""What every mesh script here shares.

Each script in this directory builds one kind of mesh with planetmodel and
writes the mesh file (gmsh `.msh`, or MFEM's own `.mesh` for the
aspherical meshes), with a JSON manifest beside it saying what every
attribute means. The build runs them (see CMakeLists.txt here) and puts
the meshes in the build tree's `data/` directory, where the C++ examples
and tests read them; nothing is written to the source tree. Run by hand,
a script writes to the current directory unless told otherwise.

The scripts are meant to be read and copied. To make a new mesh, copy the
closest script, change the skeleton, the sizing or the shells, give the
output a new name, and add a line to CMakeLists.txt. Every script takes
`--out DIR`, `--verbose` to let gmsh talk, `--scale` on the element
sizes (the canned sizes are deliberately coarse; smaller is finer) and
`--help`.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from scipy.interpolate import RegularGridInterpolator



def parser(description: str) -> argparse.ArgumentParser:
    """The command line every script accepts, ready for extra options."""
    p = argparse.ArgumentParser(
        description=description,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--out", type=Path, default=Path.cwd(), metavar="DIR",
                   help="directory to write into (default: the current directory)")
    p.add_argument("--verbose", action="store_true",
                   help="let gmsh print as it works")
    p.add_argument("--scale", type=float, default=1.0,
                   help="scale factor on the element sizes (smaller is "
                        "finer — e.g. 0.5 for a higher-resolution version "
                        "of a deliberately coarse canned mesh)")
    return p


def sizes(args, *values: float) -> tuple[float, ...]:
    """A script's sizing numbers with --scale applied."""
    return tuple(args.scale * v for v in values)


class SeaLevelBand:
    """planetmodel Refinement: fine where a surface field is near zero.

    The field is a wrapped lat-lon grid (poles and the 360-degree
    column included, so plain bilinear interpolation covers the
    sphere) in ANGULAR units — earth_coastlines.py feeds it the
    distance to the real coastline, fingerprint_coastline.py the
    angular distance to an analytic shoreline circle. The size is
    `size` wherever |field| < band and |r - radius| < depth_width (the
    plateau), and grows towards `far_size` at the rate `gradation`
    (size per unit distance) beyond. The bounded slope is what makes
    the band reliable: a steep size valley narrower than the
    background size can be stepped over by the mesher's advancing
    front — whole coastlines go missing — while at gradation ~1 every
    sample within the growth region forces smaller steps towards the
    plateau, so nothing is skipped.
    """

    def __init__(self, lats, lons, field, *, band, size, far_size,
                 radius=1.0, depth_width=0.1, gradation=1.0):
        self.size = float(size)
        self.far_size = float(far_size)
        self._band = float(band)
        self._radius = float(radius)
        self._depth = float(depth_width)
        self._gradation = float(gradation)
        order = np.argsort(lats)
        self._interp = RegularGridInterpolator(
            (lats[order], lons), field[order, :],
            bounds_error=False, fill_value=None)

    def __call__(self, points):
        pts = np.asarray(points, dtype=float)
        r = np.linalg.norm(pts, axis=1)
        safe = np.maximum(r, 1e-12)
        lat = np.degrees(np.arcsin(np.clip(pts[:, 2] / safe, -1.0, 1.0)))
        lon = np.degrees(np.arctan2(pts[:, 1], pts[:, 0])) % 360.0
        f = np.abs(self._interp(np.column_stack([lat, lon])))
        excess = np.maximum(f - self._band,
                            np.abs(r - self._radius) - self._depth)
        return np.minimum(self.far_size,
                          self.size +
                          self._gradation * np.maximum(0.0, excess))


def report(result) -> None:
    """Say what a build wrote: the files, the counts and the checks."""
    print(f"{result.msh_path.name}: {result.summary()}")
    for warning in result.validation.warnings:
        print(f"  warning: {warning}")
    print(f"  manifest: {result.manifest_path}")
