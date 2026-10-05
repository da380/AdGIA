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
`--out DIR`, `--verbose` to let gmsh talk, and `--help`.
"""
from __future__ import annotations

import argparse
from pathlib import Path



def parser(description: str) -> argparse.ArgumentParser:
    """The command line every script accepts, ready for extra options."""
    p = argparse.ArgumentParser(
        description=description,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--out", type=Path, default=Path.cwd(), metavar="DIR",
                   help="directory to write into (default: the current directory)")
    p.add_argument("--verbose", action="store_true",
                   help="let gmsh print as it works")
    return p


def report(result) -> None:
    """Say what a build wrote: the files, the counts and the checks."""
    print(f"{result.msh_path.name}: {result.summary()}")
    for warning in result.validation.warnings:
        print(f"  warning: {warning}")
    print(f"  manifest: {result.manifest_path}")
