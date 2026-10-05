"""Where benchmark output may go: the build tree, never the source tree.

outside_source(path) returns the resolved path, or stops with a message
when it lies inside the repository but not under a build* directory.
Every script of the viscoelastic box and sphere sub-families that writes
output passes its output paths through it.
"""
from __future__ import annotations

from pathlib import Path

REPOSITORY = Path(__file__).resolve().parent.parent.parent


def outside_source(path: Path) -> Path:
    path = Path(path).resolve()
    try:
        rel = path.relative_to(REPOSITORY)
    except ValueError:
        return path
    if rel.parts and rel.parts[0].startswith("build"):
        return path
    raise SystemExit(f"{path} is inside the source tree; write benchmark "
                     "output into the build tree (run the launchers in "
                     "<build>/benchmarks/..., or pass --out)")
