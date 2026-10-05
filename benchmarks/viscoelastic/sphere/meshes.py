"""Meshes of the sphere benchmarks: layered balls (3-D) and discs (2-D).

The mesh comes first. A mesh is a skeleton of radii, centre outward, and
planetmodel makes every radius a surface of the mesh, so the layer
interfaces of the models (cases.py: all on multiples of 0.2) are element
faces by construction. The element size is uniform (h at the interfaces
and away from them), the elements curved of the given order; no buffer
shell (nothing outside the body: there is no gravity).

    ./meshes --dim 2 --radii 0 0.4 0.8 1 --h 0.05 --out <build dir>

mesh_path() is what the studies call: it builds a mesh once and returns
its manifest, cached by (dim, radii, h, order) under the output
directory, which must not be inside the source tree.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent.parent / "common"))

from outputs import outside_source  # noqa: E402


def mesh_name(dim: int, radii, h: float, order: int) -> str:
    rs = "-".join(f"{r:g}" for r in radii)
    return f"sphere_{dim}d_r{rs}_h{h:g}_o{order}"


def mesh_path(out: Path, dim: int, radii, h: float, order: int = 2,
              verbose: bool = False) -> Path:
    """The manifest of the mesh, built if missing."""
    from planetmodel import Geometry, Skeleton
    from planetmodel.mesh3d import (MeshSpec, UniformInterfaces,
                                    build_layered_mesh)
    out = outside_source(out)
    out.mkdir(parents=True, exist_ok=True)
    stem = out / mesh_name(dim, radii, h, order)
    # Not with_suffix: the name holds dots ("h0.1_o2").
    manifest = Path(f"{stem}.json")
    if manifest.exists() and Path(f"{stem}.msh").exists():
        return manifest
    n = len(radii) - 1
    geometry = Geometry(Skeleton(list(radii)),
                        layer_names=[f"layer{j}" for j in range(n)],
                        interface_names=[f"r{r:g}" for r in radii[1:-1]]
                        + ["surface"])
    spec = MeshSpec(geometry, UniformInterfaces(h, h, 1.0), dimension=dim,
                    order=order)
    result = build_layered_mesh(spec, stem, verbose=verbose)
    print(f"{result.msh_path.name}: {result.summary()}", flush=True)
    return result.manifest_path


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dim", type=int, default=2)
    p.add_argument("--radii", type=float, nargs="+", default=[0.0, 1.0])
    p.add_argument("--h", type=float, default=0.1)
    p.add_argument("--order", type=int, default=2)
    p.add_argument("--out", type=Path, default=Path.cwd())
    p.add_argument("--verbose", action="store_true")
    a = p.parse_args()
    print(mesh_path(a.out, a.dim, a.radii, a.h, a.order, a.verbose))


if __name__ == "__main__":
    sys.exit(main())
