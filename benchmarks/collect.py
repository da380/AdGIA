"""Package a benchmark run tree for transfer to another machine.

Collects a runs tree (runs_campaign by default) into one tar.gz whose
paths start with the tree's own name, so that unpacking it in
<build>/benchmarks of another checkout puts the results where the
collating scripts (plot, cmb_report, the scaling summary,
talk_figures.py) expect them — they read the result files, not the case
meshes, so the default tier is enough to analyse a server campaign on
another machine. Three tiers:

  (default)     the results: *.json, *.md, *.txt, *.csv, *.png —
                everything the collating scripts read; it scales with
                the number of runs, not with the meshes, so a server
                campaign stays tens of MB
  --figures     the figures and summaries only: *.png, *.md
  --full        the whole tree: case meshes, model fields and ParaView
                exports included

  --with-cases  adds the case inputs (case.*, gmsh/) to the default
                tier, to re-run a solve against the same mesh elsewhere

The archive carries a manifest, <tree>/collect_manifest.json, recording
the host, the date, the source commit and the tier.

    ./collect
    ./collect --figures
    ./collect --tree runs_campaign --dest love_numbers_server.tar.gz

and at the other end:

    scp <server>:<build>/benchmarks/runs_campaign_<host>_<date>.tar.gz .
    tar -xzf runs_campaign_<host>_<date>.tar.gz -C <build>/benchmarks

which overlays the tree onto any results already there.
"""
from __future__ import annotations

import argparse
import datetime
import json
import socket
import subprocess
import sys
import tarfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE / "common"))

from outputs import REPOSITORY, outside_source  # noqa: E402

RESULT_SUFFIXES = {".json", ".md", ".txt", ".csv", ".png"}
FIGURE_SUFFIXES = {".png", ".md"}


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="package a benchmark run tree as one tar.gz")
    p.add_argument("--tree", type=Path, default=Path("runs_campaign"),
                   help="the run tree to collect (default: runs_campaign "
                        "where this is started)")
    p.add_argument("--dest", type=Path, default=None,
                   help="the archive to write (default: "
                        "<tree>_<host>_<date>.tar.gz beside the tree)")
    tier = p.add_mutually_exclusive_group()
    tier.add_argument("--figures", action="store_true",
                      help="figures and summaries only (*.png, *.md)")
    tier.add_argument("--full", action="store_true",
                      help="the whole tree, cases and fields included")
    p.add_argument("--with-cases", action="store_true",
                   help="add the case inputs (case.*, gmsh/) to the "
                        "default tier")
    return p


def selected(path: Path, args: argparse.Namespace) -> bool:
    """Whether a file of the tree belongs to the chosen tier."""
    if args.full:
        return True
    if args.figures:
        return path.suffix in FIGURE_SUFFIXES
    if path.suffix in RESULT_SUFFIXES:
        return True
    return args.with_cases and (path.name.startswith("case.")
                                or "gmsh" in path.parts)


def commit() -> str | None:
    try:
        return subprocess.run(
            ["git", "-C", str(REPOSITORY), "rev-parse", "HEAD"],
            capture_output=True, text=True, check=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def main() -> None:
    args = parser().parse_args()
    if args.with_cases and (args.figures or args.full):
        raise SystemExit("--with-cases refines the default tier only")

    tree = args.tree.resolve()
    if not tree.is_dir():
        raise SystemExit(f"no run tree at {tree}")

    tier = ("full" if args.full else "figures" if args.figures
            else "results+cases" if args.with_cases else "results")
    stamp = datetime.date.today().strftime("%Y%m%d")
    dest = args.dest if args.dest is not None else \
        tree.parent / f"{tree.name}_{socket.gethostname()}_{stamp}.tar.gz"
    dest = outside_source(dest)

    files = sorted(p for p in tree.rglob("*")
                   if p.is_file() and selected(p, args))
    if not files:
        raise SystemExit(f"nothing of the {tier} tier under {tree}")

    manifest = {
        "host": socket.gethostname(),
        "date": datetime.datetime.now().isoformat(timespec="seconds"),
        "tree": str(tree),
        "tier": tier,
        "commit": commit(),
        "files": len(files),
        "bytes": sum(p.stat().st_size for p in files),
    }
    manifest_file = tree / "collect_manifest.json"
    manifest_file.write_text(json.dumps(manifest, indent=2) + "\n")

    with tarfile.open(dest, "w:gz") as tar:
        tar.add(manifest_file, arcname=f"{tree.name}/collect_manifest.json")
        for path in files:
            tar.add(path, arcname=f"{tree.name}/{path.relative_to(tree)}")

    print(f"{dest}: {len(files)} files of the {tier} tier "
          f"({manifest['bytes'] / 1e6:.1f} MB before compression, "
          f"{dest.stat().st_size / 1e6:.1f} MB archived)")
    print(f"unpack with: tar -xzf {dest.name} -C <build>/benchmarks")


if __name__ == "__main__":
    main()
