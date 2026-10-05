"""Shared plumbing of the benchmark scripts: where the drivers of a
build are, and how a command is run and logged.

Every family's scripts import this (and models, make_case) from
benchmarks/common; a script begins with

    sys.path.insert(0, str(Path(__file__).resolve().parent.parent
                           / "common"))
"""
from __future__ import annotations

import shlex
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPOSITORY = HERE.parent.parent


def find_programs(given: Path | None) -> Path:
    """The directory holding the drivers: the one given, or
    <build>/benchmarks/bin of a build tree of the repository."""
    if given is not None:
        return given.resolve()
    for build in sorted(REPOSITORY.glob("build*")):
        candidate = build / "benchmarks" / "bin"
        if (candidate / "love_benchmark").exists():
            return candidate
    raise SystemExit("love_benchmark not found in a build tree of the "
                     "repository (configure with -DUSE_MPI=ON "
                     "-DBUILD_BENCHMARKS=ON), and no --programs given")


def run(command: list[str], *, log: Path | None = None,
        dry_run: bool = False) -> bool:
    """Run a command, its output to `log` and the terminal; True on
    success."""
    print("  $ " + shlex.join(command), flush=True)
    if dry_run:
        return True
    start = time.time()
    with subprocess.Popen(command, stdout=subprocess.PIPE,
                          stderr=subprocess.STDOUT, text=True) as process:
        lines = []
        for line in process.stdout:
            lines.append(line)
            sys.stdout.write("    " + line)
        process.wait()
    if log is not None:
        log.write_text("$ " + shlex.join(command) + "\n" + "".join(lines))
    print(f"  ({time.time() - start:.1f} s, exit {process.returncode})",
          flush=True)
    return process.returncode == 0
