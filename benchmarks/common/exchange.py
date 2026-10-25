"""The cross-code exchange format: a few numbers, in SI, with provenance.

Every cross-code comparison reduces each code's answer to a small text
file of named scalars and one table, which this module writes and reads
and whose layout the C++ drivers reproduce by hand.  The format, fixed
here and documented in the benchmarks README, is that of sl3dpp's stage
fixtures:

  - leading `#` lines: provenance, free text.  By convention the first
    says what the file holds, and one says `# code <name> commit <sha>
    date <iso-date>`; `provenance_line` builds it.
  - bare `name value...` lines: named scalars (one or more floats).
  - a `columns <name>...` line, then the table, one row per line, every
    value `%.17g`, SI throughout.

A comparison then reads two such files, matches scalars by name and
columns by header, and reports relative differences -- it never needs
the producing code.
"""
from __future__ import annotations

import datetime
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np


def code_commit(anywhere: Path) -> str:
    """The short commit of the repository containing `anywhere`, or
    'unknown' outside one."""
    where = Path(anywhere).resolve()
    if not where.is_dir():
        where = where.parent
    out = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=where,
                         capture_output=True, text=True, check=False)
    return out.stdout.strip() or "unknown"


def provenance_line(code: str, anywhere: Path) -> str:
    """The conventional provenance line for `code`, dated today, with
    the commit of the repository containing `anywhere`."""
    return (f"code {code} commit {code_commit(anywhere)} "
            f"date {datetime.date.today().isoformat()}")


@dataclass
class Table:
    """One exchange file: comments (without their '#'), named scalars
    (each a float array, scalars of one value included), the column
    names and the table, rows by columns."""
    comments: list[str] = field(default_factory=list)
    scalars: dict[str, np.ndarray] = field(default_factory=dict)
    columns: list[str] = field(default_factory=list)
    table: np.ndarray = field(default_factory=lambda: np.empty((0, 0)))

    def __getitem__(self, column: str) -> np.ndarray:
        return self.table[:, self.columns.index(column)]


def write_table(path: Path, *, comments: Sequence[str],
                scalars: Mapping[str, float | Sequence[float]] | None = None,
                columns: Sequence[str], table: np.ndarray) -> Path:
    """The exchange file at `path`, written whole."""
    table = np.atleast_2d(np.asarray(table, dtype=float))
    if table.shape[1] != len(columns):
        raise ValueError(f"{table.shape[1]} columns of data for "
                         f"{len(columns)} names")
    lines = [f"# {comment}" for comment in comments]
    for name, value in (scalars or {}).items():
        values = np.atleast_1d(np.asarray(value, dtype=float))
        lines.append(" ".join([name] + [f"{v:.17g}" for v in values]))
    lines.append("columns " + " ".join(columns))
    for row in table:
        lines.append(" ".join(f"{v:.17g}" for v in row))
    path = Path(path)
    path.write_text("\n".join(lines) + "\n")
    return path


def read_table(path: Path) -> Table:
    """The exchange file at `path`, parsed; rows must be rectangular."""
    out = Table()
    rows: list[list[float]] = []
    for line in Path(path).read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        if line.startswith("#"):
            out.comments.append(line[1:].strip())
            continue
        head, *rest = line.split()
        if out.columns:
            rows.append([float(v) for v in [head] + rest])
        elif head == "columns":
            out.columns = rest
        else:
            out.scalars[head] = np.array([float(v) for v in rest])
    out.table = (np.array(rows) if rows
                 else np.empty((0, len(out.columns))))
    if rows and out.table.ndim != 2:
        raise ValueError(f"{path}: ragged table")
    return out
