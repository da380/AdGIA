"""Compare exchange files: each later file against the first.

    compare reference.txt other.txt [more.txt ...] [--tol 1e-8] [--verbose]

Files are the cross-code exchange format (`common/exchange.py`): the
first column must be the degree l, the remaining shared columns are
compared.  Differences are relative to the largest reference value of
the degree, so tiny channels are judged against the degree's scale, not
their own; a NaN in either file excludes that entry (the codes mark
excluded degrees with NaN).  With --tol the exit status reports whether
every comparison stayed below it.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent.parent / "common"))

import exchange  # noqa: E402


def compare(reference: exchange.Table, other: exchange.Table,
            *, verbose: bool) -> dict[str, float]:
    if reference.columns[0] != "l" or other.columns[0] != "l":
        raise SystemExit("the first column of an exchange file must be l")
    shared = [c for c in reference.columns[1:] if c in other.columns]
    if not shared:
        raise SystemExit("no shared columns to compare")
    degrees = reference["l"].astype(int)
    if not np.array_equal(degrees, other["l"].astype(int)):
        n = min(len(degrees), len(other["l"]))
        degrees = degrees[:n]
        if not np.array_equal(degrees, other["l"].astype(int)[:n]):
            raise SystemExit("the degree columns disagree")
    worst = {c: 0.0 for c in shared}
    for i, l in enumerate(degrees):
        ref = np.array([reference[c][i] for c in shared])
        new = np.array([other[c][i] for c in shared])
        live = ~(np.isnan(ref) | np.isnan(new))
        if not live.any():
            continue
        scale = np.max(np.abs(ref[live]))
        if scale == 0.0:
            continue
        rel = np.abs(new - ref) / scale
        if verbose:
            parts = " ".join(f"{c} {rel[j]:.2e}" if live[j] else f"{c} --"
                             for j, c in enumerate(shared))
            print(f"  l {l:3d}  {parts}")
        for j, c in enumerate(shared):
            if live[j]:
                worst[c] = max(worst[c], float(rel[j]))
    return worst


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="compare exchange files against the first")
    parser.add_argument("files", nargs="+", type=Path)
    parser.add_argument("--tol", type=float, default=None,
                        help="fail (exit 1) if any difference exceeds this")
    parser.add_argument("--verbose", action="store_true",
                        help="per-degree differences")
    args = parser.parse_args(argv)
    if len(args.files) < 2:
        raise SystemExit("give a reference file and at least one other")
    reference = exchange.read_table(args.files[0])
    print(f"reference: {args.files[0]}")
    ok = True
    for path in args.files[1:]:
        print(f"{path}:")
        worst = compare(reference, exchange.read_table(path),
                        verbose=args.verbose)
        overall = max(worst.values())
        summary = "  ".join(f"{c} {v:.3e}" for c, v in worst.items())
        print(f"  worst relative difference: {summary}")
        if args.tol is not None:
            verdict = "within" if overall <= args.tol else "EXCEEDS"
            print(f"  overall {overall:.3e} {verdict} tol {args.tol:g}")
            ok = ok and overall <= args.tol
    if not ok:
        sys.exit(1)


if __name__ == "__main__":
    main()
