"""The cost of the load solves recorded in a love_benchmark results file.

A run by degree (the default) records the wall seconds and outer
iterations of every degree's solve, and its cost is their mean per
solve. A combined run (`love_benchmark -combined`, `run.py --combined`)
solves the loads of all its degrees at once: its per-degree entries are
null and the one solve's totals sit under "combined_solves", so its cost
is that solve's, attributed once to the run and never per degree.
"""
from __future__ import annotations

from typing import NamedTuple


class LoadCost(NamedTuple):
    #: wall seconds per load solve: the mean over the degrees of a run
    #: by degree, the one solve of a combined run
    seconds: float
    #: outer iterations per load solve, likewise
    iterations: float
    #: True when the run solved all its degrees' loads in one solve
    combined: bool


def load_cost(results: dict) -> LoadCost:
    """The load-solve cost of a parsed love_benchmark results file."""
    if results.get("combined", False):
        solve = results["combined_solves"]["load"]
        return LoadCost(solve["seconds"], solve["outer_iterations"], True)
    loads = [d["load"] for d in results["degrees"] if "load" in d]
    return LoadCost(sum(d["seconds"] for d in loads) / len(loads),
                    sum(d["outer_iterations"] for d in loads) / len(loads),
                    False)
