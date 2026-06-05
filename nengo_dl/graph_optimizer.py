"""Operator scheduling for the PyTorch backend.

Use Nengo's own dependency-graph construction so execution order matches the
reference simulator, including the subtle distinction between sets / incs /
reads / updates that recurrent models rely on.
"""

from typing import List
import warnings

from nengo.exceptions import BuildError
from nengo.utils.graphs import toposort as nengo_toposort
from nengo.utils.simulator import operator_dependency_graph


def topo_sort(operators: List) -> List:
    """Return operators in the same dependency order Nengo uses."""
    if not operators:
        return []

    try:
        dg = operator_dependency_graph(operators)
        return list(nengo_toposort(dg))
    except BuildError:
        warnings.warn(
            "Cycle detected in operator dependency graph; "
            "using original operator order. Simulation results may be incorrect."
        )
        return list(operators)
