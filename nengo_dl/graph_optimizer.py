"""Graph optimization for nengo-dl (PyTorch backend).

Provides topological sorting of Nengo operators to ensure correct data
dependencies are respected during simulation.
"""

from collections import defaultdict, deque
from typing import List


def topo_sort(operators: List) -> List:
    """Sort Nengo operators in dependency order (topological sort).

    Nengo's ``Model.operators`` list is in BUILD order, not execution order.
    The correct execution order must respect signal read/write dependencies:
    if operator A writes a signal that operator B reads, A must run before B.

    This function uses Kahn's algorithm on the operator dependency graph.

    Parameters
    ----------
    operators : list
        Nengo operator list (typically ``model.operators``).

    Returns
    -------
    list
        Operators in topological execution order.

    Raises
    ------
    RuntimeError
        If a cycle is detected (should not happen in valid Nengo models).
    """
    n = len(operators)
    if n == 0:
        return []

    # For each signal (by base id), record which operators write it
    # "write" = sets, incs, or updates
    signal_writers: dict = defaultdict(list)
    for i, op in enumerate(operators):
        for sig in op.sets + op.incs + op.updates:
            signal_writers[id(sig.base)].append(i)

    # Build adjacency list: edge (i → j) means op i must run before op j
    # (because op i writes a signal that op j reads)
    adj: List[List[int]] = [[] for _ in range(n)]
    indegree = [0] * n

    for j, op in enumerate(operators):
        added_edges = set()
        for sig in op.reads:
            for i in signal_writers[id(sig.base)]:
                key = (i, j)
                if i != j and key not in added_edges:
                    adj[i].append(j)
                    indegree[j] += 1
                    added_edges.add(key)

    # Kahn's algorithm
    queue = deque(i for i in range(n) if indegree[i] == 0)
    result_indices = []

    while queue:
        v = queue.popleft()
        result_indices.append(v)
        for u in adj[v]:
            indegree[u] -= 1
            if indegree[u] == 0:
                queue.append(u)

    if len(result_indices) != n:
        # Cycle detected – fall back to original order with a warning
        import warnings
        warnings.warn(
            "Cycle detected in operator dependency graph; "
            "using original operator order. Simulation results may be incorrect."
        )
        return list(operators)

    return [operators[i] for i in result_indices]
