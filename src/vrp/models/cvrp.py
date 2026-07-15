"""
Exact MILP formulation for the Capacitated Vehicle Routing Problem (CVRP).

Formulation: two-index vehicle-flow with Miller-Tucker-Zemlin (MTZ)
subtour elimination and remaining-capacity tracking. See
docs/math_formulation.md for the full mathematical statement (sets,
parameters, decision variables, objective and constraints).

This formulation scales to roughly 12-15 customers within a reasonable
time limit on CBC (the open-source solver bundled with PuLP) -- it is
meant to produce *provably optimal* baselines to validate and benchmark
the Clarke-Wright heuristic (see heuristics/clarke_wright.py and
benchmark.py), not to solve large instances directly.
"""

from __future__ import annotations

import time
from dataclasses import dataclass

import pulp

from vrp.instance import Instance


@dataclass
class RoutingSolution:
    routes: list[list[int]]     # each route: list of customer indices (0-based, depot excluded)
    total_distance: float
    status: str
    solve_time: float
    lower_bound: float | None = None   # best known bound (useful when not solved to optimality)


def solve_cvrp_milp(
    instance: Instance,
    time_limit: int | None = 120,
    msg: bool = False,
) -> RoutingSolution:
    """
    Solve a single-depot CVRP instance to (provable, given enough time) optimality.

    Node indexing internally: 0 = depot, 1..n = customers (matching `instance.customers`).
    """
    if instance.num_depots() != 1:
        raise ValueError("solve_cvrp_milp expects a single-depot instance; use mdvrp for multi-depot.")

    n = instance.num_customers()
    coords = [instance.depots[0]] + list(instance.customers)
    dist_matrix = [
        [instance.distance(coords[i], coords[j]) if i != j else 0.0 for j in range(n + 1)]
        for i in range(n + 1)
    ]
    demands = [0.0] + list(instance.demands)

    return solve_cvrp_from_distance_matrix(
        dist_matrix, demands, instance.vehicle_capacity, instance.num_vehicles_per_depot,
        time_limit=time_limit, msg=msg,
    )


def solve_cvrp_from_distance_matrix(
    dist_matrix: list[list[float]],
    demands: list[float],
    capacity: float,
    num_vehicles: int,
    time_limit: int | None = 120,
    msg: bool = False,
) -> RoutingSolution:
    """
    Same exact CVRP MILP as `solve_cvrp_milp`, but taking a pre-computed
    distance matrix directly instead of Euclidean coordinates. This is the
    piece that makes the model reusable for real road-network distances
    later on -- e.g. real driving distances instead of straight-line.

    `dist_matrix[i][j]` and `demands` must both be indexed with 0 = depot,
    1..n = customers, matching each other.
    """
    n = len(demands) - 1  # number of customers (index 0 is the depot)
    nodes = list(range(n + 1))
    customers = list(range(1, n + 1))
    demand = {i: demands[i] for i in nodes}
    Q = capacity
    K = num_vehicles

    dist = {(i, j): dist_matrix[i][j] for i in nodes for j in nodes if i != j}

    prob = pulp.LpProblem("CVRP", pulp.LpMinimize)

    # x[i,j] = 1 if a vehicle travels directly from i to j
    x = pulp.LpVariable.dicts("x", (nodes, nodes), cat="Binary")
    for i in nodes:
        x[i][i].upperBound = 0  # no self loops

    # u[i] = remaining vehicle capacity immediately after serving customer i
    u = pulp.LpVariable.dicts("u", customers, lowBound=0, upBound=Q, cat="Continuous")

    # Objective: minimize total distance traveled
    prob += pulp.lpSum(dist[i, j] * x[i][j] for i in nodes for j in nodes if i != j)

    # Each customer has exactly one incoming and one outgoing arc
    for h in customers:
        prob += pulp.lpSum(x[i][h] for i in nodes if i != h) == 1
        prob += pulp.lpSum(x[h][j] for j in nodes if j != h) == 1

    # At most K vehicles leave / return to the depot
    prob += pulp.lpSum(x[0][j] for j in customers) <= K
    prob += pulp.lpSum(x[i][0] for i in customers) <= K
    prob += pulp.lpSum(x[0][j] for j in customers) == pulp.lpSum(x[i][0] for i in customers)

    # MTZ subtour elimination + capacity, remaining-capacity form:
    # if x[i,j]=1, remaining capacity after j must be at most remaining
    # capacity after i, minus what gets delivered at j.
    for i in customers:
        for j in customers:
            if i != j:
                prob += u[j] <= u[i] - demand[j] + Q * (1 - x[i][j])

    for i in customers:
        prob += u[i] >= 0
        prob += u[i] <= Q - demand[i]

    solver = pulp.PULP_CBC_CMD(msg=msg, timeLimit=time_limit)
    start = time.time()
    prob.solve(solver)
    elapsed = time.time() - start

    status = pulp.LpStatus[prob.status]
    routes = _extract_routes(x, nodes, customers) if status in ("Optimal", "Not Solved", "Undefined") else []
    total_distance = pulp.value(prob.objective) if prob.objective is not None else float("nan")

    return RoutingSolution(
        routes=routes,
        total_distance=total_distance if total_distance is not None else float("nan"),
        status=status,
        solve_time=elapsed,
    )


def _extract_routes(x, nodes, customers) -> list[list[int]]:
    """Reconstruct depot-to-depot routes from the arc decision variables."""
    arcs = []
    for i in nodes:
        for j in nodes:
            if i != j:
                v = x[i][j].value()
                if v is not None and v > 0.5:
                    arcs.append((i, j))

    routes = []
    used = set()
    starts = [j for (i, j) in arcs if i == 0]
    for start in starts:
        if start in used:
            continue
        route = [start]
        used.add(start)
        current = start
        while True:
            nxt = next((j for (i, j) in arcs if i == current and j != 0), None)
            if nxt is None:
                break
            route.append(nxt)
            used.add(nxt)
            current = nxt
        routes.append([c - 1 for c in route])  # shift back to 0-indexed customer ids
    return routes