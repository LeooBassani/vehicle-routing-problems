"""

Extends the CVRP vehicle-flow formulation (see cvrp.py) with:
    - arrival-time variables t[i]
    - big-M time propagation constraints, which also eliminate subtours

See docs/math_formulation.md for the full mathematical statement.
"""

from __future__ import annotations

import time

import pulp

from vrp.instance import Instance
from vrp.models.cvrp import RoutingSolution, _extract_routes


def solve_vrptw_milp(instance: Instance, time_limit: int = 120, msg: bool = False) -> RoutingSolution:
    if instance.num_depots() != 1:
        raise ValueError("solve_vrptw_milp expects a single-depot instance.")
    if instance.time_windows is None:
        raise ValueError("Instance has no time_windows; generate with generate_vrptw_instance().")

    n = instance.num_customers()
    coords = [instance.depots[0]] + list(instance.customers)
    dist_matrix = [[instance.distance(coords[i], coords[j]) if i != j else 0.0 for j in range(n + 1)] for i in range(n + 1)]
    demands = [0.0] + list(instance.demands)

    return solve_vrptw_from_distance_matrix(
        dist_matrix, demands, instance.vehicle_capacity, instance.num_vehicles_per_depot,
        instance.time_windows, instance.service_times, instance.depot_time_window,
        speed=instance.speed, time_limit=time_limit, msg=msg,
    )


def solve_vrptw_from_distance_matrix(
    dist_matrix: list[list[float]],
    demands: list[float],
    capacity: float,
    num_vehicles: int,
    time_windows: list[tuple[float, float]],
    service_times: list[float],
    depot_time_window: tuple[float, float],
    speed: float = 1.0,
    time_limit: int = 120,
    msg: bool = False,
) -> RoutingSolution:
    """
    Same exact VRPTW MILP as `solve_vrptw_milp`, but taking a pre-computed
    distance matrix directly. `dist_matrix`/`demands` are indexed 0=depot,
    1..n=customers; `time_windows`/`service_times` are indexed 0..n-1,
    matching customers only (the depot's window is passed separately).
    """
    n = len(demands) - 1
    nodes = list(range(n + 1))
    customers = list(range(1, n + 1))
    demand = {i: demands[i] for i in nodes}

    tw = {0: depot_time_window}
    service = {0: 0.0}
    for idx, c in enumerate(customers):
        tw[c] = time_windows[idx]
        service[c] = service_times[idx]

    Q = capacity
    K = num_vehicles
    dist = {(i, j): dist_matrix[i][j] for i in nodes for j in nodes if i != j}
    travel_time = {k: v / speed for k, v in dist.items()}

    prob = pulp.LpProblem("VRPTW", pulp.LpMinimize)

    x = pulp.LpVariable.dicts("x", (nodes, nodes), cat="Binary")
    for i in nodes:
        x[i][i].upperBound = 0

    t = {
        i: pulp.LpVariable(f"t_{i}", lowBound=tw[i][0], upBound=tw[i][1], cat="Continuous")
        for i in nodes
    }
    u = pulp.LpVariable.dicts("u", customers, lowBound=0, upBound=Q, cat="Continuous")

    prob += pulp.lpSum(dist[i, j] * x[i][j] for i in nodes for j in nodes if i != j)

    for h in customers:
        prob += pulp.lpSum(x[i][h] for i in nodes if i != h) == 1
        prob += pulp.lpSum(x[h][j] for j in nodes if j != h) == 1

    prob += pulp.lpSum(x[0][j] for j in customers) <= K
    prob += pulp.lpSum(x[i][0] for i in customers) <= K
    prob += pulp.lpSum(x[0][j] for j in customers) == pulp.lpSum(x[i][0] for i in customers)

    # Big-M time propagation: if x[i,j]=1, arrival at j must be >= departure
    # from i (= arrival + service) + travel time. Also eliminates subtours
    # (see our conversation on why: positive travel/service time around any
    # closed cycle not through the depot is a contradiction).
    horizon = depot_time_window[1]
    for i in nodes:
        for j in customers:
            if i != j:
                M = horizon + travel_time[i, j]
                prob += t[j] >= t[i] + service[i] + travel_time[i, j] - M * (1 - x[i][j])

    # Capacity: same remaining-capacity MTZ convention as CVRP.
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