"""
MDVRP = Now a vehicle can start a nodein depot A and  finish in depot B

Extends the CVRP vehicle-flow formulation with an  variable z[i,d] (customer i served by depot d) 

"""

from __future__ import annotations

import time

import pulp

from vrp.instance import Instance
from vrp.models.cvrp import RoutingSolution


def solve_mdvrp_milp(instance: Instance, time_limit: int = 120, msg: bool = False) -> RoutingSolution:
    D = instance.num_depots()
    if D < 2:
        raise ValueError("solve_mdvrp_milp expects >= 2 depots; use cvrp for single-depot instances.")

    n = instance.num_customers()
    coords = list(instance.depots) + list(instance.customers)
    total_nodes = D + n
    dist_matrix = [
        [instance.distance(coords[i], coords[j]) if i != j else 0.0 for j in range(total_nodes)]
        for i in range(total_nodes)
    ]
    demands = [0.0] * D + list(instance.demands)

    return solve_mdvrp_from_distance_matrix(
        dist_matrix, demands, D, instance.vehicle_capacity, instance.num_vehicles_per_depot,
        time_limit=time_limit, msg=msg,
    )


def solve_mdvrp_from_distance_matrix(
    dist_matrix: list[list[float]],
    demands: list[float],
    num_depots: int,
    capacity: float,
    num_vehicles_per_depot: int,
    time_limit: int = 120,
    msg: bool = False,
) -> RoutingSolution:
    """
    Same exact MDVRP MILP, but taking a pre-computed distance matrix
    directly. `dist_matrix`/`demands` are indexed 0..num_depots-1 for
    depots, then num_depots..num_depots+n-1 for customers.
    """
    D = num_depots
    total_nodes = len(demands)
    n = total_nodes - D

    depots = list(range(D))
    customers = list(range(D, D + n))
    demand = {i: demands[i] for i in range(total_nodes)}
    Q = capacity
    K = num_vehicles_per_depot

    dist = {(i, j): dist_matrix[i][j] for i in range(total_nodes) for j in range(total_nodes) if i != j}

    prob = pulp.LpProblem("MDVRP", pulp.LpMinimize)

    x = pulp.LpVariable.dicts("x", (range(total_nodes), range(total_nodes)), cat="Binary")
    for i in range(total_nodes):
        x[i][i].upperBound = 0
    for d1 in depots:
        for d2 in depots:
            if d1 != d2:
                x[d1][d2].upperBound = 0  # no direct depot-to-depot arcs

    z = pulp.LpVariable.dicts("z", (customers, depots), cat="Binary")
    u = pulp.LpVariable.dicts("u", customers, lowBound=0, upBound=Q, cat="Continuous")

    prob += pulp.lpSum(dist[i, j] * x[i][j] for i in range(total_nodes) for j in range(total_nodes) if i != j)

    # Each customer visited exactly once, assigned to exactly one depot
    for h in customers:
        prob += pulp.lpSum(x[i][h] for i in range(total_nodes) if i != h) == 1
        prob += pulp.lpSum(x[h][j] for j in range(total_nodes) if j != h) == 1
        prob += pulp.lpSum(z[h][d] for d in depots) == 1

    # Depot arcs only allowed to/from the customer's assigned depot
    for h in customers:
        for d in depots:
            prob += x[d][h] <= z[h][d]
            prob += x[h][d] <= z[h][d]

    # Customer-customer arc implies identical depot assignment
    for i in customers:
        for j in customers:
            if i != j:
                for d in depots:
                    prob += z[i][d] - z[j][d] <= 1 - x[i][j]
                    prob += z[j][d] - z[i][d] <= 1 - x[i][j]

    # Fleet size per depot
    for d in depots:
        out_d = pulp.lpSum(x[d][h] for h in customers)
        in_d = pulp.lpSum(x[h][d] for h in customers)
        prob += out_d <= K
        prob += in_d <= K
        prob += out_d == in_d

    # Capacity (remaining-capacity MTZ convention, same as CVRP/VRPTW)
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
    routes = _extract_mdvrp_routes(x, depots, customers) if status in ("Optimal", "Not Solved", "Undefined") else []
    total_distance = pulp.value(prob.objective) if prob.objective is not None else float("nan")

    return RoutingSolution(
        routes=routes,
        total_distance=total_distance if total_distance is not None else float("nan"),
        status=status,
        solve_time=elapsed,
    )


def _extract_mdvrp_routes(x, depots, customers) -> list[list[int]]:
    """
    Reconstruct routes from arc variables. Each route starts with a
    marker: route[0] = -(depot_id + 1), followed by 0-indexed customer
    ids (shifted back from the internal depot-offset numbering) -- so
    callers can tell which depot serves each route.
    """
    all_nodes = depots + customers
    arcs = []
    for i in all_nodes:
        for j in all_nodes:
            if i != j:
                v = x[i][j].value()
                if v is not None and v > 0.5:
                    arcs.append((i, j))

    n_offset = len(depots)
    routes = []
    used = set()
    for d in depots:
        starts = [j for (i, j) in arcs if i == d]
        for start in starts:
            if start in used:
                continue
            route = [start]
            used.add(start)
            current = start
            while True:
                nxt = next((j for (i, j) in arcs if i == current and j not in depots), None)
                if nxt is None:
                    break
                route.append(nxt)
                used.add(nxt)
                current = nxt
            routes.append([-(d + 1)] + [c - n_offset for c in route])
    return routes