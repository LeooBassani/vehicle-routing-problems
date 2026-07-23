"""
Clarke-Wright Savings heuristic for the CVRP.

Start with one route per customer (depot -> customer -> depot), 
then greedily merge the pair of routes with the largest "saving"

    s(i, j) = d(depot, i) + d(depot, j) - d(i, j)

as long as the merge keeps both endpoints on the "outside" of their
routes and respects vehicle capacity.

Two-phases, to respect the size limit K:
    Phase 1 -- only merge pairs with a strictly
               positive saving.
    Phase 2 -- if more routes remain than vehicles available (K), force
               additional merges (best remaining saving first, sign
               ignored) until the route count fits the fleet, or no
               feasible merge remains at all.

See docs/math_formulation.md for the CVRP problem definition this
heuristic is solving approximately.
"""

from __future__ import annotations
import time
from typing import Callable, Optional
from vrp.instance import Instance
from vrp.models.cvrp import RoutingSolution


def _try_merge(routes, route_of, route_load, capacity, i, j, feasibility_check=None) -> bool:
    ri, rj = route_of[i], route_of[j]
    if ri == rj:
        return False
    route_i, route_j = routes[ri], routes[rj]
    if route_i[0] != i and route_i[-1] != i:
        return False
    if route_j[0] != j and route_j[-1] != j:
        return False
    if route_load[ri] + route_load[rj] > capacity:
        return False
    oriented_i = list(reversed(route_i)) if route_i[0] == i else list(route_i)
    oriented_j = list(reversed(route_j)) if route_j[-1] == j else list(route_j)
    merged = oriented_i + oriented_j

    final_route = merged
    if feasibility_check is not None:
        final_route = feasibility_check(merged)
        if final_route is None:
            return False

    routes[ri] = final_route
    route_load[ri] += route_load[rj]
    del routes[rj]
    del route_load[rj]
    for c in final_route:
        route_of[c] = ri
    return True


def _savings_merge(depot, coords, demands, capacity, dist_fn, max_routes=None, feasibility_check=None):
    n = len(coords)
    routes = {i: [i] for i in range(n)}
    route_of = {i: i for i in range(n)}
    route_load = {i: demands[i] for i in range(n)}
    savings = []
    for i in range(n):
        for j in range(i + 1, n):
            s = dist_fn(depot, coords[i]) + dist_fn(depot, coords[j]) - dist_fn(coords[i], coords[j])
            savings.append((s, i, j))
    savings.sort(key=lambda t: t[0], reverse=True)
    for s, i, j in savings:
        if s <= 0:
            break
        _try_merge(routes, route_of, route_load, capacity, i, j, feasibility_check)
    if max_routes is not None:
        idx = 0
        while len(routes) > max_routes and idx < len(savings):
            s, i, j = savings[idx]
            idx += 1
            _try_merge(routes, route_of, route_load, capacity, i, j, feasibility_check)
    return list(routes.values())


def _route_distance(depot, coords, route, dist_fn):
    if not route:
        return 0.0
    d = dist_fn(depot, coords[route[0]])
    for a, b in zip(route, route[1:]):
        d += dist_fn(coords[a], coords[b])
    d += dist_fn(coords[route[-1]], depot)
    return d


def solve_cvrp_savings(instance: Instance) -> RoutingSolution:
    if instance.num_depots() != 1:
        raise ValueError("solve_cvrp_savings expects a single-depot instance.")
    start = time.time()
    depot = instance.depots[0]
    routes = _savings_merge(depot, instance.customers, instance.demands, instance.vehicle_capacity, instance.distance, max_routes=instance.num_vehicles_per_depot)
    total = sum(_route_distance(depot, instance.customers, r, instance.distance) for r in routes)
    elapsed = time.time() - start
    status = "Heuristic" if len(routes) <= instance.num_vehicles_per_depot else "Heuristic (fleet exceeded)"
    return RoutingSolution(routes=routes, total_distance=total, status=status, solve_time=elapsed)


def _tw_feasibility_factory(instance: Instance) -> Callable[[list[int]], Optional[list[int]]]:
    """
    Builds a feasibility_check closure for VRPTW savings merges: given a
    candidate merged route, simulate it (same logic as
    tests/test_vrptw.py's _simulate_route) and reject the merge if any
    time window or the depot's closing time is violated. Tries both
    orientations of the merged route before rejecting, since the
    orientation _try_merge picked for distance/capacity reasons isn't
    necessarily the one that respects time windows.
    """
    depot = instance.depots[0]
    coords = instance.customers
    tw = instance.time_windows
    service = instance.service_times
    speed = instance.speed
    depot_close = instance.depot_time_window[1]

    def simulate(route: list[int]) -> bool:
        t = 0.0
        prev = depot
        for c in route:
            travel = instance.distance(prev, coords[c]) / speed
            arrival = t + travel
            earliest, latest = tw[c]
            if arrival > latest:
                return False
            t = max(arrival, earliest) + service[c]
            prev = coords[c]
        return t + instance.distance(prev, depot) / speed <= depot_close

    def check(route: list[int]) -> Optional[list[int]]:
        if simulate(route):
            return route
        reversed_route = list(reversed(route))
        if simulate(reversed_route):
            return reversed_route
        return None

    return check


def solve_vrptw_savings(instance: Instance) -> RoutingSolution:
    if instance.num_depots() != 1:
        raise ValueError("solve_vrptw_savings expects a single-depot instance.")
    if instance.time_windows is None:
        raise ValueError("Instance has no time_windows; generate with generate_vrptw_instance().")

    start = time.time()
    depot = instance.depots[0]
    check = _tw_feasibility_factory(instance)
    routes = _savings_merge(
        depot, instance.customers, instance.demands, instance.vehicle_capacity, instance.distance,
        max_routes=instance.num_vehicles_per_depot, feasibility_check=check,
    )
    total = sum(_route_distance(depot, instance.customers, r, instance.distance) for r in routes)
    elapsed = time.time() - start
    status = "Heuristic" if len(routes) <= instance.num_vehicles_per_depot else "Heuristic (fleet exceeded)"
    return RoutingSolution(routes=routes, total_distance=total, status=status, solve_time=elapsed)