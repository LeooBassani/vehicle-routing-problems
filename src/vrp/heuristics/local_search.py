from __future__ import annotations

import time

from vrp.instance import Instance
from vrp.models.cvrp import RoutingSolution
from vrp.heuristics.clarke_wright import solve_cvrp_savings, solve_vrptw_savings, _route_distance, _tw_simulate_factory


def two_opt(route, depot, coords, dist_fn, feasibility_check=None, max_iterations=1000):
    """
    Refine a single route's internal ordering by repeatedly undoing
    crossings: remove two arcs, reconnect the only other valid way (which
    reverses the segment between them), keep the change if it shortens
    the route. Depot stays fixed at both ends -- only the customer
    ordering in between changes.

    `feasibility_check(route) -> bool`: optional, e.g. for VRPTW. Since
    reversing a segment changes arrival times at every stop downstream of
    it, a distance-improving swap can still violate a time window -- the
    swap is only applied if it also passes this check.
    """
    route = route[:]
    n = len(route)
    if n < 2:
        return route

    improved = True
    iterations = 0
    while improved and iterations < max_iterations:
        improved = False
        iterations += 1
        for i in range(-1, n - 1):
            a = depot if i == -1 else coords[route[i]]
            b = coords[route[i + 1]]
            for j in range(i + 1, n):
                c = coords[route[j]]
                d = depot if j == n - 1 else coords[route[j + 1]]
                delta = (dist_fn(a, c) + dist_fn(b, d)) - (dist_fn(a, b) + dist_fn(c, d))
                if delta < -1e-9:
                    candidate = route[:]
                    candidate[i + 1:j + 1] = list(reversed(candidate[i + 1:j + 1]))
                    if feasibility_check is not None and not feasibility_check(candidate):
                        continue
                    route = candidate
                    improved = True
    return route


def or_opt(routes, depot, coords, demands, capacity, dist_fn, feasibility_check=None,
           segment_lengths=(1, 2, 3), max_iterations=1000):
    """
    Refine across all routes at once: try relocating a short chain of 1-3
    consecutive customers (in either orientation) to a better position,
    in the same route or a different one, subject to vehicle capacity.

    `feasibility_check(route) -> bool`: optional, e.g. for VRPTW. Applied
    to both the destination route (after insertion) and the source route
    (after removal, when moving between different routes) before a move
    is accepted -- inserting a segment shifts arrival times for every
    subsequent stop in that route, which can turn a distance-improving
    move into a time-window violation.
    """
    routes = [r[:] for r in routes]
    route_load = [sum(demands[c] for c in r) for r in routes]

    improved = True
    iterations = 0
    while improved and iterations < max_iterations:
        improved = False
        iterations += 1

        for src_idx, src_route in enumerate(routes):
            for seg_len in segment_lengths:
                if seg_len > len(src_route):
                    continue
                for start in range(len(src_route) - seg_len + 1):
                    segment = src_route[start:start + seg_len]
                    seg_demand = sum(demands[c] for c in segment)
                    remaining_src = src_route[:start] + src_route[start + seg_len:]

                    prev_point = depot if start == 0 else coords[src_route[start - 1]]
                    next_point = depot if start + seg_len == len(src_route) else coords[src_route[start + seg_len]]
                    removal_gain = (
                        dist_fn(prev_point, coords[segment[0]])
                        + dist_fn(coords[segment[-1]], next_point)
                        - dist_fn(prev_point, next_point)
                    )

                    best_delta = -1e-9
                    best_target = None  # (dst_idx, position, oriented_segment)

                    for dst_idx, dst_route in enumerate(routes):
                        if dst_idx == src_idx:
                            working_route = remaining_src
                        else:
                            if route_load[dst_idx] + seg_demand > capacity:
                                continue
                            working_route = dst_route

                        for pos in range(len(working_route) + 1):
                            u = depot if pos == 0 else coords[working_route[pos - 1]]
                            v = depot if pos == len(working_route) else coords[working_route[pos]]
                            for seg in (segment, list(reversed(segment))):
                                insertion_cost = (
                                    dist_fn(u, coords[seg[0]]) + dist_fn(coords[seg[-1]], v) - dist_fn(u, v)
                                )
                                delta = insertion_cost - removal_gain
                                if delta < best_delta:
                                    candidate_dst = working_route[:pos] + seg + working_route[pos:]
                                    if feasibility_check is not None:
                                        if not feasibility_check(candidate_dst):
                                            continue
                                        if dst_idx != src_idx and not feasibility_check(remaining_src):
                                            continue
                                    best_delta = delta
                                    best_target = (dst_idx, pos, seg)

                    if best_target is not None:
                        dst_idx, pos, seg = best_target
                        del src_route[start:start + seg_len]
                        route_load[src_idx] -= seg_demand
                        routes[dst_idx][pos:pos] = seg
                        route_load[dst_idx] += seg_demand
                        improved = True
                        break
                if improved:
                    break
            if improved:
                break

    return [r for r in routes if r]  # drop any route emptied out by relocation


def local_search(routes, depot, coords, demands, capacity, dist_fn, feasibility_check=None, max_rounds=20):
    """
    Alternates 2-opt (per route) and Or-opt (across routes) until neither
    finds any further improvement -- an Or-opt move can create a new
    crossing that 2-opt can then undo, and vice versa, so a single pass
    of each isn't enough.
    """
    for _ in range(max_rounds):
        new_routes = [two_opt(r, depot, coords, dist_fn, feasibility_check=feasibility_check) for r in routes]
        new_routes = or_opt(new_routes, depot, coords, demands, capacity, dist_fn, feasibility_check=feasibility_check)
        if new_routes == routes:
            break
        routes = new_routes
    return routes


def solve_cvrp_savings_refined(instance: Instance) -> RoutingSolution:
    """
    Runs Clarke-Wright construction (solve_cvrp_savings), then refines the
    result with 2-opt + Or-opt local search. Local search only ever keeps
    or improves the distance -- it never makes a valid solution worse or
    invalid -- so this always dominates solve_cvrp_savings on its own,
    at the cost of a bit more computation (still milliseconds, not seconds).
    """
    base = solve_cvrp_savings(instance)
    depot = instance.depots[0]

    start = time.time()
    refined_routes = local_search(
        base.routes, depot, instance.customers, instance.demands, instance.vehicle_capacity, instance.distance,
    )
    elapsed = base.solve_time + (time.time() - start)
    total = sum(_route_distance(depot, instance.customers, r, instance.distance) for r in refined_routes)

    status = base.status.replace("Heuristic", "Heuristic+2opt/Or-opt")
    return RoutingSolution(routes=refined_routes, total_distance=total, status=status, solve_time=elapsed)


def solve_vrptw_savings_refined(instance: Instance) -> RoutingSolution:
    """
    Same idea as solve_cvrp_savings_refined, but time-window aware: every
    2-opt/Or-opt candidate move is simulated and rejected if it would
    violate a time window, using the same simulation logic as
    solve_vrptw_savings' construction phase.
    """
    base = solve_vrptw_savings(instance)
    depot = instance.depots[0]
    feasibility_check = _tw_simulate_factory(instance)

    start = time.time()
    refined_routes = local_search(
        base.routes, depot, instance.customers, instance.demands, instance.vehicle_capacity, instance.distance,
        feasibility_check=feasibility_check,
    )
    elapsed = base.solve_time + (time.time() - start)
    total = sum(_route_distance(depot, instance.customers, r, instance.distance) for r in refined_routes)

    status = base.status.replace("Heuristic", "Heuristic+2opt/Or-opt")
    return RoutingSolution(routes=refined_routes, total_distance=total, status=status, solve_time=elapsed)