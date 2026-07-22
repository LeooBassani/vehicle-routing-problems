"""
Real POIs (Chelsea Industrial Area) + real road-network
shortest-path distances -> MILP + Clarke-Wright+2opt/Or-opt.

This ties together real_world.py (POI retrieval), road_network.py (street
graph + shortest paths), models/cvrp.py (solve_cvrp_from_distance_matrix),
and heuristics/*.py (the low-level, distance-matrix-agnostic functions).
"""

from __future__ import annotations

import random
import time
from dataclasses import dataclass

from vrp.models.cvrp import RoutingSolution, solve_cvrp_from_distance_matrix
from vrp.heuristics.clarke_wright import _savings_merge, _route_distance
from vrp.heuristics.local_search import local_search
from vrp.real_world import CHELSEA_INDUSTRIAL_AREA, DEFAULT_POI_TAGS, PoiRecord, _extract_poi_records
from vrp.road_network import get_road_network, nearest_node, shortest_path_distance_matrix


@dataclass
class RoadCaseStudyResult:
    graph: object
    depot_node: int
    customer_nodes: list
    poi_records: list[PoiRecord]
    dist_matrix_km: list[list[float]]
    milp_solution: RoutingSolution
    heuristic_solution: RoutingSolution


def solve_chelsea_with_road_distances(
    num_customers: int,
    depot_latlon: tuple[float, float] = CHELSEA_INDUSTRIAL_AREA,
    search_radius_m: float = 2000.0,
    poi_tags: dict = None,
    vehicle_capacity: float = 100.0,
    num_vehicles: int = 4,
    demand_range: tuple[int, int] = (5, 25),
    seed: int = 42,
    milp_time_limit: int = 60,
) -> RoadCaseStudyResult:
    import osmnx as ox  # lazy import

    if poi_tags is None:
        poi_tags = DEFAULT_POI_TAGS

    # 1. Real POIs near the depot (same source as real_world.py)
    gdf = ox.features.features_from_point(depot_latlon, tags=poi_tags, dist=search_radius_m)
    if gdf.empty:
        raise ValueError(f"No POIs found within {search_radius_m}m of {depot_latlon}.")
    all_records = _extract_poi_records(gdf, poi_tags)
    if len(all_records) < num_customers:
        raise ValueError(f"Only found {len(all_records)} POIs; requested {num_customers}.")
    rng = random.Random(seed)
    poi_records = rng.sample(all_records, num_customers)
    demands = [float(rng.randint(*demand_range)) for _ in range(num_customers)]

    # 2. Real road network around the depot
    graph = get_road_network(depot_latlon, radius_m=search_radius_m * 1.3)

    # 3. Snap depot + each POI to its nearest graph node
    depot_node = nearest_node(graph, *depot_latlon)
    customer_nodes = [nearest_node(graph, r.lat, r.lon) for r in poi_records]
    nodes = [depot_node] + customer_nodes

    # 4. Real shortest-path distance matrix (meters -> km)
    dist_matrix_m = shortest_path_distance_matrix(graph, nodes)
    dist_matrix_km = [[d / 1000.0 for d in row] for row in dist_matrix_m]

    # 5. Exact MILP, using the real distance matrix directly
    milp_demands = [0.0] + demands
    milp_solution = solve_cvrp_from_distance_matrix(
        dist_matrix_km, milp_demands, vehicle_capacity, num_vehicles, time_limit=milp_time_limit,
    )

    # 6. Clarke-Wright + 2opt/Or-opt, using the same real distances.
    # "coords" here are graph node ids (not x/y!) -- dist_fn looks them up
    # in the precomputed matrix, since _savings_merge/local_search only
    # ever call dist_fn(a, b) without assuming what a and b actually are.
    node_index = {node: i for i, node in enumerate(nodes)}

    def dist_fn(u, v):
        return dist_matrix_km[node_index[u]][node_index[v]]

    start = time.time()
    heur_routes_by_node = _savings_merge(
        depot_node, customer_nodes, demands, vehicle_capacity, dist_fn, max_routes=num_vehicles,
    )
    refined_routes = local_search(
        heur_routes_by_node, depot_node, customer_nodes, demands, vehicle_capacity, dist_fn,
    )
    elapsed = time.time() - start
    total = sum(_route_distance(depot_node, customer_nodes, r, dist_fn) for r in refined_routes)
    status = "Heuristic+2opt/Or-opt" if len(refined_routes) <= num_vehicles else "Heuristic+2opt/Or-opt (fleet exceeded)"
    heuristic_solution = RoutingSolution(routes=refined_routes, total_distance=total, status=status, solve_time=elapsed)

    return RoadCaseStudyResult(
        graph=graph, depot_node=depot_node, customer_nodes=customer_nodes, poi_records=poi_records,
        dist_matrix_km=dist_matrix_km, milp_solution=milp_solution, heuristic_solution=heuristic_solution,
    )