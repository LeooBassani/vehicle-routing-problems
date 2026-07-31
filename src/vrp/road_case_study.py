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
from vrp.models.vrptw import solve_vrptw_from_distance_matrix
from vrp.models.mdvrp import solve_mdvrp_from_distance_matrix
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


@dataclass
class VRPTWRoadCaseStudyResult:
    graph: object
    depot_node: int
    customer_nodes: list
    poi_records: list[PoiRecord]
    dist_matrix_km: list[list[float]]
    time_windows: list[tuple[float, float]]
    milp_solution: RoutingSolution


def solve_chelsea_vrptw_with_road_distances(
    num_customers: int,
    depot_latlon: tuple[float, float] = CHELSEA_INDUSTRIAL_AREA,
    search_radius_m: float = 2000.0,
    poi_tags: dict = None,
    vehicle_capacity: float = 100.0,
    num_vehicles: int = 4,
    demand_range: tuple[int, int] = (5, 25),
    seed: int = 42,
    speed_kmh: float = 30.0,
    horizon_minutes: float = 240.0,
    tw_width_range: tuple[float, float] = (30.0, 90.0),
    service_minutes: float = 10.0,
    milp_time_limit: int = 60,
) -> VRPTWRoadCaseStudyResult:
    """
    Same idea as solve_chelsea_with_road_distances, plus synthetic time
    windows generated from REAL road travel times (not straight-line) --
    same feasibility-guaranteeing logic as instance.generate_vrptw_instance
    (every customer's window allows a round trip alone within the
    horizon), just fed by the real distance matrix's depot row instead of
    Euclidean distance.
    """
    import osmnx as ox

    if poi_tags is None:
        poi_tags = DEFAULT_POI_TAGS

    gdf = ox.features.features_from_point(depot_latlon, tags=poi_tags, dist=search_radius_m)
    if gdf.empty:
        raise ValueError(f"No POIs found within {search_radius_m}m of {depot_latlon}.")
    all_records = _extract_poi_records(gdf, poi_tags)
    if len(all_records) < num_customers:
        raise ValueError(f"Only found {len(all_records)} POIs; requested {num_customers}.")
    rng = random.Random(seed)
    poi_records = rng.sample(all_records, num_customers)
    demands = [float(rng.randint(*demand_range)) for _ in range(num_customers)]

    graph = get_road_network(depot_latlon, radius_m=search_radius_m * 1.3)
    depot_node = nearest_node(graph, *depot_latlon)
    customer_nodes = [nearest_node(graph, r.lat, r.lon) for r in poi_records]
    nodes = [depot_node] + customer_nodes

    dist_matrix_m = shortest_path_distance_matrix(graph, nodes)
    dist_matrix_km = [[d / 1000.0 for d in row] for row in dist_matrix_m]

    # speed expressed in km/minute, so that dividing a km distance by it
    # gives minutes directly -- matches the minute-based horizon below.
    speed_km_per_min = speed_kmh / 60.0

    time_windows = []
    for i in range(num_customers):
        travel = dist_matrix_km[0][i + 1] / speed_km_per_min
        latest_departure = horizon_minutes - service_minutes - travel
        latest_upper = max(travel, latest_departure)
        width = rng.uniform(*tw_width_range)
        width = min(width, max(0.0, latest_upper - travel))
        span = max(0.0, latest_upper - width - travel)
        earliest = travel + rng.uniform(0, span) if span > 0 else travel
        latest = min(latest_upper, earliest + width)
        time_windows.append((round(earliest, 1), round(latest, 1)))

    milp_demands = [0.0] + demands
    service_times = [service_minutes] * num_customers
    depot_time_window = (0.0, horizon_minutes)

    milp_solution = solve_vrptw_from_distance_matrix(
        dist_matrix_km, milp_demands, vehicle_capacity, num_vehicles,
        time_windows, service_times, depot_time_window, speed=speed_km_per_min,
        time_limit=milp_time_limit,
    )

    return VRPTWRoadCaseStudyResult(
        graph=graph, depot_node=depot_node, customer_nodes=customer_nodes, poi_records=poi_records,
        dist_matrix_km=dist_matrix_km, time_windows=time_windows, milp_solution=milp_solution,
    )


@dataclass
class MDVRPRoadCaseStudyResult:
    graph: object
    depot_nodes: list
    customer_nodes: list
    poi_records: list[PoiRecord]
    dist_matrix_km: list[list[float]]
    milp_solution: RoutingSolution


def solve_boston_mdvrp_with_road_distances(
    num_customers: int,
    depot_latlons: list[tuple[float, float]] = None,
    search_radius_m: float = 2000.0,
    poi_tags: dict = None,
    vehicle_capacity: float = 80.0,
    num_vehicles_per_depot: int = 3,
    demand_range: tuple[int, int] = (5, 25),
    seed: int = 42,
    milp_time_limit: int = 90,
) -> MDVRPRoadCaseStudyResult:
    """
    MDVRP version: two (or more) real depot locations, POIs pooled from
    around all of them, one real road network large enough to cover
    everything, one real shortest-path distance matrix over depots +
    customers together, solved with the exact MDVRP MILP.
    """
    import math
    import osmnx as ox

    if depot_latlons is None:
        # Chelsea Industrial Area + a second point a few km away in Boston proper
        depot_latlons = [CHELSEA_INDUSTRIAL_AREA, (42.3550, -71.0550)]
    if poi_tags is None:
        poi_tags = DEFAULT_POI_TAGS
    D = len(depot_latlons)

    mid_lat = sum(lat for lat, _ in depot_latlons) / D
    mid_lon = sum(lon for _, lon in depot_latlons) / D

    all_records = []
    seen = set()
    for depot_latlon in depot_latlons:
        gdf = ox.features.features_from_point(depot_latlon, tags=poi_tags, dist=search_radius_m)
        if gdf.empty:
            continue
        for rec in _extract_poi_records(gdf, poi_tags):
            key = (rec.name, round(rec.lat, 5), round(rec.lon, 5))
            if key not in seen:
                seen.add(key)
                all_records.append(rec)
    if len(all_records) < num_customers:
        raise ValueError(f"Only found {len(all_records)} POIs across all depots; requested {num_customers}.")

    rng = random.Random(seed)
    poi_records = rng.sample(all_records, num_customers)
    demands = [float(rng.randint(*demand_range)) for _ in range(num_customers)]

    m_per_deg_lat = 111_320.0
    max_dist_m = 0.0
    for lat, lon in depot_latlons:
        d = math.hypot((lat - mid_lat) * m_per_deg_lat, (lon - mid_lon) * m_per_deg_lat * math.cos(math.radians(mid_lat)))
        max_dist_m = max(max_dist_m, d)
    graph = get_road_network((mid_lat, mid_lon), radius_m=max_dist_m + search_radius_m * 1.3)

    depot_nodes = [nearest_node(graph, lat, lon) for lat, lon in depot_latlons]
    customer_nodes = [nearest_node(graph, r.lat, r.lon) for r in poi_records]
    nodes = depot_nodes + customer_nodes

    dist_matrix_m = shortest_path_distance_matrix(graph, nodes)
    dist_matrix_km = [[d / 1000.0 for d in row] for row in dist_matrix_m]

    milp_demands = [0.0] * D + demands
    milp_solution = solve_mdvrp_from_distance_matrix(
        dist_matrix_km, milp_demands, D, vehicle_capacity, num_vehicles_per_depot, time_limit=milp_time_limit,
    )

    return MDVRPRoadCaseStudyResult(
        graph=graph, depot_nodes=depot_nodes, customer_nodes=customer_nodes, poi_records=poi_records,
        dist_matrix_km=dist_matrix_km, milp_solution=milp_solution,
    )