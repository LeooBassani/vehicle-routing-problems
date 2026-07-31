"""
Interactive Folium map: depot(s), real POIs, and routes drawn along their
actual street path (not straight lines). Supports single-depot (CVRP,
VRPTW) and multi-depot (MDVRP) route formats transparently.
"""

from __future__ import annotations

import folium

from vrp.road_network import shortest_path_route


ROUTE_COLORS = [
    "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd",
    "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf",
]
# folium's built-in icon colors, used for depot markers when there's more than one
DEPOT_ICON_COLORS = ["blue", "orange", "green", "red", "purple", "darkblue", "cadetblue"]


def _node_latlon(graph, node):
    data = graph.nodes[node]
    return data["y"], data["x"]  # (lat, lon)


def _decode_route(route):
    """
    Same convention as visualization.py's _decode_route: route[0] < 0
    means MDVRP format (route[0] = -(depot_id+1)); otherwise it's a plain
    single-depot route (CVRP/VRPTW), implicitly depot 0.
    """
    if route and route[0] < 0:
        return -route[0] - 1, route[1:]
    return 0, route


def build_route_map(graph, depot_nodes, customer_nodes, poi_records, routes, title=""):
    """
    `depot_nodes`: a single graph node id (single-depot case) or a list of
    them (multi-depot / MDVRP case) -- either is accepted.

    `routes`: list of routes, each a list of *local* customer indices
    (0-based, into `customer_nodes`/`poi_records`), optionally prefixed
    with a depot marker for MDVRP -- the same formats produced by every
    solver in this repo.
    """
    if not isinstance(depot_nodes, (list, tuple)):
        depot_nodes = [depot_nodes]
    multi_depot = len(depot_nodes) > 1

    center_lat, center_lon = _node_latlon(graph, depot_nodes[0])
    fmap = folium.Map(location=[center_lat, center_lon], zoom_start=14, tiles="cartodbpositron")

    for d, node in enumerate(depot_nodes):
        lat, lon = _node_latlon(graph, node)
        icon_color = DEPOT_ICON_COLORS[d % len(DEPOT_ICON_COLORS)] if multi_depot else "black"
        folium.Marker(
            [lat, lon],
            icon=folium.Icon(color=icon_color, icon="warehouse", prefix="fa"),
            popup=f"Depot {d}" if multi_depot else "Depot",
        ).add_to(fmap)

    for i, node in enumerate(customer_nodes):
        lat, lon = _node_latlon(graph, node)
        name = poi_records[i].name if poi_records else str(i)
        folium.CircleMarker(
            [lat, lon], radius=6, color="#333333", fill=True, fill_opacity=0.9, popup=name,
        ).add_to(fmap)
        folium.map.Marker(
            [lat, lon],
            icon=folium.DivIcon(html=f'<div style="font-size:10pt">{name}</div>'),
        ).add_to(fmap)

    for idx, route in enumerate(routes):
        depot_id, local_customers = _decode_route(route)
        color = ROUTE_COLORS[depot_id % len(ROUTE_COLORS)] if multi_depot else ROUTE_COLORS[idx % len(ROUTE_COLORS)]
        node_sequence = [depot_nodes[depot_id]] + [customer_nodes[c] for c in local_customers] + [depot_nodes[depot_id]]
        full_path = shortest_path_route(graph, node_sequence)
        latlon_path = [_node_latlon(graph, n) for n in full_path]
        label = f"Depot {depot_id} - route {idx}" if multi_depot else f"Route {idx}"
        folium.PolyLine(
            latlon_path, color=color, weight=4, opacity=0.85,
            tooltip=f"{label} ({len(local_customers)} stops)",
        ).add_to(fmap)

    if title:
        fmap.get_root().html.add_child(folium.Element(f'<h4 style="margin:8px">{title}</h4>'))
    return fmap