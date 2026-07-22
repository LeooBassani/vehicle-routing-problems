"""
Interactive Folium map: depot, real POIs, and routes drawn along their
actual street path.
"""

from __future__ import annotations

import folium

from vrp.road_network import shortest_path_route


ROUTE_COLORS = [
    "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd",
    "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf",
]


def _node_latlon(graph, node):
    data = graph.nodes[node]
    return data["y"], data["x"]  # (lat, lon)


def build_route_map(graph, depot_node, customer_nodes, poi_records, routes, title=""):
    """
    `routes`: list of routes, each a list of *local* customer indices
    (0-based, into `customer_nodes`/`poi_records`) -- the same format
    returned by every solver in this repo.
    """
    depot_lat, depot_lon = _node_latlon(graph, depot_node)
    fmap = folium.Map(location=[depot_lat, depot_lon], zoom_start=15, tiles="cartodbpositron")

    folium.Marker(
        [depot_lat, depot_lon],
        icon=folium.Icon(color="black", icon="warehouse", prefix="fa"),
        popup="Depot",
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
        color = ROUTE_COLORS[idx % len(ROUTE_COLORS)]
        node_sequence = [depot_node] + [customer_nodes[c] for c in route] + [depot_node]
        full_path = shortest_path_route(graph, node_sequence)
        latlon_path = [_node_latlon(graph, n) for n in full_path]
        folium.PolyLine(
            latlon_path, color=color, weight=4, opacity=0.85, tooltip=f"Route {idx} ({len(route)} stops)",
        ).add_to(fmap)

    if title:
        fmap.get_root().html.add_child(folium.Element(f'<h4 style="margin:8px">{title}</h4>'))
    return fmap