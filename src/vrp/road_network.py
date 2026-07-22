"""
Real road network and shortest-path distance computation,
using OSMnx (download) + NetworkX (shortest paths).

"""

from __future__ import annotations

import networkx as nx


def get_road_network(center_latlon: tuple[float, float], radius_m: float = 2500.0, network_type: str = "drive"):
    """
    Download the drivable street network within `radius_m` meters of
    `center_latlon` (lat, lon). Returns a networkx MultiDiGraph where every
    node has 'x' (longitude) / 'y' (latitude) attributes and every edge
    has a 'length' attribute in meters -- the standard OSMnx graph schema.
    """
    import osmnx as ox  # lazy import: only this module needs it

    graph = ox.graph_from_point(center_latlon, dist=radius_m, network_type=network_type)
    return graph


def nearest_node(graph, lat: float, lon: float):
    """Find the graph node closest to a given (lat, lon)."""
    import osmnx as ox
    return ox.distance.nearest_nodes(graph, lon, lat)


def shortest_path_distance_matrix(graph, nodes: list) -> list[list[float]]:
    """
    All-pairs shortest-path distance (meters, via edge 'length') among the
    given list of nodes. Uses one Dijkstra run per source node, which is
    efficient enough for the tens-of-nodes instances this repo's MILP can
    solve exactly.
    """
    n = len(nodes)
    matrix = [[0.0] * n for _ in range(n)]
    for i, src in enumerate(nodes):
        lengths = nx.single_source_dijkstra_path_length(graph, src, weight="length")
        for j, dst in enumerate(nodes):
            if i != j:
                if dst not in lengths:
                    raise ValueError(
                        f"Node {dst} is not reachable from node {src} on this graph "
                        "(likely a one-way-street issue -- try a larger radius_m)."
                    )
                matrix[i][j] = lengths[dst]
    return matrix


def shortest_path_route(graph, node_sequence: list) -> list:
    """
    Full node-by-node path (for map drawing), visiting `node_sequence` in
    order, each consecutive pair joined by its shortest path on the real
    street graph -- this is what makes the map show actual streets instead
    of straight lines between stops.
    """
    full_path = []
    for a, b in zip(node_sequence, node_sequence[1:]):
        segment = nx.shortest_path(graph, a, b, weight="length")
        if full_path:
            segment = segment[1:]  # avoid duplicating the shared node
        full_path.extend(segment)
    return full_path