"""
Build a CVRP `Instance` from real Points of Interest (POIs) retrieved from
OpenStreetMap, instead of randomly generated customer locations.

This is intentionally a *separate* module from `instance.py`: it depends on
`osmnx` and outbound network access (Nominatim/Overpass), which the core
synthetic instance generators do not need. Keeping it separate means
`vrp.instance` stays fast, offline, and dependency-light for anyone just
running the test suite or the synthetic benchmark.

The output is a perfectly ordinary `Instance` -- the MILP models
(`vrp.models.*`) and the Clarke-Wright heuristic (`vrp.heuristics.*`) don't
need to know or care that the coordinates originated from real POIs instead
of `random.uniform()`. That's the payoff of having designed `Instance`
around plain local (x, y) coordinates in the first place.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass

from vrp.instance import Instance


# Default: Chelsea Industrial Area, MA -- used as the depot location unless
# the caller passes a different (lat, lon).
CHELSEA_INDUSTRIAL_AREA = (42.3915, -71.0330)  # (lat, lon)

# OSM tags for the POI categories requested: supermarkets, pharmacies,
# grocery/convenience stores, and general retail shops. See
# https://wiki.openstreetmap.org/wiki/Map_features for the full tag list.
DEFAULT_POI_TAGS = {
    "shop": ["supermarket", "convenience", "grocery", "general", "variety_store"],
    "amenity": ["pharmacy"],
}


@dataclass
class PoiRecord:
    """Metadata for a single real-world customer, kept alongside the
    Instance (not inside it) since Instance stays a generic, source-agnostic
    container -- see module docstring."""
    name: str
    category: str
    lat: float
    lon: float


def _latlon_to_local_xy(lat: float, lon: float, origin_lat: float, origin_lon: float) -> tuple[float, float]:
    """
    Equirectangular approximation: converts (lat, lon) to local (x, y) in
    meters relative to an origin point. Accurate enough at city scale (a
    few km), which is all this repo's MILP is meant to handle anyway (see
    docs/math_formulation.md's scaling note -- solving exactly stops being
    practical well before "city scale" regardless of coordinate system).
    """
    m_per_deg_lat = 111_320.0
    m_per_deg_lon = 111_320.0 * math.cos(math.radians(origin_lat))
    x = (lon - origin_lon) * m_per_deg_lon
    y = (lat - origin_lat) * m_per_deg_lat
    return x, y


def _extract_poi_records(gdf, category_tags: dict) -> list[PoiRecord]:
    """Turn an OSM features GeoDataFrame into a flat list of PoiRecord,
    using each feature's centroid (handles both point and polygon
    geometries -- OSM sometimes returns a building footprint instead of a
    single point for a given shop)."""
    records = []
    for _, row in gdf.iterrows():
        centroid = row.geometry.centroid
        category = None
        for tag_key in category_tags:
            value = row.get(tag_key)
            if isinstance(value, str):
                category = f"{tag_key}={value}"
                break
        name = row.get("name")
        if not isinstance(name, str) or not name.strip():
            name = category or "Unnamed POI"
        records.append(PoiRecord(name=name, category=category or "unknown", lat=centroid.y, lon=centroid.x))
    return records


def generate_cvrp_instance_from_osm(
    num_customers: int,
    depot_latlon: tuple[float, float] = CHELSEA_INDUSTRIAL_AREA,
    search_radius_m: float = 2000.0,
    poi_tags: dict = None,
    vehicle_capacity: float = 100.0,
    num_vehicles: int = 4,
    demand_range: tuple[int, int] = (5, 25),
    seed: int = 42,
    name: str = None,
) -> tuple[Instance, list[PoiRecord]]:
    """
    Build a CVRP Instance whose depot is a real location (default: Chelsea
    Industrial Area, MA) and whose customers are real POIs (supermarkets,
    pharmacies, grocery/convenience stores, retail shops) retrieved from
    OpenStreetMap within `search_radius_m` meters of the depot.

    Demand per customer is still synthetic (OSM has no notion of "how much
    this store orders"), generated the same seeded way as the other
    generators in `vrp.instance`.

    Requires the `osmnx` package and outbound network access to
    nominatim.openstreetmap.org / the Overpass API.

    Returns (instance, poi_records): `poi_records` has the same length and
    order as `instance.customers`, so `poi_records[i]` describes the real
    place behind `instance.customers[i]` -- useful later for map labels.
    """
    import osmnx as ox  # lazy import: only needed for this function

    if poi_tags is None:
        poi_tags = DEFAULT_POI_TAGS

    depot_lat, depot_lon = depot_latlon

    gdf = ox.features.features_from_point(depot_latlon, tags=poi_tags, dist=search_radius_m)
    if gdf.empty:
        raise ValueError(
            f"No POIs found within {search_radius_m}m of {depot_latlon}. "
            "Try a larger search_radius_m or different poi_tags."
        )

    all_records = _extract_poi_records(gdf, poi_tags)

    if len(all_records) < num_customers:
        raise ValueError(
            f"Only found {len(all_records)} matching POIs within {search_radius_m}m; "
            f"requested {num_customers} customers. Increase search_radius_m."
        )

    rng = random.Random(seed)
    poi_records = rng.sample(all_records, num_customers)

    customers = [
        _latlon_to_local_xy(r.lat, r.lon, depot_lat, depot_lon) for r in poi_records
    ]
    demands = [float(rng.randint(*demand_range)) for _ in range(num_customers)]

    instance = Instance(
        name=name or f"cvrp_chelsea_n{num_customers}_seed{seed}",
        depots=[(0.0, 0.0)],  # depot is the local-coordinate origin by construction
        customers=customers,
        demands=demands,
        vehicle_capacity=vehicle_capacity,
        num_vehicles_per_depot=num_vehicles,
    )
    return instance, poi_records