"""
Instance representation and synthetic instance generation for VRP variants.

Design choice
-------------
We use a *single* Instance dataclass that can represent CVRP, VRPTW and
MDVRP alike. Fields that don't apply to a given variant are simply left at
their default (e.g. `time_windows=None` for plain CVRP). This keeps the
generator, visualization and benchmarking code variant-agnostic, while each
`models/*.py` module only reads the fields it actually needs.

All instances are synthetic (randomly generated with a fixed seed for
reproducibility) rather than pulled from classical benchmark sets. This is
intentional: it lets us control instance size, depot count and time-window
tightness directly, which is what the benchmarking narrative in this repo
needs (see docs/math_formulation.md and README.md).
"""

from __future__ import annotations

import json
import math
import random
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Optional


Coordinate = tuple[float, float]


@dataclass
class Instance:
    name: str
    depots: list[Coordinate]              # 1 depot for CVRP/VRPTW, >=2 for MDVRP
    customers: list[Coordinate]            # customer coordinates, index 0..n-1
    demands: list[float]                   # demand per customer, same order as customers
    vehicle_capacity: float
    num_vehicles_per_depot: int
    # VRPTW-only fields (None when not applicable)
    time_windows: Optional[list[tuple[float, float]]] = None   # (earliest, latest) per customer
    service_times: Optional[list[float]] = None                # service duration per customer
    depot_time_window: Optional[tuple[float, float]] = None    # operating window for all depots
    speed: float = 1.0                     # distance units per time unit, for TW instances

    def num_customers(self) -> int:
        return len(self.customers)

    def num_depots(self) -> int:
        return len(self.depots)

    def distance(self, p: Coordinate, q: Coordinate) -> float:
        return math.hypot(p[0] - q[0], p[1] - q[1])

    def all_nodes(self) -> list[Coordinate]:
        """Depots first (indices 0..D-1), then customers (indices D..D+N-1)."""
        return list(self.depots) + list(self.customers)

    def distance_matrix(self) -> list[list[float]]:
        nodes = self.all_nodes()
        n = len(nodes)
        return [[self.distance(nodes[i], nodes[j]) for j in range(n)] for i in range(n)]

    def to_json(self, path: str | Path) -> None:
        data = asdict(self)
        Path(path).write_text(json.dumps(data, indent=2))

    @staticmethod
    def from_json(path: str | Path) -> "Instance":
        data = json.loads(Path(path).read_text())
        data["depots"] = [tuple(p) for p in data["depots"]]
        data["customers"] = [tuple(p) for p in data["customers"]]
        if data.get("time_windows") is not None:
            data["time_windows"] = [tuple(tw) for tw in data["time_windows"]]
        if data.get("depot_time_window") is not None:
            data["depot_time_window"] = tuple(data["depot_time_window"])
        return Instance(**data)


def _random_points(n: int, rng: random.Random, extent: float = 100.0) -> list[Coordinate]:
    return [(rng.uniform(0, extent), rng.uniform(0, extent)) for _ in range(n)]


def generate_cvrp_instance(
    num_customers: int,
    vehicle_capacity: float = 100.0,
    num_vehicles: int = 4,
    demand_range: tuple[int, int] = (5, 25),
    extent: float = 100.0,
    seed: int = 42,
    name: Optional[str] = None,
) -> Instance:
    """Single-depot CVRP instance: one depot at the map centroid, random customers."""
    rng = random.Random(seed)
    depot = (extent / 2, extent / 2)
    customers = _random_points(num_customers, rng, extent)
    demands = [rng.randint(*demand_range) for _ in range(num_customers)]
    return Instance(
        name=name or f"cvrp_n{num_customers}_seed{seed}",
        depots=[depot],
        customers=customers,
        demands=demands,
        vehicle_capacity=vehicle_capacity,
        num_vehicles_per_depot=num_vehicles,
    )


def generate_vrptw_instance(
    num_customers: int,
    vehicle_capacity: float = 100.0,
    num_vehicles: int = 4,
    demand_range: tuple[int, int] = (5, 25),
    horizon: float = 240.0,
    tw_width_range: tuple[float, float] = (20.0, 60.0),
    service_time: float = 10.0,
    speed: float = 1.0,
    extent: float = 100.0,
    seed: int = 42,
    name: Optional[str] = None,
) -> Instance:
    """
    Single-depot VRPTW instance. Time windows are generated to guarantee
    that every customer is feasible *even when served alone*: the window is
    placed so that a vehicle can (1) arrive from the depot no earlier than
    `earliest`, and (2) after waiting if needed and completing service,
    still travel back to the depot before the depot's closing time. This
    matters because a merge heuristic (see heuristics/clarke_wright.py) can
    leave a customer in a singleton route, and a synthetic benchmark
    instance should not silently be infeasible for the trivial one-vehicle-
    per-customer solution.

    If `horizon` is too tight relative to `extent` for a given customer
    (i.e. a round trip alone cannot fit even with a zero-width window), the
    window degenerates to the tightest feasible single point in time --
    this only happens for customers placed very far from the depot on a
    very short horizon, and is a sign the caller should widen `horizon` or
    shrink `extent`.
    """
    rng = random.Random(seed)
    depot = (extent / 2, extent / 2)
    customers = _random_points(num_customers, rng, extent)
    demands = [rng.randint(*demand_range) for _ in range(num_customers)]

    time_windows = []
    for (cx, cy) in customers:
        travel = math.hypot(cx - depot[0], cy - depot[1]) / speed
        # Latest possible departure time that still allows returning to the
        # depot (with round-trip travel, assumed symmetric) before horizon.
        latest_departure = horizon - service_time - travel
        latest_upper = max(travel, latest_departure)  # don't go below arrival time

        width = rng.uniform(*tw_width_range)
        width = min(width, max(0.0, latest_upper - travel))

        span = max(0.0, latest_upper - width - travel)
        earliest = travel + rng.uniform(0, span) if span > 0 else travel
        latest = min(latest_upper, earliest + width)
        time_windows.append((round(earliest, 1), round(latest, 1)))

    return Instance(
        name=name or f"vrptw_n{num_customers}_seed{seed}",
        depots=[depot],
        customers=customers,
        demands=demands,
        vehicle_capacity=vehicle_capacity,
        num_vehicles_per_depot=num_vehicles,
        time_windows=time_windows,
        service_times=[service_time] * num_customers,
        depot_time_window=(0.0, horizon),
        speed=speed,
    )


def generate_mdvrp_instance(
    num_customers: int,
    num_depots: int = 2,
    vehicle_capacity: float = 100.0,
    num_vehicles_per_depot: int = 3,
    demand_range: tuple[int, int] = (5, 25),
    extent: float = 100.0,
    seed: int = 42,
    name: Optional[str] = None,
) -> Instance:
    """Multi-depot CVRP instance: depots spread across the map, customers random."""
    rng = random.Random(seed)
    # Spread depots roughly evenly rather than fully at random, to avoid
    # degenerate instances where two depots land on top of each other.
    depots = []
    for i in range(num_depots):
        angle = 2 * math.pi * i / num_depots
        r = extent * 0.3
        cx, cy = extent / 2, extent / 2
        depots.append((cx + r * math.cos(angle), cy + r * math.sin(angle)))

    customers = _random_points(num_customers, rng, extent)
    demands = [rng.randint(*demand_range) for _ in range(num_customers)]

    return Instance(
        name=name or f"mdvrp_n{num_customers}_d{num_depots}_seed{seed}",
        depots=depots,
        customers=customers,
        demands=demands,
        vehicle_capacity=vehicle_capacity,
        num_vehicles_per_depot=num_vehicles_per_depot,
    )