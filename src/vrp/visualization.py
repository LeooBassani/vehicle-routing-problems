"""
Route visualization for CVRP, VRPTW, and MDVRP solutions using matplotlib.
"""

from __future__ import annotations

import matplotlib.pyplot as plt

from vrp.instance import Instance
from vrp.models.cvrp import RoutingSolution


DEPOT_COLORS = [
    "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd",
    "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf",
]


def _decode_route(route: list[int]):
    """
    Routes come in two formats depending on the solver:
      - CVRP/VRPTW (single depot): plain list of customer indices, e.g. [2, 0, 5]
      - MDVRP (multi-depot): route[0] = -(depot_id+1), rest are customer
        indices, e.g. [-1, 2, 0, 5] means depot 0, customers [2, 0, 5]

    Returns (depot_id, customer_indices) either way, so the rest of the
    plotting code never needs an if/else on which variant it's drawing.
    """
    if route and route[0] < 0:
        return -route[0] - 1, route[1:]
    return 0, route


def plot_solution(
    instance: Instance,
    solution: RoutingSolution,
    labels: list[str] | None = None,
    title: str | None = None,
    save_path: str | None = None,
):
    """
    Plot depots, customers, and routes for any of the three variants in
    this repo (CVRP, VRPTW, MDVRP) -- the route format is auto-detected
    per route via `_decode_route`, and single- vs multi-depot instances
    both work without the caller needing to specify which.

    `labels`: optional list, same length/order as instance.customers, used
    to annotate each point (e.g. real store names from real_world.py's
    poi_records) instead of the default numeric index.
    """
    fig, ax = plt.subplots(figsize=(8, 8))

    for d, (dx, dy) in enumerate(instance.depots):
        color = DEPOT_COLORS[d % len(DEPOT_COLORS)]
        ax.scatter([dx], [dy], marker="s", s=200, color=color, edgecolor="black",
                   linewidth=1.5, zorder=5, label=f"Depot {d}" if instance.num_depots() > 1 else "Depot")

    cx = [c[0] for c in instance.customers]
    cy = [c[1] for c in instance.customers]
    ax.scatter(cx, cy, color="#444444", s=50, zorder=4)

    for i, (x, y) in enumerate(instance.customers):
        label = labels[i] if labels else str(i)
        ax.annotate(label, (x, y), textcoords="offset points", xytext=(6, 6), fontsize=8)

    for idx, route in enumerate(solution.routes):
        depot_id, customer_ids = _decode_route(route)
        color = DEPOT_COLORS[depot_id % len(DEPOT_COLORS)]
        depot_xy = instance.depots[depot_id]
        points = [depot_xy] + [instance.customers[c] for c in customer_ids] + [depot_xy]
        xs, ys = zip(*points)
        route_label = (f"Depot {depot_id} route {idx} ({len(customer_ids)} stops)"
                       if instance.num_depots() > 1 else f"Route {idx} ({len(customer_ids)} stops)")
        ax.plot(xs, ys, color=color, linewidth=2, alpha=0.85, marker="o", markersize=4, label=route_label)

    ax.set_title(title or f"{solution.status} | total distance = {solution.total_distance:.1f}")
    ax.set_xlabel("x (meters, local projection)")
    ax.set_ylabel("y (meters, local projection)")
    ax.legend(fontsize=7, loc="best")
    ax.set_aspect("equal", adjustable="datalim")
    ax.grid(alpha=0.2)

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig