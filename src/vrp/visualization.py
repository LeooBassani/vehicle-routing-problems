"""
Route visualization for CVRP solutions using matplotlib.
"""

from __future__ import annotations

import matplotlib.pyplot as plt

from vrp.instance import Instance
from vrp.models.cvrp import RoutingSolution


ROUTE_COLORS = [
    "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd",
    "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf",
]


def plot_solution(
    instance: Instance,
    solution: RoutingSolution,
    labels: list[str] | None = None,
    title: str | None = None,
    save_path: str | None = None,
):
    """
    Plot the depot, customers, and routes of a (single-depot) CVRP solution.

    `labels`: optional list, same length/order as instance.customers, used
    to annotate each point instead of the default numeric index.
    """
    fig, ax = plt.subplots(figsize=(8, 8))

    depot_x, depot_y = instance.depots[0]
    ax.scatter([depot_x], [depot_y], marker="s", s=200, color="black", zorder=5, label="Depot")

    cx = [c[0] for c in instance.customers]
    cy = [c[1] for c in instance.customers]
    ax.scatter(cx, cy, color="#444444", s=50, zorder=4)

    for i, (x, y) in enumerate(instance.customers):
        label = labels[i] if labels else str(i)
        ax.annotate(label, (x, y), textcoords="offset points", xytext=(6, 6), fontsize=8)

    for idx, route in enumerate(solution.routes):
        color = ROUTE_COLORS[idx % len(ROUTE_COLORS)]
        points = [instance.depots[0]] + [instance.customers[c] for c in route] + [instance.depots[0]]
        xs, ys = zip(*points)
        ax.plot(xs, ys, color=color, linewidth=2, alpha=0.85, marker="o", markersize=4,
                label=f"Route {idx} ({len(route)} stops)")

    ax.set_title(title or f"{solution.status} | total distance = {solution.total_distance:.1f}")
    ax.set_xlabel("x (meters, local projection)")
    ax.set_ylabel("y (meters, local projection)")
    ax.legend(fontsize=8, loc="best")
    ax.set_aspect("equal", adjustable="datalim")
    ax.grid(alpha=0.2)

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig