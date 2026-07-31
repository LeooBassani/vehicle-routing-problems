"""
Formal benchmark: MILP vs Clarke-Wright Savings vs Savings+2opt/Or-opt, 
across a sweep of instance sizes and seeds, for all
three variants in this repo (CVRP, VRPTW, MDVRP).

Produces one combined per-instance CSV (results/benchmark_results.csv)
and a per-variant, per-size summary printed to the console.

Run as a script:
    PYTHONPATH=src python -m vrp.benchmark
"""

from __future__ import annotations

import csv
import statistics
from dataclasses import dataclass, asdict
from pathlib import Path

from vrp.instance import generate_cvrp_instance, generate_vrptw_instance, generate_mdvrp_instance
from vrp.models.cvrp import solve_cvrp_milp
from vrp.models.vrptw import solve_vrptw_milp
from vrp.models.mdvrp import solve_mdvrp_milp
from vrp.heuristics.clarke_wright import solve_cvrp_savings, solve_vrptw_savings, solve_mdvrp_savings
from vrp.heuristics.local_search import solve_cvrp_savings_refined, solve_vrptw_savings_refined


@dataclass
class BenchmarkRow:
    variant: str
    num_customers: int
    seed: int
    milp_status: str
    milp_distance: float
    milp_time: float
    savings_distance: float
    savings_time: float
    savings_status: str
    refined_distance: float   # for MDVRP (no refinement step built), equals savings_distance
    refined_time: float
    gap_savings_pct: float | None    # None when not comparable (fleet exceeded, or MILP not Optimal)
    gap_refined_pct: float | None


def _gap(milp_status: str, milp_distance: float, heuristic_status: str, distance: float) -> float | None:
    if milp_status != "Optimal":
        return None  # MILP itself didn't prove optimality (or was infeasible) -- nothing to compare against
    if "fleet exceeded" in heuristic_status:
        return None
    return round(100.0 * (distance - milp_distance) / milp_distance, 2)


def _print_row(variant: str, n: int, seed: int, milp, savings, refined=None) -> None:
    flags = []
    if milp.status != "Optimal":
        flags.append(f"MILP status={milp.status}")
    if "fleet exceeded" in savings.status:
        flags.append("heuristic exceeded fleet")
    flag_str = f"  [{', '.join(flags)}]" if flags else ""
    refined_str = f"  +2opt/Or-opt={refined.total_distance:.1f}" if refined is not None else ""
    print(f"  [{variant:6s}] n={n:2d} seed={seed}: MILP={milp.total_distance:.1f} ({milp.solve_time:.2f}s)  "
          f"Savings={savings.total_distance:.1f}{refined_str}{flag_str}")


def run_cvrp_benchmark(sizes, seeds=(1, 2, 3), vehicle_capacity=50.0, num_vehicles=4, time_limit=60) -> list[BenchmarkRow]:
    rows = []
    for n in sizes:
        for seed in seeds:
            inst = generate_cvrp_instance(num_customers=n, vehicle_capacity=vehicle_capacity, num_vehicles=num_vehicles, seed=seed)
            milp = solve_cvrp_milp(inst, time_limit=time_limit)
            savings = solve_cvrp_savings(inst)
            refined = solve_cvrp_savings_refined(inst)
            rows.append(BenchmarkRow(
                variant="CVRP", num_customers=n, seed=seed,
                milp_status=milp.status, milp_distance=milp.total_distance, milp_time=milp.solve_time,
                savings_distance=savings.total_distance, savings_time=savings.solve_time, savings_status=savings.status,
                refined_distance=refined.total_distance, refined_time=refined.solve_time,
                gap_savings_pct=_gap(milp.status, milp.total_distance, savings.status, savings.total_distance),
                gap_refined_pct=_gap(milp.status, milp.total_distance, refined.status, refined.total_distance),
            ))
            _print_row("CVRP", n, seed, milp, savings, refined)
    return rows


def run_vrptw_benchmark(sizes, seeds=(1, 2, 3), vehicle_capacity=50.0, num_vehicles=5, time_limit=60) -> list[BenchmarkRow]:
    rows = []
    for n in sizes:
        for seed in seeds:
            inst = generate_vrptw_instance(num_customers=n, vehicle_capacity=vehicle_capacity, num_vehicles=num_vehicles, seed=seed)
            milp = solve_vrptw_milp(inst, time_limit=time_limit)
            savings = solve_vrptw_savings(inst)
            refined = solve_vrptw_savings_refined(inst)
            rows.append(BenchmarkRow(
                variant="VRPTW", num_customers=n, seed=seed,
                milp_status=milp.status, milp_distance=milp.total_distance, milp_time=milp.solve_time,
                savings_distance=savings.total_distance, savings_time=savings.solve_time, savings_status=savings.status,
                refined_distance=refined.total_distance, refined_time=refined.solve_time,
                gap_savings_pct=_gap(milp.status, milp.total_distance, savings.status, savings.total_distance),
                gap_refined_pct=_gap(milp.status, milp.total_distance, refined.status, refined.total_distance),
            ))
            _print_row("VRPTW", n, seed, milp, savings, refined)
    return rows


def run_mdvrp_benchmark(sizes, seeds=(1, 2, 3), num_depots=2, vehicle_capacity=40.0, num_vehicles_per_depot=2, time_limit=90) -> list[BenchmarkRow]:
    rows = []
    for n in sizes:
        for seed in seeds:
            inst = generate_mdvrp_instance(num_customers=n, num_depots=num_depots, vehicle_capacity=vehicle_capacity,
                                            num_vehicles_per_depot=num_vehicles_per_depot, seed=seed)
            milp = solve_mdvrp_milp(inst, time_limit=time_limit)
            savings = solve_mdvrp_savings(inst)
            # No local-search refinement built for MDVRP (see clarke_wright.py's
            # solve_mdvrp_savings docstring) -- refined columns just mirror
            # savings, kept for a uniform CSV schema across all three variants.
            rows.append(BenchmarkRow(
                variant="MDVRP", num_customers=n, seed=seed,
                milp_status=milp.status, milp_distance=milp.total_distance, milp_time=milp.solve_time,
                savings_distance=savings.total_distance, savings_time=savings.solve_time, savings_status=savings.status,
                refined_distance=savings.total_distance, refined_time=savings.solve_time,
                gap_savings_pct=_gap(milp.status, milp.total_distance, savings.status, savings.total_distance),
                gap_refined_pct=_gap(milp.status, milp.total_distance, savings.status, savings.total_distance),
            ))
            _print_row("MDVRP", n, seed, milp, savings)
    return rows


def summarize(rows: list[BenchmarkRow]) -> dict:
    """Average key metrics across seeds, grouped by (variant, instance size)."""
    keys = sorted(set((r.variant, r.num_customers) for r in rows))
    summary = {}
    for variant, n in keys:
        subset = [r for r in rows if r.variant == variant and r.num_customers == n]
        savings_gaps = [r.gap_savings_pct for r in subset if r.gap_savings_pct is not None]
        refined_gaps = [r.gap_refined_pct for r in subset if r.gap_refined_pct is not None]
        milp_optimal = [r for r in subset if r.milp_status == "Optimal"]
        summary[(variant, n)] = {
            "milp_time_mean": statistics.mean(r.milp_time for r in subset),
            "savings_time_mean": statistics.mean(r.savings_time for r in subset),
            "refined_time_mean": statistics.mean(r.refined_time for r in subset),
            "savings_gap_mean": statistics.mean(savings_gaps) if savings_gaps else None,
            "refined_gap_mean": statistics.mean(refined_gaps) if refined_gaps else None,
            "n_instances": len(subset),
            "n_milp_optimal": len(milp_optimal),
            "n_comparable": len(savings_gaps),
        }
    return summary


def save_csv(rows: list[BenchmarkRow], path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(asdict(rows[0]).keys()))
        writer.writeheader()
        for r in rows:
            writer.writerow(asdict(r))


def main():
    all_rows = []

    print("Running CVRP benchmark...")
    all_rows += run_cvrp_benchmark(sizes=[6, 8, 10, 12], seeds=[1, 2, 3], time_limit=60)

    print("\nRunning VRPTW benchmark...")
    all_rows += run_vrptw_benchmark(sizes=[6, 8, 10], seeds=[1, 2, 3], time_limit=60)

    print("\nRunning MDVRP benchmark...")
    all_rows += run_mdvrp_benchmark(sizes=[6, 8, 10], seeds=[1, 2, 3], time_limit=90)

    out_path = Path("results") / "benchmark_results.csv"
    save_csv(all_rows, out_path)
    print(f"\nSaved {len(all_rows)} rows to {out_path}")

    print("\n--- Summary by variant and instance size (averaged over seeds) ---")
    summary = summarize(all_rows)
    for (variant, n), s in summary.items():
        gap_s = f"{s['savings_gap_mean']:.2f}%" if s["savings_gap_mean"] is not None else "N/A"
        gap_r = f"{s['refined_gap_mean']:.2f}%" if s["refined_gap_mean"] is not None else "N/A"
        print(f"[{variant:6s}] n={n:2d}  MILP time={s['milp_time_mean']:.2f}s "
              f"(optimal in {s['n_milp_optimal']}/{s['n_instances']})  "
              f"Savings gap={gap_s}  +2opt/Or-opt gap={gap_r}  "
              f"[{s['n_comparable']}/{s['n_instances']} comparable]")


if __name__ == "__main__":
    main()