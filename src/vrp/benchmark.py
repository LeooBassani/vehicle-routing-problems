"""
Benchmark: MILP vs Clarke-Wright Savings vs Savings+2opt/Or-opt,
across a sweep of instance sizes and seeds.

Run as a script:
    PYTHONPATH=src python -m vrp.benchmark
"""

from __future__ import annotations

import csv
import statistics
from dataclasses import dataclass, asdict
from pathlib import Path

from vrp.instance import generate_cvrp_instance
from vrp.models.cvrp import solve_cvrp_milp
from vrp.heuristics.clarke_wright import solve_cvrp_savings
from vrp.heuristics.local_search import solve_cvrp_savings_refined


@dataclass
class BenchmarkRow:
    num_customers: int
    seed: int
    milp_status: str
    milp_distance: float
    milp_time: float
    savings_distance: float
    savings_time: float
    savings_status: str
    refined_distance: float
    refined_time: float
    gap_savings_pct: float | None    # None when not comparable (e.g. fleet exceeded)
    gap_refined_pct: float | None


def _gap(milp_distance: float, status: str, distance: float) -> float | None:
    if "fleet exceeded" in status:
        return None
    return round(100.0 * (distance - milp_distance) / milp_distance, 2)


def run_cvrp_benchmark(
    sizes: list[int],
    seeds: list[int] = (1, 2, 3),
    vehicle_capacity: float = 50.0,
    num_vehicles: int = 4,
    time_limit: int = 60,
) -> list[BenchmarkRow]:
    rows = []
    for n in sizes:
        for seed in seeds:
            inst = generate_cvrp_instance(
                num_customers=n, vehicle_capacity=vehicle_capacity, num_vehicles=num_vehicles, seed=seed,
            )
            milp = solve_cvrp_milp(inst, time_limit=time_limit)
            savings = solve_cvrp_savings(inst)
            refined = solve_cvrp_savings_refined(inst)

            rows.append(BenchmarkRow(
                num_customers=n, seed=seed,
                milp_status=milp.status, milp_distance=milp.total_distance, milp_time=milp.solve_time,
                savings_distance=savings.total_distance, savings_time=savings.solve_time, savings_status=savings.status,
                refined_distance=refined.total_distance, refined_time=refined.solve_time,
                gap_savings_pct=_gap(milp.total_distance, savings.status, savings.total_distance),
                gap_refined_pct=_gap(milp.total_distance, refined.status, refined.total_distance),
            ))
            flag = "" if "fleet exceeded" not in savings.status else "  [not comparable: heuristic exceeded fleet]"
            print(f"  n={n:2d} seed={seed}: MILP={milp.total_distance:.1f} ({milp.solve_time:.2f}s, {milp.status})  "
                  f"Savings={savings.total_distance:.1f}  +2opt/Or-opt={refined.total_distance:.1f}{flag}")
    return rows


def summarize_by_size(rows: list[BenchmarkRow]) -> dict:
    """Average key metrics across seeds, grouped by instance size."""
    sizes = sorted(set(r.num_customers for r in rows))
    summary = {}
    for n in sizes:
        subset = [r for r in rows if r.num_customers == n]
        savings_gaps = [r.gap_savings_pct for r in subset if r.gap_savings_pct is not None]
        refined_gaps = [r.gap_refined_pct for r in subset if r.gap_refined_pct is not None]
        summary[n] = {
            "milp_time_mean": statistics.mean(r.milp_time for r in subset),
            "savings_time_mean": statistics.mean(r.savings_time for r in subset),
            "refined_time_mean": statistics.mean(r.refined_time for r in subset),
            "savings_gap_mean": statistics.mean(savings_gaps) if savings_gaps else None,
            "refined_gap_mean": statistics.mean(refined_gaps) if refined_gaps else None,
            "n_instances": len(subset),
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
    print("Running CVRP benchmark (MILP vs Savings vs Savings+2opt/Or-opt)...")
    rows = run_cvrp_benchmark(sizes=[6, 8, 10, 12], seeds=[1, 2, 3], time_limit=60)

    out_path = Path("results") / "benchmark_results.csv"
    save_csv(rows, out_path)
    print(f"\nSaved {len(rows)} rows to {out_path}")

    print("\n--- Summary by instance size (averaged over seeds) ---")
    summary = summarize_by_size(rows)
    for n, s in summary.items():
        gap_s = f"{s['savings_gap_mean']:.2f}%" if s["savings_gap_mean"] is not None else "N/A"
        gap_r = f"{s['refined_gap_mean']:.2f}%" if s["refined_gap_mean"] is not None else "N/A"
        print(f"n={n:2d}  MILP time={s['milp_time_mean']:.2f}s  "
              f"Savings time={s['savings_time_mean']:.5f}s (gap {gap_s})  "
              f"+2opt/Or-opt time={s['refined_time_mean']:.5f}s (gap {gap_r})  "
              f"[{s['n_comparable']}/{s['n_instances']} comparable]")


if __name__ == "__main__":
    main()