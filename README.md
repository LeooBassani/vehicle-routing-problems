# Vehicle Routing Problems: MILP vs. Constructive Heuristics

Mixed-integer programming formulations and a Clarke-Wright Savings
heuristic (with 2-opt/Or-opt refinement) for three classical Vehicle
Routing Problem variants (CVRP, VRPTW, and MDVRP) benchmarked against
each other on optimality gap and runtime scaling, plus a real-world case
study using real points of interest and real road-network distances
around Chelsea, MA / Boston.

## Why this repo

The Vehicle Routing Problem is one of the most common topics in
Operations Research studies and one of the most common real-world
supply-chain applications of combinatorial optimization. This repo is
built around a single question every OR practitioner has to answer in
practice: **when is it worth solving exactly as a MIP, and when should you reach
for a heuristic?**

## Variants covered

| Variant | Description | Files |
|---|---|---|
| **CVRP** | Capacitated VRP — single depot, vehicle capacity limits | `models/cvrp.py` |
| **VRPTW** | + Time windows per customer | `models/vrptw.py` |
| **MDVRP** | Multiple depots, joint depot assignment + routing | `models/mdvrp.py` |

Each variant is solved two ways:
1. **Exact MILP** (PuLP + CBC, open-source — no license required to run this repo)
2. **Clarke-Wright Savings heuristic**, extended to respect time windows (VRPTW) and to decompose by nearest depot (MDVRP), plus **2-opt/Or-opt local search refinement** on top (CVRP and VRPTW; see "Known limitations" for why MDVRP doesn't have this yet)

Full mathematical formulations (sets, variables, objective, constraints)
for all three models are in
[`docs/math_formulation.md`](docs/math_formulation.md).

## Repository structure

```text
src/vrp/
├── instance.py              # Instance dataclass + synthetic instance generators
├── real_world.py            # Real POIs (Chelsea, MA) from OpenStreetMap
├── road_network.py          # Real street network + shortest-path distances (OSMnx/NetworkX)
├── road_case_study.py       # Full pipeline: real POIs + real road distances -> MILP/heuristic
├── models/
│   ├── cvrp.py                # Exact MILP: two-index vehicle-flow + MTZ (remaining-capacity form)
│   ├── vrptw.py               # + time windows (big-M time propagation)
│   └── mdvrp.py               # + multi-depot assignment variables
├── heuristics/
│   ├── clarke_wright.py     # Savings heuristic: fleet-aware, time-window-aware, multi-depot decomposition
│   └── local_search.py      # 2-opt + Or-opt refinement (time-window-aware for VRPTW)
├── benchmark.py              # MILP vs heuristic: gap + runtime scaling, all 3 variants
├── visualization.py          # Route plotting (matplotlib), all 3 variants
└── map_viz.py                 # Interactive route maps on real streets (Folium)

docs/math_formulation.md      # Full MILP formulations for all 3 variants
tests/                         # pytest suite (all 3 variants, MILP + heuristic + refinement)
results/                       # Benchmark CSV
notebooks/                     # End-to-end demo notebooks
```

## Quickstart

```bash
pip install -r requirements.txt
python -m vrp.benchmark              # runs the full benchmark, saves results/benchmark_results.csv
pytest tests/ -v                     # full test suite across all 3 variants
```

```python
from vrp.instance import generate_cvrp_instance
from vrp.models.cvrp import solve_cvrp_milp
from vrp.heuristics.local_search import solve_cvrp_savings_refined

inst = generate_cvrp_instance(num_customers=10, vehicle_capacity=50, num_vehicles=4, seed=1)

milp_solution = solve_cvrp_milp(inst, time_limit=60)          # provably optimal
heuristic_solution = solve_cvrp_savings_refined(inst)          # near-instant

print(milp_solution.total_distance, heuristic_solution.total_distance)
```

## Results

Full data in [`results/benchmark_results.csv`](results/benchmark_results.csv). Summary (averaged over 3 seeds per size):

| Variant | n | MILP time | MILP optimal | Savings gap | +2opt/Or-opt gap |
|---|---|---|---|---|---|
| CVRP | 6 | 0.31s | 3/3 | 0.92% | 0.92% |
| CVRP | 8 | 1.17s | 3/3 | 1.63% | 1.63% |
| CVRP | 10 | 5.16s | 3/3 | 2.26% | 2.13% |
| CVRP | 12 | 32.27s | 3/3 | 2.28% | 2.28% |
| VRPTW | 6 | 0.21s | 3/3 | 2.09% | 2.09% |
| VRPTW | 8 | 0.05s | 3/3 | 3.73% | 3.73% |
| VRPTW | 10 | 0.10s | 3/3 | 1.38% | 1.38% |
| MDVRP | 6 | 0.35s | 3/3 | 5.95%* | 5.95%* |
| MDVRP | 8 | 1.67s | 3/3 | 1.44%* | 1.44%* |
| MDVRP | 10 | 13.90s | 3/3 | 5.87%* | 5.87%* |

*MDVRP gaps are averaged only over the instances where the heuristic's
route count stayed within the fleet limit — see "Known limitations" below
for why that's frequently not the case, and why that's the actual finding
worth reporting.

### Insights

- **MILP solve time grows sharply with CVRP instance size** (0.31s → 32.27s from n=6 to n=12) — the expected combinatorial behavior of an exact vehicle-flow MILP, and the reason the heuristic exists at all.
- **VRPTW's MILP solves faster than CVRP's at equivalent sizes** (often under 0.1s), which is initially counter-intuitive. VRPTW looks like "CVRP plus more constraints." The explanation: time windows prune the feasible arc space before the solver even starts branching. An arc `i → j` that would arrive outside `j`'s window is infeasible regardless of what the rest of the route looks like, which shrinks the effective search space substantially compared to CVRP's capacity-only pruning.
- **The heuristic is usually within 1-6% of optimal**, and the 2-opt/Or-opt refinement provides a real but modest improvement on top (a few tenths of a percent typically). 
- **VRPTW can become genuinely infeasible, not just hard, as the fleet tightens.** During benchmark calibration, VRPTW instances at n=10 with a 4-vehicle fleet were *infeasible* in 2 of 3 seeds — not merely slow to solve. The benchmark uses a 5-vehicle fleet to keep comparisons meaningful across sizes, but the original finding is worth keeping in mind: time windows and capacity interact in a way that can eliminate feasibility entirely, not just optimality.

## Known limitations

- **MDVRP's nearest-depot decomposition heuristic frequently exceeds its per-depot fleet limit** — in benchmark instances, 5 to 8 of 9 tested instances per run produced at least one depot whose nearest-assigned customers didn't fit within that depot's own vehicle count, even though the *overall* problem was feasible (the MILP always found a feasible solution). This happens because nearest-depot assignment is fixed *before* routing starts, so it can't rebalance customers the way the MILP's joint depot-assignment-and-routing optimization can. Rather than silently return an invalid answer, `solve_mdvrp_savings` detects and reports this explicitly via `RoutingSolution.status`. This is arguably the single most useful empirical result in this repo: a concrete, measured illustration of why joint optimization beats problem decomposition when the decomposition boundary doesn't match the true optimal structure.
- **No 2-opt/Or-opt refinement for MDVRP yet** — a time-window-style extension (checking depot-assignment consistency and per-depot capacity on every candidate move) would be a natural next step, but given the finding above, the higher-value fix is almost certainly to improve the *assignment* step (e.g. capacity-aware or load-balanced assignment instead of pure nearest-depot), not to polish routes within a broken assignment.

## Real-world case study: Chelsea, MA / Boston

Every variant can be built from **real points of interest** (supermarkets,
pharmacies, convenience stores, retail shops) retrieved from OpenStreetMap
around a real depot location (Chelsea Industrial Area, MA), instead of
synthetic random coordinates — see `real_world.py`.

Going one step further, `road_network.py` + `road_case_study.py` replace
straight-line distance between those real POIs with **actual driving
distance along the real street network** (shortest path via OSMnx +
NetworkX), and `map_viz.py` renders the resulting routes on an interactive
Folium map that follows real streets — not straight lines through
buildings and water.

| | Straight-line distance | Real road-network distance |
|---|---|---|
| CVRP (8 real stores) | 13.59 km | 17.25 km |
| VRPTW (same stores + real-travel-time windows) | — | 17.83 km |
| MDVRP (2 real depots, Chelsea + Boston) | — | 25.79 km |

The gap between straight-line and real-road distance (13.59 km → 17.25 km
for the identical customer set) is itself a useful data point: Euclidean
distance systematically *underestimates* real driving distance, which
matters if a routing tool is being used to estimate delivery cost or time
rather than just to rank candidate routes against each other.

## Design choices

- **PuLP + CBC, not Gurobi.** CBC is open-source, so no commercial license required 
to reproduce any result here.
- **Synthetic instances, not classical benchmark sets** (e.g. Solomon,
  Augerat) for the core benchmark. This trades comparability with
  published literature for full control over instance size, depot count,
  and time-window tightness. Complemented by the real-world Chelsea case
  study for external validity.
- **Remaining-capacity MTZ convention**, not the textbook cumulative-load
  form (chosen deliberately (see `docs/math_formulation.md`)) because it
  matches the physical study more directly (a vehicle leaves the depot
  full and empties out as it delivers), even though both are
  mathematically equivalent.
- **The heuristic is fleet-size aware and reports infeasibility honestly.**
  A Clarke-Wright implementation can silently return more routes
  than vehicles available, or (in MDVRP) silently accept a bad
  depot-assignment split. This implementation runs a forced-merge second
  pass to respect fleet limits wherever geometrically possible, and
  explicitly reports when it isn't (`RoutingSolution.status`) instead of
  returning a falsely "clean" answer.

## Scaling to commercial size

Every exact formulation in this repo is intentionally scoped to validate
and benchmark the heuristics on small instances (roughly ≤15 customers on
CBC), not to serve as a production solver (solve time grows sharply past
that point, which is expected MILP behavior, not a bug). For a real
city-scale deployment, the natural next steps, roughly in order of
effort:

1. **A tighter exact formulation** (DFJ/branch-and-cut instead of MTZ) — MTZ's big-M constraints have a weak LP relaxation; branch-and-cut with lazy subtour-elimination constraints prunes the search tree far more effectively, typically pushing exact solving into the 50-100+ customer range. This requires a solver with native lazy-constraint callback support (Gurobi/CPLEX) — PuLP/CBC doesn't expose this cleanly.
2. **Gap-tolerance solving** — stopping the MILP as soon as it's provably within, say, 2% of optimal (`gapRel` in most commercial solvers) instead of insisting on 0% gap, which is what virtually every production solver defaults to.
3. **Local search beyond 2-opt/Or-opt** — Large Neighborhood Search (LNS) or Adaptive LNS: destroy part of the solution and intelligently reconstruct it, repeated thousands of times. This is the standard approach for large-scale VRP in industry today.
4. **Column generation / branch-and-price** — the academic state of the art for exact large-scale VRP: reformulate as a set-partitioning problem over routes, generating promising routes on demand instead of enumerating all arcs.
5. **Purpose-built libraries** — Google OR-Tools' routing solver implements LNS and local search out of the box and scales to thousands of customers in seconds to minutes; most real-world routing products (in-house or commercial) are built on something in this family rather than a hand-rolled MILP.

## Testing

```bash
pytest tests/ -v
```

The suite covers all three variants: MILP feasibility and optimality,
heuristic feasibility (capacity, time windows, fleet size, depot
assignment), 2-opt/Or-opt correctness (including a deterministic
known-crossing test) and non-regression (local search never worsens a
feasible solution), and the specific MDVRP fleet-exceeded finding above
(pinned as a regression test, not just a benchmark observation).

## How this was built

This repository was built collaboratively with Claude, used
as a pair-programming and tutoring tool throughout. Every mathematical 
formulation was derived and explained in plain language
before any code was written; every implementation was tested (often
against hand-verified small examples) before being accepted; and several
of the findings called out above (the MDVRP fleet-exceeded rate, VRPTW's
faster MILP solve time, VRPTW's infeasibility under a tight fleet) emerged
from deliberately running checks rather than being assumed. I
AI assistance was a tool in that process, the way a calculator, a textbook,
or a colleague's code review would be, not a replacement for
understanding the underlying optimization theory.
