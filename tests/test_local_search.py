import pytest

from vrp.instance import generate_cvrp_instance
from vrp.heuristics.clarke_wright import solve_cvrp_savings
from vrp.heuristics.local_search import two_opt, or_opt, local_search, solve_cvrp_savings_refined


def _euclidean(p, q):
    return ((p[0] - q[0]) ** 2 + (p[1] - q[1]) ** 2) ** 0.5


def _route_length(depot, coords, route, dist_fn):
    points = [depot] + [coords[c] for c in route] + [depot]
    return sum(dist_fn(points[k], points[k + 1]) for k in range(len(points) - 1))


class TestTwoOpt:
    def test_fixes_a_known_crossing(self):
        """A deliberately bad route (visits square corners diagonally,
        creating an X) must be fixed to the perimeter order by 2-opt."""
        depot = (0.0, 0.0)
        coords = [(10, 10), (10, 20), (20, 20), (20, 10)]
        bad_route = [0, 2, 1, 3]  # diagonal order -> crosses itself

        fixed_route = two_opt(bad_route, depot, coords, _euclidean)

        bad_length = _route_length(depot, coords, bad_route, _euclidean)
        fixed_length = _route_length(depot, coords, fixed_route, _euclidean)

        assert fixed_length < bad_length
        assert fixed_length == pytest.approx(66.50, abs=0.01)

    def test_never_increases_distance(self):
        """2-opt is a local search: it must never return a longer route
        than it started with, on any input."""
        inst = generate_cvrp_instance(num_customers=10, seed=3)
        depot = inst.depots[0]
        route = list(range(10))  # arbitrary starting order

        before = _route_length(depot, inst.customers, route, inst.distance)
        after_route = two_opt(route, depot, inst.customers, inst.distance)
        after = _route_length(depot, inst.customers, after_route, inst.distance)

        assert after <= before + 1e-9


class TestOrOpt:
    def test_respects_capacity_across_routes(self):
        inst = generate_cvrp_instance(num_customers=12, vehicle_capacity=50, num_vehicles=4, seed=5)
        base = solve_cvrp_savings(inst)

        refined_routes = or_opt(
            base.routes, inst.depots[0], inst.customers, inst.demands, inst.vehicle_capacity, inst.distance,
        )
        for r in refined_routes:
            load = sum(inst.demands[c] for c in r)
            assert load <= inst.vehicle_capacity


class TestLocalSearchIntegration:
    def test_refinement_never_worsens_savings_solution(self):
        for seed in range(1, 6):
            inst = generate_cvrp_instance(num_customers=12, vehicle_capacity=50, num_vehicles=4, seed=seed)
            base = solve_cvrp_savings(inst)
            refined = solve_cvrp_savings_refined(inst)
            assert refined.total_distance <= base.total_distance + 1e-6

    def test_refinement_still_feasible(self):
        inst = generate_cvrp_instance(num_customers=15, vehicle_capacity=50, num_vehicles=5, seed=9)
        refined = solve_cvrp_savings_refined(inst)

        served = sorted(c for r in refined.routes for c in r)
        assert served == list(range(inst.num_customers()))
        for r in refined.routes:
            assert sum(inst.demands[c] for c in r) <= inst.vehicle_capacity
        assert len(refined.routes) <= inst.num_vehicles_per_depot