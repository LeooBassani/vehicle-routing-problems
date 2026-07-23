import pytest

from vrp.instance import generate_vrptw_instance, generate_cvrp_instance
from vrp.models.vrptw import solve_vrptw_milp


def _simulate_route(instance, route):
    """Replay a route and check every time window + the return-to-depot deadline."""
    depot = instance.depots[0]
    t = 0.0
    prev = depot
    for c in route:
        travel = instance.distance(prev, instance.customers[c]) / instance.speed
        arrival = t + travel
        earliest, latest = instance.time_windows[c]
        if arrival > latest:
            return False
        t = max(arrival, earliest) + instance.service_times[c]
        prev = instance.customers[c]
    return t + instance.distance(prev, depot) / instance.speed <= instance.depot_time_window[1]


class TestVRPTWMilp:
    def test_small_instance_respects_time_windows(self):
        inst = generate_vrptw_instance(num_customers=6, vehicle_capacity=50, num_vehicles=4, seed=1)
        sol = solve_vrptw_milp(inst, time_limit=30)

        assert sol.status == "Optimal"
        for r in sol.routes:
            assert _simulate_route(inst, r)

    def test_respects_capacity(self):
        inst = generate_vrptw_instance(num_customers=6, vehicle_capacity=50, num_vehicles=4, seed=1)
        sol = solve_vrptw_milp(inst, time_limit=30)
        for r in sol.routes:
            assert sum(inst.demands[c] for c in r) <= inst.vehicle_capacity

    def test_all_customers_served_exactly_once(self):
        inst = generate_vrptw_instance(num_customers=6, vehicle_capacity=50, num_vehicles=4, seed=1)
        sol = solve_vrptw_milp(inst, time_limit=30)
        served = sorted(c for r in sol.routes for c in r)
        assert served == list(range(inst.num_customers()))

    def test_rejects_instance_without_time_windows(self):
        inst = generate_cvrp_instance(num_customers=4, seed=1)
        with pytest.raises(ValueError):
            solve_vrptw_milp(inst)

class TestVRPTWSavings:
    def test_produced_routes_respect_time_windows(self):
        from vrp.heuristics.clarke_wright import solve_vrptw_savings
        inst = generate_vrptw_instance(num_customers=8, vehicle_capacity=50, num_vehicles=4, seed=2)
        sol = solve_vrptw_savings(inst)
        for r in sol.routes:
            assert _simulate_route(inst, r), f"Route {r} violates a time window"

    def test_all_customers_served_exactly_once(self):
        from vrp.heuristics.clarke_wright import solve_vrptw_savings
        inst = generate_vrptw_instance(num_customers=8, vehicle_capacity=50, num_vehicles=4, seed=2)
        sol = solve_vrptw_savings(inst)
        served = sorted(c for r in sol.routes for c in r)
        assert served == list(range(inst.num_customers()))

    def test_respects_capacity(self):
        from vrp.heuristics.clarke_wright import solve_vrptw_savings
        inst = generate_vrptw_instance(num_customers=8, vehicle_capacity=50, num_vehicles=4, seed=2)
        sol = solve_vrptw_savings(inst)
        for r in sol.routes:
            assert sum(inst.demands[c] for c in r) <= inst.vehicle_capacity