import pytest

from vrp.instance import generate_mdvrp_instance, generate_cvrp_instance
from vrp.models.mdvrp import solve_mdvrp_milp


def _decode(route):
    """route[0] = -(depot_id+1); returns (depot_id, customer_ids)."""
    return -route[0] - 1, route[1:]


class TestMDVRPMilp:
    def test_small_instance_is_optimal_and_feasible(self):
        inst = generate_mdvrp_instance(num_customers=6, num_depots=2, vehicle_capacity=40,
                                        num_vehicles_per_depot=2, seed=3)
        sol = solve_mdvrp_milp(inst, time_limit=60)
        assert sol.status == "Optimal"

        served = []
        depot_route_count = {}
        for r in sol.routes:
            depot_id, customers = _decode(r)
            served.extend(customers)
            depot_route_count[depot_id] = depot_route_count.get(depot_id, 0) + 1
            load = sum(inst.demands[c] for c in customers)
            assert load <= inst.vehicle_capacity

        assert sorted(served) == list(range(inst.num_customers()))
        for count in depot_route_count.values():
            assert count <= inst.num_vehicles_per_depot

    def test_rejects_single_depot_instance(self):
        inst = generate_cvrp_instance(num_customers=4, seed=1)
        with pytest.raises(ValueError):
            solve_mdvrp_milp(inst)

class TestMDVRPSavings:
    def test_all_customers_served_and_capacity_respected(self):
        from vrp.heuristics.clarke_wright import solve_mdvrp_savings
        inst = generate_mdvrp_instance(num_customers=10, num_depots=2, vehicle_capacity=40,
                                        num_vehicles_per_depot=3, seed=4)
        sol = solve_mdvrp_savings(inst)
        served = []
        for r in sol.routes:
            _, customers = _decode(r)
            served.extend(customers)
            load = sum(inst.demands[c] for c in customers)
            assert load <= inst.vehicle_capacity
        assert sorted(served) == list(range(inst.num_customers()))

    def test_customers_assigned_to_nearest_depot_cluster(self):
        """Sanity check for the nearest-depot decomposition step: every
        customer should be at least as close to its assigned depot as to
        any other depot."""
        from vrp.heuristics.clarke_wright import solve_mdvrp_savings
        inst = generate_mdvrp_instance(num_customers=10, num_depots=2, vehicle_capacity=40,
                                        num_vehicles_per_depot=3, seed=4)
        sol = solve_mdvrp_savings(inst)
        for r in sol.routes:
            depot_id, customers = _decode(r)
            this_depot = inst.depots[depot_id]
            other_depots = [d for i, d in enumerate(inst.depots) if i != depot_id]
            for c in customers:
                dist_this = inst.distance(this_depot, inst.customers[c])
                dist_other_min = min(inst.distance(d, inst.customers[c]) for d in other_depots)
                assert dist_this <= dist_other_min + 1e-9

    def test_reports_fleet_exceeded_when_it_happens(self):
        """Regression test for the specific finding from our benchmark:
        this seed/config is known to produce a fleet-exceeded depot."""
        from vrp.heuristics.clarke_wright import solve_mdvrp_savings
        inst = generate_mdvrp_instance(num_customers=8, num_depots=2, vehicle_capacity=40,
                                        num_vehicles_per_depot=2, seed=2)
        sol = solve_mdvrp_savings(inst)
        assert "fleet exceeded" in sol.status