import pytest

from vrp.instance import generate_cvrp_instance
from vrp.models.cvrp import solve_cvrp_milp


def _route_load(instance, route):
    return sum(instance.demands[c] for c in route)


def _all_customers_served_once(instance, routes):
    served = [c for r in routes for c in r]
    return sorted(served) == list(range(instance.num_customers()))


class TestCVRPMilp:
    def test_small_instance_is_optimal_and_feasible(self):
        inst = generate_cvrp_instance(num_customers=6, vehicle_capacity=50, num_vehicles=3, seed=1)
        sol = solve_cvrp_milp(inst, time_limit=30)

        # test "optimal"
        assert sol.status == "Optimal"
        # check if all customer were visited and if a customer was visited twice
        assert _all_customers_served_once(inst, sol.routes)
        # capacity check
        for r in sol.routes:
            assert _route_load(inst, r) <= inst.vehicle_capacity
        assert len(sol.routes) <= inst.num_vehicles_per_depot

    # test single customer edge: only one client
    def test_single_customer_fits_in_one_route(self):
        inst = generate_cvrp_instance(num_customers=1, vehicle_capacity=100, num_vehicles=1, seed=1)
        sol = solve_cvrp_milp(inst, time_limit=10)

        assert sol.status == "Optimal"
        assert sol.routes == [[0]]

    # test reject multiple depots
    def test_rejects_multi_depot_instance(self):
        from vrp.instance import generate_mdvrp_instance
        inst = generate_mdvrp_instance(num_customers=4, num_depots=2, seed=1)
        with pytest.raises(ValueError):
            solve_cvrp_milp(inst)