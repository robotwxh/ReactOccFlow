import numpy as np
import pytest

from reactoccflow.planning import (
    CandidateGenerator,
    CandidatePlans,
    PlanFlow,
    Planner,
    PlanningConfig,
    check_kinematics,
    compute_costs,
)


def observation(speed=4.0):
    return {
        "ego_state": {"speed": speed, "acceleration": 0.0, "heading_rate": 0.0},
        "current_pose": {"x": 0.0, "y": 0.0, "yaw": 0.35},
        "route_waypoints": [[8.0, 0.0], [18.0, 1.0]],
    }


def test_candidate_count_ids_and_brake_are_deterministic():
    cfg = PlanningConfig(num_candidates=8, num_points=6)
    first = CandidateGenerator(cfg).generate(observation())
    second = CandidateGenerator(cfg).generate(observation())
    assert first.xy.shape == (9, 6, 2)
    assert len(first.ids) == 9
    assert len(set(first.ids)) == 9
    assert first.is_brake[-1]
    assert np.array_equal(first.ids, second.ids)
    np.testing.assert_allclose(first.xy, second.xy)
    assert first.latency_candidate_count == 9


def test_brake_reaches_rest_and_preserves_heading():
    plans = CandidateGenerator(PlanningConfig(num_candidates=2, max_deceleration=2.0)).generate(observation(3.0))
    brake = plans.brake
    assert brake.is_brake
    assert np.allclose(brake.heading, 0.35)
    assert np.all(np.diff(brake.xy[:, 0]) >= -1e-8)
    assert plans.speeds[-1, -1] == pytest.approx(0.0, abs=1e-8)


def test_observed_only_rejects_future_target_and_gt_keys():
    generator = CandidateGenerator()
    for bad in ("future_trajectory", "target_xy", "GT_boxes"):
        payload = observation()
        payload[bad] = 1
        with pytest.raises(ValueError, match="observed-only"):
            generator.generate(payload)


def test_kinematics_use_elapsed_timestamps_and_require_them():
    traj = np.array([[1.0, 0.0], [3.0, 0.0], [6.0, 0.0]])
    checked = check_kinematics(traj, np.array([0.5, 1.5, 3.0]), max_speed=3.0, max_acceleration=2.0)
    assert checked.max_speed == pytest.approx(2.0)
    assert checked.valid
    with pytest.raises(ValueError, match="timestamps"):
        check_kinematics(traj, None)


def test_invalid_candidate_is_retained_and_selector_falls_back_to_brake():
    plans = CandidateGenerator(PlanningConfig(num_candidates=1, max_speed=1.0)).generate(observation(10.0))
    assert len(plans.invalid_reasons[0]) > 0
    result = Planner(risk_threshold=0.2).select(plans, risks=np.array([0.01, 0.3]))
    assert result.candidate.is_brake
    assert result.no_feasible_candidate


def test_selector_tie_breaks_risk_then_cost_then_stable_id():
    plans = CandidateGenerator(PlanningConfig(num_candidates=2)).generate(observation())
    costs = np.array([2.0, 1.0, 9.0])
    result = Planner().select(plans, risks=np.array([0.1, 0.1, 0.9]), costs=costs)
    assert result.index == 1
    result = Planner().select(plans, risks=np.array([0.1, 0.1, 0.1]), costs=costs)
    assert result.index == 1


def test_costs_include_world_risk_and_comparison_modes():
    plans = CandidateGenerator().generate(observation())
    costs = compute_costs(plans, observation(), world_risk=0.25, mode="occupancy-only")
    assert costs.total.shape == (9,)
    assert np.all(costs.world_risk == pytest.approx(0.25))
    assert Planner(comparison_mode="fixed-candidates").comparison_mode == "fixed-candidates"
    assert Planner(comparison_mode="no-world").comparison_mode == "no-world"


def test_local_rng_does_not_mutate_global_state_and_k_settings_work():
    np.random.seed(123)
    before = np.random.random()
    plans_a = CandidateGenerator(PlanningConfig(num_candidates=3), seed=7).generate(observation())
    after = np.random.random()
    np.random.seed(123)
    assert before == np.random.random()
    assert after == np.random.random()
    plans_b = CandidateGenerator(PlanningConfig(num_candidates=3), seed=7).generate(observation())
    np.testing.assert_allclose(plans_a.xy, plans_b.xy)
    assert plans_a.xy.shape[0] == 4


def test_untrained_learned_head_fails_clearly():
    with pytest.raises(RuntimeError, match="untrained"):
        PlanFlow().sample(observation(), num_candidates=8, use_learned=True)


def test_observed_only_rejects_unknown_top_level_and_nested_keys():
    generator = CandidateGenerator()
    payload = observation()
    payload["other"] = 123
    with pytest.raises(ValueError, match="unknown observation key"):
        generator.generate(payload)

    payload = observation()
    payload["ego_state"] = {"speed": 4.0, "metadata": {"future_value": 1.0}}
    with pytest.raises(ValueError, match="unknown observation key"):
        generator.generate(payload)


def test_brake_is_invalid_when_horizon_cannot_reach_rest():
    cfg = PlanningConfig(num_candidates=1, max_deceleration=1.0)
    plans = CandidateGenerator(cfg).generate(observation(100.0))
    brake = plans.brake
    assert not brake.valid
    assert "stop_not_reached" in brake.invalid_reasons
    assert brake.speeds[-1] > 0.0


def _candidate_plans_for_validation(**overrides):
    values = {
        "xy": np.zeros((2, 2, 2), dtype=np.float64),
        "timestamps": np.array([0.5, 1.0], dtype=np.float64),
        "ids": ("a", "b"),
        "is_brake": np.array([False, True]),
        "valid": np.array([True, True]),
        "invalid_reasons": ((), ()),
        "heading": np.zeros((2, 2), dtype=np.float64),
        "speeds": np.zeros((2, 2), dtype=np.float64),
        "latency_candidate_count": 2,
    }
    values.update(overrides)
    return CandidatePlans(**values)


@pytest.mark.parametrize("field", ["timestamps", "heading", "speeds"])
def test_candidate_plans_reject_non_finite_numeric_arrays(field):
    values = {
        "timestamps": np.array([0.5, 1.0], dtype=np.float64),
        "heading": np.zeros((2, 2), dtype=np.float64),
        "speeds": np.zeros((2, 2), dtype=np.float64),
    }
    values[field].flat[0] = np.nan
    with pytest.raises(ValueError, match="finite"):
        _candidate_plans_for_validation(**values)


def test_candidate_plans_reject_non_increasing_timestamps():
    with pytest.raises(ValueError, match="timestamps"):
        _candidate_plans_for_validation(timestamps=np.array([1.0, 1.0]))


def test_candidate_plans_reject_malformed_trajectory_shape():
    with pytest.raises(ValueError, match="candidate xy"):
        _candidate_plans_for_validation(xy=np.zeros((2, 2)))
    with pytest.raises(ValueError, match="candidate xy"):
        _candidate_plans_for_validation(
            xy=np.zeros((2, 0, 2)),
            timestamps=np.array([], dtype=np.float64),
            heading=np.zeros((2, 0), dtype=np.float64),
            speeds=np.zeros((2, 0), dtype=np.float64),
        )
