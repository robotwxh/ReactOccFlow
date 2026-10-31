import numpy as np
import pytest

from reactoccflow.evaluation.reaction import pair_response_samples, response_metrics
from reactoccflow.evaluation.risk import (
    calibration_metrics, choose_threshold, evaluate_risk, fit_calibrator,
    rectangle_overlap_area, smoothed_event_frequency,
)


def grid():
    return {"origin_xy": [0.0, 0.0], "resolution": 1.0, "shape": [10, 10], "frame": "E0"}


def test_response_pairs_same_origin_seed_history_and_rejects_mismatch():
    a = np.array([[0., 0.], [1., 0.]])
    b = np.array([[0., 0.], [1., 1.]])
    assert pair_response_samples(a, b, reference_origin="o", counterfactual_origin="o",
                                reference_seed=2, counterfactual_seed=2,
                                reference_history=[1, 2], counterfactual_history=[1, 2]) == pytest.approx(.5)
    with pytest.raises(ValueError):
        pair_response_samples(a, b, reference_origin="o", counterfactual_origin="other",
                              reference_seed=2, counterfactual_seed=2,
                              reference_history=[1], counterfactual_history=[1])


def test_response_rejects_future_observation_labels_and_preserves_status():
    pred = np.zeros((2, 2, 2))
    with pytest.raises(ValueError):
        response_metrics(pred, [0.5, 1.0], origins=["o", "o"], actions=["a", "b"],
                         seeds=[1, 1], observed_times_s=[0., .5], future_label_times_s=[.75])
    report = response_metrics(pred, [0.5, 1.0], origins=["o", "o"], actions=["a", "b"],
                              seeds=[1, 1], statuses=["timeout", "ok"])
    assert report["status_counts"] == {"timeout": 1, "ok": 1}


def test_smoothing_and_measured_time_risk_ttc_area_overlap():
    assert smoothed_event_frequency([True, False, False]) == pytest.approx(1.5 / 4)
    from reactoccflow.evaluation.reaction import time_to_collision
    assert time_to_collision([[[2., 0.], [0., 0.]]], [[[0., 0.], [0., 0.]]], [0.5, 1.5], collision_distance=.1)[0] == pytest.approx(1.5)
    assert rectangle_overlap_area(np.array([[1,1],[1,-1],[-1,-1],[-1,1.]]),
                                 np.array([[2,1],[2,-1],[0,-1],[0,1.]])) == pytest.approx(2.)


def test_risk_uses_actor_area_and_out_of_bounds():
    worlds = [{"actors": [{"position": [[2., 2.], [2., 2.]], "lwh": [1., 1., 1.], "yaw": 0.}]}]
    report = evaluate_risk(np.array([[2., 2., 0.], [2., 2., 0.]]), worlds,
                           {"length": 2., "width": 2.}, times_s=[.5, 1.5], grid_metadata=grid())
    assert report["per_world"][0]["collision"]
    out = evaluate_risk(np.array([[-.1, 2., 0.], [-.1, 2., 0.]]), [{}],
                        {"length": 2., "width": 2.}, times_s=[.5, 1.5], grid_metadata=grid())
    assert out["per_world"][0]["status"] == "out_of_bounds"


def test_calibration_group_isolation_metrics_and_rare_support():
    with pytest.raises(ValueError):
        fit_calibrator([.1, .2], [0, 1], ["a", "b"], train_origins=["a"])
    cal = fit_calibrator([.1, .2, .8, .9], [0, 0, 1, 1], ["c", "d", "e", "f"],
                         method="isotonic", train_origins=["a"], eval_origins=["b"], K=3, M=4, NFE=8)
    p = cal.predict([.1, .9]); assert np.all((p >= 0) & (p <= 1) & np.isfinite(p))
    metrics = calibration_metrics(p, [0, 1], bins=4)
    assert metrics["brier"] == pytest.approx(0.)
    assert sum(row["count"] for row in metrics["reliability"]) == 2
    assert metrics["rare_event_support"] == 1


def test_threshold_fallback_marks_no_feasible_and_brakes():
    result = choose_threshold(np.array([[.3, .4], [.25, .35]]),
                              np.array([[1, 1], [1, 1]], bool),
                              np.array([[0, 0], [0, 0]], bool), brake_index=1)
    assert result["no_feasible_candidate"]
    assert result["selected_indices"] == [1, 1]


def test_occupancy_positive_footprint_cell_intersection_and_masks():
    meta = {"origin_xy": [0.0, 0.0], "resolution": 1.0, "shape": [1, 1], "frame": "E0"}
    # Cell center (0.5, 0.5) is outside this footprint, but the footprint overlaps it.
    candidate = np.array([[0.9, 0.5, 0.0, 0.0], [0.9, 0.5, 0.0, 0.0]])
    world = {
        "occupancy": np.ones((2, 1, 1), dtype=bool),
        "occupancy_validity_mask": np.ones((2, 1, 1), dtype=bool),
        "actors": [{"position": [[0.9, 0.5, 0.5], [0.9, 0.5, 0.5]], "lwh": [0.2, 0.2, 1.0], "yaw": 0.0}],
        "actor_validity_mask": [False],
        "status": "ok",
    }
    report = evaluate_risk(candidate, [world], {"length": 0.4, "width": 0.4, "height": 1.0, "z": 0.5, "frame": "E0"},
                           times_s=[0.5, 1.5], grid_metadata=meta)
    assert report["per_world"][0]["collision"]
    masked = dict(world, occupancy_validity_mask=np.zeros((2, 1, 1), dtype=bool))
    masked_report = evaluate_risk(candidate, [masked], {"length": 0.4, "width": 0.4, "height": 1.0, "z": 0.5, "frame": "E0"},
                                  times_s=[0.5, 1.5], grid_metadata=meta)
    assert not masked_report["per_world"][0]["collision"]


def test_risk_preserves_all_supplied_statuses_and_requires_e0_alignment():
    candidate = np.array([[2., 2., 0.], [2., 2., 0.]])
    geometry = {"length": 2., "width": 2., "frame": "E0"}
    for source_status in ("invalid", "timeout", "crash", "out_of_bounds"):
        report = evaluate_risk(candidate, [{"status": source_status}], geometry,
                               times_s=[.5, 1.5], grid_metadata={**grid(), "frame": "E0"})
        assert report["per_world"][0]["status"] == source_status
    with pytest.raises(ValueError):
        evaluate_risk(candidate, [{}], geometry, times_s=[.5, 1.5], grid_metadata={"origin_xy": [0., 0.], "resolution": 1., "shape": [10, 10]})


def test_future_label_rejection_cannot_be_bypassed_by_current_time():
    from reactoccflow.evaluation.reaction import validate_observation_input
    with pytest.raises(ValueError):
        validate_observation_input([0., 1.], future_label_times_s=[1.5], current_time_s=2.0)


def test_threshold_requires_brake_and_only_falls_back_infeasible_groups(tmp_path):
    with pytest.raises(ValueError):
        choose_threshold(np.array([[.3, .4]]), np.array([[1, 1]], bool), np.array([[0, 0]], bool))
    result = choose_threshold(np.array([[.01, .4], [.3, .01]]),
                              np.array([[0, 1], [1, 0]], bool), np.array([[1, 1], [1, 1]], bool), brake_index=1)
    assert all(i >= 0 for i in result["selected_indices"])
    assert result["no_feasible_candidate"] is False or result["selected_indices"][0] == 1


def test_calibration_artifacts_are_frozen_and_split_fingerprints_are_sha256(tmp_path):
    cal = fit_calibrator([.1, .2, .8, .9], [0, 0, 1, 1], ["c", "d", "e", "f"],
                         method="isotonic", train_origins=["a"], eval_origins=["b"], K=3, M=4, NFE=8)
    for fp in cal.artifact["split_fingerprints"].values():
        assert isinstance(fp, str) and len(fp) == 64 and all(c in "0123456789abcdef" for c in fp)
    path = tmp_path / "calibration.json"
    cal.save(path)
    from reactoccflow.evaluation.risk import load_calibrator
    loaded = load_calibrator(path, expected_feature_definition="risk_frequency")
    assert loaded.artifact["frozen"] is True
    assert np.allclose(loaded.predict([.1, .9]), cal.predict([.1, .9]))
