import json

import numpy as np
import pytest

from reactoccflow.simulation.bridge import (
    BridgeError,
    BridgeMessage,
    BridgeReceiver,
    ControlResponse,
    FreshnessError,
    FutureGroundTruthError,
    InMemoryTransport,
    MessageTooLargeError,
)
from reactoccflow.simulation.controller import Controller, ControllerConfig, VehicleControl, interpolate_xy


def message(**overrides):
    values = {
        "episode_id": "episode-a",
        "sequence": 0,
        "timestamp": 0.0,
        "sensor_bundle": {"lidar": {"frame": "E0", "count": 2}, "arrays": {"points": np.zeros((2, 3), dtype=np.float32)}},
        "ego_state": {"speed": 2.0, "x": 0.0, "y": 0.0, "yaw": 0.0},
        "navigation_condition": {"route_id": "r0", "waypoints": np.array([[0.0, 0.0], [3.0, 0.0]])},
        "candidate_outputs": {"selected_id": "candidate-0"},
        "world_outputs": {"model": "world-v0"},
    }
    values.update(overrides)
    return BridgeMessage(**values)


def test_bridge_message_round_trip_is_json_numpy_safe_and_future_keys_fail():
    encoded = message().to_json_bytes()
    decoded = BridgeMessage.from_json_bytes(encoded)
    assert decoded.episode_id == "episode-a"
    assert decoded.sequence == 0
    assert decoded.sensor_bundle["arrays"]["points"] == [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]]
    with pytest.raises(FutureGroundTruthError):
        BridgeMessage.from_mapping({**message().to_mapping(), "future_trajectory": [[1.0, 2.0]]})
    with pytest.raises(FutureGroundTruthError):
        BridgeMessage.from_mapping({**message().to_mapping(), "sensor_bundle": {"ground_truth": 1}})


def test_bridge_freshness_rejects_old_sequence_timestamp_and_stale_episode_then_resets():
    receiver = BridgeReceiver()
    receiver.receive(message(sequence=2, timestamp=2.0))
    with pytest.raises(FreshnessError, match="sequence"):
        receiver.receive(message(sequence=1, timestamp=3.0))
    with pytest.raises(FreshnessError, match="timestamp"):
        receiver.receive(message(sequence=3, timestamp=1.5))
    with pytest.raises(FreshnessError, match="episode"):
        receiver.receive(message(episode_id="episode-old", sequence=0, timestamp=0.0))
    receiver.reset_episode("episode-b")
    assert receiver.receive(message(episode_id="episode-b", sequence=0, timestamp=0.0)).episode_id == "episode-b"


def test_bridge_rejects_nonfinite_oversized_and_executable_payloads():
    with pytest.raises(ValueError, match="finite"):
        BridgeMessage(**{**message().__dict__, "timestamp": float("nan")})
    with pytest.raises(ValueError, match="executable|JSON"):
        BridgeMessage.from_mapping({**message().to_mapping(), "sensor_bundle": {"callback": lambda: None}})
    with pytest.raises(MessageTooLargeError):
        BridgeMessage.from_json_bytes(message(sensor_bundle={"x": "a" * 100}).to_json_bytes(), max_bytes=32)


def test_response_carries_fingerprints_and_deadline():
    response = ControlResponse(
        episode_id="episode-a", sequence=1, timestamp=1.0,
        requested_deadline=1.05, model_fingerprint="model-sha", config_fingerprint="config-sha",
        control={"steer": 0.0, "throttle": 0.2, "brake": 0.0},
    )
    decoded = ControlResponse.from_json_bytes(response.to_json_bytes())
    assert decoded.model_fingerprint == "model-sha"
    assert decoded.config_fingerprint == "config-sha"
    assert decoded.requested_deadline == pytest.approx(1.05)


def test_interpolation_uses_measured_plan_timestamps():
    xy = np.array([[0.0, 0.0], [2.0, 1.0], [6.0, 1.0]])
    times = np.array([0.0, 0.4, 1.2])
    np.testing.assert_allclose(interpolate_xy(xy, times, 0.8), [4.0, 1.0])


def test_controller_bounds_and_fallback_reasons_are_explicit():
    controller = Controller(ControllerConfig(fallback_brake=0.6))
    plan = {"xy": np.array([[0.0, 0.0], [2.0, 0.5], [4.0, 0.5]]), "timestamps": np.array([0.0, 0.5, 1.0]), "valid": True}
    result = controller.step(plan, {"speed": 1.0}, dt=0.5, episode_id="episode-a", sequence=0, timestamp=0.0)
    assert -1.0 <= result.control.steer <= 1.0
    assert 0.0 <= result.control.throttle <= 1.0
    assert 0.0 <= result.control.brake <= 1.0
    assert not (result.control.throttle > 0 and result.control.brake > 0)
    for reason, kwargs in [
        ("timeout", {"timed_out": True}),
        ("invalid_plan", {"selected_plan": {"xy": [[0.0, np.nan]], "timestamps": [0.0], "valid": True}}),
        ("no_candidate", {"selected_plan": None}),
        ("missing_sensor", {"sensor_available": False}),
        ("stale_episode", {"episode_id": "old-episode", "sequence": 1, "timestamp": 1.0}),
    ]:
        result = controller.step(kwargs.pop("selected_plan", plan), {"speed": 1.0}, dt=0.1, **kwargs)
        assert result.used_fallback
        assert result.failure_reason == reason
        assert result.control.throttle == 0.0
        assert result.control.brake == pytest.approx(0.6)


def test_in_memory_transport_is_typed_and_does_not_use_global_rng():
    transport = InMemoryTransport()
    transport.send(message())
    received = transport.receive()
    assert isinstance(received, BridgeMessage)
    np.random.seed(4)
    before = np.random.random()
    transport.send(message(sequence=1, timestamp=1.0))
    after = np.random.random()
    np.random.seed(4)
    assert before == np.random.random()
    assert after == np.random.random()


def test_future_ground_truth_composite_keys_are_rejected_recursively():
    base = message().to_mapping()
    for nested in (
        {"future_gt_trajectory": [[1.0, 2.0]]},
        {"foo_ground_truth": {"value": 1}},
        {"nested": {"future_gt_trajectory": [1.0]}},
    ):
        payload = {**base, "sensor_bundle": nested}
        with pytest.raises(FutureGroundTruthError):
            BridgeMessage.from_mapping(payload)


def test_concatenated_future_ground_truth_composites_fail_closed_and_future_world_is_allowed():
    base = message().to_mapping()
    for key in (
        "futuregttrajectory",
        "targettrajectory",
        "futuretargettrajectory",
        "futureego",
        "futureactor",
        "futurelabel",
        "futureoracle",
    ):
        payload = {**base, "sensor_bundle": {key: [1.0]}}
        with pytest.raises(FutureGroundTruthError):
            BridgeMessage.from_mapping(payload)

    allowed = {**base, "world_outputs": {"future_world": {"occupancy": [0.0]}}}
    assert BridgeMessage.from_mapping(allowed).world_outputs["future_world"] == {"occupancy": [0.0]}


def test_concatenated_ground_truth_composites_fail_recursively():
    base = message().to_mapping()
    for key in (
        "gttrajectory", "gtactor", "gtlabel", "groundtrajectory",
        "foo_gt_trajectory", "predicted_gt_trajectory",
        "fooGTtrajectory", "fooGTTrajectory", "fooGTActor",
    ):
        payload = {**base, "sensor_bundle": {"nested": {key: [1.0]}}}
        with pytest.raises(FutureGroundTruthError):
            BridgeMessage.from_mapping(payload)


def test_control_response_requires_bounded_mutually_exclusive_actuators():
    common = {
        "episode_id": "episode-a", "sequence": 1, "timestamp": 1.0,
        "requested_deadline": 1.05, "model_fingerprint": "model-sha",
        "config_fingerprint": "config-sha",
    }
    for control in (
        {"steer": 0.0, "throttle": 2.0, "brake": 0.0},
        {"steer": 2.0, "throttle": 0.0, "brake": 0.0},
        {"steer": 0.0, "throttle": 0.2, "brake": 0.1},
        {"steer": 0.0, "throttle": 0.0},
        {"steer": "0", "throttle": "0.2", "brake": "0"},
        {"steer": False, "throttle": 0.0, "brake": 0.0},
        {"steer": 0.0, "throttle": True, "brake": 0.0},
    ):
        with pytest.raises(BridgeError):
            ControlResponse(control=control, **common)


def test_controller_config_rejects_unbounded_actuators_and_fallback_is_safe():
    with pytest.raises(ValueError):
        ControllerConfig(max_steer=1.1)
    with pytest.raises(ValueError):
        ControllerConfig(max_throttle=1.1)
    with pytest.raises(ValueError):
        ControllerConfig(max_brake=1.1)
    with pytest.raises(ValueError):
        ControllerConfig(max_brake=0.4, fallback_brake=0.5)
    controller = Controller(ControllerConfig(max_brake=0.8, fallback_brake=0.7))
    result = controller.step(None, {"speed": 1.0}, candidate_available=False)
    assert result.used_fallback
    assert result.failure_reason == "no_candidate"
    assert result.control == VehicleControl(0.0, 0.0, 0.7)


def test_controller_invalid_deadline_and_timestamp_return_explicit_fallbacks():
    controller = Controller()
    timeout = controller.step(None, {"speed": 1.0}, candidate_available=False, requested_deadline="not-a-number")
    assert timeout.used_fallback
    assert timeout.failure_reason == "timeout"
    invalid = controller.step(None, {"speed": 1.0}, candidate_available=False, timestamp="not-a-number")
    assert invalid.used_fallback
    assert invalid.failure_reason == "invalid_plan"


def test_bridge_mapping_missing_required_fields_raises_bridge_error():
    payload = message().to_mapping()
    payload.pop("ego_state")
    with pytest.raises(BridgeError, match="missing"):
        BridgeMessage.from_mapping(payload)
    response = ControlResponse(
        episode_id="episode-a", sequence=1, timestamp=1.0,
        requested_deadline=1.05, model_fingerprint="model-sha", config_fingerprint="config-sha",
        control={"steer": 0.0, "throttle": 0.2, "brake": 0.0},
    ).to_mapping()
    response.pop("control")
    with pytest.raises(BridgeError, match="missing"):
        ControlResponse.from_mapping(response)
