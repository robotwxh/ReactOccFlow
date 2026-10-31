import numpy as np
import pytest

from reactoccflow.geometry import transform_matrix


def _lidar(points, timestamp, transform=None, frame="lidar"):
    return {
        "points": np.asarray(points, dtype=np.float64),
        "timestamp": timestamp,
        "sensor_to_e0": np.eye(4) if transform is None else transform,
        "frame": frame,
    }


def _actors(timestamp, detections, frame="E0"):
    return {"timestamp": timestamp, "detections": detections, "coordinate_frame": frame}


def test_lidar_rejects_future_gt_nonfinite_and_bad_timestamps():
    from reactoccflow.perception import LidarObservation, Perception

    with pytest.raises(ValueError, match="future_semantics"):
        LidarObservation.from_mapping(_lidar([[0, 0, 0]], 0.0) | {"future_semantics": True})
    with pytest.raises(ValueError, match="finite"):
        LidarObservation([[np.nan, 0, 0]], 0.0, np.eye(4), "lidar")
    with pytest.raises(ValueError, match="frame"):
        LidarObservation([[0, 0, 0]], 0.0, np.eye(4), "")
    with pytest.raises(ValueError, match="future"):
        Perception().observe([_lidar([[0, 0, 0]], 1.0), _lidar([[0, 0, 0]], 0.5)])
    with pytest.raises(ValueError, match="gt|target"):
        Perception().observe([_lidar([[0, 0, 0]], 0.0) | {"target": object()}])


def test_voxel_counts_height_intensity_and_out_of_bounds_exactly():
    from reactoccflow.perception import LidarObservation, VoxelBEVExtractor, VoxelGridConfig

    cfg = VoxelGridConfig(bounds=(0, 0, 0, 2, 2, 2), resolution=(1, 1, 1))
    obs = LidarObservation(
        [[0.1, 0.1, 0.1, 2.0], [0.2, 0.3, 0.9, 4.0], [1.1, 1.1, 1.1, 8.0], [-1, 0, 0, 9.0], [2, 0, 0, 10.0]],
        0.0,
        np.eye(4),
        "lidar",
    )
    result = VoxelBEVExtractor(cfg).extract([obs])
    assert result.voxel_features.shape == (1, 2, 2, 2, 4)
    assert result.voxel_mask.shape == (1, 2, 2, 2)
    assert result.out_of_bounds_count.tolist() == [2]
    np.testing.assert_allclose(result.voxel_features[0, 0, 0, 0], [2, 0.1, 0.9, 3.0])
    np.testing.assert_allclose(result.voxel_features[0, 1, 1, 1], [1, 1.1, 1.1, 8.0])
    assert np.isfinite(result.voxel_features).all()


def test_transform_roundtrip_is_applied_before_voxelization():
    from reactoccflow.perception import LidarObservation, VoxelBEVExtractor, VoxelGridConfig

    cfg = VoxelGridConfig(bounds=(0, 0, 0, 2, 2, 2), resolution=(1, 1, 1))
    # Sensor point (0, 0, 0) lands in E0 voxel (1, 0, 0).
    obs = LidarObservation([[0.1, 0.1, 0.1]], 0.0, transform_matrix([1, 0, 0]), "lidar")
    result = VoxelBEVExtractor(cfg).extract([obs])
    assert result.voxel_mask[0, 1, 0, 0]
    assert not result.voxel_mask[0, 0, 0, 0]


def test_actor_association_stable_ids_and_missing_masks():
    from reactoccflow.perception import ActorFrontend

    frontend = ActorFrontend(gating_distance=1.0)
    frames = [
        _actors(0.0, [{"x": 0, "y": 0, "z": 0, "l": 4, "w": 2, "h": 1.5, "yaw": 0, "class": "car", "score": 0.9}]),
        _actors(1.0, []),
        _actors(2.0, [{"x": 0.2, "y": 0, "z": 0, "l": 4, "w": 2, "h": 1.5, "yaw": 0.1, "class": "car", "score": 0.8}]),
    ]
    result = frontend.associate(frames)
    assert result.track_ids.tolist() == [0]
    assert result.observed_mask.tolist() == [[True, False, True]]
    assert result.current_mask.tolist() == [True]
    assert result.current.shape == (1, 8)
    assert result.past.shape == (1, 3, 7)


def test_actor_explicit_ids_and_unknown_class_fail_closed():
    from reactoccflow.perception import ActorFrontend

    frontend = ActorFrontend(allowed_classes={"car"})
    frame = _actors(0.0, [{"x": 0, "y": 0, "z": 0, "l": 4, "w": 2, "h": 1.5, "yaw": 0, "class": "alien", "score": 0.9, "track_id": "a"}])
    result = frontend.associate([frame])
    assert result.observed_mask.tolist() == [[False]]
    with pytest.raises(ValueError, match="E0"):
        frontend.associate([_actors(0.0, [], frame="lidar")])


def test_observed_bundle_has_placeholder_and_rejects_future_keys_without_rng_changes():
    from reactoccflow.perception import Perception, VoxelGridConfig

    sensor = _lidar([[0, 0, 0, 1]], 0.0) | {
        "detections": [{"x": 0, "y": 0, "z": 0, "l": 4, "w": 2, "h": 1.5, "yaw": 0, "class": "car", "score": 1.0}],
        "coordinate_frame": "E0",
    }
    rng = np.random.default_rng(123)
    before = rng.random(4)
    bundle = Perception(VoxelGridConfig(bounds=(-1, -1, -1, 1, 1, 1), resolution=(1, 1, 1))).observe([sensor])
    after = rng.random(4)
    control = np.random.default_rng(123)
    np.testing.assert_allclose(before, control.random(4))
    assert bundle.oracle is False
    assert bundle.latent_frontend_input is None
    assert bundle.source == "observed_lidar_and_detections"
    assert "trained occupancy logits" in bundle.limitation
    with pytest.raises(ValueError, match="future_actor"):
        Perception().observe([sensor | {"future_actor": []}])


def test_observe_requires_explicit_e0_actor_frame_when_detections_are_supplied():
    from reactoccflow.perception import Perception

    detection = {"x": 0, "y": 0, "z": 0, "l": 4, "w": 2, "h": 1.5, "yaw": 0, "class": "car", "score": 0.9}
    missing_frame = _lidar([[0, 0, 0]], 0.0) | {"detections": [detection]}
    with pytest.raises(ValueError, match="coordinate_frame.*E0"):
        Perception().observe([missing_frame])

    mismatched_frame = missing_frame | {"coordinate_frame": "lidar"}
    with pytest.raises(ValueError, match="coordinate_frame.*E0"):
        Perception().observe([mismatched_frame])


def test_explicit_track_id_gate_break_allocates_unique_deterministic_public_id():
    from reactoccflow.perception import ActorFrontend

    detection = lambda x: {
        "x": x, "y": 0, "z": 0, "l": 4, "w": 2, "h": 1.5,
        "yaw": 0, "class": "car", "score": 0.9, "track_id": "a",
    }
    result = ActorFrontend(gating_distance=1.0).associate([
        _actors(0.0, [detection(0.0)]),
        _actors(1.0, [detection(10.0)]),
        _actors(2.0, [detection(10.2)]),
    ])

    assert result.track_ids.tolist() == ["a", "a#1"]
    assert len(set(result.track_ids.tolist())) == 2
    assert result.observed_mask.tolist() == [[True, False, False], [False, True, True]]


def test_voxel_indices_avoid_int64_cast_warning_for_huge_finite_points():
    import warnings
    from reactoccflow.perception import VoxelGridConfig

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        indices, valid = VoxelGridConfig(bounds=(0, 0, 0, 2, 2, 2), resolution=(1, 1, 1)).indices(
            np.asarray([[1e300, 0.0, 0.0]], dtype=np.float64)
        )

    assert not any(issubclass(item.category, RuntimeWarning) for item in caught)
    assert valid.tolist() == [False]
    assert indices.dtype == np.int64
