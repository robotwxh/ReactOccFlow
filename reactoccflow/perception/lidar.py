"""Observed-only LiDAR point cloud validation and deterministic voxel features."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import numpy as np

from reactoccflow.geometry import transform_points

_FORBIDDEN = {"future_semantics", "future_actor", "future_actors", "target", "gt", "gt_occupancy", "gt_actor", "labels", "future_labels"}


def _reject_forbidden(mapping: Mapping[str, Any]) -> None:
    found = sorted(_FORBIDDEN.intersection(mapping))
    if found:
        raise ValueError(f"future/GT fields are forbidden in observed input: {', '.join(found)}")


def _validate_timestamp(value: Any) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError("timestamp must be finite") from exc
    if not np.isfinite(result):
        raise ValueError("timestamp must be finite")
    return result


@dataclass(frozen=True)
class LidarObservation:
    """One LiDAR observation expressed in a named sensor frame.

    ``sensor_to_e0`` is an explicit column-vector SE(3) transform. The frame
    name is mandatory so that an unlabeled array cannot silently enter E0.
    """

    points: np.ndarray
    timestamp: float
    sensor_to_e0: np.ndarray
    frame: str

    def __init__(self, points, timestamp, sensor_to_e0, frame, **kwargs):
        if kwargs:
            _reject_forbidden(kwargs)
            raise ValueError(f"unknown observation fields: {', '.join(sorted(kwargs))}")
        points_array = np.asarray(points, dtype=np.float64)
        if points_array.ndim != 2 or points_array.shape[1] < 3:
            raise ValueError("points must have shape (N, >=3) with xyz columns")
        if not np.isfinite(points_array).all():
            raise ValueError("points must contain only finite values")
        timestamp_value = _validate_timestamp(timestamp)
        if not isinstance(frame, str) or not frame.strip():
            raise ValueError("a non-empty coordinate frame is required")
        transform = np.asarray(sensor_to_e0, dtype=np.float64)
        if transform.shape != (4, 4) or not np.isfinite(transform).all():
            raise ValueError("sensor_to_e0 must be a finite 4x4 SE(3) transform")
        # transform_points performs the complete orthonormality/homogeneous-row check.
        transform_points(transform, np.zeros((0, 3), dtype=np.float64))
        object.__setattr__(self, "points", points_array.copy())
        object.__setattr__(self, "timestamp", timestamp_value)
        object.__setattr__(self, "sensor_to_e0", transform.copy())
        object.__setattr__(self, "frame", frame.strip())

    @classmethod
    def from_mapping(cls, mapping: Mapping[str, Any]) -> "LidarObservation":
        if not isinstance(mapping, Mapping):
            raise TypeError("LiDAR frame must be a mapping or LidarObservation")
        _reject_forbidden(mapping)
        required = {"points", "timestamp", "sensor_to_e0", "frame"}
        missing = sorted(required.difference(mapping))
        if missing:
            raise ValueError(f"missing LiDAR fields: {', '.join(missing)}")
        return cls(mapping["points"], mapping["timestamp"], mapping["sensor_to_e0"], mapping["frame"])

    def points_in_e0(self) -> np.ndarray:
        return transform_points(self.sensor_to_e0, self.points[:, :3])


@dataclass(frozen=True)
class VoxelGridConfig:
    """Axis-aligned E0 grid with an exclusive upper bound."""

    bounds: tuple[float, float, float, float, float, float] = (-40.0, -40.0, -1.0, 40.0, 40.0, 5.4)
    resolution: tuple[float, float, float] = (0.4, 0.4, 0.4)

    def __post_init__(self):
        bounds = tuple(float(v) for v in self.bounds)
        resolution = tuple(float(v) for v in self.resolution)
        if len(bounds) != 6 or len(resolution) != 3 or not np.isfinite(bounds + resolution).all():
            raise ValueError("bounds and resolution must be finite tuples")
        lo = np.asarray(bounds[:3]); hi = np.asarray(bounds[3:]); res = np.asarray(resolution)
        if np.any(hi <= lo) or np.any(res <= 0):
            raise ValueError("upper bounds must exceed lower bounds and resolution must be positive")
        shape_float = (hi - lo) / res
        shape = np.rint(shape_float).astype(int)
        if np.any(shape < 1) or not np.allclose(shape_float, shape, atol=1e-8):
            raise ValueError("bounds extents must be integer multiples of resolution")
        object.__setattr__(self, "bounds", bounds)
        object.__setattr__(self, "resolution", resolution)

    @property
    def shape(self) -> tuple[int, int, int]:
        lo = np.asarray(self.bounds[:3]); hi = np.asarray(self.bounds[3:]); res = np.asarray(self.resolution)
        return tuple(np.rint((hi - lo) / res).astype(int).tolist())

    def indices(self, points_e0: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        lo = np.asarray(self.bounds[:3]); hi = np.asarray(self.bounds[3:]); res = np.asarray(self.resolution)
        finite = np.isfinite(points_e0).all(axis=1)
        in_bounds = finite & (points_e0 >= lo).all(axis=1) & (points_e0 < hi).all(axis=1)
        indices = np.zeros(points_e0.shape, dtype=np.int64)
        valid = np.zeros(points_e0.shape[0], dtype=bool)
        if np.any(in_bounds):
            continuous = (points_e0[in_bounds] - lo) / res
            floored = np.floor(continuous)
            info = np.iinfo(np.int64)
            representable = np.all((floored >= info.min) & (floored <= info.max), axis=1)
            rows = np.flatnonzero(in_bounds)[representable]
            indices[rows] = floored[representable].astype(np.int64)
            shape = np.asarray(self.shape)
            valid[rows] = (indices[rows] >= 0).all(axis=1) & (indices[rows] < shape).all(axis=1)
        return indices, valid


@dataclass(frozen=True)
class VoxelFeatureBatch:
    voxel_features: np.ndarray
    voxel_mask: np.ndarray
    bev_features: np.ndarray
    bev_mask: np.ndarray
    out_of_bounds_count: np.ndarray
    frame_metadata: tuple[dict[str, Any], ...]

    @property
    def features(self) -> np.ndarray:
        return self.voxel_features


class VoxelBEVExtractor:
    """Count points and summarize height/intensity without labels or clamping."""

    def __init__(self, grid: VoxelGridConfig | None = None):
        self.grid = grid or VoxelGridConfig()

    @staticmethod
    def _ordered(history: Sequence[LidarObservation]) -> list[LidarObservation]:
        observations = list(history)
        timestamps = [obs.timestamp for obs in observations]
        if any(not isinstance(obs, LidarObservation) for obs in observations):
            raise TypeError("sensor history must contain LidarObservation instances")
        if any(b <= a for a, b in zip(timestamps, timestamps[1:])):
            raise ValueError("sensor timestamps must be strictly increasing; future or duplicate samples are forbidden")
        return observations

    def extract(self, history: Sequence[LidarObservation]) -> VoxelFeatureBatch:
        observations = self._ordered(history)
        nx, ny, nz = self.grid.shape
        features = np.zeros((len(observations), nx, ny, nz, 4), dtype=np.float32)
        masks = np.zeros((len(observations), nx, ny, nz), dtype=bool)
        bev = np.zeros((len(observations), nx, ny, 4), dtype=np.float32)
        bev_masks = np.zeros((len(observations), nx, ny), dtype=bool)
        out_counts = np.zeros(len(observations), dtype=np.int64)
        metadata: list[dict[str, Any]] = []
        for frame_idx, observation in enumerate(observations):
            points_e0 = observation.points_in_e0()
            indices, valid = self.grid.indices(points_e0)
            out_counts[frame_idx] = int((~valid).sum())
            valid_points = points_e0[valid]
            valid_indices = indices[valid]
            intensities = observation.points[valid, 3] if observation.points.shape[1] >= 4 else np.zeros(valid_points.shape[0], dtype=np.float64)
            # Stable integer accumulation avoids a dependency on global RNG or ordering.
            for idx, point, intensity in zip(valid_indices, valid_points, intensities):
                x, y, z = (int(v) for v in idx)
                cell = features[frame_idx, x, y, z]
                if not masks[frame_idx, x, y, z]:
                    cell[:] = (1.0, point[2], point[2], float(intensity))
                    masks[frame_idx, x, y, z] = True
                else:
                    count = cell[0] + 1.0
                    cell[1] = min(cell[1], point[2])
                    cell[2] = max(cell[2], point[2])
                    cell[3] = (cell[3] * (count - 1.0) + float(intensity)) / count
                    cell[0] = count
            bev_masks[frame_idx] = masks[frame_idx].any(axis=2)
            for x in range(nx):
                for y in range(ny):
                    occupied = masks[frame_idx, x, y]
                    if occupied.any():
                        cells = features[frame_idx, x, y, occupied]
                        bev[frame_idx, x, y, 0] = cells[:, 0].sum()
                        bev[frame_idx, x, y, 1] = cells[:, 1].min()
                        bev[frame_idx, x, y, 2] = cells[:, 2].max()
                        counts = cells[:, 0]
                        bev[frame_idx, x, y, 3] = float(np.average(cells[:, 3], weights=counts))
            metadata.append({"timestamp": observation.timestamp, "frame": observation.frame, "num_points": int(observation.points.shape[0]), "num_in_bounds": int(valid.sum()), "num_out_of_bounds": int((~valid).sum()), "coordinate_frame": "E0"})
        return VoxelFeatureBatch(features, masks, bev, bev_masks, out_counts, tuple(metadata))
