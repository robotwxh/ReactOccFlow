"""Observed-only perception bundle joining LiDAR voxels and actor tracks."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import numpy as np

from .actors import ActorFrontend, ActorTrackBatch
from .lidar import LidarObservation, VoxelBEVExtractor, VoxelFeatureBatch, VoxelGridConfig, _reject_forbidden


@dataclass(frozen=True)
class ObservationBundle:
    voxel_features: np.ndarray
    voxel_mask: np.ndarray
    bev_features: np.ndarray
    bev_mask: np.ndarray
    out_of_bounds_count: np.ndarray
    actor_history: np.ndarray
    actor_observed_mask: np.ndarray
    actor_track_ids: np.ndarray
    actor_current: np.ndarray
    actor_current_mask: np.ndarray
    actor_classes: np.ndarray
    actor_scores: np.ndarray
    frame_metadata: tuple[dict[str, Any], ...]
    latent_frontend_input: Any
    oracle: bool
    source: str
    limitation: str


class Perception:
    """CPU observed-only frontend; it emits features and tracks, never logits."""

    def __init__(self, grid: VoxelGridConfig | None = None, *, gating_distance: float = 2.0, allowed_classes: set[str] | None = None):
        self.voxel_extractor = VoxelBEVExtractor(grid)
        self.actor_frontend = ActorFrontend(gating_distance=gating_distance, allowed_classes=allowed_classes)

    def observe(self, sensor_history: Sequence[LidarObservation | Mapping[str, Any]]) -> ObservationBundle:
        if not sensor_history:
            raise ValueError("sensor_history must contain at least one current or past observation")
        observations: list[LidarObservation] = []
        actor_frames: list[Mapping[str, Any]] = []
        for item in sensor_history:
            if isinstance(item, LidarObservation):
                observations.append(item)
                actor_frames.append({"timestamp": item.timestamp, "coordinate_frame": "E0", "detections": []})
                continue
            if not isinstance(item, Mapping):
                raise TypeError("sensor_history entries must be mappings or LidarObservation")
            _reject_forbidden(item)
            observations.append(LidarObservation.from_mapping(item))
            detections = item.get("detections", ())
            if detections is None:
                detections = ()
            has_detections = bool(detections)
            if has_detections and "coordinate_frame" not in item:
                raise ValueError("actor detections require explicit coordinate_frame='E0'")
            actor_coordinate_frame = item.get("coordinate_frame", "E0")
            if has_detections and actor_coordinate_frame != "E0":
                raise ValueError("actor detections must declare coordinate_frame='E0'")
            actor_frames.append({"timestamp": item["timestamp"], "coordinate_frame": actor_coordinate_frame, "detections": detections})
        voxel = self.voxel_extractor.extract(observations)
        actors = self.actor_frontend.associate(actor_frames)
        return ObservationBundle(
            voxel_features=voxel.voxel_features,
            voxel_mask=voxel.voxel_mask,
            bev_features=voxel.bev_features,
            bev_mask=voxel.bev_mask,
            out_of_bounds_count=voxel.out_of_bounds_count,
            actor_history=actors.past,
            actor_observed_mask=actors.observed_mask,
            actor_track_ids=actors.track_ids,
            actor_current=actors.current,
            actor_current_mask=actors.current_mask,
            actor_classes=actors.classes,
            actor_scores=actors.scores,
            frame_metadata=voxel.frame_metadata,
            latent_frontend_input=None,
            oracle=False,
            source="observed_lidar_and_detections",
            limitation="This frontend emits voxel features and observed tracks; it does not emit trained occupancy logits.",
        )
