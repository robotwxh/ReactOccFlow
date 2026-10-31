"""Observed detection validation and deterministic nearest-neighbor tracks."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import numpy as np

from .lidar import _FORBIDDEN, _reject_forbidden, _validate_timestamp


@dataclass(frozen=True)
class ActorDetection:
    x: float; y: float; z: float; l: float; w: float; h: float; yaw: float
    class_name: str; score: float; track_id: Any = None; valid: bool = True

    @classmethod
    def from_mapping(cls, mapping: Mapping[str, Any], allowed_classes: set[str] | None = None) -> "ActorDetection":
        if not isinstance(mapping, Mapping):
            raise TypeError("actor detection must be a mapping")
        _reject_forbidden(mapping)
        required = {"x", "y", "z", "l", "w", "h", "yaw", "class", "score"}
        missing = sorted(required.difference(mapping))
        if missing:
            raise ValueError(f"missing actor fields: {', '.join(missing)}")
        try:
            values = [float(mapping[key]) for key in ("x", "y", "z", "l", "w", "h", "yaw", "score")]
        except (TypeError, ValueError) as exc:
            raise ValueError("actor geometry and score must be numeric") from exc
        if not np.isfinite(values).all() or any(v <= 0 for v in values[3:6]):
            raise ValueError("actor geometry and score must be finite; dimensions must be positive")
        if not 0.0 <= values[7] <= 1.0:
            raise ValueError("actor score must be in [0, 1]")
        class_name = mapping["class"]
        if not isinstance(class_name, str) or not class_name:
            raise ValueError("actor class must be a non-empty string")
        valid = allowed_classes is None or class_name in allowed_classes
        return cls(*values[:7], class_name, values[7], mapping.get("track_id"), valid)


@dataclass(frozen=True)
class ActorTrackBatch:
    current: np.ndarray
    current_mask: np.ndarray
    past: np.ndarray
    observed_mask: np.ndarray
    track_ids: np.ndarray
    classes: np.ndarray
    scores: np.ndarray
    timestamps: np.ndarray

    @property
    def actor_history(self) -> np.ndarray:
        return self.past

    @property
    def actor_observed_mask(self) -> np.ndarray:
        return self.observed_mask


@dataclass
class _Track:
    public_id: Any
    class_name: str
    states: list[ActorDetection | None]
    last_detection: ActorDetection | None = None


class ActorFrontend:
    def __init__(self, gating_distance: float = 2.0, allowed_classes: set[str] | None = None):
        if not np.isfinite(gating_distance) or gating_distance <= 0:
            raise ValueError("gating_distance must be positive and finite")
        self.gating_distance = float(gating_distance)
        self.allowed_classes = None if allowed_classes is None else set(allowed_classes)

    def _parse_frame(self, frame: Mapping[str, Any]):
        _reject_forbidden(frame)
        if not isinstance(frame, Mapping):
            raise TypeError("actor frame must be a mapping")
        coordinate_frame = frame.get("coordinate_frame", frame.get("frame"))
        if coordinate_frame != "E0":
            raise ValueError("actor detections must declare coordinate_frame='E0'")
        if "timestamp" not in frame:
            raise ValueError("actor frame requires timestamp")
        timestamp = _validate_timestamp(frame["timestamp"])
        detections = frame.get("detections", ())
        if detections is None:
            detections = ()
        parsed = [ActorDetection.from_mapping(detection, self.allowed_classes) for detection in detections]
        return timestamp, parsed

    def associate(self, frames: Sequence[Mapping[str, Any]]) -> ActorTrackBatch:
        parsed_frames = [self._parse_frame(frame) for frame in frames]
        timestamps = [item[0] for item in parsed_frames]
        if any(b <= a for a, b in zip(timestamps, timestamps[1:])):
            raise ValueError("actor timestamps must be strictly increasing")
        tracks: list[_Track] = []
        explicit: dict[Any, _Track] = {}
        used_public_ids: set[Any] = set()
        next_id = 0

        def allocate_public_id(requested: Any) -> tuple[Any, int]:
            """Return a deterministic ID that is unique among all public tracks."""
            nonlocal next_id
            if requested is None:
                while next_id in used_public_ids:
                    next_id += 1
                public_id = next_id
                next_id += 1
            elif requested not in used_public_ids:
                public_id = requested
            else:
                suffix = 1
                public_id = f"{requested}#{suffix}"
                while public_id in used_public_ids:
                    suffix += 1
                    public_id = f"{requested}#{suffix}"
            used_public_ids.add(public_id)
            return public_id, next_id
        for frame_index, (_, detections) in enumerate(parsed_frames):
            for track in tracks:
                track.states.append(None)
            invalid = [d for d in detections if not d.valid]
            for detection in sorted(invalid, key=lambda d: (d.track_id is None, str(d.track_id), d.x, d.y, d.z, d.class_name)):
                if detection.track_id is not None and detection.track_id in explicit:
                    continue
                public_id, next_id = allocate_public_id(detection.track_id)
                track = _Track(public_id, detection.class_name, [None] * (frame_index + 1))
                tracks.append(track)
                if detection.track_id is not None:
                    explicit[detection.track_id] = track
            valid = [d for d in detections if d.valid]
            valid.sort(key=lambda d: (d.track_id is None, str(d.track_id), d.x, d.y, d.z, d.class_name))
            used: set[int] = set()
            for detection in valid:
                track: _Track | None = None
                if detection.track_id is not None:
                    track = explicit.get(detection.track_id)
                    if track is not None and track.last_detection is not None:
                        distance = float(np.linalg.norm(np.asarray([detection.x - track.last_detection.x, detection.y - track.last_detection.y])))
                        if distance > self.gating_distance:
                            track = None
                else:
                    candidates = []
                    for index, candidate in enumerate(tracks):
                        if index in used or candidate.last_detection is None or candidate.class_name != detection.class_name:
                            continue
                        distance = float(np.linalg.norm(np.asarray([detection.x - candidate.last_detection.x, detection.y - candidate.last_detection.y])))
                        if distance <= self.gating_distance:
                            candidates.append((distance, str(candidate.public_id), index, candidate))
                    if candidates:
                        track = min(candidates, key=lambda item: item[:3])[-1]
                if track is None:
                    public_id, next_id = allocate_public_id(detection.track_id)
                    track = _Track(public_id, detection.class_name, [None] * frame_index + [None])
                    tracks.append(track)
                    if detection.track_id is not None:
                        explicit[detection.track_id] = track
                track.states[-1] = detection
                track.last_detection = detection
                used.add(tracks.index(track))
        count = len(tracks); length = len(parsed_frames)
        current = np.zeros((count, 8), dtype=np.float32)
        current_mask = np.zeros(count, dtype=bool)
        past = np.zeros((count, length, 7), dtype=np.float32)
        observed = np.zeros((count, length), dtype=bool)
        classes = np.empty(count, dtype=object); scores = np.zeros((count, length), dtype=np.float32)
        ids = np.empty(count, dtype=object)
        for track_index, track in enumerate(tracks):
            ids[track_index] = track.public_id; classes[track_index] = track.class_name
            previous: ActorDetection | None = None; previous_time: float | None = None
            for frame_index, detection in enumerate(track.states):
                if detection is None:
                    continue
                observed[track_index, frame_index] = True; scores[track_index, frame_index] = detection.score
                velocity = (0.0, 0.0)
                if previous is not None and previous_time is not None and timestamps[frame_index] > previous_time:
                    dt = timestamps[frame_index] - previous_time
                    velocity = ((detection.x - previous.x) / dt, (detection.y - previous.y) / dt)
                past[track_index, frame_index] = (detection.x, detection.y, detection.z, np.cos(detection.yaw), np.sin(detection.yaw), velocity[0], velocity[1])
                previous, previous_time = detection, timestamps[frame_index]
            latest = next((d for d in reversed(track.states) if d is not None), None)
            if latest is not None and track.states[-1] is not None:
                current[track_index] = (latest.x, latest.y, latest.z, latest.l, latest.w, latest.h, latest.yaw, latest.score)
                current_mask[track_index] = True
        return ActorTrackBatch(current, current_mask, past, observed, ids, classes, scores, np.asarray(timestamps, dtype=np.float64))
