"""Observed-only low-level control contract with explicit fail-safe fallback."""
from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any, Mapping

import numpy as np

from .bridge import FreshnessGate


def _finite_array(value: Any, *, name: str, shape_last: int | None = None) -> np.ndarray:
    array = np.asarray(value, dtype=np.float64)
    if array.size == 0 or not np.isfinite(array).all():
        raise ValueError(f"{name} must be finite and nonempty")
    if shape_last is not None and (array.ndim != 2 or array.shape[1] != shape_last):
        raise ValueError(f"{name} must have shape [N, {shape_last}]")
    return array


def interpolate_xy(xy: Any, timestamps: Any, query_timestamp: float) -> np.ndarray:
    points = _finite_array(xy, name="plan xy", shape_last=2)
    times = _finite_array(timestamps, name="plan timestamps")
    if times.ndim != 1 or len(times) != len(points) or len(times) == 0 or np.any(np.diff(times) <= 0):
        raise ValueError("plan timestamps must be finite, strictly increasing, and match plan xy")
    query = float(query_timestamp)
    if not math.isfinite(query):
        raise ValueError("query timestamp must be finite")
    return np.array([np.interp(query, times, points[:, axis]) for axis in range(2)], dtype=np.float64)


@dataclass(frozen=True)
class ControllerConfig:
    max_steer: float = 1.0
    max_throttle: float = 1.0
    max_brake: float = 1.0
    max_acceleration: float = 3.0
    max_deceleration: float = 6.0
    fallback_brake: float = 0.5
    steering_gain: float = 1.5

    def __post_init__(self) -> None:
        for name in ("max_steer", "max_throttle", "max_brake", "max_acceleration", "max_deceleration", "fallback_brake", "steering_gain"):
            value = float(getattr(self, name))
            if not math.isfinite(value) or value < 0:
                raise ValueError(f"{name} must be finite and nonnegative")
        if self.max_steer == 0 or self.max_throttle == 0 or self.max_brake == 0 or self.max_acceleration == 0 or self.max_deceleration == 0:
            raise ValueError("control limits and accelerations must be positive")
        if self.max_steer > 1.0 or self.max_throttle > 1.0 or self.max_brake > 1.0:
            raise ValueError("actuator limits must satisfy steer [-1,1] and throttle/brake [0,1]")
        if self.fallback_brake > self.max_brake:
            raise ValueError("fallback_brake cannot exceed max_brake")


@dataclass(frozen=True)
class VehicleControl:
    steer: float
    throttle: float
    brake: float

    def __post_init__(self) -> None:
        values = (float(self.steer), float(self.throttle), float(self.brake))
        if not all(math.isfinite(v) for v in values):
            raise ValueError("control values must be finite")
        if not -1.0 <= values[0] <= 1.0 or not 0.0 <= values[1] <= 1.0 or not 0.0 <= values[2] <= 1.0:
            raise ValueError("control values must satisfy steer [-1,1], throttle/brake [0,1]")
        if values[1] > 0 and values[2] > 0:
            raise ValueError("throttle and brake cannot both be positive")
        object.__setattr__(self, "steer", values[0]); object.__setattr__(self, "throttle", values[1]); object.__setattr__(self, "brake", values[2])


@dataclass(frozen=True)
class ControlResult:
    control: VehicleControl
    failure_reason: str | None = None
    episode_id: str | None = None
    sequence: int | None = None
    timestamp: float | None = None
    used_fallback: bool = False

    @property
    def reason(self) -> str | None:
        return self.failure_reason


class Controller:
    def __init__(self, config: ControllerConfig | None = None):
        self.config = config or ControllerConfig()
        self._freshness = FreshnessGate()

    def reset_episode(self, episode_id: str) -> None:
        self._freshness.reset_episode(episode_id)

    def _fallback(self, reason: str, *, episode_id: str | None, sequence: int | None, timestamp: float | None) -> ControlResult:
        return ControlResult(VehicleControl(0.0, 0.0, self.config.fallback_brake), reason, episode_id, sequence, timestamp, True)

    def _validate_plan(self, selected_plan: Any) -> tuple[np.ndarray, np.ndarray]:
        if not isinstance(selected_plan, Mapping) and all(hasattr(selected_plan, key) for key in ("xy", "timestamps")):
            selected_plan = {key: getattr(selected_plan, key) for key in ("xy", "timestamps", "valid") if hasattr(selected_plan, key)}
        if not isinstance(selected_plan, Mapping):
            raise ValueError("selected plan must be a mapping or typed candidate")
        unknown = set(selected_plan) - {"xy", "timestamps", "valid", "candidate_id", "is_brake"}
        if unknown:
            if any("future" in str(key).lower() or str(key).lower() in {"gt", "ground_truth", "target", "label"} for key in unknown):
                raise ValueError("selected plan is observed-only")
            raise ValueError(f"unknown selected plan key: {sorted(unknown, key=str)[0]}")
        if not bool(selected_plan.get("valid", True)):
            raise ValueError("selected plan is invalid")
        return _finite_array(selected_plan["xy"], name="plan xy", shape_last=2), _finite_array(selected_plan["timestamps"], name="plan timestamps")

    def step(self, selected_plan: Any, ego_state: Mapping[str, Any], dt: float = 0.05, *, episode_id: str | None = None, sequence: int | None = None, timestamp: float | None = None, sensor_available: bool = True, timed_out: bool = False, candidate_available: bool = True, requested_deadline: float | None = None) -> ControlResult:
        episode = episode_id
        stamp: float | None = None
        if timestamp is not None:
            try:
                stamp = float(timestamp)
            except (TypeError, ValueError, OverflowError):
                return self._fallback("invalid_plan", episode_id=episode, sequence=sequence, timestamp=None)
            if not math.isfinite(stamp):
                return self._fallback("invalid_plan", episode_id=episode, sequence=sequence, timestamp=stamp)
        deadline: float | None = None
        if requested_deadline is not None:
            try:
                deadline = float(requested_deadline)
            except (TypeError, ValueError, OverflowError):
                return self._fallback("timeout", episode_id=episode, sequence=sequence, timestamp=stamp)
            if not math.isfinite(deadline):
                return self._fallback("timeout", episode_id=episode, sequence=sequence, timestamp=stamp)
        if episode is not None:
            if self._freshness.episode_id is None:
                self._freshness.reset_episode(episode)
            elif episode != self._freshness.episode_id:
                return self._fallback("stale_episode", episode_id=episode, sequence=sequence, timestamp=stamp)
            if sequence is not None:
                if not isinstance(sequence, (int, np.integer)) or int(sequence) < 0:
                    return self._fallback("stale_episode", episode_id=episode, sequence=sequence, timestamp=stamp)
                if self._freshness.last_sequence is not None and int(sequence) <= self._freshness.last_sequence:
                    return self._fallback("stale_episode", episode_id=episode, sequence=int(sequence), timestamp=stamp)
            if stamp is not None and self._freshness.last_timestamp is not None and stamp < self._freshness.last_timestamp:
                return self._fallback("stale_episode", episode_id=episode, sequence=sequence, timestamp=stamp)
            if sequence is not None:
                self._freshness.last_sequence = int(sequence)
            if stamp is not None:
                self._freshness.last_timestamp = stamp
        if timed_out or (deadline is not None and stamp is not None and stamp > deadline):
            return self._fallback("timeout", episode_id=episode, sequence=sequence, timestamp=stamp)
        if not sensor_available:
            return self._fallback("missing_sensor", episode_id=episode, sequence=sequence, timestamp=stamp)
        if not candidate_available or selected_plan is None:
            return self._fallback("no_candidate", episode_id=episode, sequence=sequence, timestamp=stamp)
        try:
            points, times = self._validate_plan(selected_plan)
            if times.ndim != 1 or len(times) != len(points) or len(times) == 0 or np.any(np.diff(times) <= 0):
                raise ValueError("plan timestamps must be increasing and match plan xy")
            state = dict(ego_state)
            speed = float(state.get("speed", state.get("velocity", 0.0)))
            if not math.isfinite(speed) or speed < 0:
                raise ValueError("ego speed must be finite and nonnegative")
            elapsed = float(dt)
            if not math.isfinite(elapsed) or elapsed < 0:
                raise ValueError("dt must be finite and nonnegative")
            target = interpolate_xy(points, times, elapsed)
            distance = float(np.linalg.norm(target))
            desired_speed = distance / max(elapsed, 0.05)
            speed_error = desired_speed - speed
            if speed_error >= 0:
                throttle = min(self.config.max_throttle, speed_error / self.config.max_acceleration)
                brake = 0.0
            else:
                throttle = 0.0
                brake = min(self.config.max_brake, -speed_error / self.config.max_deceleration)
            steer = float(np.clip(self.config.steering_gain * target[1] / max(abs(target[0]), 1.0), -self.config.max_steer, self.config.max_steer))
            return ControlResult(VehicleControl(steer, throttle, brake), None, episode, sequence, stamp, False)
        except (TypeError, ValueError, KeyError, OverflowError):
            return self._fallback("invalid_plan", episode_id=episode, sequence=sequence, timestamp=stamp)
