"""Owned PlanFlow candidate and offline planner contract.

Inputs are observed ego state/current pose and an optional navigation route. Future
labels, expert targets, and ground-truth fields are rejected at the boundary.
"""
from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
import json
import math
from typing import Any, Iterable, Mapping, Optional, Sequence

import numpy as np


_COMPARISON_MODES = {"fixed-candidates", "no-world", "occupancy-only"}
_ALLOWED_OBSERVATION_KEYS = {"ego_state", "current_pose", "route_waypoints"}
_ALLOWED_EGO_KEYS = {"speed", "velocity", "acceleration", "heading_rate"}
_ALLOWED_POSE_KEYS = {"x", "y", "yaw", "heading"}


def _jsonable(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(k): _jsonable(v) for k, v in sorted(value.items(), key=lambda item: str(item[0]))}
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    return value


def _as_float(value: Any, default: float = 0.0) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return default
    return result


def _observed_scalar(value: Any, *, field: str) -> float:
    """Convert one allowlisted observed scalar while rejecting nested payloads."""
    if isinstance(value, (Mapping, list, tuple, np.ndarray)):
        raise ValueError(f"observed-only observation field {field!r} must be a scalar")
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"observed-only observation field {field!r} must be numeric") from exc
    if not np.isfinite(result):
        raise ValueError(f"observed-only observation field {field!r} must be finite")
    return result


def _validate_observation(observation: Mapping[str, Any]) -> tuple[dict[str, float], dict[str, float], np.ndarray, dict[str, Any]]:
    """Validate and normalize the complete observed-only input schema.

    The returned canonical mapping is built solely from allowlisted, normalized
    observed fields, so arbitrary input cannot influence candidate identities.
    """
    if not isinstance(observation, Mapping):
        raise TypeError("observation must be a mapping")

    unknown = [key for key in observation if key not in _ALLOWED_OBSERVATION_KEYS]
    if unknown:
        raise ValueError(f"observed-only observation rejects unknown observation key {unknown[0]!r}")

    ego_input = observation.get("ego_state", {})
    if not isinstance(ego_input, Mapping):
        raise TypeError("observed-only ego_state must be a mapping")
    unknown = [key for key in ego_input if key not in _ALLOWED_EGO_KEYS]
    if unknown:
        raise ValueError(f"observed-only observation rejects unknown observation key {unknown[0]!r}")
    ego = {str(key): _observed_scalar(value, field=f"ego_state.{key}") for key, value in ego_input.items()}

    pose_input = observation.get("current_pose", {})
    if not isinstance(pose_input, Mapping):
        raise TypeError("observed-only current_pose must be a mapping")
    unknown = [key for key in pose_input if key not in _ALLOWED_POSE_KEYS]
    if unknown:
        raise ValueError(f"observed-only observation rejects unknown observation key {unknown[0]!r}")
    pose = {str(key): _observed_scalar(value, field=f"current_pose.{key}") for key, value in pose_input.items()}

    route_input = observation.get("route_waypoints", [])
    route = np.asarray(route_input, dtype=np.float64)
    if route.size and (route.ndim != 2 or route.shape[1] != 2):
        raise ValueError("observed-only route_waypoints must have shape [N, 2]")
    if not np.all(np.isfinite(route)):
        raise ValueError("observed-only route_waypoints must be finite")

    canonical = {
        "ego_state": ego,
        "current_pose": pose,
        "route_waypoints": route.tolist(),
    }
    return ego, pose, route, canonical


@dataclass(frozen=True)
class PlanningConfig:
    """Configuration for deterministic local candidate generation and checks."""

    num_candidates: int = 8
    num_points: int = 6
    times: tuple[float, ...] = (0.5, 1.0, 1.5, 2.0, 2.5, 3.0)
    cruise_speed: float = 4.0
    max_deceleration: float = 6.0
    max_speed: float = 15.0
    max_acceleration: float = 6.0
    max_curvature: float = 0.35
    max_heading_rate: float = 1.0
    candidate_heading_rate: float = 0.18
    candidate_speed_spread: float = 0.25

    def __post_init__(self) -> None:
        if self.num_candidates < 0:
            raise ValueError("num_candidates must be non-negative")
        if self.num_points <= 0:
            raise ValueError("num_points must be positive")
        times = tuple(float(t) for t in self.times)
        if len(times) != self.num_points:
            raise ValueError("times must contain num_points measured future timestamps")
        if not np.all(np.isfinite(times)) or times[0] <= 0 or np.any(np.diff(times) <= 0):
            raise ValueError("times must be finite, positive, and strictly increasing")
        if self.max_deceleration <= 0:
            raise ValueError("max_deceleration must be positive")


@dataclass(frozen=True)
class KinematicCheck:
    valid: bool
    reasons: tuple[str, ...]
    speeds: np.ndarray
    accelerations: np.ndarray
    heading: np.ndarray
    heading_rate: np.ndarray
    curvature: np.ndarray
    max_speed: float
    max_acceleration: float
    max_curvature: float
    max_heading_rate: float


@dataclass(frozen=True)
class Candidate:
    xy: np.ndarray
    timestamps: np.ndarray
    candidate_id: str
    is_brake: bool
    valid: bool
    invalid_reasons: tuple[str, ...]
    heading: np.ndarray
    speeds: np.ndarray


@dataclass
class CandidatePlans:
    xy: np.ndarray
    timestamps: np.ndarray
    ids: tuple[str, ...]
    is_brake: np.ndarray
    valid: np.ndarray
    invalid_reasons: tuple[tuple[str, ...], ...]
    heading: np.ndarray
    speeds: np.ndarray
    latency_candidate_count: int
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.xy = np.asarray(self.xy, dtype=np.float64)
        self.timestamps = np.asarray(self.timestamps, dtype=np.float64)
        self.is_brake = np.asarray(self.is_brake, dtype=bool)
        self.valid = np.asarray(self.valid, dtype=bool)
        self.heading = np.asarray(self.heading, dtype=np.float64)
        self.speeds = np.asarray(self.speeds, dtype=np.float64)
        if self.xy.ndim != 3 or self.xy.shape[2] != 2 or self.xy.shape[0] <= 0 or self.xy.shape[1] <= 0:
            raise ValueError("candidate xy must have shape [N, L, 2] with positive N and L")
        n = self.xy.shape[0]
        if self.timestamps.shape != (self.xy.shape[1],):
            raise ValueError("timestamps must have shape [L]")
        if self.is_brake.shape != (n,) or self.valid.shape != (n,):
            raise ValueError("candidate masks must have shape [N]")
        if self.heading.shape != (n, self.xy.shape[1]) or self.speeds.shape != (n, self.xy.shape[1]):
            raise ValueError("heading and speeds must have shape [N, L]")
        if len(self.ids) != n or len(self.invalid_reasons) != n:
            raise ValueError("candidate metadata length mismatch")
        if self.latency_candidate_count != n:
            raise ValueError("latency_candidate_count must equal candidate count")
        if any(not isinstance(candidate_id, str) for candidate_id in self.ids):
            raise ValueError("candidate ids must be strings")
        if len(set(self.ids)) != n:
            raise ValueError("candidate ids must be unique")
        if int(np.count_nonzero(self.is_brake)) != 1:
            raise ValueError("CandidatePlans must contain exactly one brake candidate")
        if any(any(not isinstance(reason, str) for reason in reasons) for reasons in self.invalid_reasons):
            raise ValueError("invalid_reasons must contain strings")
        if not np.all(np.isfinite(self.xy)):
            raise ValueError("candidate xy must be finite")
        if not np.all(np.isfinite(self.timestamps)):
            raise ValueError("candidate timestamps must be finite")
        if self.timestamps[0] <= 0 or np.any(np.diff(self.timestamps) <= 0):
            raise ValueError("candidate timestamps must be positive and strictly increasing")
        if not np.all(np.isfinite(self.heading)):
            raise ValueError("candidate heading must be finite")
        if not np.all(np.isfinite(self.speeds)):
            raise ValueError("candidate speeds must be finite")

    @property
    def count(self) -> int:
        return int(self.xy.shape[0])

    @property
    def brake_index(self) -> int:
        indices = np.flatnonzero(self.is_brake)
        if len(indices) != 1:
            raise ValueError("CandidatePlans must contain exactly one brake candidate")
        return int(indices[0])

    @property
    def brake(self) -> Candidate:
        i = self.brake_index
        return Candidate(self.xy[i], self.timestamps, self.ids[i], True, bool(self.valid[i]), self.invalid_reasons[i], self.heading[i], self.speeds[i])

    def candidate(self, index: int) -> Candidate:
        i = int(index)
        return Candidate(self.xy[i], self.timestamps, self.ids[i], bool(self.is_brake[i]), bool(self.valid[i]), self.invalid_reasons[i], self.heading[i], self.speeds[i])


def check_kinematics(
    trajectory_xy: Sequence[Sequence[float]],
    timestamps: Optional[Sequence[float]],
    *,
    max_speed: float = math.inf,
    max_acceleration: float = math.inf,
    max_curvature: float = math.inf,
    max_heading_rate: float = math.inf,
    initial_heading: float = 0.0,
) -> KinematicCheck:
    """Check a path using its elapsed timestamps; no nominal-dt fallback exists."""
    if timestamps is None:
        raise ValueError("timestamps are required; nominal dt fallback is forbidden")
    xy = np.asarray(trajectory_xy, dtype=np.float64)
    times = np.asarray(timestamps, dtype=np.float64)
    if xy.ndim != 2 or xy.shape[1] != 2 or len(xy) == 0:
        raise ValueError("trajectory must have shape [L, 2]")
    if times.shape != (len(xy),):
        raise ValueError("timestamps must have shape [L]")
    if not np.all(np.isfinite(xy)) or not np.all(np.isfinite(times)):
        reasons = ("non_finite",)
        zeros = np.zeros(len(xy), dtype=np.float64)
        return KinematicCheck(False, reasons, zeros, zeros, zeros, zeros, zeros, math.inf, math.inf, math.inf, math.inf)
    if times[0] <= 0 or np.any(np.diff(times) <= 0):
        raise ValueError("timestamps must be positive and strictly increasing")
    dt = np.diff(np.concatenate(([0.0], times)))
    increments = np.diff(np.vstack((np.zeros((1, 2)), xy)), axis=0)
    speeds = np.linalg.norm(increments, axis=1) / dt
    # Acceleration is measured between observed future segments; the first
    # segment has no prior future measurement and is assigned zero.
    accelerations = np.concatenate(([0.0], np.diff(speeds) / dt[1:]))
    segment_heading = np.arctan2(increments[:, 1], increments[:, 0])
    segment_heading[0] = initial_heading if np.linalg.norm(increments[0]) < 1e-9 else segment_heading[0]
    heading = np.unwrap(segment_heading)
    heading_rate = np.diff(np.concatenate(([initial_heading], heading))) / dt
    curvature = np.divide(heading_rate, speeds, out=np.zeros_like(heading_rate), where=speeds > 1e-6)
    reasons: list[str] = []
    max_speed_seen = float(np.max(speeds))
    max_acc_seen = float(np.max(np.abs(accelerations)))
    max_curv_seen = float(np.max(np.abs(curvature)))
    max_rate_seen = float(np.max(np.abs(heading_rate)))
    if max_speed_seen > max_speed + 1e-9:
        reasons.append("speed_limit")
    if max_acc_seen > max_acceleration + 1e-9:
        reasons.append("acceleration_limit")
    if max_curv_seen > max_curvature + 1e-9:
        reasons.append("curvature_limit")
    if max_rate_seen > max_heading_rate + 1e-9:
        reasons.append("heading_rate_limit")
    return KinematicCheck(not reasons, tuple(reasons), speeds, accelerations, heading, heading_rate, curvature, max_speed_seen, max_acc_seen, max_curv_seen, max_rate_seen)


class CandidateGenerator:
    """Generate K deterministic local candidates and one explicit brake candidate."""

    def __init__(self, config: Optional[PlanningConfig] = None, *, seed: Optional[int] = 0):
        self.config = config or PlanningConfig()
        self.seed = seed

    def generate(self, observation: Mapping[str, Any], *, generator: Optional[np.random.Generator] = None) -> CandidatePlans:
        ego_state, pose, route, canonical_observation = _validate_observation(observation)
        speed = max(0.0, ego_state.get("speed", ego_state.get("velocity", 0.0)))
        current_yaw = pose.get("yaw", pose.get("heading", 0.0))
        times = np.asarray(self.config.times, dtype=np.float64)
        rng = generator if generator is not None else np.random.default_rng(self.seed)
        base_speed = speed if speed > 1e-6 else max(0.0, self.config.cruise_speed)
        trajectories: list[np.ndarray] = []
        headings: list[np.ndarray] = []
        speeds: list[np.ndarray] = []
        checks: list[KinematicCheck] = []
        raw = canonical_observation
        settings = _jsonable(self.config.__dict__ if hasattr(self.config, "__dict__") else {"config": str(self.config)})
        for index in range(self.config.num_candidates):
            # Local generator only: no global RNG state is touched.
            jitter = float(rng.uniform(-0.05, 0.05))
            spread = (index - (self.config.num_candidates - 1) / 2.0) if self.config.num_candidates else 0.0
            speed_scale = 1.0 + self.config.candidate_speed_spread * (spread / max(1.0, self.config.num_candidates)) + jitter
            candidate_speed = max(0.0, base_speed * speed_scale)
            heading_rate = self.config.candidate_heading_rate * (spread / max(1.0, self.config.num_candidates))
            dt = np.diff(np.concatenate(([0.0], times)))
            heading = current_yaw + heading_rate * times
            xy = np.cumsum(np.column_stack((candidate_speed * np.cos(heading), candidate_speed * np.sin(heading))) * dt[:, None], axis=0)
            if route.size:
                # Route contributes a navigation condition only: gently bias lateral offset.
                route_bias = float(np.clip(route[-1, 1] / max(abs(route[-1, 0]), 1.0), -0.5, 0.5))
                xy[:, 1] += route_bias * (times / times[-1]) * 0.25
            check = check_kinematics(xy, times, max_speed=self.config.max_speed, max_acceleration=self.config.max_acceleration, max_curvature=self.config.max_curvature, max_heading_rate=self.config.max_heading_rate, initial_heading=current_yaw)
            trajectories.append(xy)
            headings.append(check.heading)
            speeds.append(check.speeds)
            checks.append(check)
        # Explicit brake path. Its heading remains current heading after coming to rest.
        dt = np.diff(np.concatenate(([0.0], times)))
        decel = self.config.max_deceleration
        distances = speed * times - 0.5 * decel * times**2
        stop_distance = speed * speed / (2.0 * decel)
        stop_time = speed / decel if decel > 0 else 0.0
        distances = np.where(times <= stop_time, np.maximum(distances, 0.0), stop_distance)
        brake_xy = np.column_stack((distances * np.cos(current_yaw), distances * np.sin(current_yaw)))
        brake_check = check_kinematics(brake_xy, times, max_speed=self.config.max_speed, max_acceleration=self.config.max_acceleration, max_curvature=self.config.max_curvature, max_heading_rate=self.config.max_heading_rate, initial_heading=current_yaw)
        brake_reasons = list(brake_check.reasons)
        if stop_time > float(times[-1]) + 1e-9:
            brake_reasons.append("stop_not_reached")
        brake_check = KinematicCheck(
            not brake_reasons,
            tuple(brake_reasons),
            brake_check.speeds,
            brake_check.accelerations,
            brake_check.heading,
            brake_check.heading_rate,
            brake_check.curvature,
            brake_check.max_speed,
            brake_check.max_acceleration,
            brake_check.max_curvature,
            brake_check.max_heading_rate,
        )
        # At rest retain the observed heading and explicitly expose the physical speed profile.
        brake_heading = np.full(self.config.num_points, current_yaw, dtype=np.float64)
        brake_speeds = np.maximum(0.0, speed - decel * times)
        trajectories.append(brake_xy)
        headings.append(brake_heading)
        speeds.append(brake_speeds)
        checks.append(brake_check)
        ids = []
        for index in range(self.config.num_candidates + 1):
            identity = {"observation": raw, "settings": settings, "candidate_index": index}
            ids.append("plan-" + hashlib.sha256(json.dumps(identity, sort_keys=True, separators=(",", ":")).encode()).hexdigest()[:16])
        return CandidatePlans(np.stack(trajectories), times, tuple(ids), np.array([False] * self.config.num_candidates + [True]), np.array([c.valid for c in checks]), tuple(c.reasons for c in checks), np.stack(headings), np.stack(speeds), self.config.num_candidates + 1, {"coordinate_frame": "E0", "source": "deterministic_local"})


@dataclass(frozen=True)
class CostBreakdown:
    route_progress: np.ndarray
    speed_comfort: np.ndarray
    invalidity: np.ndarray
    world_risk: np.ndarray
    total: np.ndarray
    mode: str


def compute_costs(plans: CandidatePlans, observation: Optional[Mapping[str, Any]] = None, *, world_risk: Optional[float | Sequence[float]] = None, mode: str = "no-world") -> CostBreakdown:
    if mode not in _COMPARISON_MODES:
        raise ValueError(f"unknown comparison mode {mode!r}")
    route = np.asarray((observation or {}).get("route_waypoints", []), dtype=np.float64)
    final = plans.xy[:, -1]
    if route.size:
        route_progress = np.linalg.norm(final - route[-1], axis=1)
    else:
        route_progress = -final[:, 0]
    speed_comfort = np.mean(np.abs(np.diff(plans.speeds, axis=1)), axis=1) if plans.xy.shape[1] > 1 else np.zeros(plans.count)
    invalidity = (~plans.valid).astype(np.float64)
    if mode == "no-world" or world_risk is None:
        risk = np.zeros(plans.count, dtype=np.float64)
    else:
        risk = np.broadcast_to(np.asarray(world_risk, dtype=np.float64), (plans.count,)).copy()
        if not np.all(np.isfinite(risk)):
            raise ValueError("world_risk must be finite")
    total = route_progress + 0.1 * speed_comfort + 10.0 * invalidity + risk
    return CostBreakdown(route_progress, speed_comfort, invalidity, risk, total, mode)


@dataclass(frozen=True)
class SelectionResult:
    index: int
    candidate: Candidate
    no_feasible_candidate: bool
    reason: str
    risk: float
    cost: float


class Planner:
    """Offline candidate selector with explicit fallback and stable tie-breaking."""

    def __init__(self, *, risk_threshold: float = math.inf, comparison_mode: str = "no-world"):
        if comparison_mode not in _COMPARISON_MODES:
            raise ValueError(f"unknown comparison mode {comparison_mode!r}")
        self.risk_threshold = float(risk_threshold)
        self.comparison_mode = comparison_mode

    def select(self, plans: CandidatePlans, risks: Optional[Sequence[float]] = None, costs: Optional[Sequence[float] | CostBreakdown] = None) -> SelectionResult:
        if risks is None:
            risk_values = np.zeros(plans.count, dtype=np.float64)
        else:
            risk_values = np.broadcast_to(np.asarray(risks, dtype=np.float64), (plans.count,)).copy()
        if not np.all(np.isfinite(risk_values)):
            raise ValueError("risks must be finite")
        if isinstance(costs, CostBreakdown):
            cost_values = np.asarray(costs.total, dtype=np.float64)
        elif costs is None:
            cost_values = np.zeros(plans.count, dtype=np.float64)
        else:
            cost_values = np.broadcast_to(np.asarray(costs, dtype=np.float64), (plans.count,)).copy()
        if not np.all(np.isfinite(cost_values)):
            raise ValueError("costs must be finite")
        brake = plans.brake_index
        non_brake = np.flatnonzero(~plans.is_brake)
        feasible_non_brake = [int(i) for i in non_brake if plans.valid[i] and risk_values[i] <= self.risk_threshold]
        if not feasible_non_brake:
            return SelectionResult(brake, plans.candidate(brake), True, "no_feasible_candidate", float(risk_values[brake]), float(cost_values[brake]))
        feasible = [i for i in range(plans.count) if plans.valid[i] and risk_values[i] <= self.risk_threshold]
        winner = min(feasible, key=lambda i: (float(risk_values[i]), float(cost_values[i]), plans.ids[i]))
        return SelectionResult(winner, plans.candidate(winner), False, "selected", float(risk_values[winner]), float(cost_values[winner]))


class PlanFlow:
    """Typed PlanFlow boundary; learned sampling is intentionally unavailable untrained."""

    def __init__(self, config: Optional[PlanningConfig] = None, *, weights: Optional[str] = None, trained: bool = False, seed: Optional[int] = 0):
        self.config = config or PlanningConfig()
        self.weights = weights
        self.trained = bool(trained)
        self.generator = CandidateGenerator(self.config, seed=seed)

    def sample(self, observation: Mapping[str, Any], num_candidates: Optional[int] = None, generator: Optional[np.random.Generator] = None, *, use_learned: bool = False) -> CandidatePlans:
        if use_learned or self.weights is not None:
            if not self.trained:
                raise RuntimeError("PlanFlow learned head requested with untrained weights")
            raise NotImplementedError("trained PlanFlow sampling is outside this offline contract")
        if num_candidates is None or num_candidates == self.config.num_candidates:
            return self.generator.generate(observation, generator=generator)
        config = PlanningConfig(**{**self.config.__dict__, "num_candidates": int(num_candidates)})
        return CandidateGenerator(config, seed=self.generator.seed).generate(observation, generator=generator)
