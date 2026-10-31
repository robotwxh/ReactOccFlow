"""Simulator-independent, fail-closed branch collection protocol.

The module contains no CARLA imports. A future adapter implements :class:`BranchBackend`.
Observations contain current/past public history only; returned labels are evaluation targets.
"""
from __future__ import annotations

import dataclasses
import hashlib
import json
import math
import os
import tempfile
from pathlib import Path
from typing import Any, Mapping, Protocol, Sequence

import numpy as np

SCHEMA_VERSION = 1
SCENARIO_FAMILIES = (
    "following_braking", "merge", "unprotected_turn", "pedestrian_crossing",
    "lane_change", "cut_in", "intersection_stop", "yield_to_pedestrian",
    "obstacle_avoidance", "roundabout_entry",
)
RESULT_ARRAY_KEYS = (
    "ego_trajectory", "actor_trajectories", "occupancy_labels", "occupancy_masks",
    "collision_events", "tracking_error", "timestamps",
)


def _plain(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.integer, np.floating, np.bool_)):
        return value.item()
    if dataclasses.is_dataclass(value):
        return _plain(dataclasses.asdict(value))
    if isinstance(value, Mapping):
        return {str(k): _plain(v) for k, v in sorted(value.items(), key=lambda x: str(x[0]))}
    if isinstance(value, (list, tuple)):
        return [_plain(v) for v in value]
    if isinstance(value, float) and not math.isfinite(value):
        raise ValueError("non-finite configuration value")
    return value


def _digest(value: Any) -> str:
    return hashlib.sha256(json.dumps(_plain(value), sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def deterministic_origin_split(origin_id: str) -> str:
    """Assign a complete origin group using a stable SHA256 bucket."""
    bucket = int(hashlib.sha256(origin_id.encode()).hexdigest()[:8], 16) % 100
    return "train" if bucket < 70 else "tune" if bucket < 80 else "calibration" if bucket < 90 else "evaluation"


@dataclasses.dataclass(frozen=True)
class ScenarioOrigin:
    origin_id: str
    family: str
    initial_index: int = 0
    action_id: str | None = None
    map_requirement: str = "unspecified"

    def __post_init__(self):
        if self.family not in SCENARIO_FAMILIES:
            raise ValueError(f"unknown scenario family: {self.family}")
        if not self.origin_id or self.initial_index < 0:
            raise ValueError("origin_id and nonnegative initial_index are required")


@dataclasses.dataclass(frozen=True)
class BranchCollectionConfig:
    output_root: Path | str
    origins: tuple[ScenarioOrigin, ...]
    action_ids: tuple[str, ...] = tuple(f"action_{i}" for i in range(8))
    npc_seeds: tuple[int, ...] = (0, 1, 2)
    settings: Mapping[str, Any] = dataclasses.field(default_factory=dict)
    observation_schema_version: str = "reactoccflow.observation.v1"
    evaluation_target_schema_version: str = "reactoccflow.evaluation_target.v1"
    coordinate_frame: str = "sim_ego_x_forward_y_left_z_up"
    semantic_taxonomy: str = "simulator_declared"

    def __post_init__(self):
        if not self.action_ids or len(set(self.action_ids)) != len(self.action_ids):
            raise ValueError("candidate action IDs must be unique")
        if not self.npc_seeds:
            raise ValueError("at least one NPC seed is required")
        if len({o.origin_id for o in self.origins}) != len(self.origins):
            raise ValueError("origin IDs must be unique")
        if any(not isinstance(s, int) for s in self.npc_seeds):
            raise ValueError("NPC seeds must be integers")
        Path(self.output_root).mkdir(parents=True, exist_ok=True)

    @property
    def binding(self) -> dict[str, Any]:
        return {"schema_version": SCHEMA_VERSION, "origins": self.origins,
                "action_ids": self.action_ids, "npc_seeds": self.npc_seeds,
                "settings": self.settings, "observation_schema_version": self.observation_schema_version,
                "evaluation_target_schema_version": self.evaluation_target_schema_version,
                "coordinate_frame": self.coordinate_frame, "semantic_taxonomy": self.semantic_taxonomy}

    @property
    def binding_hash(self) -> str:
        return _digest(self.binding)


@dataclasses.dataclass
class ResetState:
    initial_state_fingerprint: str
    common_history_fingerprint: str
    public_history: Mapping[str, Any]
    metadata: Mapping[str, Any]


@dataclasses.dataclass
class BranchResult:
    ego_trajectory: np.ndarray
    actor_trajectories: np.ndarray
    occupancy_labels: np.ndarray
    occupancy_masks: np.ndarray
    collision_events: np.ndarray
    tracking_error: np.ndarray
    timestamps: np.ndarray
    execution_status: str
    metadata: Mapping[str, Any] = dataclasses.field(default_factory=dict)


class BranchBackend(Protocol):
    def reset(self, scenario: ScenarioOrigin, npc_seed: int) -> ResetState: ...
    def execute(self, candidate_reference: Mapping[str, Any], *, responsive_npc: bool) -> BranchResult: ...


@dataclasses.dataclass(frozen=True)
class CollectionReport:
    expected_branches: int
    successful_branches: int
    failed_branches: int
    split_counts: Mapping[str, int]

    @property
    def complete(self) -> bool:
        """True only when every expected branch has a recorded status."""
        return self.successful_branches + self.failed_branches == self.expected_branches


class BranchCollector:
    def __init__(self, backend: BranchBackend, config: BranchCollectionConfig):
        self.backend, self.config = backend, config
        self.root = Path(config.output_root)

    def collect(self, *, resume: bool = False) -> CollectionReport:
        manifest = self.root / "_collection.json"
        if resume and manifest.exists():
            old = json.loads(manifest.read_text())
            if old.get("binding_hash") != self.config.binding_hash or old.get("schema_version") != SCHEMA_VERSION:
                raise ValueError("resume binding/settings mismatch")
        elif resume and any(self.root.glob("*.json")):
            raise ValueError("resume metadata missing")
        self._atomic_json(manifest, {"schema_version": SCHEMA_VERSION, "binding_hash": self.config.binding_hash,
                                     "binding": self.config.binding})
        expected = len(self.config.origins) * len(self.config.action_ids) * len(self.config.npc_seeds)
        successful = failed = 0
        split_counts = {s: 0 for s in ("train", "tune", "calibration", "evaluation")}
        candidate_references = self.config.settings.get("candidate_references", {})
        if not isinstance(candidate_references, Mapping):
            raise ValueError("candidate_references must be a mapping")

        for origin in self.config.origins:
            split = deterministic_origin_split(origin.origin_id)
            split_counts[split] += 1
            for seed in self.config.npc_seeds:
                pending = []
                existing = {}
                for action in self.config.action_ids:
                    scenario = dataclasses.replace(origin, action_id=action)
                    branch_id = _digest({"origin_id": origin.origin_id, "action_id": action, "npc_seed": seed})[:24]
                    status_path, npz_path = self.root / f"{branch_id}.json", self.root / f"{branch_id}.npz"
                    spec = (scenario, action, branch_id, status_path, npz_path)
                    if resume and status_path.exists():
                        status = json.loads(status_path.read_text())
                        self._validate_resume_status(status, origin, action, seed, split, branch_id, npz_path)
                        existing[action] = status
                        if status.get("status") == "success":
                            successful += 1
                        else:
                            failed += 1
                    else:
                        if resume and npz_path.exists():
                            raise ValueError("resume branch status missing")
                        pending.append(spec)
                if not pending:
                    continue

                # Reset every pending branch before execution.  The first action
                # in a group is the only permitted baseline source; an invalid
                # first reset invalidates all pending branches in that group.
                resets: dict[str, ResetState] = {}
                reset_errors: dict[str, str] = {}
                for scenario, action, _branch_id, _status_path, _npz_path in pending:
                    try:
                        reset = self.backend.reset(scenario, seed)
                        self._validate_reset(reset)
                        resets[action] = reset
                    except Exception as exc:
                        reset_errors[action] = f"{type(exc).__name__}: {exc}"

                baseline = None
                baseline_error = None
                first_action = self.config.action_ids[0]
                if first_action in existing:
                    prior = existing[first_action]
                    if prior.get("initial_state_fingerprint") and prior.get("common_history_fingerprint"):
                        baseline = (prior["initial_state_fingerprint"], prior["common_history_fingerprint"])
                    else:
                        baseline_error = "recorded first reset has no valid baseline"
                elif first_action in reset_errors:
                    baseline_error = f"first reset invalid: {reset_errors[first_action]}"
                elif first_action in resets:
                    first = resets[first_action]
                    baseline = (first.initial_state_fingerprint, first.common_history_fingerprint)
                else:
                    baseline_error = "first reset did not produce a baseline"

                for scenario, action, branch_id, status_path, npz_path in pending:
                    status = self._new_status(origin, action, seed, split, branch_id)
                    reset = resets.get(action)
                    if baseline_error is not None:
                        self._write_failure(status_path, npz_path, status, f"group baseline invalid: {baseline_error}")
                        failed += 1
                        continue
                    if action in reset_errors:
                        self._write_failure(status_path, npz_path, status, reset_errors[action])
                        failed += 1
                        continue
                    assert reset is not None
                    fingerprints = (reset.initial_state_fingerprint, reset.common_history_fingerprint)
                    status.update({"initial_state_fingerprint": fingerprints[0],
                                   "common_history_fingerprint": fingerprints[1]})
                    if fingerprints != baseline:
                        self._write_failure(status_path, npz_path, status,
                                             "initial state/common history fingerprint mismatch")
                        failed += 1
                        continue
                    if action not in candidate_references or candidate_references[action] is None:
                        self._write_failure(status_path, npz_path, status,
                                             f"candidate reference unavailable for action {action}")
                        failed += 1
                        continue
                    try:
                        result = self.backend.execute(
                            {"action_id": action, "reference": candidate_references[action]},
                            responsive_npc=True,
                        )
                        self._validate_result(result)
                        arrays = {name: np.asarray(getattr(result, name)) for name in RESULT_ARRAY_KEYS}
                        _atomic_npz(npz_path, arrays)
                        status.update({"status": "success" if result.execution_status == "success" else "failed",
                                       "reason": "" if result.execution_status == "success" else f"execution status: {result.execution_status}",
                                       "execution_status": result.execution_status,
                                       "npz_sha256": _file_sha(npz_path)})
                        self._atomic_json(status_path, status)
                        if result.execution_status == "success":
                            successful += 1
                        else:
                            failed += 1
                    except Exception as exc:  # retain every failure as a branch artifact
                        self._write_failure(status_path, npz_path, status,
                                             f"{type(exc).__name__}: {exc}")
                        failed += 1
        return CollectionReport(expected, successful, failed, split_counts)

    def _new_status(self, origin: ScenarioOrigin, action: str, seed: int, split: str, branch_id: str) -> dict[str, Any]:
        return {
            "schema_version": SCHEMA_VERSION,
            "settings_hash": self.config.binding_hash,
            "branch_id": branch_id,
            "origin_id": origin.origin_id,
            "family": origin.family,
            "action_id": action,
            "npc_seed": seed,
            "split": split,
            "status": "failed",
            "execution_status": "invalid",
            "reason": "",
            "initial_state_fingerprint": "",
            "common_history_fingerprint": "",
            "observation_schema_version": self.config.observation_schema_version,
            "evaluation_target_schema_version": self.config.evaluation_target_schema_version,
            "coordinate_frame": self.config.coordinate_frame,
            "semantic_taxonomy": self.config.semantic_taxonomy,
        }

    def _write_failure(self, status_path: Path, npz_path: Path, status: dict[str, Any], reason: str) -> None:
        status["status"] = "failed"
        status["execution_status"] = "invalid"
        status["reason"] = reason
        _atomic_npz(npz_path, {"empty": np.empty(0, dtype=np.float32)})
        status["npz_sha256"] = _file_sha(npz_path)
        self._atomic_json(status_path, status)

    def _validate_resume_status(self, status: Mapping[str, Any], origin: ScenarioOrigin, action: str,
                                seed: int, split: str, branch_id: str, npz_path: Path) -> None:
        expected = self._new_status(origin, action, seed, split, branch_id)
        for key in ("schema_version", "settings_hash", "branch_id", "origin_id", "family", "action_id",
                    "npc_seed", "split", "observation_schema_version", "evaluation_target_schema_version",
                    "coordinate_frame", "semantic_taxonomy"):
            if status.get(key) != expected[key]:
                raise ValueError(f"resume branch identity mismatch: {key}")
        if status.get("status") not in {"success", "failed"}:
            raise ValueError("invalid recorded branch status")
        if status.get("status") == "success" and status.get("execution_status") != "success":
            raise ValueError("successful branch has non-success execution status")
        if status.get("status") == "failed" and status.get("execution_status") not in {"timeout", "crash", "invalid"}:
            raise ValueError("failed branch has invalid execution status")
        if not npz_path.exists() or _file_sha(npz_path) != status.get("npz_sha256"):
            raise ValueError("corrupt branch payload")
        self._validate_npz_payload(npz_path, status["status"], status["execution_status"])

    @staticmethod
    def _validate_npz_payload(path: Path, status: str, execution_status: str) -> None:
        try:
            with np.load(path, allow_pickle=False) as payload:
                keys = set(payload.files)
                if status == "success":
                    if keys != set(RESULT_ARRAY_KEYS):
                        raise ValueError("NPZ schema keys mismatch")
                    arrays = {name: payload[name] for name in RESULT_ARRAY_KEYS}
                    _validate_result_arrays(arrays)
                elif execution_status == "invalid":
                    if keys != {"empty"} or payload["empty"].shape != (0,):
                        raise ValueError("invalid branch NPZ schema mismatch")
                elif execution_status in {"timeout", "crash"}:
                    if keys != set(RESULT_ARRAY_KEYS):
                        raise ValueError("failed result NPZ schema keys mismatch")
                    _validate_result_arrays({name: payload[name] for name in RESULT_ARRAY_KEYS})
                else:
                    raise ValueError("failed branch execution status is unsupported")
        except (OSError, ValueError, EOFError) as exc:
            raise ValueError(f"invalid branch NPZ payload: {exc}") from exc

    @staticmethod
    def _validate_reset(reset: ResetState) -> None:
        if not reset.initial_state_fingerprint or not reset.common_history_fingerprint:
            raise ValueError("missing reset fingerprints")
        if not isinstance(reset.metadata, Mapping):
            raise ValueError("reset metadata is required")
        _require_responsive_npc(reset.metadata, "reset")
        for key, value in reset.public_history.items():
            if any(token in str(key).lower() for token in ("future", "target", "label")):
                raise ValueError(f"future observation field rejected: {key}")
            _validate_numeric(value, f"public_history.{key}")
        if "timestamps" not in reset.public_history:
            raise ValueError("public history timestamps are required")
        times = np.asarray(reset.public_history["timestamps"], dtype=float)
        _validate_times(times, "public history timestamps")
        if not reset.metadata.get("coordinate_frame") or not reset.metadata.get("semantic_taxonomy"):
            raise ValueError("coordinate frame and semantic taxonomy are required")

    @staticmethod
    def _validate_result(result: BranchResult) -> None:
        if result.execution_status not in {"success", "timeout", "crash", "invalid"}:
            raise ValueError("invalid execution status")
        metadata = result.metadata if isinstance(result.metadata, Mapping) else {}
        _require_responsive_npc(metadata, "result")
        arrays = {name: getattr(result, name) for name in RESULT_ARRAY_KEYS}
        _validate_result_arrays(arrays)

    @staticmethod
    def _atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
        _atomic_bytes(path, (json.dumps(_plain(payload), sort_keys=True, indent=2) + "\n").encode())


def _require_responsive_npc(metadata: Mapping[str, Any], phase: str) -> None:
    modes = [metadata[key] for key in ("npc_control_mode", "npc_trajectory_mode") if key in metadata]
    if any(mode in ("frozen", "prerecorded") for mode in modes):
        raise ValueError("declared frozen/prerecorded NPC trajectory rejected")
    if not modes or any(mode != "responsive" for mode in modes):
        raise ValueError(f"responsive NPC declaration required in {phase}")


def _validate_numeric(value: Any, name: str) -> None:
    array = np.asarray(value)
    if array.dtype.kind not in "biuf":
        raise ValueError(f"{name} must be numeric")
    if np.issubdtype(array.dtype, np.inexact) and not np.isfinite(array).all():
        raise ValueError(f"non-finite values in {name}")


def _validate_result_arrays(arrays: Mapping[str, Any]) -> None:
    missing = set(RESULT_ARRAY_KEYS) - set(arrays)
    if missing:
        raise ValueError(f"result arrays missing: {sorted(missing)}")
    converted = {}
    for name in RESULT_ARRAY_KEYS:
        _validate_numeric(arrays[name], name)
        converted[name] = np.asarray(arrays[name])
    timestamps = converted["timestamps"]
    _validate_times(timestamps, "result timestamps")
    steps = timestamps.shape[0]
    ego = converted["ego_trajectory"]
    actors = converted["actor_trajectories"]
    if ego.ndim != 2 or ego.shape[0] != steps or ego.shape[1] < 1:
        raise ValueError("ego_trajectory must have shape (T, D) matching timestamps")
    if actors.ndim != 3 or actors.shape[1] != steps or actors.shape[2] != ego.shape[1]:
        raise ValueError("actor_trajectories must have shape (A, T, D) matching ego trajectory")
    labels = converted["occupancy_labels"]
    masks = converted["occupancy_masks"]
    if labels.ndim < 1 or labels.shape[0] != steps or masks.shape != labels.shape:
        raise ValueError("occupancy labels and masks must share a T-leading shape")
    for name in ("collision_events", "tracking_error"):
        value = converted[name]
        if value.ndim != 1 or value.shape[0] != steps:
            raise ValueError(f"{name} must have length T matching timestamps")

def _validate_times(times: np.ndarray, name: str) -> None:
    if times.ndim != 1 or times.size == 0 or not np.isfinite(times).all() or np.any(np.diff(times) <= 0):
        raise ValueError(f"invalid timestamps in {name}")

def _file_sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _atomic_bytes(path: Path, data: bytes) -> None:
    fd, temp = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(fd, "wb") as f: f.write(data); f.flush(); os.fsync(f.fileno())
        os.replace(temp, path)
    finally:
        Path(temp).unlink(missing_ok=True)


def _atomic_npz(path: Path, arrays: Mapping[str, np.ndarray]) -> None:
    fd, temp = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".npz", dir=path.parent)
    try:
        with os.fdopen(fd, "wb") as f:
            np.savez(f, **arrays); f.flush(); os.fsync(f.fileno())
        os.replace(temp, path)
    finally:
        Path(temp).unlink(missing_ok=True)


def catalog_counts(*, pilot: bool) -> dict[str, int]:
    origins_per_family = 10 if pilot else 100
    branches = len(SCENARIO_FAMILIES) * origins_per_family * 8 * 3
    return {"origins": len(SCENARIO_FAMILIES) * origins_per_family, "actions": 8, "seeds": 3, "branches": branches}
