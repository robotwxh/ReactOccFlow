"""Simulator-independent, fail-closed bridge message contract.

The bridge is deliberately transport-agnostic.  A future simulator adapter may
send the JSON bytes over a process boundary, while tests use :class:`InMemoryTransport`.
Only current and past observations cross this boundary; future truth keys are rejected.
"""
from __future__ import annotations

from dataclasses import dataclass, field
import json
import math
import re
from queue import Empty, Queue
from typing import Any, Mapping, Protocol

import numpy as np

SCHEMA_VERSION = "reactoccflow.bridge.v1"
RESPONSE_SCHEMA_VERSION = "reactoccflow.control_response.v1"
DEFAULT_MAX_BYTES = 1_000_000
DEFAULT_MAX_ARRAY_ELEMENTS = 1_000_000
DEFAULT_MAX_ARRAY_RANK = 4


class BridgeError(ValueError):
    """Base class for malformed or unsafe bridge data."""


class FutureGroundTruthError(BridgeError):
    """Raised when future labels or ground-truth values cross the bridge."""


class FreshnessError(BridgeError):
    """Raised when an episode, sequence, or timestamp is stale."""


class MessageTooLargeError(BridgeError):
    """Raised when a message exceeds the configured byte/array bounds."""


_FORBIDDEN_KEY_WORDS = {
    "future", "futuregt", "futuregroundtruth", "futuretrajectory", "futureego", "futureactor",
    "futurelabel", "futuretarget", "groundtruth", "gt", "target", "targets", "label", "labels", "oracle",
}
_FORBIDDEN_KEY_TOKENS = _FORBIDDEN_KEY_WORDS - {"future"}
_FUTURE_TRUTH_TOKENS = {"gt", "ground", "truth", "groundtruth", "target", "targets", "label", "labels", "oracle"}
_FUTURE_NON_WORLD_TOKENS = {"trajectory", "ego", "actor"}
_NORMALIZED_FORBIDDEN_COMPOSITES = ("groundtruth", "targettrajectory")
_NORMALIZED_GROUND_TRUTH_PREFIXES = ("gt", "ground")
_NORMALIZED_FUTURE_FORBIDDEN_PARTS = (
    "groundtruth", "ground", "truth", "gt", "trajectory", "ego", "actor",
    "targets", "target", "labels", "label", "oracle",
)


def _key_tokens(key: str) -> tuple[str, ...]:
    # Split separators and common camel-case spellings so composite names are
    # checked recursively with the same policy as simple keys.
    text = re.sub(r"([a-z0-9])([A-Z])", r"\1_\2", str(key))
    return tuple(token for token in re.split(r"[^a-zA-Z0-9]+", text.lower()) if token)


def _key_is_forbidden(key: str) -> bool:
    tokens = _key_tokens(key)
    normalized = "".join(tokens)
    # Keep the one explicitly supported predicted-world name available while
    # treating concatenated composites as aggressively as separator-delimited
    # names.  For example, ``futuregttrajectory`` has no token boundaries but
    # still carries future ground-truth trajectory semantics.
    if normalized == "futureworld":
        return False
    # ``gt`` is a common ground-truth abbreviation and may be glued to an
    # arbitrary namespace or CamelCase suffix, defeating token boundaries.
    if "gt" in normalized:
        return True
    if (
        normalized in _FORBIDDEN_KEY_WORDS
        or any(token in _FORBIDDEN_KEY_TOKENS for token in tokens)
        or any(marker in normalized for marker in _NORMALIZED_FORBIDDEN_COMPOSITES)
    ):
        return True
    if any(normalized.startswith(prefix) and len(normalized) > len(prefix) for prefix in _NORMALIZED_GROUND_TRUTH_PREFIXES):
        return True
    if "ground" in tokens and "truth" in tokens:
        return True
    if "future" in tokens:
        # A predicted world may legitimately be named future_world. Any
        # future field naming truth, labels, or trajectory state is rejected.
        if any(token in _FUTURE_TRUTH_TOKENS | _FUTURE_NON_WORLD_TOKENS for token in tokens):
            return True
        if len(tokens) == 1:
            return True
    # ``_key_tokens`` cannot split arbitrary concatenations such as
    # ``futuregttrajectory``.  Detect forbidden future composites directly in
    # the normalized spelling, preserving names such as ``future_world``.
    future_start = normalized.find("future")
    if future_start >= 0:
        future_suffix = normalized[future_start + len("future"):]
        if not future_suffix or any(part in future_suffix for part in _NORMALIZED_FUTURE_FORBIDDEN_PARTS):
            return True
    return False


def _plain(value: Any, *, path: str = "$", max_array_elements: int = DEFAULT_MAX_ARRAY_ELEMENTS, max_array_rank: int = DEFAULT_MAX_ARRAY_RANK, depth: int = 0) -> Any:
    if depth > 32:
        raise BridgeError("JSON payload nesting exceeds limit")
    if isinstance(value, np.ndarray):
        if value.dtype.kind not in "biufc":
            raise BridgeError(f"{path} arrays must use numeric NumPy dtypes")
        if value.ndim > max_array_rank or value.size > max_array_elements:
            raise MessageTooLargeError(f"{path} array exceeds shape/element limit")
        if not np.isfinite(value).all():
            raise BridgeError(f"{path} must contain finite values")
        return [_plain(item, path=f"{path}[]", max_array_elements=max_array_elements, max_array_rank=max_array_rank, depth=depth + 1) for item in value.tolist()]
    if isinstance(value, np.generic):
        return _plain(value.item(), path=path, max_array_elements=max_array_elements, max_array_rank=max_array_rank, depth=depth + 1)
    if isinstance(value, Mapping):
        output = {}
        for key, item in value.items():
            if not isinstance(key, str):
                raise BridgeError(f"{path} mapping keys must be strings")
            if _key_is_forbidden(key):
                raise FutureGroundTruthError(f"future/ground-truth key is forbidden: {key}")
            output[key] = _plain(item, path=f"{path}.{key}", max_array_elements=max_array_elements, max_array_rank=max_array_rank, depth=depth + 1)
        return output
    if isinstance(value, (list, tuple)):
        if len(value) > max_array_elements:
            raise MessageTooLargeError(f"{path} sequence exceeds element limit")
        return [_plain(item, path=f"{path}[]", max_array_elements=max_array_elements, max_array_rank=max_array_rank, depth=depth + 1) for item in value]
    if isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise BridgeError(f"{path} must be finite")
        return value
    if value is None:
        return None
    if isinstance(value, (bytes, bytearray, memoryview)) or callable(value):
        raise BridgeError(f"{path} contains executable or non-JSON payload")
    raise BridgeError(f"{path} contains unsupported non-JSON value {type(value).__name__}")


def _mapping(value: Mapping[str, Any], *, name: str) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise BridgeError(f"{name} must be a mapping")
    return _plain(value)


def _finite_timestamp(value: Any, *, name: str = "timestamp") -> float:
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise BridgeError(f"{name} must be finite") from exc
    if not math.isfinite(result) or result < 0:
        raise BridgeError(f"{name} must be finite and nonnegative")
    return result


def _sequence(value: Any) -> int:
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
        raise BridgeError("sequence must be a nonnegative integer")
    result = int(value)
    if result < 0:
        raise BridgeError("sequence must be a nonnegative integer")
    return result


@dataclass(frozen=True)
class BridgeMessage:
    episode_id: str
    sequence: int
    timestamp: float
    sensor_bundle: Mapping[str, Any]
    ego_state: Mapping[str, Any]
    navigation_condition: Mapping[str, Any]
    candidate_outputs: Mapping[str, Any] = field(default_factory=dict)
    world_outputs: Mapping[str, Any] = field(default_factory=dict)
    schema_version: str = SCHEMA_VERSION

    def __post_init__(self) -> None:
        if self.schema_version != SCHEMA_VERSION:
            raise BridgeError(f"unsupported bridge schema version: {self.schema_version!r}")
        if not isinstance(self.episode_id, str) or not self.episode_id or len(self.episode_id) > 256:
            raise BridgeError("episode_id must be a nonempty bounded string")
        object.__setattr__(self, "sequence", _sequence(self.sequence))
        object.__setattr__(self, "timestamp", _finite_timestamp(self.timestamp))
        for name in ("sensor_bundle", "ego_state", "navigation_condition", "candidate_outputs", "world_outputs"):
            object.__setattr__(self, name, _mapping(getattr(self, name), name=name))

    @classmethod
    def from_mapping(cls, payload: Mapping[str, Any], *, max_array_elements: int = DEFAULT_MAX_ARRAY_ELEMENTS, max_array_rank: int = DEFAULT_MAX_ARRAY_RANK) -> "BridgeMessage":
        if not isinstance(payload, Mapping):
            raise BridgeError("bridge message must be a mapping")
        allowed = {"schema_version", "episode_id", "sequence", "timestamp", "sensor_bundle", "ego_state", "navigation_condition", "navigation", "candidate_outputs", "world_outputs"}
        unknown = set(payload) - allowed
        if unknown:
            unknown_key = sorted(unknown, key=str)[0]
            if _key_is_forbidden(str(unknown_key)):
                raise FutureGroundTruthError(f"future/ground-truth key is forbidden: {unknown_key}")
            raise BridgeError(f"unknown bridge message key: {unknown_key}")
        # Validate size/shape limits before dataclass normalization.
        normalized = _plain(payload, max_array_elements=max_array_elements, max_array_rank=max_array_rank)
        if "navigation_condition" not in normalized and "navigation" in normalized:
            normalized["navigation_condition"] = normalized.pop("navigation")
        required = {"episode_id", "sequence", "timestamp", "sensor_bundle", "ego_state", "navigation_condition"}
        missing = sorted(required - set(normalized))
        if missing:
            raise BridgeError(f"missing bridge message field(s): {', '.join(missing)}")
        return cls(**{key: normalized[key] for key in ("episode_id", "sequence", "timestamp", "sensor_bundle", "ego_state", "navigation_condition", "candidate_outputs", "world_outputs", "schema_version") if key in normalized})

    @classmethod
    def from_json_bytes(cls, payload: bytes | bytearray | str, *, max_bytes: int = DEFAULT_MAX_BYTES, max_array_elements: int = DEFAULT_MAX_ARRAY_ELEMENTS, max_array_rank: int = DEFAULT_MAX_ARRAY_RANK) -> "BridgeMessage":
        if isinstance(payload, str):
            raw = payload.encode("utf-8")
        elif isinstance(payload, (bytes, bytearray)):
            raw = bytes(payload)
        else:
            raise BridgeError("bridge payload must be JSON bytes or text")
        if len(raw) > max_bytes:
            raise MessageTooLargeError("bridge message exceeds max_bytes")
        try:
            decoded = json.loads(raw.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise BridgeError("bridge payload must be valid JSON") from exc
        return cls.from_mapping(decoded, max_array_elements=max_array_elements, max_array_rank=max_array_rank)

    def to_mapping(self) -> dict[str, Any]:
        return _plain({"schema_version": self.schema_version, "episode_id": self.episode_id, "sequence": self.sequence, "timestamp": self.timestamp, "sensor_bundle": self.sensor_bundle, "ego_state": self.ego_state, "navigation_condition": self.navigation_condition, "candidate_outputs": self.candidate_outputs, "world_outputs": self.world_outputs})

    def to_json_bytes(self, *, max_bytes: int = DEFAULT_MAX_BYTES) -> bytes:
        raw = json.dumps(self.to_mapping(), sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")
        if len(raw) > max_bytes:
            raise MessageTooLargeError("bridge message exceeds max_bytes")
        return raw


def _validate_control_mapping(control: Mapping[str, Any]) -> dict[str, float]:
    required = {"steer", "throttle", "brake"}
    missing = sorted(required - set(control))
    if missing:
        raise BridgeError(f"control missing actuator field(s): {', '.join(missing)}")
    values: dict[str, float] = {}
    for name in required:
        raw = control[name]
        if isinstance(raw, bool) or not isinstance(raw, (int, float)):
            raise BridgeError(f"control {name} must be finite numeric")
        try:
            value = float(raw)
        except (TypeError, ValueError, OverflowError) as exc:
            raise BridgeError(f"control {name} must be finite numeric") from exc
        if not math.isfinite(value):
            raise BridgeError(f"control {name} must be finite numeric")
        values[name] = value
    if not -1.0 <= values["steer"] <= 1.0:
        raise BridgeError("control steer must satisfy [-1, 1]")
    if not 0.0 <= values["throttle"] <= 1.0:
        raise BridgeError("control throttle must satisfy [0, 1]")
    if not 0.0 <= values["brake"] <= 1.0:
        raise BridgeError("control brake must satisfy [0, 1]")
    if values["throttle"] > 0.0 and values["brake"] > 0.0:
        raise BridgeError("control throttle and brake cannot both be positive")
    return values


@dataclass(frozen=True)
class ControlResponse:
    episode_id: str
    sequence: int
    timestamp: float
    requested_deadline: float
    model_fingerprint: str
    config_fingerprint: str
    control: Mapping[str, Any]
    status: str = "ok"
    failure_reason: str | None = None
    schema_version: str = RESPONSE_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if self.schema_version != RESPONSE_SCHEMA_VERSION:
            raise BridgeError(f"unsupported response schema version: {self.schema_version!r}")
        if not isinstance(self.episode_id, str) or not self.episode_id:
            raise BridgeError("response episode_id must be nonempty")
        object.__setattr__(self, "sequence", _sequence(self.sequence))
        object.__setattr__(self, "timestamp", _finite_timestamp(self.timestamp))
        object.__setattr__(self, "requested_deadline", _finite_timestamp(self.requested_deadline, name="requested_deadline"))
        for name in ("model_fingerprint", "config_fingerprint", "status"):
            if not isinstance(getattr(self, name), str) or not getattr(self, name):
                raise BridgeError(f"response {name} must be nonempty")
        control = _mapping(self.control, name="control")
        control = _validate_control_mapping(control)
        object.__setattr__(self, "control", control)
        if self.failure_reason is not None and not isinstance(self.failure_reason, str):
            raise BridgeError("failure_reason must be a string or null")

    @classmethod
    def from_mapping(cls, payload: Mapping[str, Any]) -> "ControlResponse":
        if not isinstance(payload, Mapping):
            raise BridgeError("response must be a mapping")
        allowed = {"schema_version", "episode_id", "sequence", "timestamp", "requested_deadline", "model_fingerprint", "config_fingerprint", "control", "status", "failure_reason"}
        unknown = set(payload) - allowed
        if unknown:
            raise BridgeError(f"unknown response key: {sorted(unknown, key=str)[0]}")
        normalized = _plain(payload)
        required = {"episode_id", "sequence", "timestamp", "requested_deadline", "model_fingerprint", "config_fingerprint", "control"}
        missing = sorted(required - set(normalized))
        if missing:
            raise BridgeError(f"missing response field(s): {', '.join(missing)}")
        return cls(**normalized)

    def to_mapping(self) -> dict[str, Any]:
        return _plain({"schema_version": self.schema_version, "episode_id": self.episode_id, "sequence": self.sequence, "timestamp": self.timestamp, "requested_deadline": self.requested_deadline, "model_fingerprint": self.model_fingerprint, "config_fingerprint": self.config_fingerprint, "control": self.control, "status": self.status, "failure_reason": self.failure_reason})

    def to_json_bytes(self, *, max_bytes: int = DEFAULT_MAX_BYTES) -> bytes:
        raw = json.dumps(self.to_mapping(), sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")
        if len(raw) > max_bytes:
            raise MessageTooLargeError("response exceeds max_bytes")
        return raw

    @classmethod
    def from_json_bytes(cls, payload: bytes | str, *, max_bytes: int = DEFAULT_MAX_BYTES) -> "ControlResponse":
        if isinstance(payload, str):
            payload = payload.encode("utf-8")
        if not isinstance(payload, (bytes, bytearray)):
            raise BridgeError("response payload must be JSON bytes or text")
        if len(payload) > max_bytes:
            raise MessageTooLargeError("response exceeds max_bytes")
        try:
            decoded = json.loads(bytes(payload).decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise BridgeError("response payload must be valid JSON") from exc
        return cls.from_mapping(decoded)


class BridgeTransport(Protocol):
    def send(self, message: BridgeMessage) -> None: ...
    def receive(self, timeout_s: float | None = None) -> BridgeMessage: ...


class InMemoryTransport:
    """Small typed fake transport for CPU tests; no sockets or simulator imports."""

    def __init__(self, *, max_bytes: int = DEFAULT_MAX_BYTES, max_queue: int = 64):
        self.max_bytes = max_bytes
        self._queue: Queue[bytes] = Queue(maxsize=max_queue)
        self._closed = False

    def send(self, message: BridgeMessage) -> None:
        if self._closed:
            raise BridgeError("transport is closed")
        if not isinstance(message, BridgeMessage):
            raise TypeError("transport accepts BridgeMessage only")
        self._queue.put(message.to_json_bytes(max_bytes=self.max_bytes))

    put = send

    def receive(self, timeout_s: float | None = None) -> BridgeMessage:
        if timeout_s is not None and (not math.isfinite(float(timeout_s)) or float(timeout_s) < 0):
            raise ValueError("timeout_s must be finite and nonnegative")
        try:
            payload = self._queue.get(timeout=timeout_s)
        except Empty as exc:
            raise TimeoutError("bridge receive timed out") from exc
        return BridgeMessage.from_json_bytes(payload, max_bytes=self.max_bytes)

    get = receive

    def close(self) -> None:
        self._closed = True


class FreshnessGate:
    def __init__(self) -> None:
        self.episode_id: str | None = None
        self.last_sequence: int | None = None
        self.last_timestamp: float | None = None

    def reset_episode(self, episode_id: str) -> None:
        if not isinstance(episode_id, str) or not episode_id:
            raise ValueError("episode_id must be nonempty")
        self.episode_id, self.last_sequence, self.last_timestamp = episode_id, None, None

    def accept(self, message: BridgeMessage) -> BridgeMessage:
        if self.episode_id is None:
            self.reset_episode(message.episode_id)
        elif message.episode_id != self.episode_id:
            raise FreshnessError(f"stale episode {message.episode_id!r}; active episode is {self.episode_id!r}")
        if self.last_sequence is not None and message.sequence <= self.last_sequence:
            raise FreshnessError(f"old sequence {message.sequence}; last accepted sequence is {self.last_sequence}")
        if self.last_timestamp is not None and message.timestamp < self.last_timestamp:
            raise FreshnessError(f"timestamp regression {message.timestamp}; last accepted timestamp is {self.last_timestamp}")
        self.last_sequence, self.last_timestamp = message.sequence, message.timestamp
        return message


class BridgeReceiver:
    def __init__(self, *, max_bytes: int = DEFAULT_MAX_BYTES, max_array_elements: int = DEFAULT_MAX_ARRAY_ELEMENTS, max_array_rank: int = DEFAULT_MAX_ARRAY_RANK):
        self.max_bytes = max_bytes
        self.max_array_elements = max_array_elements
        self.max_array_rank = max_array_rank
        self.freshness = FreshnessGate()

    def receive(self, payload: BridgeMessage | bytes | str) -> BridgeMessage:
        message = payload if isinstance(payload, BridgeMessage) else BridgeMessage.from_json_bytes(payload, max_bytes=self.max_bytes, max_array_elements=self.max_array_elements, max_array_rank=self.max_array_rank)
        return self.freshness.accept(message)

    def reset_episode(self, episode_id: str) -> None:
        self.freshness.reset_episode(episode_id)
