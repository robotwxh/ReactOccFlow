"""Pure numeric response metrics with strict counterfactual pairing contracts.

The functions in this module consume already materialized predictions or logs. They
never infer causality from unrelated origins and never fabricate simulator results.
"""
from __future__ import annotations

from collections import Counter, defaultdict
import hashlib
import json
from typing import Any, Iterable, Mapping

import numpy as np


def _finite(name: str, value: Any, *, ndim: int | None = None) -> np.ndarray:
    arr = np.asarray(value)
    if ndim is not None and arr.ndim != ndim:
        raise ValueError(f"{name} must have ndim={ndim}, got {arr.ndim}")
    if arr.size and (not np.issubdtype(arr.dtype, np.number) or not np.all(np.isfinite(arr))):
        raise ValueError(f"{name} must contain finite numeric values")
    return arr


def _metadata(name: str, value: Iterable[Any], n: int) -> np.ndarray:
    arr = np.asarray(list(value), dtype=object)
    if arr.ndim != 1 or len(arr) != n:
        raise ValueError(f"{name} must have length {n}")
    return arr


def validate_observation_input(observation_times_s, *, future_label_times_s=None,
                               current_time_s: float | None = None) -> None:
    """Reject labels that occur after the observation cutoff."""
    times = _finite("observation_times_s", observation_times_s, ndim=1).astype(float)
    if len(times) == 0 or np.any(np.diff(times) < 0):
        raise ValueError("observation times must be non-empty and sorted")
    # The observation history, rather than a caller-supplied wall-clock value,
    # defines the information cutoff. current_time_s is accepted only as a
    # consistency check so it cannot bypass future-label rejection.
    cutoff = float(times[-1])
    if current_time_s is not None and not np.isfinite(float(current_time_s)):
        raise ValueError("current_time_s must be finite")
    if current_time_s is not None and float(current_time_s) + 1e-9 < cutoff:
        raise ValueError("current_time_s cannot precede the observation cutoff")
    if future_label_times_s is not None:
        labels = _finite("future_label_times_s", future_label_times_s, ndim=1).astype(float)
        if np.any(labels > cutoff + 1e-9):
            raise ValueError("future labels are not valid observation inputs")


def history_fingerprint(history: Any) -> str:
    """Return a deterministic fingerprint for an observed history."""
    if isinstance(history, str):
        return history
    arr = np.asarray(history)
    if arr.dtype.kind in "OUS":
        payload = json.dumps(history, sort_keys=True, default=str, separators=(",", ":")).encode()
    else:
        if not np.all(np.isfinite(arr)):
            raise ValueError("history must be finite")
        payload = arr.tobytes(order="C") + repr(arr.shape).encode() + str(arr.dtype).encode()
    return hashlib.sha256(payload).hexdigest()


def time_to_collision(relative_positions, relative_velocities, times_s, *, radius: float = 0.0,
                      collision_distance: float | None = None) -> np.ndarray:
    """Estimate first measured-time collision for each trajectory.

    Inputs are ``(..., T, D)`` positions/velocities and an explicit ``T`` time grid.
    The returned TTC is ``inf`` if no measured sample reaches the configured distance.
    No nominal ``dt`` is inferred.
    """
    pos = _finite("relative_positions", relative_positions)
    vel = _finite("relative_velocities", relative_velocities)
    ts = _finite("times_s", times_s, ndim=1).astype(float)
    if pos.shape != vel.shape or pos.ndim < 2 or pos.shape[-2] != len(ts):
        raise ValueError("relative position/velocity shapes must be (..., T, D) matching times")
    if len(ts) < 1 or np.any(np.diff(ts) <= 0) or np.any(ts < 0):
        raise ValueError("times_s must be strictly increasing and non-negative")
    threshold = float(radius if collision_distance is None else collision_distance)
    if not np.isfinite(threshold) or threshold < 0:
        raise ValueError("collision distance must be finite and non-negative")
    distances = np.linalg.norm(pos, axis=-1)
    hit = distances <= threshold
    first = np.argmax(hit, axis=-1)
    return np.where(np.any(hit, axis=-1), ts[first], np.inf)


def _speed(pred: np.ndarray, times: np.ndarray) -> np.ndarray:
    if pred.shape[-1] >= 4:
        return np.linalg.norm(pred[..., 2:4], axis=-1)
    if pred.shape[-1] >= 2:
        delta = np.diff(pred[..., :2], axis=-2, prepend=pred[:, :1, :2])
        dt = np.diff(times, prepend=times[:1])
        dt[0] = np.nan
        speed = np.linalg.norm(delta, axis=-1) / np.where(dt > 0, dt, np.nan)
        speed[..., 0] = 0.0
        return speed
    raise ValueError("predictions must contain at least x,y coordinates")


def response_metrics(predictions, times_s, *, origins, actions, seeds, histories=None,
                     statuses=None, observed_times_s=None, future_label_times_s=None,
                     collision_events=None, ttc_s=None) -> dict[str, Any]:
    """Summarize paired response distributions.

    Every comparison key is ``(origin, seed, history_fingerprint)``; action branches
    are compared only within that key. ``predictions`` has shape ``(N,T,D)``.
    """
    pred = _finite("predictions", predictions)
    ts = _finite("times_s", times_s, ndim=1).astype(float)
    if pred.ndim != 3 or pred.shape[1] != len(ts) or pred.shape[1] < 1:
        raise ValueError("predictions must have shape (N,T,D) matching times_s")
    if np.any(np.diff(ts) <= 0) or np.any(ts < 0):
        raise ValueError("times_s must be strictly increasing and non-negative")
    n = pred.shape[0]
    origin = _metadata("origins", origins, n)
    action = _metadata("actions", actions, n)
    seed = _metadata("seeds", seeds, n)
    if histories is None:
        hist = np.asarray(["<unspecified>"] * n, dtype=object)
    else:
        hist = np.asarray([history_fingerprint(h) for h in histories], dtype=object)
        if len(hist) != n:
            raise ValueError(f"histories must have length {n}")
    if observed_times_s is not None:
        validate_observation_input(observed_times_s, future_label_times_s=future_label_times_s)
    elif future_label_times_s is not None:
        raise ValueError("observed_times_s is required when future labels are supplied")
    stat = statuses if statuses is not None else ["ok"] * n
    status = _metadata("statuses", stat, n).astype(str)
    if collision_events is not None:
        collisions = np.asarray(collision_events, dtype=bool)
        if collisions.shape != (n, len(ts)) and collisions.shape != (n,):
            raise ValueError("collision_events must be (N,T) or (N,)")
        collision_any = np.any(collisions, axis=1) if collisions.ndim == 2 else collisions
    else:
        collision_any = np.zeros(n, dtype=bool)
    if ttc_s is not None:
        ttc = _finite("ttc_s", ttc_s).astype(float)
        if ttc.shape not in {(n,), (n, len(ts))}:
            raise ValueError("ttc_s must be (N,) or (N,T)")
        ttc = np.min(ttc, axis=1) if ttc.ndim == 2 else ttc
    else:
        ttc = np.full(n, np.inf)

    speed = _speed(pred, ts)
    speed_change = speed[:, -1] - speed[:, 0]
    lateral = pred[:, -1, 1] - pred[:, 0, 1]
    keys = list(zip(origin.tolist(), seed.tolist(), hist.tolist()))
    groups: dict[str, list[float]] = defaultdict(list)
    discrepancy: list[float] = []
    for key in sorted(set(keys), key=str):
        inds = [i for i, k in enumerate(keys) if k == key]
        acts = sorted(set(action[i] for i in inds), key=str)
        if len(acts) > 1:
            for ai, a in enumerate(acts):
                ia = [i for i in inds if action[i] == a]
                for b in acts[ai + 1:]:
                    ib = [i for i in inds if action[i] == b]
                    # Pair branch distributions deterministically without claiming one-to-one causality.
                    for i in ia:
                        for j in ib:
                            discrepancy.append(float(np.mean(np.linalg.norm(pred[i, :, :2] - pred[j, :, :2], axis=-1))))
        for a in acts:
            ia = [i for i in inds if action[i] == a]
            prefix = f"{key[0]}|{key[1]}|{key[2]}|{a}"
            groups[prefix].extend(speed_change[ia].tolist())
    valid = status == "ok"
    reports = {name: {"count": len(vals), "values": [float(v) for v in vals],
                      "mean": float(np.mean(vals)) if vals else None}
               for name, vals in sorted(groups.items())}
    return {
        "times_s": ts.tolist(), "sample_count": n,
        "speed_change": {"values": speed_change.tolist(), "mean": float(np.mean(speed_change[valid])) if np.any(valid) else None},
        "lateral_displacement": {"values": lateral.tolist(), "mean": float(np.mean(lateral[valid])) if np.any(valid) else None},
        "ttc_s": {"values": ttc.tolist(), "finite_count": int(np.isfinite(ttc).sum()),
                   "minimum": float(np.min(ttc[np.isfinite(ttc)])) if np.any(np.isfinite(ttc)) else None},
        "collision_event_rate": float(np.mean(collision_any)) if n else None,
        "response_discrepancy": {"values": discrepancy, "mean": float(np.mean(discrepancy)) if discrepancy else None, "count": len(discrepancy)},
        "by_origin_seed_history_action": reports,
        "status_counts": dict(Counter(status.tolist())),
        "valid_count": int(valid.sum()),
        "pairing_keys": int(len(set(keys))),
        "causal_claim": False,
    }

# Public aliases kept explicit for downstream evaluation scripts.
compute_response_metrics = response_metrics
response_metric_report = response_metrics


def pair_response_samples(reference, counterfactual, *, reference_origin, counterfactual_origin,
                           reference_seed, counterfactual_seed, reference_history,
                           counterfactual_history):
    """Validate a direct counterfactual pair before computing discrepancy."""
    if reference_origin != counterfactual_origin:
        raise ValueError("counterfactual samples must share the same origin")
    if reference_seed != counterfactual_seed:
        raise ValueError("counterfactual samples must share the same random seed")
    if history_fingerprint(reference_history) != history_fingerprint(counterfactual_history):
        raise ValueError("counterfactual samples must share the same history fingerprint")
    a, b = _finite("reference", reference), _finite("counterfactual", counterfactual)
    if a.shape != b.shape:
        raise ValueError("paired response samples must have matching shapes")
    if a.ndim < 2:
        raise ValueError("paired response samples require a time axis")
    return float(np.mean(np.linalg.norm(a[..., :2] - b[..., :2], axis=-1)))
