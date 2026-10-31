"""Deterministic geometry, multi-world risk, and probability calibration metrics."""
from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any, Iterable, Mapping

import numpy as np


def _finite(name, value, ndim=None):
    arr = np.asarray(value)
    if ndim is not None and arr.ndim != ndim:
        raise ValueError(f"{name} must have ndim={ndim}")
    if arr.size and (not np.issubdtype(arr.dtype, np.number) or not np.all(np.isfinite(arr))):
        raise ValueError(f"{name} must contain finite numeric values")
    return arr


def smoothed_event_frequency(events, M: int | None = None) -> float:
    arr = np.asarray(events, dtype=bool).reshape(-1)
    m = len(arr) if M is None else int(M)
    if m < 1 or len(arr) != m:
        raise ValueError("M must equal the number of world samples and be positive")
    return float((int(arr.sum()) + 0.5) / (m + 1.0))

risk_frequency = smoothed_event_frequency


def _grid(meta: Mapping[str, Any]):
    try:
        origin = _finite("grid.origin_xy", meta["origin_xy"], 1).astype(float)
        shape = np.asarray(meta["shape"], dtype=int)
        resolution = float(meta["resolution"])
        frame = str(meta.get("frame", meta.get("coordinate_frame")))
    except (KeyError, TypeError) as exc:
        raise ValueError("grid metadata requires origin_xy, resolution, shape, and frame='E0'") from exc
    if frame != "E0":
        raise ValueError("grid metadata must be expressed in E0")
    if origin.shape != (2,) or shape.shape != (2,) or np.any(shape <= 0) or not np.isfinite(resolution) or resolution <= 0:
        raise ValueError("invalid grid metadata")
    return origin, resolution, shape


def _corners(center, length, width, yaw):
    c = np.asarray(center, dtype=float)
    if c.shape != (2,):
        raise ValueError("center must be x,y")
    l, w, y = float(length), float(width), float(yaw)
    if not np.all(np.isfinite([l, w, y])) or l <= 0 or w <= 0:
        raise ValueError("footprint dimensions must be positive and finite")
    local = np.array([[l/2,w/2],[l/2,-w/2],[-l/2,-w/2],[-l/2,w/2]])
    rot = np.array([[np.cos(y), -np.sin(y)], [np.sin(y), np.cos(y)]])
    return local @ rot.T + c


def _poly_area(poly):
    if len(poly) < 3:
        return 0.0
    p = np.asarray(poly)
    return float(abs(np.dot(p[:, 0], np.roll(p[:, 1], -1)) - np.dot(p[:, 1], np.roll(p[:, 0], -1))) / 2)


def _clip(subject, edge_a, edge_b):
    if len(subject) == 0:
        return []
    out = []
    edge = edge_b - edge_a
    inside = lambda p: np.cross(edge, p - edge_a) >= -1e-10
    for a, b in zip(subject, subject[1:] + subject[:1]):
        ia, ib = inside(a), inside(b)
        if ia and ib:
            out.append(b)
        elif ia and not ib:
            d = b - a; den = np.cross(edge, d)
            if abs(den) > 1e-12: out.append(a + d * (-np.cross(edge, a - edge_a) / den))
        elif not ia and ib:
            d = b - a; den = np.cross(edge, d)
            if abs(den) > 1e-12: out.extend([a + d * (-np.cross(edge, a - edge_a) / den), b])
    return out


def rectangle_overlap_area(a, b) -> float:
    """Intersection area of two convex oriented rectangles."""
    poly = [np.asarray(x, dtype=float) for x in a]
    clipper = np.asarray(b, dtype=float)
    if clipper.shape != (4, 2) or np.asarray(a).shape != (4, 2):
        raise ValueError("rectangles must be corner arrays with shape (4,2)")
    # Ensure clipper is counter-clockwise for the inside test.
    if _poly_area(clipper) and np.cross(clipper[1]-clipper[0], clipper[2]-clipper[1]) < 0:
        clipper = clipper[::-1]
    for x, y in zip(clipper, np.roll(clipper, -1, axis=0)):
        poly = _clip(poly, x, y)
        if not poly: return 0.0
    return _poly_area(poly)


def _ego_box(row, ego_geometry):
    if isinstance(ego_geometry, Mapping):
        length = ego_geometry.get("length", ego_geometry.get("l"))
        width = ego_geometry.get("width", ego_geometry.get("w"))
        height = ego_geometry.get("height", ego_geometry.get("h", 1.0))
        z = ego_geometry.get("z", ego_geometry.get("center_z", 0.0))
        frame = ego_geometry.get("frame", "E0")
    else:
        length, width = ego_geometry[:2]
        height, z, frame = 1.0, 0.0, "E0"
    if frame != "E0":
        raise ValueError("ego geometry must be expressed in E0")
    row = np.asarray(row, dtype=float)
    if row.shape[-1] < 2:
        raise ValueError("ego rows require x,y coordinates")
    yaw = row[2] if row.shape[-1] >= 3 else 0.0
    # Candidate rows use x,y,yaw,z when a fourth coordinate is supplied.
    if row.shape[-1] >= 4:
        z = row[3]
    if not np.all(np.isfinite([length, width, height, z])) or float(height) <= 0:
        raise ValueError("ego dimensions and z must be finite; height must be positive")
    return _corners(row[:2], length, width, yaw), float(z), float(height)


def _ego_rect(row, ego_geometry):
    return _ego_box(row, ego_geometry)[0]


def _actor_box(actor, t):
    if isinstance(actor, Mapping):
        pos = np.asarray(actor.get("position", actor.get("center")), dtype=float)
        pos = pos[t] if pos.ndim > 1 else pos
        dims = actor.get("lwh", actor.get("dimensions", actor.get("size")))
        if dims is None:
            dims = [actor.get("length"), actor.get("width"), actor.get("height", 1.0)]
        dims = np.asarray(dims, dtype=float)
        if dims.shape != (3,):
            raise ValueError("actor dimensions must be L,W,H")
        yaw = actor.get("yaw", 0.0)
        yaw = np.asarray(yaw, dtype=float)
        yaw = yaw[t] if yaw.ndim else yaw
        z = float(pos[2]) if pos.shape[0] >= 3 else float(actor.get("z", actor.get("center_z", 0.0)))
        return _corners(pos[:2], dims[0], dims[1], yaw), z, float(dims[2])
    arr = np.asarray(actor, dtype=float)
    row = arr[t] if arr.ndim > 1 else arr
    if row.shape[-1] >= 7:
        # x,y,z,length,width,height,yaw
        return _corners(row[:2], row[3], row[4], row[6]), float(row[2]), float(row[5])
    if row.shape[-1] < 5:
        raise ValueError("actor rows require x,y,length,width,yaw or x,y,z,l,w,h,yaw")
    return _corners(row[:2], row[2], row[3], row[4]), 0.0, 1.0


def _actor_rect(actor, t):
    return _actor_box(actor, t)[0]


def _vertical_overlap(a_z, a_h, b_z, b_h):
    return min(a_z + a_h / 2.0, b_z + b_h / 2.0) - max(a_z - a_h / 2.0, b_z - b_h / 2.0) > 1e-12


def _occupancy_overlap(rect, occupancy_t, meta, class_mask=None, validity_mask=None) -> float:
    origin, res, shape = _grid(meta)
    occ = np.asarray(occupancy_t)
    if occ.shape[:2] != tuple(shape):
        raise ValueError("occupancy spatial shape does not match grid metadata")
    if class_mask is None:
        mask = occ.astype(bool) if occ.ndim == 2 else np.any(occ.astype(bool), axis=-1)
    else:
        cm = np.asarray(class_mask)
        if cm.ndim == 0:
            mask = (occ == cm)
        elif occ.ndim == 2:
            mask = np.isin(occ, cm)
        else:
            mask = np.any(np.isin(occ, cm), axis=-1)
    if mask.shape != tuple(shape):
        raise ValueError("occupancy class mask shape mismatch")
    if validity_mask is not None:
        vm = np.asarray(validity_mask, dtype=bool)
        if vm.shape != tuple(shape):
            raise ValueError("occupancy validity mask shape mismatch")
        mask = mask & vm
    # Compute true polygon intersection against each occupied cell. A center point
    # test misses partial intersections at cell boundaries.
    ys, xs = np.nonzero(mask)
    area = 0.0
    for y, x in zip(ys.tolist(), xs.tolist()):
        cell = np.array([[origin[0] + x * res, origin[1] + y * res],
                         [origin[0] + (x + 1) * res, origin[1] + y * res],
                         [origin[0] + (x + 1) * res, origin[1] + (y + 1) * res],
                         [origin[0] + x * res, origin[1] + (y + 1) * res]], dtype=float)
        area += rectangle_overlap_area(rect, cell)
    return float(area)

def evaluate_risk(candidate, worlds, ego_geometry, *, times_s, grid_metadata,
                  occupancy_class_mask=None, actors_key="actors", occupancy_key="occupancy",
                  collision_area_epsilon: float = 1e-9, observed_ego_geometry=None,
                  current_ego_geometry=None) -> dict[str, Any]:
    """Evaluate ego footprint collision events independently for each sampled world."""
    traj = _finite("candidate", candidate)
    if traj.ndim == 3:
        reports = [evaluate_risk(row, worlds, ego_geometry, times_s=times_s, grid_metadata=grid_metadata,
                                 occupancy_class_mask=occupancy_class_mask, actors_key=actors_key, occupancy_key=occupancy_key,
                                 collision_area_epsilon=collision_area_epsilon,
                                 observed_ego_geometry=observed_ego_geometry,
                                 current_ego_geometry=current_ego_geometry) for row in traj]
        return {"candidate_count": len(reports), "candidates": reports,
                "collision_event_rates": [r["collision_event_rate"] for r in reports], "M": reports[0]["M"] if reports else 0}
    ts = _finite("times_s", times_s, 1).astype(float)
    if traj.ndim != 2 or traj.shape[0] != len(ts) or traj.shape[1] < 2:
        raise ValueError("candidate must have shape (T,>=2)")
    if len(ts) < 1 or np.any(np.diff(ts) <= 0) or np.any(ts < 0) or ts[-1] > 3.0 + 1e-9:
        raise ValueError("times_s must be explicit, strictly increasing, non-negative, and within 3 seconds")
    origin, res, shape = _grid(grid_metadata)
    if observed_ego_geometry is not None and current_ego_geometry is not None:
        raise ValueError("provide only one of observed_ego_geometry and current_ego_geometry")
    aligned_ego = observed_ego_geometry if observed_ego_geometry is not None else current_ego_geometry
    if aligned_ego is not None:
        if not isinstance(aligned_ego, Mapping) or str(aligned_ego.get("frame", aligned_ego.get("coordinate_frame", "E0"))) != "E0":
            raise ValueError("observed/current ego geometry must be expressed in E0")
        _ego_box(np.zeros(3, dtype=float), aligned_ego)
    worlds_list = list(worlds) if not isinstance(worlds, Mapping) else [worlds]
    if not worlds_list: raise ValueError("at least one world sample is required")
    per_world = []
    for wi, world in enumerate(worlds_list):
        if not isinstance(world, Mapping):
            world = {occupancy_key: world}
        occupancy = world.get(occupancy_key)
        occupancy_validity = world.get("occupancy_validity_mask", world.get("occupancy_valid_mask"))
        actors = world.get(actors_key, []) or []
        actor_validity = world.get("actor_validity_mask", world.get("actor_valid_mask"))
        if occupancy is not None:
            occ = np.asarray(occupancy)
            if occ.shape[0] != len(ts):
                raise ValueError("occupancy must have leading measured-time axis")
            if occupancy_validity is not None:
                ov = np.asarray(occupancy_validity, dtype=bool)
                if ov.shape not in {(len(ts), *shape), tuple(shape)}:
                    raise ValueError("occupancy validity mask must be (T,H,W) or (H,W)")
        collision_times, overlap_areas = [], []
        oob_times = []
        for t, row in enumerate(traj):
            rect, ego_z, ego_h = _ego_box(row, ego_geometry)
            lo, hi = origin, origin + shape * res
            oob = bool(np.any(rect[:, 0] < lo[0]) or np.any(rect[:, 1] < lo[1]) or np.any(rect[:, 0] > hi[0]) or np.any(rect[:, 1] > hi[1]))
            if oob:
                oob_times.append(float(ts[t]))
            area = 0.0
            if occupancy is not None:
                ov_t = None if occupancy_validity is None else (np.asarray(occupancy_validity, dtype=bool)[t] if np.asarray(occupancy_validity).ndim == 3 else occupancy_validity)
                area = max(area, _occupancy_overlap(rect, occ[t], grid_metadata, occupancy_class_mask, ov_t))
            for ai, actor in enumerate(actors):
                valid = True
                if actor_validity is not None:
                    av = np.asarray(actor_validity, dtype=bool)
                    if av.ndim == 1:
                        if len(av) != len(actors):
                            raise ValueError("actor validity mask must have one entry per actor")
                        valid = bool(av[ai])
                    elif av.ndim == 2:
                        if av.shape[0] != len(actors) or av.shape[1] != len(ts):
                            raise ValueError("actor validity mask must be (A,T)")
                        valid = bool(av[ai, t])
                    else:
                        raise ValueError("actor validity mask must be (A,) or (A,T)")
                if not valid:
                    continue
                if isinstance(actor, Mapping):
                    own_valid = actor.get("valid_mask", actor.get("validity_mask"))
                    if own_valid is not None:
                        own = np.asarray(own_valid, dtype=bool)
                        valid = bool(own[t] if own.ndim else own)
                if valid:
                    actor_rect, actor_z, actor_h = _actor_box(actor, t)
                    if _vertical_overlap(ego_z, ego_h, actor_z, actor_h):
                        area = max(area, rectangle_overlap_area(rect, actor_rect))
            overlap_areas.append(float(area))
            if area > collision_area_epsilon:
                collision_times.append(float(ts[t]))
        source_status = world.get("status") if isinstance(world, Mapping) else None
        # Preserve any supplied non-OK event/validity status verbatim. Computed
        # statuses are used only when the producer explicitly marked the world OK.
        if source_status is not None and str(source_status) != "ok":
            status = str(source_status)
        else:
            status = "out_of_bounds" if oob_times else ("collision" if collision_times else "ok")
        per_world.append({"world_index": wi, "collision": bool(collision_times), "collision_times_s": collision_times,
                          "overlap_area": overlap_areas, "out_of_bounds": bool(oob_times), "out_of_bounds_times_s": oob_times,
                          "status": status})
    events = np.asarray([x["collision"] for x in per_world], dtype=bool)
    return {"times_s": ts.tolist(), "world_count": len(per_world), "per_world": per_world,
            "collision_count": int(events.sum()), "raw_collision_event_rate": float(events.mean()),
            "collision_event_rate": smoothed_event_frequency(events),
            "M": len(events), "no_world_repair": True}

risk_evaluation = evaluate_risk
risk_metrics = evaluate_risk
evaluate_risk_samples = evaluate_risk


# ---------- calibration ----------
def _pava(x, y, w):
    blocks = [[float(xi), float(yi), float(wi)] for xi, yi, wi in zip(x, y, w)]
    i = 0
    while i < len(blocks) - 1:
        if blocks[i][1] > blocks[i+1][1]:
            a, b = blocks[i], blocks[i+1]; wt = a[2] + b[2]
            blocks[i:i+2] = [[(a[0]*a[2]+b[0]*b[2])/wt, (a[1]*a[2]+b[1]*b[2])/wt, wt]]
            i = max(0, i-1)
        else: i += 1
    return np.asarray(blocks)

@dataclass
class Calibrator:
    method: str
    feature_definition: str
    knots: np.ndarray
    values: np.ndarray
    weights: np.ndarray | None = None
    coef: np.ndarray | None = None
    artifact: dict[str, Any] | None = None

    def predict(self, features):
        x = _finite("features", features)
        if self.method == "isotonic":
            p = np.interp(x.reshape(-1), self.knots, self.values, left=self.values[0], right=self.values[-1])
        else:
            z = self.coef[0] + (x.reshape(-1, 1) @ self.coef[1:].reshape(1, -1)).reshape(-1)
            p = 1 / (1 + np.exp(-np.clip(z, -60, 60)))
        return np.clip(p.reshape(x.shape[:-1] if x.ndim > 1 else x.shape), 0, 1).astype(float)

    def to_dict(self):
        return {"schema_version": 2, "method": self.method, "feature_definition": self.feature_definition,
                "knots": self.knots.tolist(), "values": self.values.tolist(),
                "weights": None if self.weights is None else self.weights.tolist(),
                "coef": None if self.coef is None else self.coef.tolist(), "artifact": self.artifact or {}}

    def save(self, path):
        # Saving is the freeze boundary: evaluation must consume this immutable
        # schema rather than a mutable in-memory fitter.
        self.artifact = dict(self.artifact or {})
        self.artifact["frozen"] = True
        self.artifact["schema_version"] = 2
        Path(path).write_text(json.dumps(self.to_dict(), indent=2, allow_nan=False) + "\n")


def load_calibrator(path, *, expected_feature_definition=None,
                    expected_split_fingerprints=None, require_frozen=True):
    payload = json.loads(Path(path).read_text())
    if payload.get("schema_version", 0) < 1:
        raise ValueError("unsupported calibration artifact schema")
    artifact = payload.get("artifact") or {}
    if require_frozen and artifact.get("frozen") is not True:
        raise ValueError("calibration artifact must be frozen before evaluation")
    if expected_feature_definition is not None and payload.get("feature_definition") != expected_feature_definition:
        raise ValueError("calibration feature definition mismatch")
    if expected_split_fingerprints is not None and artifact.get("split_fingerprints") != expected_split_fingerprints:
        raise ValueError("calibration split fingerprint mismatch")
    method = payload.get("method")
    if method not in {"isotonic", "logistic"}:
        raise ValueError("invalid calibration method")
    return Calibrator(method, payload["feature_definition"], np.asarray(payload["knots"], dtype=float),
                      np.asarray(payload["values"], dtype=float),
                      None if payload.get("weights") is None else np.asarray(payload["weights"], dtype=float),
                      None if payload.get("coef") is None else np.asarray(payload["coef"], dtype=float), artifact)


def _sha256_payload(payload):
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str).encode("utf-8")
    import hashlib
    return hashlib.sha256(encoded).hexdigest()


def fit_calibrator(features, labels, origin_groups, *, method="isotonic", feature_definition="risk_frequency",
                   calibration_origins=None, train_origins=None, eval_origins=None, K=None, M=None, NFE=None):
    x = _finite("features", features); y = _finite("labels", labels).astype(float).reshape(-1); groups = np.asarray(list(origin_groups), dtype=object)
    if x.ndim == 1: x = x[:, None]
    if x.ndim != 2 or x.shape[1] != 1:
        raise ValueError("calibrator currently accepts one scalar feature")
    if x.shape[0] != len(y) or len(groups) != len(y) or np.any((y < 0) | (y > 1)) or np.any(y != np.floor(y)):
        raise ValueError("features, labels, and origin_groups must align; labels must be binary")
    if calibration_origins is None: calibration_origins = set(groups.tolist())
    else: calibration_origins = set(calibration_origins)
    train_set = set(train_origins or [])
    eval_set = set(eval_origins or [])
    if train_set & eval_set:
        raise ValueError("train and eval origins overlap")
    if train_set and calibration_origins & train_set:
        raise ValueError("calibration and train origins overlap")
    if eval_set and calibration_origins & eval_set:
        raise ValueError("calibration and eval origins overlap")
    observed_origins = set(groups.tolist())
    if len(calibration_origins) == 0 or calibration_origins != observed_origins: raise ValueError("missing calibration labels or origin groups")
    if method not in {"isotonic", "logistic"}: raise ValueError("method must be isotonic or logistic")
    z = x[:, 0]
    if method == "isotonic":
        order = np.argsort(z, kind="mergesort"); xx, yy = z[order], y[order]
        block = _pava(xx, yy, np.ones_like(yy)); knots, vals = block[:, 0], np.clip(block[:, 1], 0, 1)
        cal = Calibrator(method, feature_definition, knots, vals, block[:, 2])
    else:
        X = np.column_stack([np.ones(len(z)), z]); beta = np.zeros(X.shape[1])
        for _ in range(100):
            p = 1 / (1 + np.exp(-np.clip(X @ beta, -60, 60))); grad = X.T @ (p-y); h = X.T @ (p[:, None] * (1-p)[:, None] * X) + 1e-6*np.eye(X.shape[1])
            step = np.linalg.solve(h, grad); beta -= step
            if np.linalg.norm(step) < 1e-9: break
        cal = Calibrator(method, feature_definition, np.array([z.min(), z.max()]), np.array([0., 1.]), coef=beta)
    records = [{"origin": str(g), "feature": float(v), "label": int(label)}
               for g, v, label in zip(groups.tolist(), z.tolist(), y.tolist())]
    cal.artifact = {"feature_definition": feature_definition,
                    "calibration_origins": sorted(map(str, calibration_origins)),
                    "split_fingerprints": {
                        "calibration": _sha256_payload({"origins": sorted(map(str, calibration_origins)),
                                                         "records": sorted(records, key=lambda r: (r["origin"], r["feature"], r["label"]))}),
                        "train": _sha256_payload({"origins": sorted(map(str, train_set))}),
                        "eval": _sha256_payload({"origins": sorted(map(str, eval_set))}),
                    },
                    "K": K, "M": M, "NFE": NFE, "sample_count": int(len(y)), "method": method,
                    "frozen": False, "schema_version": 2}
    return cal


def calibration_metrics(probabilities, labels, *, bins=10):
    p = np.clip(_finite("probabilities", probabilities).astype(float).reshape(-1), 0, 1); y = _finite("labels", labels).astype(float).reshape(-1)
    if len(p) != len(y) or len(p) == 0 or np.any((y < 0) | (y > 1)) or np.any(y != np.floor(y)):
        raise ValueError("probabilities and binary labels must align")
    brier = float(np.mean((p-y)**2)); nll = float(-np.mean(y*np.log(np.maximum(p, 1e-15)) + (1-y)*np.log(np.maximum(1-p, 1e-15))))
    edges = np.linspace(0, 1, int(bins)+1); rows=[]; ece=0.
    for i in range(len(edges)-1):
        mask = (p >= edges[i]) & ((p < edges[i+1]) if i < len(edges)-2 else (p <= edges[i+1])); n=int(mask.sum())
        row={"bin":i,"lower":float(edges[i]),"upper":float(edges[i+1]),"count":n,"mean_pred":float(np.mean(p[mask])) if n else None,"event_rate":float(np.mean(y[mask])) if n else None}
        rows.append(row)
        if n: ece += n/len(p)*abs(row["mean_pred"]-row["event_rate"])
    order=np.argsort(-p); yy=y[order]; risk_cov=[]; cumulative=0
    for k in range(1,len(y)+1): cumulative += yy[k-1]; risk_cov.append({"coverage":k/len(y),"risk":float(cumulative/k),"count":k})
    return {"brier":brier,"log_loss":nll,"nll":nll,"ece":float(ece),"reliability":rows,"risk_coverage":risk_cov,"sample_count":len(y),"event_count":int(y.sum()),"rare_event_support":int(y.sum())}


def choose_threshold(risks, collisions, completions, *, thresholds=(0.05, 0.10, 0.20),
                     brake_index=None, artifact_path=None):
    r = _finite("risks", risks).astype(float)
    c = np.asarray(collisions, dtype=bool)
    comp = np.asarray(completions, dtype=bool)
    if r.ndim != 2 or c.shape != r.shape or comp.shape != r.shape:
        raise ValueError("risks, collisions, completions must be (groups,candidates)")
    if len(thresholds) == 0 or any(not np.isfinite(t) or t <= 0 or t >= 1 for t in thresholds):
        raise ValueError("invalid thresholds")
    if brake_index is not None and (int(brake_index) < 0 or int(brake_index) >= r.shape[1]):
        raise ValueError("brake_index must identify a candidate")
    scores = []
    for t in thresholds:
        feasible = r <= t
        selected = []
        fallback_count = 0
        for g in range(r.shape[0]):
            idx = np.flatnonzero(feasible[g])
            if len(idx):
                selected.append(int(idx[np.argmin(r[g, idx])]))
            elif brake_index is not None:
                selected.append(int(brake_index)); fallback_count += 1
            else:
                selected.append(-1)
        valid = [i for i in selected if i >= 0]
        if len(valid) != len(selected):
            scores.append((np.inf, np.inf, float(t), selected, fallback_count))
        else:
            scores.append((float(np.mean([c[g, i] for g, i in enumerate(selected)])),
                           -float(np.mean([comp[g, i] for g, i in enumerate(selected)])),
                           float(t), selected, fallback_count))
    scores.sort(key=lambda x: (x[0], x[1], x[2]))
    if not np.isfinite(scores[0][0]):
        raise ValueError("at least one threshold leaves an infeasible group; provide brake_index")
    chosen = scores[0]
    threshold, selected = chosen[2], chosen[3]
    result = {"threshold": threshold, "selected_indices": [int(i) for i in selected],
              "no_feasible_candidate": bool(chosen[4] > 0),
              "brake_index": None if brake_index is None else int(brake_index),
              "tune_scores": [{"threshold": x[2], "collision_rate": None if not np.isfinite(x[0]) else x[0],
                               "completion_rate": None if not np.isfinite(x[1]) else -x[1],
                               "fallback_count": int(x[4])} for x in scores],
              "artifact_schema": 1, "frozen": False}
    if artifact_path is not None:
        save_threshold_artifact(result, artifact_path)
    return result


def save_threshold_artifact(result, path):
    payload = dict(result)
    payload["frozen"] = True
    payload["artifact_schema"] = 1
    Path(path).write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n")


def load_threshold_artifact(path):
    payload = json.loads(Path(path).read_text())
    if payload.get("frozen") is not True:
        raise ValueError("threshold artifact must be frozen before evaluation")
    if payload.get("artifact_schema") != 1:
        raise ValueError("unsupported threshold artifact schema")
    if any(int(i) < 0 for i in payload.get("selected_indices", [])):
        raise ValueError("threshold artifact contains an infeasible selection")
    return payload

threshold_helper = choose_threshold
select_risk_threshold = choose_threshold
calibrate_risk = fit_calibrator
evaluate_calibration = calibration_metrics
