"""Shared history-input rules, independent of models and future targets.

Project-owned protocol glue; reference searches and reused encoding boundaries
are recorded in docs/provenance/dense_history_sources_20261116.json.
"""
from __future__ import annotations

import numpy as np


HISTORY_VISIBILITIES = ('dense_semantic', 'mask_lidar', 'mask_lidar_temporal_ground')
DEFAULT_HISTORY_VISIBILITY = 'mask_lidar'
HISTORY_CACHE_DIRECTORIES = {
    'mask_lidar': 'e0_full',
    'dense_semantic': 'e0_dense_history',
    'mask_lidar_temporal_ground': 'e0_mask_lidar_temporal_ground',
}
KNOWN_POLICY = 'in_bounds & semantics!=255'
SENSOR_POLICY = 'selected_sensor_mask & in_bounds & semantics!=255'
TEMPORAL_GROUND_POLICY = (
    'mask_lidar_known | causal_observed_ground11_14_min2_since_last_visible_conflict_v3;'
    'all_four_xy_neighbours_directly_observed_same_label;'
    'fill_with_past_observed_label_only;in_bounds'
)
GROUND_CLASSES = (11, 12, 13, 14)


def mode_from_visibility_key(key):
    """None selects dense known semantics; retain explicit sensor-mask APIs."""
    if key is None:
        return 'dense_semantic'
    if not isinstance(key, str) or not key or key == 'dense_semantic':
        raise ValueError('visibility_key must be None or a nonempty sensor mask name')
    return key


def visibility_key_from_mode(mode):
    if not isinstance(mode, str) or not mode:
        raise ValueError('history_visibility must name an input mode')
    if mode == 'mask_lidar_temporal_ground':
        return 'mask_lidar'
    return None if mode == 'dense_semantic' else mode


def visibility_metadata(key, *, history_visibility=None):
    mode = mode_from_visibility_key(key) if history_visibility is None else history_visibility
    if visibility_key_from_mode(mode) != key:
        raise ValueError('history_visibility disagrees with visibility key')
    policy = (TEMPORAL_GROUND_POLICY if mode == 'mask_lidar_temporal_ground'
              else KNOWN_POLICY if key is None else SENSOR_POLICY)
    return {'history_visibility': mode,
            'input_visibility_key': key,
            'input_visibility_policy': policy}


def resolve_history_visibility(values, *, key_field='input_visibility_key'):
    """Read new explicit modes or legacy mask keys, rejecting contradictions."""
    if 'history_visibility' in values:
        mode = values['history_visibility']
        key = visibility_key_from_mode(mode)
        if key_field in values and values[key_field] != key:
            raise ValueError('history_visibility disagrees with visibility key')
    elif key_field in values:
        mode = mode_from_visibility_key(values[key_field])
        key = values[key_field]
    else:
        raise ValueError('Missing history visibility mode/key')
    expected = visibility_metadata(key, history_visibility=mode)['input_visibility_policy']
    if 'input_visibility_policy' in values and values['input_visibility_policy'] != expected:
        raise ValueError('history_visibility disagrees with input visibility policy')
    return mode


def history_visibility_mask(semantics, in_bounds, sensor_mask=None):
    """Only known in-bounds voxels are valid, optionally restricted by a sensor."""
    semantics, in_bounds = np.asarray(semantics), np.asarray(in_bounds, dtype=bool)
    if semantics.ndim != 3 or semantics.dtype.kind not in 'ui' or semantics.shape != in_bounds.shape:
        raise ValueError('Semantic labels/in_bounds must have matching 3-D integer/bool grid shape')
    if not np.isin(semantics, [*range(18), 255]).all():
        raise ValueError('Semantic labels must contain Occ3D18 classes or unknown255')
    visible = in_bounds & (semantics != 255)
    if sensor_mask is not None:
        sensor_mask = np.asarray(sensor_mask, dtype=bool)
        if sensor_mask.shape != semantics.shape:
            raise ValueError('Observation visibility mask shape must match the semantic grid')
        visible &= sensor_mask
    return visible


def complete_masked_ground_history(semantics, in_bounds, sensor_mask):
    """Causally carry repeatedly observed ground into hidden E0 cells.

    Inputs are chronological, already-warped grids from at most four observed
    frames. Two direct observations of the same ground class are required since
    the last visible conflicting label. Visible free/object/other-ground labels
    reset that evidence. All four current-frame XY neighbours must directly
    observe the same class at the same height, avoiding stale vertical ground
    surfaces after nearest-voxel ego alignment. Filled cells never count as observations. A completion
    carries the observed label, never the current hidden ground-truth label.
    This project-owned protocol is documented in masked_ground_sources_20261118.
    """
    semantics = np.asarray(semantics)
    in_bounds = np.asarray(in_bounds, dtype=bool)
    sensor_mask = np.asarray(sensor_mask, dtype=bool)
    if (semantics.ndim != 4 or not 1 <= semantics.shape[0] <= 4
            or semantics.shape != in_bounds.shape or semantics.shape != sensor_mask.shape):
        raise ValueError('Ground completion requires one to four matching chronological 3-D grids')
    visible = np.stack([history_visibility_mask(s, b, m)
                        for s, b, m in zip(semantics, in_bounds, sensor_mask)])
    labels = np.where(visible, semantics, 17).astype(np.uint8)
    encoded_visible = visible.copy()
    memory_label = np.full(semantics.shape[1:], 17, dtype=np.uint8)
    observations = np.zeros(semantics.shape[1:], dtype=np.uint8)
    for slot in range(len(semantics)):
        seen = visible[slot]
        ground = seen & np.isin(labels[slot], GROUND_CLASSES)
        conflict = seen & ((labels[slot] != memory_label) | ~ground)
        observations[conflict] = 0
        memory_label[seen] = labels[slot][seen]
        observations[ground] += 1
        supported_x = np.zeros_like(seen)
        supported_y = np.zeros_like(seen)
        supported_x[1:-1] = (seen[:-2] & seen[2:]
                            & (labels[slot][:-2] == memory_label[1:-1])
                            & (labels[slot][2:] == memory_label[1:-1]))
        supported_y[:, 1:-1] = (seen[:, :-2] & seen[:, 2:]
                               & (labels[slot][:, :-2] == memory_label[:, 1:-1])
                               & (labels[slot][:, 2:] == memory_label[:, 1:-1]))
        fill = ~seen & in_bounds[slot] & (observations >= 2) & supported_x & supported_y
        labels[slot][fill] = memory_label[fill]
        encoded_visible[slot][fill] = True
    return labels, encoded_visible
