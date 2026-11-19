"""Encode already-warped 3-D E0 occupancy with explicit information boundaries.

Adapted after copying OccFM tools/cache_vae.py's no-grad/CPU NumPy cache
boundary. Source commit, checksums, dependency decisions and the untouched
source snapshot are recorded in docs/provenance/e0_cache_sources.json.
The upstream OccFM license remains unresolved; see THIRD_PARTY_NOTICES.md.

The pretrained codec has no unknown/mask channel. Hidden/unknown voxels are
therefore masked to empty17 inside its embedding, an encoding approximation,
never evidence that those voxels are free. Future semantic-known masks remain
separate target fields for consistency losses and evaluation.
"""
from __future__ import annotations

from contextlib import contextmanager
from copy import deepcopy

import numpy as np
import torch

from reactoccflow.geometry import GridSpec
from .history_visibility import (
    DEFAULT_HISTORY_VISIBILITY, complete_masked_ground_history, history_visibility_mask, mode_from_visibility_key,
    visibility_key_from_mode, visibility_metadata,
)


_OBSERVED_HISTORY = 4
_LATENT_SHAPE = (16, 50, 50)
_VERSION = 1
_DEFAULT_VISIBILITY = object()


def _validate_options(history_slots, hist_last, source_frame, visibility_key):
    if source_frame not in {'ego', 'lidar'}:
        raise ValueError('source_frame must be explicitly ego or lidar')
    if not isinstance(history_slots, int) or isinstance(history_slots, bool) or history_slots < _OBSERVED_HISTORY:
        raise ValueError('history_slots must be an integer of at least four')
    if not isinstance(hist_last, int) or isinstance(hist_last, bool) or not 1 <= hist_last <= _OBSERVED_HISTORY:
        raise ValueError('hist_last must be an integer between one and four')
    mode_from_visibility_key(visibility_key)


@contextmanager
def _inference_codec(codec):
    """Keep stochastic training layers disabled without changing caller state."""
    modes = [(module, module.training) for module in codec.modules()]
    codec.eval()
    try:
        with torch.inference_mode():
            yield
    finally:
        # Restore mixed train/eval submodule states as well as the root state.
        for module, training in modes:
            module.training = training


def _encode_arrays(semantics, visibility, codec, *, device):
    """Shared posterior-mean boundary for original and completed history grids."""
    labels = torch.tensor(semantics[None], dtype=torch.long, device=device)
    visible = torch.tensor(visibility[None], dtype=torch.bool, device=device)
    latent = codec.encode(labels, visible, sample=False)
    if not isinstance(latent, torch.Tensor) or tuple(latent.shape) != (1, *_LATENT_SHAPE):
        raise ValueError('Codec must return one posterior-mean latent with shape [1,16,50,50]')
    latent = latent[0].detach().to(device='cpu', dtype=torch.float32).numpy().copy()
    if not np.isfinite(latent).all():
        raise ValueError('Codec produced a nonfinite posterior-mean latent')
    return latent


def _encode_grid(index, source_token, current_token, codec, *, device, source_frame,
                 visibility_key=None, return_semantics=False):
    """Load the raw 3-D warp before passing any semantic grid to the codec."""
    grid = index.load_occupancy_e0(source_token, current_token, source_frame=source_frame,
                                   visibility_keys=() if visibility_key is None else (visibility_key,))
    semantics = np.asarray(grid['semantics'])
    in_bounds = np.asarray(grid['in_bounds'], dtype=bool)
    known = history_visibility_mask(semantics, in_bounds)
    visibility = known.copy()
    if visibility_key is not None:
        masks = grid.get('visibility', {})
        if not isinstance(masks, dict) or visibility_key not in masks:
            raise ValueError(f'Missing requested observation visibility mask: {visibility_key}')
        visibility = history_visibility_mask(semantics, in_bounds, masks[visibility_key])
    latent = _encode_arrays(semantics, visibility, codec, device=device)
    if return_semantics:
        return latent, known, np.array(semantics, dtype=np.uint8, copy=True)
    return latent, known


def encode_e0_observation(index, scene, current_token, codec, device='cpu', *,
                          history_slots=6, hist_last=4, source_frame='ego',
                          visibility_key=_DEFAULT_VISIBILITY, history_visibility=None):
    """Encode only current/past oracle observations; never inspect future data.

    Actor/ego histories retain four observed slots. Latent history and its mask,
    times and metadata tokens are right-aligned into ``history_slots`` slots.
    ``hist_last`` selects up to four recent slots, without filling missing ones.
    Every present semantic grid is warped in 3-D into the current ego frame by
    the independent index before deterministic posterior-mean encoding.
    The caller places its codec on ``device``; output values are NumPy arrays.
    By default use the first-version LiDAR mask without ground completion.
    An explicit visibility_key=None selects dense known semantics. Explicit
    history modes retain their recorded meaning for historical reproduction.
    """
    if visibility_key is _DEFAULT_VISIBILITY:
        visibility_key = visibility_key_from_mode(
            DEFAULT_HISTORY_VISIBILITY if history_visibility is None else history_visibility)
    mode = mode_from_visibility_key(visibility_key) if history_visibility is None else history_visibility
    if history_visibility is not None:
        requested_key = visibility_key_from_mode(mode)
        if visibility_key is not None and visibility_key != requested_key:
            raise ValueError('history_visibility disagrees with visibility key')
        visibility_key = requested_key
    _validate_options(history_slots, hist_last, source_frame, visibility_key)
    observed, original_metadata = index.build_observation(
        scene, current_token, history=_OBSERVED_HISTORY, max_actors=8)
    observation = {key: np.array(value, copy=True) for key, value in observed.items()}
    metadata = deepcopy(original_metadata)
    tokens = list(metadata['history_sample_tokens'])
    original_mask = np.asarray(observation['history_mask'], dtype=bool)
    times = np.asarray(observation['history_time_s'])
    if len(tokens) != _OBSERVED_HISTORY or original_mask.shape != (_OBSERVED_HISTORY,) or times.shape != (_OBSERVED_HISTORY,):
        raise ValueError('Observation history tokens, mask and times must have four aligned slots')
    if not np.array_equal(original_mask, np.asarray([bool(token) for token in tokens])) or tokens[-1] != current_token:
        raise ValueError('Observation history token validity must match its mask and current sample')
    latents = np.zeros((history_slots, *_LATENT_SHAPE), np.float32)
    history_mask = np.zeros(history_slots, bool)
    history_time = np.zeros(history_slots, np.float64)
    latent_tokens = [''] * history_slots
    slots = [slot for slot in range(_OBSERVED_HISTORY - hist_last, _OBSERVED_HISTORY)
             if original_mask[slot]]
    with _inference_codec(codec):
        completed = None
        if mode == 'mask_lidar_temporal_ground':
            grids = [index.load_occupancy_e0(tokens[slot], current_token, source_frame=source_frame,
                                            visibility_keys=('mask_lidar',)) for slot in slots]
            if any('mask_lidar' not in grid.get('visibility', {}) for grid in grids):
                raise ValueError('Missing requested observation visibility mask: mask_lidar')
            completed = complete_masked_ground_history(
                np.stack([grid['semantics'] for grid in grids]),
                np.stack([grid['in_bounds'] for grid in grids]),
                np.stack([grid['visibility']['mask_lidar'] for grid in grids]))
        for position, observed_slot in enumerate(slots):
            latent_slot = history_slots - _OBSERVED_HISTORY + observed_slot
            token = tokens[observed_slot]
            if completed is None:
                latents[latent_slot], _ = _encode_grid(
                    index, token, current_token, codec, device=device, source_frame=source_frame,
                    visibility_key=visibility_key)
            else:
                latents[latent_slot] = _encode_arrays(completed[0][position], completed[1][position],
                                                     codec, device=device)
            history_mask[latent_slot] = True
            history_time[latent_slot] = times[observed_slot]
            latent_tokens[latent_slot] = token
    observation.update({'history_latents': latents, 'history_mask': history_mask,
                        'history_time_s': history_time,
                        'observed_history_time_s': np.array(times, copy=True)})
    metadata.update({
        'version': _VERSION, 'format': 'reactoccflow_e0_encoded_sample',
        'oracle': True, 'coordinate_frame': 'E0', 'source_frame': source_frame,
        'grid': GridSpec().as_dict(), 'posterior': 'mean', 'latent_scaling': 'unscaled',
        'history_slots': history_slots, 'hist_last': hist_last,
        'history_latent_sample_tokens': latent_tokens,
        **visibility_metadata(visibility_key, history_visibility=mode),
        'unknown_policy': 'hidden_or_unknown_masked_to_empty17_for_pretrained_embedding_only;not_valid_free_evidence',
        'spatial_alignment': 'raw_3d_nearest_warp_before_encoding;no_latent_2d_warp',
        'provenance': 'docs/provenance/e0_cache_sources.json',
    })
    return {'observation': observation, 'metadata': metadata}


def encode_e0_training_sample(index, scene, current_token, codec, device='cpu', *,
                              history_slots=6, hist_last=4, future=6,
                              source_frame='ego', visibility_key=_DEFAULT_VISIBILITY,
                              include_future_semantics=False, history_visibility=None):
    """Add separate future oracle targets to an observation-only encoding.

    Future labels use complete in-bounds known semantics, independent of future
    LiDAR/camera visibility. Unknown255 and warped-out cells remain false in
    ``future_known_mask``. Only truly absent scene-tail samples receive zero
    latents and a false ``future_latent_mask``; missing files propagate errors.
    Future data never enters ``observation`` or the observation-only API.
    Set ``include_future_semantics=True`` to retain the same warped semantic
    uint8 grids under target; absent scene-tail grids contain unknown255.
    """
    if not isinstance(include_future_semantics, bool):
        raise ValueError('include_future_semantics must be boolean')
    if not isinstance(future, int) or isinstance(future, bool) or future < 1:
        raise ValueError('future must be a positive integer')
    result = encode_e0_observation(
        index, scene, current_token, codec, device=device, history_slots=history_slots,
        hist_last=hist_last, source_frame=source_frame, visibility_key=visibility_key,
        history_visibility=history_visibility)
    sample = index.build_sample(scene, current_token, history=_OBSERVED_HISTORY,
                                future=future, max_actors=8)
    target = {key: np.array(value, copy=True) for key, value in sample['target'].items()}
    tokens = list(sample['metadata']['future_sample_tokens'])
    present = np.asarray([bool(token) for token in tokens])
    if len(tokens) != future or not np.array_equal(present, target['ego_target_mask']):
        raise ValueError('Future sample tokens must match future length and ego target validity')
    if np.any(np.diff(present.astype(np.int8)) > 0):
        raise ValueError('Future samples must form a contiguous prefix before the scene tail')
    latents = np.zeros((future, *_LATENT_SHAPE), np.float32)
    known_masks = None
    future_semantics = None
    with _inference_codec(codec):
        for step, token in enumerate(tokens):
            if not token:
                continue
            encoded = _encode_grid(
                index, token, current_token, codec, device=device, source_frame=source_frame,
                return_semantics=include_future_semantics)
            latents[step], known = encoded[:2]
            if known_masks is None:
                known_masks = np.zeros((future, *known.shape), bool)
            elif known.shape != known_masks.shape[1:]:
                raise ValueError('Future semantic grids must have a consistent shape')
            known_masks[step] = known
            if include_future_semantics:
                if future_semantics is None:
                    future_semantics = np.full(known_masks.shape, 255, np.uint8)
                future_semantics[step] = encoded[2]
    if known_masks is None:
        known_masks = np.zeros((future, *GridSpec().shape), bool)
    target.update({'future_latents': latents, 'future_latent_mask': present,
                   'future_known_mask': known_masks})
    if include_future_semantics:
        if future_semantics is None:
            future_semantics = np.full(known_masks.shape, 255, np.uint8)
        target['future_semantics'] = future_semantics
    result['target'] = target
    result['metadata'].update({
        'future_sample_tokens': tokens, 'future': future,
        'target_visibility_policy': 'in_bounds & semantics!=255;independent_of_future_sensor_visibility',
        'target_known_mask': 'future_known_mask',
    })
    return result
