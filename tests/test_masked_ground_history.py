"""Information-boundary and evidence tests for observed-only ground memory."""
import numpy as np
import pytest

from reactoccflow.data.history_visibility import (
    complete_masked_ground_history, resolve_history_visibility,
    visibility_key_from_mode, visibility_metadata,
)


def series(values, seen, ground=12):
    labels = np.full((len(values), 3, 3, 1), ground, np.uint8)
    labels[:, 1, 1, 0] = values
    masks = np.ones_like(labels, bool)
    masks[:, 1, 1, 0] = seen
    return labels, masks


def test_hidden_labels_do_not_supply_completion_and_inputs_are_unchanged():
    labels, masks = series([12, 12, 4, 255], [1, 1, 0, 0])
    bounds = np.ones_like(masks)
    original = labels.copy()
    completed, visible = complete_masked_ground_history(labels, bounds, masks)
    np.testing.assert_array_equal(completed[:, 1, 1, 0], [12, 12, 12, 12])
    assert visible.all()
    np.testing.assert_array_equal(labels, original)
    changed = labels.copy()
    changed[~masks] = 11
    other = complete_masked_ground_history(changed, bounds, masks)
    np.testing.assert_array_equal(other[0], completed)
    np.testing.assert_array_equal(other[1], visible)


@pytest.mark.parametrize('ground', [11, 12, 13, 14])
def test_two_direct_observations_required_and_filled_cells_are_not_evidence(ground):
    labels, masks = series([ground]*4, [1, 0, 1, 0], ground)
    completed, visible = complete_masked_ground_history(labels, np.ones_like(masks), masks)
    np.testing.assert_array_equal(visible[:, 1, 1, 0], [1, 0, 1, 1])
    assert completed[1, 1, 1, 0] == 17
    assert completed[-1, 1, 1, 0] == ground
    masks[2, 1, 1, 0] = False
    assert not complete_masked_ground_history(labels, np.ones_like(masks), masks)[1][1:, 1, 1, 0].any()


@pytest.mark.parametrize('conflict', [0, 4, 11, 13, 14, 17, 255])
def test_visible_conflicts_preserved_and_reset_ground_memory(conflict):
    labels, masks = series([12, 12, conflict, 12], [1, 1, 1, 0])
    completed, visible = complete_masked_ground_history(labels, np.ones_like(masks), masks)
    if conflict == 255:
        # Unknown is not a direct observation and cannot invalidate observed ground.
        assert completed[2, 1, 1, 0] == 12
        assert visible[-1, 1, 1, 0]
    else:
        assert completed[2, 1, 1, 0] == conflict
        assert not visible[-1, 1, 1, 0]


@pytest.mark.parametrize('category', [0, 1, 4, 7, 15, 16, 17])
def test_non_ground_and_out_of_bounds_are_never_completed(category):
    labels = np.full((4, 2, 1, 1), category, np.uint8)
    labels[:, 1] = 12
    masks = np.ones_like(labels, bool)
    masks[2:] = False
    bounds = np.ones_like(masks)
    bounds[2:, 1] = False
    _, visible = complete_masked_ground_history(labels, bounds, masks)
    assert not visible[2:].any()


def test_completion_is_prefix_causal():
    labels, masks = series([12, 12, 11, 11], [1, 0, 1, 1])
    bounds = np.ones_like(masks)
    full = complete_masked_ground_history(labels, bounds, masks)
    for length in range(1, 4):
        prefix = complete_masked_ground_history(labels[:length], bounds[:length], masks[:length])
        for actual, expected in zip(prefix, full):
            np.testing.assert_array_equal(actual, expected[:length])


def test_policy_identity_distinguishes_base_mask_from_completion():
    mode = 'mask_lidar_temporal_ground'
    key = visibility_key_from_mode(mode)
    assert key == 'mask_lidar'
    metadata = visibility_metadata(key, history_visibility=mode)
    assert resolve_history_visibility(metadata) == mode
    assert metadata['input_visibility_policy'] != visibility_metadata(key)['input_visibility_policy']
    with pytest.raises(ValueError, match='policy'):
        resolve_history_visibility({**metadata, 'input_visibility_policy': visibility_metadata(key)['input_visibility_policy']})


def test_completion_cache_keeps_future_and_actor_targets_unchanged():
    from test_e0_cache import RecordingCodec, SmallIndex
    from reactoccflow.data.e0_cache import encode_e0_training_sample
    index = SmallIndex()
    for token in ('past', 'previous', 'current'):
        grid = index.grids[token]
        grid['in_bounds'] = np.ones((3, 3, 1), bool)
        grid['semantics'] = np.full((3, 3, 1), 12, np.uint8)
        grid['visibility']['mask_lidar'] = np.ones((3, 3, 1), bool)
        grid['visibility']['mask_lidar'][1, 1, 0] = token != 'current'
    base = encode_e0_training_sample(index, 'scene-1', 'current', RecordingCodec(),
                                    history_visibility='mask_lidar')
    codec = RecordingCodec()
    new = encode_e0_training_sample(index, 'scene-1', 'current', codec,
                                   history_visibility='mask_lidar_temporal_ground')
    assert codec.calls[2][1].all()
    assert not np.array_equal(base['observation']['history_latents'], new['observation']['history_latents'])
    for key in base['target']:
        np.testing.assert_array_equal(base['target'][key], new['target'][key])
    for key in base['observation']:
        if key != 'history_latents':
            np.testing.assert_array_equal(base['observation'][key], new['observation'][key])
    index.grids['next']['semantics'][:] = 11
    changed = encode_e0_training_sample(index, 'scene-1', 'current', RecordingCodec(),
                                       history_visibility='mask_lidar_temporal_ground')
    np.testing.assert_array_equal(new['observation']['history_latents'], changed['observation']['history_latents'])


def test_height_support_uses_direct_observations_and_rejects_one_sided_extension():
    labels, masks = series([12]*4, [1, 1, 0, 0])
    masks[2:, 0, 1] = False
    masks[2:, 1, 0] = False
    completed, visible = complete_masked_ground_history(labels, np.ones_like(masks), masks)
    assert not visible[2:, 1, 1].any()
    assert (completed[2:, 1, 1] == 17).all()


def test_a_single_opposite_pair_does_not_supply_four_neighbour_support():
    labels, masks = series([12]*4, [1, 1, 0, 0])
    masks[2:, 0, 1] = False
    assert not complete_masked_ground_history(labels, np.ones_like(masks), masks)[1][2:, 1, 1].any()
