"""Small 3-D labels verify the E0 encoder's information and validity boundary."""
from copy import deepcopy

import numpy as np
import pytest
import torch


class RecordingCodec(torch.nn.Module):
    """Cheap stand-in for the already-tested VAE, retaining the real torch API."""

    def __init__(self):
        super().__init__()
        self.calls = []

    def encode(self, semantics, visibility, *, sample=False):
        self.calls.append((semantics.cpu().numpy().copy(), visibility.cpu().numpy().copy(),
                           sample, torch.is_grad_enabled(), self.training))
        # Expose transformed values so skipping warping or visibility affects output.
        known = torch.where(visibility, semantics, 17)
        value = known.float().sum()
        return value.expand(1, 16, 50, 50).clone()


class SmallIndex:
    """An explicit observation-only index; future traversal is guarded by method."""

    def __init__(self, *, history_tokens=('', 'past', 'previous', 'current'), future_tokens=('next', '', '', '', '', '')):
        self.history_tokens = list(history_tokens)
        self.future_tokens = list(future_tokens)
        self.loads = []
        self.sample_calls = 0
        self.grids = {}
        for token, value in [('past', 1), ('previous', 2), ('current', 3), ('next', 4)]:
            semantics = np.array([[[value], [255]], [[17], [value + 1]]], np.uint8)
            self.grids[token] = {'semantics': semantics, 'in_bounds': np.array([[[True], [True]], [[False], [True]]]),
                                 'visibility': {'mask_lidar': np.array([[[True], [True]], [[True], [False]]]),
                                                'mask_camera': np.ones((2, 2, 1), bool)}}

    def build_observation(self, scene, current_token, *, history=4, max_actors=8):
        assert scene == 'scene-1' and current_token == 'current'
        assert history == 4 and max_actors == 8
        mask = np.array([bool(token) for token in self.history_tokens])
        return {'actor_history': np.zeros((8, 4, 6), np.float32),
                'actor_observed_mask': np.tile(mask, (8, 1)),
                'ego_history': np.zeros((4, 6), np.float32),
                'ego_observed_mask': mask.copy(), 'history_mask': mask,
                'history_time_s': np.array([-1.5, -1., -.5, 0.]),
                'ego_state': np.zeros(4, np.float32)}, {
                'coordinate_frame': 'E0', 'oracle': True, 'scene': scene, 'token': current_token,
                'history_sample_tokens': self.history_tokens.copy()}

    def build_sample(self, scene, current_token, *, history=4, future=6, max_actors=8):
        self.sample_calls += 1
        observation, metadata = self.build_observation(scene, current_token, history=history, max_actors=max_actors)
        metadata['future_sample_tokens'] = self.future_tokens.copy()
        return {'observation': observation, 'metadata': metadata,
                'target': {'ego_target_mask': np.array([bool(t) for t in self.future_tokens]),
                           'actor_future': np.zeros((8, future, 6), np.float32)}}

    def load_occupancy_e0(self, source_token, current_token, *, source_frame, visibility_keys=()):
        assert current_token == 'current' and source_frame in {'ego', 'lidar'}
        self.loads.append((source_token, current_token, source_frame))
        return deepcopy(self.grids[source_token])


def test_observation_is_future_independent_and_aligns_padded_latents():
    from reactoccflow.data.e0_cache import encode_e0_observation
    index, codec = SmallIndex(), RecordingCodec()
    result = encode_e0_observation(index, 'scene-1', 'current', codec)
    obs = result['observation']
    assert index.sample_calls == 0
    assert [call[0] for call in index.loads] == ['past', 'previous', 'current']
    assert obs['history_latents'].shape == (6, 16, 50, 50)
    np.testing.assert_array_equal(obs['history_mask'], [False, False, False, True, True, True])
    assert not obs['history_latents'][:3].any()
    assert obs['history_latents'][3, 0, 0, 0] == 52  # visible1, three maskedempty17
    assert obs['actor_history'].shape == (8, 4, 6)
    assert obs['ego_history'].shape == (4, 6)
    np.testing.assert_array_equal(obs['history_time_s'], [0, 0, 0, -1, -.5, 0])
    assert result['metadata']['history_latent_sample_tokens'] == ['', '', '', 'past', 'previous', 'current']
    assert 'target' not in result and not any('future' in key for key in obs)
    assert all(isinstance(value, np.ndarray) for value in obs.values())
    assert codec.training  # evaluation mode during encoding is restored afterwards
    assert all(not sampled and not gradients and not training for _, _, sampled, gradients, training in codec.calls)


def test_history_encodes_warped_3d_semantics_and_intersection_of_validity_masks():
    from reactoccflow.data.e0_cache import encode_e0_observation
    index, codec = SmallIndex(), RecordingCodec()
    encode_e0_observation(index, 'scene-1', 'current', codec, source_frame='lidar', visibility_key='mask_lidar')
    semantics, visibility, *_ = codec.calls[0]
    assert semantics.shape == (1, 2, 2, 1)
    np.testing.assert_array_equal(semantics[0], index.grids['past']['semantics'])
    np.testing.assert_array_equal(visibility[0], [[[True], [False]], [[False], [False]]])
    assert all(call[2] == 'lidar' for call in index.loads)


def test_training_keeps_complete_known_future_grid_and_tail_masks_target_only():
    from reactoccflow.data.e0_cache import encode_e0_training_sample
    index, codec = SmallIndex(), RecordingCodec()
    result = encode_e0_training_sample(index, 'scene-1', 'current', codec)
    obs, target = result['observation'], result['target']
    assert index.sample_calls == 1
    assert [call[0] for call in index.loads] == ['past', 'previous', 'current', 'next']
    assert target['future_latents'].shape == (6, 16, 50, 50)
    np.testing.assert_array_equal(target['future_latent_mask'], [True, False, False, False, False, False])
    assert not target['future_latents'][1:].any()
    assert target['future_latents'][0, 0, 0, 0] == 43  # future known4+5, two maskedempty17
    assert target['future_known_mask'].shape == (6, 2, 2, 1)
    np.testing.assert_array_equal(target['future_known_mask'][0], [[[True], [False]], [[False], [True]]])
    assert not target['future_known_mask'][1:].any()
    assert target['actor_future'].shape == (8, 6, 6)
    assert not any('future' in key for key in obs)
    assert result['metadata']['posterior'] == 'mean'
    assert result['metadata']['coordinate_frame'] == 'E0'


def test_hist_last_selects_recent_observed_slots_without_changing_actor_history():
    from reactoccflow.data.e0_cache import encode_e0_observation
    index, codec = SmallIndex(), RecordingCodec()
    result = encode_e0_observation(index, 'scene-1', 'current', codec, hist_last=2)
    assert [call[0] for call in index.loads] == ['previous', 'current']
    np.testing.assert_array_equal(result['observation']['history_mask'], [False, False, False, False, True, True])
    assert result['observation']['actor_history'].shape == (8, 4, 6)


@pytest.mark.parametrize('options', [{'source_frame': 'guess'}, {'history_slots': 3}, {'hist_last': 5}, {'hist_last': 0}])
def test_invalid_contract_options_fail_before_loading(options):
    from reactoccflow.data.e0_cache import encode_e0_observation
    index = SmallIndex()
    with pytest.raises(ValueError):
        encode_e0_observation(index, 'scene-1', 'current', RecordingCodec(), **options)
    assert not index.loads


def test_missing_occupancy_is_an_error_rather_than_a_synthetic_scene_tail():
    from reactoccflow.data.e0_cache import encode_e0_training_sample
    index = SmallIndex()
    del index.grids['next']
    with pytest.raises(KeyError, match='next'):
        encode_e0_training_sample(index, 'scene-1', 'current', RecordingCodec())


def test_requested_input_visibility_must_exist_and_masks_match_grid():
    from reactoccflow.data.e0_cache import encode_e0_observation
    index = SmallIndex()
    del index.grids['past']['visibility']['mask_lidar']
    with pytest.raises(ValueError, match='mask_lidar'):
        encode_e0_observation(index, 'scene-1', 'current', RecordingCodec(), visibility_key='mask_lidar')
    index = SmallIndex()
    index.grids['past']['in_bounds'] = np.ones((2, 2), bool)
    with pytest.raises(ValueError, match='shape'):
        encode_e0_observation(index, 'scene-1', 'current', RecordingCodec())


def test_history_and_future_land_in_same_current_e0_xyz_before_encoding():
    from reactoccflow.data.e0_cache import encode_e0_training_sample
    from reactoccflow.geometry import GridSpec, warp_occupancy_nearest

    class MovingIndex(SmallIndex):
        def __init__(self):
            super().__init__(history_tokens=('', '', 'previous', 'current'))
            self.grid = GridSpec(shape=(4, 2, 3), origin=(0, 0, 0), voxel_size=(1, 1, 1))
            self.sources = {}
            # A fixed global object at xyz[2.5,.5,2.5]; ego moves +x,+z each frame.
            for token, cell in [('previous', (2, 0, 2)), ('current', (1, 0, 1)), ('next', (0, 0, 0))]:
                labels = np.full(self.grid.shape, 17, np.uint8)
                labels[cell] = 4
                self.sources[token] = labels

        def load_occupancy_e0(self, source_token, current_token, *, source_frame, visibility_keys=()):
            self.loads.append((source_token, current_token, source_frame))
            transform = np.eye(4)
            offset = {'previous': -1, 'current': 0, 'next': 1}[source_token]
            transform[:3, 3] = [offset, 0, offset]
            return warp_occupancy_nearest(
                self.sources[source_token], transform,
                visibility={'mask_lidar': np.ones(self.grid.shape, bool)},
                source_grid=self.grid, target_grid=self.grid)

    index, codec = MovingIndex(), RecordingCodec()
    result = encode_e0_training_sample(index, 'scene-1', 'current', codec)
    assert [load[0] for load in index.loads] == ['previous', 'current', 'next']
    for semantics, visibility, *_ in codec.calls:
        np.testing.assert_array_equal(np.argwhere(semantics[0] == 4), [[1, 0, 1]])
        assert visibility[0, 1, 0, 1]
    assert result['target']['future_known_mask'][0, 1, 0, 1]


def test_scene_tail_has_no_future_file_reads_and_has_false_known_voxels():
    from reactoccflow.data.e0_cache import encode_e0_training_sample
    index = SmallIndex(future_tokens=('', '', '', '', '', ''))
    result = encode_e0_training_sample(index, 'scene-1', 'current', RecordingCodec())
    assert [load[0] for load in index.loads] == ['past', 'previous', 'current']
    target = result['target']
    assert target['future_known_mask'].shape == (6, 200, 200, 16)
    assert not target['future_known_mask'].any()
    assert not target['future_latent_mask'].any()
    assert not target['future_latents'].any()


def test_future_sensor_visibility_is_neither_required_nor_used():
    from reactoccflow.data.e0_cache import encode_e0_training_sample
    index, codec = SmallIndex(), RecordingCodec()
    del index.grids['next']['visibility']
    result = encode_e0_training_sample(index, 'scene-1', 'current', codec)
    assert result['target']['future_latents'][0, 0, 0, 0] == 43
    assert codec.calls[-1][1][0, 1, 1, 0]


def test_visibility_key_selects_observed_sensor_without_changing_actor_features():
    from reactoccflow.data.e0_cache import encode_e0_observation
    index, codec = SmallIndex(), RecordingCodec()
    result = encode_e0_observation(index, 'scene-1', 'current', codec, visibility_key='mask_camera')
    assert result['observation']['history_latents'][3, 0, 0, 0] == 37  #1+2+17+17
    np.testing.assert_array_equal(result['observation']['actor_history'], np.zeros((8, 4, 6)))
    assert result['metadata']['input_visibility_key'] == 'mask_camera'


@pytest.mark.parametrize('failure', ['shape', 'nonfinite'])
def test_invalid_codec_output_is_rejected_and_training_mode_restored(failure):
    from reactoccflow.data.e0_cache import encode_e0_observation

    class InvalidCodec(RecordingCodec):
        def encode(self, semantics, visibility, *, sample=False):
            if failure == 'shape':
                return torch.zeros(1, 16, 2, 2)
            return torch.full((1, 16, 50, 50), float('nan'))

    codec = InvalidCodec()
    with pytest.raises(ValueError, match='shape|nonfinite'):
        encode_e0_observation(SmallIndex(), 'scene-1', 'current', codec)
    assert codec.training


def test_opt_in_future_semantics_retains_same_warp_unknowns_and_scene_tail():
    from reactoccflow.data.e0_cache import encode_e0_training_sample
    index, codec = SmallIndex(), RecordingCodec()
    result = encode_e0_training_sample(
        index, 'scene-1', 'current', codec, include_future_semantics=True)
    target = result['target']
    labels = target['future_semantics']
    assert labels.shape == (6, 2, 2, 1) and labels.dtype == np.uint8
    np.testing.assert_array_equal(labels[0], [[[4], [255]], [[17], [5]]])
    np.testing.assert_array_equal(target['future_known_mask'][0], [[[True], [False]], [[False], [True]]])
    assert (labels[1:] == 255).all() and not target['future_known_mask'][1:].any()
    assert np.isfinite(labels).all() and np.isin(labels, [*range(18), 255]).all()
    assert [load[0] for load in index.loads] == ['past', 'previous', 'current', 'next']
    np.testing.assert_array_equal(labels[0], codec.calls[-1][0][0])
    assert not any('future' in key for key in result['observation'])


def test_future_semantics_disabled_keeps_original_default_outputs_exactly():
    from reactoccflow.data.e0_cache import encode_e0_training_sample
    original = encode_e0_training_sample(SmallIndex(), 'scene-1', 'current', RecordingCodec())
    disabled = encode_e0_training_sample(
        SmallIndex(), 'scene-1', 'current', RecordingCodec(), include_future_semantics=False)
    assert set(disabled['target']) == {
        'ego_target_mask', 'actor_future', 'future_latents', 'future_latent_mask', 'future_known_mask'}
    assert original['metadata'] == disabled['metadata']
    for group in ('observation', 'target'):
        assert original[group].keys() == disabled[group].keys()
        for key in original[group]:
            np.testing.assert_array_equal(original[group][key], disabled[group][key], err_msg=key)


def test_opt_in_future_semantics_all_missing_scene_tail_is_unknown():
    from reactoccflow.data.e0_cache import encode_e0_training_sample
    index = SmallIndex(future_tokens=('', '', '', '', '', ''))
    target = encode_e0_training_sample(
        index, 'scene-1', 'current', RecordingCodec(), include_future_semantics=True)['target']
    assert target['future_semantics'].shape == target['future_known_mask'].shape == (6, 200, 200, 16)
    assert target['future_semantics'].dtype == np.uint8
    assert (target['future_semantics'] == 255).all() and not target['future_known_mask'].any()


def test_future_semantics_option_requires_boolean_before_loading():
    from reactoccflow.data.e0_cache import encode_e0_training_sample
    index = SmallIndex()
    with pytest.raises(ValueError, match='include_future_semantics'):
        encode_e0_training_sample(
            index, 'scene-1', 'current', RecordingCodec(), include_future_semantics='yes')
    assert not index.loads and not index.sample_calls


def test_explicit_dense_preserves_known_invisible_voxels_and_needs_no_sensor_masks():
    from reactoccflow.data.e0_cache import encode_e0_observation
    codec = RecordingCodec()
    original = encode_e0_observation(SmallIndex(), 'scene-1', 'current', codec, visibility_key=None)
    np.testing.assert_array_equal(codec.calls[0][1][0], [[[True], [False]], [[False], [True]]])
    assert original['metadata']['history_visibility'] == 'dense_semantic'
    assert original['metadata']['input_visibility_key'] is None
    assert original['metadata']['input_visibility_policy'] == 'in_bounds & semantics!=255'
    for replacement in ('missing', 'false', 'malformed'):
        index = SmallIndex()
        for grid in index.grids.values():
            if replacement == 'missing':
                del grid['visibility']
            else:
                grid['visibility'] = {'mask_lidar': np.zeros((2, 2, 1) if replacement == 'false' else (1,), bool)}
        result = encode_e0_observation(index, 'scene-1', 'current', RecordingCodec(), visibility_key=None)
        np.testing.assert_array_equal(result['observation']['history_latents'], original['observation']['history_latents'])


def test_switching_history_mode_keeps_actor_inputs_and_all_future_targets_identical():
    from reactoccflow.data.e0_cache import encode_e0_training_sample
    dense = encode_e0_training_sample(SmallIndex(), 'scene-1', 'current', RecordingCodec(),
                                     visibility_key=None, include_future_semantics=True)
    masked = encode_e0_training_sample(SmallIndex(), 'scene-1', 'current', RecordingCodec(),
                                      visibility_key='mask_lidar', include_future_semantics=True)
    assert dense['observation']['history_latents'][3, 0, 0, 0] == 37
    assert masked['observation']['history_latents'][3, 0, 0, 0] == 52
    for key in dense['observation']:
        if key != 'history_latents':
            np.testing.assert_array_equal(dense['observation'][key], masked['observation'][key])
    for key in dense['target']:
        np.testing.assert_array_equal(dense['target'][key], masked['target'][key])


def test_restored_default_keeps_hidden_ground_masked_without_completion(monkeypatch):
    import reactoccflow.data.e0_cache as encoder
    index = SmallIndex()
    for token in ('past', 'previous', 'current'):
        index.grids[token] = {'semantics': np.full((3, 3, 1), 12, np.uint8),
                             'in_bounds': np.ones((3, 3, 1), bool),
                             'visibility': {'mask_lidar': np.ones((3, 3, 1), bool)}}
    index.grids['current']['visibility']['mask_lidar'][1, 1, 0] = False
    def unexpected_completion(*args):
        raise AssertionError('First-version mask must not complete hidden ground')
    monkeypatch.setattr(encoder, 'complete_masked_ground_history', unexpected_completion)
    codec = RecordingCodec()
    default = encoder.encode_e0_training_sample(index, 'scene-1', 'current', codec)
    assert not codec.calls[2][1][0, 1, 1, 0]
    assert default['metadata']['history_visibility'] == 'mask_lidar'
    explicit = encoder.encode_e0_training_sample(index, 'scene-1', 'current', RecordingCodec(),
                                               history_visibility='mask_lidar')
    assert default['metadata'] == explicit['metadata']
    for section in ('observation', 'target'):
        for key in default[section]:
            np.testing.assert_array_equal(default[section][key], explicit[section][key])


@pytest.mark.parametrize('visibility_key', [None, 'mask_lidar'])
def test_future_grid_changes_cannot_change_history_encoding(visibility_key):
    from reactoccflow.data.e0_cache import encode_e0_training_sample
    index = SmallIndex()
    first = encode_e0_training_sample(index, 'scene-1', 'current', RecordingCodec(), visibility_key=visibility_key)
    index.grids['next']['semantics'][:] = 10
    changed = encode_e0_training_sample(index, 'scene-1', 'current', RecordingCodec(), visibility_key=visibility_key)
    for key in first['observation']:
        np.testing.assert_array_equal(first['observation'][key], changed['observation'][key])
    assert not np.array_equal(first['target']['future_latents'], changed['target']['future_latents'])
