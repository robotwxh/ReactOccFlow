"""Independent analytic checks for the held-out generation evaluation protocol."""
import json

import numpy as np
import pytest
import torch

from reactoccflow.evaluation.occupancy import OccupancyAccumulator
from reactoccflow.evaluation.reactflow_generation import (
    SCORE_NAMES, aggregate, atomic_json, checkpoint_model_options, make_initial_noise, make_windows,
    paper_horizon_summary, partition_indices, project_prediction_to_future,
    window_seed,
)
from reactoccflow.geometry import GridSpec, transform_matrix


def test_checkpoint_architecture_overrides_default_and_loads_trained_weights():
    from reactoccflow.models.react_flow import ReactFlow
    from reactoccflow.models.standard_dit import StandardDiT
    config = {'mode': 'joint', 'backbone': 'dit', 'dit': {
        'in_channels': 16, 'input_size': 50, 'frames': 12, 'trajectory_length': 6,
        'patch_size': 1, 'hidden_size': 32, 'depth': 2, 'num_heads': 4,
        'mlp_ratio': 2., 'activation_checkpointing': False}}
    options = checkpoint_model_options({'config': config})
    source = ReactFlow(backbone='dit', dit_config=options['dit_config'])
    restored = ReactFlow(**options)
    restored.load_state_dict(source.state_dict(), strict=True)
    assert isinstance(restored.transition_model.trunk, StandardDiT)
    assert restored.transition_model.trunk.depth == 2
    assert restored.transition_model.trunk.hidden_size == 32
    assert config['dit']['frames'] == 12  # Saved metadata is not mutated.


def test_checkpoint_selection_retains_legacy_u_shape_and_rejects_other_modes():
    assert checkpoint_model_options({'config': {'mode': 'joint'}}) == {}
    assert checkpoint_model_options({'config': {'mode': 'joint', 'backbone': 'occfm_unet'}}) == {}
    with pytest.raises(ValueError, match='Unsupported checkpoint backbone'):
        checkpoint_model_options({'config': {'mode': 'joint', 'backbone': 'unknown'}})
    with pytest.raises(ValueError, match='joint'):
        checkpoint_model_options({'config': {'mode': 'occupancy_only', 'backbone': 'dit'}})


def test_checkpoint_with_different_grid_is_rejected_before_weight_loading():
    with pytest.raises(ValueError, match='input_size'):
        checkpoint_model_options({'config': {'mode': 'joint', 'backbone': 'dit',
                                            'dit': {'in_channels': 16, 'input_size': 40}}})


def test_paper_average_uses_one_two_three_seconds_and_not_six_frame_average():
    accumulator = OccupancyAccumulator()
    truth = np.zeros((6, 1), dtype=np.uint8)
    prediction = np.array([[0], [17], [0], [17], [0], [17]], dtype=np.uint8)
    accumulator.update(prediction, truth)
    report = paper_horizon_summary(accumulator)
    assert report['standard']['miou_percent_at_1_2_3s'] == [0., 0., 0.]
    assert report['standard']['miou_percent_mean_1_2_3s'] == 0.
    assert report['standard']['mean_over_horizons'] == .5


def test_mask_unknown_and_hallucinated_classes_are_counted_correctly():
    accumulator = OccupancyAccumulator()
    truth = np.tile(np.array([0, 0, 255, 17]), (6, 1))
    prediction = np.tile(np.array([0, 1, 4, 2]), (6, 1))
    mask = np.tile(np.array([True, True, True, False]), (6, 1))
    accumulator.update(prediction, truth, visibility=mask)
    report = paper_horizon_summary(accumulator)
    assert report['valid_voxels'] == [2]*6
    assert report['standard']['miou_percent_mean_1_2_3s'] == 25.
    assert report['legacy']['miou_percent_mean_1_2_3s'] > 25.


def test_partition_has_exact_once_coverage_without_padding():
    pieces = [partition_indices(11, rank, 3) for rank in range(3)]
    assert sorted(sum(pieces, [])) == list(range(11))
    assert set(pieces[0]).isdisjoint(pieces[1])
    with pytest.raises(ValueError):
        partition_indices(11, 3, 3)


def test_window_noise_is_independent_of_batch_and_partition():
    samples = [{'scene': 'a', 'token': 'first'}, {'scene': 'b', 'token': 'second'}]
    whole = make_initial_noise(samples, 2026, torch.device('cpu'))
    second = make_initial_noise(samples[1:], 2026, torch.device('cpu'))
    for key in whole:
        torch.testing.assert_close(whole[key][1:], second[key], rtol=0, atol=0)
    assert window_seed(2026, 'a', 'first') != window_seed(2026, 'b', 'first')


def test_prediction_to_future_uses_e0_to_future_direction_and_empty_outside():
    grid = GridSpec(shape=(4, 1, 1), origin=(0, 0, 0), voxel_size=(1, 1, 1))
    prediction = np.array([4, 5, 6, 7], dtype=np.uint8).reshape(grid.shape)
    # Future ego moved +1m in global x. Its first voxel samples E0 x=1.5m.
    warped = project_prediction_to_future(prediction, np.eye(4), transform_matrix([1, 0, 0]), grid=grid)
    np.testing.assert_array_equal(warped['semantics'].reshape(-1), [5, 6, 7, 17])
    np.testing.assert_array_equal(warped['in_bounds'].reshape(-1), [True, True, True, False])


def test_prediction_projection_rotates_future_basis_and_preserves_class_ids():
    grid = GridSpec(shape=(2, 2, 1), origin=(-1, -1, 0), voxel_size=(1, 1, 1))
    prediction = np.array([1, 2, 3, 4], dtype=np.uint8).reshape(grid.shape)
    root = np.sqrt(.5)
    future = transform_matrix(quaternion_wxyz=[root, 0, 0, root])
    warped = project_prediction_to_future(prediction, np.eye(4), future, grid=grid)
    np.testing.assert_array_equal(warped['semantics'][:, :, 0], [[3, 1], [4, 2]])
    assert warped['in_bounds'].all()


def test_window_identity_binds_history_and_future_without_swapping():
    resource = {'frames': [['scene', str(i)] for i in range(12)],
                'latent_gt_indices': list(range(12)), 'flow_windows': [list(range(12))]}
    frames = [{'scene': 'scene', 'token': str(i)} for i in range(12)]
    frames[5].update(history_sample_tokens=['2', '3', '4', '5'],
                     future_sample_tokens=['6', '7', '8', '9', '10', '11'])
    windows = make_windows(resource, {'frames': frames})
    assert windows[0]['token'] == '5'
    frames[5]['future_sample_tokens'][0] = '5'
    with pytest.raises(ValueError, match='Future identity'):
        make_windows(resource, {'frames': frames})


def test_rank_aggregation_sums_counts_and_rejects_duplicate_coverage(tmp_path):
    for rank in range(2):
        directory = tmp_path / f'rank_{rank:03d}'
        directory.mkdir()
        accumulator = OccupancyAccumulator()
        accumulator.update(np.zeros((6, 1), dtype=np.uint8), np.zeros((6, 1), dtype=np.uint8))
        state = {'identity_sha256': 'same', 'completed': [rank], 'finished': True,
                 'accumulators': {name: accumulator.state_dict() for name in SCORE_NAMES},
                 'coverage': {'valid_target_voxels': np.ones(6, dtype=np.int64),
                              'supported_target_voxels': np.ones(6, dtype=np.int64)}}
        atomic_json(directory/'state.json', state)
    result = aggregate(tmp_path, 'same', 2, 2)
    assert result['windows'] == 2
    assert result['metrics']['paper_frame_full']['windows'] == 2
    assert result['metrics']['paper_frame_full']['standard']['miou_percent_mean_1_2_3s'] == 100.
    path = tmp_path/'rank_001/state.json'
    state = json.loads(path.read_text())
    state['completed'] = [0]
    atomic_json(path, state)
    with pytest.raises(ValueError, match='duplicated or incomplete'):
        aggregate(tmp_path, 'same', 2, 2)
