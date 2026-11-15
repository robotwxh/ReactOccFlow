"""Adversarial checks for full count and protocol comparison integrity."""
from copy import deepcopy
import csv
import json

import numpy as np
import pytest

from reactoccflow.evaluation.architecture_report import (
    MATCHED_SETTINGS, SCORE_NAMES, baseline_metric, delta, identity_digest,
    metric_from_counts, validate_completed, validate_ground_truth,
    validate_shared_protocol, write_report,
)
from reactoccflow.evaluation.occupancy import OccupancyAccumulator


def cached_score(accumulator):
    report = accumulator.compute()
    for protocol in ('standard', 'legacy'):
        selected = [100*report[protocol]['per_time_mean_iou'][i] for i in (1, 3, 5)]
        report[protocol]['miou_percent_at_1_2_3s'] = selected
        report[protocol]['miou_percent_mean_1_2_3s'] = float(np.mean(selected))
    return report


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, default=lambda x: x.tolist()))


def make_full_fixture(root):
    identity = {'world_size': 2, 'windows': [['scene', str(i)] for i in range(4)]}
    digest = identity_digest(identity)
    write_json(root/'full/identity.json', identity)
    total = OccupancyAccumulator()
    for rank in range(2):
        accumulator = OccupancyAccumulator()
        for index in range(rank, 4, 2):
            target = np.tile(np.array([0, 1, 17]), (6, 1))
            prediction = target.copy()
            prediction[:, 0] = index % 2
            accumulator.update(prediction, target)
            total.update(prediction, target)
        score = cached_score(accumulator)
        coverage = {'valid_target_voxels': np.full(6, 6), 'supported_target_voxels': np.full(6, 5)}
        part = {'status': 'completed', 'checkpoint_step': 100000, 'rank': rank,
                'windows': 2, 'identity_sha256': digest,
                'metrics': {name: score for name in SCORE_NAMES}, 'paper_frame_coverage': coverage}
        state = {'identity_sha256': digest, 'finished': True, 'completed': list(range(rank, 4, 2)),
                 'accumulators': {name: accumulator.state_dict() for name in SCORE_NAMES},
                 'coverage': coverage}
        write_json(root/f'full/rank_{rank:03d}/result.json', part)
        write_json(root/f'full/rank_{rank:03d}/state.json', state)
    result = {'status': 'completed', 'windows': 4, 'world_size': 2, 'identity_sha256': digest,
              'metrics': {name: cached_score(total) for name in SCORE_NAMES},
              'paper_frame_coverage': {'valid_target_voxels': [12]*6, 'supported_target_voxels': [10]*6}}
    write_json(root/'full/result.json', result)
    return result


def settings_identity():
    identity = {key: None for key in MATCHED_SETTINGS}
    identity.update(nfe=10, seed=2026, precision='float32', batch_size=4, world_size=2,
                    encode_chunk=4, decode_chunk=2, windows=[['scene', 'a']],
                    condition='oracle_actor_history_and_logged_future_ego_action',
                    input_sha256={'/project/data/research/normalization.json': 'norm',
                                  '/raw/scene/token/labels.npz': 'gt',
                                  '/project/models/model.py': 'source'}, protocol_version=1)
    return identity


def test_horizon_average_cannot_silently_become_all_six_frame_average():
    accumulator = OccupancyAccumulator()
    truth = np.zeros((6, 1), dtype=np.uint8)
    prediction = np.array([[0], [17], [0], [17], [0], [17]], dtype=np.uint8)
    accumulator.update(prediction, truth)
    score = cached_score(accumulator)
    report = metric_from_counts(score, 1)
    assert report['standard']['miou_percent_mean_1_2_3s'] == 0.
    score['standard']['miou_percent_mean_1_2_3s'] = 50.
    with pytest.raises(ValueError, match='count-derived'):
        metric_from_counts(score, 1)


def test_completed_fixture_proves_rank_coverage_and_count_sum(tmp_path):
    make_full_fixture(tmp_path)
    report = validate_completed(tmp_path, 100000, expected_windows=4)
    assert report['rank_coverage_verified'] is True
    assert report['metrics']['paper_frame_full']['windows'] == 4


@pytest.mark.parametrize('mutation', ['duplicate', 'wrong_step', 'unfinished', 'rank_count', 'merged_count'])
def test_invalid_rank_coverage_or_forged_counts_are_rejected(tmp_path, mutation):
    make_full_fixture(tmp_path)
    state_path = tmp_path/'full/rank_001/state.json'
    state = json.loads(state_path.read_text())
    if mutation == 'duplicate':
        state['completed'] = [0, 2]
    elif mutation == 'unfinished':
        state['finished'] = False
    elif mutation == 'rank_count':
        state['accumulators']['paper_frame_full']['seen'][0][0] += 1
    elif mutation == 'wrong_step':
        path = tmp_path/'full/rank_001/result.json'
        value = json.loads(path.read_text())
        value['checkpoint_step'] = 52000
        write_json(path, value)
    else:
        path = tmp_path/'full/result.json'
        value = json.loads(path.read_text())
        value['metrics']['paper_frame_full']['seen'][0][0] += 1
        write_json(path, value)
    write_json(state_path, state)
    with pytest.raises(ValueError):
        validate_completed(tmp_path, 100000, expected_windows=4)


@pytest.mark.parametrize('field,value', [('seed', 123), ('nfe', 8), ('precision', 'bf16'),
                                        ('windows', [['scene', 'different']]), ('batch_size', 2)])
def test_protocol_differences_reject_false_paired_comparison(field, value):
    previous = settings_identity()
    current = deepcopy(previous)
    current[field] = value
    with pytest.raises(ValueError, match='settings differ'):
        validate_shared_protocol(current, previous, check_evaluator_code=False)


def test_architecture_and_source_upgrade_are_allowed_but_data_changes_are_not():
    previous = settings_identity()
    current = deepcopy(previous)
    current.update(protocol_version=2, model_options={'backbone': 'dit'})
    current['input_sha256']['/project/models/model.py'] = 'different_architecture'
    result = validate_shared_protocol(current, previous, check_evaluator_code=False)
    assert result['identity_protocol_versions'] == [1, 2]
    current['input_sha256']['/project/data/research/normalization.json'] = 'different_norm'
    with pytest.raises(ValueError, match='Shared data'):
        validate_shared_protocol(current, previous, check_evaluator_code=False)


def test_ground_truth_mismatch_with_occfm_is_rejected(tmp_path):
    result = make_full_fixture(tmp_path)
    seen = np.asarray(result['metrics']['paper_frame_full']['seen'])
    matrix = np.stack([np.diag(row) for row in seen])
    baseline = {'metrics': {'class_confusion_counts': matrix.tolist()}}
    assert validate_ground_truth(result, deepcopy(result), baseline)['full_grid_GT_matches_original_OccFM']
    baseline['metrics']['class_confusion_counts'][0][0][0] += 1
    with pytest.raises(ValueError, match='original OccFM'):
        validate_ground_truth(result, deepcopy(result), baseline)


def test_baseline_is_recomputed_from_confusion_not_trusted_reported_mean():
    matrix = np.zeros((6, 18, 18), dtype=np.int64)
    matrix[:, 0, 0] = 3
    matrix[:, 1, 0] = 1
    # Legacy: class0=.75, class1=0, remaining15 absent classes=1.
    value = 100*(.75+15)/17
    baseline = {'status': 'completed', 'validation_scope': 'full',
                'metrics': {'samples': 4219, 'class_confusion_counts': matrix.tolist(),
                            'miou_percent_at_1_2_3s': [value]*3,
                            'miou_percent_mean_1_2_3s': value}}
    assert baseline_metric(baseline)['avg'] == pytest.approx(value)
    baseline['metrics']['miou_percent_mean_1_2_3s'] += 1.
    with pytest.raises(ValueError, match='average differs'):
        baseline_metric(baseline)


def test_comparison_delta_has_no_assumption_about_result_direction():
    current = {'miou_1_2_3s': [20., 20., 20.], 'avg': 20.}
    previous = {'miou_1_2_3s': [30., 25., 20.], 'avg': 25.}
    assert delta(current, previous) == {'per_horizon_percentage_points': [-10., -5., 0.],
                                        'average_percentage_points': -5.}


def test_report_emits_four_models_and_all_seventeen_classes_at_three_horizons(tmp_path):
    full = make_full_fixture(tmp_path)
    metrics = {name: metric_from_counts(score, 4) for name, score in full['metrics'].items()}
    row = metrics['paper_frame_full']['legacy']
    values = {'miou_1_2_3s': row['miou_percent_at_1_2_3s'], 'avg': row['miou_percent_mean_1_2_3s']}
    comparison = {'rows': [{'name': name, **values} for name in ('paper', 'reproduction', 'U-CFM', 'DiT')],
                  'standard_dit': values, 'all_score_protocols': {'standard_dit': metrics, 'actor_u_cfm': metrics},
                  'paper_frame_coverage': {'fraction_per_horizon': [5/6]*6},
                  'delta_vs_actor_u_cfm': {'average_percentage_points': 0.},
                  'delta_vs_occfm_paper_future': {'average_percentage_points': -1.},
                  'delta_vs_occfm_reproduced_future': {'average_percentage_points': 1.}}
    write_report(tmp_path, comparison, {'class_iou': row['per_time_class_iou']})
    with (tmp_path/'comparison.csv').open() as stream:
        assert len(list(csv.DictReader(stream))) == 4
    with (tmp_path/'per_class_comparison.csv').open() as stream:
        classes = list(csv.DictReader(stream))
    assert len(classes) == 51
    assert {int(item['class_id']) for item in classes} == set(range(17))
    assert {int(item['horizon_s']) for item in classes} == {1, 2, 3}
    assert json.loads((tmp_path/'comparison.json').read_text())['standard_dit'] == values
    text = (tmp_path/'标准DiT与CFM及OccFM_mIoU对照.md').read_text()
    assert '不是只改变主干的受控消融' in text
    assert 'https://arxiv.org/html/2506.23434v2' in text
