"""Small official-schema tables test observation/target information boundaries."""
import json
from pathlib import Path
import numpy as np
import pytest


def official_fixture(root: Path):
    tables = {
        'scene': [{'token': 'scene-token', 'name': 'scene-0001', 'first_sample_token': 's0', 'last_sample_token': 's3', 'nbr_samples': 4}],
        'sample': [], 'sample_annotation': [], 'sample_data': [], 'ego_pose': [],
        'instance': [{'token': t, 'category_token': 'car'} for t in ['actor-a', 'actor-new', 'invisible']],
        'category': [{'token': 'car', 'name': 'vehicle.car'}],
        'sensor': [{'token': 'lidar-sensor', 'channel': 'LIDAR_TOP', 'modality': 'lidar'}],
        'calibrated_sensor': [{'token': 'lidar-cal', 'sensor_token': 'lidar-sensor', 'translation': [1, 0, 1], 'rotation': [1, 0, 0, 0]}],
    }
    q = [np.sqrt(.5), 0, 0, np.sqrt(.5)]
    for i in range(4):
        tables['sample'].append({'token': f's{i}', 'timestamp': i * 500000, 'prev': f's{i-1}' if i else '', 'next': f's{i+1}' if i < 3 else '', 'scene_token': 'scene-token'})
        tables['sample_data'].append({'token': f'l{i}', 'sample_token': f's{i}', 'is_key_frame': True, 'ego_pose_token': f'e{i}', 'calibrated_sensor_token': 'lidar-cal'})
        tables['ego_pose'].append({'token': f'e{i}', 'translation': [i, 0, 0], 'rotation': q})
        for actor in (['actor-a'] if i < 3 else ['actor-new']):
            tables['sample_annotation'].append({'token': f'{actor}-{i}', 'sample_token': f's{i}', 'instance_token': actor, 'translation': [i + 1, 4, 0], 'rotation': [1, 0, 0, 0], 'size': [2, 4, 1], 'visibility_token': '4', 'num_lidar_pts': 4})
    tables['sample_annotation'].append({'token': 'inv', 'sample_token': 's2', 'instance_token': 'invisible', 'translation': [2, 0, 0], 'rotation': [1, 0, 0, 0], 'size': [2, 4, 1], 'visibility_token': '4', 'num_lidar_pts': 0})
    folder = root / 'v1.0-trainval'
    folder.mkdir(parents=True)
    for table, rows in tables.items():
        (folder / f'{table}.json').write_text(json.dumps(rows))
    return tables


def index_fixture(tmp_path):
    from reactoccflow.data.research import NuScenesLabelIndex
    official_fixture(tmp_path)
    return NuScenesLabelIndex(tmp_path)


def test_actor_history_is_fixed_e0_backward_velocity_and_current_size(tmp_path):
    sample = index_fixture(tmp_path).build_sample('scene-0001', 's2', history=4, future=6, max_actors=8)
    obs, target, meta = sample['observation'], sample['target'], sample['metadata']
    assert obs['actor_history'].shape == (8, 4, 6)
    np.testing.assert_allclose(obs['actor_history'][0, -1], [4, -1, -1, 0, 0, -2], atol=1e-6)
    assert obs['actor_observed_mask'][0].tolist() == [False, True, True, True]
    assert obs['actor_velocity_mask'][0].tolist() == [False, False, True, True]
    np.testing.assert_array_equal(obs['actor_size'][0], [4, 2, 1])
    assert obs['actor_class'][0] == 4
    assert meta['actor_instance_tokens'][0] == 'actor-a'
    assert meta['oracle'] is True
    assert not target['actor_target_mask'].any()
    assert not target['actor_future_exists'].any()
    np.testing.assert_allclose(target['ego_future'][0, :2], [0, -1], atol=1e-6)
    assert target['ego_target_mask'].tolist() == [True, False, False, False, False, False]


def test_future_records_cannot_change_observation_or_selection(tmp_path):
    from reactoccflow.data.research import NuScenesLabelIndex
    tables = official_fixture(tmp_path)
    first, metadata = NuScenesLabelIndex(tmp_path).build_observation('scene-0001', 's2')
    for table in ('sample', 'sample_annotation', 'sample_data', 'ego_pose'):
        rows = tables[table]
        if table == 'sample': rows = [r for r in rows if r['token'] != 's3']
        elif table in ('sample_annotation', 'sample_data'): rows = [r for r in rows if r['sample_token'] != 's3']
        else: rows = [r for r in rows if r['token'] != 'e3']
        (tmp_path / 'v1.0-trainval' / f'{table}.json').write_text(json.dumps(rows))
    second, after_metadata = NuScenesLabelIndex(tmp_path).build_observation('scene-0001', 's2')
    for key in first: np.testing.assert_array_equal(first[key], second[key], err_msg=key)
    assert metadata == after_metadata


def test_initial_frame_no_actors_and_scene_tail_masks(tmp_path):
    index = index_fixture(tmp_path)
    initial = index.build_sample('scene-0001', 's0')
    assert initial['observation']['history_mask'].tolist() == [False, False, False, True]
    tail = index.build_sample('scene-0001', 's3')
    assert not tail['target']['ego_target_mask'].any()
    index.annotations['s3'] = []
    empty = index.build_sample('scene-0001', 's3')
    assert not empty['observation']['actor_observed_mask'].any()
    assert not empty['target']['actor_target_mask'].any()
    assert np.isfinite(empty['observation']['actor_history']).all()


def test_bad_sample_time_gap_and_cross_scene_are_rejected(tmp_path):
    index = index_fixture(tmp_path)
    index.samples['s1']['timestamp'] = 1500000
    with pytest.raises(ValueError, match='timestamp|gap'):
        index.build_sample('scene-0001', 's2')


def test_research_shards_are_numeric_and_scene_scoped(tmp_path):
    from reactoccflow.data.research import build_research_labels
    index = index_fixture(tmp_path / 'source')
    manifest = build_research_labels(index, tmp_path / 'output', scenes=['scene-0001'])
    assert manifest['coordinate_frame'] == 'E0'
    assert manifest['num_frames'] == 4
    with np.load(tmp_path / 'output' / manifest['shards'][0]['file'], allow_pickle=False) as shard:
        assert shard['observation.actor_history'].shape == (4, 8, 4, 6)
        assert not any(shard[key].dtype.kind == 'O' for key in shard.files)
    saved = json.loads((tmp_path / 'output/manifest.json').read_text())
    assert saved['frames'][2]['token'] == 's2'


def test_occupancy_source_frame_uses_explicit_ego_or_lidar(tmp_path):
    index = index_fixture(tmp_path)
    np.testing.assert_allclose(index.occupancy_to_e0('s2', 's2', source_frame='ego'), np.eye(4), atol=1e-12)
    transform = index.occupancy_to_e0('s2', 's2', source_frame='lidar')
    np.testing.assert_allclose(transform[:3, 3], [1, 0, 1], atol=1e-12)
    with pytest.raises(ValueError):
        index.occupancy_to_e0('s2', 's2', source_frame='guess')


def test_distance_token_sort_current_metadata_and_missing_history_no_bridging(tmp_path):
    from reactoccflow.data.research import NuScenesLabelIndex
    tables = official_fixture(tmp_path)
    # Actor a at distance sqrt(17), two tied actors at distance1, far actor100.
    for token, x in [('z-tie', 3), ('b-tie', 1), ('far', 102)]:
        tables['instance'].append({'token': token, 'category_token': 'car'})
        tables['sample_annotation'].append({'token': token, 'sample_token': 's2', 'instance_token': token, 'translation': [x, 0, 0], 'rotation': [1, 0, 0, 0], 'size': [2, 5, 1], 'visibility_token': '2', 'num_lidar_pts': 1})
    # Current actor a has a missing intermediate observed annotation, then reappears.
    for row in tables['sample_annotation']:
        if row['token'] == 'actor-a-1': row['num_lidar_pts'] = 0
    tables['sample_annotation'].append({'token': 'a-future', 'sample_token': 's3', 'instance_token': 'actor-a', 'translation': [10, 0, 2], 'rotation': [1, 0, 0, 0], 'size': [20, 40, 10], 'visibility_token': '1', 'num_lidar_pts': 0})
    folder = tmp_path / 'v1.0-trainval'
    for table in ('instance', 'sample_annotation'): (folder / f'{table}.json').write_text(json.dumps(tables[table]))
    sample = NuScenesLabelIndex(tmp_path).build_sample('scene-0001', 's2', max_actors=3)
    assert sample['metadata']['actor_instance_tokens'] == ['b-tie', 'z-tie', 'actor-a']
    assert sample['observation']['actor_observed_mask'][2].tolist() == [False, True, False, True]
    assert not sample['observation']['actor_velocity_mask'][2].any()
    np.testing.assert_array_equal(sample['observation']['actor_size'][2], [4, 2, 1])
    assert sample['target']['actor_target_mask'][2, 0]
    assert sample['target']['actor_existence_mask'][:, 0].all()
    assert not sample['target']['actor_existence_mask'][:, 1:].any()
    assert sample['target']['actor_future_z'][2, 0] == 2


def test_cross_scene_chain_and_missing_samples_do_not_bridge(tmp_path):
    index = index_fixture(tmp_path)
    index.samples['s1']['scene_token'] = 'other'
    with pytest.raises(ValueError, match='cross-scene'):
        index.build_sample('scene-0001', 's2')
    del index.samples['s1']
    with pytest.raises(ValueError, match='Missing sample'):
        index.build_sample('scene-0001', 's2')


def test_streamed_records_and_on_demand_occupancy_warp(tmp_path):
    from reactoccflow.data.research import _json_records
    table = tmp_path / 'tiny.json'
    table.write_text(' [ {"token": "with spaces", "value": 3}, {"token": "two"} ] ')
    assert list(_json_records(table, chunk_size=8)) == [{'token': 'with spaces', 'value': 3}, {'token': 'two'}]
    index = index_fixture(tmp_path / 'source')
    from reactoccflow.geometry import GridSpec
    path = index.root / 'gts/scene-0001/s2'
    path.mkdir(parents=True)
    semantic = np.array([[[4], [17]], [[255], [1]]], dtype=np.uint8)
    np.savez(path / 'labels.npz', semantics=semantic, mask_camera=semantic == 4, mask_lidar=semantic != 255)
    grid = GridSpec(shape=(2, 2, 1), origin=(0, 0, 0), voxel_size=(1, 1, 1))
    result = index.load_occupancy_e0('s2', 's2', source_frame='ego', source_grid=grid, target_grid=grid)
    np.testing.assert_array_equal(result['semantics'], semantic)
    np.testing.assert_array_equal(result['visibility']['mask_camera'], semantic == 4)
    assert result['oracle'] is True


def test_selection_distance_is_planar_e0_even_when_ego_is_tilted(tmp_path):
    from reactoccflow.data.research import NuScenesLabelIndex
    tables = official_fixture(tmp_path)
    # At current ego [2,0,0], pitch +90: world vertical -z becomes E0 +x.
    # Actor a is globally nearby but 10m above; actor near-e0-x is2m E0-away.
    for pose in tables['ego_pose']:
        pose['rotation'] = [np.sqrt(.5), 0, np.sqrt(.5), 0]
    for ann in tables['sample_annotation']:
        if ann['token'] == 'actor-a-2': ann['translation'] = [2, 0, 10]
    tables['instance'].append({'token': 'near-e0-x', 'category_token': 'car'})
    tables['sample_annotation'].append({'token': 'near', 'sample_token': 's2', 'instance_token': 'near-e0-x', 'translation': [2, 2, 0], 'rotation': [1, 0, 0, 0], 'size': [2, 4, 1], 'visibility_token': '4', 'num_lidar_pts': 1})
    for name in ('ego_pose', 'instance', 'sample_annotation'):
        (tmp_path / 'v1.0-trainval' / f'{name}.json').write_text(json.dumps(tables[name]))
    observed, metadata = NuScenesLabelIndex(tmp_path).build_observation('scene-0001', 's2', max_actors=1)
    assert metadata['actor_instance_tokens'] == ['near-e0-x']


def test_official_jitter_edge_preserves_actual_time_and_backward_velocity(tmp_path):
    from reactoccflow.data.research import NuScenesLabelIndex, build_research_labels
    tables = official_fixture(tmp_path)
    # Actual official reciprocal sample-chain edge observed in scene-0994.
    for sample, timestamp in zip(tables['sample'], [0, 500000, 1250945, 1750945]):
        sample['timestamp'] = timestamp
    (tmp_path / 'v1.0-trainval/sample.json').write_text(json.dumps(tables['sample']))
    index = NuScenesLabelIndex(tmp_path)
    observed, _ = index.build_observation('scene-0001', 's2')
    assert index.sample_tokens('scene-0001') == ['s0', 's1', 's2', 's3']
    np.testing.assert_allclose(observed['history_time_s'], [0, -1.250945, -.750945, 0], atol=1e-12)
    # Actor moves global+x1m, i.e. E0-y1m; velocity uses .750945s.
    np.testing.assert_allclose(observed['actor_history'][0, -1, 4:], [0, -1/.750945], atol=1e-6)
    assert observed['actor_velocity_mask'][0, -1]
    manifest = build_research_labels(index, tmp_path / 'labels', ['scene-0001'])
    assert manifest['sample_period_s'] == .5
    assert manifest['max_gap_s'] == .8
    # Explicit stricter policy still rejects this edge rather than interpolating.
    with pytest.raises(ValueError, match='timestamp gap'):
        NuScenesLabelIndex(tmp_path, max_gap_s=.75).build_observation('scene-0001', 's2')


def test_gap_above_explicit_point8_policy_still_rejected(tmp_path):
    index = index_fixture(tmp_path)
    index.samples['s2']['timestamp'] = 1300001
    with pytest.raises(ValueError, match='timestamp gap'):
        index.build_observation('scene-0001', 's2')
