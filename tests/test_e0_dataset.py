"""Research cache boundaries, restart safety, and complete-only publication."""
from copy import deepcopy
import hashlib
import json
from pathlib import Path

import numpy as np
import pytest


def numeric_sample(scene='scene-a', token='s3'):
    return {
        'observation': {
            'history_latents': np.zeros((6, 16, 50, 50), np.float32),
            'history_mask': np.array([False, False, True, True, True, True]),
            'history_time_s': np.array([0, 0, -1.5, -1, -.5, 0.]),
            'observed_history_time_s': np.array([-1.5, -1, -.5, 0.]),
            'actor_history': np.zeros((8, 4, 6), np.float32),
            'actor_observed_mask': np.zeros((8, 4), bool),
            'actor_velocity_mask': np.zeros((8, 4), bool),
            'actor_size': np.zeros((8, 3), np.float32),
            'actor_class': np.full(8, -1, np.int16),
            'actor_z': np.zeros(8, np.float32),
            'actor_visibility': np.zeros(8, np.int8),
            'ego_velocity_mask': np.ones(4, bool),
            'ego_current': np.array([0, 0, 0, 1, 0, 0], np.float32),
            'ego_state': np.zeros(4, np.float32),
            'ego_state_mask': np.ones(4, bool),
            'ego_history': np.zeros((4, 6), np.float32),
            'ego_observed_mask': np.ones(4, bool),
        },
        'target': {
            'future_latents': np.ones((6, 16, 50, 50), np.float32),
            'future_latent_mask': np.ones(6, bool),
            'future_time_s': np.arange(1, 7, dtype=np.float64) / 2,
            'future_known_mask': np.ones((6, 2, 2, 1), bool),
            'future_semantics': np.full((6, 2, 2, 1), 4, np.uint8),
            'actor_future': np.zeros((8, 6, 6), np.float32),
            'actor_target_mask': np.zeros((8, 6), bool),
            'actor_future_exists': np.zeros((8, 6), bool),
            'actor_existence_mask': np.zeros((8, 6), bool),
            'actor_velocity_mask': np.zeros((8, 6), bool),
            'actor_future_z': np.zeros((8, 6), np.float32),
            'ego_future': np.zeros((6, 6), np.float32),
            'ego_target_mask': np.ones(6, bool),
        },
        'metadata': {
            'version': 1, 'format': 'reactoccflow_e0_encoded_sample',
            'scene': scene, 'token': token, 'coordinate_frame': 'E0',
            'oracle': True, 'source_frame': 'ego', 'actor_instance_tokens': [''] * 8,
            'posterior': 'mean', 'latent_scaling': 'unscaled',
            'input_visibility_key': 'mask_lidar',
            'input_visibility_policy': 'selected_sensor_mask & in_bounds & semantics!=255',
            'history_slots': 6, 'hist_last': 4, 'future': 6,
            'grid': {'shape': [2, 2, 1]}, 'timestamp_us': 1000000,
            'history_sample_tokens': ['s0', 's1', 's2', token],
            'history_latent_sample_tokens': ['', '', 's0', 's1', 's2', token],
            'future_sample_tokens': ['f0', 'f1', 'f2', 'f3', 'f4', 'f5'],
        },
    }


def write_fixture(root, *, complete=True, sample=None):
    sample = numeric_sample() if sample is None else sample
    root.mkdir(exist_ok=True)
    filename = 'scene-a/s3.npz'
    (root / 'scene-a').mkdir()
    np.savez_compressed(root / filename, **{f'{section}.{key}': value for section in ('observation', 'target') for key, value in sample[section].items()})
    digest = hashlib.sha256((root / filename).read_bytes()).hexdigest()
    entry = {'scene': 'scene-a', 'token': 's3', 'global_index': 0, 'file': filename,
             'record_file': 'scene-a/s3.json', 'sha256': digest, 'settings_sha256': 'a' * 64}
    (root / entry['record_file']).write_text(json.dumps({**entry, 'metadata': sample['metadata']}))
    manifest = {'version': 1, 'format': 'reactoccflow_e0_research_cache', 'coordinate_frame': 'E0',
                'complete': complete, 'settings_sha256': 'a' * 64, 'sample_count': 1,
                'expected_count': 1, 'records': [entry]}
    (root / 'manifest.json').write_text(json.dumps(manifest))
    return manifest


def test_loader_returns_independent_numeric_observations_and_targets(tmp_path):
    from reactoccflow.data.e0_dataset import E0ResearchDataset
    write_fixture(tmp_path)
    dataset = E0ResearchDataset(tmp_path, verify_checksums=True)
    assert len(dataset) == 1
    sample = dataset[0]
    assert sample['metadata']['scene'] == 'scene-a'
    assert sample['target']['future_semantics'].dtype == np.uint8
    assert sample['observation']['history_mask'].tolist() == [False, False, True, True, True, True]
    assert all('future' not in key for key in sample['observation'])
    assert all(value.dtype.kind in 'buif' for section in ('observation', 'target') for value in sample[section].values())
    sample['observation']['history_latents'].fill(7)
    assert not dataset[0]['observation']['history_latents'].any()


def test_loader_rejects_incomplete_unless_explicitly_requested(tmp_path):
    from reactoccflow.data.e0_dataset import E0ResearchDataset
    write_fixture(tmp_path, complete=False)
    with pytest.raises(ValueError, match='incomplete'):
        E0ResearchDataset(tmp_path)
    assert len(E0ResearchDataset(tmp_path, allow_incomplete=True)) == 1


@pytest.mark.parametrize('mutation', ['duplicate', 'escape', 'absolute', 'count', 'frame', 'version'])
def test_loader_rejects_invalid_manifest(tmp_path, mutation):
    from reactoccflow.data.e0_dataset import E0ResearchDataset
    manifest = write_fixture(tmp_path)
    if mutation == 'duplicate': manifest['records'].append(deepcopy(manifest['records'][0]))
    if mutation == 'escape': manifest['records'][0]['file'] = '../outside.npz'
    if mutation == 'absolute': manifest['records'][0]['record_file'] = '/tmp/outside.json'
    if mutation == 'count': manifest['expected_count'] = 2
    if mutation == 'frame': manifest['coordinate_frame'] = 'future_ego'
    if mutation == 'version': manifest['version'] = 999
    (tmp_path / 'manifest.json').write_text(json.dumps(manifest))
    with pytest.raises(ValueError): E0ResearchDataset(tmp_path)


@pytest.mark.parametrize('mutation', ['future_input', 'object', 'wrong_shape', 'invalid_mask', 'identity', 'settings', 'known_unknown', 'nonfinite'])
def test_loader_rejects_corrupt_record_contract(tmp_path, mutation):
    from reactoccflow.data.e0_dataset import E0ResearchDataset
    sample = numeric_sample()
    if mutation == 'future_input': sample['observation']['future_time_s'] = np.zeros(6)
    if mutation == 'object': sample['target']['bad'] = np.array([{}], dtype=object)
    if mutation == 'wrong_shape': sample['target']['future_latents'] = np.zeros((5, 16, 50, 50))
    if mutation == 'invalid_mask': sample['target']['future_latent_mask'][3] = False
    if mutation == 'identity': sample['metadata']['token'] = 'wrong'
    if mutation == 'settings': sample['metadata']['coordinate_frame'] = 'other'
    if mutation == 'known_unknown': sample['target']['future_semantics'][0, 0, 0, 0] = 255
    if mutation == 'nonfinite': sample['observation']['history_latents'][0, 0, 0, 0] = np.nan
    write_fixture(tmp_path, sample=sample)
    with pytest.raises(ValueError): E0ResearchDataset(tmp_path)[0]


def test_checksum_detects_altered_valid_numpy_payload(tmp_path):
    from reactoccflow.data.e0_dataset import E0ResearchDataset
    write_fixture(tmp_path)
    with (tmp_path / 'scene-a/s3.npz').open('ab') as stream: stream.write(b'changed')
    with pytest.raises(ValueError, match='checksum'):
        E0ResearchDataset(tmp_path, verify_checksums=True)[0]


@pytest.mark.parametrize('field,value', [('history_visibility', 'dense_semantic'),
                                       ('input_visibility_policy', 'in_bounds & semantics!=255'),
                                       ('input_visibility_key', None)])
def test_loader_rejects_contradictory_history_metadata(tmp_path, field, value):
    from reactoccflow.data.e0_dataset import E0ResearchDataset
    sample = numeric_sample()
    sample['metadata'][field] = value
    write_fixture(tmp_path, sample=sample)
    with pytest.raises(ValueError, match='disagrees'):
        E0ResearchDataset(tmp_path)[0]


def anchor_fixture(count=4):
    return [{'scene': 'scene-a', 'token': f'a{i}', 'global_index': i} for i in range(count)]


def test_two_rank_generation_resumes_and_only_publishes_complete_after_all_ranks(tmp_path):
    from scripts.cache_research import run_cache
    from reactoccflow.data.e0_dataset import E0ResearchDataset
    anchors, calls = anchor_fixture(), []
    def encode(anchor):
        calls.append(anchor['token'])
        return numeric_sample(anchor['scene'], anchor['token'])
    settings = {'world_size': 2, 'fixture': True}
    first = run_cache(tmp_path, settings, anchors, encode, rank=0, world_size=2, limit=1)
    assert first['complete'] is False and calls == ['a0']
    with pytest.raises(ValueError, match='incomplete'): E0ResearchDataset(tmp_path)
    run_cache(tmp_path, settings, anchors, encode, rank=1, world_size=2)
    assert calls == ['a0', 'a1', 'a3']
    final = run_cache(tmp_path, settings, anchors, encode, rank=0, world_size=2)
    assert calls == ['a0', 'a1', 'a3', 'a2']
    assert final['complete'] and len(E0ResearchDataset(tmp_path, verify_checksums=True)) == 4
    assert [row['token'] for row in final['records']] == ['a0', 'a1', 'a2', 'a3']
    run_cache(tmp_path, settings, anchors, encode, rank=0, world_size=2)
    assert len(calls) == 4


def test_cache_rejects_mixed_input_modes_and_cross_mode_resume(tmp_path):
    from scripts.cache_research import run_cache
    from reactoccflow.data.e0_dataset import E0ResearchDataset
    from reactoccflow.data.history_visibility import visibility_metadata
    settings = {'history_visibility': 'dense_semantic', 'visibility_key': None}
    anchors = anchor_fixture(1)
    def dense(anchor):
        sample = numeric_sample(anchor['scene'], anchor['token'])
        sample['metadata'].update(visibility_metadata(None))
        return sample
    def masked(anchor):
        return numeric_sample(anchor['scene'], anchor['token'])
    with pytest.raises(ValueError, match='history visibility'):
        run_cache(tmp_path, settings, anchors, masked)
    result = run_cache(tmp_path, settings, anchors, dense)
    assert result['history_visibility'] == 'dense_semantic'
    assert E0ResearchDataset(tmp_path)[0]['metadata']['input_visibility_key'] is None
    with pytest.raises(ValueError, match='settings'):
        run_cache(tmp_path, {'history_visibility': 'mask_lidar', 'visibility_key': 'mask_lidar'}, anchors, masked)
    record_path = tmp_path / 'scene-a/a0.json'
    record = json.loads(record_path.read_text())
    record['metadata'].update(visibility_metadata('mask_lidar'))
    record_path.write_text(json.dumps(record))
    with pytest.raises(ValueError, match='history visibility'):
        E0ResearchDataset(tmp_path)[0]
    # A corrupted but otherwise schema-valid mode is not accepted as a valid resume.
    calls = []
    def reencode(anchor):
        calls.append(anchor['token'])
        return dense(anchor)
    run_cache(tmp_path, settings, anchors, reencode)
    assert calls == ['a0']


@pytest.mark.parametrize('mode', [None, 'dense_semantic', 'mask_lidar', 'mask_lidar_temporal_ground'])
def test_cache_cli_selects_mode_specific_group_directory_without_preprocessing(tmp_path, monkeypatch, mode):
    import scripts.cache_research as cache
    class StopPreflight(Exception):
        pass
    received = []
    def inspect(args):
        received.append(args)
        raise StopPreflight
    monkeypatch.setattr(cache, '_prepare', inspect)
    with pytest.raises(StopPreflight):
        cache.main(['--group', 'tune']+([] if mode is None else ['--history-visibility', mode]))
    args = received[0]
    mode = 'mask_lidar' if mode is None else mode
    assert args.history_visibility == mode
    name = {'dense_semantic': 'e0_dense_history', 'mask_lidar': 'e0_full',
            'mask_lidar_temporal_ground': 'e0_mask_lidar_temporal_ground'}[mode]
    assert args.output == Path('data/research') / name / 'tune'


def test_completed_ground_cache_cannot_be_reused_as_plain_mask_history(tmp_path):
    from scripts.cache_research import run_cache
    from reactoccflow.data.e0_dataset import E0ResearchDataset
    from reactoccflow.data.history_visibility import visibility_metadata
    mode = 'mask_lidar_temporal_ground'
    policy = visibility_metadata('mask_lidar', history_visibility=mode)
    settings = {'history_visibility': mode, 'visibility_key': 'mask_lidar',
                'input_visibility_policy': policy['input_visibility_policy']}
    def encode(anchor):
        sample = numeric_sample(anchor['scene'], anchor['token'])
        sample['metadata'].update(policy)
        return sample
    result = run_cache(tmp_path, settings, anchor_fixture(1), encode)
    assert result['history_visibility'] == mode
    assert E0ResearchDataset(tmp_path)[0]['metadata']['history_visibility'] == mode
    with pytest.raises(ValueError, match='settings|identity|different'):
        run_cache(tmp_path, {'history_visibility': 'mask_lidar', 'visibility_key': 'mask_lidar'},
                  anchor_fixture(1), encode)


def test_resume_reencodes_only_missing_or_hash_corrupt_records_and_rejects_settings_changes(tmp_path):
    from scripts.cache_research import run_cache
    anchors, calls = anchor_fixture(3), []
    def encode(anchor):
        calls.append(anchor['token'])
        return numeric_sample(anchor['scene'], anchor['token'])
    settings = {'world_size': 1, 'normalization_hash': 'one'}
    run_cache(tmp_path, settings, anchors, encode, rank=0, world_size=1)
    (tmp_path / 'scene-a/a0.json').unlink()
    with (tmp_path / 'scene-a/a1.npz').open('ab') as stream: stream.write(b'corrupt')
    run_cache(tmp_path, settings, anchors, encode, rank=0, world_size=1)
    assert calls == ['a0', 'a1', 'a2', 'a0', 'a1']
    with pytest.raises(ValueError, match='settings'):
        run_cache(tmp_path, {**settings, 'normalization_hash': 'two'}, anchors, encode, rank=0, world_size=1)
    with pytest.raises(ValueError, match='anchor'):
        run_cache(tmp_path, settings, anchors[:-1], encode, rank=0, world_size=1)


def test_anchor_census_uses_scene_chains_instead_of_table_order():
    from scripts.cache_research import enumerate_anchors
    scenes = [{'name': 'scene-a', 'token': 'scene', 'first_sample_token': 's0', 'last_sample_token': 's11', 'nbr_samples': 12}]
    samples = [{'token': f's{i}', 'scene_token': 'scene', 'timestamp': i * 500000,
                'prev': f's{i-1}' if i else '', 'next': f's{i+1}' if i < 11 else ''} for i in reversed(range(12))]
    anchors = enumerate_anchors(scenes, samples, ['scene-a'])
    assert [row['token'] for row in anchors] == ['s3', 's4', 's5']
    assert [row['global_index'] for row in anchors] == [0, 1, 2]
    samples[0]['prev'] = 's0'
    with pytest.raises(ValueError, match='chain'): enumerate_anchors(scenes, samples, ['scene-a'])


def test_uncommitted_payload_after_interruption_is_reencoded(tmp_path, monkeypatch):
    import scripts.cache_research as cache
    anchors, calls = anchor_fixture(2), []
    def encode(anchor):
        calls.append(anchor['token'])
        return numeric_sample(anchor['scene'], anchor['token'])
    original = cache._atomic_json
    def interrupt_metadata(path, value):
        if Path(path).name == 'a0.json': raise RuntimeError('simulated interruption')
        return original(path, value)
    monkeypatch.setattr(cache, '_atomic_json', interrupt_metadata)
    with pytest.raises(RuntimeError, match='interruption'):
        cache.run_cache(tmp_path, {}, anchors, encode)
    assert (tmp_path / 'scene-a/a0.npz').exists()
    assert not (tmp_path / 'scene-a/a0.json').exists()
    assert not json.loads((tmp_path / 'manifest.json').read_text())['complete']
    monkeypatch.setattr(cache, '_atomic_json', original)
    complete = cache.run_cache(tmp_path, {}, anchors, encode)
    assert complete['complete'] and calls == ['a0', 'a0', 'a1']
    assert not list(tmp_path.rglob('*.tmp'))


def _parallel_cache_worker(root, rank):
    from scripts.cache_research import run_cache
    def encode(anchor): return numeric_sample(anchor['scene'], anchor['token'])
    run_cache(root, {'fixture': 'parallel'}, anchor_fixture(6), encode, rank=rank, world_size=2)


def test_concurrent_rank_initialization_and_atomic_publication(tmp_path):
    import multiprocessing
    from reactoccflow.data.e0_dataset import E0ResearchDataset
    context = multiprocessing.get_context('fork')
    workers = [context.Process(target=_parallel_cache_worker, args=(str(tmp_path), rank)) for rank in range(2)]
    for worker in workers: worker.start()
    for worker in workers:
        worker.join(timeout=15)
        assert worker.exitcode == 0
    dataset = E0ResearchDataset(tmp_path, verify_checksums=True)
    assert len(dataset) == 6
    assert [dataset[i]['metadata']['token'] for i in range(6)] == ['a0', 'a1', 'a2', 'a3', 'a4', 'a5']
    assert not list(tmp_path.rglob('*.tmp'))


def test_loader_refuses_symlinks_and_hardlinks(tmp_path):
    from reactoccflow.data.e0_dataset import E0ResearchDataset
    write_fixture(tmp_path)
    payload = tmp_path / 'scene-a/s3.npz'
    original = tmp_path / 'original.npz'
    payload.rename(original)
    payload.symlink_to(original)
    with pytest.raises(ValueError, match='symlink'): E0ResearchDataset(tmp_path)
    payload.unlink()
    import os
    os.link(original, payload)
    with pytest.raises(ValueError, match='hardlink'): E0ResearchDataset(tmp_path)


def test_changed_official_table_bytes_cannot_enter_cache(tmp_path):
    from scripts.cache_research import _verify_sources
    table_root = tmp_path / 'v1.0-trainval'
    table_root.mkdir()
    tables = {}
    for table in ('scene', 'sample', 'sensor', 'calibrated_sensor', 'sample_data', 'ego_pose', 'category', 'instance', 'sample_annotation'):
        path = table_root / (table + '.json')
        path.write_text('[]')
        tables[path.name] = {'bytes': 2, 'sha256': hashlib.sha256(b'[]').hexdigest()}
    manifest = tmp_path / 'source.json'
    manifest.write_text(json.dumps({'tables': tables}))
    assert len(_verify_sources(tmp_path, 'v1.0-trainval', manifest)) == 9
    (table_root / 'sample.json').write_text('{}')
    with pytest.raises(ValueError, match='checksum'): _verify_sources(tmp_path, 'v1.0-trainval', manifest)


def test_restarted_audit_preserves_published_partial_progress_after_failure(tmp_path):
    from scripts.cache_research import run_cache
    anchors = anchor_fixture(3)
    def encode(anchor): return numeric_sample(anchor['scene'], anchor['token'])
    run_cache(tmp_path, {}, anchors, encode, limit=1)
    old_manifest = json.loads((tmp_path / 'manifest.json').read_text())
    assert old_manifest['sample_count'] == 1
    def fail(anchor): raise RuntimeError('encoding failed')
    with pytest.raises(RuntimeError, match='encoding failed'):
        run_cache(tmp_path, {}, anchors, fail)
    progress = json.loads((tmp_path / 'rank-00000.json').read_text())
    assert progress['complete'] is False
    assert not json.loads((tmp_path / 'manifest.json').read_text())['complete']


def test_preflight_accepts_existing_canonical_normalization_split_identity(tmp_path):
    """Stats bind canonical split JSON; label manifests bind exact file bytes."""
    import argparse
    from scripts.cache_research import _prepare
    source = tmp_path / 'raw' / 'v1.0-trainval'
    source.mkdir(parents=True)
    scenes = [{'name': 'scene-a', 'token': 'scene', 'first_sample_token': 's0', 'last_sample_token': 's9', 'nbr_samples': 10}]
    samples = [{'token': f's{i}', 'scene_token': 'scene', 'timestamp': 1000000 + i * 500000,
                'prev': f's{i-1}' if i else '', 'next': f's{i+1}' if i < 9 else ''} for i in range(10)]
    source_entries = {}
    for name in ('scene', 'sample', 'sensor', 'calibrated_sensor', 'sample_data', 'ego_pose', 'category', 'instance', 'sample_annotation'):
        path = source / (name + '.json')
        path.write_text(json.dumps(scenes if name == 'scene' else samples if name == 'sample' else []))
        source_entries[path.name] = {'bytes': path.stat().st_size, 'sha256': hashlib.sha256(path.read_bytes()).hexdigest()}
    source_manifest = tmp_path / 'sources.json'
    source_manifest.write_text(json.dumps({'tables': source_entries}))
    split = {'groups': {'train': ['scene-a'], 'evaluation': []}}
    split_path = tmp_path / 'split.json'; split_path.write_text(json.dumps(split, indent=2))
    split_bytes_sha = hashlib.sha256(split_path.read_bytes()).hexdigest()
    labels = {'group': 'train', 'coordinate_frame': 'E0', 'occupancy_source_frame': 'ego',
              'split_manifest_sha256': split_bytes_sha, 'grid': {'shape': [2, 2, 1]},
              'source_scene_sha256': source_entries['scene.json']['sha256'],
              'source_sample_sha256': source_entries['sample.json']['sha256'],
              'frames': [{'scene': 'scene-a', 'token': row['token'], 'timestamp_us': row['timestamp']} for row in samples], 'num_frames': 10}
    label_path = tmp_path / 'labels.json'; label_path.write_text(json.dumps(labels))
    normalization = {'fit_group': 'train', 'coordinate_frame': 'E0',
                     'split_sha256': hashlib.sha256(json.dumps(split, sort_keys=True, separators=(',', ':')).encode()).hexdigest(),
                     'source_manifest_sha256': hashlib.sha256(label_path.read_bytes()).hexdigest()}
    stats_path = tmp_path / 'normalization.json'; stats_path.write_text(json.dumps(normalization))
    for row in samples:
        path = tmp_path / 'raw' / 'gts' / 'scene-a' / row['token'] / 'labels.npz'
        path.parent.mkdir(parents=True); path.touch()
    checkpoint = tmp_path / 'checkpoint.ckpt'; checkpoint.write_bytes(b'fingerprint only, never loaded')
    args = argparse.Namespace(root=tmp_path / 'raw', version='v1.0-trainval', group='train',
                              split=split_path, label_manifest=label_path, normalization=stats_path,
                              source_manifest=source_manifest, checkpoint=checkpoint, device='cpu',
                              history_visibility='mask_lidar')
    anchors, settings = _prepare(args)
    assert len(anchors) == 1 and anchors[0]['token'] == 's3'
    assert settings['split_sha256'] == split_bytes_sha
    assert settings['normalization']['split_sha256'] != split_bytes_sha
    # A raw occupancy edit must change the shared identity before any resume skips.
    from scripts.cache_research import run_cache
    def encode(anchor): return numeric_sample(anchor['scene'], anchor['token'])
    output = tmp_path / 'cache'
    run_cache(output, settings, anchors, encode)
    changed = args.root / 'gts' / 'scene-a' / 's0' / 'labels.npz'
    changed.write_bytes(b'changed raw occupancy payload')
    updated_anchors, updated_settings = _prepare(args)
    assert updated_settings != settings
    with pytest.raises(ValueError, match='settings'):
        run_cache(output, updated_settings, updated_anchors, encode)


@pytest.mark.parametrize('section,key', [
    ('observation', 'actor_velocity_mask'), ('observation', 'actor_size'),
    ('observation', 'actor_class'), ('observation', 'actor_z'),
    ('observation', 'actor_visibility'), ('observation', 'ego_velocity_mask'),
    ('observation', 'ego_current'), ('observation', 'ego_state'),
    ('observation', 'ego_state_mask'), ('target', 'actor_velocity_mask'),
    ('target', 'actor_future_exists'), ('target', 'actor_existence_mask'),
    ('target', 'actor_future_z'),
])
@pytest.mark.parametrize('mutation', ['missing', 'shape'])
def test_complete_training_fields_are_required_with_correct_shapes(tmp_path, section, key, mutation):
    from reactoccflow.data.e0_dataset import E0ResearchDataset
    sample = numeric_sample()
    if mutation == 'missing': del sample[section][key]
    else: sample[section][key] = sample[section][key][:-1]
    write_fixture(tmp_path, sample=sample)
    with pytest.raises(ValueError, match='shape'):
        E0ResearchDataset(tmp_path)[0]


@pytest.mark.parametrize('section,key', [
    ('observation', 'actor_velocity_mask'), ('observation', 'ego_velocity_mask'),
    ('observation', 'ego_state_mask'), ('target', 'actor_velocity_mask'),
    ('target', 'actor_future_exists'), ('target', 'actor_existence_mask'),
])
def test_complete_training_masks_must_be_boolean(tmp_path, section, key):
    from reactoccflow.data.e0_dataset import E0ResearchDataset
    sample = numeric_sample()
    sample[section][key] = sample[section][key].astype(np.int8)
    write_fixture(tmp_path, sample=sample)
    with pytest.raises(ValueError, match='boolean'): E0ResearchDataset(tmp_path)[0]


def test_resume_reencodes_committed_metadata_that_loader_would_reject(tmp_path):
    from scripts.cache_research import run_cache
    from reactoccflow.data.e0_dataset import E0ResearchDataset
    anchors, calls = anchor_fixture(2), []
    def encode(anchor):
        calls.append(anchor['token'])
        return numeric_sample(anchor['scene'], anchor['token'])
    run_cache(tmp_path, {}, anchors, encode)
    path = tmp_path / 'scene-a/a0.json'
    record = json.loads(path.read_text())
    record['metadata']['coordinate_frame'] = 'future_ego'
    path.write_text(json.dumps(record))
    final = run_cache(tmp_path, {}, anchors, encode)
    assert calls == ['a0', 'a1', 'a0']
    assert final['complete']
    assert E0ResearchDataset(tmp_path, verify_checksums=True)[0]['metadata']['coordinate_frame'] == 'E0'


def test_unknown_training_field_requires_explicit_schema_version(tmp_path):
    from reactoccflow.data.e0_dataset import E0ResearchDataset
    sample = numeric_sample()
    sample['observation']['unversioned_input'] = np.zeros(1, np.float32)
    write_fixture(tmp_path, sample=sample)
    with pytest.raises(ValueError, match='schema'): E0ResearchDataset(tmp_path)[0]


@pytest.mark.parametrize('mutation', ['format', 'actors', 'source_frame', 'timestamp'])
def test_committed_metadata_schema_is_validated(tmp_path, mutation):
    from reactoccflow.data.e0_dataset import E0ResearchDataset
    sample = numeric_sample()
    if mutation == 'format': sample['metadata']['format'] = 'other'
    if mutation == 'actors': sample['metadata']['actor_instance_tokens'] = ['same'] * 8
    if mutation == 'source_frame': sample['metadata']['source_frame'] = 'future'
    if mutation == 'timestamp': sample['metadata']['timestamp_us'] = 'not-numeric'
    write_fixture(tmp_path, sample=sample)
    with pytest.raises(ValueError): E0ResearchDataset(tmp_path)[0]
