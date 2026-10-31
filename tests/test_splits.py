import json
import numpy as np
import pytest


def test_scene_groups_deterministic_disjoint_and_eval_untouched():
    from reactoccflow.data.splits import grouped_scene_split
    train = [f'scene-{i:04}' for i in range(20)]
    val = ['val-a', 'val-b']
    split = grouped_scene_split(train, val, seed=2026)
    assert split == grouped_scene_split(list(reversed(train)), list(reversed(val)), seed=2026)
    groups = split['groups']
    assert [len(groups[k]) for k in ('train','tune','calibration','evaluation')] == [16,2,2,2]
    assert set(groups['evaluation']) == set(val)
    assert len(set(sum(groups.values(), []))) == 22
    assert split != grouped_scene_split(train,val,seed=1)


def test_duplicate_overlap_and_unknown_scenes_rejected():
    from reactoccflow.data.splits import grouped_scene_split, validate_scene_membership
    with pytest.raises(ValueError): grouped_scene_split(['a']*10, ['b'])
    with pytest.raises(ValueError): grouped_scene_split([str(i) for i in range(10)], ['0'])
    split = grouped_scene_split([str(i) for i in range(10)], ['val'])
    with pytest.raises(ValueError, match='Unknown'): validate_scene_membership(split, ['missing'])
    assert validate_scene_membership(split, ['val']) == {'val':'evaluation'}


def test_normalization_only_observed_training_not_targets_or_eval(tmp_path):
    from reactoccflow.data.splits import grouped_scene_split, fit_observation_normalization
    split=grouped_scene_split([str(i) for i in range(10)],['val'])
    train=split['groups']['train'][0]
    obs = {'observation.actor_history':np.zeros((1,1,2,6),np.float32),
           'observation.actor_observed_mask':np.array([[[True,False]]]),
           'observation.actor_velocity_mask':np.array([[[True,False]]]),
           'observation.ego_state':np.array([[2,3,4,5]],np.float32),
           'observation.ego_state_mask':np.ones((1,4),bool)}
    obs['observation.actor_history'][0,0,0]=[1,2,0,1,3,4]
    obs['observation.actor_history'][0,0,1]=10000
    np.savez(tmp_path/'train.npz',**obs)
    # Evaluation shards deliberately absent: must never be loaded to fit.
    manifest={'version':1,'coordinate_frame':'E0','shards':[{'scene':train,'file':'train.npz'},{'scene':'val','file':'MUST_NOT_READ.npz'}]}
    (tmp_path/'manifest.json').write_text(json.dumps(manifest))
    stats=fit_observation_normalization(tmp_path,split)
    assert stats['fit_group']=='train'
    assert stats['actor_xy']['count']==[1,1]
    assert stats['actor_xy']['mean']==[1,2]
    assert stats['actor_velocity']['mean']==[3,4]
    assert stats['ego_state']['mean']==[2,3,4,5]
    assert stats['actor_xy']['std']==[1,1]  # zero variance explicitly unit-scaled


@pytest.mark.parametrize('seed', [True, False, 2026.1, 2026.0, '2026', None])
def test_seed_must_be_integral_and_reproducibly_recorded(seed):
    from reactoccflow.data.splits import grouped_scene_split
    with pytest.raises(ValueError, match='seed|Seed'):
        grouped_scene_split([str(i) for i in range(10)], ['val'], seed=seed)


def test_numpy_integer_seed_matches_native_integer():
    from reactoccflow.data.splits import grouped_scene_split
    scenes = [str(i) for i in range(10)]
    assert grouped_scene_split(scenes, ['val'], seed=np.int64(2026)) == grouped_scene_split(scenes, ['val'], seed=2026)


def _normalization_fixture(tmp_path):
    from reactoccflow.data.splits import grouped_scene_split
    split = grouped_scene_split([str(i) for i in range(10)], ['val'])
    observations = {
        'observation.actor_history': np.array([[[[1, 2, 0, 1, 3, 4], [999, 999, 0, 1, 999, 999]]]], dtype=np.float32),
        'observation.actor_observed_mask': np.array([[[True, False]]]),
        'observation.actor_velocity_mask': np.array([[[True, False]]]),
        'observation.ego_state': np.array([[2, 3, 4, 5]], dtype=np.float32),
        'observation.ego_state_mask': np.ones((1, 4), dtype=bool),
    }
    shard = {'scene': split['groups']['train'][0], 'file': 'train.npz'}
    manifest = {'version': 1, 'coordinate_frame': 'E0', 'shards': [shard]}
    np.savez(tmp_path / shard['file'], **observations)
    (tmp_path / 'manifest.json').write_text(json.dumps(manifest))
    return split, observations, manifest


@pytest.mark.parametrize('change', [
    'actor_rank', 'actor_features', 'observed_shape', 'observed_dtype',
    'velocity_shape', 'velocity_dtype', 'velocity_unobserved',
    'ego_rank', 'ego_features', 'ego_mask_shape', 'ego_mask_dtype',
])
def test_normalization_rejects_malformed_observation_schema(tmp_path, change):
    from reactoccflow.data.splits import fit_observation_normalization
    split, obs, _ = _normalization_fixture(tmp_path)
    if change == 'actor_rank': obs['observation.actor_history'] = np.ones((1, 2, 6))
    elif change == 'actor_features': obs['observation.actor_history'] = np.ones((1, 1, 2, 7))
    elif change == 'observed_shape': obs['observation.actor_observed_mask'] = np.array([True, False])
    elif change == 'observed_dtype': obs['observation.actor_observed_mask'] = np.array([[[1, 0]]])
    elif change == 'velocity_shape': obs['observation.actor_velocity_mask'] = np.array(True)
    elif change == 'velocity_dtype': obs['observation.actor_velocity_mask'] = np.array([[[1, 0]]])
    elif change == 'velocity_unobserved': obs['observation.actor_velocity_mask'] = np.array([[[True, True]]])
    elif change == 'ego_rank': obs['observation.ego_state'] = np.array([2, 3, 4, 5])
    elif change == 'ego_features': obs['observation.ego_state'] = np.ones((2, 2))
    elif change == 'ego_mask_shape': obs['observation.ego_state_mask'] = np.array(True)
    elif change == 'ego_mask_dtype': obs['observation.ego_state_mask'] = np.ones((1, 4), dtype=np.uint8)
    np.savez(tmp_path / 'train.npz', **obs)
    with pytest.raises(ValueError, match='actor|velocity|ego|mask'):
        fit_observation_normalization(tmp_path, split)


@pytest.mark.parametrize('duplicate', ['scene', 'file'])
def test_normalization_rejects_duplicate_manifest_scene_or_file(tmp_path, duplicate):
    from reactoccflow.data.splits import fit_observation_normalization
    split, _, manifest = _normalization_fixture(tmp_path)
    first = manifest['shards'][0]
    if duplicate == 'scene':
        manifest['shards'].append({'scene': first['scene'], 'file': 'another.npz'})
    else:
        manifest['shards'].append({'scene': split['groups']['train'][1], 'file': './train.npz'})
    (tmp_path / 'manifest.json').write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match='Duplicate|duplicate'):
        fit_observation_normalization(tmp_path, split)


def test_masked_population_moments_ignore_invalid_nan_and_pool_shards(tmp_path):
    from reactoccflow.data.splits import fit_observation_normalization
    split, obs, manifest = _normalization_fixture(tmp_path)
    obs['observation.actor_history'][0, 0, 1] = np.nan
    np.savez(tmp_path / 'train.npz', **obs)
    second = {key: value.copy() for key, value in obs.items()}
    second['observation.actor_history'][0, 0, 0] = [5, 8, 0, 1, 7, 12]
    second['observation.ego_state'][0] = [6, 7, 8, 9]
    np.savez(tmp_path / 'second.npz', **second)
    manifest['shards'].append({'scene': split['groups']['train'][1], 'file': 'second.npz'})
    (tmp_path / 'manifest.json').write_text(json.dumps(manifest))
    stats = fit_observation_normalization(tmp_path, split)
    assert stats['actor_xy']['count'] == [2, 2]
    np.testing.assert_allclose(stats['actor_xy']['mean'], [3, 5])
    np.testing.assert_allclose(stats['actor_xy']['std'], [2, 3])
    np.testing.assert_allclose(stats['actor_velocity']['mean'], [5, 8])
    np.testing.assert_allclose(stats['actor_velocity']['std'], [2, 4])
    np.testing.assert_allclose(stats['ego_state']['mean'], [4, 5, 6, 7])
    np.testing.assert_allclose(stats['ego_state']['std'], [2, 2, 2, 2])
