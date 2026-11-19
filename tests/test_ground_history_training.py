from copy import deepcopy
import hashlib
import json

import numpy as np
import pytest

from reactoccflow.data.history_recache import reuse_targets, verify_reuse_cache
from reactoccflow.evaluation.architecture_report import MATCHED_SETTINGS, SCORE_FUNCTIONS
from reactoccflow.evaluation.ground_history_comparison import (
    MODE, validate_policy_only_evaluator_change, validate_training_pair,
)
from scripts.cache_research import run_cache
from scripts.run_dense_history_training import verify_completion


def test_observation_recache_preserves_targets_and_rejects_actor_changes():
    from test_e0_dataset import numeric_sample
    previous = numeric_sample()
    observed = {'observation': deepcopy(previous['observation']), 'metadata': deepcopy(previous['metadata'])}
    observed['observation']['history_latents'][:] = 2
    result = reuse_targets(previous, observed, source_record_sha256='f'*64)
    assert result['target'] is previous['target']
    np.testing.assert_array_equal(result['observation']['history_latents'], 2)
    observed['observation']['actor_history'][0, 0, 0] = 3
    with pytest.raises(ValueError, match='Non-latent'):
        reuse_targets(previous, observed, source_record_sha256='f'*64)


def test_reuse_requires_same_source_codec_and_window_order(tmp_path):
    from test_e0_dataset import anchor_fixture, numeric_sample
    fields = ('group', 'source_occupancy_sha256', 'source_manifest_sha256', 'label_manifest_sha256',
              'split_sha256', 'normalization_sha256', 'checkpoint_sha256', 'source_frame', 'history_slots',
              'hist_last', 'future', 'posterior', 'dtype', 'include_future_semantics', 'torch_version',
              'cuda_version', 'allow_tf32_matmul', 'allow_tf32_cudnn', 'cudnn_benchmark')
    settings = {**dict.fromkeys(fields, 'same'), 'history_visibility': 'mask_lidar', 'visibility_key': 'mask_lidar'}
    anchors = anchor_fixture(2)
    run_cache(tmp_path, settings, anchors, lambda a: numeric_sample(a['scene'], a['token']))
    _, audit = verify_reuse_cache(tmp_path, settings, anchors)
    assert audit['windows'] == 2
    for field in ('source_occupancy_sha256', 'checkpoint_sha256'):
        with pytest.raises(ValueError, match=field):
            verify_reuse_cache(tmp_path, {**settings, field: 'changed'}, anchors)
    with pytest.raises(ValueError, match='order'):
        verify_reuse_cache(tmp_path, settings, anchors[::-1])


@pytest.mark.parametrize('mode', ['dense_semantic', 'mask_lidar'])
def test_three_way_training_audit_allows_only_history_mode(mode):
    base = {'history_visibility': mode, 'max_steps': 10000, 'seed': 2026}
    new = {**base, 'history_visibility': MODE}
    assert validate_training_pair(base, new, mode)['differing_field'] == 'history_visibility'
    with pytest.raises(ValueError, match='beyond history'):
        validate_training_pair(base, {**new, 'seed': 42}, mode)


def test_controller_checks_the_declared_new_cache_mode(tmp_path):
    (tmp_path/'manifest.json').write_text(json.dumps({'complete': True, 'history_visibility': MODE,
                                                    'sample_count': 3, 'expected_count': 3}))
    stage = {'name': 'cache', 'cache_root': str(tmp_path), 'expected_count': 3, 'history_visibility': MODE}
    verify_completion(stage)
    with pytest.raises(RuntimeError, match='inconsistent'):
        verify_completion({**stage, 'history_visibility': 'mask_lidar'})


def test_evaluator_audit_accepts_only_the_declared_input_change(tmp_path):
    top = ''.join(f'def {name}():\n    return 0\n\n' for name in SCORE_FUNCTIONS if '.' not in name)
    old = top+'''class OnlineEvaluationData:
    def _labels(self):
        return 0
    def _grid(self):
        return 0
    def __getitem__(self, index):
        history, visible = [], []
        for frame in self.frames:
            transformed = frame
            history.append(transformed['semantics'])
            visible.append(transformed['visibility'])
        future = self.future
        return history, visible, future
'''
    new = old.replace('history, visible = [], []', 'history, visible, bounds = [], [], []')
    new = new.replace("            history.append(transformed['semantics'])\n",
                      "            history.append(transformed['semantics'])\n            bounds.append(transformed['in_bounds'])\n")
    new = new.replace('        future = self.future',
                      "        if self.history_visibility == 'mask_lidar_temporal_ground':\n"
                      '            history, visible = complete_masked_ground_history(history, bounds, visible)\n'
                      '        future = self.future')
    identities = []
    for name, source in [('current', new), ('previous', old)]:
        path = tmp_path/name/'reactoccflow/evaluation/reactflow_generation.py'
        path.parent.mkdir(parents=True); path.write_text(source)
        identities.append({**dict.fromkeys(MATCHED_SETTINGS, None), 'input_sha256': {
            str(path): hashlib.sha256(source.encode()).hexdigest(), '/data/research/example': 'same'}})
    audit = validate_policy_only_evaluator_change(*identities)
    assert len(audit['unchanged_generation_and_score_functions']) == len(SCORE_FUNCTIONS)
    path = next(p for p in identities[0]['input_sha256'] if p.endswith('.py'))
    changed = new.replace('future = self.future', 'future = self.future[1:]')
    from pathlib import Path
    Path(path).write_text(changed)
    identities[0]['input_sha256'][path] = hashlib.sha256(changed.encode()).hexdigest()
    with pytest.raises(ValueError, match='beyond the declared'):
        validate_policy_only_evaluator_change(*identities)
