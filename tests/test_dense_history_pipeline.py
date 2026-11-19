import hashlib
import json
from pathlib import Path
import sys

import pytest

from scripts.run_dense_history_training import run_pipeline, verify_source


def config_fixture(tmp_path):
    source = tmp_path/'source'
    source.mkdir()
    (source/'module.py').write_text('# frozen\n')
    manifest = tmp_path/'source.manifest.json'
    manifest.write_text(json.dumps({'files': {'module.py': hashlib.sha256((source/'module.py').read_bytes()).hexdigest()}}))
    return {'source_root': str(source), 'source_manifest': str(manifest), 'run_dir': str(tmp_path),
            'stop_request_file': str(tmp_path/'STOP_REQUESTED.json'), 'environment': {}, 'stages': []}


def test_pipeline_refuses_changed_frozen_source(tmp_path):
    config = config_fixture(tmp_path)
    Path(config['source_root'], 'module.py').write_text('changed')
    with pytest.raises(ValueError, match='checksum'):
        verify_source(config)
    assert not (tmp_path/'pipeline.json').exists()


def test_pipeline_existing_stop_marker_blocks_all_children(tmp_path):
    config = config_fixture(tmp_path)
    Path(config['stop_request_file']).write_text('{}')
    config['stages'] = [{'name': 'forbidden', 'commands': [[sys.executable, '-c', "raise RuntimeError('must not launch')"]]}]
    state = run_pipeline(config, poll_seconds=.01)
    assert state['status'] == 'paused_by_user' and state['stages'] == []


def test_pipeline_failure_never_advances_or_retries(tmp_path):
    config = config_fixture(tmp_path)
    sentinel = tmp_path/'should-not-exist'
    config['stages'] = [
        {'name': 'cache', 'commands': [[sys.executable, '-c', 'raise SystemExit(3)']], 'cache_root': str(tmp_path/'cache')},
        {'name': 'training', 'commands': [[sys.executable, '-c', f'from pathlib import Path; Path({str(sentinel)!r}).touch()']]},
    ]
    state = run_pipeline(config, poll_seconds=.01)
    assert state['status'] == 'failed'
    assert len(state['stages']) == 1 and state['stages'][0]['processes'][0]['exit_code'] == 3
    assert not sentinel.exists()


def test_pipeline_advances_only_after_complete_dense_cache(tmp_path):
    config = config_fixture(tmp_path)
    cache = tmp_path/'cache'
    cache.mkdir()
    (cache/'manifest.json').write_text(json.dumps({'complete': True, 'history_visibility': 'dense_semantic',
                                                  'sample_count': 2, 'expected_count': 2}))
    config['stages'] = [{'name': 'cache', 'commands': [[sys.executable, '-c', 'pass']],
                         'cache_root': str(cache), 'expected_count': 2}]
    assert run_pipeline(config, poll_seconds=.01)['status'] == 'completed'
    with pytest.raises(FileExistsError, match='runtime record'):
        run_pipeline(config, poll_seconds=.01)


def test_pipeline_checks_both_evaluation_ranks_before_aggregation(tmp_path):
    config = config_fixture(tmp_path)
    evaluation = tmp_path/'full'
    for rank in range(2):
        directory = evaluation/f'rank_{rank:03d}'
        directory.mkdir(parents=True)
        (directory/'result.json').write_text(json.dumps({'status': 'completed', 'checkpoint_step': 10000,
                                                        'windows': 2}))
    aggregate = tmp_path/'result.json'
    config['stages'] = [
        {'name': 'evaluation', 'evaluation_root': str(evaluation), 'expected_windows': 4,
         'commands': [[sys.executable, '-c', 'pass']]},
        {'name': 'aggregation', 'completion_json': str(aggregate), 'expected_windows': 4,
         'commands': [[sys.executable, '-c',
                       f'import json; from pathlib import Path; Path({str(aggregate)!r}).write_text(json.dumps(dict(status="completed",windows=4)))']]},
    ]
    state = run_pipeline(config, poll_seconds=.01)
    assert state['status'] == 'completed' and len(state['stages']) == 2
