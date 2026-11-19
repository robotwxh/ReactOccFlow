"""Supervise the explicitly authorized dense-history cache -> joint training run.

Commands, source identities and outputs are fixed in the run JSON. Failures halt
the pipeline without retry; a run-specific stop marker blocks later stages.
Reference search/provenance: docs/provenance/dense_training_sources_20261116.json.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import time


def atomic_json(path, value):
    path = Path(path)
    temporary = path.with_name(path.name + f'.{os.getpid()}.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')
    temporary.replace(path)


def verify_source(config):
    source = Path(config['source_root'])
    manifest = json.loads(Path(config['source_manifest']).read_text())
    for relative, expected in manifest['files'].items():
        path = source / relative
        if path.is_symlink() or not path.is_file() or path.stat().st_nlink != 1:
            raise ValueError(f'Frozen source must be an ordinary independent file: {relative}')
        if hashlib.sha256(path.read_bytes()).hexdigest() != expected:
            raise ValueError(f'Frozen source checksum mismatch: {relative}')


def stage_progress(stage):
    if 'cache_root' in stage:
        root = Path(stage['cache_root'])
        ranks = []
        for rank in range(2):
            path = root / f'rank-{rank:05d}.json'
            if path.exists():
                row = json.loads(path.read_text())
                ranks.append({key: row.get(key) for key in ('rank', 'completed_count', 'expected_count',
                                                           'seconds', 'complete', 'new_records')})
        return {'cache_ranks': ranks}
    if 'evaluation_root' in stage:
        ranks = []
        for rank in range(2):
            path = Path(stage['evaluation_root']) / f'rank_{rank:03d}/status.json'
            if path.exists():
                ranks.append(json.loads(path.read_text()))
        return {'evaluation_ranks': ranks}
    if 'training_log' not in stage:
        return {}
    path = Path(stage['training_log'])
    if not path.exists():
        return {}
    with path.open('rb') as stream:
        stream.seek(max(0, path.stat().st_size - 65536))
        lines = stream.read().splitlines()
    for line in reversed(lines):
        try:
            row = json.loads(line)
        except (ValueError, UnicodeError):
            continue
        if 'global_step' in row:
            return {'last_training_event': {key: row.get(key) for key in ('event', 'global_step', 'utc',
                                                                        'elapsed_seconds', 'loss')}}
    return {}


def verify_completion(stage):
    if 'completion_json' in stage:
        result = json.loads(Path(stage['completion_json']).read_text())
        if result.get('status') != 'completed':
            raise RuntimeError(f"Missing completed output: {stage['name']}")
        if 'expected_windows' in stage and result.get('windows') != stage['expected_windows']:
            raise RuntimeError('Completed output has the wrong window count')
    elif 'evaluation_root' in stage:
        for rank in range(2):
            path = Path(stage['evaluation_root']) / f'rank_{rank:03d}/result.json'
            result = json.loads(path.read_text())
            if (result.get('status') != 'completed' or result.get('checkpoint_step') != 10000
                    or result.get('windows') != len(range(rank, stage['expected_windows'], 2))):
                raise RuntimeError('Incomplete evaluation rank or wrong checkpoint step')
    elif 'cache_root' in stage:
        manifest = json.loads((Path(stage['cache_root']) / 'manifest.json').read_text())
        if (manifest.get('complete') is not True
                or manifest.get('history_visibility') != stage.get('history_visibility', 'dense_semantic')
                or manifest.get('sample_count') != stage['expected_count']
                or manifest.get('expected_count') != stage['expected_count']):
            raise RuntimeError(f"Incomplete or inconsistent history cache: {stage['name']}")
    else:
        progress = stage_progress(stage).get('last_training_event', {})
        if progress.get('event') != 'finished':
            raise RuntimeError('Training exited without its finished record')


def run_pipeline(config, *, poll_seconds=10):
    verify_source(config)
    run_dir = Path(config['run_dir'])
    stop_file = Path(config['stop_request_file'])
    state_path = run_dir / 'pipeline.json'
    if state_path.exists():
        raise FileExistsError('Pipeline already has a runtime record; inspect it before any explicit restart')
    env = dict(os.environ)
    for key in ('PYTHONPATH', 'PYTHONHOME'):
        env.pop(key, None)
    env.update(config['environment'])
    state = {'status': 'starting', 'controller_pid': os.getpid(), 'stages': [],
             'started_at': datetime.now(timezone.utc).isoformat(), 'source_manifest': config['source_manifest']}

    def request_stop(signum, frame):
        atomic_json(stop_file, {'reason': 'controller_signal', 'signal': signum,
                               'utc': datetime.now(timezone.utc).isoformat()})

    previous_handlers = {sig: signal.signal(sig, request_stop) for sig in (signal.SIGTERM, signal.SIGINT)}
    try:
        for stage in config['stages']:
            if stop_file.exists():
                state.update(status='paused_by_user', current_stage=None)
                break
            verify_source(config)
            entry = {'name': stage['name'], 'status': 'running', 'started_at': datetime.now(timezone.utc).isoformat(),
                     'commands': stage['commands'], 'processes': []}
            state['stages'].append(entry)
            state.update(status='running', current_stage=stage['name'])
            children, logs = [], []
            for index, command in enumerate(stage['commands']):
                log_path = run_dir / f"{stage['name']}.rank-{index}.log"
                stream = log_path.open('xb')
                logs.append(stream)
                child = subprocess.Popen(command, cwd=config['source_root'], env=env, stdin=subprocess.DEVNULL,
                                         stdout=stream, stderr=subprocess.STDOUT, start_new_session=True)
                children.append(child)
                entry['processes'].append({'pid': child.pid, 'log': str(log_path), 'exit_code': None})
            atomic_json(state_path, state)
            try:
                while any(child.poll() is None for child in children):
                    if stop_file.exists() and 'training_log' not in stage:
                        for child in children:
                            if child.poll() is None:
                                child.terminate()
                    for process, child in zip(entry['processes'], children):
                        process['exit_code'] = child.poll()
                    entry.update(stage_progress(stage))
                    state['heartbeat_utc'] = datetime.now(timezone.utc).isoformat()
                    atomic_json(state_path, state)
                    time.sleep(poll_seconds)
                for process, child in zip(entry['processes'], children):
                    process['exit_code'] = child.wait()
            finally:
                for stream in logs:
                    stream.close()
            entry.update(stage_progress(stage), finished_at=datetime.now(timezone.utc).isoformat())
            if stop_file.exists():
                entry['status'] = 'paused_by_user'
                state.update(status='paused_by_user', current_stage=None)
                break
            if any(process['exit_code'] != 0 for process in entry['processes']):
                entry['status'] = 'failed'
                state.update(status='failed', current_stage=stage['name'])
                break
            verify_completion(stage)
            entry['status'] = 'completed'
            atomic_json(state_path, state)
        else:
            state.update(status='completed', current_stage=None)
        atomic_json(state_path, state)
        return state
    except Exception as error:
        state.update(status='failed', error=f'{type(error).__name__}: {error}')
        atomic_json(state_path, state)
        raise
    finally:
        for sig, handler in previous_handlers.items():
            signal.signal(sig, handler)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, required=True)
    args = parser.parse_args(argv)
    state = run_pipeline(json.loads(args.config.read_text()))
    print(json.dumps({'status': state['status'], 'stage': state.get('current_stage')}), flush=True)
    return 1 if state['status'] == 'failed' else 0


if __name__ == '__main__':
    raise SystemExit(main())
