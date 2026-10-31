"""Resumable rank-sharded full E0 research cache (never a training job).

Run from the independent project: python -m scripts.cache_research --help.
Only ordinary NPZ/JSON files are published. Each payload is hashed and renamed
before its metadata commit marker; interrupted pairs are re-encoded on resume.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
import fcntl
import hashlib
import json
import os
from pathlib import Path
import tempfile
import time

# Bound thread pools only in this owned generator process, before NumPy/torch.
for _thread_env in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ[_thread_env] = '4'

import numpy as np

from reactoccflow.data.e0_dataset import (
    CACHE_FORMAT, CACHE_VERSION, payload_sha256, read_record, safe_cache_path,
    validate_sample,
)


def _canonical(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()


def _fingerprint(value):
    return hashlib.sha256(_canonical(value)).hexdigest()


def _atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=f'.{path.name}.', suffix='.tmp', dir=path.parent)
    try:
        with os.fdopen(descriptor, 'wb') as stream:
            stream.write(_canonical(value) + b'\n'); stream.flush(); os.fsync(stream.fileno())
        os.replace(temporary, path)
        directory = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
        try: os.fsync(directory)
        finally: os.close(directory)
    finally:
        if os.path.exists(temporary): os.unlink(temporary)


@contextmanager
def _lock(root, filename):
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    with safe_cache_path(root, filename).open('a') as stream:
        fcntl.flock(stream, fcntl.LOCK_EX)
        try: yield
        finally: fcntl.flock(stream, fcntl.LOCK_UN)


def enumerate_anchors(scene_rows, sample_rows, scene_names):
    """Census official linked keyframes: indices 3 through n-7 inclusive."""
    requested = set(scene_names)
    if len(requested) != len(scene_names):
        raise ValueError('Duplicate split scene names')
    scenes = {row['name']: row for row in scene_rows if row['name'] in requested}
    if set(scenes) != requested:
        raise ValueError('Split scene absent from official scene table')
    selected_tokens = {row['token'] for row in scenes.values()}
    samples = {}
    for row in sample_rows:
        if row['scene_token'] in selected_tokens:
            if row['token'] in samples: raise ValueError('Duplicate official sample token')
            samples[row['token']] = row
    anchors = []
    for name in sorted(requested):
        scene = scenes[name]
        chain, token, previous, last_timestamp = [], scene['first_sample_token'], '', None
        seen = set()
        while token:
            if token in seen or token not in samples: raise ValueError('Broken/cyclic official scene chain')
            row = samples[token]
            if row['scene_token'] != scene['token'] or row['prev'] != previous:
                raise ValueError('Cross-scene or nonreciprocal official scene chain')
            timestamp = row['timestamp']
            if last_timestamp is not None and not 0 < (timestamp - last_timestamp) / 1e6 <= .8:
                raise ValueError('Invalid official scene chain timestamp gap')
            chain.append(row); seen.add(token)
            previous, token, last_timestamp = token, row['next'], timestamp
        if len(chain) != scene['nbr_samples'] or not chain or chain[-1]['token'] != scene['last_sample_token']:
            raise ValueError('Incomplete official scene chain census')
        extras = sum(row['scene_token'] == scene['token'] for row in samples.values())
        if extras != len(chain): raise ValueError('Disconnected official scene chain samples')
        for position in range(3, len(chain) - 6):
            row = chain[position]
            anchors.append({'scene': name, 'token': row['token'], 'timestamp_us': row['timestamp'],
                            'global_index': len(anchors)})
    return anchors


def _record_entry(anchor, settings_sha256):
    scene, token = anchor['scene'], anchor['token']
    for value in (scene, token):
        if not isinstance(value, str) or not value or Path(value).name != value or value in ('.', '..') or '\\' in value:
            raise ValueError('Unsafe anchor identity')
    return {'scene': scene, 'token': token, 'global_index': anchor['global_index'],
            'file': f'{scene}/{token}.npz', 'record_file': f'{scene}/{token}.json',
            'settings_sha256': settings_sha256}


def _existing_record(root, anchor, settings_sha256):
    expected = _record_entry(anchor, settings_sha256)
    try:
        record = json.loads(safe_cache_path(root, expected['record_file']).read_text())
        if any(record.get(key) != value for key, value in expected.items()): return None
        entry = {**expected, 'sha256': record['sha256']}
        # Resume and complete-manifest assembly apply exactly the reader's full
        # numeric/metadata schema, in addition to hashing actual payload bytes.
        read_record(root, entry, settings_sha256, verify_checksum=True)
        return entry
    except (OSError, ValueError, KeyError, TypeError):
        return None


def _write_record(root, anchor, settings_sha256, sample):
    validate_sample(sample)
    if (sample['metadata'].get('scene'), sample['metadata'].get('token')) != (anchor['scene'], anchor['token']):
        raise ValueError('Encoded sample anchor identity mismatch')
    entry = _record_entry(anchor, settings_sha256)
    path = safe_cache_path(root, entry['file'])
    path.parent.mkdir(parents=True, exist_ok=True)
    arrays = {f'{section}.{key}': value for section in ('observation', 'target') for key, value in sample[section].items()}
    descriptor, temporary = tempfile.mkstemp(prefix=f'.{path.name}.', suffix='.tmp', dir=path.parent)
    try:
        with os.fdopen(descriptor, 'wb') as stream:
            np.savez_compressed(stream, **arrays); stream.flush(); os.fsync(stream.fileno())
        entry['sha256'] = payload_sha256(temporary)
        os.replace(temporary, path)
        _atomic_json(safe_cache_path(root, entry['record_file']), {**entry, 'metadata': sample['metadata']})
    finally:
        if os.path.exists(temporary): os.unlink(temporary)
    return entry


def _base_manifest(settings_sha256, anchors, settings):
    return {'version': CACHE_VERSION, 'format': CACHE_FORMAT, 'coordinate_frame': 'E0',
            'oracle': True, 'complete': False, 'settings_sha256': settings_sha256,
            'expected_count': len(anchors), 'sample_count': 0, 'records': [],
            'normalization': settings.get('normalization'), 'group': settings.get('group')}


def _initialize(root, settings, anchors, world_size):
    enriched = {**settings, 'world_size': world_size, 'expected_count': len(anchors),
                'anchor_identity_sha256': _fingerprint(anchors)}
    digest = _fingerprint(enriched)
    with _lock(root, '.settings.lock'):
        path = safe_cache_path(root, 'settings.json')
        if path.exists():
            existing = json.loads(path.read_text())
            if existing.get('settings_sha256') != digest or existing.get('settings') != enriched:
                raise ValueError('Resume settings or anchor identities differ from initialized cache')
        else:
            if (root / 'manifest.json').exists():
                raise ValueError('Cannot initialize settings over an existing manifest')
            _atomic_json(path, {'settings_sha256': digest, 'settings': enriched})
        manifest = safe_cache_path(root, 'manifest.json')
        if not manifest.exists(): _atomic_json(manifest, _base_manifest(digest, anchors, settings))
    return digest


def assemble_manifest(root, settings, anchors, settings_sha256, world_size):
    """Publish complete:true only after every rank commits its exact census."""
    with _lock(root, '.assemble.lock'):
        records, all_complete = [], True
        for rank in range(world_size):
            path = safe_cache_path(root, f'rank-{rank:05d}.json')
            if not path.exists(): all_complete = False; continue
            progress = json.loads(path.read_text())
            if progress.get('settings_sha256') != settings_sha256 or progress.get('rank') != rank:
                raise ValueError('Rank settings/identity mismatch')
            expected = [row for row in anchors if row['global_index'] % world_size == rank]
            rank_records = progress.get('records', [])
            expected_by_key = {(row['scene'], row['token']): row for row in expected}
            seen = set()
            valid_records = []
            for entry in rank_records:
                identity = (entry['scene'], entry['token'])
                if identity not in expected_by_key or identity in seen:
                    raise ValueError('Rank contains duplicate or foreign anchor')
                anchor = expected_by_key[identity]
                verified = _existing_record(root, anchor, settings_sha256)
                if verified != entry:
                    all_complete = False
                    continue
                seen.add(identity); valid_records.append(entry)
            records.extend(valid_records)
            if progress.get('complete') is not True or len(valid_records) != len(expected): all_complete = False
        records.sort(key=lambda row: row['global_index'])
        complete = all_complete and len(records) == len(anchors)
        manifest = _base_manifest(settings_sha256, anchors, settings)
        manifest.update(complete=complete, sample_count=len(records), records=records)
        _atomic_json(safe_cache_path(root, 'manifest.json'), manifest)
        return manifest


def run_cache(root, settings, anchors, encode, *, rank=0, world_size=1, limit=None, assemble_only=False):
    """Generate one rank; ``limit`` counts new encodings, never valid resumes.

    The encoder receives an anchor dict and returns the independent E0 sample.
    This seam permits realistic tiny CPU fixtures without a GPU/checkpoint.
    """
    root = Path(root).resolve()
    if world_size < 1 or not 0 <= rank < world_size or (limit is not None and limit < 0):
        raise ValueError('Invalid rank/world-size/new-record limit')
    if [row['global_index'] for row in anchors] != list(range(len(anchors))):
        raise ValueError('Anchor indices must be globally ordered contiguous integers')
    identities = [(row['scene'], row['token']) for row in anchors]
    if len(set(identities)) != len(identities): raise ValueError('Duplicate anchor identities')
    digest = _initialize(root, settings, anchors, world_size)
    if assemble_only: return assemble_manifest(root, settings, anchors, digest, world_size)
    assigned = [row for row in anchors if row['global_index'] % world_size == rank]
    with _lock(root, f'.rank-{rank:05d}.lock'):
        started, new, reused, records = time.monotonic(), 0, 0, []
        progress = {'version': 1, 'rank': rank, 'world_size': world_size,
                    'settings_sha256': digest, 'expected_count': len(assigned), 'complete': False,
                    'records': records, 'new_records': 0, 'reused_records': 0}
        progress_path = safe_cache_path(root, f'rank-{rank:05d}.json')
        _atomic_json(progress_path, progress)
        # Immediately revoke a previously complete view while this rank is audited.
        with _lock(root, '.assemble.lock'):
            manifest_path = safe_cache_path(root, 'manifest.json')
            manifest = json.loads(manifest_path.read_text()); manifest['complete'] = False
            _atomic_json(manifest_path, manifest)
        for anchor in assigned:
            entry = _existing_record(root, anchor, digest)
            if entry is None:
                if limit is not None and new >= limit: continue
                entry = _write_record(root, anchor, digest, encode(anchor)); new += 1
            else: reused += 1
            records.append(entry)
            progress.update(new_records=new, reused_records=reused, seconds=time.monotonic() - started,
                            completed_count=len(records))
            if len(records) % 10 == 0:
                _atomic_json(progress_path, progress)
                print(json.dumps({'stage': 'encode', 'rank': rank, 'done': len(records), 'assigned': len(assigned),
                                  'new': new, 'reused': reused, 'seconds': progress['seconds']}), flush=True)
        progress.update(complete=len(records) == len(assigned), seconds=time.monotonic() - started)
        _atomic_json(progress_path, progress)
    manifest = assemble_manifest(root, settings, anchors, digest, world_size)
    print(json.dumps({'stage': 'rank_finished', 'rank': rank, 'new': new, 'reused': reused,
                      'rank_complete': progress['complete'], 'complete': manifest['complete'],
                      'sample_count': manifest['sample_count'], 'expected_count': len(anchors),
                      'seconds': progress['seconds']}), flush=True)
    return manifest


def _verify_sources(root, version, source_manifest):
    expected_tables = {'scene.json', 'sample.json', 'sensor.json', 'calibrated_sensor.json',
                       'sample_data.json', 'ego_pose.json', 'category.json', 'instance.json', 'sample_annotation.json'}
    manifest = json.loads(source_manifest.read_text())
    if set(manifest.get('tables', {})) != expected_tables:
        raise ValueError('Source manifest must bind all nine official source tables')
    verified = {}
    for table in sorted(expected_tables):
        path = root / version / table
        entry = manifest['tables'][table]
        if path.stat().st_size != entry['bytes'] or payload_sha256(path) != entry['sha256']:
            raise ValueError(f'Official source table checksum mismatch: {table}')
        verified[table] = entry
    return verified


def _runtime_settings(args, split, labels, normalization, source_tables, anchors):
    import torch
    project = Path(__file__).resolve().parents[1]
    owned = [Path(__file__), project / 'reactoccflow/data/e0_cache.py', project / 'reactoccflow/data/e0_dataset.py',
             project / 'reactoccflow/data/research.py', project / 'reactoccflow/models/codec.py',
             project / 'reactoccflow/models/_utils.py', project / 'reactoccflow/provenance.py']
    owned += sorted((project / 'reactoccflow/geometry').rglob('*.py'))
    owned += sorted((project / 'reactoccflow/models/modules').rglob('*.py'))
    device = torch.device(args.device)
    if device.type == 'cuda':
        if not torch.cuda.is_available(): raise RuntimeError('Requested CUDA is unavailable')
        gpu_model = torch.cuda.get_device_name(device)
        capability = list(torch.cuda.get_device_capability(device))
    elif device.type == 'cpu': gpu_model, capability = 'cpu', None
    else: raise ValueError('Cache encoding supports cpu or explicit cuda device')
    return {
        'version': 1, 'group': args.group, 'metadata_version': args.version,
        'coordinate_frame': 'E0', 'history': 4, 'history_slots': 6, 'hist_last': 4, 'future': 6,
        'anchor_selection': 'official_scene_chain_i=3..n-7_inclusive;sorted_scene_name',
        'source_frame': 'ego', 'visibility_key': 'mask_lidar', 'posterior': 'mean',
        'include_future_semantics': True, 'dtype': 'float32', 'gpu_model': gpu_model,
        'gpu_capability': capability, 'torch_version': torch.__version__, 'cuda_version': torch.version.cuda,
        'numpy_version': np.__version__, 'intraop_threads': torch.get_num_threads(),
        'allow_tf32_matmul': torch.backends.cuda.matmul.allow_tf32,
        'allow_tf32_cudnn': torch.backends.cudnn.allow_tf32,
        'cudnn_benchmark': torch.backends.cudnn.benchmark,
        'source_tables': source_tables, 'source_manifest_sha256': payload_sha256(args.source_manifest),
        'split_sha256': payload_sha256(args.split), 'label_manifest_sha256': payload_sha256(args.label_manifest),
        'normalization_sha256': payload_sha256(args.normalization),
        'checkpoint_sha256': payload_sha256(args.checkpoint),
        'normalization': normalization,
        'generation_code_sha256': {str(path.relative_to(project)): payload_sha256(path) for path in owned},
        'pretraining_caveat': split.get('pretraining_caveat'),
        'label_grid': labels['grid'], 'source_split_groups': split['groups'],
    }


def _prepare(args):
    split = json.loads(args.split.read_text())
    if args.group not in split['groups']: raise ValueError('Unknown split group')
    groups = split['groups']
    flattened = [name for names in groups.values() for name in names]
    if len(flattened) != len(set(flattened)): raise ValueError('Overlapping split scene groups')
    labels = json.loads(args.label_manifest.read_text())
    normalization = json.loads(args.normalization.read_text())
    if labels.get('group') != args.group or labels.get('coordinate_frame') != 'E0' or labels.get('occupancy_source_frame') != 'ego':
        raise ValueError('Label manifest group/frame convention mismatch')
    split_hash = payload_sha256(args.split)
    # Existing label manifests bind file bytes; normalization binds canonical JSON.
    if labels.get('split_manifest_sha256') != split_hash or normalization.get('split_sha256') != _fingerprint(split):
        raise ValueError('Label/normalization split fingerprint mismatch')
    if args.group == 'train' and normalization.get('source_manifest_sha256') != payload_sha256(args.label_manifest):
        raise ValueError('Normalization was fitted against another training-label manifest')
    if normalization.get('fit_group') != 'train' or normalization.get('coordinate_frame') != 'E0':
        raise ValueError('Normalization must be train-only in E0')
    source_tables = _verify_sources(args.root, args.version, args.source_manifest)
    if labels.get('source_scene_sha256') != source_tables['scene.json']['sha256'] or labels.get('source_sample_sha256') != source_tables['sample.json']['sha256']:
        raise ValueError('Label source-table identity mismatch')
    from reactoccflow.data.research import _json_records
    anchors = enumerate_anchors(_json_records(args.root / args.version / 'scene.json'),
                                _json_records(args.root / args.version / 'sample.json'), groups[args.group])
    frames = {(row['scene'], row['token']): row for row in labels['frames']}
    if len(frames) != len(labels['frames']) or labels['num_frames'] != len(frames):
        raise ValueError('Duplicate/mismatched label frame census')
    if {scene for scene, token in frames} != set(groups[args.group]): raise ValueError('Label/split scene sets disagree')
    for anchor in anchors:
        key = (anchor['scene'], anchor['token'])
        if key not in frames or frames[key]['timestamp_us'] != anchor['timestamp_us']:
            raise ValueError('Anchor missing/misaligned in labels')
    source_occupancy = {}
    for scene, token in sorted(frames):
        relative = f'{scene}/{token}/labels.npz'
        path = safe_cache_path(args.root / 'gts', relative)
        if not path.is_file():
            raise FileNotFoundError(f'Missing source occupancy: {scene}/{token}')
        source_occupancy[relative] = {'bytes': path.stat().st_size, 'sha256': payload_sha256(path)}
    settings = _runtime_settings(args, split, labels, normalization, source_tables, anchors)
    # Bind every source occupancy, including edge frames that are context but
    # never anchors. Recomputed on every invocation before any resume can skip.
    settings['source_occupancy'] = source_occupancy
    settings['source_occupancy_sha256'] = _fingerprint(source_occupancy)
    settings['source_occupancy_count'] = len(source_occupancy)
    return anchors, settings


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=Path('/home/lthpc/data/nuscenes'))
    parser.add_argument('--version', default='v1.0-trainval')
    parser.add_argument('--group', default='train')
    parser.add_argument('--split', type=Path, default=Path('data/research/splits.json'))
    parser.add_argument('--label-manifest', type=Path, default=Path('data/research/train/manifest.json'))
    parser.add_argument('--normalization', type=Path, default=Path('data/research/normalization.json'))
    parser.add_argument('--source-manifest', type=Path, default=Path('data/research/source_manifest.json'))
    parser.add_argument('--checkpoint', type=Path, default=Path('assets/checkpoints/occfm_vae.ckpt'))
    parser.add_argument('--output', type=Path, default=Path('data/research/e0_full/train'))
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--rank', type=int, default=0)
    parser.add_argument('--world-size', type=int, default=2)
    parser.add_argument('--limit', type=int, default=None, help='Maximum NEW records this invocation; existing verified records do not count')
    parser.add_argument('--assemble-only', action='store_true')
    args = parser.parse_args(argv)
    if args.world_size < 1 or not 0 <= args.rank < args.world_size or (args.limit is not None and args.limit < 0):
        parser.error('world-size must be positive, rank in range, and limit nonnegative')
    import torch
    torch.set_num_threads(4)
    anchors, settings = _prepare(args)
    print(json.dumps({'stage': 'census', 'expected_count': len(anchors), 'rank': args.rank,
                      'world_size': args.world_size, 'assigned': sum(a['global_index'] % args.world_size == args.rank for a in anchors)}), flush=True)
    state = {}
    def encode(anchor):
        if not state:
            from reactoccflow.data.research import NuScenesLabelIndex
            from reactoccflow.data.e0_cache import encode_e0_training_sample
            from reactoccflow.models import OccupancyCodec
            scenes = sorted(settings['source_split_groups'][args.group])
            state['index'] = NuScenesLabelIndex(args.root, args.version, scenes=scenes)
            state['codec'] = OccupancyCodec().to(args.device).eval()
            state['codec'].load_pretrained(args.checkpoint)
            state['encode'] = encode_e0_training_sample
            for parameter in state['codec'].parameters(): parameter.requires_grad_(False)
        return state['encode'](state['index'], anchor['scene'], anchor['token'], state['codec'], args.device,
                               history_slots=6, hist_last=4, future=6, source_frame='ego',
                               visibility_key='mask_lidar', include_future_semantics=True)
    manifest = run_cache(args.output, settings, anchors, encode, rank=args.rank, world_size=args.world_size,
                         limit=args.limit, assemble_only=args.assemble_only)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
