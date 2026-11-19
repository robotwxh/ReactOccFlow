"""Independent numeric E0 research-cache reader; raw annotations are unnecessary.

The resumable NPZ/JSON format is project-owned. Reference searches and existing
encoder provenance are recorded in docs/provenance/full_cache_sources.json.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path, PurePosixPath

import numpy as np

from .history_visibility import resolve_history_visibility

CACHE_FORMAT = 'reactoccflow_e0_research_cache'
CACHE_VERSION = 1


def payload_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def safe_cache_path(root, relative):
    """Resolve ordinary cache files, rejecting traversal and linked payloads."""
    if not isinstance(relative, str) or not relative or '\\' in relative:
        raise ValueError('Cache paths must be nonempty relative POSIX paths')
    parts = PurePosixPath(relative)
    if parts.is_absolute() or any(part in ('.', '..') for part in parts.parts):
        raise ValueError('Cache path escapes root')
    root = Path(root).resolve()
    path = root.joinpath(*parts.parts)
    if root not in path.resolve().parents:
        raise ValueError('Cache path escapes root')
    current = root
    for part in parts.parts:
        current = current / part
        if current.is_symlink():
            raise ValueError('Cache symlinks are forbidden')
    if path.exists() and path.is_file() and path.stat().st_nlink != 1:
        raise ValueError('Cache hardlinks are forbidden')
    return path


def validate_sample(sample):
    """Validate the observation/target boundary and aligned full-window arrays."""
    metadata = sample['metadata']
    if metadata.get('version') != 1 or metadata.get('coordinate_frame') != 'E0':
        raise ValueError('Unsupported E0 sample version/coordinate frame')
    if metadata.get('format') != 'reactoccflow_e0_encoded_sample' or metadata.get('source_frame') != 'ego' or metadata.get('oracle') is not True:
        raise ValueError('Unsupported E0 sample metadata schema/source convention')
    if not isinstance(metadata.get('timestamp_us'), int) or isinstance(metadata['timestamp_us'], bool):
        raise ValueError('Sample timestamp_us must be an integer')
    actor_tokens = metadata.get('actor_instance_tokens')
    if not isinstance(actor_tokens, list) or len(actor_tokens) != 8 or not all(isinstance(token, str) for token in actor_tokens):
        raise ValueError('Eight explicit actor instance identities are required')
    present_tokens = [token for token in actor_tokens if token]
    if len(present_tokens) != len(set(present_tokens)):
        raise ValueError('Duplicate actor instance identities')
    if metadata.get('posterior') != 'mean' or metadata.get('latent_scaling') != 'unscaled':
        raise ValueError('E0 cache requires unscaled posterior means')
    resolve_history_visibility(metadata)
    obs, target = sample['observation'], sample['target']
    for section, values in (('observation', obs), ('target', target)):
        for key, array in values.items():
            if not isinstance(array, np.ndarray) or array.dtype.kind not in 'buif':
                raise ValueError(f'Only numeric non-object arrays permitted: {section}.{key}')
            if array.dtype.kind == 'f' and not np.isfinite(array).all():
                raise ValueError(f'Nonfinite cache array: {section}.{key}')
            if section == 'observation' and any(word in key.lower() for word in ('future', 'target')):
                raise ValueError('Future/target keys are forbidden in observation')
    shapes = {
        'observation': {'history_latents': (6, 16, 50, 50), 'history_mask': (6,),
                        'history_time_s': (6,), 'observed_history_time_s': (4,),
                        'actor_history': (8, 4, 6), 'actor_observed_mask': (8, 4),
                        'actor_velocity_mask': (8, 4), 'actor_size': (8, 3),
                        'actor_class': (8,), 'actor_z': (8,), 'actor_visibility': (8,),
                        'ego_history': (4, 6), 'ego_observed_mask': (4,),
                        'ego_velocity_mask': (4,), 'ego_current': (6,),
                        'ego_state': (4,), 'ego_state_mask': (4,)},
        'target': {'future_latents': (6, 16, 50, 50), 'future_latent_mask': (6,),
                   'future_time_s': (6,), 'actor_future': (8, 6, 6),
                   'actor_target_mask': (8, 6), 'actor_future_exists': (8, 6),
                   'actor_existence_mask': (8, 6), 'actor_velocity_mask': (8, 6),
                   'actor_future_z': (8, 6), 'ego_future': (6, 6),
                   'ego_target_mask': (6,)},
    }
    grid_shape = tuple(metadata.get('grid', {}).get('shape', ()))
    if len(grid_shape) != 3 or any(not isinstance(v, int) or v < 1 for v in grid_shape):
        raise ValueError('Missing 3-D semantic grid shape')
    shapes['target'].update(future_known_mask=(6, *grid_shape), future_semantics=(6, *grid_shape))
    for section, expected in shapes.items():
        for key, shape in expected.items():
            if key not in sample[section] or sample[section][key].shape != shape:
                raise ValueError(f'Invalid/missing shape: {section}.{key}, expected {shape}')
    for section in ('observation', 'target'):
        if set(sample[section]) != set(shapes[section]):
            raise ValueError(f'Unknown E0 array schema fields in {section}')
        for key, array in sample[section].items():
            if (key.endswith('_mask') or key == 'actor_future_exists') and array.dtype.kind != 'b':
                raise ValueError(f'Validity mask must be boolean: {key}')
    if obs['history_mask'].tolist() != [False, False, True, True, True, True]:
        raise ValueError('Full E0 window requires four observed history frames in six slots')
    if not target['future_latent_mask'].all() or not target['ego_target_mask'].all():
        raise ValueError('Full E0 window requires all six future frames')
    if not obs['ego_observed_mask'].all():
        raise ValueError('Full E0 window requires all four ego history frames')
    if not (np.diff(obs['history_time_s'][2:]) > 0).all() or obs['history_time_s'][-1] != 0:
        raise ValueError('History times must increase to current time zero')
    if not np.array_equal(obs['history_time_s'][2:], obs['observed_history_time_s']):
        raise ValueError('Actor and latent history times disagree')
    times = target['future_time_s']
    if times[0] <= 0 or not (np.diff(times) > 0).all():
        raise ValueError('Future times must be strictly increasing positive timestamps')
    semantics = target['future_semantics']
    if semantics.dtype != np.uint8 or not np.isin(semantics, [*range(18), 255]).all():
        raise ValueError('Future semantics must be uint8 Occ3D18/unknown255')
    if (target['future_known_mask'] & (semantics == 255)).any():
        raise ValueError('Unknown255 cannot be marked known')
    if metadata.get('history_slots') != 6 or metadata.get('hist_last') != 4 or metadata.get('future') != 6:
        raise ValueError('Unsupported E0 temporal window settings')
    for key, length in (('history_sample_tokens', 4), ('history_latent_sample_tokens', 6), ('future_sample_tokens', 6)):
        if len(metadata.get(key, [])) != length:
            raise ValueError(f'Missing aligned token identities: {key}')
    if metadata['history_sample_tokens'][-1] != metadata['token'] or metadata['history_latent_sample_tokens'][2:] != metadata['history_sample_tokens']:
        raise ValueError('History token identities do not match current anchor')
    if any(not token for token in metadata['history_sample_tokens'] + metadata['future_sample_tokens']):
        raise ValueError('Full windows cannot contain absent sample identities')


def read_record(root, entry, settings_sha256, *, verify_checksum=False, history_visibility=None):
    payload = safe_cache_path(root, entry['file'])
    record_path = safe_cache_path(root, entry['record_file'])
    record = json.loads(record_path.read_text())
    for key in ('scene', 'token', 'file', 'record_file', 'sha256', 'settings_sha256', 'global_index'):
        if record.get(key) != entry.get(key):
            raise ValueError(f'Record identity/settings mismatch: {key}')
    if record['settings_sha256'] != settings_sha256:
        raise ValueError('Record settings mismatch')
    if verify_checksum and payload_sha256(payload) != record['sha256']:
        raise ValueError('Cache payload checksum mismatch')
    metadata = record['metadata']
    if history_visibility is not None and resolve_history_visibility(metadata) != history_visibility:
        raise ValueError('Cache record history visibility disagrees with settings/manifest')
    if (metadata.get('scene'), metadata.get('token')) != (entry['scene'], entry['token']):
        raise ValueError('Record metadata identity mismatch')
    sample = {'observation': {}, 'target': {}, 'metadata': metadata}
    with np.load(payload, allow_pickle=False) as arrays:
        for name in arrays.files:
            section, separator, key = name.partition('.')
            if not separator or section not in ('observation', 'target') or not key:
                raise ValueError('NPZ arrays must have explicit observation/target names')
            sample[section][key] = np.array(arrays[name], copy=True)
    validate_sample(sample)
    return sample


class E0ResearchDataset:
    """Load independently usable full-window examples, refusing partial caches.

    ``allow_incomplete=True`` is an explicit debugging option. Checksums can be
    enabled per read without loading the entire cache into RAM. Each item is
    a fresh dict of observation/target NumPy arrays and JSON metadata; no torch,
    raw tables, occupancy sources, checkpoint, or reference repository is read.
    Normalization statistics are available in ``normalization`` and remain a
    downstream training choice; cached actor states retain physical units.
    """
    def __init__(self, root, verify_checksums=False, *, allow_incomplete=False):
        self.root = Path(root).resolve()
        self.verify_checksums = bool(verify_checksums)
        self.manifest = json.loads(safe_cache_path(self.root, 'manifest.json').read_text())
        manifest = self.manifest
        if manifest.get('version') != CACHE_VERSION or manifest.get('format') != CACHE_FORMAT or manifest.get('coordinate_frame') != 'E0':
            raise ValueError('Unsupported research cache format/version/coordinate frame')
        if manifest.get('complete') is not True and not allow_incomplete:
            raise ValueError('Research cache is incomplete; explicit allow_incomplete is for debugging only')
        self.records = manifest['records']
        if manifest.get('sample_count') != len(self.records):
            raise ValueError('Manifest sample count disagrees with records')
        if manifest.get('complete') is True and manifest.get('expected_count') != len(self.records):
            raise ValueError('Complete manifest does not contain the expected anchor census')
        seen, ordinals = set(), set()
        for entry in self.records:
            identity = (entry['scene'], entry['token'])
            if identity in seen or entry['global_index'] in ordinals:
                raise ValueError('Duplicate cache anchor identity/index')
            seen.add(identity); ordinals.add(entry['global_index'])
            if entry.get('settings_sha256') != manifest['settings_sha256']:
                raise ValueError('Manifest record settings mismatch')
            safe_cache_path(self.root, entry['file'])
            safe_cache_path(self.root, entry['record_file'])
        self.normalization = manifest.get('normalization')
        self.history_visibility = (resolve_history_visibility(manifest)
                                   if 'history_visibility' in manifest else None)

    def __len__(self):
        return len(self.records)

    def __getitem__(self, item):
        return read_record(self.root, self.records[item], self.manifest['settings_sha256'],
                           verify_checksum=self.verify_checksums,
                           history_visibility=self.history_visibility)
