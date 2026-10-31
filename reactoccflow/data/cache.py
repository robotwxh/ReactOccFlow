"""Sharded adapter of OccFM's nuScenes latent cache/sequence logic.

Copied from forecast/datasets/nuscenes_dataset.py before adaptation. Source
hash/commit: docs/provenance/data_sources.json. The legacy cache is imported
once; training workers only open independent NumPy shards and a JSON index.
This compatibility cache retains each frame's original coordinate convention;
it does not claim to be the E0-aligned research dataset.
"""
from __future__ import annotations

import json
from pathlib import Path, PurePosixPath
import pickle
import tempfile
import shutil

import numpy as np


def parse_label_identity(path) -> tuple[str, str]:
    while isinstance(path, (list, tuple)):
        if len(path) != 1:
            raise ValueError('Expected a single label path per cached frame')
        path = path[0]
    if not isinstance(path, str):
        raise ValueError('Expected a label path string')
    parts = PurePosixPath(path.replace('\\', '/')).parts
    if len(parts) < 3 or parts[-1] != 'labels.npz' or not parts[-3].startswith('scene-'):
        raise ValueError(f'Invalid occupancy label identity: {path}')
    scene, token = parts[-3:-1]
    if token in {'.', '..'}:
        raise ValueError('Invalid sample token')
    return scene, token


def _latent(record) -> np.ndarray:
    array = np.asarray(record['x_sampled'], dtype=np.float32)
    if array.ndim == 4 and array.shape[0] == 1:
        array = array[0]
    if array.ndim != 3 or not np.isfinite(array).all():
        raise ValueError('Invalid latent shape or nonfinite values')
    return array


def convert_legacy_cache(cache_path, output_dir, *, info_path=None, shard_size=512, duplicate_policy='error') -> dict:
    """Convert a trusted local pickle cache, publishing only complete datasets.

    Metadata supplies original within-scene indices: missing cached frames do
    not silently become adjacent. No original label paths survive in the index.
    """
    if info_path is None:
        raise ValueError('info_path is required: provide scene metadata with timestamps to establish chronology')
    if duplicate_policy not in {'error', 'first'}:
        raise ValueError('duplicate_policy must be error or first')
    if shard_size < 1:
        raise ValueError('shard_size must be positive')
    output_dir = Path(output_dir)
    if output_dir.exists():
        raise FileExistsError(output_dir)
    with Path(cache_path).open('rb') as stream:
        records = pickle.load(stream)
    if not records:
        raise ValueError('Empty latent cache')
    by_identity, first_indices, duplicate_records = {}, {}, []
    for record_index, record in enumerate(records):
        identity = parse_label_identity(record['gt_path'])
        if identity in by_identity:
            if duplicate_policy == 'error':
                raise ValueError(f'Duplicate scene/token: {identity}')
            original = by_identity[identity]
            if not np.array_equal(original['gt_trajs'], record['gt_trajs']):
                raise ValueError(f'Duplicate with conflicting trajectory: {identity}')
            duplicate_records.append({'scene': identity[0], 'token': identity[1],
                                      'kept_index': first_indices[identity], 'ignored_index': record_index,
                                      'same_latent': bool(np.array_equal(original['x_sampled'], record['x_sampled']))})
            continue
        by_identity[identity] = record
        first_indices[identity] = record_index
    with Path(info_path).open('rb') as stream:
        infos = pickle.load(stream)['infos']
    order = []
    metadata_identities = set()
    for scene, frames in infos.items():
        previous_stamp = None
        for position, frame in enumerate(frames):
            identity = (scene, frame['token'])
            if identity in metadata_identities:
                raise ValueError(f'Duplicate metadata identity: {identity}')
            metadata_identities.add(identity)
            stamp = frame.get('timestamp')
            if stamp is None:
                raise ValueError(f'Missing metadata timestamp for {identity}')
            try:
                stamp = int(stamp)
            except (TypeError, ValueError, OverflowError) as error:
                raise ValueError(f'Invalid metadata timestamp for {identity}') from error
            if previous_stamp is not None and stamp <= previous_stamp:
                raise ValueError(f'Metadata timestamps must be strictly increasing within scene {scene}')
            previous_stamp = stamp
            if identity in by_identity:
                order.append((identity, position, stamp))
    unknown = by_identity.keys() - metadata_identities
    if unknown:
        raise ValueError(f'Cached tokens absent from metadata: {len(unknown)}')
    shape = _latent(by_identity[order[0][0]]).shape
    # Validate before writing; inconsistent records cannot publish an index.
    for record in records:
        if _latent(record).shape != shape:
            raise ValueError('Latent shape mismatch across frames')
        traj = np.asarray(record['gt_trajs'], dtype=np.float32)
        if traj.size != 2 or not np.isfinite(traj).all():
            raise ValueError('Each compatibility trajectory must contain one finite 2D displacement')
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix='.cache-', dir=output_dir.parent))
    try:
        rows, shards, trajectories = [], [], []
        for start in range(0, len(order), shard_size):
            chunk = order[start:start + shard_size]
            name = f'latents-{len(shards):05d}.npy'
            np.save(staging / name, np.stack([_latent(by_identity[identity]) for identity, _, _ in chunk]))
            shard_id = len(shards)
            shards.append({'file': name, 'count': len(chunk)})
            for local, ((scene, token), position, stamp) in enumerate(chunk):
                rows.append({'scene': scene, 'token': token, 'scene_position': position,
                             'timestamp_us': stamp, 'shard': shard_id, 'row': local})
                trajectories.append(np.asarray(by_identity[(scene, token)]['gt_trajs'], dtype=np.float32).reshape(2))
        np.save(staging / 'trajectory.npy', np.stack(trajectories))
        manifest = {'version': 1, 'coordinate_frame': 'legacy_per_frame_ego',
                    'latent_shape': list(shape), 'dtype': 'float32',
                    'num_frames': len(rows), 'shards': shards, 'frames': rows,
                    'duplicate_policy': duplicate_policy, 'duplicate_records': duplicate_records,
                    'ordering_basis': 'metadata_scene_order_with_strictly_increasing_timestamps',
                    'trajectory_semantics': 'legacy_single_frame_ego_displacement'}
        (staging / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
        staging.rename(output_dir)
    finally:
        if staging.exists():
            shutil.rmtree(staging)
    return manifest


class LatentSequenceDataset:
    """Gap-safe, scene-local history/target windows backed by read-only memmaps."""

    def __init__(self, root, *, history=6, future=6, hist_last=None):
        self.root = Path(root)
        self.manifest = json.loads((self.root / 'manifest.json').read_text())
        if self.manifest.get('version') != 1:
            raise ValueError('Unsupported cache format')
        self.history, self.future = int(history), int(future)
        self.hist_last = self.history if hist_last is None else int(hist_last)
        if self.history < 1 or self.future < 1 or not 0 <= self.hist_last <= self.history:
            raise ValueError('Invalid sequence lengths')
        self.frames = self.manifest['frames']
        self._arrays = {}
        self._trajectory = None
        for shard in self.manifest['shards']:
            name = Path(shard['file'])
            if name.is_absolute() or '..' in name.parts or len(name.parts) != 1:
                raise ValueError('Shard paths must be plain filenames')
        length = self.history + self.future
        self.valid_indices = []
        for start in range(max(0, len(self.frames) - length + 1)):
            window = self.frames[start:start + length]
            scene, pos = window[0]['scene'], window[0]['scene_position']
            if all(f['scene'] == scene and f['scene_position'] == pos + j for j, f in enumerate(window)):
                stamps = [f['timestamp_us'] for f in window]
                if all(t is not None for t in stamps):
                    differences = np.diff(stamps)
                    if not np.all((differences > 0) & (differences <= 750000)):
                        continue
                self.valid_indices.append(start)

    def __len__(self):
        return len(self.valid_indices)

    def __getstate__(self):
        state = dict(self.__dict__)
        state['_arrays'], state['_trajectory'] = {}, None
        return state

    def __getitem__(self, index):
        start = self.valid_indices[index]
        rows = self.frames[start:start + self.history + self.future]
        arrays = []
        for frame in rows:
            sid = frame['shard']
            if sid not in self._arrays:
                self._arrays[sid] = np.load(self.root / self.manifest['shards'][sid]['file'], mmap_mode='r')
            arrays.append(self._arrays[sid][frame['row']])
        latents = np.stack(arrays)
        if self._trajectory is None:
            self._trajectory = np.load(self.root / 'trajectory.npy', mmap_mode='r')
        trajectory = np.array(self._trajectory[start:start + len(rows)], copy=True)
        history_mask = np.arange(self.history) >= self.history - self.hist_last
        latents[:self.history][~history_mask] = 0
        trajectory[:self.history][~history_mask] = 0
        return {'history_latents': latents[:self.history], 'history_mask': history_mask,
                'future_latents': latents[self.history:], 'trajectory': trajectory,
                'scene': rows[0]['scene'], 'tokens': [f['token'] for f in rows]}
