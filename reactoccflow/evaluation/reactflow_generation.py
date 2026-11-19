"""Independent held-out ReactFlow generation evaluation.

Reuse of the copied OccFM count metric and IR-WM-derived SE(3)/voxel geometry
is recorded in docs/provenance/reactflow_generation_sources_20261104.json.
Checkpoint-bound dual-backbone construction is recorded in
docs/provenance/standard_dit_generation_sources_20261107.json.
The official reference repositories are never imported or executed.

The predictor receives four observed grids/actor states and an explicitly
supplied logged ego action. Future semantics and poses enter the scorer only.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import fcntl
import hashlib
import json
import os
from pathlib import Path
import time

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from reactoccflow.data.e0_dataset import payload_sha256
from reactoccflow.data.loader import shutdown_data_loaders
from reactoccflow.data.history_visibility import (
    HISTORY_VISIBILITIES, complete_masked_ground_history, history_visibility_mask, visibility_key_from_mode,
)
from reactoccflow.evaluation.occupancy import OccupancyAccumulator
from reactoccflow.geometry import GridSpec, inverse_transform, warp_occupancy_nearest


PROTOCOL_VERSION = 3
HORIZON_INDICES = (1, 3, 5)
SCORE_NAMES = ('native_e0_known', 'paper_frame_full', 'paper_frame_covered')


def atomic_json(path, value):
    path = Path(path)
    temporary = path.with_name(path.name + f'.{os.getpid()}.tmp')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False,
                                   default=lambda x: x.tolist()) + '\n')
    temporary.replace(path)


def window_seed(seed, scene, token):
    identity = f'{seed}:{scene}:{token}'.encode()
    return int.from_bytes(hashlib.sha256(identity).digest()[:8], 'big') % (2**63 - 1)


def partition_indices(length, rank, world_size):
    if world_size < 1 or not 0 <= rank < world_size or length < 0:
        raise ValueError('Invalid evaluation partition')
    return list(range(rank, length, world_size))


def checkpoint_model_options(checkpoint):
    """Select the trained architecture; retain the original legacy U default."""
    config = checkpoint['config']
    if config.get('mode') != 'joint':
        raise ValueError('This benchmark requires a joint occupancy-actor checkpoint')
    backbone = config.get('backbone', 'occfm_unet')
    if backbone == 'occfm_unet':
        return {}
    if backbone != 'dit':
        raise ValueError(f'Unsupported checkpoint backbone: {backbone}')
    options = dict(config['dit'])
    for name, expected in {'in_channels': 16, 'input_size': 50, 'frames': 12,
                           'trajectory_length': 6}.items():
        if options.pop(name, None) != expected:
            raise ValueError(f'Checkpoint DiT {name} differs from the benchmark grid/slots')
    required = {'patch_size', 'hidden_size', 'depth', 'num_heads', 'mlp_ratio',
                'activation_checkpointing'}
    if set(options) != required:
        raise ValueError('Incomplete or unsupported checkpoint DiT configuration')
    return {'backbone': 'dit', 'dit_config': options}


def checkpoint_history_visibility(checkpoint):
    """Legacy joint weights retain their original LiDAR-masked observations."""
    mode = checkpoint['config'].get('history_visibility', 'mask_lidar')
    if mode not in HISTORY_VISIBILITIES:
        raise ValueError(f'Unsupported checkpoint history visibility: {mode}')
    return mode


def paper_horizon_summary(accumulator):
    """Keep pooled/all-six metrics separate from the paper's 1/2/3s mean."""
    report = accumulator.compute()
    for protocol in ('standard', 'legacy'):
        values = report[protocol]['per_time_mean_iou']
        selected = [values[i] for i in HORIZON_INDICES]
        report[protocol]['miou_percent_at_1_2_3s'] = [None if x is None else 100*x for x in selected]
        report[protocol]['miou_percent_mean_1_2_3s'] = (
            None if any(x is None for x in selected) else 100*float(np.mean(selected)))
        report[protocol]['pooled_miou_percent'] = (
            None if report[protocol]['mean_iou'] is None else 100*report[protocol]['mean_iou'])
    return report


def project_prediction_to_future(prediction_e0, e0_to_global, future_to_global, *, grid=GridSpec()):
    """Use future pose ONLY for scoring; unknown prediction support is empty17.

    warp_occupancy_nearest expects source-to-destination, hence E0 -> future
    is inverse(T_global_future) @ T_global_E0. Its reverse map samples E0 cells
    at future-grid voxel centres. Semantic class IDs are never interpolated.
    """
    e0_to_future = inverse_transform(future_to_global) @ np.asarray(e0_to_global)
    return warp_occupancy_nearest(np.asarray(prediction_e0, dtype=np.uint8), e0_to_future,
                                 source_grid=grid, target_grid=grid, unknown=17)


def make_windows(resource_manifest, label_manifest, limit=None):
    labels = {(row['scene'], row['token']): row for row in label_manifest['frames']}
    if len(labels) != len(label_manifest['frames']):
        raise ValueError('Duplicate label identities')
    windows = []
    for index, window in enumerate(resource_manifest['flow_windows']):
        frames = [resource_manifest['frames'][resource_manifest['latent_gt_indices'][k]] for k in window]
        if len(frames) != 12 or len({frame[0] for frame in frames}) != 1:
            raise ValueError('Expected one-scene twelve-slot window')
        scene, token = frames[5]
        metadata = labels[(scene, token)]
        if metadata['history_sample_tokens'] != [frame[1] for frame in frames[2:6]]:
            raise ValueError('History identity differs from the official compatibility window')
        if metadata['future_sample_tokens'] != [frame[1] for frame in frames[6:]]:
            raise ValueError('Future identity differs from the official compatibility window')
        windows.append({'index': index, 'scene': scene, 'token': token,
                        'metadata': metadata,
                        'history_metadata': [labels[tuple(frame)] for frame in frames[2:6]],
                        'future_metadata': [labels[tuple(frame)] for frame in frames[6:]]})
    if limit is not None:
        if limit < 1:
            raise ValueError('limit must be positive')
        windows = windows[:limit]
    return windows


class OnlineEvaluationData(Dataset):
    """CPU geometry/labels; no VAE or predictor is accessible to this dataset."""

    def __init__(self, windows, label_manifest, label_root, occupancy_root, *, history_visibility='mask_lidar'):
        if history_visibility not in HISTORY_VISIBILITIES:
            raise ValueError(f'Unsupported evaluation history visibility: {history_visibility}')
        self.history_visibility = history_visibility
        self.windows = windows
        self.label_manifest = label_manifest
        self.label_root = Path(label_root)
        self.occupancy_root = Path(occupancy_root)
        self.shard_id = None
        self.shard = None

    def __len__(self):
        return len(self.windows)

    def _labels(self, metadata):
        shard_id = metadata['shard']
        if self.shard_id != shard_id:
            path = self.label_root / self.label_manifest['shards'][shard_id]['file']
            with np.load(path, allow_pickle=False) as data:
                self.shard = {key: data[key].copy() for key in data.files}
            self.shard_id = shard_id
        row = metadata['row']
        observation = {key.split('.', 1)[1]: value[row].copy()
                       for key, value in self.shard.items() if key.startswith('observation.')}
        candidate = self.shard['target.ego_future'][row, :, :2].copy()
        if not self.shard['target.ego_target_mask'][row].all():
            raise ValueError('Incomplete future logged-action condition')
        return observation, candidate

    def _grid(self, metadata, *, visibility_key=None):
        path = self.occupancy_root / metadata['scene'] / metadata['token'] / 'labels.npz'
        with np.load(path, allow_pickle=False) as data:
            semantics = data['semantics'].astype(np.uint8, copy=True)
            sensor_mask = data[visibility_key].astype(bool, copy=True) if visibility_key is not None else None
        if semantics.shape != GridSpec().shape or (sensor_mask is not None and sensor_mask.shape != semantics.shape):
            raise ValueError('Invalid original occupancy shape')
        if not np.isin(semantics, [*range(18), 255]).all():
            raise ValueError('Invalid original semantic labels')
        return semantics, sensor_mask

    def __getitem__(self, index):
        entry = self.windows[index]
        metadata = entry['metadata']
        observation, candidate = self._labels(metadata)
        current_to_global = np.asarray(metadata['e0_to_global'], dtype=np.float64)
        global_to_current = inverse_transform(current_to_global)
        history, visible, bounds = [], [], []
        for frame in entry['history_metadata']:
            semantics, sensor_mask = self._grid(frame, visibility_key=visibility_key_from_mode(self.history_visibility))
            transformed = warp_occupancy_nearest(
                semantics, global_to_current @ np.asarray(frame['e0_to_global']), visibility=sensor_mask)
            history.append(transformed['semantics'])
            bounds.append(transformed['in_bounds'])
            visible.append(history_visibility_mask(transformed['semantics'], transformed['in_bounds'],
                                                   transformed.get('visibility')))
        if self.history_visibility == 'mask_lidar_temporal_ground':
            history, visible = complete_masked_ground_history(history, bounds, visible)
        # Future grids/poses are deliberately separate scorer fields.
        original, native, native_known, future_poses = [], [], [], []
        for frame in entry['future_metadata']:
            semantics, _ = self._grid(frame)
            pose = np.asarray(frame['e0_to_global'], dtype=np.float64)
            transformed = warp_occupancy_nearest(semantics, global_to_current @ pose)
            original.append(semantics)
            native.append(transformed['semantics'])
            native_known.append(transformed['in_bounds'] & (transformed['semantics'] != 255))
            future_poses.append(pose)
        # The source observation has four actor slots; only latent slots are padded.
        times = observation.pop('history_time_s')
        observation.pop('history_mask')
        observation['observed_history_time_s'] = times
        observation['history_time_s'] = np.r_[np.zeros(2, dtype=times.dtype), times]
        observation['history_mask'] = np.array([False, False, True, True, True, True])
        return {'index': entry['index'], 'scene': entry['scene'], 'token': entry['token'],
                'observation': observation, 'candidate': candidate,
                'history_semantics': np.stack(history), 'history_visible': np.stack(visible),
                'target_native': np.stack(native), 'target_native_known': np.stack(native_known),
                'target_original': np.stack(original), 'future_to_global': np.stack(future_poses),
                'current_to_global': current_to_global}


def list_collate(batch):
    return batch


def make_initial_noise(batch, seed, device):
    occupancy, actors = [], []
    for sample in batch:
        generator = torch.Generator(device=device).manual_seed(window_seed(seed, sample['scene'], sample['token']))
        occupancy.append(torch.randn((1, 1, 6, 16, 50, 50), generator=generator, device=device))
        actors.append(torch.randn((1, 1, 8, 6, 6), generator=generator, device=device))
    return {'future_latents': torch.cat(occupancy), 'actor_future': torch.cat(actors)}


def predict_batch(flow, codec, batch, device, *, seed, nfe, encode_chunk, decode_chunk):
    """Only explicitly selected observation fields and known action enter flow."""
    observation = {key: torch.from_numpy(np.stack([sample['observation'][key] for sample in batch])).to(device)
                   for key in batch[0]['observation']}
    labels = torch.from_numpy(np.stack([sample['history_semantics'] for sample in batch])).to(device).long()
    visible = torch.from_numpy(np.stack([sample['history_visible'] for sample in batch])).to(device)
    labels = labels.flatten(0, 1)
    visible = visible.flatten(0, 1)
    encoded = torch.cat([codec.encode(labels[i:i+encode_chunk], visible[i:i+encode_chunk], sample=False)
                         for i in range(0, len(labels), encode_chunk)])
    history = torch.zeros((len(batch), 6, 16, 50, 50), device=device)
    history[:, 2:] = encoded.reshape(len(batch), 4, 16, 50, 50)
    observation['history_latents'] = history
    candidate = torch.from_numpy(np.stack([sample['candidate'] for sample in batch])).to(device)
    initial_noise = make_initial_noise(batch, seed, device)
    generator = torch.Generator(device=device).manual_seed(seed)
    result = flow.sample(observation, candidate, num_worlds=1, generator=generator,
                         nfe=nfe, initial_noise=initial_noise)
    latents = result['future_latents'][:, 0].flatten(0, 1)
    if not torch.isfinite(latents).all():
        raise RuntimeError('Nonfinite predicted occupancy latents')
    predicted = torch.cat([codec.decode(latents[i:i+decode_chunk]).to('cpu', dtype=torch.uint8)
                           for i in range(0, len(latents), decode_chunk)])
    return predicted.reshape(len(batch), 6, 200, 200, 16).numpy()


def _accumulator_from_state(state=None):
    result = OccupancyAccumulator()
    if state is not None:
        result.load_state_dict(state)
    return result


def _state(accumulators, coverage, completed, identity):
    return {'identity_sha256': identity, 'completed': completed,
            'accumulators': {key: value.state_dict() for key, value in accumulators.items()},
            'coverage': coverage}


def _result(accumulators, coverage):
    valid = np.asarray(coverage['valid_target_voxels'], dtype=np.float64)
    supported = np.asarray(coverage['supported_target_voxels'], dtype=np.float64)
    return {'metrics': {key: paper_horizon_summary(value) for key, value in accumulators.items()},
            'paper_frame_coverage': {'valid_target_voxels': valid.astype(np.int64).tolist(),
                                     'supported_target_voxels': supported.astype(np.int64).tolist(),
                                     'fraction_per_horizon': np.divide(supported, valid, out=np.zeros(6), where=valid>0).tolist()}}


def prepare_identity(args, resource, labels, windows):
    root = Path(__file__).resolve().parents[2]
    paths = [args.checkpoint, args.codec_checkpoint, args.normalization,
             args.resource_manifest, args.label_manifest,
             *sorted(root.joinpath('reactoccflow/models').rglob('*.py')),
             root/'reactoccflow/geometry/__init__.py', root/'reactoccflow/evaluation/occupancy.py', Path(__file__)]
    paths += [root/'reactoccflow/data/history_visibility.py', root/'reactoccflow/data/loader.py']
    # Bind source bytes before any resumed counters are reused.
    paths += [args.label_manifest.parent / shard['file'] for shard in labels['shards']]
    frames = {(x['scene'], x['token']) for w in windows
              for x in [*w['history_metadata'], *w['future_metadata']]}
    paths += [args.occupancy_root / scene / token / 'labels.npz' for scene, token in sorted(frames)]
    hashes = {str(path.resolve()): payload_sha256(path) for path in paths}
    checkpoint = torch.load(args.checkpoint, map_location='cpu', weights_only=True)
    identity = {'protocol_version': PROTOCOL_VERSION, 'input_sha256': hashes,
                'model_options': checkpoint_model_options(checkpoint),
                'history_visibility': checkpoint_history_visibility(checkpoint),
                'nfe': args.nfe, 'seed': args.seed, 'precision': 'float32',
                'encode_chunk': args.encode_chunk, 'decode_chunk': args.decode_chunk,
                'batch_size': args.batch_size, 'world_size': args.world_size,
                'numerical_settings': {'deterministic_algorithms': True,
                                       'cudnn_deterministic': True, 'cudnn_benchmark': False,
                                       'cublas_workspace_config': ':4096:8',
                                       'matmul_allow_tf32': False, 'cudnn_allow_tf32': True},
                'torch_version': torch.__version__, 'cuda_version': torch.version.cuda,
                'limit': args.limit, 'windows': [[w['scene'], w['token']] for w in windows],
                'condition': 'oracle_actor_history_and_logged_future_ego_action',
                'predictor_frame': 'fixed_current_ego_E0',
                'paper_score_frame': 'each_future_ego;GT_pose_used_in_scorer_only;unsupported_prediction_empty17',
                'native_score_mask': 'future_GT_known_and_inside_current_E0',
                'paper_score_mask': 'all_future_original_GT_except_unknown255;no_camera_or_lidar_mask',
                'heading': 'nominal 0.5s slots; report slots2,4,6 as1s,2s,3s; actual source timestamps retained'}
    encoded = json.dumps(identity, sort_keys=True, separators=(',', ':')).encode()
    return identity, hashlib.sha256(encoded).hexdigest()


def aggregate(output, identity, world_size, expected):
    accumulators = {name: OccupancyAccumulator() for name in SCORE_NAMES}
    coverage = {'valid_target_voxels': np.zeros(6, dtype=np.int64), 'supported_target_voxels': np.zeros(6, dtype=np.int64)}
    completed = []
    for rank in range(world_size):
        path = output / f'rank_{rank:03d}' / 'state.json'
        if not path.exists():
            raise ValueError(f'Missing rank state: {path}')
        state = json.loads(path.read_text())
        if state['identity_sha256'] != identity or not state.get('finished'):
            raise ValueError('All rank states must be complete and share identity')
        completed.extend(state['completed'])
        for key, accumulator in accumulators.items():
            other = state['accumulators'][key]
            for name in ('seen', 'positive', 'correct', 'valid_voxels'):
                setattr(accumulator, name, getattr(accumulator, name) + np.asarray(other[name], dtype=np.int64))
            accumulator.windows += other['windows']
        for key in coverage:
            coverage[key] += state['coverage'][key]
    if sorted(completed) != list(range(expected)):
        raise ValueError('Rank coverage is duplicated or incomplete')
    result = {'status': 'completed', 'identity_sha256': identity, 'windows': expected,
              'world_size': world_size, **_result(accumulators, coverage)}
    atomic_json(output/'result.json', result)
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', type=Path, default=Path('outputs/reactflow_joint/seed_2026/best.pt'))
    parser.add_argument('--codec-checkpoint', type=Path, default=Path('assets/checkpoints/occfm_vae.ckpt'))
    parser.add_argument('--normalization', type=Path, default=Path('data/research/normalization.json'))
    parser.add_argument('--resource-manifest', type=Path, default=Path('resources/occfm_audit/manifest.json'))
    parser.add_argument('--label-manifest', type=Path, default=Path('data/research/evaluation/manifest.json'))
    parser.add_argument('--occupancy-root', type=Path, default=Path('/home/lthpc/data/nuscenes/gts'))
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--rank', type=int, default=0)
    parser.add_argument('--world-size', type=int, default=2)
    parser.add_argument('--limit', type=int)
    parser.add_argument('--batch-size', type=int, default=2)
    parser.add_argument('--workers', type=int, default=4)
    parser.add_argument('--encode-chunk', type=int, default=4)
    parser.add_argument('--decode-chunk', type=int, default=2)
    parser.add_argument('--nfe', type=int, default=10)
    parser.add_argument('--seed', type=int, default=2026)
    parser.add_argument('--aggregate-only', action='store_true')
    args = parser.parse_args(argv)
    if min(args.batch_size, args.encode_chunk, args.decode_chunk, args.nfe) < 1 or args.workers < 0:
        parser.error('Invalid batch/chunk/NFE/worker count')
    torch.set_num_threads(4)
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = True
    resource = json.loads(args.resource_manifest.read_text())
    labels = json.loads(args.label_manifest.read_text())
    if labels.get('group') != 'evaluation' or labels.get('oracle') is not True:
        raise ValueError('Official held-out evaluation labels required')
    windows = make_windows(resource, labels, args.limit)
    assigned = partition_indices(len(windows), args.rank, args.world_size)
    args.output.mkdir(parents=True, exist_ok=True)
    identity, identity_hash = prepare_identity(args, resource, labels, windows)
    with (args.output/'identity.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        path = args.output/'identity.json'
        if path.exists():
            if json.loads(path.read_text()) != identity:
                raise ValueError('Output identity differs; use a fresh output directory')
        else:
            atomic_json(path, identity)
    if args.aggregate_only:
        result = aggregate(args.output, identity_hash, args.world_size, len(windows))
        print(json.dumps({'status': result['status'], 'windows': result['windows'],
                          'paper_frame': result['metrics']['paper_frame_full']['standard']['miou_percent_at_1_2_3s']}), flush=True)
        return
    rank_output = args.output/f'rank_{args.rank:03d}'
    rank_output.mkdir(exist_ok=True)
    with (rank_output/'run.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        state_path = rank_output/'state.json'
        state = json.loads(state_path.read_text()) if state_path.exists() else None
        if state and state['identity_sha256'] != identity_hash:
            raise ValueError('Rank resume identity differs')
        completed = [] if state is None else state['completed']
        if completed != assigned[:len(completed)]:
            raise ValueError('Rank cursor is not the exact assigned prefix')
        accumulators = {name: _accumulator_from_state(None if state is None else state['accumulators'][name])
                        for name in SCORE_NAMES}
        coverage = {name: np.zeros(6, dtype=np.int64) if state is None else np.asarray(state['coverage'][name], dtype=np.int64)
                    for name in ('valid_target_voxels', 'supported_target_voxels')}
        if any(acc.windows != len(completed) for acc in accumulators.values()):
            raise ValueError('Counter/cursor mismatch')
        if len(completed) == len(assigned):
            print(json.dumps({'status': 'already_completed', 'rank': args.rank, 'windows': len(completed)}), flush=True)
            return
        from reactoccflow.models.codec import OccupancyCodec
        from reactoccflow.models.react_flow import ReactFlow
        device = torch.device(args.device)
        if device.type == 'cuda':
            torch.cuda.set_device(device)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        torch.backends.cuda.matmul.allow_tf32 = False
        normalization = json.loads(args.normalization.read_text())
        if normalization.get('fit_group') != 'train':
            raise ValueError('Normalization must be train-only')
        checkpoint = torch.load(args.checkpoint, map_location='cpu', weights_only=True)
        model_options = checkpoint_model_options(checkpoint)
        flow = ReactFlow(normalization=normalization, **model_options).to(device).eval().requires_grad_(False)
        expected_trunk = ('StandardDiT' if model_options.get('backbone') == 'dit'
                          else 'FLOW_MATCHING_DOWN_X4_DiT')
        if type(flow.transition_model.trunk).__name__ != expected_trunk:
            raise ValueError('The loaded model does not match the checkpoint backbone')
        for name in ('actor_mean', 'actor_std'):
            if not torch.equal(getattr(flow, name).detach().cpu(), checkpoint['ema_state_dict'][name]):
                raise ValueError('Checkpoint normalization buffers differ from the supplied train statistics')
        flow.load_state_dict(checkpoint['ema_state_dict'], strict=True)
        codec = OccupancyCodec().to(device).eval().requires_grad_(False)
        codec.load_pretrained(args.codec_checkpoint)
        dataset = OnlineEvaluationData(windows, labels, args.label_manifest.parent, args.occupancy_root,
                                       history_visibility=identity['history_visibility'])
        remaining = torch.utils.data.Subset(dataset, assigned[len(completed):])
        loader = DataLoader(remaining, batch_size=args.batch_size, num_workers=args.workers,
                            shuffle=False, collate_fn=list_collate, persistent_workers=args.workers>0)
        started = time.monotonic()
        initial_completed = len(completed)
        atomic_json(rank_output/'status.json', {'status': 'running', 'pid': os.getpid(), 'rank': args.rank,
                                               'completed': len(completed), 'total': len(assigned)})
        with torch.inference_mode():
            for batch in loader:
                prediction = predict_batch(flow, codec, batch, device, seed=args.seed, nfe=args.nfe,
                                           encode_chunk=args.encode_chunk, decode_chunk=args.decode_chunk)
                for sample, predicted in zip(batch, prediction):
                    accumulators['native_e0_known'].update(predicted, sample['target_native'], visibility=sample['target_native_known'])
                    projected, support = [], []
                    for t in range(6):
                        warped = project_prediction_to_future(predicted[t], sample['current_to_global'], sample['future_to_global'][t])
                        projected.append(warped['semantics'])
                        support.append(warped['in_bounds'])
                        valid = sample['target_original'][t] != 255
                        coverage['valid_target_voxels'][t] += int(valid.sum())
                        coverage['supported_target_voxels'][t] += int((valid & warped['in_bounds']).sum())
                    accumulators['paper_frame_full'].update(np.stack(projected), sample['target_original'])
                    accumulators['paper_frame_covered'].update(np.stack(projected), sample['target_original'], visibility=np.stack(support))
                    completed.append(sample['index'])
                state = _state(accumulators, coverage, completed, identity_hash)
                atomic_json(state_path, state)
                elapsed = time.monotonic() - started
                processed = len(completed) - initial_completed
                progress = {'status': 'running', 'pid': os.getpid(), 'rank': args.rank,
                            'completed': len(completed), 'total': len(assigned), 'elapsed_seconds': elapsed,
                            'windows_per_second': processed/elapsed,
                            'estimated_remaining_seconds': (len(assigned)-len(completed))*elapsed/processed,
                            'checkpoint_step': checkpoint['global_step'],
                            'utc': datetime.now(timezone.utc).isoformat()}
                atomic_json(rank_output/'status.json', progress)
                print(json.dumps(progress), flush=True)
        state = _state(accumulators, coverage, completed, identity_hash)
        state['finished'] = True
        atomic_json(state_path, state)
        result = {'status': 'completed', 'rank': args.rank, 'identity_sha256': identity_hash,
                  'checkpoint_step': checkpoint['global_step'], 'windows': len(completed),
                  'elapsed_seconds': time.monotonic()-started, **_result(accumulators, coverage)}
        atomic_json(rank_output/'result.json', result)
        atomic_json(rank_output/'status.json', {'status': 'completed', 'rank': args.rank,
                                               'completed': len(completed), 'total': len(assigned)})
        shutdown_data_loaders(loader)
        print(json.dumps({'status': 'completed', 'rank': args.rank, 'windows': len(completed)}), flush=True)


if __name__ == '__main__':
    main()
