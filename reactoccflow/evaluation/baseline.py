"""Resumable dataset evaluation adapted from the copied OccFM validation loop.

Run python -m reactoccflow.evaluation.baseline --help. Compatibility predictions
remain in their original per-frame coordinate convention. Ground-truth future
ego conditioning is a prediction-task condition, not a deployed planner.
"""
from __future__ import annotations

import argparse
from functools import lru_cache
import hashlib
import json
import os
from pathlib import Path
import time

import numpy as np

from .occupancy import OccupancyAccumulator


def evaluation_payload_hashes(dataset, count, data_root):
    """Bind resume state to array bytes and the requested future-label set.

    The manifest describes layout, not contents. Hashing is intentionally done
    before resuming, with bounded-memory streaming in the provenance helper.
    """
    from reactoccflow.provenance import sha256_file

    cache_files = {'trajectory.npy', *(shard['file'] for shard in dataset.manifest['shards'])}
    cache_hashes = {name: sha256_file(dataset.root / name) for name in sorted(cache_files)}
    identities = set()
    for start in dataset.valid_indices[:count]:
        future = dataset.frames[start + dataset.history:start + dataset.history + dataset.future]
        identities.update((frame['scene'], frame['token']) for frame in future)
    label_hashes = {f'{scene}/{token}': sha256_file(Path(data_root) / 'gts' / scene / token / 'labels.npz')
                    for scene, token in sorted(identities)}
    return {'cache_payload_sha256': cache_hashes, 'label_payload_sha256': label_hashes}


def window_seed(seed, scene, tokens):
    identity = json.dumps([int(seed), scene, list(tokens)], separators=(',', ':'))
    return int.from_bytes(hashlib.sha256(identity.encode()).digest()[:8], 'little') % (2**63 - 1)


class EvaluationState:
    def __init__(self, settings, times=6):
        self.settings = settings
        self.masked, self.unmasked = OccupancyAccumulator(times), OccupancyAccumulator(times)
        self.processed = 0
        self.compute_seconds = 0.
        self.batch_latencies = []
        self.peak_cuda_bytes = None

    def update(self, prediction, target, visibility):
        self.masked.update(prediction, target, visibility=visibility)
        self.unmasked.update(prediction, target)
        self.processed += 1

    def save(self, path):
        path = Path(path)
        state = {'settings': np.array(json.dumps(self.settings, sort_keys=True)),
                 'processed': np.array(self.processed), 'compute_seconds': np.array(self.compute_seconds),
                 'batch_latencies': np.asarray(self.batch_latencies, dtype=np.float64),
                 'peak_cuda_bytes': np.array(-1 if self.peak_cuda_bytes is None else self.peak_cuda_bytes, dtype=np.int64)}
        for prefix, acc in [('masked', self.masked), ('unmasked', self.unmasked)]:
            for name, value in acc.state_dict().items():
                state[f'{prefix}_{name}'] = value
        temporary = path.with_suffix('.tmp.npz')
        np.savez(temporary, **state)
        os.replace(temporary, path)

    def load(self, path):
        with np.load(path, allow_pickle=False) as state:
            if json.loads(str(state['settings'])) != self.settings:
                raise ValueError('Resume settings differ; use a new output directory')
            self.processed = int(state['processed'])
            self.compute_seconds = float(state['compute_seconds'])
            self.batch_latencies = state['batch_latencies'].tolist()
            peak = int(state['peak_cuda_bytes'])
            self.peak_cuda_bytes = None if peak == -1 else peak
            for prefix, acc in [('masked', self.masked), ('unmasked', self.unmasked)]:
                acc.load_state_dict({name: state[f'{prefix}_{name}'] for name in ['seen', 'positive', 'correct', 'valid_voxels', 'windows']})
            if self.masked.windows != self.processed or self.unmasked.windows != self.processed:
                raise ValueError('Corrupt evaluation window count')


def _json(path, value):
    path = Path(path)
    temporary = path.with_suffix('.tmp.json')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    os.replace(temporary, path)


def evaluate(args):
    import torch
    from reactoccflow.data.cache import LatentSequenceDataset
    from reactoccflow.models import OccupancyCodec, ConditionalFlowMatching
    from reactoccflow.provenance import sha256_file

    torch.set_num_threads(args.cpu_threads)
    device = torch.device(args.device)
    if device.type == 'cuda':
        torch.cuda.set_device(device)
    dataset = LatentSequenceDataset(args.cache, hist_last=4)
    count = len(dataset) if args.limit is None else min(len(dataset), args.limit)
    if count < 1 or args.batch_size < 1 or args.decode_chunk < 1:
        raise ValueError('Positive window count and batch/decode chunk required')
    checkpoint = Path(args.checkpoints) / f'occfm_{args.variant}.ckpt'
    settings = {'variant': args.variant, 'nfe': args.nfe, 'seed': args.seed, 'weights': args.weights,
                'batch_size': args.batch_size, 'decode_chunk': args.decode_chunk, 'total_windows': count,
                'dataset_windows': len(dataset), 'cache_manifest_sha256': sha256_file(Path(args.cache) / 'manifest.json'),
                'checkpoint_sha256': sha256_file(checkpoint),
                'vae_sha256': sha256_file(Path(args.checkpoints) / 'occfm_vae.ckpt'),
                'dtype': 'float32', 'data_root': str(Path(args.data_root).resolve()),
                'seed_scheme': 'sha256(seed,scene,window_tokens)', 'hist_last': 4,
                'metric_protocol_version': 2, 'full_validation_requested': count == len(dataset),
                'device': str(device), 'gpu_name': torch.cuda.get_device_name(device) if device.type == 'cuda' else None,
                'cpu_threads': args.cpu_threads}
    settings.update(evaluation_payload_hashes(dataset, count, args.data_root))
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    state = EvaluationState(settings)
    state_path = output / 'state.npz'
    if state_path.exists():
        state.load(state_path)
    if state.processed > count:
        raise ValueError('Saved state exceeds requested dataset')

    @lru_cache(maxsize=128)
    def label(scene, token):
        path = Path(args.data_root) / 'gts' / scene / token / 'labels.npz'
        with np.load(path) as data:
            if 'mask_camera' not in data:
                raise ValueError(f'Camera visibility missing: {path}')
            return data['semantics'].copy(), data['mask_camera'].astype(bool)

    def write_report():
        report = {'status': 'complete' if state.processed == count else 'running', 'settings': settings,
                  'processed_windows': state.processed, 'total_windows': count,
                  'complete_fraction': state.processed / count,
                  'masked_camera': state.masked.compute(), 'unmasked': state.unmasked.compute(),
                  'compute_seconds': state.compute_seconds,
                  'batch_latencies_seconds': {'mean': float(np.mean(state.batch_latencies)) if state.batch_latencies else None,
                                              'p95': float(np.percentile(state.batch_latencies,95)) if state.batch_latencies else None},
                  'device': state.settings['device'], 'gpu_name': state.settings['gpu_name'],
                  'peak_cuda_bytes': state.peak_cuda_bytes,
                  'note': 'Counts aggregate across windows before IoU. Legacy unmasked per-time mean matches source absent-class rule; standard camera-masked is separate. Overlapping windows are not independent samples.'}
        _json(output / 'metrics.json', report)
        return report

    if state.processed == count:
        return write_report()

    model = ConditionalFlowMatching(trajectory_length=6 if args.variant == 'history' else 12).to(device).eval()
    model.load_pretrained(checkpoint, weights=args.weights)
    codec = OccupancyCodec().to(device).eval()
    codec.load_pretrained(Path(args.checkpoints) / 'occfm_vae.ckpt')
    if device.type == 'cuda':
        torch.cuda.reset_peak_memory_stats(device)
    write_report()
    with torch.inference_mode():
        for start in range(state.processed, count, args.batch_size):
            batch = [dataset[i] for i in range(start, min(start + args.batch_size, count))]
            history = torch.from_numpy(np.stack([sample['history_latents'] for sample in batch])).to(device)
            trajectories = torch.from_numpy(np.stack([sample['trajectory'] for sample in batch])).to(device)
            if args.variant == 'history':
                trajectories = trajectories[:, :6]
            noise = torch.stack([torch.randn((6, 16, 50, 50), device=device,
                        generator=torch.Generator(device=device).manual_seed(window_seed(args.seed, sample['scene'], sample['tokens']))) for sample in batch])
            # Populate labels before timing GPU prediction and decode.
            truths, masks = [], []
            for sample in batch:
                rows = [label(sample['scene'], token) for token in sample['tokens'][6:]]
                truths.append(np.stack([row[0] for row in rows]))
                masks.append(np.stack([row[1] for row in rows]))
            if device.type == 'cuda':
                torch.cuda.synchronize(device)
            begin = time.perf_counter()
            latents = model.sample(history, trajectories, initial_noise=noise, nfe=args.nfe)
            flattened = latents.flatten(0, 1)
            decoded = []
            for first in range(0, len(flattened), args.decode_chunk):
                decoded.append(codec.decode(flattened[first:first + args.decode_chunk]).to(torch.uint8).cpu().numpy())
            predictions = np.concatenate(decoded).reshape(len(batch), 6, 200, 200, 16)
            if device.type == 'cuda':
                torch.cuda.synchronize(device)
            elapsed = time.perf_counter() - begin
            for predicted, target, visibility in zip(predictions, truths, masks):
                state.update(predicted, target, visibility)
            state.compute_seconds += elapsed
            state.batch_latencies.append(elapsed)
            if device.type == 'cuda':
                state.peak_cuda_bytes = max(state.peak_cuda_bytes or 0, torch.cuda.max_memory_allocated(device))
            state.save(state_path)
            if start == 0 or state.processed % max(args.batch_size, 32) == 0 or state.processed == count:
                write_report()
                print(json.dumps({'processed': state.processed, 'total': count,
                                  'compute_seconds_per_window': state.compute_seconds / state.processed}), flush=True)
    report = write_report()
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--cache', default='data/latents/validation')
    parser.add_argument('--data-root', required=True)
    parser.add_argument('--checkpoints', default='assets/checkpoints')
    parser.add_argument('--variant', choices=['history','future'], required=True)
    parser.add_argument('--weights', choices=['state_dict','ema_model'], default='ema_model')
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--nfe', type=int, default=10)
    parser.add_argument('--seed', type=int, default=2026)
    parser.add_argument('--batch-size', type=int, default=4)
    parser.add_argument('--decode-chunk', type=int, default=4)
    parser.add_argument('--cpu-threads', type=int, default=4)
    parser.add_argument('--limit', type=int)
    parser.add_argument('--output', required=True)
    evaluate(parser.parse_args())


if __name__ == '__main__':
    main()
