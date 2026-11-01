"""T7 small-sample training with continuous decoded endpoint consistency.

Run as python -m scripts.train_joint_smoke. Partial E0 caches require the
explicit --allow-incomplete debug flag. This is an oracle-observation training
diagnostic with separately supplied logged ego action, not planning evaluation.
Existing copied components and unchanged T3 helpers are documented in
``docs/provenance/training_joint_sources.json``.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import random
import time

import numpy as np
import torch

from reactoccflow.data.e0_dataset import E0ResearchDataset
from reactoccflow.models import OccupancyCodec
from reactoccflow.models.react_flow import ReactFlow
from reactoccflow.provenance import sha256_file
from reactoccflow.training.joint import JointTrainingAdapter
from reactoccflow.training.diagnostics import atomic_write, evaluate_probe, run_training


def array_fingerprint(array):
    contiguous = np.ascontiguousarray(array)
    return {'shape': list(array.shape), 'dtype': str(array.dtype),
            'sha256': hashlib.sha256(contiguous.tobytes()).hexdigest()}


def module_fingerprint(module):
    digest = hashlib.sha256()
    for name, value in module.state_dict().items():
        digest.update(name.encode())
        digest.update(str(value.dtype).encode())
        digest.update(str(tuple(value.shape)).encode())
        digest.update(value.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--cache', default='data/research/e0_full/train')
    parser.add_argument('--output', default='outputs/dit_joint_training_acceptance')
    parser.add_argument('--codec-checkpoint', default='assets/checkpoints/occfm_vae.ckpt')
    parser.add_argument('--device', default='cuda:1')
    parser.add_argument('--windows', type=int, default=32)
    parser.add_argument('--batch-size', type=int, default=1)
    parser.add_argument('--steps', type=int, default=200)
    parser.add_argument('--lr', type=float, default=1e-4, help='Standard DiT and actor/interaction learning rate')
    parser.add_argument('--threads', type=int, default=4)
    parser.add_argument('--seed', type=int, default=2026)
    parser.add_argument('--decode-chunk', type=int, default=1)
    parser.add_argument('--allow-incomplete', action='store_true', help='Explicit partial-cache debug diagnostic only')
    args = parser.parse_args()
    if args.steps < 1 or args.windows < 1 or args.batch_size < 1 or args.windows % args.batch_size:
        parser.error('Positive steps/windows/batch-size required; windows must be divisible by batch-size')
    # Refuse partial cache before creating outputs or allocating a model.
    dataset = E0ResearchDataset(args.cache, verify_checksums=True, allow_incomplete=args.allow_incomplete)
    if len(dataset) < args.windows:
        raise ValueError(f'Requested {args.windows} distinct windows; cache contains {len(dataset)}')
    normalization = dataset.normalization
    if not isinstance(normalization, dict) or normalization.get('fit_group') != 'train' or normalization.get('coordinate_frame') != 'E0':
        raise ValueError('Cache must embed actual train-only E0 normalization statistics')
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    if any((output / name).exists() for name in ('results.json', 'state.pt', 'steps.jsonl')):
        raise FileExistsError('Use a fresh output directory; existing diagnostic evidence is retained')
    torch.set_num_threads(args.threads)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device(args.device)
    if device.type == 'cuda':
        torch.cuda.set_device(device)
        torch.cuda.manual_seed_all(args.seed)
        torch.cuda.reset_peak_memory_stats(device)
    started = time.perf_counter()
    started_utc = datetime.now(timezone.utc).isoformat()
    samples = [dataset[index] for index in range(args.windows)]
    selected = []
    file_hashes = {str(dataset.root / 'manifest.json'): sha256_file(dataset.root / 'manifest.json')}
    settings = dataset.root / 'settings.json'
    if settings.is_file():
        file_hashes[str(settings)] = sha256_file(settings)
    for index, sample in enumerate(samples):
        entry = dataset.records[index]
        for key in ('file', 'record_file'):
            path = dataset.root / entry[key]
            file_hashes[str(path)] = sha256_file(path)
        selected.append({'dataset_index': index, 'global_index': entry['global_index'],
                         'scene': entry['scene'], 'token': entry['token'],
                         'history_tokens': sample['metadata']['history_sample_tokens'],
                         'future_tokens': sample['metadata']['future_sample_tokens'],
                         'arrays': {section: {key: array_fingerprint(array) for key, array in sample[section].items()}
                                    for section in ('observation', 'target')}})
    if len({(entry['scene'], entry['token']) for entry in selected}) != args.windows:
        raise RuntimeError('Selected cache records are not distinct')
    batches = []
    for offset in range(0, args.windows, args.batch_size):
        group = samples[offset:offset + args.batch_size]
        observation, target = [{key: torch.from_numpy(np.stack([sample[section][key] for sample in group])).to(device)
                                for key in group[0][section]} for section in ('observation', 'target')]
        # A separately supplied logged action conditions this supervised task.
        candidate = target['ego_future'][..., :2].clone()
        batches.append((observation, target, candidate))
    del samples
    flow = ReactFlow(normalization=normalization).to(device)
    codec = OccupancyCodec().to(device)
    codec_loading = codec.load_pretrained(args.codec_checkpoint, weights='state_dict')
    adapter = JointTrainingAdapter(flow, codec, decode_chunk=args.decode_chunk, require_frozen_trunk=False)
    frozen_before = {'vae': module_fingerprint(codec)}
    frozen_status = adapter.frozen_status()
    components_before = []
    # evaluate_probe uses a replayed local generator and eval/no_grad mode.
    initial_probe = evaluate_probe(adapter, batches, seed=args.seed + 2)
    initial_last_components = dict(adapter.last_components)
    if device.type == 'cuda':
        torch.cuda.synchronize(device)
    train_start = time.perf_counter()
    with (output / 'steps.jsonl').open('x') as stream:
        def log(entry):
            entry.update({'components': dict(adapter.last_components), 'frozen': adapter.frozen_status(),
                          'elapsed_seconds': time.perf_counter() - started})
            line = json.dumps(entry, allow_nan=False)
            stream.write(line + '\n')
            stream.flush()
            print(line, flush=True)
        print(json.dumps({'initial_probe': initial_probe['mean_loss'], 'windows': args.windows,
                          'batch_size': args.batch_size, 'allow_incomplete': args.allow_incomplete,
                          'cache_complete': dataset.manifest['complete']}), flush=True)
        training = run_training(adapter, batches, steps=args.steps, lr=args.lr,
                                weight_decay=.01, seed=args.seed + 1, max_norm=1., log_callback=log)
    if device.type == 'cuda':
        torch.cuda.synchronize(device)
    train_seconds = time.perf_counter() - train_start
    final_probe = evaluate_probe(adapter, batches, seed=args.seed + 2)
    final_last_components = dict(adapter.last_components)
    frozen_status = adapter.frozen_status()
    frozen_after = {'vae': module_fingerprint(codec)}
    if frozen_before != frozen_after:
        raise RuntimeError('Frozen VAE tensors changed during diagnostic')
    config = {'mode': 'joint', 'steps': args.steps, 'windows': args.windows, 'batch_size': args.batch_size,
              'backbone': 'dit', 'dit': flow.transition_model.trunk.config,
              'optimizer': 'AdamW', 'lr': args.lr, 'weight_decay': .01, 'max_grad_norm': 1.,
              'gradient_accumulation': 1, 'precision': 'float32', 'decode_chunk': args.decode_chunk,
              'occ_weight': 1., 'actor_weight': 1., 'existence_weight': .1,
              'occ_actor_weight': .1, 'temporal_weight': .05, 'kinematics_weight': .01,
              'freeze_backbone': False, 'freeze_vae': True, 'device': str(device),
              'seed_global': args.seed, 'seed_training': args.seed + 1, 'seed_probe': args.seed + 2,
              'threads': args.threads, 'allow_incomplete': args.allow_incomplete,
              'cache_complete': dataset.manifest['complete'], 'scheduler': None, 'ema': None}
    state_path = output / 'state.pt'
    state = {name: value.detach().cpu() for name, value in flow.state_dict().items()}
    atomic_write(state_path, lambda path: torch.save({'state_dict': state, 'config': config,
                                                      'normalization': normalization}, path))
    input_paths = ['scripts/train_joint_smoke.py', 'reactoccflow/training/diagnostics.py', 'reactoccflow/models/react_flow.py',
                   'reactoccflow/models/joint_transition.py', 'reactoccflow/models/codec.py',
                   'reactoccflow/losses/consistency.py', 'reactoccflow/data/e0_dataset.py',
                   'reactoccflow/models/standard_dit.py', args.codec_checkpoint]
    report = {'version': 1, 'status': 'completed', 'started_utc': started_utc, 'config': config,
              'training': training, 'initial_probe': initial_probe, 'final_probe': final_probe,
              'probe_loss_decreased': final_probe['mean_loss'] < initial_probe['mean_loss'],
              'probe_final_to_initial_ratio': final_probe['mean_loss'] / initial_probe['mean_loss'],
              'initial_probe_last_batch_components': initial_last_components,
              'final_probe_last_batch_components': final_last_components,
              'smoke_200_steps_passed': args.steps == 200 and training['optimizer_steps'] == 200,
              'small_sample_32_window_loss_decreased': args.windows == 32 and final_probe['mean_loss'] < initial_probe['mean_loss'],
              'frozen_status': frozen_status, 'frozen_check_count': adapter.frozen_checks,
              'frozen_initial_sha256': frozen_before, 'frozen_final_sha256': frozen_after,
              'loading_reports': {'initialization': 'standard DiT from scratch', 'codec': codec_loading},
              'data_identity': {'cache_root': str(dataset.root), 'cache_complete': dataset.manifest['complete'],
                                'allow_incomplete': args.allow_incomplete, 'expected_count': dataset.manifest['expected_count'],
                                'available_count': len(dataset), 'selected_window_count': args.windows,
                                'selected_scene_count': len({entry['scene'] for entry in selected}),
                                'sample_indices': list(range(args.windows)), 'records': selected,
                                'normalization_canonical_sha256': hashlib.sha256(json.dumps(normalization, sort_keys=True, separators=(',', ':')).encode()).hexdigest(),
                                'file_hashes': file_hashes,
                                'selection': 'First32 distinct scene-scoped anchor windows may overlap; not32 independent scenes.'},
              'input_hashes': {str(path): sha256_file(path) for path in input_paths},
              'checkpoint': {'path': str(state_path), 'sha256': sha256_file(state_path),
                             'tensor_count': len(state), 'format': 'ReactFlow tensor state_dict plus plain config/normalization only; codec excluded'},
              'timing': {'training_seconds': train_seconds, 'total_seconds': time.perf_counter() - started,
                         'concurrent_engineering_work': True, 'not_isolated_T16_benchmark': True},
              'peak_memory_allocated_bytes': torch.cuda.max_memory_allocated(device) if device.type == 'cuda' else None,
              'peak_memory_reserved_bytes': torch.cuda.max_memory_reserved(device) if device.type == 'cuda' else None,
              'environment': {'torch': torch.__version__, 'cuda': torch.version.cuda,
                              'gpu': torch.cuda.get_device_name(device) if device.type == 'cuda' else None},
              'scope': 'Oracle E0 small-sample training with logged action; no future targets enter observation or inference.',
              'limitations': ['Partial-cache use is an explicitly enabled debug diagnostic when allow_incomplete is true.',
                             'A lower fixed32-window probe is fitting evidence, not full overfit convergence or formal dataset training.',
                             'No generalization, calibrated reaction, planning, closed-loop or safety result is established.',
                             'VAE stays frozen; DiT and actor/interaction are optimized. No argmax decode; checkpoint is not a production resume checkpoint.']}
    atomic_write(output / 'results.json', lambda path: Path(path).write_text(json.dumps(report, indent=2, allow_nan=False) + '\n'))
    print(json.dumps({'output': str(output), 'optimizer_steps': training['optimizer_steps'],
                      'initial_probe': initial_probe['mean_loss'], 'final_probe': final_probe['mean_loss'],
                      'probe_loss_decreased': report['probe_loss_decreased'], 'frozen_status': frozen_status}, allow_nan=False), flush=True)


if __name__ == '__main__':
    main()
