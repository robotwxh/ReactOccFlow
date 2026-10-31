"""Run the T3 200-step smoke and fixed 32-window overfit diagnostic.

The train/zero-grad/backward/clip/step order adapts the verbatim OccFM trainer
snapshot recorded in docs/provenance/training_sources.json. Only project-owned
runtime modules and independent resources are used. This compatibility check
retains the legacy historical-trajectory convention, including its last
history slot's one-step future ego displacement: it is not a future-free
planning experiment or a formal training/quality benchmark.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import random
import tempfile
import time

import numpy as np
import torch
from torch.nn.utils import clip_grad_norm_

from reactoccflow.data.cache import LatentSequenceDataset
from reactoccflow.models.flow import ConditionalFlowMatching
from reactoccflow.provenance import sha256_file


def evaluate_probe(model, batches, *, seed):
    """Replay noise/time/conditioning draws in eval mode without gradients."""
    was_training = model.training
    device = next(model.parameters()).device
    generator = torch.Generator(device=device).manual_seed(seed)
    losses = []
    model.eval()
    try:
        with torch.no_grad():
            for history, target, trajectory in batches:
                loss = model.training_loss(history, target, trajectory, generator=generator)
                if loss.ndim != 0 or not torch.isfinite(loss).item():
                    raise RuntimeError('Nonfinite or nonscalar probe loss')
                losses.append(float(loss))
    finally:
        model.train(was_training)
    if not losses:
        raise ValueError('Probe batches must be nonempty')
    return {'mean_loss': sum(losses) / len(losses), 'batch_losses': losses,
            'seed': seed, 'mode': 'eval_with_no_grad', 'fixed_noise_time_condition_draws': True}


def run_training(model, batches, *, steps, lr=2e-5, weight_decay=.01,
                 seed=2027, max_norm=1., log_callback=None):
    """Perform exactly the requested optimizer steps and fail on invalid states."""
    if steps < 1 or not batches:
        raise ValueError('Positive steps and nonempty batches are required')
    named = [(name, p) for name, p in model.named_parameters() if p.requires_grad]
    if not named:
        raise ValueError('No trainable parameters')
    parameters = [p for _, p in named]
    name, selected = named[0]
    before = selected.detach().cpu().clone()
    generator = torch.Generator(device=selected.device).manual_seed(seed)
    optimizer = torch.optim.AdamW(parameters, lr=lr, weight_decay=weight_decay)
    model.train()
    losses, gradient_norms = [], []
    gradient_parameter_counts = []
    for index in range(steps):
        history, target, trajectory = batches[index % len(batches)]
        optimizer.zero_grad(set_to_none=True)
        loss = model.training_loss(history, target, trajectory, generator=generator)
        if loss.ndim != 0 or not torch.isfinite(loss).item():
            raise RuntimeError(f'Nonfinite or nonscalar training loss at step {index + 1}')
        if not loss.requires_grad:
            raise RuntimeError(f'Training loss has no gradient at step {index + 1}')
        loss.backward()
        gradients = [p.grad for p in parameters if p.grad is not None]
        if not gradients or not torch.stack([torch.isfinite(g).all() for g in gradients]).all().item():
            raise RuntimeError(f'Missing or nonfinite gradient at step {index + 1}')
        norm = clip_grad_norm_(parameters, max_norm, error_if_nonfinite=True)
        optimizer.step()
        if not torch.stack([torch.isfinite(p).all() for p in parameters]).all().item():
            raise RuntimeError(f'Nonfinite parameter after optimizer step {index + 1}')
        losses.append(float(loss.detach()))
        gradient_norms.append(float(norm))
        gradient_parameter_counts.append(len(gradients))
        if log_callback is not None and ((index + 1) % 20 == 0 or index + 1 == steps):
            log_callback({'step': index + 1, 'loss': losses[-1],
                          'gradient_norm_before_clip': gradient_norms[-1],
                          'gradient_parameters': len(gradients), 'finite': True})
    after = selected.detach().cpu()
    delta = (after - before).abs()
    changed = int(torch.count_nonzero(delta))
    if changed == 0:
        raise RuntimeError(f'No optimizer update observed in trainable parameter {name}')
    return {'optimizer_steps': len(losses), 'losses': losses,
            'gradient_norms_before_clip': gradient_norms,
            'finite_loss_steps': len(losses), 'finite_gradient_steps': len(losses),
            'finite_parameter_steps': len(losses),
            'gradient_parameter_count_range': [min(gradient_parameter_counts), max(gradient_parameter_counts)],
            'trainable_parameter_count': sum(p.numel() for p in parameters),
            'parameter_update': {'name': name, 'changed_elements': changed,
                                 'numel': selected.numel(), 'max_abs_delta': float(delta.max()),
                                 'l2_delta': float(torch.linalg.vector_norm(delta)),
                                 'initial_sha256': hashlib.sha256(before.numpy().tobytes()).hexdigest(),
                                 'final_sha256': hashlib.sha256(after.numpy().tobytes()).hexdigest()}}


def atomic_write(path, writer):
    """Publish a complete output from an ordinary temporary file."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix='.' + path.name + '-', dir=path.parent)
    os.close(descriptor)
    try:
        writer(temporary)
        os.replace(temporary, path)
    finally:
        Path(temporary).unlink(missing_ok=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--cache', default='data/latents/training')
    parser.add_argument('--checkpoint', default='assets/checkpoints/occfm_history.ckpt')
    parser.add_argument('--output', default='outputs/training_acceptance')
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--steps', type=int, default=200)
    parser.add_argument('--windows', type=int, default=32)
    parser.add_argument('--batch-size', type=int, default=2)
    parser.add_argument('--threads', type=int, default=4)
    parser.add_argument('--seed', type=int, default=2026)
    args = parser.parse_args()
    if args.windows < 1 or args.batch_size < 1 or args.windows % args.batch_size:
        parser.error('windows must be positive and divisible by positive batch-size')
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    result_path = output / 'results.json'
    checkpoint_path = output / 'trained_state.pt'
    if result_path.exists() or checkpoint_path.exists():
        raise FileExistsError('Use a new output directory; completed acceptance evidence is immutable')
    torch.set_num_threads(args.threads)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device(args.device)
    if device.type == 'cuda':
        torch.cuda.set_device(device)
        torch.cuda.manual_seed_all(args.seed)
        torch.cuda.reset_peak_memory_stats(device)
    start = time.perf_counter()
    started_utc = datetime.now(timezone.utc).isoformat()
    dataset = LatentSequenceDataset(args.cache, hist_last=4)
    if len(dataset) < args.windows:
        raise ValueError(f'Need {args.windows} distinct windows; cache supplies {len(dataset)}')
    indices = list(range(args.windows))
    samples = [dataset[index] for index in indices]
    windows = []
    shard_ids = set()
    for index, sample in zip(indices, samples):
        start_frame = dataset.valid_indices[index]
        rows = dataset.frames[start_frame:start_frame + dataset.history + dataset.future]
        if len({row['scene'] for row in rows}) != 1:
            raise RuntimeError('Selected window crosses a scene boundary')
        shard_ids.update(row['shard'] for row in rows)
        windows.append({'dataset_index': index, 'frame_start': start_frame,
                        'scene': sample['scene'], 'tokens': sample['tokens']})
    if len({tuple(w['tokens']) for w in windows}) != args.windows:
        raise RuntimeError('Selected windows are not distinct')
    batches = []
    for offset in range(0, len(samples), args.batch_size):
        group = samples[offset:offset + args.batch_size]
        batches.append(tuple(torch.from_numpy(np.stack([sample[key][:length] for sample in group])).to(device)
                             for key, length in [('history_latents', 6), ('future_latents', 6), ('trajectory', 6)]))
    used_paths = [Path(args.cache) / 'manifest.json', Path(args.cache) / 'trajectory.npy']
    used_paths += [Path(args.cache) / dataset.manifest['shards'][sid]['file'] for sid in sorted(shard_ids)]
    identity = {'cache_root': str(Path(args.cache).resolve()), 'dataset_window_count': len(dataset),
                'selected_window_count': len(indices), 'sample_indices': indices, 'windows': windows,
                'overlap_policy': 'First32 distinct scene-local windows may overlap; not32 independent scenes.',
                'resource_hashes': {str(path): sha256_file(path) for path in used_paths},
                'checkpoint': str(Path(args.checkpoint).resolve()),
                'checkpoint_sha256': sha256_file(args.checkpoint)}
    model = ConditionalFlowMatching().to(device)
    loading = model.load_pretrained(args.checkpoint, weights='state_dict')
    if device.type == 'cuda':
        torch.cuda.synchronize(device)
    setup_seconds = time.perf_counter() - start
    probe_start = time.perf_counter()
    initial_probe = evaluate_probe(model, batches, seed=args.seed + 2)
    if device.type == 'cuda':
        torch.cuda.synchronize(device)
    initial_probe_seconds = time.perf_counter() - probe_start
    log_path = output / 'steps.jsonl'
    with log_path.open('x') as stream:
        def log(entry):
            entry['elapsed_seconds'] = time.perf_counter() - start
            line = json.dumps(entry, allow_nan=False)
            stream.write(line + '\n')
            stream.flush()
            print(line, flush=True)
        print(json.dumps({'initial_probe': initial_probe['mean_loss'], 'windows': len(indices),
                          'batch_size': args.batch_size, 'concurrent_workload': True}), flush=True)
        train_start = time.perf_counter()
        training = run_training(model, batches, steps=args.steps, lr=2e-5,
                                weight_decay=.01, seed=args.seed + 1, log_callback=log)
        if device.type == 'cuda':
            torch.cuda.synchronize(device)
        training_seconds = time.perf_counter() - train_start
    final_probe = evaluate_probe(model, batches, seed=args.seed + 2)
    if device.type == 'cuda':
        torch.cuda.synchronize(device)
    config = {'steps': args.steps, 'batch_size': args.batch_size, 'windows': args.windows,
              'optimizer': 'AdamW', 'lr': 2e-5, 'weight_decay': .01, 'max_grad_norm': 1.,
              'precision': 'float32', 'gradient_accumulation': 1, 'hist_last': 4,
              'history_length': 6, 'future_length': 6, 'trajectory_length': 6,
              'seed_global': args.seed, 'seed_training_noise_time_keep': args.seed + 1,
              'seed_fixed_probe_noise_time_keep': args.seed + 2,
              'data_order': 'deterministic first-window order; cycle16 batches for200 steps',
              'vae': 'frozen by exclusion; codec not instantiated; cached latents only',
              'scheduler': None, 'ema': None, 'device': str(device), 'torch_threads': args.threads}
    state = {name: tensor.detach().cpu() for name, tensor in model.state_dict().items()}
    atomic_write(checkpoint_path, lambda path: torch.save({'state_dict': state, 'config': config}, path))
    report = {'version': 1, 'status': 'completed', 'started_utc': started_utc,
              'config': config, 'data_identity': identity, 'loading_report': loading,
              'training': training, 'initial_probe': initial_probe, 'final_probe': final_probe,
              'probe_loss_decreased': final_probe['mean_loss'] < initial_probe['mean_loss'],
              'probe_loss_ratio_final_to_initial': final_probe['mean_loss'] / initial_probe['mean_loss'],
              'smoke_200_steps_passed': args.steps == 200 and training['optimizer_steps'] == 200,
              'small_overfit_32_window_diagnostic_passed': args.windows == 32 and final_probe['mean_loss'] < initial_probe['mean_loss'],
              'task_scope': 'Canonical OccFM history compatibility training diagnostic; not future-free planning.',
              'legacy_conditioning_caveat': 'Cache trajectory[t] encodes ego displacement from t to t+1; trajectory[:6] includes one-step future ego displacement in last history slot. Retained transparently for canonical checkpoint compatibility.',
              'limitations': ['32distinct scene-local windows may overlap; not a validation/generalization result.',
                             '200steps are a pre-formal-training diagnostic; no warmup/cosine/EMA/DDP production training claim.',
                             'Timing measured concurrently with fullvalidation jobs; not isolated T16 performance.'],
              'timing': {'concurrent_workload': True, 'setup_seconds': setup_seconds,
                         'initial_probe_seconds': initial_probe_seconds, 'training_seconds': training_seconds,
                         'total_seconds_before_result_write': time.perf_counter() - start},
              'environment': {'torch': torch.__version__, 'cuda': torch.version.cuda,
                              'gpu': torch.cuda.get_device_name(device) if device.type == 'cuda' else None},
              'peak_memory_allocated_bytes': torch.cuda.max_memory_allocated(device) if device.type == 'cuda' else None,
              'peak_memory_reserved_bytes': torch.cuda.max_memory_reserved(device) if device.type == 'cuda' else None,
              'checkpoint': {'path': str(checkpoint_path), 'sha256': sha256_file(checkpoint_path),
                             'format': 'Tensor state_dict plus plain config, no serialized model/optimizer object.'}}
    atomic_write(result_path, lambda path: Path(path).write_text(json.dumps(report, indent=2, allow_nan=False) + '\n'))
    print(json.dumps({'result': str(result_path), 'optimizer_steps': training['optimizer_steps'],
                      'initial_probe': initial_probe['mean_loss'], 'final_probe': final_probe['mean_loss'],
                      'probe_loss_decreased': report['probe_loss_decreased'],
                      'peak_memory_allocated_bytes': report['peak_memory_allocated_bytes']}, allow_nan=False), flush=True)


if __name__ == '__main__':
    main()
