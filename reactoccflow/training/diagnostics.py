"""Backbone-independent optimizer/probe diagnostics.

Relocated from the owned smoke runner; provenance: remove_unet_sources.json
and training_sources.json. No reference repository is imported.
"""
from __future__ import annotations
import hashlib
import os
from pathlib import Path
import tempfile
import torch
from torch.nn.utils import clip_grad_norm_


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

