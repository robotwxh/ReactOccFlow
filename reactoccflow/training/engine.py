"""Small, deterministic PyTorch training primitives for ReactOccFlow."""
from __future__ import annotations

import math
from contextlib import nullcontext

import torch
import torch.distributed as dist
from torch.nn.utils import clip_grad_norm_
from torch.optim.swa_utils import AveragedModel, get_ema_avg_fn


def build_react_flow_optimizer(flow, *, trunk_lr=2e-5, new_lr=1e-4, weight_decay=0.01):
    """Create distinct learning-rate groups for the migrated trunk and new heads."""
    for name, value in (("trunk_lr", trunk_lr), ("new_lr", new_lr)):
        if not math.isfinite(value) or value <= 0:
            raise ValueError(f"{name} must be finite and positive")
    if not math.isfinite(weight_decay) or weight_decay < 0:
        raise ValueError("weight_decay must be finite and nonnegative")

    trunk, new = [], []
    for name, parameter in flow.named_parameters():
        if not parameter.requires_grad:
            continue
        (trunk if name.startswith("transition_model.trunk.") else new).append(parameter)
    groups = []
    if trunk:
        groups.append({"params": trunk, "lr": trunk_lr, "group_name": "trunk"})
    if new:
        groups.append({"params": new, "lr": new_lr, "group_name": "new_branches"})
    if not groups:
        raise ValueError("flow has no trainable parameters")
    return torch.optim.AdamW(groups, weight_decay=weight_decay)


def build_warmup_cosine_scheduler(optimizer, *, total_steps, warmup_steps=1000, min_lr_ratio=0.01):
    """Linear warmup followed by cosine decay, stepped once per optimizer update."""
    if isinstance(total_steps, bool) or not isinstance(total_steps, int) or total_steps < 1:
        raise ValueError("total_steps must be a positive integer")
    if isinstance(warmup_steps, bool) or not isinstance(warmup_steps, int) or not 0 <= warmup_steps < total_steps:
        raise ValueError("warmup_steps must be an integer in [0, total_steps)")
    if not math.isfinite(min_lr_ratio) or not 0 <= min_lr_ratio <= 1:
        raise ValueError("min_lr_ratio must be finite and in [0, 1]")

    def multiplier(step):
        if warmup_steps and step < warmup_steps:
            return (step + 1) / warmup_steps
        progress = min(1.0, max(0.0, (step - warmup_steps) / (total_steps - warmup_steps)))
        return min_lr_ratio + (1 - min_lr_ratio) * 0.5 * (1 + math.cos(math.pi * progress))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, multiplier)


def create_ema(model, *, decay=0.9999):
    """Make a frozen EMA copy whose first update averages from initial weights."""
    if not math.isfinite(decay) or not 0 < decay < 1:
        raise ValueError("EMA decay must be finite and in (0, 1)")
    ema = AveragedModel(model, avg_fn=get_ema_avg_fn(decay), use_buffers=True)
    ema.n_averaged.fill_(1)
    ema.requires_grad_(False).eval()
    return ema


def synchronize_buffers(model, *, source_rank=0):
    """Synchronize immutable model buffers before creating rank-local EMA copies."""
    if not dist.is_available() or not dist.is_initialized():
        return
    for buffer in model.buffers():
        dist.broadcast(buffer, src=source_rank)


def optimizer_step(model, optimizer, microbatches, *, loss_fn, max_grad_norm=1.0):
    """Run one gradient-accumulated update, syncing DDP only on the final batch."""
    if not microbatches:
        raise ValueError("microbatches must not be empty")
    if not math.isfinite(max_grad_norm) or max_grad_norm <= 0:
        raise ValueError("max_grad_norm must be finite and positive")
    parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
    if not parameters:
        raise ValueError("model has no trainable parameters")

    optimizer.zero_grad(set_to_none=True)
    losses = []
    for index, batch in enumerate(microbatches):
        no_sync = getattr(model, "no_sync", None)
        context = no_sync() if no_sync is not None and index + 1 < len(microbatches) else nullcontext()
        with context:
            loss = loss_fn(model, batch)
            if not torch.is_tensor(loss) or loss.ndim != 0 or not torch.isfinite(loss).item():
                raise RuntimeError(f"loss must be a finite scalar at microbatch {index}")
            if not loss.requires_grad:
                raise RuntimeError(f"loss has no gradient at microbatch {index}")
            (loss / len(microbatches)).backward()
        losses.append(float(loss.detach()))

    gradients = [parameter.grad for parameter in parameters if parameter.grad is not None]
    if not gradients or not all(torch.isfinite(gradient).all().item() for gradient in gradients):
        raise RuntimeError("optimizer step has missing or nonfinite gradients")
    clip_grad_norm_(parameters, max_grad_norm, error_if_nonfinite=True)
    optimizer.step()
    if not all(torch.isfinite(parameter).all().item() for parameter in parameters):
        raise RuntimeError("optimizer step produced nonfinite parameters")
    return sum(losses) / len(losses)
