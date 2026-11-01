"""Shared OccFM straight probability path and shifted grid; no network."""
import torch


def interpolate(noise, target, time):
    """The upstream straight training path: t*target + (1-t)*noise."""
    if noise.shape != target.shape:
        raise ValueError('noise and target shapes must match')
    time = torch.as_tensor(time, device=noise.device, dtype=noise.dtype)
    if time.ndim == 1:
        time = time.reshape(-1, *([1] * (noise.ndim - 1)))
    return time * target + (1 - time) * noise


def shifted_time_grid(nfe, alpha=3., *, device=None, dtype=torch.float32):
    if not isinstance(nfe, int) or nfe < 1 or alpha <= 0:
        raise ValueError('nfe must be positive integer and alpha positive')
    time = torch.linspace(0, 1, nfe + 1, device=device, dtype=dtype)
    return (1 - alpha * time / (1 + (alpha - 1) * time)).flip(0)
