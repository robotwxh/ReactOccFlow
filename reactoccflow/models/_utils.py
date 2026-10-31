"""Small utilities extracted from the copied common_utils snapshot."""
from collections.abc import Iterable
from itertools import repeat
from pathlib import Path
import torch


def to_2tuple(value):
    # Source: common_utils._ntuple(2); provenance records the original file hash.
    if isinstance(value, Iterable) and not isinstance(value, str):
        return tuple(value)
    return tuple(repeat(value, 2))


def read_state_dict(path, *, weights="state_dict"):
    """Load tensor checkpoints on CPU, stripping only documented wrapper prefixes."""
    checkpoint = torch.load(Path(path), map_location='cpu', weights_only=True)
    if weights not in ('state_dict', 'ema_model'):
        raise ValueError('weights must be state_dict or ema_model')
    if weights == 'ema_model' and weights not in checkpoint:
        raise ValueError('checkpoint has no ema_model weights')
    state = checkpoint.get(weights, checkpoint)
    if weights == 'ema_model':
        state = {key: value for key, value in state.items() if key != 'n_averaged'}
    result = {}
    for key, value in state.items():
        while key.startswith(('module.', 'model.')):
            key = key.split('.', 1)[1]
        if key in result:
            raise ValueError(f'Checkpoint prefix mapping collision: {key}')
        result[key] = value
    return result


def strict_load(module, state):
    result = module.load_state_dict(state, strict=True)
    return {'missing_keys': list(result.missing_keys), 'unexpected_keys': list(result.unexpected_keys)}
