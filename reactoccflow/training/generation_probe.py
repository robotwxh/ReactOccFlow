"""Fixed tune-subset free-generation monitoring, separate from joint loss.

Reuse of owned count metrics, identity-seeded noise and independent E0 caches
is recorded in docs/provenance/generation_probe_sources_20261107.json. This
module neither imports nor executes either read-only reference repository.
"""
from __future__ import annotations

from contextlib import contextmanager
import math
import random

import numpy as np
import torch
from torch import nn
from torch.utils.data import Subset

from reactoccflow.evaluation.occupancy import OccupancyAccumulator
from reactoccflow.evaluation.reactflow_generation import paper_horizon_summary, window_seed


def _metadata(dataset, index):
    if isinstance(dataset, Subset):
        return _metadata(dataset.dataset, int(dataset.indices[index]))
    if hasattr(dataset, 'records'):
        value = dataset.records[index]
    else:
        value = dataset[index].get('metadata', {})
    if not all(isinstance(value.get(key), str) and value[key] for key in ('scene', 'token')):
        raise ValueError('Generation probe requires explicit scene/token metadata')
    return value


def select_probe_indices(dataset, count=8):
    """Evenly select sorted scene IDs, then their first dataset-relative window.

    E0ResearchDataset records and nested Subset mappings are inspected without
    reading large NPZ payloads. Generic datasets may expose sample metadata.
    Fewer than ``count`` scenes yield one window per available scene.
    """
    if isinstance(count, bool) or not isinstance(count, int) or count < 1:
        raise ValueError('Probe scene count must be a positive integer')
    first = {}
    for index in range(len(dataset)):
        metadata = _metadata(dataset, index)
        first.setdefault(metadata['scene'], index)
    scenes = sorted(first)
    if not scenes:
        raise ValueError('Cannot select generation probes from an empty dataset')
    selected = np.linspace(0, len(scenes)-1, min(count, len(scenes)), dtype=np.int64)
    return [first[scenes[int(position)]] for position in selected]


@contextmanager
def _preserve_training_state(flow, codec, device):
    """Restore mixed module modes, parameter flags and all global RNG streams."""
    modules = list({id(module): module for root in (flow, codec) for module in root.modules()}.values())
    parameters = list({id(parameter): parameter for root in (flow, codec) for parameter in root.parameters()}.values())
    modes = [(module, module.training) for module in modules]
    flags = [(parameter, parameter.requires_grad) for parameter in parameters]
    python_rng, numpy_rng = random.getstate(), np.random.get_state()
    devices = [] if device.type != 'cuda' else [device.index if device.index is not None else torch.cuda.current_device()]
    try:
        with torch.random.fork_rng(devices=devices):
            flow.eval()
            codec.eval()
            for parameter in parameters:
                parameter.requires_grad_(False)
            with torch.inference_mode():
                yield
    finally:
        for module, training in modes:
            module.training = training
        for parameter, requires_grad in flags:
            parameter.requires_grad_(requires_grad)
        random.setstate(python_rng)
        np.random.set_state(numpy_rng)


def _architecture_diagnostics(flow, compute_output_rank):
    trunk = getattr(getattr(flow, 'transition_model', None), 'trunk', None)
    linear = getattr(getattr(trunk, 'final_layer', None), 'linear', None)
    result = {'backbone_class': None if trunk is None else type(trunk).__name__,
              'patch_size': getattr(trunk, 'patch_size', None),
              'hidden_size': getattr(trunk, 'hidden_size', None),
              'patch_values': None, 'rank_capacity': None, 'learned_output_rank': None,
              'forced_noise_variance_floor_at_unit_scale': None,
              'checkpoint_rank_noise_variance_floor_at_unit_scale': None,
              'rank_note': 'Not applicable to a trunk without a fixed final patch-linear head.'}
    if not isinstance(linear, nn.Linear):
        return result
    output, hidden = linear.weight.shape
    capacity = min(output, hidden)
    result.update(patch_values=output, rank_capacity=capacity,
                  forced_noise_variance_floor_at_unit_scale=(output-capacity)/output,
                  rank_note='Dimension hard-cap and current learned rank are distinct. A zero-initialized output naturally has rank0; this alone is not a dimension hard-cap.')
    if compute_output_rank:
        weight = linear.weight.detach().to(device='cpu', dtype=torch.float64)
        if not torch.isfinite(weight).all():
            raise RuntimeError('Nonfinite final projection weights during generation probe')
        singular = torch.linalg.svdvals(weight)
        tolerance = max(weight.shape)*torch.finfo(torch.float64).eps*singular[0]
        rank = int((singular > tolerance).sum())
        result.update(learned_output_rank=rank, output_rank_tolerance=float(tolerance),
                      checkpoint_rank_noise_variance_floor_at_unit_scale=(output-rank)/output)
    scale = float(getattr(flow, 'latent_scale', 1.))
    if not math.isfinite(scale) or scale <= 0:
        raise ValueError('Invalid occupancy latent scale')
    result.update(latent_scale=scale,
                  forced_noise_variance_floor_after_unscale=result['forced_noise_variance_floor_at_unit_scale']/scale**2)
    return result


def _add_moments(state, value):
    value = value.detach().to(device='cpu', dtype=torch.float64)
    if not torch.isfinite(value).all():
        raise RuntimeError('Nonfinite generated/target occupancy latents')
    state['count'] += value.numel()
    state['sum'] += float(value.sum())
    state['sum_squares'] += float(value.square().sum())


def _moments_result(state):
    if state['count'] == 0:
        return {'count': 0, 'mean': None, 'std': None}
    mean = state['sum']/state['count']
    return {'count': state['count'], 'mean': mean,
            'std': math.sqrt(max(0., state['sum_squares']/state['count']-mean**2))}


def evaluate_generation_probe(flow, codec, dataset, indices, *, device,
                              nfe=10, seed=20261107, decode_chunk=2,
                              compute_output_rank=True):
    """Generate fixed tune windows from noise without changing training state.

    Only cached observations and explicitly supplied logged ego XY enter the
    predictor. Future semantics, known masks and latents are scoring/diagnostic
    targets. Counts are merged across windows before computing any mIoU.
    """
    device = torch.device(device)
    indices = list(indices)
    if not indices or any(isinstance(i, bool) or not isinstance(i, (int, np.integer))
                          or not 0 <= i < len(dataset) for i in indices):
        raise ValueError('Probe indices must select existing dataset windows')
    indices = [int(index) for index in indices]
    if len(set(indices)) != len(indices):
        raise ValueError('Duplicate generation probe indices')
    if (isinstance(nfe, bool) or not isinstance(nfe, int) or nfe < 1
            or isinstance(decode_chunk, bool) or not isinstance(decode_chunk, int) or decode_chunk < 1):
        raise ValueError('Probe NFE and decode chunk must be positive integers')
    future = int(flow.future_frames)
    if future != 6:
        raise ValueError('The 1/2/3s generation probe requires six future slots')
    accumulator = OccupancyAccumulator(times=future)
    generated = {'count': 0, 'sum': 0., 'sum_squares': 0.}
    target_moments = dict(generated)
    identities = []
    with _preserve_training_state(flow, codec, device):
        architecture = _architecture_diagnostics(flow, compute_output_rank)
        for index in indices:
            sample = dataset[index]
            metadata = sample['metadata']
            if not all(isinstance(metadata.get(key), str) and metadata[key] for key in ('scene', 'token')):
                raise ValueError('Probe sample requires scene/token identity')
            local_seed = window_seed(seed, metadata['scene'], metadata['token'])
            generator = torch.Generator(device=device).manual_seed(local_seed)
            observation = {key: torch.as_tensor(value, device=device).unsqueeze(0)
                           for key, value in sample['observation'].items()}
            history = observation['history_latents']
            # Extract only the explicitly supplied action condition, never a
            # future occupancy/actor label, before entering the predictor.
            candidate = torch.as_tensor(sample['target']['ego_future'][..., :2], device=device).unsqueeze(0)
            noise = {
                'future_latents': torch.randn((1, 1, future, *flow.latent_shape), device=device,
                                              dtype=history.dtype, generator=generator),
                'actor_future': torch.randn((1, 1, flow.max_actors, future, 6), device=device,
                                            dtype=history.dtype, generator=generator),
            }
            result = flow.sample(observation, candidate, num_worlds=1, generator=generator,
                                 nfe=nfe, initial_noise=noise)
            latents = result['future_latents'][0, 0]
            if tuple(latents.shape) != (future, *flow.latent_shape) or not torch.isfinite(latents).all():
                raise RuntimeError('Generation probe returned invalid/nonfinite occupancy latents')
            predictions = []
            for start in range(0, future, decode_chunk):
                prediction = codec.decode(latents[start:start+decode_chunk])
                if not torch.isfinite(prediction).all():
                    raise RuntimeError('Nonfinite decoded occupancy in generation probe')
                predictions.append(prediction.detach().to(device='cpu', dtype=torch.uint8))
            prediction = torch.cat(predictions).numpy()
            target = sample['target']
            semantics = torch.as_tensor(target['future_semantics']).cpu().numpy()
            known = torch.as_tensor(target['future_known_mask']).bool().cpu().numpy()
            accumulator.update(prediction, semantics, visibility=known)
            valid_frames = torch.as_tensor(target['future_latent_mask'], device=device).bool()
            truth_latents = torch.as_tensor(target['future_latents'], device=device)
            _add_moments(generated, latents[valid_frames])
            _add_moments(target_moments, truth_latents[valid_frames])
            identities.append({'dataset_index': index, 'scene': metadata['scene'],
                               'token': metadata['token'], 'noise_seed': local_seed})
    generated_stats, target_stats = _moments_result(generated), _moments_result(target_moments)
    return {'scope': 'fixed_tune_generation_probe', 'official_validation': False,
            'full_tune_evaluation': False, 'windows': len(indices),
            'scenes': len({item['scene'] for item in identities}), 'indices': indices,
            'identities': identities, 'nfe': nfe, 'seed': seed,
            'num_worlds_per_condition': 1, 'decode_chunk': decode_chunk,
            'condition': 'oracle_cached_history_and_explicit_logged_future_ego_xy',
            'native_e0_known': paper_horizon_summary(accumulator),
            'metric_protocol': {'semantic_classes': '0..16; exclude empty17',
                                'ignore_gt_label': 255,
                                'visibility': 'future_known_mask & GT!=255',
                                'aggregation': 'sum whole-probe per-horizon counts, then mean mIoU at slots2,4,6',
                                'primary': 'standard',
                                'legacy_absent_gt_rule': 'IoU=1 when class has no GT; retained only for source compatibility and can inflate small probes'},
            'generated_latents': generated_stats, 'target_latents': target_stats,
            'generated_latent_std': generated_stats['std'], 'target_latent_std': target_stats['std'],
            'latent_std_definition': 'Population std over selected valid future latent elements; not an ensemble noise-variance estimate.',
            'architecture': architecture,
            'interpretation': 'Small fixed tune-scene diagnostic. This is neither the full tune set nor the official held-out mIoU benchmark.'}
