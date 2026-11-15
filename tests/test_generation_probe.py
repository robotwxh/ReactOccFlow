"""Generation monitoring must preserve training state and target boundaries."""
from copy import deepcopy
import json
import random

import numpy as np
import pytest
import torch
from torch import nn
from torch.utils.data import Dataset, Subset

from reactoccflow.training.generation_probe import evaluate_generation_probe, select_probe_indices


class Samples(Dataset):
    def __init__(self, entries):
        self.entries = entries

    def __len__(self):
        return len(self.entries)

    def __getitem__(self, index):
        return self.entries[index]


def sample(scene='scene-a', token='a0', marker=0.):
    return {'metadata': {'scene': scene, 'token': token},
            'observation': {'history_latents': np.full((6, 1, 2, 2), marker, np.float32),
                            'history_mask': np.array([False, False, True, True, True, True]),
                            'actor_history': np.zeros((1, 4, 6), np.float32)},
            'target': {'ego_future': np.zeros((6, 6), np.float32),
                       'actor_future': np.full((1, 6, 6), 999., np.float32),
                       'future_latents': np.zeros((6, 1, 2, 2), np.float32),
                       'future_latent_mask': np.ones(6, bool),
                       'future_semantics': np.zeros((6, 2, 2, 1), np.uint8),
                       'future_known_mask': np.ones((6, 2, 2, 1), bool)}}


class ToyFlow(nn.Module):
    future_frames, max_actors, latent_shape, latent_scale = 6, 1, (1, 2, 2), 10.

    def __init__(self, *, mode='noise', fail=False):
        super().__init__()
        self.weight = nn.Parameter(torch.tensor(.5))
        self.dropout = nn.Dropout(.5)
        self.mode, self.fail, self.calls = mode, fail, []

    def sample(self, observation, candidate, *, num_worlds, generator, nfe, initial_noise):
        assert not self.training and not self.dropout.training
        assert not torch.is_grad_enabled() and torch.is_inference_mode_enabled()
        assert all(not p.requires_grad for p in self.parameters())
        assert set(observation) == {'history_latents', 'history_mask', 'actor_history'}
        assert candidate.shape == (1, 6, 2)
        assert num_worlds == 1 and nfe == 10
        self.calls.append({'occupancy_noise': initial_noise['future_latents'].clone(),
                           'actor_noise': initial_noise['actor_future'].clone(),
                           'candidate': candidate.clone()})
        # Even an accidental global RNG draw inside a model must not perturb
        # the real trainer's streams after this monitoring call returns.
        torch.rand(3)
        np.random.rand()
        random.random()
        if self.fail:
            raise RuntimeError('deliberate generation failure')
        latents = initial_noise['future_latents'].clone()
        if self.mode == 'condition':
            value = 1. if observation['history_latents'][0, -1, 0, 0, 0] > 0 else -1.
            latents.fill_(value)
        return {'future_latents': latents}


class ToyCodec(nn.Module):
    def __init__(self):
        super().__init__()
        self.gain = nn.Parameter(torch.tensor(1.), requires_grad=False)
        self.dropout = nn.Dropout(.7)

    def encode(self, *args, **kwargs):
        raise AssertionError('Cached generation probes must not VAE-encode future or history data')

    def decode(self, latents):
        assert not self.training and not self.dropout.training
        assert not torch.is_grad_enabled()
        return (latents[:, 0] > 0).long()[..., None]


def test_scene_selection_uses_metadata_without_loading_all_cache_payloads():
    class RecordsOnly:
        records = [{'scene': scene, 'token': str(i)} for i, scene in enumerate(('b', 'a', 'd', 'a', 'c', 'e'))]
        def __len__(self):
            return len(self.records)
        def __getitem__(self, index):
            raise AssertionError('Selection should inspect E0 cache records only')
    dataset = RecordsOnly()
    assert select_probe_indices(dataset, 3) == [1, 4, 5]
    subset = Subset(dataset, [5, 4, 0, 3])
    assert select_probe_indices(subset, 2) == [3, 0]
    nested = Subset(subset, [0, 2, 3])
    assert select_probe_indices(nested, 8) == [2, 1, 0]


def test_generic_dataset_selection_is_deterministic_and_keeps_one_first_window_per_scene():
    dataset = Samples([sample('c', 'c0'), sample('a', 'a0'), sample('b', 'b0'), sample('a', 'a1')])
    assert select_probe_indices(dataset, 2) == [1, 0]
    assert select_probe_indices(dataset, 8) == [1, 2, 0]
    with pytest.raises(ValueError, match='empty'):
        select_probe_indices(Samples([]))


def test_future_targets_are_never_predictor_inputs_and_fixed_noise_replays():
    data = sample()
    data['target']['ego_future'][:, :2] = np.array([3., 4.])
    dataset = Samples([data])
    flow, codec = ToyFlow(), ToyCodec()
    first = evaluate_generation_probe(flow, codec, dataset, [0], device='cpu')
    data['target']['future_semantics'].fill(1)
    data['target']['future_latents'][:] = np.arange(24).reshape(6, 1, 2, 2)
    data['target']['actor_future'].fill(-999.)
    second = evaluate_generation_probe(flow, codec, dataset, [0], device='cpu')
    for key in ('occupancy_noise', 'actor_noise', 'candidate'):
        torch.testing.assert_close(flow.calls[0][key], flow.calls[1][key], rtol=0, atol=0)
    torch.testing.assert_close(flow.calls[0]['candidate'], torch.tensor([[[3., 4.]]*6]))
    assert first['target_latent_std'] != second['target_latent_std']
    assert first['generated_latent_std'] == second['generated_latent_std']
    assert not first['official_validation'] and not first['full_tune_evaluation']
    json.dumps(second, allow_nan=False)


@pytest.mark.parametrize('fail', [False, True])
def test_probe_restores_global_rng_mixed_modes_grad_flags_and_existing_gradients(fail):
    flow, codec = ToyFlow(fail=fail), ToyCodec()
    flow.train()
    flow.dropout.eval()
    codec.eval()
    codec.dropout.train()
    flow.weight.grad = torch.tensor(.75)
    states = [(m, m.training) for root in (flow, codec) for m in root.modules()]
    flags = [(p, p.requires_grad, None if p.grad is None else p.grad.clone()) for root in (flow, codec) for p in root.parameters()]
    torch_rng, numpy_rng, python_rng = torch.get_rng_state().clone(), np.random.get_state(), random.getstate()
    if fail:
        with pytest.raises(RuntimeError, match='deliberate'):
            evaluate_generation_probe(flow, codec, Samples([sample()]), [0], device='cpu')
    else:
        evaluate_generation_probe(flow, codec, Samples([sample()]), [0], device='cpu')
    assert all(module.training == training for module, training in states)
    for parameter, requires_grad, gradient in flags:
        assert parameter.requires_grad == requires_grad
        if gradient is None:
            assert parameter.grad is None
        else:
            torch.testing.assert_close(parameter.grad, gradient, rtol=0, atol=0)
    torch.testing.assert_close(torch.get_rng_state(), torch_rng, rtol=0, atol=0)
    after = np.random.get_state()
    assert after[0] == numpy_rng[0] and after[2:] == numpy_rng[2:]
    np.testing.assert_array_equal(after[1], numpy_rng[1])
    assert random.getstate() == python_rng


def test_probe_aggregates_counts_before_miou_instead_of_averaging_windows():
    first, second = sample('a', 'a0', marker=0.), sample('b', 'b0', marker=1.)
    first['target']['future_known_mask'][:] = False
    first['target']['future_known_mask'][:, 0, 0, 0] = True
    result = evaluate_generation_probe(ToyFlow(mode='condition'), ToyCodec(), Samples([first, second]), [0, 1], device='cpu')
    score = result['native_e0_known']
    assert score['windows'] == 2 and score['valid_voxels'] == [5]*6
    assert score['standard']['miou_percent_at_1_2_3s'] == pytest.approx([10.]*3)
    assert score['standard']['miou_percent_mean_1_2_3s'] == pytest.approx(10.)
    assert score['legacy']['miou_percent_mean_1_2_3s'] > 10.


def test_probe_noise_depends_on_identity_not_subset_order():
    a, b = sample('a', 'a0'), sample('b', 'b0')
    flow, codec = ToyFlow(), ToyCodec()
    evaluate_generation_probe(flow, codec, Samples([a, b]), [0, 1], device='cpu')
    original = flow.calls[1]
    evaluate_generation_probe(flow, codec, Subset(Samples([a, b]), [1]), [0], device='cpu')
    torch.testing.assert_close(original['occupancy_noise'], flow.calls[-1]['occupancy_noise'], rtol=0, atol=0)
    torch.testing.assert_close(original['actor_noise'], flow.calls[-1]['actor_noise'], rtol=0, atol=0)


@pytest.mark.parametrize('hidden,forced', [(2, .5), (6, 0.)])
def test_dimension_hard_cap_is_distinguished_from_natural_zero_initialized_rank(hidden, forced):
    flow, codec = ToyFlow(), ToyCodec()
    trunk = nn.Module()
    trunk.patch_size, trunk.hidden_size = 2, hidden
    trunk.final_layer = nn.Module()
    trunk.final_layer.linear = nn.Linear(hidden, 4)
    nn.init.zeros_(trunk.final_layer.linear.weight)
    flow.transition_model = nn.Module()
    flow.transition_model.trunk = trunk
    result = evaluate_generation_probe(flow, codec, Samples([sample()]), [0], device='cpu')
    architecture = result['architecture']
    assert architecture['rank_capacity'] == min(hidden, 4)
    assert architecture['learned_output_rank'] == 0
    assert architecture['forced_noise_variance_floor_at_unit_scale'] == forced
    assert architecture['checkpoint_rank_noise_variance_floor_at_unit_scale'] == 1.
    assert architecture['forced_noise_variance_floor_after_unscale'] == forced/100


def test_unknown_gt_and_known_mask_are_scoring_only():
    data = sample()
    data['target']['future_semantics'][:, 0, 0, 0] = 255
    data['target']['future_known_mask'][:, 1, 0, 0] = False
    result = evaluate_generation_probe(ToyFlow(), ToyCodec(), Samples([data]), [0], device='cpu')
    assert result['native_e0_known']['valid_voxels'] == [2]*6
    assert result['metric_protocol']['ignore_gt_label'] == 255
