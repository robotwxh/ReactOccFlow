"""Validation aggregates each masked mean with its own sufficient statistics."""
import copy

import pytest
import torch
from torch import nn
from torch.nn import functional as F

from reactoccflow.geometry import GridSpec
from reactoccflow.losses import ConsistencyLosses
from reactoccflow.models.react_flow import ReactFlow
from reactoccflow.training.joint import JointTrainingAdapter


class SmallField(nn.Module):
    def __init__(self):
        super().__init__()
        self.occ = nn.Parameter(torch.tensor(.2))
        self.actor = nn.Parameter(torch.tensor(.1))
        self.existence = nn.Parameter(torch.tensor(.4))

    def forward(self, data):
        return {
            'predicted_latent': self.occ + .3 * data['noised_sequence'],
            'actor_velocity': self.actor + .2 * data['actor_noised'],
            'actor_existence_logits': self.existence.expand_as(data['actor_noised'][..., 0]),
        }


class SmallCodec(nn.Module):
    def decode(self, latents, *, return_logits=False):
        assert return_logits
        features = latents[:, 0, :, :, None, None]
        classes = torch.arange(18, device=latents.device, dtype=latents.dtype)
        return features * classes


def fixture(mode='joint'):
    flow = ReactFlow(mode=mode, transition_model=SmallField(), latent_shape=(1, 2, 2),
                     history_frames=1, future_frames=3, actor_history_frames=1,
                     max_actors=2, latent_scale=1., occ_weight=.7,
                     actor_weight=1.3, existence_weight=.2)
    observation = {
        'history_latents': torch.zeros(3, 1, 1, 2, 2),
        'history_mask': torch.ones(3, 1, dtype=torch.bool),
        'actor_history': torch.zeros(3, 2, 1, 6),
        'actor_observed_mask': torch.ones(3, 2, 1, dtype=torch.bool),
        'actor_velocity_mask': torch.ones(3, 2, 1, dtype=torch.bool),
        'actor_size': torch.full((3, 2, 3), 2.),
        'actor_class': torch.tensor([[4, 2], [1, 4], [4, -1]]),
        'actor_z': torch.zeros(3, 2),
    }
    observation['actor_history'][..., 3] = 1
    observation['actor_observed_mask'][2, 1] = False
    actor = torch.zeros(3, 2, 3, 6)
    actor[..., 0] = torch.tensor([.1, .2, .3])
    actor[..., 3] = 1
    actor[..., 4] = torch.tensor([1., 4., 0.])[:, None, None]
    actor[..., 5] = torch.tensor([.2, .7, 0.])[:, None, None]
    actor[0, 1, 2, 0] = 100.  # Supervised but outside the geometry grid.
    poses = torch.tensor([[[1, 1, 1], [1, 1, 1]],
                          [[1, 0, 0], [0, 0, 0]],
                          [[1, 1, 0], [1, 1, 1]]], dtype=torch.bool)
    velocity = poses.clone()
    velocity[0, 1, 2] = False
    velocity[1] = False
    velocity[2, 0, 1] = False
    existence_mask = poses.clone()
    existence_mask[1, 0] = True
    known = torch.ones(3, 3, 2, 2, 1, dtype=torch.bool)
    known[1] = False
    known[1, 0, 0, 0, 0] = True
    known[2, 1:] = False
    known[2, 1, 0, 0, 0] = True
    semantics = torch.full(known.shape, 11, dtype=torch.long)
    semantics[0, 1, 0, 0, 0] = 12
    target = {
        'future_latents': torch.arange(36, dtype=torch.float32).reshape(3, 3, 1, 2, 2) / 20,
        'future_latent_mask': torch.tensor([[1, 1, 1], [1, 0, 0], [1, 1, 0]], dtype=torch.bool),
        'actor_future': actor,
        'actor_target_mask': poses,
        'actor_velocity_mask': velocity,
        'actor_future_exists': poses.clone(),
        'actor_existence_mask': existence_mask,
        'actor_future_z': torch.zeros(3, 2, 3),
        'future_time_s': torch.tensor([[1., 2., 3.]]).expand(3, -1),
        'future_semantics': semantics,
        'future_known_mask': known,
    }
    grid = GridSpec(shape=(2, 2, 1), origin=(-1., -1., -.5), voxel_size=(1., 1., 1.))
    consistency = ConsistencyLosses(grid=grid, occ_actor_weight=.13,
                                    temporal_weight=.17, kinematics_weight=.19,
                                    low_speed_threshold=.1)
    return flow, SmallCodec(), consistency, observation, target, torch.zeros(3, 3, 2)


def mean_from_statistic(statistic):
    return statistic['numerator'] / statistic['count'].clamp_min(1)


def total_from_statistics(statistics):
    return sum(item['weight'] * mean_from_statistic(item) for item in statistics.values())


def assert_detached_statistics(statistics):
    for item in statistics.values():
        assert set(item) == {'numerator', 'count', 'weight'}
        assert item['numerator'].ndim == item['count'].ndim == 0
        assert not item['numerator'].requires_grad and not item['count'].requires_grad
        assert isinstance(item['weight'], float)


@pytest.mark.parametrize('mode', ['joint', 'parallel', 'occupancy_only'])
def test_flow_statistics_keep_original_masked_formulas_and_counts(mode):
    flow, _, _, observation, target, candidate = fixture(mode)
    ordinary = flow.training_loss(observation, target, candidate,
                                  generator=torch.Generator().manual_seed(23))
    actual = flow.training_loss(observation, target, candidate,
                                generator=torch.Generator().manual_seed(23),
                                return_statistics=True)
    for key in ordinary:
        torch.testing.assert_close(actual[key], ordinary[key], atol=0, rtol=0)
    statistics = actual['statistics']
    assert_detached_statistics(statistics)
    assert statistics['occ_fm']['count'] == 24
    assert statistics['actor_fm']['count'] == (0 if mode == 'occupancy_only' else 48)
    assert statistics['actor_existence']['count'] == (0 if mode == 'occupancy_only' else 11)
    # Independently recreate the original public scalar loss formulas.
    generator = torch.Generator().manual_seed(23)
    time = torch.sigmoid(torch.randn(3, generator=generator))
    occ_noise = torch.randn(3, 3, 1, 2, 2, generator=generator)
    actor_noise = torch.randn(3, 2, 3, 6, generator=generator)
    occ_mask = target['future_latent_mask'][..., None, None, None].expand_as(occ_noise)
    clean_occ = torch.where(occ_mask, target['future_latents'], 0.)
    t = time[:, None, None, None, None]
    predicted_occ = .2 + .3 * (t * clean_occ + (1 - t) * occ_noise)
    expected_occ = (predicted_occ - (clean_occ - occ_noise)).square()[occ_mask].mean()
    torch.testing.assert_close(actual['occ_fm'], expected_occ)
    if mode != 'occupancy_only':
        selected = observation['actor_observed_mask'][..., -1, None]
        poses = target['actor_target_mask'] & selected
        velocities = target['actor_velocity_mask'] & poses
        mask = torch.cat((poses[..., None].expand(-1, -1, -1, 4),
                          velocities[..., None].expand(-1, -1, -1, 2)), -1)
        clean_actor = torch.where(mask, target['actor_future'], 0.)
        t = time[:, None, None, None]
        predicted_actor = .1 + .2 * (t * clean_actor + (1 - t) * actor_noise)
        expected_actor = (predicted_actor - (clean_actor - actor_noise)).square()[mask].mean()
        exists = target['actor_existence_mask'] & selected
        expected_existence = F.binary_cross_entropy_with_logits(
            torch.full_like(actor_noise[..., 0], .4), target['actor_future_exists'].float(),
            reduction='none')[exists].mean()
        torch.testing.assert_close(actual['actor_fm'], expected_actor)
        torch.testing.assert_close(actual['actor_existence'], expected_existence)
    for key, item in statistics.items():
        torch.testing.assert_close(mean_from_statistic(item), actual[key], atol=0, rtol=0)
    torch.testing.assert_close(total_from_statistics(statistics), ordinary['total'], atol=0, rtol=0)


def test_consistency_statistics_keep_roi_temporal_and_kinematic_denominators_separate():
    _, _, consistency, observation, target, _ = fixture()
    logits = torch.arange(3 * 3 * 2 * 2 * 18, dtype=torch.float32).reshape(3, 3, 2, 2, 1, 18) / 100
    logits.requires_grad_()
    actor = target['actor_future'].clone().requires_grad_()
    ordinary = consistency(logits, actor, observation, target)
    actual = consistency(logits, actor, observation, target, return_statistics=True)
    for key in ordinary:
        torch.testing.assert_close(actual[key], ordinary[key], atol=0, rtol=0)
    statistics = actual['statistics']
    assert_detached_statistics(statistics)
    assert {name: item['count'].item() for name, item in statistics.items()} == {
        'occ_actor_consistency': 8, 'temporal_consistency': 42,
        'kinematic_velocity': 12, 'heading_alignment': 5,
    }
    for key in ('occ_actor_consistency', 'temporal_consistency'):
        torch.testing.assert_close(mean_from_statistic(statistics[key]), ordinary[key])
    torch.testing.assert_close(mean_from_statistic(statistics['kinematic_velocity'])
                               + mean_from_statistic(statistics['heading_alignment']), ordinary['kinematics'])
    torch.testing.assert_close(total_from_statistics(statistics), ordinary['total'])
    ordinary['total'].backward()
    assert actor.grad is not None and logits.grad is not None


@pytest.mark.parametrize('cuts', [(1, 3), (2, 3), (1, 2, 3)])
def test_joint_statistics_are_invariant_to_batch_partition_with_fixed_noise(monkeypatch, cuts):
    flow, codec, consistency, observation, target, candidate = fixture()
    adapter = JointTrainingAdapter(flow, codec, consistency=consistency, require_frozen_trunk=False)
    # Hold each example's stochastic realization fixed while changing batch grouping.
    monkeypatch.setattr(torch, 'randn', lambda *shape, **kwargs: torch.zeros(
        *shape, device=kwargs.get('device'), dtype=kwargs.get('dtype')))
    full = adapter.validation_statistics(observation, target, candidate, generator=torch.Generator())
    full_loss = adapter.training_loss(observation, target, candidate, generator=torch.Generator())
    assert_detached_statistics(full)
    assert len(full) == 7
    totals = {name: {'numerator': torch.zeros_like(item['numerator']),
                     'count': torch.zeros_like(item['count']), 'weight': item['weight']}
              for name, item in full.items()}
    old_batch_weighted_total = 0.
    start = 0
    for stop in cuts:
        obs_part = {key: value[start:stop] for key, value in observation.items()}
        target_part = {key: value[start:stop] for key, value in target.items()}
        partial = adapter.validation_statistics(obs_part, target_part, candidate[start:stop],
                                                generator=torch.Generator())
        old_batch_weighted_total += (stop - start) * adapter.training_loss(
            obs_part, target_part, candidate[start:stop], generator=torch.Generator())
        for name, item in partial.items():
            totals[name]['numerator'] += item['numerator']
            totals[name]['count'] += item['count']
            assert item['weight'] == full[name]['weight']
        start = stop
    for name in full:
        torch.testing.assert_close(totals[name]['numerator'], full[name]['numerator'])
        torch.testing.assert_close(totals[name]['count'], full[name]['count'], atol=0, rtol=0)
    torch.testing.assert_close(total_from_statistics(totals), full_loss)
    assert not torch.isclose(old_batch_weighted_total / 3, full_loss, atol=1e-5, rtol=1e-5)


def test_empty_joint_statistics_keep_zero_counts_and_fixed_weights():
    flow, codec, consistency, observation, target, candidate = fixture()
    observation = copy.deepcopy(observation)
    target = copy.deepcopy(target)
    observation['actor_observed_mask'].zero_()
    for key in ('future_latent_mask', 'future_known_mask', 'actor_target_mask',
                'actor_velocity_mask', 'actor_existence_mask'):
        target[key].zero_()
    adapter = JointTrainingAdapter(flow, codec, consistency=consistency, require_frozen_trunk=False)
    statistics = adapter.validation_statistics(observation, target, candidate,
                                               generator=torch.Generator().manual_seed(31))
    assert_detached_statistics(statistics)
    assert {name: item['weight'] for name, item in statistics.items()} == {
        'occ_fm': .7, 'actor_fm': 1.3, 'actor_existence': .2,
        'occ_actor_consistency': .13, 'temporal_consistency': .17,
        'kinematic_velocity': .19, 'heading_alignment': .19,
    }
    assert all(item['numerator'] == 0 and item['count'] == 0 for item in statistics.values())
    assert total_from_statistics(statistics) == 0


def test_statistics_reuse_one_field_evaluation_and_preserve_loss_gradients_and_rng():
    flow, codec, consistency, observation, target, candidate = fixture()
    adapter = JointTrainingAdapter(flow, codec, consistency=consistency, require_frozen_trunk=False)
    calls = []
    handle = flow.transition_model.register_forward_hook(lambda *args: calls.append(None))
    generator = torch.Generator().manual_seed(41)
    ordinary = adapter.training_loss(observation, target, candidate, generator=generator)
    ordinary_components = adapter.last_components.copy()
    ordinary_gradients = torch.autograd.grad(ordinary, tuple(flow.parameters()))
    expected_rng_state = generator.get_state()
    calls.clear()
    generator = torch.Generator().manual_seed(41)
    actual, statistics = adapter.training_loss(observation, target, candidate,
                                                generator=generator, return_statistics=True)
    actual_gradients = torch.autograd.grad(actual, tuple(flow.parameters()))
    assert len(calls) == 1
    assert torch.equal(generator.get_state(), expected_rng_state)
    assert adapter.last_components == ordinary_components
    torch.testing.assert_close(actual, ordinary, atol=0, rtol=0)
    for actual_gradient, ordinary_gradient in zip(actual_gradients, ordinary_gradients):
        torch.testing.assert_close(actual_gradient, ordinary_gradient, atol=0, rtol=0)
    assert_detached_statistics(statistics)
    calls.clear()
    generator = torch.Generator().manual_seed(41)
    validation = adapter.validation_statistics(observation, target, candidate, generator=generator)
    assert len(calls) == 1
    assert torch.equal(generator.get_state(), expected_rng_state)
    for name, item in validation.items():
        torch.testing.assert_close(item['numerator'], statistics[name]['numerator'], atol=0, rtol=0)
        torch.testing.assert_close(item['count'], statistics[name]['count'], atol=0, rtol=0)
    handle.remove()
