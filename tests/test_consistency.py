"""T7 regression contracts: continuous gradients, actor-local geometry and masks."""
import copy
import importlib.util

import pytest
import torch
from torch import nn

from reactoccflow.geometry import GridSpec


def test_consistency_api_exists():
    assert importlib.util.find_spec('reactoccflow.losses.consistency') is not None, 'T7 continuous losses are missing'


def fixture():
    grid = GridSpec(shape=(8, 8, 3), origin=(-4., -4., -1.5), voxel_size=(1., 1., 1.))
    actor = torch.tensor([[[[.3, .2, .2, .9, .3, .2], [.5, .4, .2, .9, .2, .2], [.6, .5, .2, .9, .1, .1]]]], requires_grad=True)
    logits = torch.zeros(1, 3, 8, 8, 3, 18, requires_grad=True)
    obs = {
        'actor_history': torch.tensor([[[[0., 0., 0., 1., 0., 0.]]]]),
        'actor_observed_mask': torch.ones(1, 1, 1, dtype=torch.bool),
        'actor_velocity_mask': torch.ones(1, 1, 1, dtype=torch.bool),
        'actor_class': torch.tensor([[4]]), 'actor_size': torch.tensor([[[2., 1., 1.]]]),
        'actor_z': torch.zeros(1, 1),
    }
    target = {'actor_future': actor.detach().clone(),
              'actor_future_z': torch.zeros(1, 1, 3),
              'actor_target_mask': torch.ones(1, 1, 3, dtype=torch.bool),
              'actor_velocity_mask': torch.ones(1, 1, 3, dtype=torch.bool),
              'future_time_s': torch.tensor([[1., 2., 3.]]),
              'future_semantics': torch.full((1, 3, 8, 8, 3), 17, dtype=torch.long),
              'future_known_mask': torch.ones(1, 3, 8, 8, 3, dtype=torch.bool),
              'future_latent_mask': torch.ones(1, 3, dtype=torch.bool)}
    return grid, logits, actor, obs, target


def loss_fixture(**kwargs):
    from reactoccflow.losses import ConsistencyLosses
    grid, logits, actor, obs, target = fixture()
    return ConsistencyLosses(grid=grid, roi_margin=.2, **kwargs), logits, actor, obs, target


def test_geometry_gradient_reaches_actor_and_continuous_voxel_logits():
    loss, logits, actor, obs, target = loss_fixture()
    value = loss(logits, actor, obs, target)['occ_actor_consistency']
    value.backward()
    assert torch.isfinite(value) and value > 0
    assert actor.grad[..., :2].abs().sum() > 0
    assert actor.grad[..., 2:4].abs().sum() > 0
    assert logits.grad.abs().sum() > 0
    changed = logits.detach().clone()
    changed[..., 4] += 3.
    assert not torch.isclose(value, loss(changed, actor, obs, target)['occ_actor_consistency'])


def test_geometry_ignores_voxels_outside_selected_actor_roi():
    loss, logits, actor, obs, target = loss_fixture()
    initial = loss(logits, actor, obs, target)['occ_actor_consistency']
    changed = logits.detach().clone()
    changed[:, :, 0, :, :, 4] = 20.
    torch.testing.assert_close(initial, loss(changed, actor, obs, target)['occ_actor_consistency'], atol=0, rtol=0)
    # A false target/observation slot can carry arbitrary unselected actors.
    obs['actor_observed_mask'].zero_()
    target['actor_future'].fill_(float('nan'))
    target['actor_future_z'].fill_(float('nan'))
    obs['actor_size'].fill_(float('nan'))
    value = loss(logits, actor, obs, target)['occ_actor_consistency']
    assert value == 0 and value.requires_grad


@pytest.mark.parametrize('case', ['all_unknown', 'out_of_bounds', 'all_masks_empty'])
def test_empty_geometry_has_finite_differentiable_zero(case):
    loss, logits, actor, obs, target = loss_fixture()
    if case == 'all_unknown':
        target['future_known_mask'].zero_()
    elif case == 'out_of_bounds':
        target['actor_future'][..., :2] = 1000.
    else:
        target['actor_target_mask'].zero_()
        target['actor_velocity_mask'].zero_()
        target['future_latent_mask'].zero_()
        target['actor_future'].fill_(float('nan'))
        target['actor_future_z'].fill_(float('nan'))
        target['future_time_s'].fill_(float('nan'))
    result = loss(logits, actor, obs, target)
    assert result['occ_actor_consistency'] == 0
    result['total'].backward()
    assert torch.isfinite(logits.grad).all() and torch.isfinite(actor.grad).all()
    if case == 'all_masks_empty':
        assert all(value == 0 and value.requires_grad for value in result.values())


def test_temporal_only_compares_same_known_static_class():
    loss, logits, actor, obs, target = loss_fixture()
    logits = logits.detach().clone()
    logits[:, 1:, ..., 11] = 4.
    logits.requires_grad_()
    target['future_semantics'][:, :, 0, 0, 0] = 11
    static = loss(logits, actor, obs, target)['temporal_consistency']
    assert static > 0
    static.backward()
    assert logits.grad[:, :, 0, 0, 0, 11].abs().sum() > 0
    # Remove both static transitions with unknown and true class change.
    target['future_known_mask'][:, 0, 0, 0, 0] = False
    target['future_semantics'][:, 2, 0, 0, 0] = 12
    assert loss(logits, actor, obs, target)['temporal_consistency'] == 0
    target['future_semantics'].fill_(4)
    assert loss(logits, actor, obs, target)['temporal_consistency'] == 0


def test_required_target_data_is_not_silently_replaced_by_zero():
    loss, logits, actor, obs, target = loss_fixture()
    del target['future_semantics']
    with pytest.raises(ValueError, match='future_semantics'):
        loss(logits, actor, obs, target)
    target['future_semantics'] = torch.full((1, 3, 8, 8, 3), 17)
    target['actor_future'][0, 0, 0, 0] = float('nan')
    with pytest.raises(ValueError, match='finite'):
        loss(logits, actor, obs, target)


def test_kinematics_uses_elapsed_seconds_and_first_current_transition():
    loss, logits, actor, obs, target = loss_fixture()
    # Pedestrian lateral velocity is allowed; every endpoint is 2 m/s in y.
    obs['actor_class'].fill_(7)
    actor = torch.tensor([[[[0., 2., 0., 1., 0., 2.], [0., 6., 0., 1., 0., 2.], [0., 12., 0., 1., 0., 2.]]]], requires_grad=True)
    target['future_time_s'] = torch.tensor([[1., 3., 6.]])
    assert loss(logits, actor, obs, target)['kinematics'] == 0
    target['future_time_s'] = torch.tensor([[.5, 1., 1.5]])
    assert loss(logits, actor, obs, target)['kinematics'] > 0
    target['future_time_s'] = torch.tensor([[1., 3., 6.]])
    obs['actor_history'][..., :2] = torch.tensor([0., -4.])
    assert loss(logits, actor, obs, target)['kinematics'] > 0


def test_class_specific_heading_and_speed_gates_do_not_force_pedestrian_motion():
    loss, logits, actor, obs, target = loss_fixture()
    actor = torch.tensor([[[[0., 2., 0., 1., 0., 2.], [0., 4., 0., 1., 0., 2.], [0., 6., 0., 1., 0., 2.]]]], requires_grad=True)
    obs['actor_class'].fill_(4)
    vehicle = loss(logits, actor, obs, target)['kinematics']
    obs['actor_class'].fill_(2)
    cycle = loss(logits, actor, obs, target)['kinematics']
    obs['actor_class'].fill_(7)
    pedestrian = loss(logits, actor, obs, target)['kinematics']
    assert vehicle > cycle > pedestrian == 0
    torch.testing.assert_close(cycle, vehicle / 2)
    obs['actor_class'].fill_(4)
    slow = actor.detach().clone()
    slow[..., [1, 5]] *= .01
    assert loss(logits, slow, obs, target)['kinematics'] == 0


def test_kinematics_never_bridges_missing_intermediate_pose_or_velocity():
    loss, logits, actor, obs, target = loss_fixture()
    obs['actor_class'].fill_(7)
    target['actor_target_mask'][:, :, 1] = False
    target['actor_velocity_mask'][:, :, 1] = False
    # Only step zero survives; last velocity is missing its immediate start pose.
    actor = torch.tensor([[[[1., 0., 0., 1., 1., 0.], [999., 999., 0., 1., 0., 0.], [20., 0., 0., 1., 100., 0.]]]], requires_grad=True)
    assert loss(logits, actor, obs, target)['kinematics'] == 0
    target['actor_target_mask'].fill_(True)
    target['actor_velocity_mask'].zero_()
    assert loss(logits, actor, obs, target)['kinematics'] == 0


def test_training_endpoints_preserve_default_loss_rng_and_model_call_count():
    from test_react_flow import batch, recording_model
    model = recording_model()
    obs, target, candidate = batch(1)
    generator = torch.Generator().manual_seed(4)
    plain = model.training_loss(obs, target, candidate, generator=generator)
    plain_rng = generator.get_state()
    generator.manual_seed(4)
    enriched = model.training_loss(obs, target, candidate, generator=generator, return_endpoints=True)
    assert set(enriched) == {*plain, 'endpoints'}
    assert torch.equal(generator.get_state(), plain_rng)
    assert len(model.transition_model.calls) == 2
    for key in plain:
        torch.testing.assert_close(enriched[key], plain[key], atol=0, rtol=0)
    endpoints = enriched['endpoints']
    call = model.transition_model.calls[-1]
    torch.testing.assert_close(endpoints['future_latents'], call['noised_sequence'][:, 6:] / 10)
    torch.testing.assert_close(endpoints['actor_future'], call['actor_noised'])
    assert endpoints['future_latents'].requires_grad
    assert endpoints['actor_future'].requires_grad
    (endpoints['future_latents'].square().sum() + endpoints['actor_future'].square().sum()).backward()
    assert model.transition_model.value.grad.abs() > 0


class TinyCodec(nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = nn.Parameter(torch.tensor(2.))

    def decode(self, latents, *, return_logits=False):
        assert return_logits
        return (latents[:, 0, :, :, None, None] * self.weight).expand(-1, -1, -1, 1, 18)


def test_frozen_codec_retains_endpoint_input_gradients_and_logits():
    from reactoccflow.losses import decode_endpoints
    codec = TinyCodec().train()
    latents = torch.ones(2, 3, 1, 2, 2, requires_grad=True)
    decoded = decode_endpoints(codec, latents, chunk=2)
    assert decoded.shape == (2, 3, 2, 2, 1, 18)
    decoded.square().mean().backward()
    assert latents.grad.abs().sum() > 0
    assert codec.weight.grad is None and not codec.weight.requires_grad
    assert not codec.training


def test_geometry_loss_uses_fixed_supervision_roi_and_not_predicted_roi():
    loss, logits, actor, obs, target = loss_fixture()
    original = loss(logits, actor, obs, target)['occ_actor_consistency']
    distant = actor.detach().clone()
    distant[..., 0] = 2.5
    distant.requires_grad_()
    changed = loss(logits, distant, obs, target)['occ_actor_consistency']
    assert changed > original
    changed.backward()
    assert distant.grad[..., 0].abs().sum() > 0
    # Extremely small but finite heading vectors must not produce NaN gradients.
    small_heading = actor.detach().clone()
    small_heading[..., 2:4] = 1e-12
    small_heading.requires_grad_()
    value = loss(logits, small_heading, obs, target)['occ_actor_consistency']
    value.backward()
    assert torch.isfinite(small_heading.grad).all()


def test_known_mask_and_frame_mask_gate_geometry_and_temporal_voxels():
    loss, logits, actor, obs, target = loss_fixture()
    target['future_latent_mask'].zero_()
    result = loss(logits, actor, obs, target)
    assert result['occ_actor_consistency'] == result['temporal_consistency'] == 0
    target['future_known_mask'].zero_()
    # Unknown labels may retain their sentinel without being treated as free.
    target['future_semantics'].fill_(255)
    target['future_latent_mask'].fill_(True)
    result = loss(logits, actor, obs, target)
    assert result['occ_actor_consistency'] == result['temporal_consistency'] == 0


def test_invalid_physical_timestamps_raise_on_valid_transitions():
    loss, logits, actor, obs, target = loss_fixture()
    target['future_time_s'][0, 1] = 0.
    with pytest.raises(ValueError, match='increase'):
        loss(logits, actor, obs, target)


def test_masked_target_nan_pose_and_velocity_does_not_change_valid_losses():
    loss, logits, actor, obs, target = loss_fixture()
    target['actor_target_mask'][..., 1] = False
    target['actor_velocity_mask'][..., 1] = False
    before = loss(logits, actor, obs, target)
    target['actor_future'][..., 1, :] = float('nan')
    target['actor_future_z'][..., 1] = float('nan')
    target['actor_future'][..., 0, 4:] = float('nan')
    after = loss(logits, actor, obs, target)
    for key in before:
        torch.testing.assert_close(after[key], before[key], atol=0, rtol=0)
    after['total'].backward()
    assert torch.isfinite(actor.grad).all() and torch.isfinite(logits.grad).all()


def test_finite_half_precision_logits_have_finite_differentiable_empty_zero():
    loss, logits, actor, obs, target = loss_fixture()
    logits = torch.full_like(logits, 20., dtype=torch.float16, requires_grad=True)
    actor = actor.detach().half().requires_grad_()
    target['future_latent_mask'].zero_()
    target['actor_target_mask'].zero_()
    target['actor_velocity_mask'].zero_()
    result = loss(logits, actor, obs, target)
    assert all(torch.isfinite(value) and value == 0 for value in result.values())
    result['total'].backward()
    assert torch.isfinite(logits.grad).all() and torch.isfinite(actor.grad).all()


def test_half_precision_large_static_temporal_mean_accumulates_stably():
    from reactoccflow.losses import ConsistencyLosses
    grid, _, actor, obs, target = fixture()
    grid = GridSpec(shape=(260, 260, 1), origin=(-130., -130., -.5), voxel_size=(1., 1., 1.))
    logits = torch.full((1, 3, 260, 260, 1, 18), -10., dtype=torch.float16)
    logits[:, 0, ..., 11] = 10.
    logits[:, 1:, ..., 12] = 10.
    logits.requires_grad_()
    target['future_semantics'] = torch.full(logits.shape[:-1], 11, dtype=torch.long)
    target['future_known_mask'] = torch.ones(logits.shape[:-1], dtype=torch.bool)
    target['actor_target_mask'].zero_()
    target['actor_velocity_mask'].zero_()
    value = ConsistencyLosses(grid=grid)(logits, actor, obs, target)['temporal_consistency']
    # First adjacent pair has differences+1/-1 among six classes; second is zero.
    torch.testing.assert_close(value.float(), torch.tensor(1/6), atol=1e-4, rtol=1e-4)
    value.backward()
    assert torch.isfinite(logits.grad).all()
