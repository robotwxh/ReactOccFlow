"""T6 contracts: no future conditioning, safe masks, and real DiT token exchange."""
import importlib.util


def test_research_react_flow_api_exists():
    assert importlib.util.find_spec('reactoccflow.models.react_flow') is not None, 'T6 ReactFlow API is missing'

import copy
from pathlib import Path
import pytest
import torch
from torch import nn

# CPU tests stay small; full default spatial smoke is an explicit opt-in.
torch.set_num_threads(2)


def batch(batch_size=2, actors=3):
    g = torch.Generator().manual_seed(71)
    obs = {
        'history_latents': torch.randn(batch_size, 6, 2, 4, 4, generator=g),
        'history_mask': torch.tensor([[False, False, True, True, True, True]]).expand(batch_size, -1).clone(),
        'actor_history': torch.randn(batch_size, actors, 4, 6, generator=g),
        'actor_observed_mask': torch.ones(batch_size, actors, 4, dtype=torch.bool),
        'actor_velocity_mask': torch.ones(batch_size, actors, 4, dtype=torch.bool),
        'actor_size': torch.ones(batch_size, actors, 3),
        'actor_class': torch.full((batch_size, actors), 4, dtype=torch.long),
    }
    obs['actor_history'][..., 2:4] = torch.tensor([.6, .8])
    obs['actor_observed_mask'][:, -1] = False
    obs['actor_velocity_mask'][:, -1] = False
    obs['actor_class'][:, -1] = -1
    target = {
        'future_latents': torch.randn(batch_size, 6, 2, 4, 4, generator=g),
        'future_latent_mask': torch.tensor([[True, True, True, False, False, False]]).expand(batch_size, -1).clone(),
        'actor_future': torch.randn(batch_size, actors, 6, 6, generator=g),
        'actor_target_mask': torch.ones(batch_size, actors, 6, dtype=torch.bool),
        'actor_velocity_mask': torch.ones(batch_size, actors, 6, dtype=torch.bool),
        'actor_future_exists': torch.ones(batch_size, actors, 6, dtype=torch.bool),
        'actor_existence_mask': torch.ones(batch_size, actors, 6, dtype=torch.bool),
    }
    target['actor_target_mask'][:, -1] = False
    target['actor_velocity_mask'][:, :, -2:] = False
    target['actor_existence_mask'][:, -1] = False
    target['actor_future_exists'][:, :, -1] = False
    target['actor_target_mask'][:, :, -1] = False
    return obs, target, torch.zeros(batch_size, 6, 2)


class RecordingTransition(nn.Module):
    """Deterministic velocity field keeps loss/ODE assertions hand-checkable."""
    def __init__(self, value=0.):
        super().__init__()
        self.value = nn.Parameter(torch.tensor(float(value)))
        self.calls = []

    def forward(self, data):
        self.calls.append(data)
        return {'predicted_latent': torch.zeros_like(data['noised_sequence']) + self.value,
                'actor_velocity': torch.zeros_like(data['actor_noised']) + self.value,
                'actor_existence_logits': torch.zeros_like(data['actor_noised'][..., 0]) + self.value}


def small_model(mode='joint', *, freeze=False):
    from reactoccflow.models.react_flow import ReactFlow
    from reactoccflow.models.standard_dit import StandardDiT
    trunk = StandardDiT(in_channels=2, input_size=4, frames=12,
                        patch_size=2, hidden_size=32, depth=2, num_heads=4,
                        activation_checkpointing=False)
    # Emulate a learned field for coupling checks; pristine zero initialization
    # has its own architecture test in test_standard_dit.py.
    with torch.no_grad():
        trunk.final_layer.linear.weight.normal_(std=.02)
        trunk.final_layer.modulation[-1].weight.normal_(std=.02)
        for block in trunk.blocks:
            block.adaLN_modulation[-1].weight.normal_(std=.02)
    return ReactFlow(mode=mode, trunk=trunk, latent_shape=(2, 4, 4),
                     max_actors=3, freeze_backbone=freeze)


def recording_model(**kwargs):
    from reactoccflow.models.react_flow import ReactFlow
    return ReactFlow(transition_model=RecordingTransition(), latent_shape=(2, 4, 4), max_actors=3, **kwargs)


def flow_inputs(obs, candidate, *, occ_value=.2, actor_value=.3):
    return {'noised_sequence': torch.cat((obs['history_latents'], torch.full_like(obs['history_latents'], occ_value)), 1),
            'timesteps': torch.full((obs['history_latents'].shape[0],), 300.),
            'trajectory': candidate,
            'actor_noised': torch.full((obs['history_latents'].shape[0], 3, 6, 6), actor_value),
            **{key: value for key, value in obs.items() if key.startswith('actor_')}}


@pytest.mark.parametrize('mode', ['joint', 'parallel', 'occupancy_only'])
def test_modes_sample_shapes_and_reproducibility_without_global_rng(mode):
    model = small_model(mode).eval()
    obs, _, candidate = batch(1)
    state = torch.random.get_rng_state()
    a = model.sample(obs, candidate, generator=torch.Generator().manual_seed(4), num_worlds=2, nfe=2)
    b = model.sample(obs, candidate, generator=torch.Generator().manual_seed(4), num_worlds=2, nfe=2)
    assert a['future_latents'].shape == (1, 2, 6, 2, 4, 4)
    assert a['actor_future'].shape == (1, 2, 3, 6, 6)
    assert a['actor_existence'].shape == (1, 2, 3, 6)
    for key in a:
        assert torch.equal(a[key], b[key])
        assert torch.isfinite(a[key]).all()
    assert torch.equal(state, torch.random.get_rng_state())
    assert not a['actor_future'][:, :, -1].any()
    assert not a['actor_existence'][:, :, -1].any()


@pytest.mark.parametrize('mode', ['joint', 'parallel'])
def test_token_exchange_is_bidirectional_only_in_joint(mode):
    model = small_model(mode).eval()
    obs, _, candidate = batch(1)
    data = flow_inputs(obs, candidate)
    original = model.transition_model(data)
    changed_actor = {**data, 'actor_noised': data['actor_noised'] + .7}
    changed_occ = {**data, 'noised_sequence': data['noised_sequence'] + .8}
    actor_changed = model.transition_model(changed_actor)
    occ_changed = model.transition_model(changed_occ)
    if mode == 'joint':
        assert not torch.allclose(original['predicted_latent'], actor_changed['predicted_latent'], atol=1e-7, rtol=1e-7)
        assert not torch.allclose(original['actor_velocity'], occ_changed['actor_velocity'], atol=1e-7, rtol=1e-7)
        both = original['predicted_latent'].square().mean() + original['actor_velocity'].square().mean()
        both.backward()
        assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in model.transition_model.interactions.parameters())
    else:
        assert torch.equal(original['predicted_latent'], actor_changed['predicted_latent'])
        assert torch.equal(original['actor_velocity'], occ_changed['actor_velocity'])


def test_real_token_path_matches_unchanged_trunk_when_no_actor():
    model = small_model().eval()
    obs, _, candidate = batch(1)
    obs['actor_observed_mask'].zero_()
    data = flow_inputs(obs, candidate)
    expected = model.transition_model.trunk({'noised_sequence': data['noised_sequence'], 'timesteps': data['timesteps'], 'trajectory': candidate})['predicted_latent']
    actual = model.transition_model(data)
    torch.testing.assert_close(actual['predicted_latent'], expected, rtol=0, atol=0)
    assert torch.isfinite(actual['actor_velocity']).all()
    assert not actual['actor_velocity'].any()


def test_padding_garbage_and_history_masks_do_not_change_valid_outputs():
    model = small_model().eval()
    obs, _, candidate = batch(2)
    original = model.sample(obs, candidate, generator=torch.Generator().manual_seed(2), nfe=1)
    dirty = copy.deepcopy(obs)
    dirty['history_latents'][:, :2] = float('nan')
    dirty['actor_history'][:, -1] = float('nan')
    dirty['actor_size'][:, -1] = float('nan')
    dirty['actor_class'][:, -1] = 98765
    dirty['actor_velocity_mask'][:, -1] = True
    again = model.sample(dirty, candidate, generator=torch.Generator().manual_seed(2), nfe=1)
    for key in original:
        torch.testing.assert_close(original[key], again[key], rtol=0, atol=0)


def test_missing_velocity_and_unobserved_history_are_not_conditions():
    model = small_model().eval()
    obs, _, candidate = batch(1)
    obs['actor_observed_mask'][:, 0, :2] = False
    obs['actor_velocity_mask'][:, 0] = False
    a = model.sample(obs, candidate, generator=torch.Generator().manual_seed(2), nfe=1)
    obs['actor_history'][:, 0, :2] = float('nan')
    obs['actor_history'][:, 0, :, 4:] = float('nan')
    b = model.sample(obs, candidate, generator=torch.Generator().manual_seed(2), nfe=1)
    for key in a:
        torch.testing.assert_close(a[key], b[key], rtol=0, atol=0)


def test_training_masks_are_future_only_and_modal_element_means():
    model = recording_model()
    obs, target, candidate = batch()
    losses = model.training_loss(obs, target, candidate, generator=torch.Generator().manual_seed(6))
    call = model.transition_model.calls[-1]
    t = call['timesteps'] / 1000
    occ_t = t[:, None, None, None, None]
    actor_t = t[:, None, None, None]
    clean_occ = torch.where(target['future_latent_mask'][..., None, None, None], target['future_latents'] * 10, 0)
    occ_noise = (call['noised_sequence'][:, 6:] - occ_t * clean_occ) / (1 - occ_t)
    occ_valid = target['future_latent_mask'][..., None, None, None].expand_as(clean_occ)
    torch.testing.assert_close(losses['occ_fm'], (clean_occ - occ_noise)[occ_valid].square().mean())
    selected = obs['actor_observed_mask'][:, :, -1]
    valid = target['actor_target_mask'] & selected[..., None]
    dims = valid[..., None].expand_as(target['actor_future']).clone()
    dims[..., 4:] &= target['actor_velocity_mask'][..., None]
    clean_actor = torch.where(dims, target['actor_future'], 0)
    actor_noise = (call['actor_noised'] - actor_t * clean_actor) / (1 - actor_t)
    torch.testing.assert_close(losses['actor_fm'], (clean_actor - actor_noise)[dims].square().mean())
    torch.testing.assert_close(losses['actor_existence'], torch.tensor(.69314718056))
    torch.testing.assert_close(losses['total'], losses['occ_fm'] + losses['actor_fm'] + .1 * losses['actor_existence'])
    assert set(losses) == {'total', 'occ_fm', 'actor_fm', 'actor_existence'}
    losses['total'].backward()
    assert model.transition_model.value.grad is not None
    assert torch.isfinite(model.transition_model.value.grad)


def test_target_tail_garbage_does_not_affect_losses_and_no_target_fields_enter_forward():
    model = recording_model()
    obs, target, candidate = batch()
    a = model.training_loss(obs, target, candidate, generator=torch.Generator().manual_seed(5))
    target['future_latents'][:, 3:] = float('nan')
    target['actor_future'][:, -1] = float('nan')
    target['actor_future'][:, :, -1] = float('nan')
    target['actor_future'][:, :, -2:, 4:] = float('nan')
    b = model.training_loss(obs, target, candidate, generator=torch.Generator().manual_seed(5))
    for key in a:
        torch.testing.assert_close(a[key], b[key], rtol=0, atol=0)
    call = model.transition_model.calls[-1]
    assert not (set(target) - {'actor_velocity_mask'}) & set(call)
    torch.testing.assert_close(call['actor_velocity_mask'], obs['actor_velocity_mask'] & obs['actor_observed_mask'])


def test_losses_all_empty_are_differentiable_zero_and_local_rng():
    model = recording_model()
    obs, target, candidate = batch()
    obs['actor_observed_mask'].zero_()
    for key in ('future_latent_mask', 'actor_target_mask', 'actor_velocity_mask', 'actor_existence_mask'):
        target[key].zero_()
    state = torch.random.get_rng_state()
    losses = model.training_loss(obs, target, candidate, generator=torch.Generator().manual_seed(9))
    assert all(value.item() == 0 and value.requires_grad for value in losses.values())
    losses['total'].backward()
    assert model.transition_model.value.grad == 0
    assert torch.equal(state, torch.random.get_rng_state())


def test_independent_noise_shared_time_and_no_history_loss():
    model = recording_model()
    obs, target, candidate = batch()
    model.training_loss(obs, target, candidate, generator=torch.Generator().manual_seed(99))
    call = model.transition_model.calls[-1]
    g = torch.Generator().manual_seed(99)
    t = torch.sigmoid(torch.randn(2, generator=g))
    occ_noise = torch.randn(2, 6, 2, 4, 4, generator=g)
    actor_noise = torch.randn(2, 3, 6, 6, generator=g)
    torch.testing.assert_close(call['timesteps'], t * 1000)
    expected = t[:, None, None, None, None] * target['future_latents'] * 10 + (1-t[:, None, None, None, None]) * occ_noise
    torch.testing.assert_close(call['noised_sequence'][:, 6:9], expected[:, :3])
    expected_actor = t[:, None, None, None] * target['actor_future'] + (1-t[:, None, None, None]) * actor_noise
    torch.testing.assert_close(call['actor_noised'][:, :2, :4], expected_actor[:, :2, :4])
    assert not torch.equal(occ_noise.flatten()[:actor_noise.numel()], actor_noise.flatten())
    assert not call['noised_sequence'][:, :2].any()


def test_endpoint_normalization_heading_fallback_and_autograd():
    stats = {'actor_xy': {'mean': [10., -2.], 'std': [2., 3.]},
             'actor_velocity': {'mean': [1., 4.], 'std': [5., 6.]}}
    model = recording_model(normalization=stats)
    obs, _, candidate = batch(1)
    noise = {'future_latents': torch.zeros(1, 1, 6, 2, 4, 4),
             'actor_future': torch.zeros(1, 1, 3, 6, 6)}
    out = model.sample(obs, candidate, generator=torch.Generator().manual_seed(1), initial_noise=noise, nfe=2)
    torch.testing.assert_close(out['actor_future'][0, 0, 0, 0], torch.tensor([10., -2., .6, .8, 1., 4.]))
    assert out['future_latents'].requires_grad
    assert out['actor_future'].requires_grad
    out['future_latents'].sum().backward()
    assert model.transition_model.value.grad is not None


def test_future_observation_keys_shapes_and_rng_are_validated():
    model = recording_model()
    obs, target, candidate = batch(1)
    for key in ('future_latents', 'actor_target_mask', 'target', 'future_time_s', 'actor_future_exists', 'target.future_known_mask'):
        with pytest.raises(ValueError, match='future|target|observation'):
            model.sample({**obs, key: target['actor_future']}, candidate, generator=torch.Generator())
    with pytest.raises(ValueError, match='generator'):
        model.sample(obs, candidate)
    with pytest.raises(ValueError, match='candidate|trajectory|shape'):
        model.sample(obs, candidate[:, :-1], generator=torch.Generator())
    with pytest.raises(ValueError, match='shape'):
        model.sample({**obs, 'history_mask': torch.ones(1, 3, dtype=torch.bool)}, candidate, generator=torch.Generator())
    with pytest.raises(ValueError, match='nfe'):
        model.sample(obs, candidate, generator=torch.Generator(), nfe=0)


def test_occupancy_only_uses_original_head_and_skips_missing_actor_targets():
    model = small_model('occupancy_only').eval()
    obs, target, candidate = batch(1)
    losses = model.training_loss(obs, {k: v for k, v in target.items() if k.startswith('future_latent')}, candidate, generator=torch.Generator().manual_seed(3))
    assert losses['actor_fm'] == 0 and losses['actor_existence'] == 0
    data = flow_inputs(obs, candidate)
    actual = model.transition_model(data)['predicted_latent']
    expected = model.transition_model.trunk({'noised_sequence': data['noised_sequence'], 'timesteps': data['timesteps'], 'trajectory': candidate})['predicted_latent']
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)




def test_real_trunk_backward_reaches_both_new_branches_with_frozen_trunk():
    model = small_model(freeze=True)
    obs, target, candidate = batch(1)
    losses = model.training_loss(obs, target, candidate, generator=torch.Generator().manual_seed(3))
    losses['total'].backward()
    named = dict(model.transition_model.named_parameters())
    assert all(p.grad is None for n, p in named.items() if n.startswith('trunk.'))
    for prefix in ('actor_encoder.', 'actor_velocity_head.', 'existence_head.', 'interactions.'):
        assert any(p.grad is not None and p.grad.abs().sum() > 0 for n, p in named.items() if n.startswith(prefix))
    assert all(torch.isfinite(value) for value in losses.values())


def test_sample_nfe_counts_vector_field_evaluations_and_fixed_conditions():
    model = recording_model()
    obs, _, candidate = batch(1)
    model.sample(obs, candidate, generator=torch.Generator().manual_seed(7), nfe=3)
    calls = model.transition_model.calls
    assert len(calls) == 3
    for call in calls:
        torch.testing.assert_close(call['noised_sequence'][:, 2:6], obs['history_latents'][:, 2:] * 10)
        torch.testing.assert_close(call['trajectory'], candidate)
    assert calls[0]['timesteps'].item() == 0
    assert calls[0]['timesteps'].item() < calls[1]['timesteps'].item() < calls[2]['timesteps'].item() < 1000


def test_mixed_all_padding_rows_have_finite_backward_and_ignore_slot_garbage():
    model = small_model(freeze=True)
    obs, target, candidate = batch(2)
    obs['actor_observed_mask'][0].zero_()
    obs['actor_history'][0] = float('nan')
    obs['actor_size'][0] = float('nan')
    obs['actor_class'][0] = 999
    losses = model.training_loss(obs, target, candidate, generator=torch.Generator().manual_seed(17))
    losses['total'].backward()
    assert all(torch.isfinite(value) for value in losses.values())
    assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)
    output = model.sample(obs, candidate, generator=torch.Generator().manual_seed(17), nfe=1)
    assert not output['actor_future'][0].any()
    assert not output['actor_existence'][0].any()
    assert all(torch.isfinite(value).all() for value in output.values())


def test_existence_absence_is_a_loss_label_and_never_suppresses_actor_sample():
    model = recording_model()
    obs, target, candidate = batch(1)
    loss1 = model.training_loss(obs, target, candidate, generator=torch.Generator().manual_seed(2))
    call1 = model.transition_model.calls[-1]
    target['actor_future_exists'] = ~target['actor_future_exists']
    loss2 = model.training_loss(obs, target, candidate, generator=torch.Generator().manual_seed(2))
    call2 = model.transition_model.calls[-1]
    for key in call1:
        torch.testing.assert_close(call1[key], call2[key])
    # Zero existence logits mean 0.5, independent of current selected states.
    generated = model.sample(obs, candidate, generator=torch.Generator().manual_seed(2), nfe=1)
    torch.testing.assert_close(generated['actor_existence'][0, 0, :2], torch.full((2, 6), .5))
    assert generated['actor_future'][0, 0, :2].abs().sum() > 0
    assert torch.isfinite(loss1['total']) and torch.isfinite(loss2['total'])


def test_invalid_target_shape_noise_normalization_and_mode_are_explicit_errors():
    from reactoccflow.models.react_flow import ReactFlow
    obs, target, candidate = batch(1)
    with pytest.raises(ValueError, match='mode'):
        ReactFlow(mode='unknown')
    with pytest.raises(ValueError, match='normalization|statistics'):
        recording_model(normalization={})
    model = recording_model()
    target['future_latent_mask'] = target['future_latent_mask'][:, :2]
    with pytest.raises(ValueError, match='shape'):
        model.training_loss(obs, target, candidate, generator=torch.Generator())
    with pytest.raises(ValueError, match='noise'):
        model.sample(obs, candidate, generator=torch.Generator(), initial_noise={})
    with pytest.raises(ValueError, match='num_worlds'):
        model.sample(obs, candidate, generator=torch.Generator(), num_worlds=0)


@pytest.mark.parametrize('all_empty', [False, True])
def test_occupancy_only_default_real_trunk_loss_supports_backward(all_empty):
    from reactoccflow.models.react_flow import ReactFlow
    trunk = small_model('occupancy_only', freeze=False).transition_model.trunk
    # Deliberately omit the freeze argument: this tests the public default.
    model = ReactFlow(mode='occupancy_only', trunk=trunk, latent_shape=(2, 4, 4), max_actors=3)
    obs, target, candidate = batch(1)
    if all_empty:
        target['future_latent_mask'].zero_()
    losses = model.training_loss(obs, target, candidate, generator=torch.Generator().manual_seed(29))
    assert losses['total'].requires_grad
    losses['total'].backward()
    grads = [p.grad for p in model.transition_model.trunk.parameters() if p.requires_grad]
    assert grads and any(grad is not None for grad in grads)
    assert all(torch.isfinite(grad).all() for grad in grads if grad is not None)
    if all_empty:
        assert losses['total'].item() == 0
        assert all(not grad.any() for grad in grads if grad is not None)
    else:
        assert any(grad is not None and grad.abs().sum() > 0 for grad in grads)


@pytest.mark.parametrize('mode', ['joint', 'parallel', 'occupancy_only'])
@pytest.mark.parametrize('origin', ['constructor_default', 'yaml'])
def test_each_mode_default_and_yaml_has_active_trainable_gradient_path(mode, origin):
    from reactoccflow.models.react_flow import ReactFlow
    import yaml
    trunk = small_model(mode, freeze=False).transition_model.trunk
    kwargs = {'mode': mode}
    if origin == 'yaml':
        kwargs = yaml.safe_load(Path(f'configs/react_flow_{mode}.yaml').read_text())['model']
        kwargs.pop('kind')
    if origin == 'yaml':
        kwargs['dit_config'] = {'patch_size': 2, 'hidden_size': 32, 'depth': 2,
                                'num_heads': 4, 'activation_checkpointing': False}
    else:
        kwargs['trunk'] = trunk
    kwargs.update(latent_shape=(2, 4, 4), max_actors=3)
    model = ReactFlow(**kwargs)
    obs, target, candidate = batch(1)
    loss = model.training_loss(obs, target, candidate, generator=torch.Generator().manual_seed(30))['total']
    assert loss.requires_grad
    loss.backward()
    active = [p for p in model.parameters() if p.requires_grad]
    assert active and any(p.grad is not None and p.grad.abs().sum() > 0 for p in active)
    assert all(torch.isfinite(p.grad).all() for p in active if p.grad is not None)
    trunk_trainable = any(p.requires_grad for p in model.transition_model.trunk.parameters())
    assert trunk_trainable
