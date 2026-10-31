import pytest
import torch
from torch import nn
from reactoccflow.models.flow import ConditionalFlowMatching, interpolate, shifted_time_grid
from reactoccflow.models.modules.base.attn_base import DiTAttention


class RecordingField(nn.Module):
    def __init__(self, value=2.0):
        super().__init__()
        self.value = value
        self.calls = []

    def forward(self, data):
        self.calls.append({k: v.clone() if torch.is_tensor(v) else v for k, v in data.items()})
        return {**data, 'predicted_latent': torch.full_like(data['noised_sequence'], self.value)}


def test_interpolation():
    noise, data = torch.zeros(2, 3, 1, 2, 2), torch.ones(2, 3, 1, 2, 2)
    assert torch.equal(interpolate(noise, data, torch.tensor([0., 1.])), torch.stack([noise[0], data[1]]))
    grid = shifted_time_grid(10, alpha=3)
    assert grid[0] == 0 and grid[-1] == 1 and (grid.diff() > 0).all()
    t = torch.linspace(0, 1, 11)
    assert torch.equal(grid, (1 - 3 * t / (1 + 2 * t)).flip(0))


@pytest.mark.parametrize('batch', [1, 2])
def test_constant_field_exact_history_separate_and_conditions(batch):
    field = RecordingField()
    model = ConditionalFlowMatching(field, latent_scale=10)
    history = torch.arange(batch * 2 * 4.).reshape(batch, 2, 1, 2, 2)
    noise = torch.full((batch, 3, 1, 2, 2), .3)
    trajectory = torch.zeros(batch, 12, 2)
    result = model.sample(history, trajectory, initial_noise=noise, nfe=4)
    assert torch.allclose(result, (noise + 2) / 10)
    assert len(field.calls) == 4
    for call in field.calls:
        assert torch.equal(call['noised_sequence'][batch:, :2], history * 10)
        assert not call['noised_sequence'][:batch, :2].any()
        assert call['trajectory'].shape == (2 * batch, 12, 2)
        assert call['timesteps'].shape == (2 * batch,)
    assert torch.equal(history, torch.arange(batch * 2 * 4.).reshape(batch, 2, 1, 2, 2))


def test_fixed_noise_and_generator_do_not_use_global_rng():
    model = ConditionalFlowMatching(RecordingField())
    history = torch.zeros(2, 6, 1, 2, 2)
    state = torch.random.get_rng_state()
    first = model.sample(history, future_shape=(2, 6, 1, 2, 2), generator=torch.Generator().manual_seed(4), nfe=2)
    second = model.sample(history, future_shape=(2, 6, 1, 2, 2), generator=torch.Generator().manual_seed(4), nfe=2)
    assert torch.equal(first, second)
    assert torch.equal(state, torch.random.get_rng_state())
    with pytest.raises(ValueError, match='generator|noise'):
        model.sample(history, future_shape=(2, 6, 1, 2, 2))


def test_dit_preserves_upstream_head_major_reshape():
    layer = DiTAttention(8, num_heads=2).eval()
    x = torch.arange(24.).reshape(1, 3, 8) / 24
    q, k, v = layer.qkv(x).reshape(1, 3, 3, 2, 4).permute(2, 0, 3, 1, 4).contiguous().unbind(0)
    expected = layer.proj(torch.nn.functional.scaled_dot_product_attention(q, k, v).reshape(1, 3, 8))
    torch.testing.assert_close(layer(x), expected)


def test_classifier_free_guidance_keeps_trajectory_in_both_branches():
    class ConditionalField(nn.Module):
        def forward(self, data):
            assert data['trajectory'] is not None
            values = data['noised_sequence'][:, :1].mean((1, 2, 3, 4), keepdim=True)
            return {**data, 'predicted_latent': torch.ones_like(data['noised_sequence']) * values}
    sampler = ConditionalFlowMatching(ConditionalField(), guidance_scale=2, latent_scale=10)
    history = torch.ones(2, 6, 1, 2, 2)
    noise = torch.zeros(2, 6, 1, 2, 2)
    assert torch.allclose(sampler.sample(history, torch.zeros(2, 12, 2), initial_noise=noise, nfe=2), torch.full_like(noise, 2))


def test_real_backbone_batch_two_and_trajectory_branch():
    from reactoccflow.models.modules.transition_models.FlowMatchingV2 import FLOW_MATCHING_DOWN_X4_DiT
    backbone = FLOW_MATCHING_DOWN_X4_DiT(
        in_channels=2, out_channels=2, model_channels=32, channel_multi=[2, 4],
        input_size=[4, 4], trajectory_length=12, init_kernel_size=3,
        init_3d_conv_channels=16, attn_dim=8, temporal_attn_head=2, spatial_attn_head=2).eval()
    x = torch.zeros(2, 12, 2, 4, 4)
    with torch.no_grad():
        absent = backbone.forward_single(x, torch.tensor([100., 200.]), None)
        conditioned = backbone.forward_single(x, torch.tensor([100., 200.]), torch.ones(2, 12, 2))
    assert conditioned.shape == x.shape
    assert torch.isfinite(conditioned).all()
    assert not torch.equal(conditioned, absent)


def test_training_loss_future_only_gradients_and_local_rng():
    class TrainableField(nn.Module):
        def __init__(self):
            super().__init__()
            self.slots = nn.Parameter(torch.zeros(12))
            self.calls = []
        def forward(self, data):
            self.calls.append(data)
            output = self.slots[None, :, None, None, None].expand_as(data['noised_sequence'])
            return {**data, 'predicted_latent': output}
    field = TrainableField()
    model = ConditionalFlowMatching(field, latent_scale=10)
    history = torch.ones(2, 6, 1, 2, 2)
    target = torch.ones_like(history)
    state = torch.random.get_rng_state()
    first = model.training_loss(history, target, torch.zeros(2, 12, 2), generator=torch.Generator().manual_seed(6))
    second = model.training_loss(history, target, torch.zeros(2, 12, 2), generator=torch.Generator().manual_seed(6))
    assert first.ndim == 0 and first.requires_grad
    assert torch.equal(first, second)
    assert torch.equal(state, torch.random.get_rng_state())
    first.backward()
    assert torch.count_nonzero(field.slots.grad[:6]) == 0
    assert torch.count_nonzero(field.slots.grad[6:]) == 6
    assert field.calls[0]['trajectory'].shape == (2, 12, 2)
    assert field.calls[0]['timesteps'].shape == (2,)
    assert ((field.calls[0]['timesteps'] > 0) & (field.calls[0]['timesteps'] < 1000)).all()
    with pytest.raises(ValueError, match='generator'):
        model.training_loss(history, target)
