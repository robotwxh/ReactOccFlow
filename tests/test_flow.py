"""Analytic flow math and attention contracts used by standard DiT."""
import pytest
import torch
from reactoccflow.models.flow import interpolate, shifted_time_grid
from reactoccflow.models.modules.base.attn_base import DiTAttention


def test_interpolation_and_shifted_grid_match_analytic_path():
    noise, data = torch.zeros(2, 3, 1, 2, 2), torch.ones(2, 3, 1, 2, 2)
    assert torch.equal(interpolate(noise, data, torch.tensor([0., 1.])), torch.stack([noise[0], data[1]]))
    grid = shifted_time_grid(10, alpha=3)
    assert grid[0] == 0 and grid[-1] == 1 and (grid.diff() > 0).all()
    t = torch.linspace(0, 1, 11)
    assert torch.equal(grid, (1 - 3 * t / (1 + 2 * t)).flip(0))
    with pytest.raises(ValueError, match='shape'):
        interpolate(noise, data[:, :2], .5)


@pytest.mark.parametrize('batch_size', [1, 2])
def test_reactflow_constant_field_integrates_exactly_and_keeps_history_fixed(batch_size):
    from test_react_flow import batch, recording_model
    model = recording_model()
    model.transition_model.value.data.fill_(2.)
    obs, _, candidate = batch(batch_size)
    noise = {'future_latents': torch.full((batch_size, 1, 6, 2, 4, 4), .3),
             'actor_future': torch.zeros(batch_size, 1, 3, 6, 6)}
    with torch.no_grad():
        output = model.sample(obs, candidate, initial_noise=noise, nfe=4, generator=torch.Generator())
    torch.testing.assert_close(output['future_latents'], (noise['future_latents'] + 2) / 10)
    assert len(model.transition_model.calls) == 4
    history = torch.where(obs['history_mask'][..., None, None, None], obs['history_latents'], 0.) * 10
    for call in model.transition_model.calls:
        torch.testing.assert_close(call['noised_sequence'][:, :6], history, rtol=0, atol=0)
        torch.testing.assert_close(call['trajectory'], candidate, rtol=0, atol=0)


def test_standard_dit_math_attention_merges_heads_in_token_order():
    layer = DiTAttention(8, num_heads=2, attention_mode='math').eval()
    x = torch.arange(24.).reshape(1, 3, 8) / 24
    q, k, v = layer.qkv(x).reshape(1, 3, 3, 2, 4).permute(2, 0, 3, 1, 4).unbind(0)
    expected = layer.proj(torch.nn.functional.scaled_dot_product_attention(q, k, v).transpose(1, 2).reshape(1, 3, 8))
    torch.testing.assert_close(layer(x), expected)
