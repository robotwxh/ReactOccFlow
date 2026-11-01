"""Joint acceptance uses continuous endpoint losses and preserves frozen weights."""
import importlib

import pytest
import torch
from torch import nn

from reactoccflow.geometry import GridSpec
from reactoccflow.losses import ConsistencyLosses
from reactoccflow.models.react_flow import ReactFlow


def adapter_class():
    try:
        return importlib.import_module('scripts.train_joint_smoke').JointTrainingAdapter
    except ModuleNotFoundError:
        pytest.fail('joint endpoint-training adapter is not implemented')


class TinyJointField(nn.Module):
    def __init__(self):
        super().__init__()
        self.occ = nn.Parameter(torch.tensor(.2))
        self.actor = nn.Parameter(torch.tensor(.1))
        self.existence = nn.Parameter(torch.tensor(.1))

    def forward(self, data):
        return {'predicted_latent': self.occ.expand_as(data['noised_sequence']),
                'actor_velocity': self.actor.expand_as(data['actor_noised']),
                'actor_existence_logits': self.existence.expand_as(data['actor_noised'][..., 0])}


class TinyContinuousCodec(nn.Module):
    def __init__(self):
        super().__init__()
        self.gain = nn.Parameter(torch.tensor(2.))

    def decode(self, latents, *, return_logits=False):
        assert return_logits, 'joint losses must receive continuous semantic logits'
        assert not self.training, 'codec must remain in eval mode during training'
        features = latents[:, :1, :, :].permute(0, 2, 3, 1)[..., None]
        classes = torch.arange(18, device=latents.device, dtype=latents.dtype)
        return features * self.gain * classes


def fixture():
    flow = ReactFlow(transition_model=TinyJointField(), latent_shape=(1, 4, 4),
                     max_actors=1, latent_scale=1.)
    codec = TinyContinuousCodec()
    grid = GridSpec(shape=(4, 4, 1), origin=(-2., -2., -.5), voxel_size=(1., 1., 1.))
    obs = {'history_latents': torch.zeros(1, 6, 1, 4, 4), 'history_mask': torch.ones(1, 6, dtype=torch.bool),
           'actor_history': torch.zeros(1, 1, 4, 6), 'actor_observed_mask': torch.ones(1, 1, 4, dtype=torch.bool),
           'actor_velocity_mask': torch.ones(1, 1, 4, dtype=torch.bool), 'actor_size': torch.ones(1, 1, 3),
           'actor_class': torch.full((1, 1), 4), 'actor_z': torch.zeros(1, 1)}
    obs['actor_history'][..., 3] = 1
    target = {'future_latents': torch.ones(1, 6, 1, 4, 4), 'future_latent_mask': torch.ones(1, 6, dtype=torch.bool),
              'actor_future': torch.zeros(1, 1, 6, 6), 'actor_target_mask': torch.ones(1, 1, 6, dtype=torch.bool),
              'actor_velocity_mask': torch.ones(1, 1, 6, dtype=torch.bool),
              'actor_future_exists': torch.ones(1, 1, 6, dtype=torch.bool),
              'actor_existence_mask': torch.ones(1, 1, 6, dtype=torch.bool),
              'actor_future_z': torch.zeros(1, 1, 6), 'future_time_s': torch.arange(1., 7.).unsqueeze(0),
              'future_semantics': torch.full((1, 6, 4, 4, 1), 11, dtype=torch.uint8),
              'future_known_mask': torch.ones(1, 6, 4, 4, 1, dtype=torch.bool)}
    target['actor_future'][..., 3] = 1
    return flow, codec, ConsistencyLosses(grid=grid), obs, target, torch.zeros(1, 6, 2)


def test_adapter_combines_endpoint_losses_and_backpropagates_without_codec_gradients():
    flow, codec, consistency, obs, target, candidate = fixture()
    adapter = adapter_class()(flow, codec, consistency=consistency)
    adapter.train()
    total = adapter.training_loss(obs, target, candidate, generator=torch.Generator().manual_seed(7))
    assert total.ndim == 0 and total.requires_grad and torch.isfinite(total)
    assert adapter.last_components['consistency.total'] > 0
    assert adapter.last_components['total'] > adapter.last_components['base.total']
    total.backward()
    assert all(p.grad is not None and torch.isfinite(p.grad) for p in flow.parameters())
    assert all(not p.requires_grad and p.grad is None for p in codec.parameters())
    assert not codec.training


def test_zero_consistency_weights_recover_base_loss_for_identical_local_draws():
    flow, codec, _, obs, target, candidate = fixture()
    from reactoccflow.losses import ConsistencyLosses
    zero = ConsistencyLosses(grid=GridSpec(shape=(4, 4, 1), origin=(-2., -2., -.5), voxel_size=(1., 1., 1.)),
                             occ_actor_weight=0., temporal_weight=0., kinematics_weight=0.)
    expected = flow.training_loss(obs, target, candidate, generator=torch.Generator().manual_seed(8))['total']
    adapter = adapter_class()(flow, codec, consistency=zero)
    actual = adapter.training_loss(obs, target, candidate, generator=torch.Generator().manual_seed(8))
    torch.testing.assert_close(actual, expected, atol=0, rtol=0)


def test_adapter_forward_is_ddp_callable_and_supports_trainable_trunk_mode():
    flow, codec, consistency, obs, target, candidate = fixture()
    adapter = adapter_class()(flow, codec, consistency=consistency,
                              require_frozen_trunk=False, record_components=False)
    total = adapter(obs, target, candidate, generator=torch.Generator().manual_seed(9))
    assert total.ndim == 0 and torch.isfinite(total)
    assert adapter.last_components == {}


def test_adapter_train_mode_keeps_frozen_codec_in_eval():
    flow, codec, consistency, _, _, _ = fixture()
    adapter = adapter_class()(flow, codec, consistency=consistency)

    adapter.train()

    assert adapter.training
    assert adapter.flow.training
    assert not adapter.codec.training


def test_joint_adapter_is_available_from_installed_training_package():
    from reactoccflow.training.joint import JointTrainingAdapter

    assert JointTrainingAdapter is adapter_class()


def test_cli_refuses_partial_cache_before_output_or_model_setup(tmp_path, monkeypatch):
    import json
    import sys
    api = importlib.import_module('scripts.train_joint_smoke')
    root = tmp_path / 'cache'
    root.mkdir()
    (root / 'manifest.json').write_text(json.dumps({'format': 'reactoccflow_e0_research_cache',
                                                  'version': 1, 'coordinate_frame': 'E0', 'complete': False}))
    output = tmp_path / 'never-created'
    monkeypatch.setattr(sys, 'argv', ['train_joint_smoke', '--cache', str(root), '--output', str(output)])
    with pytest.raises(ValueError, match='incomplete'):
        api.main()
    assert not output.exists()


def test_adapter_rejects_a_trunk_gradient_leak():
    flow, codec, consistency, _, _, _ = fixture()
    trunk = nn.Linear(1, 1)
    trunk(torch.ones(1, 1)).sum().backward()
    trunk.requires_grad_(False)
    flow.transition_model.trunk = trunk
    adapter = adapter_class()(flow, codec, consistency=consistency, require_frozen_trunk=True)
    with pytest.raises(RuntimeError, match='trunk.*gradients None'):
        adapter.frozen_status()
