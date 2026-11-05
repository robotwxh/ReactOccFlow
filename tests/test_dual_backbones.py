"""Restoration, real bidirectional coupling and architecture checkpoint boundaries."""
import copy

import pytest
import torch

from reactoccflow.models import ReactFlow, StandardDiT, FLOW_MATCHING_DOWN_X4_DiT
from test_react_flow import batch, flow_inputs


UNET = dict(model_channels=32, channel_multi=[1, 2], init_kernel_size=3,
            init_3d_conv_channels=32, attn_dim=8, temporal_attn_head=4,
            spatial_attn_head=4)
DIT = dict(patch_size=2, hidden_size=32, depth=2, num_heads=4,
           activation_checkpointing=False)


def make_flow(backbone='occfm_unet', mode='joint'):
    options = {'unet_config': UNET} if backbone == 'occfm_unet' else {'dit_config': DIT}
    return ReactFlow(mode=mode, backbone=backbone, latent_shape=(2, 4, 4),
                     max_actors=3, **options)


def test_restored_default_and_explicit_dit_keep_distinct_topologies():
    model = ReactFlow()
    trunk = model.transition_model.trunk
    assert model.backbone == 'occfm_unet'
    assert isinstance(trunk, FLOW_MATCHING_DOWN_X4_DiT)
    assert len(trunk.downs) == len(trunk.ups) == 2
    assert len(trunk.state_dict()) == 204
    assert all(p.requires_grad for k, p in trunk.named_parameters()
               if k not in {'pos_embed', 'temp_embed'})
    assert not trunk.pos_embed.requires_grad and not trunk.temp_embed.requires_grad
    dit = make_flow('dit')
    assert isinstance(dit.transition_model.trunk, StandardDiT)
    assert not hasattr(dit.transition_model.trunk, 'downs')


@pytest.mark.parametrize('mode', ['joint', 'parallel', 'occupancy_only'])
def test_restored_modes_finite_backward_and_reproducible_sampling(mode):
    model = make_flow(mode=mode)
    obs, target, plan = batch(2)
    loss = model.training_loss(obs, target, plan,
                              generator=torch.Generator().manual_seed(5))['total']
    loss.backward()
    assert torch.isfinite(loss)
    assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)
    model.eval()
    with torch.inference_mode():
        a = model.sample(obs, plan, nfe=2, generator=torch.Generator().manual_seed(7))
        b = model.sample(obs, plan, nfe=2, generator=torch.Generator().manual_seed(7))
    for key in a:
        assert torch.isfinite(a[key]).all()
        torch.testing.assert_close(a[key], b[key], rtol=0, atol=0)


def test_unet_bottleneck_coupling_and_no_actor_identity_path():
    model = make_flow().eval()
    obs, _, plan = batch(2)
    obs['actor_observed_mask'][0].zero_()
    data = flow_inputs(obs, plan)
    with torch.inference_mode():
        a = model.transition_model(data)
        actor = model.transition_model({**data, 'actor_noised': data['actor_noised'] + .7})
        occ = model.transition_model({**data, 'noised_sequence': data['noised_sequence'] + .7})
        standalone = model.transition_model.trunk(data)['predicted_latent']
    torch.testing.assert_close(a['predicted_latent'][0], standalone[0], rtol=0, atol=0)
    assert not a['actor_velocity'][0].any()
    assert not torch.allclose(a['predicted_latent'][1], actor['predicted_latent'][1])
    assert not torch.allclose(a['actor_velocity'][1], occ['actor_velocity'][1])


@pytest.mark.parametrize('backbone', ['occfm_unet', 'dit'])
def test_checkpoints_round_trip_strictly_and_reject_other_backbone(backbone):
    a, b = make_flow(backbone), make_flow(backbone)
    b.load_state_dict(a.state_dict(), strict=True)
    with pytest.raises(RuntimeError):
        make_flow('dit' if backbone == 'occfm_unet' else 'occfm_unet').load_state_dict(a.state_dict(), strict=True)


def test_occfm_partition_loading_does_not_touch_actor_parameters(tmp_path):
    model = make_flow()
    actor_before = copy.deepcopy(model.transition_model.actor_encoder.state_dict())
    expected = copy.deepcopy(model.transition_model.trunk.state_dict())
    path = tmp_path / 'upstream.pt'
    torch.save({'state_dict': {**{'transition_model.' + k: v for k, v in expected.items()},
                               'encoder.unrelated': torch.ones(1)}}, path)
    with torch.no_grad():
        model.transition_model.trunk.init_conv.weight.zero_()
    report = model.load_pretrained_trunk(path)
    assert report['trunk_key_count'] == len(expected)
    assert report['excluded_compressor_keys'] == ['encoder.unrelated']
    for key, value in expected.items():
        torch.testing.assert_close(model.transition_model.trunk.state_dict()[key], value, rtol=0, atol=0)
    for key, value in actor_before.items():
        torch.testing.assert_close(model.transition_model.actor_encoder.state_dict()[key], value, rtol=0, atol=0)


def test_wrong_architecture_options_rejected_before_training():
    with pytest.raises(ValueError, match='match the selected backbone'):
        ReactFlow(backbone='dit', unet_config=UNET)
    with pytest.raises(ValueError, match='match the selected backbone'):
        ReactFlow(backbone='occfm_unet', dit_config=DIT)
    with pytest.raises(ValueError, match='Unknown U-Net'):
        ReactFlow(unet_config={'input_size': 4})


def test_resume_and_initialization_cannot_be_combined_before_reading_data():
    from scripts.train_react_flow import main
    with pytest.raises(SystemExit):
        main(['--backbone', 'occfm_unet', '--resume', 'saved.pt',
              '--trunk-checkpoint', 'initialization.pt'])
