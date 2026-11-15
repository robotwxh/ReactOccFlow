"""Standard DiT architecture, conditioning, masking and joint-flow contracts."""
import copy

import pytest
import torch
from torch import nn

from reactoccflow.models.standard_dit import StandardDiT
from reactoccflow.models.react_flow import ReactFlow
from test_react_flow import batch, flow_inputs


def tiny_trunk(**kwargs):
    return StandardDiT(in_channels=2, input_size=4, frames=12,
                       patch_size=2, hidden_size=32, depth=2, num_heads=4,
                       activation_checkpointing=False, **kwargs)


def tiny_flow(mode='joint', **kwargs):
    return ReactFlow(mode=mode, latent_shape=(2, 4, 4), max_actors=3,
                     dit_config={'patch_size': 2, 'hidden_size': 32, 'depth': 2,
                                 'num_heads': 4, 'activation_checkpointing': False}, **kwargs)


def open_zero_gates(model):
    """A nonzero field emulates learned weights for dependency checks."""
    with torch.no_grad():
        trunk = model.transition_model.trunk
        trunk.final_layer.linear.weight.normal_(std=.02)
        trunk.final_layer.modulation[-1].weight.normal_(std=.02)
        for block in trunk.blocks:
            block.adaLN_modulation[-1].weight.normal_(std=.02)


def test_patchify_round_trip_preserves_frames_channels_and_pixel_order():
    trunk = tiny_trunk()
    x = torch.arange(2 * 12 * 2 * 4 * 4).reshape(2, 12, 2, 4, 4)
    patches = trunk.patchify(x)
    assert patches.shape == (2, 48, 8)
    torch.testing.assert_close(trunk.unpatchify(patches), x, rtol=0, atol=0)
    torch.testing.assert_close(patches[0, 0], x[0, 0, :, :2, :2].permute(1, 2, 0).reshape(-1))


def test_default_flow_is_isotropic_dit_and_trainable_from_scratch():
    flow = ReactFlow(backbone='dit')
    trunk = flow.transition_model.trunk
    assert isinstance(trunk, StandardDiT) and flow.backbone == 'dit'
    assert trunk.token_count == 1200 and len(trunk.blocks) == 8
    assert trunk.hidden_size == 512
    assert trunk.patch_embed.in_features == 400
    assert trunk.final_layer.linear.out_features == 400
    assert not any(isinstance(m, (nn.Conv2d, nn.Conv3d, nn.ConvTranspose2d, nn.ConvTranspose3d))
                   for m in flow.modules())
    assert not hasattr(trunk, 'downs') and not hasattr(trunk, 'ups')
    assert all(p.requires_grad for p in trunk.parameters())


def test_zero_initialization_and_constant_token_resolution():
    trunk = tiny_trunk().eval()
    shapes = []
    handles = [b.register_forward_hook(lambda m, args, output: shapes.append(output.shape)) for b in trunk.blocks]
    out = trunk({'noised_sequence': torch.randn(2, 12, 2, 4, 4),
                 'timesteps': torch.tensor([100., 500.]), 'trajectory': torch.randn(2, 6, 2)})
    for handle in handles:
        handle.remove()
    assert shapes == [torch.Size([2, 48, 32])] * 2
    assert out['predicted_latent'].shape == (2, 12, 2, 4, 4)
    assert not out['predicted_latent'].any()
    assert all(not b.adaLN_modulation[-1].weight.any() for b in trunk.blocks)


@pytest.mark.parametrize('mode', ['joint', 'parallel', 'occupancy_only'])
def test_dit_joint_modes_train_and_sample_reproducibly(mode):
    flow = tiny_flow(mode)
    obs, target, candidate = batch(2)
    opt = torch.optim.AdamW(flow.parameters(), lr=1e-3)
    for _ in range(3):
        opt.zero_grad(set_to_none=True)
        loss = flow.training_loss(obs, target, candidate, generator=torch.Generator().manual_seed(4))['total']
        assert torch.isfinite(loss)
        loss.backward()
        assert all(torch.isfinite(p.grad).all() for p in flow.parameters() if p.grad is not None)
        opt.step()
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in flow.transition_model.trunk.parameters())
    flow.eval()
    state = torch.random.get_rng_state()
    with torch.inference_mode():
        a = flow.sample(obs, candidate, generator=torch.Generator().manual_seed(3), nfe=2)
        b = flow.sample(obs, candidate, generator=torch.Generator().manual_seed(3), nfe=2)
    for name in a:
        torch.testing.assert_close(a[name], b[name], rtol=0, atol=0)
        assert torch.isfinite(a[name]).all()
    assert torch.equal(state, torch.random.get_rng_state())


@pytest.mark.parametrize('mode', ['joint', 'parallel'])
def test_dit_actor_exchange_and_action_history_conditioning(mode):
    flow = tiny_flow(mode).eval()
    open_zero_gates(flow)
    obs, _, candidate = batch(1)
    data = flow_inputs(obs, candidate)
    with torch.no_grad():
        a = flow.transition_model(data)
        actor = flow.transition_model({**data, 'actor_noised': data['actor_noised'] + .8})
        occ = flow.transition_model({**data, 'noised_sequence': data['noised_sequence'] + .7})
        plan = flow.transition_model({**data, 'trajectory': candidate + .5})
        history = data['noised_sequence'].clone()
        history[:, :6] += .3
        changed_history = flow.transition_model({**data, 'noised_sequence': history})
    if mode == 'joint':
        assert not torch.allclose(a['predicted_latent'], actor['predicted_latent'])
        assert not torch.allclose(a['actor_velocity'], occ['actor_velocity'])
    else:
        assert torch.equal(a['predicted_latent'], actor['predicted_latent'])
        assert torch.equal(a['actor_velocity'], occ['actor_velocity'])
    assert not torch.allclose(a['predicted_latent'], plan['predicted_latent'])
    assert not torch.allclose(a['predicted_latent'][:, 6:], changed_history['predicted_latent'][:, 6:])


def test_all_padding_rows_are_finite_and_match_standalone_dit():
    flow = tiny_flow().eval()
    open_zero_gates(flow)
    obs, target, candidate = batch(2)
    obs['actor_observed_mask'][0].zero_()
    obs['actor_history'][0] = float('nan')
    obs['actor_size'][0] = float('nan')
    obs['actor_class'][0] = 9999
    loss = flow.training_loss(obs, target, candidate, generator=torch.Generator().manual_seed(17))['total']
    loss.backward()
    assert torch.isfinite(loss)
    assert all(torch.isfinite(p.grad).all() for p in flow.parameters() if p.grad is not None)
    clean, plan = flow._observation(obs, candidate)
    data = flow_inputs(clean, plan)
    with torch.no_grad():
        joint = flow.transition_model(data)
        standalone = flow.transition_model.trunk(data)
    torch.testing.assert_close(joint['predicted_latent'][0], standalone['predicted_latent'][0], rtol=0, atol=0)
    assert not joint['actor_velocity'][0].any()


def test_checkpointed_blocks_match_eager_forward_and_gradients():
    a = tiny_flow().train()
    open_zero_gates(a)
    b = copy.deepcopy(a)
    b.transition_model.trunk.activation_checkpointing = True
    obs, target, candidate = batch(1)
    losses = []
    for model in (a, b):
        loss = model.training_loss(obs, target, candidate, generator=torch.Generator().manual_seed(8))['total']
        loss.backward()
        losses.append(loss)
    torch.testing.assert_close(*losses, rtol=0, atol=0)
    for pa, pb in zip(a.parameters(), b.parameters()):
        if pa.grad is not None:
            torch.testing.assert_close(pa.grad, pb.grad, rtol=0, atol=0)


def test_strict_dit_state_round_trip_and_legacy_initialization_rejection(tmp_path):
    a = tiny_flow()
    open_zero_gates(a)
    path = tmp_path / 'dit.pt'
    torch.save(a.state_dict(), path)
    b = tiny_flow()
    b.load_state_dict(torch.load(path, weights_only=True), strict=True)
    obs, _, candidate = batch(1)
    data = flow_inputs(obs, candidate)
    for name, value in a.transition_model(data).items():
        torch.testing.assert_close(value, b.transition_model(data)[name], rtol=0, atol=0)
    with pytest.raises(ValueError, match='requires backbone=occfm_unet'):
        a.load_pretrained_trunk(path)


@pytest.mark.parametrize('options', [
    {'patch_size': 3}, {'hidden_size': 30}, {'depth': 0}, {'num_heads': 3},
    {'depth': True}, {'mlp_ratio': float('nan')}, {'activation_checkpointing': 'false'},
])
def test_invalid_architecture_options_are_rejected(options):
    config = dict(in_channels=2, input_size=4, patch_size=2, hidden_size=32, depth=2, num_heads=4)
    with pytest.raises(ValueError, match='DiT'):
        StandardDiT(**{**config, **options})


def test_direct_velocity_backbone_rejects_forced_noise_nullspace_before_training():
    with pytest.raises(ValueError, match='Gaussian noise directions'):
        StandardDiT(in_channels=16, input_size=50, patch_size=5,
                    hidden_size=256, depth=8, num_heads=8)


def test_safe_width_has_capacity_to_observe_and_control_every_patch_direction():
    # Nonzero weights emulate a learned head; zero-init is retained for real training.
    trunk = StandardDiT(in_channels=4, input_size=4, frames=2,
                        trajectory_length=1, patch_size=2, hidden_size=32,
                        depth=1, num_heads=4, activation_checkpointing=False)
    generator = torch.Generator().manual_seed(61)
    with torch.no_grad():
        trunk.final_layer.linear.weight.copy_(torch.randn(
            trunk.final_layer.linear.weight.shape, generator=generator))
    assert torch.linalg.matrix_rank(trunk.patch_embed.weight.double()) == 16
    assert torch.linalg.matrix_rank(trunk.final_layer.linear.weight.double()) == 16
    # A full-row-rank velocity head admits a correction for arbitrary patch noise.
    noise = torch.randn(7, 16, generator=generator, dtype=torch.float64)
    weight = trunk.final_layer.linear.weight.double()
    features = -noise @ torch.linalg.pinv(weight).T
    torch.testing.assert_close(noise + features @ weight.T,
                               torch.zeros_like(noise), rtol=0, atol=1e-12)


def test_standard_dit_supports_explicit_nonlegacy_temporal_slots():
    flow = ReactFlow(history_frames=3, future_frames=2, latent_shape=(2, 4, 4), max_actors=3,
                     dit_config={'patch_size': 2, 'hidden_size': 32, 'depth': 2, 'num_heads': 4})
    assert flow.transition_model.trunk.frames == 5


def test_runner_rejects_cross_architecture_arguments_before_any_data_access():
    from scripts.train_react_flow import main
    with pytest.raises(SystemExit):
        main(['--backbone', 'dit', '--trunk-checkpoint', 'legacy.pt'])
    with pytest.raises(SystemExit):
        main(['--backbone', 'occfm_unet', '--dit-config', 'missing.yaml'])


def test_both_backbones_are_available_without_reviving_stale_baseline_api():
    import ast
    import importlib.util
    from pathlib import Path
    import reactoccflow.models as models

    assert not hasattr(models, 'ConditionalFlowMatching')
    assert importlib.util.find_spec('reactoccflow.evaluation.baseline') is None
    assert Path('reactoccflow/models/modules/transition_models/FlowMatchingV2.py').exists()
    assert not Path('configs/occfm_history.yaml').exists()
    assert not Path('configs/occfm_future.yaml').exists()
    for folder in ('reactoccflow', 'scripts'):
        for path in Path(folder).rglob('*.py'):
            for node in ast.walk(ast.parse(path.read_text())):
                if isinstance(node, ast.Call) and getattr(node.func, 'id', None) == 'ReactFlow':
                    for keyword in node.keywords:
                        if keyword.arg == 'backbone' and isinstance(keyword.value, ast.Constant):
                            assert keyword.value.value in {'dit', 'occfm_unet'}, str(path)
