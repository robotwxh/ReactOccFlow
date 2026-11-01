"""Training acceptance catches unsafe optimizer progress and unreplayed probes."""
import importlib

import pytest
import torch
from torch import nn



def runner():
    # Import inside tests so missing acceptance functionality is a test failure.
    try:
        return importlib.import_module('reactoccflow.training.diagnostics')
    except ModuleNotFoundError:
        pytest.fail('standalone training acceptance runner is not implemented')


class TinyVelocity(nn.Module):
    def __init__(self):
        super().__init__()
        self.velocity = nn.Parameter(torch.tensor(0.))

    def forward(self, data):
        return {'predicted_latent': self.velocity.expand_as(data['noised_sequence'])}

    def training_loss(self, history, target, trajectory, *, generator):
        noise = torch.randn(target.shape, generator=generator, device=target.device) * .1
        prediction = self({'noised_sequence': history})['predicted_latent']
        return (prediction - target - noise).square().mean()


def small_batches():
    return [(torch.zeros(2, 6, 1, 2, 2), torch.ones(2, 6, 1, 2, 2), torch.zeros(2, 6, 2))]


def test_real_optimizer_updates_and_probe_replays_without_gradients():
    api = runner()
    model = TinyVelocity()
    model.train()
    initial_probe = api.evaluate_probe(model, small_batches(), seed=90)
    assert model.training
    assert all(p.grad is None for p in model.parameters())
    assert initial_probe == api.evaluate_probe(model, small_batches(), seed=90)
    stats = api.run_training(model, small_batches(), steps=20, lr=.08, seed=91)
    final_probe = api.evaluate_probe(model, small_batches(), seed=90)
    assert stats['optimizer_steps'] == 20
    assert len(stats['losses']) == 20
    assert stats['finite_loss_steps'] == stats['finite_gradient_steps'] == stats['finite_parameter_steps'] == 20
    assert stats['parameter_update']['changed_elements'] == 1
    assert stats['parameter_update']['max_abs_delta'] > .1
    assert final_probe['mean_loss'] < initial_probe['mean_loss']


class InvalidLoss(nn.Module):
    def __init__(self, kind):
        super().__init__()
        self.weight = nn.Parameter(torch.tensor(1.))
        self.kind = kind

    def training_loss(self, history, target, trajectory, *, generator):
        if self.kind == 'loss':
            return self.weight * float('nan')
        if self.kind == 'gradient':
            return self.weight.sqrt() * 0 + (self.weight - 1).sqrt()
        return self.weight * 0 + 1


@pytest.mark.parametrize('kind,reason', [('loss', 'loss'), ('gradient', 'gradient'), ('no_update', 'update')])
def test_rejects_nonfinite_loss_gradient_or_missing_update(kind, reason):
    api = runner()
    with pytest.raises(RuntimeError, match=reason):
        api.run_training(InvalidLoss(kind), small_batches(), steps=2, lr=.01, weight_decay=0., seed=1)


def test_probe_disables_dropout_and_gradients_and_restores_mode_on_error():
    api = runner()

    class GuardedVelocity(TinyVelocity):
        def __init__(self):
            super().__init__()
            self.dropout = nn.Dropout(.75)

        def forward(self, data):
            assert not torch.is_grad_enabled(), 'probe must not construct an autograd graph'
            assert not self.training, 'probe must disable model dropout'
            return {'predicted_latent': self.dropout(self.velocity.expand_as(data['noised_sequence']))}

    model = GuardedVelocity()
    api.evaluate_probe(model, small_batches(), seed=1)
    assert model.training
    broken = InvalidLoss('loss').train()
    with pytest.raises(RuntimeError, match='probe loss'):
        api.evaluate_probe(broken, small_batches(), seed=1)
    assert broken.training
