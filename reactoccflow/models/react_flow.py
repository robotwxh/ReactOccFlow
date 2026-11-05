"""First research occupancy–actor conditional flow-matching model (T6).

Inputs use the E0 cache's observation/target dictionaries. Observations contain
only current/past measurements; the separately supplied ego candidate is a known
planned control. Sampling keeps gradients for later endpoint losses: callers
performing inference should use ``torch.inference_mode()`` to bound memory.

The straight probability path, sigmoid generative time, occupancy scale and
shifted Euler grid reuse the migrated OccFM math. The joint research branch is
new and is NOT a trained joint model. Provenance: react_flow_sources.json.
"""
from __future__ import annotations

from collections.abc import Mapping
import math
import torch
from torch import nn
from torch.nn import functional as F

from .flow import interpolate, shifted_time_grid
from .joint_transition import JointTransition
from .standard_dit import StandardDiT
from ._utils import read_state_dict, strict_load
from .modules.transition_models.FlowMatchingV2 import FLOW_MATCHING_DOWN_X4_DiT


_OBSERVATION_KEYS = frozenset({
    'history_latents', 'history_mask', 'history_time_s', 'observed_history_time_s',
    'actor_history', 'actor_observed_mask', 'actor_velocity_mask', 'actor_size',
    'actor_class', 'actor_z', 'actor_visibility', 'ego_history',
    'ego_observed_mask', 'ego_velocity_mask', 'ego_current', 'ego_state', 'ego_state_mask',
})


def _masked_mean(values, mask, *, statistics=None, name=None, weight=1.):
    """Mean over valid elements only; empty sets retain a zero gradient path."""
    mask = mask.expand_as(values)
    numerator = torch.where(mask, values, torch.zeros_like(values)).sum()
    count = mask.sum()
    if statistics is not None:
        statistics[name] = {'numerator': numerator.detach(), 'count': count.detach(),
                            'weight': float(weight)}
    return numerator / count.clamp_min(1)


class ReactFlow(nn.Module):
    """Shared-time, independent-noise occupancy and actor flow.

    Defaults: B×6×16×50×50 history/future occupancy, B×8×4×6 actor history,
    B×8×6×6 actor futures, six candidate XY controls. Only the last observed
    actor mask selects slots. ``normalization`` is the train-only statistics
    dictionary with actor_xy/actor_velocity mean/std pairs; None is identity.

    The default backbone is the restored OccFM-style U-shaped DiT. Both trunks
    are initialized from scratch unless weights are explicitly supplied,
    and trainable in all modes. ``transition_model`` supplies a custom field; ``trunk``
    supplies an explicit backbone. No codec/VAE is registered in this module.
    """
    def __init__(self, mode='joint', *, transition_model=None, trunk=None,
                 normalization=None, latent_shape=(16, 50, 50), history_frames=6,
                 future_frames=6, max_actors=8, actor_history_frames=4,
                 actor_hidden=128, actor_heads=4, interaction_layers=2,
                 latent_scale=10., time_scalar=1000., alpha=3.,
                 occ_weight=1., actor_weight=1., existence_weight=.1,
                 freeze_backbone=False, backbone=None, dit_config=None, unet_config=None):
        super().__init__()
        if mode not in {'joint', 'parallel', 'occupancy_only'}:
            raise ValueError('mode must be joint, parallel, or occupancy_only')
        if transition_model is not None and trunk is not None:
            raise ValueError('Provide either transition_model or trunk')
        if backbone not in {None, 'occfm_unet', 'dit'}:
            raise ValueError('backbone must be occfm_unet or dit')
        if len(latent_shape) != 3 or any(int(x) != x or x <= 0 for x in latent_shape):
            raise ValueError('latent_shape must contain positive C,H,W')
        if min(history_frames, future_frames, max_actors, actor_history_frames) < 1:
            raise ValueError('Frame and actor counts must be positive')
        if min(latent_scale, time_scalar, alpha) <= 0:
            raise ValueError('Flow scales and alpha must be positive')
        self.mode = mode
        self.latent_shape = tuple(latent_shape)
        self.history_frames = history_frames
        self.future_frames = future_frames
        self.max_actors = max_actors
        self.actor_history_frames = actor_history_frames
        self.latent_scale, self.time_scalar, self.alpha = latent_scale, time_scalar, alpha
        self.occ_weight, self.actor_weight, self.existence_weight = occ_weight, actor_weight, existence_weight
        self.transition_model = transition_model if transition_model is not None else JointTransition(
            trunk, mode=mode, future_frames=future_frames, history_frames=history_frames,
            actor_history_frames=actor_history_frames, actor_hidden=actor_hidden,
            actor_heads=actor_heads, interaction_layers=interaction_layers,
            backbone=backbone, dit_config=dit_config, unet_config=unet_config,
            latent_shape=self.latent_shape)
        self.backbone = getattr(self.transition_model, 'backbone', 'custom')
        means, scales = [], []
        for name in ('actor_xy', 'actor_velocity'):
            stats = None if normalization is None else normalization.get(name)
            if normalization is not None and stats is None:
                raise ValueError(f'normalization requires {name} train statistics')
            mean = [0., 0.] if stats is None else stats['mean']
            scale = [1., 1.] if stats is None else stats['std']
            if len(mean) != 2 or len(scale) != 2 or not all(math.isfinite(float(v)) for v in [*mean, *scale]) or min(scale) <= 0:
                raise ValueError(f'{name} normalization requires two finite means and positive stds')
            means.append(mean)
            scales.append(scale)
        self.register_buffer('actor_mean', torch.tensor([*means[0], 0., 0., *means[1]], dtype=torch.float32))
        self.register_buffer('actor_std', torch.tensor([*scales[0], 1., 1., *scales[1]], dtype=torch.float32))
        if hasattr(self.transition_model, 'trunk'):
            self.freeze_trunk(frozen=freeze_backbone)

    def freeze_trunk(self, frozen=True):
        """Freeze the selected trunk only; actor heads/exchange stay trainable.

        In occupancy_only, frozen=True is for inference: all active loss paths
        are then frozen. Pass frozen=False to train that mode.
        """
        if not hasattr(self.transition_model, 'trunk'):
            raise ValueError('The supplied transition_model has no backbone trunk')
        self.transition_model.trunk.requires_grad_(not frozen)
        # Source fixed positional embeddings are never trainable.
        self.transition_model.trunk.pos_embed.requires_grad_(False)
        self.transition_model.trunk.temp_embed.requires_grad_(False)
        return self

    def load_pretrained_trunk(self, path, *, weights='state_dict'):
        """Restore strict OccFM trunk loading, without loading actor/VAE weights.

        Partition selection reuses the archived owned CFM loader. This is
        initialization only, never an optimizer/cursor resume operation.
        """
        trunk = getattr(self.transition_model, 'trunk', None)
        if not isinstance(trunk, FLOW_MATCHING_DOWN_X4_DiT):
            raise ValueError('OccFM pretrained trunk loading requires backbone=occfm_unet')
        state = read_state_dict(path, weights=weights)
        prefix = 'transition_model.'
        if any(key.startswith(prefix) for key in state):
            selected = {key[len(prefix):]: value for key, value in state.items() if key.startswith(prefix)}
            unrelated = [key for key in state if not key.startswith(prefix)]
            unexpected = [key for key in unrelated if not key.startswith(('embedding.', 'encoder.', 'quantization.', 'decoder.'))]
            if unexpected:
                raise RuntimeError(f'Unexpected checkpoint partitions: {unexpected}')
        else:
            selected, unrelated = state, []
        report = strict_load(trunk, selected)
        report.update(excluded_compressor_keys=unrelated, trunk_key_count=len(trunk.state_dict()),
                      new_branch_keys=[key for key in self.transition_model.state_dict() if not key.startswith('trunk.')])
        return report


    def _normalise_actor(self, state):
        return (state - self.actor_mean.to(state)) / self.actor_std.to(state)

    def _denormalise_actor(self, state):
        return state * self.actor_std.to(state) + self.actor_mean.to(state)

    def _observation(self, observation, candidate):
        if not isinstance(observation, Mapping):
            raise ValueError('observation must be a current/past tensor dictionary')
        forbidden = set(observation) - _OBSERVATION_KEYS
        if forbidden:
            raise ValueError(f'Unexpected observation keys (future/target inputs forbidden): {sorted(forbidden)}')
        if 'history_latents' not in observation:
            raise ValueError('observation requires history_latents')
        history = observation['history_latents']
        if not torch.is_tensor(history) or not history.is_floating_point() or history.ndim != 5:
            raise ValueError('history_latents shape must be floating B,T,C,H,W')
        batch = history.shape[0]
        expected = {
            'history_latents': (batch, self.history_frames, *self.latent_shape),
            'history_mask': (batch, self.history_frames),
            'actor_history': (batch, self.max_actors, self.actor_history_frames, 6),
            'actor_observed_mask': (batch, self.max_actors, self.actor_history_frames),
            'actor_velocity_mask': (batch, self.max_actors, self.actor_history_frames),
            'actor_size': (batch, self.max_actors, 3),
            'actor_class': (batch, self.max_actors),
        }
        for key, shape in expected.items():
            if key not in observation or not torch.is_tensor(observation[key]) or tuple(observation[key].shape) != shape:
                raise ValueError(f'observation {key} shape must be {shape}')
            if observation[key].device != history.device:
                raise ValueError(f'observation {key} must share history_latents device')
        if not torch.is_tensor(candidate) or tuple(candidate.shape) != (batch, self.future_frames, 2):
            raise ValueError(f'ego_candidate_xy shape must be B,{self.future_frames},2')
        candidate = candidate.to(history)
        observed = observation['actor_observed_mask'].bool()
        selected = observed[..., -1]
        observed = observed & selected[..., None]
        velocity = observation['actor_velocity_mask'].bool() & observed
        hist = torch.where(observation['history_mask'].bool()[..., None, None, None], history, torch.zeros_like(history))
        actor = torch.where(observed[..., None], observation['actor_history'].to(history), torch.zeros_like(observation['actor_history'].to(history)))
        actor = torch.cat((actor[..., :4], torch.where(velocity[..., None], actor[..., 4:], torch.zeros_like(actor[..., 4:]))), -1)
        actor = self._normalise_actor(actor)
        actor = torch.where(observed[..., None], actor, torch.zeros_like(actor))
        actor = torch.cat((actor[..., :4], torch.where(velocity[..., None], actor[..., 4:], torch.zeros_like(actor[..., 4:]))), -1)
        size = torch.where(selected[..., None], observation['actor_size'].to(history), torch.zeros_like(observation['actor_size'].to(history)))
        classes = torch.where(selected, observation['actor_class'].long(), torch.full_like(observation['actor_class'].long(), -1))
        if ((classes[selected] < 0) | (classes[selected] > 17)).any():
            raise ValueError('Selected actor_class must be in [0, 17]')
        if not all(torch.isfinite(x).all() for x in (hist, actor, size, candidate)):
            raise ValueError('Valid observed features and ego candidate must be finite')
        return {'history_latents': hist, 'actor_history': actor,
                'actor_observed_mask': observed, 'actor_velocity_mask': velocity,
                'actor_size': size, 'actor_class': classes}, candidate

    def _target(self, target, name, shape, reference, *, boolean=False):
        value = target.get(name)
        if not torch.is_tensor(value) or tuple(value.shape) != tuple(shape):
            raise ValueError(f'target {name} shape must be {tuple(shape)}')
        return value.to(device=reference.device, dtype=torch.bool if boolean else reference.dtype)

    def _predict(self, observation, candidate, occ_state, actor_state, time):
        batch = occ_state.shape[0]
        data = {key: value for key, value in observation.items() if key.startswith('actor_')}
        data.update({'noised_sequence': torch.cat((observation['history_latents'] * self.latent_scale, occ_state), 1),
                     'timesteps': time.reshape(batch) * self.time_scalar,
                     'trajectory': candidate, 'actor_noised': actor_state})
        result = self.transition_model(data)
        expected = {'predicted_latent': data['noised_sequence'].shape,
                    'actor_velocity': actor_state.shape,
                    'actor_existence_logits': actor_state.shape[:-1]}
        for key, shape in expected.items():
            if key not in result or result[key].shape != shape:
                raise ValueError(f'Transition {key} shape must be {tuple(shape)}')
        return result

    def training_loss(self, observation, target, ego_candidate_xy, *, generator=None,
                      return_endpoints=False, return_statistics=False):
        """Compute the original training objective, optionally exposing detached sums.

        ``statistics`` contains each component's masked numerator, raw element
        count (zero for empty masks), and objective weight. Sum each numerator
        and count across validation batches before dividing; kinematics are
        supplied separately by the endpoint consistency module.
        """
        if generator is None:
            raise ValueError('training_loss requires an explicit local generator')
        observation, candidate = self._observation(observation, ego_candidate_xy)
        history = observation['history_latents']
        batch = history.shape[0]
        occ_shape = (batch, self.future_frames, *self.latent_shape)
        actor_shape = (batch, self.max_actors, self.future_frames, 6)
        occ_mask = self._target(target, 'future_latent_mask', occ_shape[:2], history, boolean=True)
        clean_occ = self._target(target, 'future_latents', occ_shape, history) * self.latent_scale
        clean_occ = torch.where(occ_mask[..., None, None, None], clean_occ, torch.zeros_like(clean_occ))
        selected = observation['actor_observed_mask'][..., -1]
        if self.mode != 'occupancy_only':
            valid = self._target(target, 'actor_target_mask', actor_shape[:-1], history, boolean=True) & selected[..., None]
            vel_valid = self._target(target, 'actor_velocity_mask', actor_shape[:-1], history, boolean=True) & valid
            actor_mask = torch.cat((valid[..., None].expand(*valid.shape, 4), vel_valid[..., None].expand(*valid.shape, 2)), -1)
            clean_actor = self._normalise_actor(self._target(target, 'actor_future', actor_shape, history))
            clean_actor = torch.where(actor_mask, clean_actor, torch.zeros_like(clean_actor))
            existence_mask = self._target(target, 'actor_existence_mask', actor_shape[:-1], history, boolean=True) & selected[..., None]
            existence = self._target(target, 'actor_future_exists', actor_shape[:-1], history)
            existence = torch.where(existence_mask, existence, torch.zeros_like(existence))
        else:
            clean_actor = history.new_zeros(actor_shape)
        # One sigmoid-normal generative time, independent modal Gaussian states.
        time = torch.sigmoid(torch.randn(batch, device=history.device, dtype=history.dtype, generator=generator))
        occ_noise = torch.randn(occ_shape, device=history.device, dtype=history.dtype, generator=generator)
        actor_noise = torch.randn(actor_shape, device=history.device, dtype=history.dtype, generator=generator)
        occ_state = interpolate(occ_noise, clean_occ, time)
        actor_state = interpolate(actor_noise, clean_actor, time)
        result = self._predict(observation, candidate, occ_state, actor_state, time)
        occ_velocity = result['predicted_latent'][:, self.history_frames:]
        statistics = {} if return_statistics else None
        occ_loss = _masked_mean((occ_velocity - (clean_occ - occ_noise)).square(),
                                occ_mask[..., None, None, None], statistics=statistics,
                                name='occ_fm', weight=self.occ_weight)
        if self.mode != 'occupancy_only':
            actor_loss = _masked_mean((result['actor_velocity'] - (clean_actor - actor_noise)).square(),
                                      actor_mask, statistics=statistics,
                                      name='actor_fm', weight=self.actor_weight)
            existence_loss = _masked_mean(F.binary_cross_entropy_with_logits(result['actor_existence_logits'], existence, reduction='none'),
                                          existence_mask, statistics=statistics,
                                          name='actor_existence', weight=self.existence_weight)
        else:
            actor_loss = result['actor_velocity'].sum() * 0
            existence_loss = result['actor_existence_logits'].sum() * 0
        total = self.occ_weight * occ_loss + self.actor_weight * actor_loss + self.existence_weight * existence_loss
        losses = {'total': total, 'occ_fm': occ_loss, 'actor_fm': actor_loss, 'actor_existence': existence_loss}
        if return_statistics:
            if self.mode == 'occupancy_only':
                for name, weight in (('actor_fm', self.actor_weight),
                                     ('actor_existence', self.existence_weight)):
                    statistics[name] = {'numerator': history.new_zeros(()),
                                        'count': history.new_zeros((), dtype=torch.long),
                                        'weight': float(weight)}
            losses['statistics'] = statistics
        if return_endpoints:
            # Same field evaluation and RNG draws as the original loss. Unknown
            # cached latents are an empty-label approximation; voxel consistency
            # must separately gate on future_known_mask in the loss module.
            losses['endpoints'] = {
                'future_latents': (occ_state + (1 - time[:, None, None, None, None]) * occ_velocity) / self.latent_scale,
                'actor_future': self._denormalise_actor(actor_state + (1 - time[:, None, None, None]) * result['actor_velocity']),
                'actor_existence_logits': result['actor_existence_logits'],
                'generative_time': time,
            }
        return losses

    def sample(self, observation, ego_candidate_xy, *, num_worlds=1, generator=None,
               nfe=8, initial_noise=None):
        """Generate one candidate's worlds, preserving gradients by construction.

        Optional noise is a dictionary with ``future_latents`` B,M,F,C,H,W and
        ``actor_future`` B,M,A,F,6 in normalized flow coordinates. Sampling has
        no target argument. Heading is normalized ONLY at the endpoint, with
        current observed heading as the degenerate-vector fallback. Absent
        future existence never gates current-selected actor states. Existence uses
        the last ODE evaluation's separate head, preserving the exact NFE budget.
        """
        if generator is None:
            raise ValueError('sample requires an explicit local generator')
        if not isinstance(num_worlds, int) or isinstance(num_worlds, bool) or num_worlds < 1:
            raise ValueError('num_worlds must be a positive integer')
        if not isinstance(nfe, int) or isinstance(nfe, bool) or nfe < 1:
            raise ValueError('nfe must be a positive integer')
        observation, candidate = self._observation(observation, ego_candidate_xy)
        history = observation['history_latents']
        batch = history.shape[0]
        occ_shape = (batch, num_worlds, self.future_frames, *self.latent_shape)
        actor_shape = (batch, num_worlds, self.max_actors, self.future_frames, 6)
        if initial_noise is None:
            occ = torch.randn(occ_shape, device=history.device, dtype=history.dtype, generator=generator)
            actor = torch.randn(actor_shape, device=history.device, dtype=history.dtype, generator=generator)
        else:
            if not isinstance(initial_noise, Mapping) or set(initial_noise) != {'future_latents', 'actor_future'}:
                raise ValueError('initial_noise requires future_latents and actor_future tensors')
            occ, actor = initial_noise['future_latents'], initial_noise['actor_future']
            if not torch.is_tensor(occ) or not torch.is_tensor(actor) or tuple(occ.shape) != occ_shape or tuple(actor.shape) != actor_shape:
                raise ValueError(f'initial_noise shapes must be {occ_shape} and {actor_shape}')
            occ, actor = occ.to(history).clone(), actor.to(history).clone()
            if not torch.isfinite(occ).all() or not torch.isfinite(actor).all():
                raise ValueError('initial_noise must be finite')
        occ = occ.flatten(0, 1)
        actor = actor.flatten(0, 1)
        observation = {key: value.repeat_interleave(num_worlds, dim=0) for key, value in observation.items()}
        candidate = candidate.repeat_interleave(num_worlds, dim=0)
        grid = shifted_time_grid(nfe, self.alpha, device=history.device)
        for current, following in zip(grid[:-1], grid[1:]):
            result = self._predict(observation, candidate, occ, actor, current.expand(batch * num_worlds))
            occ = occ + (following - current) * result['predicted_latent'][:, self.history_frames:]
            actor = actor + (following - current) * result['actor_velocity']
        # Reuse the final vector-field call's existence logits so nfe is the
        # exact neural evaluation count (no hidden endpoint trunk evaluation).
        actor = self._denormalise_actor(actor)
        heading = actor[..., 2:4]
        length = torch.linalg.vector_norm(heading, dim=-1, keepdim=True)
        current_heading = observation['actor_history'][..., -1, 2:4]
        current_length = torch.linalg.vector_norm(current_heading, dim=-1, keepdim=True)
        fallback = torch.where(current_length > 1e-6, current_heading / current_length.clamp_min(1e-6), torch.tensor([0., 1.], device=actor.device, dtype=actor.dtype))
        heading = torch.where(length > 1e-6, heading / length.clamp_min(1e-6), fallback[..., None, :])
        actor = torch.cat((actor[..., :2], heading, actor[..., 4:]), -1)
        selected = observation['actor_observed_mask'][..., -1]
        existence = result['actor_existence_logits'].sigmoid()
        if self.mode == 'occupancy_only':
            selected = torch.zeros_like(selected)
        actor = torch.where(selected[..., None, None], actor, torch.zeros_like(actor))
        existence = torch.where(selected[..., None], existence, torch.zeros_like(existence))
        return {'future_latents': (occ / self.latent_scale).reshape(occ_shape),
                'actor_future': actor.reshape(actor_shape),
                'actor_existence': existence.reshape(actor_shape[:-1])}
