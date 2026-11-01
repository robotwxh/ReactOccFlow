"""Standard DiT with actor self-attention and bidirectional token exchange.

Shared conditioning components reuse the independently migrated OccFM closure.
See docs/provenance/standard_dit_sources.json, remove_unet_sources.json and
THIRD_PARTY_NOTICES.md for source attribution and licensing boundaries.
"""
from __future__ import annotations

import torch
from torch import nn

from .modules.base.postion_embed import timestep_embedding, get_fourier_embeds_from_coordinates
from .standard_dit import StandardDiT


def _safe_actor_attention(module, query, memory, valid):
    """Use a deterministic null key only for all-padding rows, never NaN softmax."""
    memory = torch.where(valid[..., None], memory, torch.zeros_like(memory))
    has_actor = valid.any(-1, keepdim=True)
    sentinel = torch.zeros_like(memory[:, :1])
    memory = torch.cat((memory, sentinel), dim=1)
    # The null key is masked when real actor keys exist.
    padding = torch.cat((~valid, has_actor), dim=1)
    result = module(query, memory, memory, key_padding_mask=padding, need_weights=False)[0]
    return torch.where(has_actor[..., None], result, torch.zeros_like(result))


class TokenInteraction(nn.Module):
    """One actor self-attention/FFN plus simultaneous bidirectional exchange."""
    def __init__(self, hidden=128, heads=4):
        super().__init__()
        self.actor_norm = nn.LayerNorm(hidden)
        self.occ_norm = nn.LayerNorm(hidden)
        self.actor_self_attention = nn.MultiheadAttention(hidden, heads, dropout=0., batch_first=True)
        self.occ_to_actor = nn.MultiheadAttention(hidden, heads, dropout=0., batch_first=True)
        self.actor_to_occ = nn.MultiheadAttention(hidden, heads, dropout=0., batch_first=True)
        self.ff_norm = nn.LayerNorm(hidden)
        self.ff = nn.Sequential(nn.Linear(hidden, hidden * 4), nn.GELU(), nn.Linear(hidden * 4, hidden))
        self.actor_gate = nn.Parameter(torch.tensor(.05))
        self.occ_gate = nn.Parameter(torch.tensor(.05))

    def forward(self, occupancy, actors, valid, *, exchange):
        norm_actor = self.actor_norm(actors)
        actors = actors + _safe_actor_attention(self.actor_self_attention, norm_actor, norm_actor, valid)
        actors = torch.where(valid[..., None], actors, torch.zeros_like(actors))
        if exchange:
            norm_actor, norm_occ = self.actor_norm(actors), self.occ_norm(occupancy)
            actor_delta = self.occ_to_actor(norm_actor, norm_occ, norm_occ, need_weights=False)[0]
            occ_delta = _safe_actor_attention(self.actor_to_occ, norm_occ, norm_actor, valid)
            occupancy = occupancy + self.occ_gate * occ_delta
            actors = actors + self.actor_gate * actor_delta
        actors = actors + self.ff(self.ff_norm(actors))
        return occupancy, torch.where(valid[..., None], actors, torch.zeros_like(actors))


class JointTransition(nn.Module):
    """Predict occupancy/actor velocities and independent future existence logits.

    Input is an ordinary tensor dictionary: the source ``noised_sequence``,
    ``timesteps``, ``trajectory`` keys plus normalized ``actor_history`` and
    ``actor_noised``; current observed masks, dimensions and classes. Future
    target masks/existence/timestamps are deliberately absent from this API.
    """
    def __init__(self, trunk=None, *, mode='joint', future_frames=6,
                 actor_history_frames=4, actor_hidden=128, actor_heads=4,
                 interaction_layers=2, history_frames=6, backbone='dit',
                 dit_config=None, latent_shape=(16, 50, 50)):
        super().__init__()
        if mode not in {'joint', 'parallel', 'occupancy_only'}:
            raise ValueError('mode must be joint, parallel, or occupancy_only')
        self.mode = mode
        self.history_frames = history_frames
        self.future_frames = future_frames
        if backbone != 'dit':
            raise ValueError('Only standard DiT is supported')
        if trunk is None:
            channels, height, width = latent_shape
            if height != width:
                raise ValueError('Standard DiT requires square spatial latent shape')
            trunk = StandardDiT(in_channels=channels, input_size=height,
                                frames=history_frames + future_frames,
                                trajectory_length=future_frames, **(dit_config or {}))
        if not isinstance(trunk, StandardDiT):
            raise ValueError('trunk must be a StandardDiT; use transition_model for custom test fields')
        self.trunk = trunk
        self.backbone = 'dit'
        self.actor_hidden = actor_hidden
        self.actor_class_embedding = nn.Embedding(19, 16, padding_idx=0)
        self.actor_encoder = nn.Sequential(nn.Linear(actor_history_frames * 8 + 3 + 16, actor_hidden), nn.SiLU(), nn.Linear(actor_hidden, actor_hidden))
        self.actor_state_projection = nn.Linear(6, actor_hidden)
        self.actor_time_encoder = nn.Sequential(nn.Linear(actor_hidden, actor_hidden), nn.SiLU(), nn.Linear(actor_hidden, actor_hidden))
        self.actor_action_encoder = nn.Sequential(nn.Linear(future_frames * 8 * 4, actor_hidden), nn.SiLU(), nn.Linear(actor_hidden, actor_hidden))
        self.register_buffer('future_slot_embedding', timestep_embedding(torch.arange(1, future_frames + 1) * .5, actor_hidden), persistent=False)
        self.occ_projection = nn.Linear(self.trunk.hidden_size, actor_hidden)
        self.occ_delta_projection = nn.Linear(actor_hidden, self.trunk.hidden_size, bias=False)
        self.interactions = nn.ModuleList([TokenInteraction(actor_hidden, actor_heads) for _ in range(interaction_layers)])
        self.actor_velocity_head = nn.Sequential(nn.LayerNorm(actor_hidden), nn.Linear(actor_hidden, 6))
        self.existence_head = nn.Sequential(nn.LayerNorm(actor_hidden), nn.Linear(actor_hidden, 1))

    def _actor_tokens(self, data):
        observed = data['actor_observed_mask'].bool()
        selected = observed[..., -1]
        observed = observed & selected[..., None]
        velocity_mask = data['actor_velocity_mask'].bool() & observed
        history = torch.where(observed[..., None], data['actor_history'], torch.zeros_like(data['actor_history']))
        history = torch.cat((history[..., :4], torch.where(velocity_mask[..., None], history[..., 4:], torch.zeros_like(history[..., 4:]))), -1)
        size = torch.where(selected[..., None], data['actor_size'], torch.zeros_like(data['actor_size']))
        classes = torch.where(selected, data['actor_class'].long() + 1, torch.zeros_like(data['actor_class'], dtype=torch.long))
        if ((classes < 0) | (classes > 18)).any():
            raise ValueError('Selected actor_class must be in [0, 17]')
        features = torch.cat((history, observed[..., None].to(history.dtype), velocity_mask[..., None].to(history.dtype)), -1).flatten(-2)
        features = torch.cat((features, size, self.actor_class_embedding(classes)), -1)
        condition = self.actor_encoder(features)
        batch, slots, future, _ = data['actor_noised'].shape
        valid = selected[..., None].expand(batch, slots, future).reshape(batch, -1)
        noised = torch.where(selected[..., None, None], data['actor_noised'], torch.zeros_like(data['actor_noised']))
        time = self.actor_time_encoder(timestep_embedding(data['timesteps'], self.actor_hidden).to(history.dtype))
        action = get_fourier_embeds_from_coordinates(8, data['trajectory'].reshape(-1, 2)).reshape(batch, -1)
        action = self.actor_action_encoder(action)
        tokens = (self.actor_state_projection(noised) + condition[:, :, None] +
                  self.future_slot_embedding.to(history.dtype)[None, None] +
                  time[:, None, None] + action[:, None, None])
        tokens = tokens.reshape(batch, -1, self.actor_hidden)
        return torch.where(valid[..., None], tokens, torch.zeros_like(tokens)), valid


    def _dit_forward(self, x, timesteps, trajectory, actors, valid):
        """Exchange actors between isotropic DiT blocks, at fixed resolution."""
        trunk = self.trunk
        tokens, condition = trunk.encode_tokens(x, timesteps, trajectory)
        exchange_after = max(1, len(trunk.blocks) // 2) - 1
        for index, block in enumerate(trunk.blocks):
            tokens = trunk.run_block(block, tokens, condition)
            if index == exchange_after:
                original = self.occ_projection(tokens)
                occupancy = original
                for interaction in self.interactions:
                    occupancy, actors = interaction(occupancy, actors, valid, exchange=True)
                tokens = tokens + self.occ_delta_projection(occupancy - original)
        return trunk.decode_tokens(tokens, condition), actors

    def forward(self, data):
        if self.mode == 'occupancy_only':
            predicted = self.trunk({key: data[key] for key in ('noised_sequence', 'timesteps', 'trajectory')})['predicted_latent']
            zero = predicted.sum() * 0
            return {'predicted_latent': predicted,
                    'actor_velocity': torch.zeros_like(data['actor_noised']) + zero,
                    'actor_existence_logits': torch.zeros_like(data['actor_noised'][..., 0]) + zero}
        actors, valid = self._actor_tokens(data)
        if self.mode == 'joint':
            predicted, actors = self._dit_forward(data['noised_sequence'], data['timesteps'], data['trajectory'], actors, valid)
        else:
            predicted = self.trunk({key: data[key] for key in ('noised_sequence', 'timesteps', 'trajectory')})['predicted_latent']
            for interaction in self.interactions:
                _, actors = interaction(None, actors, valid, exchange=False)
        shape = data['actor_noised'].shape
        velocity = self.actor_velocity_head(actors).reshape(shape)
        logits = self.existence_head(actors).reshape(shape[:-1])
        selected = data['actor_observed_mask'][..., -1].bool()
        return {'predicted_latent': predicted,
                'actor_velocity': torch.where(selected[..., None, None], velocity, torch.zeros_like(velocity)),
                'actor_existence_logits': torch.where(selected[..., None], logits, torch.zeros_like(logits))}
