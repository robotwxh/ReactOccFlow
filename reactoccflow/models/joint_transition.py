"""Selectable U-shaped or isotropic DiT with bidirectional actor exchange.

Shared conditioning components reuse the independently migrated OccFM closure.
See docs/provenance/standard_dit_sources.json, remove_unet_sources.json and
THIRD_PARTY_NOTICES.md for source attribution and licensing boundaries.
"""
from __future__ import annotations

import torch
from torch import nn
from einops import rearrange, repeat

from .modules.base.postion_embed import timestep_embedding, get_fourier_embeds_from_coordinates
from .standard_dit import StandardDiT
from .modules.transition_models.FlowMatchingV2 import FLOW_MATCHING_DOWN_X4_DiT


def default_trunk(future_frames=6, *, history_frames=6,
                  latent_shape=(16, 50, 50), unet_config=None):
    """Restore the source-compatible U-shaped trunk; retain its state keys."""
    options = dict(model_channels=128, channel_multi=[2, 4], init_kernel_size=7,
                   init_3d_conv_channels=64, attn_dim=32, temporal_attn_head=8,
                   spatial_attn_head=8, attention_mode='flash')
    if unet_config is not None:
        if not isinstance(unet_config, dict) or set(unet_config) - options.keys():
            raise ValueError('Unknown U-Net architecture options')
        options.update(unet_config)
    integer_names = set(options) - {'channel_multi', 'attention_mode'}
    multipliers = options['channel_multi']
    if (not isinstance(multipliers, (list, tuple)) or not multipliers or
            any(isinstance(v, bool) or not isinstance(v, int) or v < 1
                for v in [*(options[k] for k in integer_names), *multipliers])):
        raise ValueError('U-Net dimensions and channel multipliers must be positive integers')
    channels, height, width = latent_shape
    if height != width or height % (2 ** (len(multipliers) - 1)):
        raise ValueError('U-Net requires square latent dimensions divisible by its downsampling factor')
    widths = [options['init_3d_conv_channels'],
              *(options['init_3d_conv_channels'] * m for m in multipliers)]
    if (widths[1] != options['model_channels'] or options['model_channels'] % 4 or
            options['init_kernel_size'] % 2 != 1 or
            any(w % options['spatial_attn_head'] for w in widths)):
        raise ValueError('Incompatible U-Net embedding, attention heads, or convolution kernel')
    if options['attention_mode'] not in {'flash', 'math'}:
        raise ValueError('U-Net attention_mode must be flash or math')
    return FLOW_MATCHING_DOWN_X4_DiT(
        in_channels=channels, out_channels=channels, input_size=[height, width],
        trajectory_length=future_frames, frames=history_frames + future_frames,
        **options)


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
                 interaction_layers=2, history_frames=6, backbone=None,
                 dit_config=None, unet_config=None, latent_shape=(16, 50, 50)):
        super().__init__()
        if mode not in {'joint', 'parallel', 'occupancy_only'}:
            raise ValueError('mode must be joint, parallel, or occupancy_only')
        self.mode = mode
        self.history_frames = history_frames
        self.future_frames = future_frames
        if backbone is None:
            backbone = 'dit' if isinstance(trunk, StandardDiT) or dit_config is not None else 'occfm_unet'
        if backbone not in {'occfm_unet', 'dit'}:
            raise ValueError('backbone must be occfm_unet or dit')
        if (backbone == 'dit' and unet_config is not None) or (backbone == 'occfm_unet' and dit_config is not None):
            raise ValueError('Architecture options must match the selected backbone')
        if trunk is not None and (dit_config is not None or unet_config is not None):
            raise ValueError('Provide an explicit trunk or architecture options, not both')
        if trunk is None:
            channels, height, width = latent_shape
            if height != width:
                raise ValueError('Standard DiT requires square spatial latent shape')
            if backbone == 'dit':
                trunk = StandardDiT(in_channels=channels, input_size=height,
                                    frames=history_frames + future_frames,
                                    trajectory_length=future_frames, **(dit_config or {}))
            else:
                trunk = default_trunk(future_frames, history_frames=history_frames,
                                      latent_shape=latent_shape, unet_config=unet_config)
        expected_type = StandardDiT if backbone == 'dit' else FLOW_MATCHING_DOWN_X4_DiT
        if not isinstance(trunk, expected_type):
            raise ValueError('Explicit trunk does not match the selected backbone')
        self.trunk = trunk
        self.backbone = backbone
        self.actor_hidden = actor_hidden
        self.actor_class_embedding = nn.Embedding(19, 16, padding_idx=0)
        self.actor_encoder = nn.Sequential(nn.Linear(actor_history_frames * 8 + 3 + 16, actor_hidden), nn.SiLU(), nn.Linear(actor_hidden, actor_hidden))
        self.actor_state_projection = nn.Linear(6, actor_hidden)
        self.actor_time_encoder = nn.Sequential(nn.Linear(actor_hidden, actor_hidden), nn.SiLU(), nn.Linear(actor_hidden, actor_hidden))
        self.actor_action_encoder = nn.Sequential(nn.Linear(future_frames * 8 * 4, actor_hidden), nn.SiLU(), nn.Linear(actor_hidden, actor_hidden))
        self.register_buffer('future_slot_embedding', timestep_embedding(torch.arange(1, future_frames + 1) * .5, actor_hidden), persistent=False)
        middle_channels = (trunk.hidden_size if backbone == 'dit'
                           else trunk.mid_block2.block2.proj.out_channels)
        self.occ_projection = nn.Linear(middle_channels, actor_hidden)
        self.occ_delta_projection = nn.Linear(actor_hidden, middle_channels, bias=False)
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


    def _middle_forward(self, x, timesteps, trajectory, actors, valid):
        """Copied OccFM forward path; exchange immediately after mid_block2.

        Source math/reshape/residual ordering remains unchanged. Interaction
        inputs/outputs are explicit local values, with no hooks or global state.
        """
        trunk = self.trunk
        x = rearrange(x, 'b f c h w -> b c f h w')
        time_rel_pos_bias = trunk.time_rel_pos_bias(x.shape[2], device=x.device)
        x = trunk.init_conv(x)
        x_temporal = trunk.init_temporal_attn(x, pos_bias=time_rel_pos_bias)
        x = x + x_temporal
        r = x.clone()
        emb = trunk.t_embedder(timestep_embedding(timesteps, trunk.model_channels, repeat_only=False))
        if trajectory is not None:
            emb = trunk.encode_pose(trajectory[:, :trunk.traj_length, :]) + emb
        h = []
        timestep_spatial = repeat(emb, 'n d -> (n c) d', c=trunk.temp_embed.shape[1])
        temp_repeat_time = trunk.pos_embed.shape[1]
        height_img = trunk.input_size[0]
        for idx, (block1, block2, spatial_attn, temporal_attn, identity, downsample) in enumerate(trunk.downs):
            timestep_temp = repeat(emb, 'n d -> (n c) d', c=temp_repeat_time)
            x = block1(x, emb)
            x = block2(x, emb)
            x = rearrange(x, 'b c f h w -> (b f) (h w) c')
            x = x + trunk.pos_embed if idx == 0 else x
            x = spatial_attn(x, timestep_spatial)
            x = rearrange(x, '(b f) t c -> (b t) f c', b=r.shape[0])
            x = x + trunk.temp_embed if idx == 0 else x
            x = temporal_attn(x, timestep_temp)
            x = rearrange(x, '(b t) f c -> b c f t', b=r.shape[0])
            x = rearrange(x, 'b c f (h w) -> b c f h w', h=height_img, w=height_img)
            h.append(x)
            x = downsample(x)
            height_img = x.shape[-2]
            temp_repeat_time = x.shape[-1] * x.shape[-2]
        x = trunk.mid_block1(x, emb)
        x = trunk.mid_spatial_attn(x)
        x = trunk.mid_temporal_attn(x, pos_bias=time_rel_pos_bias)
        x = trunk.mid_block2(x, emb)
        # Research addition: project actual 3-D middle features, exchange twice,
        # and inject ONLY the exchanged delta before the source decoder path.
        middle_shape = x.shape
        original_tokens = self.occ_projection(rearrange(x, 'b c f h w -> b (f h w) c'))
        occ_tokens = original_tokens
        for interaction in self.interactions:
            occ_tokens, actors = interaction(occ_tokens, actors, valid, exchange=True)
        delta = self.occ_delta_projection(occ_tokens - original_tokens)
        x = x + delta.reshape(middle_shape[0], middle_shape[2], middle_shape[3], middle_shape[4], middle_shape[1]).permute(0, 4, 1, 2, 3)
        for block1, block2, spatial_attn, temporal_attn, upsample in trunk.ups:
            timestep_temp = repeat(emb, 'n d -> (n c) d', c=temp_repeat_time)
            x = torch.cat((x, h.pop()), dim=1)
            height_img = x.shape[-2]
            x = block1(x, emb)
            x = block2(x, emb)
            x = rearrange(x, 'b c f h w -> (b f) (h w) c')
            x = spatial_attn(x, timestep_spatial)
            x = rearrange(x, '(b f) t c -> (b t) f c', b=r.shape[0])
            x = temporal_attn(x, timestep_temp)
            x = rearrange(x, '(b t) f c -> b t f c', b=r.shape[0])
            x = rearrange(x, 'b (h w) f c -> b h w f c', h=height_img)
            x = rearrange(x, 'b h w f c -> b c f h w')
            x = upsample(x)
            temp_repeat_time = x.shape[-1] * x.shape[-2]
        x = trunk.final_conv(torch.cat((x, r), dim=1))
        return rearrange(x, 'b c f h w -> b f c h w'), actors

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
            forward = self._dit_forward if self.backbone == 'dit' else self._middle_forward
            predicted, actors = forward(data['noised_sequence'], data['timesteps'], data['trajectory'], actors, valid)
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
