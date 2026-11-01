"""Isotropic space-time DiT velocity field, without a U-Net encoder/decoder.

DiTBlock and sinusoidal/Fourier embeddings reuse the independently migrated
OccFM dependency closure. See docs/provenance/standard_dit_sources.json and
THIRD_PARTY_NOTICES.md for source hashes and unresolved upstream licensing.
Patch projection, unpatchification and the adaLN output layer are project-owned.
"""
from __future__ import annotations

import math

import torch
from torch import nn
from torch.utils.checkpoint import checkpoint
from einops import rearrange

from .modules.base.dit import DiTBlock
from .modules.base.postion_embed import (
    get_2d_sincos_pos_embed, get_1d_sincos_temp_embed,
    get_fourier_embeds_from_coordinates, timestep_embedding,
)


class DiTOutput(nn.Module):
    """Conditioned final normalization and zero-initialized patch prediction."""

    def __init__(self, hidden_size, patch_values):
        super().__init__()
        self.norm = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.modulation = nn.Sequential(nn.SiLU(), nn.Linear(hidden_size, 2 * hidden_size))
        self.linear = nn.Linear(hidden_size, patch_values)

    def forward(self, tokens, condition):
        shift, scale = self.modulation(condition).chunk(2, dim=-1)
        return self.linear(self.norm(tokens) * (1 + scale[:, None]) + shift[:, None])


class StandardDiT(nn.Module):
    """Patchify -> fixed-width global DiT blocks -> unpatchify.

    A frame is patched spatially; all frame/patch tokens share global attention.
    Generation time and the supplied ego plan condition adaLN-Zero. Physical
    time has a separate fixed positional embedding. No down/up stages or
    encoder-to-decoder skip connections are constructed. Output is a flow
    velocity, even though the reusable block is named Diffusion Transformer.
    """

    architecture = 'standard_dit'

    def __init__(self, *, in_channels=16, input_size=50, frames=12,
                 trajectory_length=6, patch_size=5, hidden_size=256, depth=8,
                 num_heads=8, mlp_ratio=4., activation_checkpointing=True):
        super().__init__()
        integer_options = {'in_channels': in_channels, 'input_size': input_size,
                           'frames': frames, 'trajectory_length': trajectory_length,
                           'patch_size': patch_size, 'hidden_size': hidden_size,
                           'depth': depth, 'num_heads': num_heads}
        if any(isinstance(v, bool) or not isinstance(v, int) or v < 1 for v in integer_options.values()):
            raise ValueError('DiT dimensions, depth and heads must be positive integers')
        if input_size % patch_size or hidden_size % 4 or hidden_size % num_heads:
            raise ValueError('DiT requires divisible image/patch size and hidden size divisible by 4 and heads')
        if not math.isfinite(mlp_ratio) or mlp_ratio <= 0 or int(hidden_size * mlp_ratio) < 1:
            raise ValueError('DiT mlp_ratio must be finite and positive')
        if not isinstance(activation_checkpointing, bool):
            raise ValueError('DiT activation_checkpointing must be boolean')
        self.in_channels, self.input_size, self.frames = in_channels, input_size, frames
        self.traj_length, self.patch_size = trajectory_length, patch_size
        self.hidden_size, self.depth, self.num_heads = hidden_size, depth, num_heads
        self.grid_size = input_size // patch_size
        self.activation_checkpointing = activation_checkpointing
        self.config = {**integer_options, 'mlp_ratio': float(mlp_ratio),
                       'activation_checkpointing': activation_checkpointing}
        patch_values = patch_size ** 2 * in_channels
        self.patch_embed = nn.Linear(patch_values, hidden_size)
        self.t_embedder = nn.Sequential(nn.Linear(hidden_size, hidden_size), nn.SiLU(),
                                       nn.Linear(hidden_size, hidden_size))
        self.traj_encoder = nn.Sequential(nn.Linear(trajectory_length * 8 * 4, hidden_size),
                                         nn.SiLU(), nn.Linear(hidden_size, hidden_size))
        self.register_buffer('pos_embed', torch.from_numpy(
            get_2d_sincos_pos_embed(hidden_size, self.grid_size)).float().unsqueeze(0))
        self.register_buffer('temp_embed', torch.from_numpy(
            get_1d_sincos_temp_embed(hidden_size, frames)).float().unsqueeze(0))
        # The copied math attention correctly merges heads; see provenance for
        # why the source flash branch is deliberately not selected here.
        self.blocks = nn.ModuleList([
            DiTBlock(hidden_size, num_heads, mlp_ratio=mlp_ratio,
                     cond_size=hidden_size, attention_mode='math') for _ in range(depth)
        ])
        self.final_layer = DiTOutput(hidden_size, patch_values)
        self._initialize_weights()

    @property
    def token_count(self):
        return self.frames * self.grid_size ** 2

    def _initialize_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
        for block in self.blocks:
            nn.init.zeros_(block.adaLN_modulation[-1].weight)
            nn.init.zeros_(block.adaLN_modulation[-1].bias)
        for layer in (self.t_embedder[0], self.t_embedder[2]):
            nn.init.normal_(layer.weight, std=.02)
        nn.init.zeros_(self.final_layer.modulation[-1].weight)
        nn.init.zeros_(self.final_layer.modulation[-1].bias)
        nn.init.zeros_(self.final_layer.linear.weight)
        nn.init.zeros_(self.final_layer.linear.bias)

    def patchify(self, sequence):
        if sequence.ndim != 5 or tuple(sequence.shape[1:]) != (
                self.frames, self.in_channels, self.input_size, self.input_size):
            raise ValueError('DiT sequence shape must be B,frames,C,H,W matching its configuration')
        return rearrange(sequence, 'b f c (h p) (w q) -> b (f h w) (p q c)',
                         p=self.patch_size, q=self.patch_size)

    def unpatchify(self, patches):
        if patches.ndim != 3 or tuple(patches.shape[1:]) != (
                self.token_count, self.patch_size ** 2 * self.in_channels):
            raise ValueError('DiT output patches do not match its configured token/patch shape')
        return rearrange(patches, 'b (f h w) (p q c) -> b f c (h p) (w q)',
                         f=self.frames, h=self.grid_size, w=self.grid_size,
                         p=self.patch_size, q=self.patch_size, c=self.in_channels)

    def encode_tokens(self, sequence, timesteps, trajectory):
        patches = self.patchify(sequence)
        batch = sequence.shape[0]
        if timesteps is None or tuple(timesteps.shape) != (batch,):
            raise ValueError('DiT timesteps shape must be B')
        if trajectory is None or tuple(trajectory.shape) != (batch, self.traj_length, 2):
            raise ValueError('DiT trajectory shape must be B,trajectory_length,2')
        tokens = self.patch_embed(patches)
        position = self.pos_embed[:, None] + self.temp_embed[:, :, None]
        tokens = tokens + position.reshape(1, self.token_count, self.hidden_size).to(tokens)
        time = timestep_embedding(timesteps, self.hidden_size).to(tokens)
        action = get_fourier_embeds_from_coordinates(8, trajectory.reshape(-1, 2))
        condition = self.t_embedder(time) + self.traj_encoder(action.reshape(batch, -1).to(tokens))
        return tokens, condition

    def run_block(self, block, tokens, condition):
        if self.activation_checkpointing and self.training and torch.is_grad_enabled():
            return checkpoint(block, tokens, condition, use_reentrant=False)
        return block(tokens, condition)

    def decode_tokens(self, tokens, condition):
        return self.unpatchify(self.final_layer(tokens, condition))

    def forward(self, data):
        tokens, condition = self.encode_tokens(
            data['noised_sequence'], data['timesteps'], data['trajectory'])
        for block in self.blocks:
            tokens = self.run_block(block, tokens, condition)
        return {'predicted_latent': self.decode_tokens(tokens, condition)}
