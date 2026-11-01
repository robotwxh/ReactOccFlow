import torch.nn as nn
import torch

from .others import Normalize, exists, Mlp
from einops import rearrange
from einops_exts import check_shape, rearrange_many
from torch.nn.attention import sdpa_kernel, SDPBackend

from ...ops.flash_attention.flash_attention import FlashAttention
from torch import einsum

attn_backend = [SDPBackend.FLASH_ATTENTION, SDPBackend.EFFICIENT_ATTENTION]

class AttnBlock(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.in_channels = in_channels

        self.norm = Normalize(in_channels)

        self.attention = FlashAttention(dim_head=64, dim=in_channels, heads=8)

        self.proj_out = torch.nn.Conv2d(in_channels, in_channels, kernel_size=1, stride=1, padding=0)

    def forward(self, x):

        if len(x.shape) == 3:
            x = rearrange(x, 'b (h w) c -> b c h w', h=int(x.shape[1] ** 0.5))
        h_ = x

        h_ = self.norm(h_)

        h, w = x.shape[-2:]
        h_ = rearrange(h_, 'b c h w -> b (h w) c')
        h_ = self.attention(h_)
        h_ = rearrange(h_, 'b (h w) c -> b c h w', h=h, w=w)

        h_ = self.proj_out(h_)

        return x+h_


class DiTAttention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, attn_drop=0., proj_drop=0., use_lora=False, attention_mode='flash'):
        super().__init__()
        assert dim % num_heads == 0, 'dim should be divisible by num_heads'
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5
        self.attention_mode = attention_mode
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4).contiguous()
        q, k, v = qkv.unbind(0)   # make torchscript happy (cannot use tensor as tuple)

        if self.attention_mode == 'flash':
            # cause loss nan while using with amp
            # Optionally use the context manager to ensure one of the fused kerenels is run
            #with torch.nn.attention.sdpa_kernel(attn_backend):
            with sdpa_kernel([SDPBackend.FLASH_ATTENTION, SDPBackend.EFFICIENT_ATTENTION, SDPBackend.MATH]):
                x = torch.nn.functional.scaled_dot_product_attention(q, k, v).reshape(B, N, C) # require pytorch 2.0

        elif self.attention_mode == 'math':
            attn = (q @ k.transpose(-2, -1)) * self.scale
            attn = attn.softmax(dim=-1)
            attn = self.attn_drop(attn)
            x = (attn @ v).transpose(1, 2).reshape(B, N, C)

        else:
            raise NotImplemented

        x = self.proj(x)
        x = self.proj_drop(x)
        return x
