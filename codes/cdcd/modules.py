import math
import torch
import torch.nn.functional as F
from torch import nn
from x_transformers.x_transformers import apply_rotary_pos_emb


# Sinusoidal positional embedding
class SinusPositionEmbedding(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, x, scale=1000):
        device = x.device
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device).float() * -emb)
        emb = scale * x.unsqueeze(1) * emb.unsqueeze(0)
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb


# Convolutional positional embedding
class ConvPositionEmbedding(nn.Module):
    def __init__(self, dim, kernel_size=31, groups=16):
        super().__init__()
        self.conv1d = nn.Sequential(nn.Conv1d(dim, dim, kernel_size, groups=groups, 
                                              padding=kernel_size // 2), nn.Mish(),
                                    nn.Conv1d(dim, dim, kernel_size, groups=groups, 
                                              padding=kernel_size // 2), nn.Mish())

    def forward(self, x, mask=None):
        if mask is not None:
            mask = mask.unsqueeze(-1)
            x = x.masked_fill(~mask, 0.0)

        x = x.transpose(1, 2)
        x = self.conv1d(x)
        y = x.transpose(1, 2)

        if mask is not None:
            y = y.masked_fill(~mask, 0.0)

        return y


# Global Response Normalization layer
class GRN(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.gamma = nn.Parameter(torch.zeros(1, 1, dim))
        self.beta = nn.Parameter(torch.zeros(1, 1, dim))

    def forward(self, x):
        Gx = torch.norm(x, p=2, dim=1, keepdim=True)
        Nx = Gx / (Gx.mean(dim=-1, keepdim=True) + 1e-6)
        return self.gamma * (x * Nx) + self.beta + x


# ConvNeXt-V2 Block
class ConvNeXtV2Block(nn.Module):
    def __init__(self, dim, inner_dim):
        super().__init__()
        self.dwconv = nn.Conv1d(dim, dim, kernel_size=7, padding=3, groups=dim, dilation=1)
        self.norm = nn.LayerNorm(dim, eps=1e-6)
        self.pwconv1 = nn.Linear(dim, inner_dim)
        self.act = nn.GELU()
        self.grn = GRN(inner_dim)
        self.pwconv2 = nn.Linear(inner_dim, dim)

    def forward(self, x):
        residual = x
        x = x.transpose(1, 2)
        x = self.dwconv(x)
        x = x.transpose(1, 2)
        x = self.norm(x)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.grn(x)
        x = self.pwconv2(x)
        return residual + x


# AdaLayerNorm v1
# returns modulated x for attn input and params for later mlp modulation
class AdaLayerNorm_v1(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.silu = nn.SiLU()
        self.linear = nn.Linear(dim, dim * 6)
        self.norm = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)

    def forward(self, x, emb):
        emb = self.linear(self.silu(emb))
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = torch.chunk(emb, 6, dim=1)
        x = self.norm(x) * (1 + scale_msa.unsqueeze(1)) + shift_msa.unsqueeze(1)
        return x, gate_msa, shift_mlp, scale_mlp, gate_mlp


# AdaLayerNorm v2
# returns only modulated x
class AdaLayerNorm_v2(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.silu = nn.SiLU()
        self.linear = nn.Linear(dim, dim * 2)
        self.norm = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)

    def forward(self, x, emb):
        emb = self.linear(self.silu(emb))
        scale, shift = torch.chunk(emb, 2, dim=1)
        x = self.norm(x) * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)
        return x


# FeedForward Transformer layer 
class FeedForward(nn.Module):
    def __init__(self, dim, dim_out=None, mult=4, dropout=0.0, approximate="none"):
        super().__init__()
        inner_dim = int(dim * mult)
        dim_out = dim_out if dim_out is not None else dim

        self.ff = nn.Sequential(nn.Linear(dim, inner_dim), nn.GELU(approximate=approximate), 
                                nn.Dropout(dropout), nn.Linear(inner_dim, dim_out))

    def forward(self, x):
        return self.ff(x)


# Multi-head self-attention with RoPE
class Attention(nn.Module):
    def __init__(self, dim, heads=8, dim_head=64, dropout=0.0):
        super().__init__()
        self.dim = dim
        self.heads = heads
        self.dim_head = dim_head
        self.inner_dim = dim_head * heads

        self.to_q = nn.Linear(dim, self.inner_dim)
        self.to_k = nn.Linear(dim, self.inner_dim)
        self.to_v = nn.Linear(dim, self.inner_dim)
        self.to_out = nn.Linear(self.inner_dim, dim)
        self.out_dropout = nn.Dropout(dropout)

    def forward(self, x, mask=None, rope=None):
        batch_size = x.shape[0]

        # input projections
        query = self.to_q(x)
        key = self.to_k(x)
        value = self.to_v(x)

        # split into heads
        query = query.view(batch_size, -1, self.heads, self.dim_head).transpose(1, 2)
        key = key.view(batch_size, -1, self.heads, self.dim_head).transpose(1, 2)
        value = value.view(batch_size, -1, self.heads, self.dim_head).transpose(1, 2)

        # apply rotary position embedding
        if rope is not None:
            freqs, xpos_scale = rope
            if xpos_scale is not None:
                q_xpos_scale, k_xpos_scale = (xpos_scale, xpos_scale**(-1.0))
            else:
                q_xpos_scale, k_xpos_scale = (1.0, 1.0)
            query = apply_rotary_pos_emb(query, freqs, q_xpos_scale)
            key = apply_rotary_pos_emb(key, freqs, k_xpos_scale)

        # apply attention mask
        if mask is not None:
            attn_mask = mask.unsqueeze(1).unsqueeze(1)
            attn_mask = attn_mask.expand(batch_size, self.heads, query.shape[-2], key.shape[-2])
        else:
            attn_mask = None

        # apply dot product
        x = F.scaled_dot_product_attention(query, key, value, attn_mask=attn_mask, 
                                           dropout_p=0.0, is_causal=False)
        x = x.transpose(1, 2).reshape(batch_size, -1, self.inner_dim)
        x = x.to(query.dtype)

        # output projection and dropout
        x = self.to_out(x)
        x = self.out_dropout(x)

        if mask is not None:
            mask = mask.unsqueeze(-1)
            x = x.masked_fill(~mask, 0.0)

        return x


# DiT Block
class DiTBlock(nn.Module):
    def __init__(self, dim, heads=8, dim_head=64, ff_mult=4, dropout=0.1):
        super().__init__()

        self.pre_attn_norm = AdaLayerNorm_v1(dim)
        self.attn = Attention(dim=dim, heads=heads, dim_head=dim_head, dropout=dropout)
        self.pre_ff_norm = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        self.ff = FeedForward(dim=dim, mult=ff_mult, dropout=dropout, approximate="tanh")

    def forward(self, x, t, mask=None, rope=None):
        x_norm, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.pre_attn_norm(x, emb=t)
        attn_output = self.attn(x=x_norm, mask=mask, rope=rope)
        x = x + gate_msa.unsqueeze(1) * attn_output

        x_norm = self.pre_ff_norm(x) * (1 + scale_mlp.unsqueeze(1)) + shift_mlp.unsqueeze(1)
        ff_output = self.ff(x_norm)
        x = x + gate_mlp.unsqueeze(1) * ff_output

        if mask is not None:
            mask = mask.unsqueeze(-1)
            x = x.masked_fill(~mask, 0.0)

        return x


# Diffusion timestep embedding
class TimestepEmbedding(nn.Module):
    def __init__(self, dim, freq_embed_dim=256):
        super().__init__()
        self.time_embed = SinusPositionEmbedding(freq_embed_dim)
        self.time_mlp = nn.Sequential(nn.Linear(freq_embed_dim, dim), nn.SiLU(), 
                                      nn.Linear(dim, dim))

    def forward(self, timestep):
        hidden = self.time_embed(timestep)
        hidden = hidden.to(timestep.dtype)
        time = self.time_mlp(hidden)
        return time
