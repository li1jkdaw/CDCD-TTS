import numpy as np
import torch


# utils for positional embedding computation
def precompute_freqs_cis(dim, end, theta=10000.0, theta_rescale_factor=1.0):
    theta *= theta_rescale_factor ** (dim / (dim - 2))
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2)[:(dim // 2)].float() / dim))
    t = torch.arange(end, device=freqs.device)
    freqs = torch.outer(t, freqs).float()
    freqs_cos = torch.cos(freqs)
    freqs_sin = torch.sin(freqs)
    return torch.cat([freqs_cos, freqs_sin], dim=-1)

def get_pos_embed_indices(start, length, max_pos, scale=1.0):
    scale = scale * torch.ones_like(start, dtype=torch.float32)
    pos = torch.arange(length, device=start.device, dtype=torch.float32)
    pos = (pos.unsqueeze(0) * scale.unsqueeze(1)).long()
    pos += start.unsqueeze(1)
    pos = torch.where(pos < max_pos, pos, max_pos - 1)
    return pos


# conversion between FSQ token idx and vectors in R^n
def fsq_token_to_vector(token, base=3, dim=8):
    s = np.base_repr(token, base=base)
    s = s.rjust(dim, "0")[:dim]
    v = torch.zeros(dim, dtype=torch.short)
    shift = (base - 1) // 2
    for i in range(dim):
        v[i] = int(s[dim-1-i]) - shift
    return v

def vector_to_fsq_token(v, base=3, dim=8):
    p = base ** torch.arange(dim)
    p = p.to(v.device)
    shift = (base - 1) // 2
    v = torch.clamp(torch.round(v), min=-shift, max=shift) + shift
    token = torch.sum(v * p)
    token = int(token.item())
    return token


# get mask from a batch of lengths
def sequence_mask(lengths, add_pad=0):
    max_length = torch.max(lengths) + add_pad
    x = torch.arange(max_length, dtype=lengths.dtype, device=lengths.device)
    return x.unsqueeze(0) < lengths.unsqueeze(1)
