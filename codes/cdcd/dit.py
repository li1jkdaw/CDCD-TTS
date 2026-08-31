import torch
import torch.nn.functional as F
from torch import nn
from x_transformers.x_transformers import RotaryEmbedding

from modules import AdaLayerNorm_v2, ConvPositionEmbedding, TimestepEmbedding
from modules import ConvNeXtV2Block, DiTBlock
from utils import get_pos_embed_indices, precompute_freqs_cis


# Text embedding
class TextEmbedding(nn.Module):
    def __init__(self, vocab_size, text_dim, mask_filler_tokens=False, diffusion_context=1024, 
                 conv_layers=4, conv_mult=2):
        super().__init__()
        self.text_dim = text_dim
        self.mask_filler_tokens = mask_filler_tokens  # whether to apply masks to filler tokens
        self.diffusion_context = diffusion_context  # 1024 context is ~41 seconds of speech

        self.text_embed = nn.Embedding(vocab_size + 1, text_dim)  # new filler token with id=0
        self.vocab_size = vocab_size

        self.register_buffer("freqs_cis", precompute_freqs_cis(text_dim, diffusion_context), 
                             persistent=False)
        self.text_blocks = nn.Sequential(*[ConvNeXtV2Block(text_dim, text_dim * conv_mult)
                                           for _ in range(conv_layers)])

    def extract_text_embeddings(self, x):
        return self.text_embed(x)

    def forward(self, text, seq_len, mask=None, drop_text=False):
        text = text + 1  # filler token is a token with id=0, so increment "real" tokens
        text = text[:, :seq_len]  # cut to target sequence length
        batch_size, text_len = text.shape[0], text.shape[1]
        text = F.pad(text, (0, seq_len - text_len), value=0)
        if self.mask_filler_tokens:
            text_mask = (text == 0)
        elif mask is not None:
            text_mask = ~mask
        else:
            text_mask = None

        if drop_text:  # cfg for text
            text = torch.zeros_like(text)

        text = self.extract_text_embeddings(text)

        batch_start = torch.zeros((batch_size, ), dtype=torch.long)
        pos_idx = get_pos_embed_indices(batch_start, seq_len, max_pos=self.diffusion_context)
        text_pos_embed = self.freqs_cis[pos_idx]
        text = text + text_pos_embed

        if text_mask is not None:
            text_mask = text_mask.unsqueeze(-1).expand(-1, -1, self.text_dim)
            text = text.masked_fill(text_mask, 0.0)
            for block in self.text_blocks:
                text = block(text)
                text = text.masked_fill(text_mask, 0.0)
        else:
            text = self.text_blocks(text)

        return text


# Noisy input embedding
class InputEmbedding(nn.Module):
    def __init__(self, speech_dim, text_dim, out_dim):
        super().__init__()
        self.proj = nn.Linear(speech_dim * 2 + text_dim, out_dim)
        self.conv_pos_embed = ConvPositionEmbedding(dim=out_dim)

    def forward(self, x, cond, text_embed, mask=None):
        x = self.proj(torch.cat((x, cond, text_embed), dim=-1))
        x = self.conv_pos_embed(x, mask=mask) + x

        if mask is not None:
            mask = mask.unsqueeze(-1)
            x = x.masked_fill(~mask, 0.0)

        return x


# Diffusion Transformer backbone
class DiT(nn.Module):
    def __init__(self, dim=512, depth=8, heads=8, dim_head=64, dropout=0.1, ff_mult=4, 
                text_dim=256, mask_filler_tokens=False, diffusion_context=1024, conv_layers=4, 
                fsq_dim=8, fsq_base=3, vocab_size=34, long_skip_connection=False):
        super().__init__()

        self.dim = dim
        self.depth = depth
        self.fsq_dim = fsq_dim
        self.fsq_base = fsq_base
        self.diffusion_context = diffusion_context

        self.time_embed = TimestepEmbedding(dim)
        self.text_embed = TextEmbedding(vocab_size, text_dim, 
                                        mask_filler_tokens=mask_filler_tokens, 
                                        diffusion_context=diffusion_context, 
                                        conv_layers=conv_layers)
        self.text_cond, self.text_uncond = None, None  # text cache
        self.input_embed = InputEmbedding(fsq_dim, text_dim, dim)

        self.rotary_embed = RotaryEmbedding(dim_head)
        self.transformer_blocks = nn.ModuleList([DiTBlock(dim=dim, heads=heads, 
                                                          dim_head=dim_head, 
                                                          ff_mult=ff_mult, 
                                                          dropout=dropout) 
                                                 for _ in range(depth)])
        if long_skip_connection:
            self.long_skip_connection = nn.Linear(dim * 2, dim, bias=False)
        else:
            self.long_skip_connection = None

        self.norm_out = AdaLayerNorm_v2(dim)
        self.softmax_layer = nn.Linear(dim, fsq_base ** fsq_dim)

        self.initialize_weights()

    def initialize_weights(self):
        # Zero-out AdaLN layers in DiT blocks:
        for block in self.transformer_blocks:
            nn.init.constant_(block.pre_attn_norm.linear.weight, 0)
            nn.init.constant_(block.pre_attn_norm.linear.bias, 0)

        # Zero-out output layers:
        nn.init.constant_(self.norm_out.linear.weight, 0)
        nn.init.constant_(self.norm_out.linear.bias, 0)

    def clear_cache(self):
        self.text_cond, self.text_uncond = None, None

    def forward(self, x, cond, text, time, mask=None, cfg=False, cache=False):
        batch_size, seq_len = x.shape[0], x.shape[1]

        if time.ndim == 0:
            time = time.repeat(batch_size)

        if cfg:
            batch_size = batch_size // 2

        t = self.time_embed(time)
        if cache:
            if cfg:
                if self.text_cond is None:
                    self.text_cond = self.text_embed(text, seq_len, mask[:batch_size, :], 
                                                     drop_text=False)
                if self.text_uncond is None:
                    self.text_uncond = self.text_embed(text, seq_len, mask[:batch_size, :], 
                                                       drop_text=True)
                text_embed = torch.cat([self.text_cond, self.text_uncond], dim=0)
            else:
                if self.text_cond is None:
                    self.text_cond = self.text_embed(text, seq_len, mask)
                text_embed = self.text_cond
        else:
            if cfg:
                text_embed_cond = self.text_embed(text, seq_len, mask[:batch_size, :], 
                                                  drop_text=False)
                text_embed_uncond = self.text_embed(text, seq_len, mask[:batch_size, :], 
                                                    drop_text=True)
                text_embed = torch.cat([text_embed_cond, text_embed_uncond], dim=0)
            else:
                text_embed = self.text_embed(text, seq_len, mask)
        x = self.input_embed(x, cond, text_embed, mask=mask)

        rope = self.rotary_embed.forward_from_seq_len(seq_len)

        if self.long_skip_connection is not None:
            residual = x

        for block in self.transformer_blocks:
            x = block(x, t, mask=mask, rope=rope)

        if self.long_skip_connection is not None:
            x = self.long_skip_connection(torch.cat((x, residual), dim=-1))

        x = self.norm_out(x, t)
        logits = self.softmax_layer(x)

        if mask is not None:
            mask = mask.unsqueeze(-1)
            logits = logits.masked_fill(~mask, 0.0)

        return logits
