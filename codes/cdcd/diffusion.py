import math
import random
import copy

import torch
import torch.nn as nn
import torch.nn.functional as F

from utils import fsq_token_to_vector, vector_to_fsq_token, sequence_mask


class Diffusion(nn.Module):
    def __init__(self, backbone, cond_min=0.05, cond_max=0.3):
        super().__init__()
        self.backbone = backbone
        self.backbone_ema = None
        self.fsq_dim = backbone.fsq_dim
        self.fsq_base = backbone.fsq_base
        self.diffusion_context = backbone.diffusion_context
        self.num_tokens = self.fsq_base ** self.fsq_dim
        es = [fsq_token_to_vector(i, base=self.fsq_base, dim=self.fsq_dim) 
              for i in range(self.num_tokens)]
        self.es = torch.stack(es, dim=0).float()
        self.cond_min = cond_min
        self.cond_max = cond_max

    def exists_ema_model(self):
        return self.backbone_ema is not None

    def init_ema_model(self):
        self.backbone_ema = copy.deepcopy(self.backbone)
        for param in self.backbone_ema.parameters():
            param.detach_()

    def update_ema_model(self, ema_weight=0.999):
        for (name, value), (name_ema, value_ema) in zip(self.backbone.named_parameters(), 
                                                        self.backbone_ema.named_parameters()):
            value_ema.data = ema_weight*value_ema.data + (1.0 - ema_weight)*value.data

    def alpha_t(self, t):
        t = t.unsqueeze(-1).unsqueeze(-1)
        return torch.clamp(1.0 - t, min=1e-6)

    def sigma_t(self, t):
        t = t.unsqueeze(-1).unsqueeze(-1)
        return torch.sqrt(torch.clamp(t*(2.0 - t), min=1e-12))

    def add_gaussian_noise(self, x0, t, mask=None):
        noise = torch.randn_like(x0)
        alpha_t = self.alpha_t(t)
        sigma_t = self.sigma_t(t)
        xt = alpha_t*x0 + sigma_t*noise
        if mask is not None:
            mask = mask.unsqueeze(-1)
            xt = xt.masked_fill(~mask, 0.0)
        return xt

    @torch.no_grad()
    def estimate_x0(self, xt, cond, cond_lens, text, t, mask=None, 
                    w=None, temperature=1.0, cache=False, use_ema=False):
        batch_size = xt.shape[0]
        if w is None:
            if use_ema:
                logits = self.backbone_ema(xt, cond, text, t, mask=mask,
                                           cfg=False, cache=cache)
            else:
                logits = self.backbone(xt, cond, text, t, mask=mask, 
                                       cfg=False, cache=cache)
        else:
            xt_joint = torch.cat([xt]*2, dim=0)
            cond_joint = torch.cat([cond]*2, dim=0)
            if mask is not None:
                mask_joint = torch.cat([mask]*2, dim=0)
            else:
                mask_joint = None
            t_joint = torch.cat([t]*2, dim=0)
            if use_ema:
                logits_joint = self.backbone_ema(xt_joint, cond_joint, text, t_joint, 
                                                 mask=mask_joint, cfg=True, cache=cache)
            else:
                logits_joint = self.backbone(xt_joint, cond_joint, text, t_joint, 
                                             mask=mask_joint, cfg=True, cache=cache)
            logits_cond, logits_uncond = torch.chunk(logits_joint, chunks=2, dim=0)
            logits = (1.0 + w)*logits_cond - w*logits_uncond

        probs = F.softmax(logits / temperature, dim=-1).unsqueeze(-1)
        e = (self.es).unsqueeze(0).unsqueeze(1).to(xt)
        x0_estim = torch.sum(e * probs, dim=2)

        for i in range(batch_size):
            x0_estim[i, :cond_lens[i], :] = cond[i, :cond_lens[i], :]

        if mask is not None:
            mask = mask.unsqueeze(-1)
            x0_estim = x0_estim.masked_fill(~mask, 0.0)

        return x0_estim

    @torch.no_grad()
    def estimate_noise(self, xt, cond, cond_lens, text, t, mask=None, 
                       w=None, temperature=1.0, cache=False, use_ema=False):
        alpha_t = self.alpha_t(t)
        sigma_t = self.sigma_t(t)
        x0_estim = self.estimate_x0(xt, cond, cond_lens, text, t, mask=mask, 
                                    w=w, temperature=temperature, cache=cache, 
                                    use_ema=use_ema)
        noise_estim = (xt - alpha_t*x0_estim) / sigma_t
        return noise_estim

    @torch.no_grad()
    def ddim_sampler(self, z, cond, cond_lens, text, mask=None, 
                     w=None, temperature=1.0, use_ema=True, n_timesteps=25):
        batch_size = z.shape[0]
        h = 1.0 / n_timesteps
        xt = z

        if use_ema:
            self.backbone_ema.clear_cache()
        else:
            self.backbone.clear_cache()

        for i in range(n_timesteps):
            t = (1.0 - i*h) * torch.ones(batch_size, dtype=z.dtype, device=z.device)
            s = (1.0 - (i + 1)*h) * torch.ones(batch_size, dtype=z.dtype, device=z.device)
            alpha_t, sigma_t = self.alpha_t(t), self.sigma_t(t)
            alpha_s, sigma_s = self.alpha_t(s), self.sigma_t(s)
            if mask is not None:
                xt = xt.masked_fill(~mask.unsqueeze(-1), 0.0)
            noise_estim = self.estimate_noise(xt, cond, cond_lens, text, t, mask=mask, 
                                              w=w, temperature=temperature, 
                                              cache=True, use_ema=use_ema)
            xt = alpha_s/alpha_t*xt - (alpha_s/alpha_t*sigma_t - sigma_s)*noise_estim

        if mask is not None:
            return xt.masked_fill(~mask.unsqueeze(-1), 0.0)
        else:
            return xt

    @torch.no_grad()
    def forward(self, seq_length, ref_tokens, text, w=None, prior_stddev=1.0, 
                temperature=1.0, use_ema=True, padding=0, n_timesteps=25):
        ref_length = len(ref_tokens)
        ref_vectors = [fsq_token_to_vector(token, base=self.fsq_base, dim=self.fsq_dim)
                       for token in ref_tokens]
        ref_vectors = torch.stack(ref_vectors, dim=0).float().to(text.device)

        prior_sample = torch.randn((1, ref_length + seq_length + padding, self.fsq_dim))
        prior_sample = prior_sample.to(text.device) * prior_stddev

        cond = torch.zeros_like(prior_sample)
        cond[0, :ref_length, :] = ref_vectors

        length = torch.LongTensor([seq_length + ref_length]).to(text.device)
        mask = sequence_mask(length, add_pad=padding)

        text = text.unsqueeze(0)

        res_vector = self.ddim_sampler(prior_sample, cond, [ref_length], text, 
                                       mask=mask, w=w, temperature=temperature, 
                                       use_ema=use_ema, n_timesteps=n_timesteps)

        res_tokens = torch.zeros(1, seq_length, dtype=torch.long, device=text.device)
        for j in range(seq_length):
            res_tokens[0, j] = vector_to_fsq_token(res_vector[0, ref_length + j, :], 
                                                   base=self.fsq_base, dim=self.fsq_dim)
        return res_tokens

    def loss_t(self, x0, cond, text, t, mask, gt_tokens):
        batch_size, seq_len = x0.shape[0], x0.shape[1]
        xt = self.add_gaussian_noise(x0, t, mask=mask)
        logits = self.backbone(xt, cond, text, t, mask=mask)
        logits = torch.reshape(logits, (batch_size*seq_len, self.num_tokens))
        targets = torch.reshape(gt_tokens, (batch_size*seq_len, ))
        ce_loss = F.cross_entropy(logits, targets, reduction="none")
        ce_loss = torch.reshape(ce_loss, (batch_size, seq_len))
        if mask is None:
            return torch.mean(ce_loss, dim=1)
        else:
            return torch.sum(ce_loss*mask, dim=1) / torch.sum(mask, dim=1)

    def compute_diff_loss(self, x0, text, lengths, gt_tokens, eps=1e-5):
        batch_size, seq_len = x0.shape[0], x0.shape[1]
        t = torch.rand(batch_size, dtype=x0.dtype, device=x0.device)
        t = torch.clamp(t, eps, 1.0-eps)

        cond = torch.zeros_like(x0)
        cond_len = random.randint(math.ceil(seq_len*self.cond_min), 
                                  math.floor(seq_len*self.cond_max))
        cond[:, :cond_len, :] = x0[:, :cond_len, :]

        mask = sequence_mask(lengths)
        loss = torch.mean(self.loss_t(x0, cond, text, t, mask, gt_tokens))
        return loss
