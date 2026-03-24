"""
Shared building blocks for Encoder.

Components:
    - modulate: AdaLN modulation
    - rotate_half, broadcat: RoPE helpers
    - get_2d_sincos_pos_embed: positional embedding
    - VisionRotaryEmbeddingFast: 1D RoPE
    - SwiGLUFFN: SwiGLU feedforward
    - RMSNorm: Root Mean Square normalization
    - NormAttention: Multi-head attention with QK normalization + RoPE
    - GaussianFourierEmbedding: Timestep embedding (handles t<0 for reconstruction)
    - LabelEmbedder: Class label embedding with CFG dropout
"""

import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat


# ---------------------------------------------------------------------------
# AdaLN modulation
# ---------------------------------------------------------------------------

def modulate(x, shift, scale):
    if shift is not None:
        return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)
    return x * (1 + scale.unsqueeze(1))


# ---------------------------------------------------------------------------
# RoPE helpers
# ---------------------------------------------------------------------------

def broadcat(tensors, dim=-1):
    num_tensors = len(tensors)
    shape_lens = set(len(t.shape) for t in tensors)
    assert len(shape_lens) == 1, "tensors must all have the same number of dimensions"
    shape_len = list(shape_lens)[0]
    dim = (dim + shape_len) if dim < 0 else dim
    dims = list(zip(*(list(t.shape) for t in tensors)))
    expandable_dims = [(i, val) for i, val in enumerate(dims) if i != dim]
    assert all(len(set(t[1])) <= 2 for t in expandable_dims), (
        "invalid dimensions for broadcastable concatenation"
    )
    max_dims = [(t[0], max(t[1])) for t in expandable_dims]
    expanded_dims = [(t[0], (t[1],) * num_tensors) for t in max_dims]
    expanded_dims.insert(dim, (dim, dims[dim]))
    expandable_shapes = list(zip(*(t[1] for t in expanded_dims)))
    tensors = [t.expand(*s) for t, s in zip(tensors, expandable_shapes)]
    return torch.cat(tensors, dim=dim)


def rotate_half(x):
    x = rearrange(x, "... (d r) -> ... d r", r=2)
    x1, x2 = x.unbind(dim=-1)
    x = torch.stack((-x2, x1), dim=-1)
    return rearrange(x, "... d r -> ... (d r)")


# ---------------------------------------------------------------------------
# Sinusoidal positional embedding
# ---------------------------------------------------------------------------

def get_1d_sincos_pos_embed_from_grid(embed_dim, pos):
    assert embed_dim % 2 == 0
    omega = np.arange(embed_dim // 2, dtype=np.float64)
    omega /= embed_dim / 2.0
    omega = 1.0 / 10000**omega
    pos = pos.reshape(-1)
    out = np.einsum("m,d->md", pos, omega)
    return np.concatenate([np.sin(out), np.cos(out)], axis=1)


def get_2d_sincos_pos_embed_from_grid(embed_dim, grid):
    assert embed_dim % 2 == 0
    emb_h = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[0])
    emb_w = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[1])
    return np.concatenate([emb_h, emb_w], axis=1)



def get_2d_sincos_pos_embed(embed_dim, grid_size, cls_token=False, extra_tokens=0):
    grid_h = np.arange(grid_size, dtype=np.float32)
    grid_w = np.arange(grid_size, dtype=np.float32)
    grid = np.meshgrid(grid_w, grid_h)
    grid = np.stack(grid, axis=0).reshape([2, 1, grid_size, grid_size])
    pos_embed = get_2d_sincos_pos_embed_from_grid(embed_dim, grid)
    if cls_token and extra_tokens > 0:
        pos_embed = np.concatenate([np.zeros([extra_tokens, embed_dim]), pos_embed], axis=0)
    return pos_embed


# ---------------------------------------------------------------------------
# 1D Rotary Positional Embedding
# ---------------------------------------------------------------------------

class VisionRotaryEmbeddingFast(nn.Module):
    def __init__(
        self,
        dim,
        pt_seq_len=16,
        ft_seq_len=None,
        custom_freqs=None,
        freqs_for="lang",
        theta=10000,
        max_freq=10,
        num_freqs=1,
        num_cls_token=0,
    ):
        super().__init__()
        if custom_freqs:
            freqs = custom_freqs
        elif freqs_for == "lang":
            freqs = 1.0 / (theta ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim))
        elif freqs_for == "pixel":
            freqs = torch.linspace(1.0, max_freq / 2, dim // 2) * math.pi
        elif freqs_for == "constant":
            freqs = torch.ones(num_freqs).float()
        else:
            raise ValueError(f"unknown modality {freqs_for}")

        if ft_seq_len is None:
            ft_seq_len = pt_seq_len
        t = torch.arange(ft_seq_len) / ft_seq_len * pt_seq_len
        freqs = torch.einsum("..., f -> ... f", t, freqs)
        freqs = repeat(freqs, "... n -> ... (n r)", r=2)

        freqs_cos = freqs.cos().view(-1, freqs.shape[-1])
        freqs_sin = freqs.sin().view(-1, freqs.shape[-1])

        if num_cls_token > 0:
            N_img, D = freqs_cos.shape
            cos_pad = torch.ones(num_cls_token, D, dtype=freqs_cos.dtype)
            sin_pad = torch.zeros(num_cls_token, D, dtype=freqs_sin.dtype)
            freqs_cos = torch.cat([cos_pad, freqs_cos], dim=0)
            freqs_sin = torch.cat([sin_pad, freqs_sin], dim=0)

        self.register_buffer("freqs_cos", freqs_cos)
        self.register_buffer("freqs_sin", freqs_sin)

    def forward(self, t):
        _, _, Lt, _ = t.shape  # B, num_heads, L, dim
        freqs_cos = self.freqs_cos[:Lt]
        freqs_sin = self.freqs_sin[:Lt]
        return t * freqs_cos + rotate_half(t) * freqs_sin




# ---------------------------------------------------------------------------
# SwiGLU feedforward
# ---------------------------------------------------------------------------

class SwiGLUFFN(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, drop=0.0, bias=True):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.w12 = nn.Linear(in_features, 2 * hidden_features, bias=bias)
        self.w3 = nn.Linear(hidden_features, out_features, bias=bias)

    def forward(self, x):
        x12 = self.w12(x)
        x1, x2 = x12.chunk(2, dim=-1)
        hidden = F.silu(x1) * x2
        return self.w3(hidden)


# ---------------------------------------------------------------------------
# RMSNorm
# ---------------------------------------------------------------------------

class RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-6, weighted_norm=True):
        super().__init__()
        self.eps = eps
        self.weighted_norm = weighted_norm
        if weighted_norm:
            self.weight = nn.Parameter(torch.ones(dim))

    def _norm(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x):
        output = self._norm(x.float()).type_as(x)
        if not self.weighted_norm:
            return output
        return output * self.weight.to(output.dtype)


# ---------------------------------------------------------------------------
# NormAttention: multi-head attention with optional QK-norm + RoPE
# ---------------------------------------------------------------------------

class NormAttention(nn.Module):
    def __init__(
        self,
        dim,
        num_heads=8,
        qkv_bias=False,
        qk_norm=False,
        attn_drop=0.0,
        proj_drop=0.0,
        norm_layer=nn.LayerNorm,
        fused_attn=True,
        use_rmsnorm=False,
        return_attn=False,
    ):
        super().__init__()
        assert dim % num_heads == 0, "dim should be divisible by num_heads"
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim**-0.5
        self.fused_attn = fused_attn

        if use_rmsnorm:
            norm_layer = RMSNorm

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.q_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
        self.k_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        self.return_attn = return_attn

    def forward(self, x, rope=None):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)
        q, k = self.q_norm(q), self.k_norm(k)

        if rope is not None:
            q = rope(q)
            k = rope(k)

        if self.fused_attn:
            q = q.to(v.dtype)
            k = k.to(v.dtype)
            x = F.scaled_dot_product_attention(
                q, k, v, dropout_p=self.attn_drop.p if self.training else 0.0
            )
        else:
            q = q * self.scale
            attn = q @ k.transpose(-2, -1)
            attn = attn.softmax(dim=-1)
            if self.return_attn:
                h = attn.shape[1]
                attn_weights = attn[:, :h].detach().float().cpu()
            attn = self.attn_drop(attn)
            x = attn @ v

        x = x.transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        if self.return_attn:
            return x, attn_weights
        return x


# ---------------------------------------------------------------------------
# GaussianFourierEmbedding: timestep embedding with special reconstruction token
# ---------------------------------------------------------------------------

class GaussianFourierEmbedding(nn.Module):
    """
    Gaussian Fourier Embedding for continuous timesteps t in [0, 1],
    plus a special embedding for reconstruction phase when t < 0.
    """
    def __init__(self, hidden_size, embedding_size=256, scale=1.0):
        super().__init__()
        self.embedding_size = embedding_size
        self.scale = scale

        # fixed random frequencies (no grad)
        W = torch.normal(mean=0.0, std=self.scale, size=(embedding_size,))
        self.W = nn.Parameter(W, requires_grad=False)

        self.mlp = nn.Sequential(
            nn.Linear(embedding_size * 2, hidden_size, bias=True),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size, bias=True),
        )
        self.hidden_size = hidden_size

        ## this is not used in the latest model.
        self.special_time_emb = nn.Parameter(torch.zeros(hidden_size))
        nn.init.normal_(self.special_time_emb, std=0.02)


    def forward(self, t):
        # allow scalar t
        if t.dim() == 0:
            t = t[None]

        dev = next(self.parameters()).device
        t = t.to(device=dev, dtype=torch.float32)

        mask_special = (t < 0)
        mask_normal  = ~mask_special

        out = torch.empty(t.shape[0], self.hidden_size, device=dev, dtype=torch.float32)

        if mask_normal.any():
            tn = t[mask_normal][:, None]  # [N, 1]
            angles = tn * self.W[None, :] * (2.0 * math.pi)  # [N, E]
            feats = torch.cat([torch.sin(angles), torch.cos(angles)], dim=-1)  # [N, 2E]
            # Cast to float32 to handle bf16/fp16 autocast
            out[mask_normal] = self.mlp(feats).to(dtype=torch.float32)

        return out


# ---------------------------------------------------------------------------
# LabelEmbedder: class labels with CFG dropout + special -1 token
# ---------------------------------------------------------------------------

class LabelEmbedder(nn.Module):
    """
    Embeds class labels into vector representations.
    Supports:
      - classifier-free guidance dropout
      - labels == -1 as a special 'rep-learning' token with its own embedding
    """
    def __init__(self, num_classes, hidden_size, dropout_prob):
        super().__init__()
        use_cfg_embedding = 1 if dropout_prob > 0 else 0
        self.null_label_index = num_classes if use_cfg_embedding else None
        self.embedding_table = nn.Embedding(num_classes + use_cfg_embedding + 1, hidden_size) # +1 is not required.
        self.num_classes = num_classes
        self.dropout_prob = dropout_prob
        nn.init.normal_(self.embedding_table.weight, std=0.02)

    def token_drop(self, labels, force_drop_ids):
        """
        Drops *valid* labels (0..num_classes-1) to enable classifier-free guidance.
        Special labels (-1) are not dropped.
        """
        if self.null_label_index is None:
            return labels

        valid_mask = (labels >= 0)
        if force_drop_ids is None:
            drop_mask = torch.rand(labels.shape[0], device=labels.device) < self.dropout_prob
        else:
            drop_mask = (force_drop_ids == 1)
        drop_mask = drop_mask & valid_mask
        out = labels.clone()
        out[drop_mask] = self.null_label_index
        return out

    def forward(self, labels, train, force_drop_ids=None):
        """
        labels: int64 tensor [N]; use -1 for the special 'rep-learning' condition.
        """
        if labels.dim() == 0:
            labels = labels[None]
        use_dropout = self.dropout_prob > 0
        if (train and use_dropout) or (force_drop_ids is not None):
            labels = self.token_drop(labels, force_drop_ids)
        return self.embedding_table(labels)
