from __future__ import annotations

from typing import Optional

import torch
from torch import nn
from torch.nn import functional as F

from .kv_cache import KVCache
from .rope import RotaryPositionalEmbedding


class _FallbackRMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        normed = x * torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + self.eps)
        return normed * self.weight


def _rms_norm(dim: int, eps: float = 1e-6) -> nn.Module:
    if hasattr(nn, "RMSNorm"):
        return nn.RMSNorm(dim, eps=eps)
    return _FallbackRMSNorm(dim, eps=eps)


def _prepare_attention_mask(
    attention_mask: Optional[torch.Tensor],
    query: torch.Tensor,
) -> Optional[torch.Tensor]:
    if attention_mask is None:
        return None

    mask = attention_mask.to(device=query.device)

    if mask.ndim == 1:
        mask = mask.unsqueeze(0)

    if mask.ndim == 2:
        mask = mask[:, None, None, :]

    if mask.ndim == 3:
        mask = mask[:, None, :, :]

    if mask.dtype == torch.bool:
        zero = torch.zeros((), dtype=query.dtype, device=query.device)
        neg_inf = torch.full((), torch.finfo(query.dtype).min, dtype=query.dtype, device=query.device)
        return torch.where(mask, zero, neg_inf)

    if mask.dtype.is_floating_point:
        return mask.to(dtype=query.dtype)

    zero = torch.zeros((), dtype=query.dtype, device=query.device)
    neg_inf = torch.full((), torch.finfo(query.dtype).min, dtype=query.dtype, device=query.device)
    return torch.where(mask > 0, zero, neg_inf)


class MultiHeadAttention(nn.Module):
    def __init__(
        self,
        config: dict,
        is_causal: Optional[bool] = False,
    ):
        super().__init__()
        self.config = config
        self.embed_dim = config["embed_dim"]
        self.num_heads = config["num_heads"]
        self.num_kv_heads = config["num_kv_heads"]
        self.head_dim = config["head_dim"]
        self.max_seq_len = config["max_seq_len"]
        self.kv_cache: KVCache | None = None

        self.attn_bias = config["attn_bias"]
        self.attn_dropout = config["attn_dropout"]
        self.is_causal = bool(is_causal)

        if config["rope_scaling"] is not None:
            self.pos_embeddings = RotaryPositionalEmbedding(
                self.head_dim,
                self.max_seq_len,
                config["rope_theta"],
                is_scaling=True,
                scale_factor=config["rope_scaling_factor"],
                low_freq_factor=config["rope_low_freq_factor"],
                high_freq_factor=config["rope_high_freq_factor"],
                old_context_len=config["rope_original_max_pos_embeddings"],
            )
        else:
            self.pos_embeddings = RotaryPositionalEmbedding(
                self.head_dim,
                self.max_seq_len,
                config["rope_theta"],
            )

        self.k_norm = _rms_norm(self.head_dim) if config["qk_norm"] else None
        self.q_norm = _rms_norm(self.head_dim) if config["qk_norm"] else None

        self.q_proj = nn.Linear(
            self.embed_dim,
            self.num_heads * self.head_dim,
            bias=self.attn_bias,
        )
        self.k_proj = nn.Linear(
            self.embed_dim,
            self.num_kv_heads * self.head_dim,
            bias=self.attn_bias,
        )
        self.v_proj = nn.Linear(
            self.embed_dim,
            self.num_kv_heads * self.head_dim,
            bias=self.attn_bias,
        )
        self.o_proj = nn.Linear(
            self.num_heads * self.head_dim,
            self.embed_dim,
            bias=self.attn_bias,
        )

    def forward(
        self,
        x: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        batch_size, seq_len, _ = x.shape

        k = self.k_proj(x)
        v = self.v_proj(x)
        q = self.q_proj(x)

        q_per_kv = self.num_heads // self.num_kv_heads
        q = q.reshape(batch_size, seq_len, self.num_kv_heads * q_per_kv, self.head_dim)
        k = k.reshape(batch_size, seq_len, self.num_kv_heads, self.head_dim)
        v = v.reshape(batch_size, seq_len, self.num_kv_heads, self.head_dim)

        if position_ids is None:
            position_ids = torch.arange(seq_len, device=x.device)
        elif position_ids.ndim == 2:
            position_ids = position_ids[0]

        if self.q_norm is not None:
            q = self.q_norm(q)
        if self.k_norm is not None:
            k = self.k_norm(k)

        q, k = self.pos_embeddings(q, k, position_ids)

        if self.kv_cache is None:
            self.kv_cache = KVCache(
                self.max_seq_len,
                batch_size,
                self.num_kv_heads,
                self.max_seq_len,
                self.head_dim,
                dtype=x.dtype,
                device=x.device,
            )

        self.kv_cache.update(k, v)
        k, v = self.kv_cache.get()

        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        if self.num_kv_heads != self.num_heads:
            q_per_kv = self.num_heads // self.num_kv_heads
            k = k.repeat_interleave(q_per_kv, dim=1)
            v = v.repeat_interleave(q_per_kv, dim=1)

        attn_mask = _prepare_attention_mask(attention_mask, q)
        use_causal = self.is_causal if attn_mask is None else False

        dropout_p = self.attn_dropout if self.training else 0.0
        attn_out = F.scaled_dot_product_attention(
            q,
            k,
            v,
            attn_mask=attn_mask,
            dropout_p=dropout_p,
            is_causal=use_causal,
        )

        attn_out = attn_out.transpose(1, 2).reshape(batch_size, seq_len, -1)
        return self.o_proj(attn_out)
