from typing import Optional

import tinygrad as tg

from .kv_cache import KVCache
from .rope import RotaryPositionalEmbedding

class MultiHeadAttention:
    def __init__(
        self,
        config: dict,
        is_causal: Optional[bool] = False,
        use_norm_k: Optional[bool] = False,
        use_norm_q: Optional[bool] = False,
        use_rope: Optional[bool] = True,
    ):
        self.embed_dim = config["embed_dim"]
        self.num_heads = config["num_heads"]
        self.num_kv_heads = config["num_kv_heads"]
        self.head_dim = config["head_dim"]
        self.max_seq_len = config["max_seq_len"]
        # initialize internally; don't rely on config keys that may not exist
        self.kv_cache = None
        self.attn_bias = config["attn_bias"]
        self.attn_dropout = config["attn_dropout"]
        self.use_rope = use_rope
        self.is_causal = is_causal

        if self.use_rope:
            if config["rope_scaling"] is not None:
                self.pos_embeddings = RotaryPositionalEmbedding(
                    config["head_dim"],
                    config["max_seq_len"],
                    config["rope_theta"],
                    is_scaling=True,
                    scale_factor=config["rope_scaling_factor"],
                    low_freq_factor=config["rope_low_freq_factor"],
                    high_freq_factor=config["rope_high_freq_factor"],
                    old_context_len=config["rope_original_max_pos_embeddings"]
                )
            else:
                self.pos_embeddings = RotaryPositionalEmbedding(
                    self.head_dim,
                    self.max_seq_len,
                    config["rope_theta"]
                )

        self.k_norm = tg.nn.RMSNorm(self.embed_dim) if use_norm_k else None
        self.q_norm = tg.nn.RMSNorm(self.embed_dim) if use_norm_q else None

        self.q_proj = tg.nn.Linear(self.embed_dim, self.num_heads * self.head_dim, bias=self.attn_bias)
        self.k_proj = tg.nn.Linear(self.embed_dim, self.num_kv_heads * self.head_dim, bias=self.attn_bias)
        self.v_proj = tg.nn.Linear(self.embed_dim, self.num_kv_heads * self.head_dim, bias=self.attn_bias)
        self.o_proj = tg.nn.Linear(self.embed_dim, self.embed_dim, bias=self.attn_bias)

    def __call__(
        self,
        x: tg.Tensor,
        attention_mask: Optional[tg.Tensor] = None,
        position_ids: Optional[tg.Tensor] = None,
    ) -> tg.Tensor:
        """
        Written for purely self attention so missing cross attention
        methods
        """
        batch_size, x_seq_len, _ = x.shape

        # k,v shape [batch, embed_dim, num_kv_heads * head_dim]
        # q shape [batch, embed_dim, num_heads * head_dim]
        k,v,q = self.k_proj(x), self.v_proj(x), self.q_proj(x)

        q_per_kv = self.num_heads // self.num_kv_heads
        # Q should have num_heads heads; K/V may have fewer (GQA)
        q = q.reshape(batch_size, x_seq_len, self.num_kv_heads * q_per_kv, self.head_dim)
        k = k.reshape(batch_size, x_seq_len, self.num_kv_heads, self.head_dim)
        v = v.reshape(batch_size, x_seq_len, self.num_kv_heads, self.head_dim)

        if position_ids is None:
            position_ids = tg.Tensor.arange(x_seq_len, device=x.device)
        elif position_ids.ndim == 2:
            position_ids = position_ids[0]

        if self.use_rope:
            q, k = self.pos_embeddings(q, k, position_ids)

        # initialize cache with correct dtype/device lazily
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

        # Update KV cache using pre-transpose K/V: [B, S, Kv, D]
        self.kv_cache.update(k, v)
        keys, values = self.kv_cache.get()  # [B, T, Kv, D]

        # transpose for SDPA
        # q: [B, H, S, D], k/v: [B, Kv, T, D]
        q = q.transpose(1, 2)
        keys = keys.transpose(1, 2)
        values = values.transpose(1, 2)

        if self.q_norm: q = self.q_norm(q)
        if self.k_norm: keys = self.k_norm(keys)

        # Repeat Kv heads to match H (GQA)
        if self.num_kv_heads != self.num_heads:
            B, Kv, T, D = keys.shape
            q_per_kv = self.num_heads // self.num_kv_heads
            try:
                keys = keys.reshape(B, Kv, 1, T, D).expand((B, Kv, q_per_kv, T, D)).flatten(1, 2)
                values = values.reshape(B, Kv, 1, T, D).expand((B, Kv, q_per_kv, T, D)).flatten(1, 2)
            except Exception:
                zeros_k = tg.Tensor.zeros((B, Kv, q_per_kv, T, D), device=keys.device)
                zeros_v = tg.Tensor.zeros((B, Kv, q_per_kv, T, D), device=values.device)
                keys = (keys.reshape(B, Kv, 1, T, D) + zeros_k).flatten(1, 2)
                values = (values.reshape(B, Kv, 1, T, D) + zeros_v).flatten(1, 2)

        # If we provide an explicit mask (causal/pad), disable built-in causal to avoid double-masking
        use_causal = self.is_causal if attention_mask is None else False
        attn_out = tg.Tensor.scaled_dot_product_attention(
            q,
            keys,
            values,
            attn_mask=attention_mask,
            dropout_p=self.attn_dropout,
            is_causal=use_causal
        )

        attn_out = attn_out.transpose(1, 2).reshape(batch_size, x_seq_len, -1)
        return self.o_proj(attn_out)
