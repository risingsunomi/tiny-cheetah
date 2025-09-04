from typing import Optional

import tinygrad as tg

from .kv_cache import KVCache
from .rope import RotaryPositionalEmbedding

class MultiHeadAttention:
    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        num_kv_heads: int,
        head_dim: int,
        max_seq_len: int,
        kv_cache: Optional[KVCache] = None,
        is_causal: bool = False,
        attn_bias: Optional[bool] = False,
        attn_dropout: Optional[float] = 0.0,
        use_rope: Optional[bool] = True,
        rope_base: Optional[int] = 10000,
        rope_scaling_opt: Optional[dict] = None
    ):
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim
        self.max_seq_len = max_seq_len
        self.kv_cache = kv_cache
        self.is_causal = is_causal
        self.attn_bias = attn_bias
        self.attn_dropout = attn_dropout
        self.use_rope = use_rope
        self.rope_base = rope_base
        self.rope_scaling_opt = rope_scaling_opt

        if self.use_rope:
            if rope_scaling_opt:
                self.pos_embeddings = RotaryPositionalEmbedding(
                    head_dim,
                    max_seq_len,
                    rope_base,
                    is_scaling=True,
                    **rope_scaling_opt
                )
            else:
                self.pos_embeddings = RotaryPositionalEmbedding(
                    head_dim,
                    max_seq_len,
                    rope_base
                )
        
        self.q_proj = tg.nn.Linear(embed_dim, num_heads * head_dim, bias=attn_bias)
        self.k_proj = tg.nn.Linear(embed_dim, num_kv_heads * head_dim, bias=attn_bias)
        self.v_proj = tg.nn.Linear(embed_dim, num_kv_heads * head_dim, bias=attn_bias)
        self.out_proj = tg.nn.Linear(embed_dim, embed_dim, bias=attn_bias)

    def __call__(
        self,
        x: tg.Tensor,
        mask: Optional[tg.Tensor] = None,
        input_pos: Optional[tg.Tensor] = None,
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
        q = q.reshape(batch_size, x_seq_len, self.num_heads, self.head_dim)

        if self.use_rope:
            # q embed shape the same as input
            q = self.pos_embeddings(q, input_pos)

        # q shape [batch, q_per_kv, x_seq_len, head_dim]
        q = q.transpose(1, 2)

        if self.kv_cache is None:
            # initialize cache with max_seq_len capacity
            self.kv_cache = KVCache(self.max_seq_len)
            self.kv_cache.cache_k = tg.Tensor.zeros((
                batch_size, self.max_seq_len, self.head_dim
            ))
            self.kv_cache.cache_v = tg.Tensor.zeros((
                batch_size, self.max_seq_len, self.head_dim
            ))

        k = k.reshape(batch_size, x_seq_len, self.num_kv_heads, self.head_dim)
        v = v.reshape(batch_size, x_seq_len, self.num_kv_heads, self.head_dim)

        if self.use_rope:
            k = self.pos_embeddings(k, input_pos)

        # k,v shape [batch, num_kv_heads, seq_len, head_dim]
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        self.kv_cache.update(k, v)

        if self.num_kv_heads != self.num_heads:
            expand_shape = (batch_size, self.num_kv_heads, q_per_kv, -1, self.head_dim)
            k = k.reshape(*expand_shape).flatten(1, 2)
            v = v.reshape(*expand_shape).flatten(1, 2)

        attn_out = tg.Tensor.scaled_dot_product_attention(
            q,
            k,
            v,
            attn_mask=mask,
            dropout_p=self.attn_dropout,
            is_causal=self.is_causal
        )

        attn_out = attn_out.transpose(1, 2).reshape(batch_size, x_seq_len, -1)
        return self.out_proj(attn_out)
