import math
from typing import Optional

import tinygrad

class RotaryPositionalEmbedding:
    """
    A class for Rotary Position Embedding (ROPE) used in LLMs.
    Supports scaling and non-scaling ROPE
    """
    def __init__(
        self,
        dim: int,
        max_seq_len: int,
        base: int = 100000,
        is_scaling: bool = False,
        scale_factor: float = 8.0,
        low_freq_factor: int = 1,
        high_freq_factor: int = 1,
        old_context_len: int = 8192
    ):
        self.dim = dim
        self.max_seq_len = max_seq_len
        self.base = base
        self.is_scaling = is_scaling
        self.low_freq_factor = low_freq_factor
        self.high_freq_factor = high_freq_factor
        self.old_context_len = old_context_len
        self.scale_factor = scale_factor
        self.cache = None

        # inv_freq = 1 / (base ** (i / dim)) for i in [0,2,...,dim-2]
        self.theta = 1.0 / (self.base ** (
            tinygrad.Tensor.arange(0, self.dim, 2)[: (self.dim//2)].float() / self.dim
        ))
        if self.is_scaling:
            self.theta = self.apply_scaling(self.theta)

    def build_rope_cache(self) -> tinygrad.Tensor:
        # Build the rotary position embedding cache
        seq_idx = tinygrad.Tensor.arange(self.max_seq_len)
        idx_theta = tinygrad.Tensor.einsum("i, j -> ij", seq_idx, self.theta).float()
        self.cache = tinygrad.Tensor.stack([
            tinygrad.Tensor.cos(idx_theta),
            tinygrad.Tensor.sin(idx_theta)
        ], dim=-1)

    def apply_scaling(
        self,
        freqs: tinygrad.Tensor
    ) -> tinygrad.Tensor:
        low_freq_wavelen = self.old_context_len / self.low_freq_factor
        high_freq_wavelen = self.old_context_len / self.high_freq_factor
        new_freqs = []
        for freq in freqs:
            wavelen = 2 * math.pi / freq
            if wavelen < high_freq_wavelen:
                new_freqs.append(freq)
            elif wavelen > low_freq_wavelen:
                new_freqs.append(freq / self.scale_factor)
            else:
                assert low_freq_wavelen != high_freq_wavelen
                smooth = (self.old_context_len / wavelen - self.low_freq_factor) / (
                     self.high_freq_factor -  self.low_freq_factor
                )
                new_freqs.append(
                    (1 - smooth) * freq /  self.scale_factor + smooth * freq
                )

        return tinygrad.Tensor(new_freqs, dtype=freqs.dtype, device=freqs.device)

    def forward(self, x: tinygrad.Tensor) -> tinygrad.Tensor:
        if self.cache is None:
            self.build_rope_cache()

        # Support inputs with shape (..., T, D) where D==self.dim and T is sequence length
        assert x.shape[-1] == self.dim, "Last dim must equal RoPE dim"
        assert self.dim % 2 == 0, "RoPE dim must be even"

        seq_len = x.shape[-2]
        # cache: (max_seq_len, D//2, 2) -> (T, D//2, 2)
        cs = self.cache[:seq_len]

        # Pair the last dimension: (..., T, D//2, 2)
        x_pairs = x.float().reshape(*x.shape[:-1], -1, 2)

        # Add leading singleton dims to cache for broadcasting
        lead = x_pairs.ndim - cs.ndim
        if lead > 0:
            cs = cs.reshape(*((1,) * lead), *cs.shape)

        cos = cs[..., 0]
        sin = cs[..., 1]

        x0 = x_pairs[..., 0]
        x1 = x_pairs[..., 1]
        y0 = x0 * cos - x1 * sin
        y1 = x0 * sin + x1 * cos

        y_pairs = tinygrad.Tensor.stack([y0, y1], dim=-1)
        y = y_pairs.reshape(*x.shape)
        return y.cast(x.dtype)
