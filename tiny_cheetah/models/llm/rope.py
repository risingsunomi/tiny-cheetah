import math
from typing import Optional, Tuple

import tinygrad as tg

# From https://github.com/tinygrad/tinygrad/blob/master/extra/models/llama.py
# From https://github.com/BatSmacker84/exo/blob/fbec1d2b10ccf3a804294c0742b69d508f7b5fb8/exo/inference/tinygrad/models/llama.py
def complex_multi(A: tg.Tensor, C: tg.Tensor, D: tg.Tensor) -> tg.Tensor:
  """
  real and imaginary complex multiplication
  """
  a,b = A[..., 0:1], A[..., 1:2]
  ro = a*C - b*D
  co = a*D + b*C
  return ro.cat(co, dim=-1)

class RotaryPositionalEmbedding:
    """
    A class for Rotary Position Embedding (ROPE) used in LLMs.
    Supports scaling and non-scaling ROPE
    """
    def __init__(
        self,
        dim: int,
        max_seq_len: int,
        theta: float = 100000.0,
        is_scaling: bool = False,
        scale_factor: float = 8.0,
        low_freq_factor: float = 1.0,
        high_freq_factor: float = 1.0,
        old_context_len: int = 8192,
        dtype = tg.dtypes.half
    ):
        self.dim = dim
        self.max_seq_len = max_seq_len
        self.theta = theta
        self.is_scaling = is_scaling
        self.low_freq_factor = low_freq_factor
        self.high_freq_factor = high_freq_factor
        self.old_context_len = old_context_len
        self.scale_factor = scale_factor
        self.cache = None
        self.dtype = dtype

        # precompute freqs
        self.freqs = 1.0 / (self.theta ** (
            tg.Tensor.arange(
                0,
                self.dim,
                2
            )[: (self.dim//2)] / self.dim
        ))
    
        if self.is_scaling:
            self.freqs = self.apply_scaling(self.freqs)

        end = self.max_seq_len * 2
        self.freqs = tg.Tensor.arange(end).unsqueeze(dim=1) * self.freqs.unsqueeze(dim=0)
        self.freqs = tg.Tensor.stack(
            self.freqs.cos().cast(self.dtype),
            self.freqs.sin().cast(self.dtype), 
            dim=-1
        ).reshape(
            1,
            end,
            1,
            self.dim//2,
            2
        )

    # def build_rope_cache(self) -> tg.Tensor:
    #     seq_idx = tg.Tensor.arange(self.max_seq_len)
    #     idx_theta = tg.Tensor.einsum(
    #         "i, j -> ij", seq_idx, self.theta).float()
    #     self.cache = tg.Tensor.stack([
    #         tg.Tensor.cos(idx_theta),
    #         tg.Tensor.sin(idx_theta)
    #     ], dim=-1)

    def build_rope_cache(self, device) -> tg.Tensor:
        half = self.dim // 2
        self.theta = (1.0 / (self.base ** (
            tg.Tensor.arange(0, self.dim, 2, device=device).float() / self.dim
        )))[:half].float()
        seq_idx = tg.Tensor.arange(self.max_seq_len, device=device).float()
        idx_theta = tg.Tensor.einsum("s,d->sd", seq_idx, self.theta).float()
        self.cache = tg.Tensor.stack([
            idx_theta.cos(),
            idx_theta.sin()
        ], dim=-1)

    def apply_scaling(
        self,
        freqs: tg.Tensor
    ) -> tg.Tensor:
        """
        Applies frequency scaling (e.g., YaRN-like) to the input frequency vector.
        Returns a tensor of the same shape as `freqs`.
        """
        # guard defaults if optional values are None
        low = 1.0 if (self.low_freq_factor is None) else float(self.low_freq_factor)
        high = 1.0 if (self.high_freq_factor is None) else float(self.high_freq_factor)
        # if old_context_len is not provided, don't change frequencies with this term
        if self.old_context_len is None or self.scale_factor is None:
            m = 1.0
        else:
            m = float(self.old_context_len) / float(self.max_seq_len)
            m = m ** (1.0 / float(self.scale_factor))

        # make a copy to avoid in-place on shared tensors
        out = (freqs * m).contiguous()
        # split low/high frequency bands roughly in half of the freqs (i.e., D/4 boundary for D/2 freqs)
        split_idx = self.dim // 4
        if split_idx > 0:
            out[:split_idx] = out[:split_idx] * low
            out[split_idx:] = out[split_idx:] * high
        else:
            out = out * high
        return out

    def __call__(
        self,
        q: tg.Tensor, # [B, S, Hq, D]
        k: tg.Tensor, # [B, S, Hk, D]
        position_ids: Optional[tg.Tensor] = None
    ) -> Tuple[tg.Tensor, tg.Tensor]:
        # if self.cache is None:
        #     self.build_rope_cache(q.device)

        """
        Returns rotated (q,k). Handles position_ids as [S], [B,S], or None.
        """
        B, S, Hq, D = q.shape
        _, _, Hk, Dk = k.shape
        assert D % 2 == 0 and Dk == D, "q/k head dims must match and be even"

        # positions -> [S]
        if position_ids is None:
            pos = tg.Tensor.arange(S, device=q.device)
        else:
            pos = position_ids
            if pos.ndim == 2:  # [B,S] -> take row 0 (RoPE equal across batch)
                pos = pos[0]
            elif pos.ndim == 0:
                pos = pos.reshape(1)

        # reshape into complex pairs
        q_pairs = q.reshape(B, S, Hq, D // 2, 2).float()
        k_pairs = k.reshape(B, S, Hk, D // 2, 2).float()

        # select cos/sin for the provided positions and move to the correct device
        # self.freqs shape: [1, 2*max_seq_len, 1, D/2, 2]
        table = self.freqs.to(q.device)
        cs = table[:, pos.cast(tg.dtypes.default_int), :, :, :]  # [1, S, 1, D/2, 2]

        # split into cos/sin and apply rotation via complex multiply
        C, D_ = cs[..., 0:1], cs[..., 1:2]                      # [1, S, 1, D/2, 1]
        q_out = complex_multi(q_pairs, C, D_)                   # [B, S, Hq, D/2, 2]
        k_out = complex_multi(k_pairs, C, D_)                   # [B, S, Hk, D/2, 2]

        # flatten back to last dim and cast to original dtype
        return q_out.flatten(3).cast(q.dtype), k_out.flatten(3).cast(k.dtype)
