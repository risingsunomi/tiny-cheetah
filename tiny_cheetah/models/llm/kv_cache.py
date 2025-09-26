from typing import Tuple
import tinygrad as tg

class KVCache:
    def __init__(
        self,
        max_cache_len: int,
        batch_size,
        num_kv_heads,
        max_seq_len,
        head_dim
    ):
        self.max_cache_len = max_cache_len
        self.cache_k = tg.Tensor.zeros((
            batch_size, num_kv_heads, max_seq_len, head_dim
        )).contiguous().realize()
        self.cache_v = tg.Tensor.zeros((
            batch_size, num_kv_heads, max_seq_len, head_dim
        )).contiguous().realize()
        self.cache_pos = 0

    def reset(self):
        self.cache_k = None
        self.cache_v = None
        self.cache_pos = 0

    def update(
        self,
        k: tg.Tensor,
        v: tg.Tensor
    ) -> Tuple[tg.Tensor, tg.Tensor]:
        batch_size, _, seq_len, _ = k.shape
        if batch_size > self.cache_k.shape[0]:
            raise ValueError(f"Batch size {batch_size} exceeds cache size {self.cache_k.shape[0]}")
        assert (self.cache_pos + seq_len) <= self.max_cache_len, \
            f"Input seq len {seq_len} exceeds max cache len {self.max_cache_len}"
        
        # k_save = self.cache_k[:, :, self.cache_pos[:seq_len]]
        # k_save = k_save.contiguous()
        # v_save = self.cache_v[:, :, self.cache_pos[:seq_len]]
        # v_save = v_save.contiguous()
        sl = slice(self.cache_pos, self.cache_pos + seq_len)
        self.cache_k[:, :, sl, :] = k
        self.cache_v[:, :, sl, :] = v
        self.cache_pos += seq_len

    def serialize(self) -> dict:
        return {
            "max_len": self.max_len,
            "cache_k": self.cache_k.numpy(),
            "cache_v": self.cache_v.numpy()
        }