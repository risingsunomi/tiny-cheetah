from __future__ import annotations

import os
import torch
from torch import nn

from ...shard import Shard
from .transformer import TransformerBlock

class Model(nn.Module):
    def __init__(
        self,
        config: dict,
        shard: Shard,
        use_tied: bool = False,
    ):
        super().__init__()
        self.config = config
        self.shard = shard

        print(f"loading shard: {shard}")

        self.embed_tokens = nn.Embedding(
            num_embeddings=self.config["vocab_size"],
            embedding_dim=self.config["embed_dim"],
        ).to(os.getenv("TC_DEVICE", "cpu"))

        self.norm = nn.LayerNorm(
            self.config["embed_dim"],
            eps=self.config["norm_eps"]
        ).to(os.getenv("TC_DEVICE", "cpu"))
        self.layers = nn.ModuleList(
            [TransformerBlock(self.config) for _ in range(self.shard.start_layer, self.shard.end_layer)]
        )

        # output == lm_head
        self.output = nn.Linear(
            self.config["embed_dim"],
            self.config["vocab_size"]
        ).to(os.getenv("TC_DEVICE", "cpu"))
        if use_tied:
            self.output.weight = self.embed_tokens.weight

    def reset_kv_cache(self) -> None:
        for layer in self.layers:
            attn = getattr(layer, "self_attn", None)
            if attn is not None and getattr(attn, "kv_cache", None) is not None:
                attn.kv_cache.clear()

    def forward(
        self,
        x: torch.Tensor,
        position_ids: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
        hidden_state: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if hidden_state is None:
            x = self.embed_tokens(x.long())
        else:
            x = hidden_state

        for layer in self.layers:
            x = layer(x, attention_mask, position_ids)

        if self.shard.end_layer == self.shard.total_layers - 1:
            x = self.norm(x)
            x = self.output(x)

        return x
