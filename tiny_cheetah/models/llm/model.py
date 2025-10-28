import tinygrad as tg

from .shard import Shard
from .transformer import TransformerBlock

class Model:
    def __init__(
        self,
        config: dict,
        shard: Shard,
        use_tied: bool = False
    ):
        
        self.config = config
        self.shard = shard

        print(f"loading shard: {shard}")

        self.embed_tokens = tg.nn.Embedding(
            vocab_size=self.config["vocab_size"],
            embed_size=self.config["embed_dim"]
        )

        self.norm = tg.nn.LayerNorm(self.config["embed_dim"], eps=self.config["norm_eps"])

        self.layers = [None for _ in range(self.shard.start_layer, self.shard.end_layer)]

        for i in range(self.shard.start_layer, self.shard.end_layer):
            self.layers[i] = TransformerBlock(self.config)

        # output == lm_head
        self.output = tg.nn.Linear(self.config["embed_dim"], self.config["vocab_size"])
        if use_tied:
            self.output.weight = self.embed_tokens.weight

    def reset_kv_cache(self) -> None:
        for layer in self.layers:
            if layer is None:
                continue
            attn = getattr(layer, "self_attn", None)
            if attn is not None and getattr(attn, "kv_cache", None) is not None:
                attn.kv_cache.clear()
    
    def __call__(
        self,
        x,
        position_ids: tg.Tensor | None=None,
        attention_mask: tg.Tensor | None=None
    ):
        x = self.embed_tokens(x)
        for i in range(self.shard.start_layer, self.shard.end_layer):
            x = self.layers[i](x, attention_mask, position_ids)

        if self.shard.end_layer == self.shard.total_layers-1:
            x = self.norm(x)
            x = self.output(x)
        
        return x
