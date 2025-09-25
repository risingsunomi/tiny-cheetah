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
        print(f"model config: {self.config.model_config}")


        self.embed_tokens = tg.nn.Embedding(
            vocab_size=self.config["vocab_size"],
            embed_size=self.config["embed_dim"]
        )

        self.norm = tg.nn.LayerNorm(self.config["embed_dim"], eps=self.config["norm_eps"])

        self.layers = [None for _ in range(self.shard.start_layer, self.shard.end_layer)]

        for i in range(self.shard.start_layer, self.shard.end_layer):
            self.layers[i] = TransformerBlock(
                embed_dim=self.config["embed_dim"],
                hidden_dim=self.config["intermediate_dim"],
                num_heads=self.config["num_heads"],
                num_kv_heads=self.config["num_kv_heads"],
                head_dim=self.config["head_dim"],
                max_seq_len=self.config["max_seq_len"]
            )

        # output == lm_head
        self.output = tg.nn.Linear(self.config["embed_dim"], self.config["vocab_size"])
        if use_tied:
            self.output.weight = self.embed_tokens.weight


    def __call__(self, x):
        x = self.embed_tokens(x)
        x = self.norm(x)
        
        for i in range(self.shard.start_layer, self.shard.end_layer):
            x = self.layers[i](x)

        if not self.shard.end_layer == self.shard.total_layers-1:
            x = self.output(x)
        
        return x
