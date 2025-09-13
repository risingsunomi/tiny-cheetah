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

        self.tok_embeddings = tg.nn.Embedding(
            num_embeddings=self.config["vocab_size"],
            embedding_dim=self.config["embed_dim"]
        )

        self.norm = tg.nn.LayerNorm(self.config["embed_dim"], eps=self.config["norm_eps"])

        self.transformer_layers = [None for _ in range(self.shard.start_layer, self.shard.end_layer)]

        for i in range(self.shard.start_layer, self.shard.end_layer):
            self.transformer_layers[i] = TransformerBlock(
                embed_dim=self.config["embed_dim"],
                hidden_dim=self.config["hidden_dim"],
                num_heads=self.config["num_heads"],
                num_kv_heads=self.config["num_kv_heads"],
                head_dim=self.config["head_dim"],
                max_seq_len=self.config["max_seq_len"]
            )

        # output == lm_head
        self.output = tg.nn.Linear(self.config["embed_dim"], self.config["vocab_size"])
        if use_tied:
            self.output.weight = self.tok_embeddings.weight


    def __call__(self, x):
        x = self.tok_embeddings(x)
        x = self.norm(x)
        
        for layer in range(self.shard.start_layer, self.shard.end_layer):
            x = layer(x)
        
        if not self.shard.end_layer == self.shard.n_layer-1:
            x = self.output(x)
        
        return x
