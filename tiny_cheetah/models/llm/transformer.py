import tinygrad as tg

from .attention import MultiHeadAttention
from .mlp import MLP

class TransformerBlock:
    def __init__(
        self,
        config: dict
    ):
        self.input_layernorm = tg.nn.RMSNorm(config["embed_dim"])
        self.mlp = MLP(
            config["embed_dim"],
            config["intermediate_dim"],
            config["hidden_act"]
        )
        self.self_attn = MultiHeadAttention(
            config=config,
            is_causal=any("CausalLM" in arch for arch in config["architectures"])
        )
        self.post_attention_layernorm = tg.nn.RMSNorm(config["embed_dim"])

        self.attn_scale = config["attn_scale"]
        self.mlp_scale = config["mlp_scale"]

    def __call__(self, x, attention_mask, position_ids):
        h = self.input_layernorm(x)
        attn_out = self.self_attn(h, attention_mask, position_ids)

        if self.attn_scale:
            x = x + self.attn_scale(attn_out)
        else:
            x = x + attn_out

        mlp_out = self.mlp(self.post_attention_layernorm(x))

        if self.mlp_scale:
            out = x + self.mlp_scale(mlp_out)
        else:
            out = x + mlp_out
        
        return out