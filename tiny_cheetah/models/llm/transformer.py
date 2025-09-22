import tinygrad as tg

from .attention import MultiHeadAttention
from .mlp import MLP

class TransformerBlock:
    def __init__(
        self,
        embed_dim,
        hidden_dim,
        num_heads,
        num_kv_heads,
        head_dim,
        max_seq_len,
        hidden_act: str = "relu",
        attn_scale: any = False,
        mlp_scale: any = None
    ):
        self.input_layernorm = tg.nn.RMSNorm(embed_dim)
        self.mlp = MLP(embed_dim, hidden_dim, hidden_act)
        self.self_attn = MultiHeadAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            num_kv_heads=num_kv_heads,
            head_dim=head_dim,
            max_seq_len=max_seq_len
        )
        self.post_attention_layernorm = tg.nn.RMSNorm(embed_dim)

        self.attn_scale = attn_scale
        self.mlp_scale = mlp_scale

    def __call__(self, x, mask=None):
        h = self.input_layernorm(x)
        attn_out = self.self_attn(h, mask=mask)

        if self.attn_scale:
            x = x + self.attn_scale(attn_out)
        else:
            x = x + attn_out

        mlp_out = self.mlp(self.post_attention_layernorm(x))

        if self.mlp_scale:
            out = h + self.mlp_scale(mlp_out)
        else:
            out = h + mlp_out
        
        return out