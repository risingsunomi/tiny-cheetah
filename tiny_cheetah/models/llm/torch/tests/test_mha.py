import unittest

try:
    import torch
except ModuleNotFoundError:
    torch = None

if torch is not None:
    from tiny_cheetah.models.llm.torch.attention import MultiHeadAttention


@unittest.skipIf(torch is None, "torch is not installed")
class TestMultiHeadAttention(unittest.TestCase):
    def setUp(self):
        self.config = {
            "embed_dim": 32,
            "num_heads": 4,
            "num_kv_heads": 2,
            "head_dim": 8,
            "max_seq_len": 16,
            "attn_bias": False,
            "attn_dropout": 0.0,
            "rope_scaling": None,
            "rope_theta": 10000.0,
            "rope_scaling_factor": None,
            "rope_low_freq_factor": 0.0,
            "rope_high_freq_factor": 0.0,
            "rope_original_max_pos_embeddings": 0,
            "qk_norm": False,
        }

    def test_forward_shape(self):
        mha = MultiHeadAttention(config=self.config)
        x = torch.randn(3, 5, self.config["embed_dim"])
        out = mha(x)
        self.assertEqual(tuple(out.shape), (3, 5, self.config["embed_dim"]))

    def test_prefill_then_decode_step(self):
        mha = MultiHeadAttention(config=self.config, is_causal=True)
        x = torch.randn(2, 4, self.config["embed_dim"])
        mask = torch.ones((2, 4), dtype=torch.long)
        pos = torch.arange(4).unsqueeze(0).expand(2, 4)

        prefill_out = mha(x, attention_mask=mask, position_ids=pos)
        self.assertEqual(tuple(prefill_out.shape), (2, 4, self.config["embed_dim"]))

        x_step = torch.randn(2, 1, self.config["embed_dim"])
        mask_step = torch.ones((2, 5), dtype=torch.long)
        pos_step = torch.tensor([4], dtype=torch.long)

        decode_out = mha(x_step, attention_mask=mask_step, position_ids=pos_step)
        self.assertEqual(tuple(decode_out.shape), (2, 1, self.config["embed_dim"]))


if __name__ == "__main__":
    unittest.main()
