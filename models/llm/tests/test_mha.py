import unittest

import tinygrad as tg
from ..attention import MultiHeadAttention

class TestMultiHeadAttention(unittest.TestCase):
    def setUp(self):
        # Match attention.MultiHeadAttention signature
        # embed_dim must equal num_heads * head_dim
        self.embed_dim = 4
        self.num_heads = 2
        self.num_kv_heads = 2
        self.head_dim = 2
        self.max_seq_len = 8
        self.mha = MultiHeadAttention(
            embed_dim=self.embed_dim,
            num_heads=self.num_heads,
            num_kv_heads=self.num_kv_heads,
            head_dim=self.head_dim,
            max_seq_len=self.max_seq_len,
            use_rope=False,
        )

    def test_forward_shape(self):
        # Batch size 3, sequence length 5, input dim 4
        x = tg.Tensor.randn(3, 5, self.embed_dim)
        out = self.mha(x)
        # Output should have shape (3, 5, 4)
        self.assertEqual(out.shape, (3, 5, self.embed_dim))

    # def test_forward_consistency(self):
    #     x = tg.Tensor.ones(2, 3, self.embed_dim)
    #     out1 = self.mha(x)
    #     out2 = self.mha(x)
    #     # For deterministic weights, outputs should be the same
    #     diff = (out1 - out2).abs().max().item()
    #     self.assertLessEqual(diff, 1e-6)

    # def test_invalid_input_dim(self):
    #     x = tg.Tensor.randn(2, 3, self.embed_dim + 1)  # Wrong input dim
    #     with self.assertRaises(Exception):
    #         _ = self.mha(x)
