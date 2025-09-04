import unittest

import tinygrad
from models.llm.mlp import MLP

class TestMLP(unittest.TestCase):

    def setUp(self):
        self.mlp = MLP(embed_dim=128, hidden_dim=256, activation="relu")

    def test_forward(self):
        # Test the forward pass with a dummy input
        input_tensor = tinygrad.Tensor.randn(1, 128)
        output = self.mlp(input_tensor)
        print(f"Output shape: {output.shape}")
        print(f"Output tensor: {output}")
        self.assertEqual(output.shape, (1, 128))
