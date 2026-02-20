import unittest

try:
    import torch
except ModuleNotFoundError:
    torch = None

if torch is not None:
    from tiny_cheetah.models.llm.torch.mlp import MLP


@unittest.skipIf(torch is None, "torch is not installed")
class TestMLP(unittest.TestCase):
    def setUp(self):
        self.mlp = MLP(embed_dim=128, hidden_dim=256, activation="relu")

    def test_forward(self):
        input_tensor = torch.randn(1, 128)
        output = self.mlp(input_tensor)
        self.assertEqual(tuple(output.shape), (1, 128))


if __name__ == "__main__":
    unittest.main()
