import json
import tempfile
import unittest
from pathlib import Path

try:
    import torch
except ModuleNotFoundError:
    torch = None

if torch is not None:
    from tiny_cheetah.models.llm.torch.model_config import ModelConfig


@unittest.skipIf(torch is None, "torch is not installed")
class TestModelConfig(unittest.TestCase):
    def test_load_gpt_oss_moe_fields(self):
        payload = {
            "model_type": "gpt_oss",
            "architectures": ["GptOssForCausalLM"],
            "hidden_size": 2880,
            "num_attention_heads": 64,
            "num_key_value_heads": 8,
            "head_dim": 64,
            "max_position_embeddings": 131072,
            "intermediate_size": 2880,
            "num_hidden_layers": 24,
            "vocab_size": 201088,
            "experts_per_token": 4,
            "num_local_experts": 32,
            "num_experts_per_tok": 4,
            "hidden_act": "silu",
            "rms_norm_eps": 1e-5,
            "rope_theta": 150000.0,
            "initial_context_length": 4096,
            "router_aux_loss_coef": 0.9,
            "output_router_logits": False,
            "use_cache": True,
            "torch_dtype": "bfloat16",
        }

        with tempfile.TemporaryDirectory() as tmp:
            cfg = Path(tmp) / "config.json"
            cfg.write_text(json.dumps(payload), encoding="utf-8")
            model_config = ModelConfig()
            model_config.load(cfg)

        c = model_config.config
        self.assertEqual(c["model_type"], "gpt_oss")
        self.assertTrue(c["moe"])
        self.assertEqual(c["num_local_experts"], 32)
        self.assertEqual(c["experts_per_token"], 4)
        self.assertEqual(c["num_experts_per_tok"], 4)
        self.assertEqual(c["initial_context_length"], 4096)
        self.assertEqual(c["router_aux_loss_coef"], 0.9)
        self.assertFalse(c["output_router_logits"])
        self.assertTrue(c["use_cache"])
        self.assertTrue(c["attn_bias"])
        self.assertTrue(c["mlp_bias"])

    def test_load_generation_config_keeps_missing_sampling_fields_unset(self):
        payload = {
            "model_type": "gpt_oss",
            "architectures": ["GptOssForCausalLM"],
            "hidden_size": 2880,
            "num_attention_heads": 64,
            "num_key_value_heads": 8,
            "max_position_embeddings": 131072,
            "intermediate_size": 2880,
            "num_hidden_layers": 24,
            "vocab_size": 201088,
        }
        gen_payload = {
            "bos_token_id": 199998,
            "eos_token_id": [200002, 199999, 200012],
            "pad_token_id": 199999,
        }

        with tempfile.TemporaryDirectory() as tmp:
            cfg = Path(tmp) / "config.json"
            gen_cfg = Path(tmp) / "generation_config.json"
            cfg.write_text(json.dumps(payload), encoding="utf-8")
            gen_cfg.write_text(json.dumps(gen_payload), encoding="utf-8")

            model_config = ModelConfig()
            model_config.load(cfg)
            model_config.load_generation_config(gen_cfg)

        c = model_config.config
        self.assertIsNone(c["temperature"])
        self.assertIsNone(c["top_k"])
        self.assertIsNone(c["top_p"])
        self.assertEqual(c["eos_token_id"], [200002, 199999, 200012])


if __name__ == "__main__":
    unittest.main()
