import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
from safetensors.numpy import save_file

try:
    import torch
except ModuleNotFoundError:
    torch = None

if torch is not None:
    from tiny_cheetah.models.llm.torch.helpers import load_safetensors
    from tiny_cheetah.models.llm.torch.helpers import permute as helpers_permute


if torch is not None:
    class _DummyQProjModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.q_proj = torch.nn.Linear(4, 8, bias=False)


    class _DummyExperts(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.gate_up_proj_blocks = torch.nn.Parameter(
                torch.zeros((2, 8, 1, 16), dtype=torch.uint8),
                requires_grad=False,
            )


    class _DummyMlp(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.router = torch.nn.Linear(4, 2, bias=True)
            self.experts = _DummyExperts()


    class _DummyLayer(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.mlp = _DummyMlp()


    class _DummyMoEModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.layers = torch.nn.ModuleList([_DummyLayer()])


def _write_model_index(model_dir: Path, weight_map: dict[str, str]) -> None:
    index_payload = {
        "metadata": {"total_size": 0},
        "weight_map": weight_map,
    }
    (model_dir / "model.safetensors.index.json").write_text(json.dumps(index_payload))


@unittest.skipIf(torch is None, "torch is not installed")
class TestHelpersLoader(unittest.TestCase):
    def test_load_safetensors_applies_qproj_permute_for_non_gpt_oss(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            model_dir = Path(temp_dir)
            weight_file = "model.safetensors"
            key = "model.q_proj.weight"

            raw_weight = np.arange(32, dtype=np.float32).reshape(8, 4)
            save_file({key: raw_weight}, str(model_dir / weight_file))
            _write_model_index(model_dir, {key: weight_file})

            model = _DummyQProjModel()
            load_safetensors(
                model,
                model_dir,
                model_config={"num_heads": 2, "num_kv_heads": 1, "model_type": "llama"},
                weight_device="cpu",
                use_tied=False,
            )

            expected = helpers_permute(torch.from_numpy(raw_weight), 2).to(dtype=model.q_proj.weight.dtype)
            torch.testing.assert_close(model.q_proj.weight.detach().cpu(), expected)

    def test_load_safetensors_skips_qproj_permute_for_gpt_oss(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            model_dir = Path(temp_dir)
            weight_file = "model.safetensors"
            key = "model.q_proj.weight"

            raw_weight = np.arange(32, dtype=np.float32).reshape(8, 4)
            save_file({key: raw_weight}, str(model_dir / weight_file))
            _write_model_index(model_dir, {key: weight_file})

            model = _DummyQProjModel()
            load_safetensors(
                model,
                model_dir,
                model_config={"num_heads": 2, "num_kv_heads": 1, "model_type": "gpt_oss"},
                weight_device="cpu",
                use_tied=False,
            )

            expected = torch.from_numpy(raw_weight).to(dtype=model.q_proj.weight.dtype)
            torch.testing.assert_close(model.q_proj.weight.detach().cpu(), expected)

    def test_load_safetensors_loads_gpt_oss_moe_expert_tensor(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            model_dir = Path(temp_dir)
            weight_file = "model.safetensors"
            key = "model.layers.0.mlp.experts.gate_up_proj_blocks"

            raw_weight = np.full((2, 8, 1, 16), 0x22, dtype=np.uint8)
            save_file({key: raw_weight}, str(model_dir / weight_file))
            _write_model_index(model_dir, {key: weight_file})

            model = _DummyMoEModel()
            load_safetensors(
                model,
                model_dir,
                model_config={"num_heads": 2, "num_kv_heads": 1, "model_type": "gpt_oss"},
                weight_device="cpu",
                use_tied=False,
            )

            expected = torch.from_numpy(raw_weight)
            self.assertTrue(
                torch.equal(
                    model.layers[0].mlp.experts.gate_up_proj_blocks.detach().cpu(),
                    expected,
                )
            )

    def test_load_safetensors_without_index_scans_all_shards(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            model_dir = Path(temp_dir)
            key = "model.q_proj.weight"
            raw_weight = np.arange(32, dtype=np.float32).reshape(8, 4)

            save_file({"model.embed_tokens.weight": np.zeros((4, 4), dtype=np.float32)}, str(model_dir / "a.safetensors"))
            save_file({key: raw_weight}, str(model_dir / "b.safetensors"))

            model = _DummyQProjModel()
            load_safetensors(
                model,
                model_dir,
                model_config={"num_heads": 2, "num_kv_heads": 1, "model_type": "gpt_oss"},
                weight_device="cpu",
                use_tied=False,
            )

            expected = torch.from_numpy(raw_weight).to(dtype=model.q_proj.weight.dtype)
            torch.testing.assert_close(model.q_proj.weight.detach().cpu(), expected)

    def test_load_safetensors_with_index_infers_model_prefix_even_if_first_key_is_lm_head(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            model_dir = Path(temp_dir)
            weight_file = "model.safetensors"
            qproj_key = "model.q_proj.weight"
            raw_weight = np.arange(32, dtype=np.float32).reshape(8, 4)

            save_file(
                {
                    "lm_head.weight": np.zeros((4, 4), dtype=np.float32),
                    qproj_key: raw_weight,
                },
                str(model_dir / weight_file),
            )
            _write_model_index(
                model_dir,
                {
                    "lm_head.weight": weight_file,
                    qproj_key: weight_file,
                },
            )

            model = _DummyQProjModel()
            load_safetensors(
                model,
                model_dir,
                model_config={"num_heads": 2, "num_kv_heads": 1, "model_type": "gpt_oss"},
                weight_device="cpu",
                use_tied=False,
            )

            expected = torch.from_numpy(raw_weight).to(dtype=model.q_proj.weight.dtype)
            torch.testing.assert_close(model.q_proj.weight.detach().cpu(), expected)


if __name__ == "__main__":
    unittest.main()
