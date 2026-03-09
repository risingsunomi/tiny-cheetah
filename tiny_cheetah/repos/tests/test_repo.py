import json
from pathlib import Path
import tempfile
import unittest

from ..repo_huggingface import RepoHuggingFace
from ..repo_custom import RepoCustom

class TestRepo(unittest.IsolatedAsyncioTestCase):
    async def test_custom_download(self):
        repo = RepoCustom("unsloth/Llama-3.2-1B-Instruct")
        model_path, model_config, messages = await repo.download()
        self.assertTrue(model_path.exists())
        self.assertIsInstance(model_config, type(repo.model_config))
        self.assertIsInstance(messages, list)

    # def test_hf_download(self):
    #     repo = RepoHuggingFace("unsloth/Llama-3.2-1B-Instruct")
    #     model_path = repo.download()
    #     self.assertTrue(model_path.exists())


class TestRepoCustomCacheFiles(unittest.TestCase):
    def test_gpt_oss_cache_requires_chat_template(self):
        payload = {
            "model_type": "gpt_oss",
            "hidden_size": 32,
            "num_attention_heads": 4,
            "num_key_value_heads": 4,
            "max_position_embeddings": 64,
            "intermediate_size": 32,
            "num_hidden_layers": 2,
            "vocab_size": 128,
        }

        with tempfile.TemporaryDirectory() as tmp:
            repo = RepoCustom("openai/gpt-oss-20b", cache_root=Path(tmp))
            (repo.base_dir / "config.json").write_text(json.dumps(payload), encoding="utf-8")
            repo._load_configs()

            self.assertEqual(repo._missing_cached_files(), ["chat_template.jinja"])

            (repo.base_dir / "chat_template.jinja").write_text("{{ messages }}", encoding="utf-8")
            self.assertEqual(repo._missing_cached_files(), [])

    def test_repo_custom_can_load_torch_model_config(self):
        payload = {
            "model_type": "gpt_oss",
            "hidden_size": 32,
            "num_attention_heads": 4,
            "num_key_value_heads": 4,
            "max_position_embeddings": 64,
            "intermediate_size": 32,
            "num_hidden_layers": 2,
            "vocab_size": 128,
            "num_local_experts": 8,
            "num_experts_per_tok": 2,
        }

        with tempfile.TemporaryDirectory() as tmp:
            repo = RepoCustom("openai/gpt-oss-20b", cache_root=Path(tmp), backend="torch")
            (repo.base_dir / "config.json").write_text(json.dumps(payload), encoding="utf-8")
            repo._load_configs()

            self.assertTrue(repo.model_config.config.get("moe"))
            self.assertEqual(repo.model_config.config.get("num_local_experts"), 8)
            self.assertEqual(repo.model_config.config.get("num_experts_per_tok"), 2)
