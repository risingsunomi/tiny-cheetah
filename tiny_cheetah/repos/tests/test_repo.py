import unittest

from ..repo_huggingface import RepoHuggingFace
from ..repo_custom import RepoCustom

class TestRepo(unittest.IsolatedAsyncioTestCasepytho):
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