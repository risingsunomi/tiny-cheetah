import unittest

from ..repo_huggingface import RepoHuggingFace

class TestRepoHuggingFace(unittest.TestCase):
    def test_download(self):
        repo = RepoHuggingFace("unsloth/Llama-3.2-1B-Instruct")
        model_path = repo.download()
        self.assertTrue(model_path.exists())