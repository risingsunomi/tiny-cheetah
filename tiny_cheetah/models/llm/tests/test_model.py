import unittest

import transformers as hf_transformers
import tinygrad as tg

from tiny_cheetah.models.llm.model import Model
from tiny_cheetah.models.llm.helpers import load_safetensors
from tiny_cheetah.models.llm.shard import Shard
from tiny_cheetah.repos import RepoHuggingFace

class TestModel(unittest.TestCase):
   def setUp(self):
       test_model = "unsloth/Llama-3.2-1B-Instruct"
       repo = RepoHuggingFace(test_model)
       self.model_path, self.model_config = repo.download()
       shard = Shard(
           test_model,
           start_layer=0,
           end_layer=self.model_config["num_layers"],
           total_layers=self.model_config["num_layers"],
       )
       self.model = Model(self.model_config, shard)

   def test_llama32_1B_load_weights_apple(self):
        try:
            load_safetensors(
                self.model,
                self.model_path,
                weight_device="METAL",
                use_tied=True
            )
        except Exception as e:
            self.fail(f"Loading weights failed: {e}")
        
        self.assertTrue(True, "Weights loaded successfully")