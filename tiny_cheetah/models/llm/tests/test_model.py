import unittest

import transformers as hf_transformers
import tinygrad as tg

from tiny_cheetah.models.llm.model import Model
from tiny_cheetah.models.llm.helpers import load_safetensors, sample
from tiny_cheetah.models.llm.shard import Shard
from tiny_cheetah.repos import RepoHuggingFace

TOP_K = 0
TEMP = 0.8
TOP_P = 0.0

class TestModel(unittest.TestCase):
    def setUp(self):
       self.test_model = "unsloth/Llama-3.2-1B-Instruct"
       repo = RepoHuggingFace(self.test_model)
       self.model_path, self.model_config = repo.download()
       shard = Shard(
           self.test_model,
           start_layer=0,
           end_layer=self.model_config["num_layers"],
           total_layers=self.model_config["num_layers"],
       )
       self.model = Model(self.model_config, shard)

    # def test_llama32_1B_load_weights_apple(self):
    #     try:
    #         load_safetensors(
    #             self.model,
    #             self.model_path,
    #             weight_device="METAL",
    #             use_tied=True
    #         )
    #     except Exception as e:
    #         self.fail(f"Loading weights failed: {e}")
        
    #     self.assertTrue(True, "Weights loaded successfully")

    def test_llama32_1B_forward_apple(self):
        try:
            load_safetensors(
                self.model,
                self.model_path,
                self.model_config,
                weight_device="METAL",
                use_tied=True
            )
        except Exception as e:
            self.fail(f"Loading weights failed: {e}")

        prompt = "What is the capital of France?"

        tokenizer = hf_transformers.AutoTokenizer.from_pretrained(
            self.test_model,
            local_files_only=True
        )

        inputs = tokenizer.encode(prompt, return_tensors="np")
        input_ids = tg.Tensor(inputs, device="METAL")
        print(f"input_ids: {input_ids}")
        try:
            logits = self.model(input_ids)
            print(f"logits {logits}")
            self.assertIsNotNone(logits, "Model logits should not be None")
            self.assertTrue(hasattr(logits, "shape"), "logits should have a 'shape' attribute")
            self.assertEqual(logits.shape[0], input_ids.shape[0], "Batch size of logits should match input")
        except Exception as e:
            self.fail(f"Forward pass failed: {e}")

        out_tokens = sample(logits[:, -1, :].flatten(), temp=TEMP, k=TOP_K, p=TOP_P)
        out_decoded = tokenizer.decode(out_tokens.tolist())
        print(f"{out_decoded=}")