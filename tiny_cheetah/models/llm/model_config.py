
from pathlib import Path
import json
import os

import tinygrad as tg
import json
import os

import tinygrad as tg
from tinygrad import dtypes

class ModelConfig:
    def __init__(self):
        self.model_config = {}
        self.qk_norm_models = ["qwen3", "qwen2"] # TODO: replace with config.json

    def load(self, config_file: Path):
        with open(config_file, "r") as f:
            base_config = json.loads(f.read())

        HF_PRECISION_STR_TO_DTYPE = {
            "float16": dtypes.float16,
            "bfloat16": dtypes.bfloat16,
            "float32": dtypes.float32,
            "float64": dtypes.float64,
            "int8": dtypes.int8,
            "int16": dtypes.int16,
            "int32": dtypes.int32,
            "int64": dtypes.int64,
            "uint8": dtypes.uint8,
            "uint16": dtypes.uint16,
            "uint32": dtypes.uint32,
            "uint64": dtypes.uint64,
            "bool": dtypes.bool,
        }

        self.model_config = {
            "architectures": base_config.get("architectures", []),
            "embed_dim": base_config["hidden_size"],
            "num_heads": base_config["num_attention_heads"],
            "head_dim": base_config.get(
                "head_dim",
                base_config["hidden_size"] // base_config["num_attention_heads"],
            ),  # Assuming embed_dim = hidden_size
            "num_kv_heads": base_config["num_key_value_heads"],
            "max_seq_len": base_config["max_position_embeddings"],
            "intermediate_dim": base_config["intermediate_size"],
            "attn_dropout": base_config.get("attention_dropout", 0.0),
            "norm_eps": base_config["rms_norm_eps"],
            "rope_scaling": base_config.get("rope_scaling"),
            "rope_theta": base_config["rope_theta"],
            "vocab_size": base_config["vocab_size"],
            "num_layers": base_config["num_hidden_layers"],
            "attn_bias": base_config.get("attention_bias", False),
            "hidden_act": base_config.get("hidden_act", "silu"),
            "tinygrad_dtype": HF_PRECISION_STR_TO_DTYPE.get(
                base_config.get("torch_dtype", "bfloat16"),
                dtypes.bfloat16
            ),
            "tie_word_embeddings": base_config.get("tie_word_embeddings", False),
            "model_type": base_config.get("model_type", "")
        }

        self.model_config["rope_scaling_factor"] = None
        self.model_config["rope_low_freq_factor"] = 0.0
        self.model_config["rope_high_freq_factor"] = 0.0
        self.model_config["rope_original_max_pos_embeddings"] = 0

        rope_scaling = self.model_config.get("rope_scaling")
        if rope_scaling is not None:
            self.model_config["rope_scaling_factor"] = rope_scaling.get("factor", rope_scaling.get("rope_factor", 0))
            self.model_config["rope_low_freq_factor"] = rope_scaling.get("low_freq_factor", rope_scaling.get("beta_fast", 0))
            self.model_config["rope_high_freq_factor"] = rope_scaling.get("high_freq_factor", rope_scaling.get("beta_slow", 0))
            self.model_config["rope_original_max_pos_embeddings"] = rope_scaling.get("original_max_position_embeddings", rope_scaling.get("original_max_pos", 0))
            self.model_config["rope_type"] = rope_scaling.get("rope_type", "llama3")

        # attention qk norm flag (default False if missing)
        attn_qk_norm = base_config.get("attn_qk_norm")
        if isinstance(attn_qk_norm, bool):
            self.model_config["qk_norm"] = attn_qk_norm
        elif isinstance(attn_qk_norm, str):
            # treat string variants like "rmsnorm" as enabled
            self.model_config["qk_norm"] = len(attn_qk_norm) > 0
        elif self.model_config.get("model_type",  "") in self.qk_norm_models:
            self.model_config["qk_norm"] = True
        else:
            self.model_config["qk_norm"] = False

        custom_seq = os.getenv("XOT_MAX_SEQ_LEN")
        if custom_seq is not None:
            self.model_config["max_seq_len"] = custom_seq

    def load_generation_config(self, gen_config_file: Path):
        with open(gen_config_file, "r") as f:
            gen_config = json.loads(f.read())
        
        self.model_config["temperature"] = gen_config.get("temperature") if os.getenv("TC_TEMP") is None else float(os.getenv("TC_TEMP"))
        self.model_config["top_k"] = gen_config.get("top_k") if os.getenv("TC_TOP_K") is None else int(os.getenv("TC_TOP_K"))
        self.model_config["top_p"] = gen_config.get("top_p") if os.getenv("TC_TOP_P") is None else float(os.getenv("TC_TOP_P"))
        self.model_config["eos_token_id"] = gen_config.get("eos_token_id")
        self.model_config["pad_token_id"] = gen_config.get("pad_token_id")
        self.model_config["bos_token_id"] = gen_config.get("bos_token_id")

    def __getitem__(self, key):
        return self.model_config.get(key, None)
    
    def __str__(self):
        return str(self.model_config)
