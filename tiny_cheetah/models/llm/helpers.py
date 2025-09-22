"""
Helpers for LLM models.
"""
from pathlib import Path
import json

import safetensors
import tinygrad as tg

from tiny_cheetah.models.llm.model import Model

def load_safetensors(
        model: Model,
        model_path: Path,
        weight_device: str = "CPU",
        use_tied: bool = False
    ) -> None:
    """
    Loading weights into model
    """
    model_files = list(model_path.glob("*.safetensors"))
    weight_map = {}

    # get weight map if there is one
    weight_map_json = model_path / "model.safetensors.index.json"
    if weight_map_json.exists():
        with weight_map_json.open("r") as f:
            safe_index = json.load(f)
            weight_map = safe_index["weight_map"]
    else:
        # get weightmap from safetensor
        # this is usually when the model only has one weight
        weight_data = safetensors.safe_open(str(model_files[0]), framework="numpy")
        for key in weight_data.keys():
            weight_map[key] = model_files[0].name

    model_state_dict = tg.nn.state.get_state_dict(model)
    prefix_check = next(iter(weight_map.keys())).split(".")[0]
    if prefix_check in ["model", "base_model", "transformer", "gpt_neox"]:
        prefix = prefix_check + "."
    else:
        prefix = ""
    
    for key in model_state_dict.keys():
        model_weight_key = prefix + key
        if use_tied and model_weight_key in ["model.output.weight", "model.output.bias"]:
            # print(f"!!! WARNING: tying weights for {model_weight_key}")
            continue

        if model_weight_key not in weight_map.keys():
            # print(f"!!! WARNING: {model_weight_key} not in weight map")
            continue

        weight_file = model_path / weight_map[model_weight_key]
        weights = tg.nn.state.safe_load(str(weight_file))
        weight = weights.get(model_weight_key)

        if weight is None:
            # print(f"!!! WARNING: {model_weight_key} not found in {weight_file}")
            continue
        weight = weight.to(weight_device)
        param = model_state_dict[key]
        
        if param.shape != weight.shape:
            # print(f"!!! WARNING: {key} shape mismatch, model: {param.shape}, weight: {weight.shape}")
            continue
        
        param.assign(weight)
