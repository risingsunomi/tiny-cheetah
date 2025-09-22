"""
Loading weights into model
"""
from pathlib import Path
import json

import safetensors
import tinygrad as tg

from model import Model

def load_safetensors(model: Model, model_path: Path):
    model_files = list(model_path.glob("*.safetensors"))
    weight_map = {}

    # get weight map if there is one
    weight_map = model_path / "model.safetensors.index.json"
    if weight_map.exists():
        with weight_map.open("r") as f:
            safe_index = json.load(f)
            weight_map = safe_index["weight_map"]
    else:
        # get weightmap from safetensor
        # this is usually when the model only has one weight
        weight_data = safetensors.safe_open(str(model_files[0]), framework="numpy")
        for key in weight_data.keys():
            weight_map[key] = model_files[0].name

    model_state_dict = tg.nn.state.get_state_dict(model)
    prefix_check = weight_map.keys()[0].split(".")[0]
    if prefix_check in ["model", "base_model", "transformer", "gpt_neox"]:
        prefix = prefix_check + "."
    else:
        prefix = ""
    
    for key in model_state_dict.keys():
        if prefix + key not in weight_map:
            print(f"WARNING: {prefix + key} not in weight map")
            continue
        
        weight_file = model_path / weight_map[prefix + key]
        weights = safetensors.safe_open(str(weight_file), framework="numpy")
        if prefix + key not in weights:
            print(f"WARNING: {prefix + key} not in {weight_file}, available keys: {list(weights.keys())}")
            continue
        
        weight = weights.get_tensor(prefix + key)
        param = model_state_dict[key]
        if param.shape != weight.shape:
            print(f"WARNING: {key} shape mismatch, model: {param.shape}, weight: {weight.shape}")
            continue
        
        param.assign(tg.Tensor(weight))
        print(f"Loaded {key} from {weight_file}")
