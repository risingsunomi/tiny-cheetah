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

    
