"""
Helpers for LLM models.
"""
from pathlib import Path
from typing import Optional, Any
import json

import safetensors
import tinygrad as tg

from tiny_cheetah.models.llm.model import Model


# From https://github.com/tinygrad/tinygrad/blob/master/extra/models/llama.py#L204C3-L205C138
# permute for weights using huggingface
def permute(v: tg.Tensor, n_heads: int):
    return v.reshape(n_heads, 2, v.shape[0] // n_heads // 2, v.shape[1] if len(v.shape) > 1 else 1).transpose(1, 2).reshape(*v.shape[:2])

def apply_weight(model_weight_key: str, key: str, weight_map, model_path, weight_device, model_state_dict, model_config):
    weight_file = model_path / weight_map[model_weight_key]
    weights = tg.nn.state.safe_load(str(weight_file))
    weight = weights.get(model_weight_key)

    if weight is None:
        # print(f"!!! WARNING: {model_weight_key} not found in {weight_file}")
        return

    weight = weight.to(weight_device)

    if tg.dtypes.bfloat16:
        # bfloat16 fix for tinygrad. Need to research reasoning
        # From https://github.com/tinygrad/tinygrad/blob/master/extra/models/llama.py#L251
        weight = weight.cast(tg.dtypes.float32).cast(tg.dtypes.float16)
    
    if model_state_dict[key].shape != weight.shape:
        return

    if "q_proj" in key or "q_norm" in key:
        weight = permute(weight, model_config["num_heads"])
    elif "k_proj" in key or "k_norm" in key:
        weight = permute(weight, model_config["num_kv_heads"])
    
    model_state_dict[key] = weight
    
# Load safetensor weights into model
def load_safetensors(
        model: Model,
        model_path: Path,
        model_config: dict,
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
        if model_weight_key not in weight_map.keys():
            if use_tied and key == "output.weight":
                embed_weight_key = "model.embed_tokens.weight"
                apply_weight(
                    embed_weight_key,
                    key,
                    weight_map,
                    model_path,
                    weight_device,
                    model_state_dict,
                    model_config
                )

                continue
            
            continue
        
        apply_weight(
            model_weight_key,
            key,
            weight_map,
            model_path,
            weight_device,
            model_state_dict,
            model_config
        )
    
    tg.nn.state.load_state_dict(model, model_state_dict)

"""
LLM sampling
"""

# From https://github.com/tinygrad/tinygrad/blob/master/extra/models/llama.py#L119
# standard openai sampling
def sample(logits: tg.Tensor, temp: float=1.0, k: Optional[int]=None, p: Optional[float]=None, af: Optional[float] = None, ap: Optional[float] = None):
    assert logits.ndim == 1, "only works on 1d tensors"
    assert 0 <= p <= 1, "p must be between 0 and 1"
    assert 0 <= k <= logits.numel(), "k must be between 0 and numel"

    # if temperature is very low just use argmax
    if temp < 1e-6: return logits.argmax()

    logits = logits.to(tg.Device.DEFAULT)

    # alpha sampling
    if af or ap:
        if not hasattr(sample, "alpha_counter"):
            setattr(sample, "alpha_counter", tg.Tensor.zeros_like(logits, dtype=tg.dtypes.int32).contiguous())
        logits = logits - (sample.alpha_counter * af + (sample.alpha_counter > 0) * ap)

    # replace NaNs with -inf
    logits = (logits != logits).where(-float("inf"), logits)

    # softmax
    t = (logits / temp).softmax()

    counter, counter2 = tg.Tensor.arange(t.numel(), device=logits.device).contiguous(), tg.Tensor.arange(t.numel() - 1, -1, -1, device=logits.device).contiguous()
    # top k
    if k:
        output, output_indices = tg.Tensor.zeros(k, device=logits.device).contiguous(), tg.Tensor.zeros(k, device=logits.device, dtype=tg.dtypes.int32).contiguous()
        for i in range(k):
            t_argmax = (t.numel() - ((t == (t_max := t.max())) * counter2).max() - 1).cast(tg.dtypes.default_int)
            output = output + t_max.unsqueeze(0).pad(((i, k - i - 1),))
            output_indices = output_indices + t_argmax.unsqueeze(0).pad(((i, k - i - 1),))
            t = (counter == t_argmax).where(0, t)

        # approximate top p
        # because we are already limited to top k elements we can do top p "without sorting"
        output_cumsum = output[::-1].cumsum()[::-1] + t.sum()
        output = (output_cumsum >= (1 - p)) * output
        output_indices = (output_cumsum >= (1 - p)) * output_indices

        # sample
        output_idx = output.multinomial()
        output_token = output_indices[output_idx]
    else:
        output_token = t.multinomial()

    # increase alpha counter
    if af or ap:
        sample.alpha_counter = (counter == output_token).where(sample.alpha_counter + 1, sample.alpha_counter)

    return output_token

