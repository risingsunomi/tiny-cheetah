"""
Helpers for LLM models.
"""
from pathlib import Path
from typing import Optional, Any
import json
import os, time

import safetensors
import transformers as hf_transformers
import tinygrad as tg

from tiny_cheetah.models.llm.model import Model
from tiny_cheetah.models.llm.shard import Shard
from tiny_cheetah.repos import RepoHuggingFace


# From https://github.com/tinygrad/tinygrad/blob/master/extra/models/llama.py#L204C3-L205C138
# permute for weights using huggingface
def permute(v: tg.Tensor, n_heads: int):
    return v.reshape(n_heads, 2, v.shape[0] // n_heads // 2, v.shape[1] if len(v.shape) > 1 else 1).transpose(1, 2).reshape(*v.shape[:2])

def apply_weight(
    model_weight_key: str,
    key: str,
    weight_map,
    model_path,
    weight_device,
    model_state_dict,
    model_config
):
    
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

    if "q_proj" in key:
        weight = permute(weight, model_config["num_heads"])
    elif "k_proj" in key:
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
        with safetensors.safe_open(str(model_files[0]), framework="numpy") as weight_data:
            for key in weight_data.keys():
                weight_map[key] = model_files[0].name

    model_state_dict = tg.nn.state.get_state_dict(model)
    prefix_check = list(weight_map.keys())[1].split(".")[0]
    if prefix_check in ["model", "base_model", "transformer", "gpt_neox", "blk"]:
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
            elif "lm_head.weight" in weight_map.keys() and key == "output.weight":
                apply_weight(
                    "lm_head.weight",
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
def sample(
    logits: tg.Tensor,
    temp: float=1.0,
    k: Optional[int]=0,
    p: Optional[float]=0.8,
    af: Optional[float] = 0.0,
    ap: Optional[float] = 0.0
):
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

def generate(
    model: Model,
    input_ids: tg.Tensor, # [B, S]
    attention_mask: tg.Tensor,
    tokenizer: Any,
    max_new_tokens: int = 2048,
    temp: float=1.0,
    top_k: Optional[int] = 0,
    top_p: Optional[float] = 0.8,
    alpha_f: Optional[float] = 0.0,
    alpha_p: Optional[float] = 0.0,
    curr_pos: int = 0,
    verbose: bool = False
) -> list:
    # select device
    if os.getenv("TC_DEVICE") is not None:
        device = os.getenv("TC_DEVICE")
    else:
        available_devices = tg.Device.DEFAULT.split(",")    
        if "METAL" in available_devices:
            device = "METAL"
        elif "AMD" in available_devices:
            device = "AMD"
        elif "CUDA" in available_devices:
            device = "CUDA"
        else:
            print(f"Using default CPU device")
            device = "CPU"

    input_ids = input_ids.to(device)
    attention_mask = attention_mask.to(device)
    out_tokens = []

    position_ids = ((attention_mask.cumsum(axis=1) - 1) * attention_mask).to(device) # [B, S]

    print("Generating")
    print("_________________________________\n\n")
    if verbose:
        print(f"input_ids: {input_ids.tolist()}\nposition_ids: {position_ids.tolist()}\nattention_mask: {attention_mask.tolist()}")

    # get logits
    logits = model(
        input_ids,
        attention_mask=attention_mask,
        position_ids=position_ids
    ) # [B, S, V]

    next_logit = logits[:, -1, :].flatten() # [B, V]
    tok = sample(next_logit, temp=temp, k=top_k, p=top_p, af=alpha_f, ap=alpha_p)
    tok = tok.item()
    out_tokens.append(tok)
    # toks/sec timer
    t0 = time.time()
    generated = 1
    curr_pos += 1

    eos_hit = False

    # first token sampled; appended to out_tokens above

    limit = max_new_tokens - 1 if max_new_tokens > 0 else None

    while True:
        if tok == tokenizer.eos_token_id:
            elapsed = time.time() - t0
            tok_s = generated / elapsed if elapsed > 0 else float("inf")
            print(f"[decode] {generated} tokens in {elapsed:.3f}s  ->  {tok_s:.2f} tok/s")
            eos_hit = True
            break

        if limit is not None and generated >= limit:
            break

        generated += 1

        next_tok = tg.Tensor([[tok]], device=device)  # [B, 1]
        # grow attention mask and use absolute position for the new token
        attention_mask = attention_mask.cat(
            tg.Tensor.ones((attention_mask.shape[0], 1), device=device), dim=1
        )
        position_ids = tg.Tensor([curr_pos], device=device)

        if verbose:
            print(
                f"next_tok: {[next_tok.item()]}\n position_ids: {position_ids.tolist()}\n attention_mask_len: {attention_mask.shape[1]}"
            )

        logits = model(
            next_tok,
            attention_mask=attention_mask,
            position_ids=position_ids
        )
        next_logit = logits[:, -1, :].flatten()  # [B, V]
        tok = sample(next_logit, temp=temp, k=top_k, p=top_p, af=alpha_f, ap=alpha_p)
        tok = tok.item()
        out_tokens.append(tok)
        curr_pos += 1

    if not eos_hit:
        elapsed = time.time() - t0
        tok_s = generated / elapsed if elapsed > 0 else float("inf")
        if verbose:
            print(f"[decode] {generated} tokens in {elapsed:.3f}s  ->  {tok_s:.2f} tok/s (no EOS)")

    return out_tokens
