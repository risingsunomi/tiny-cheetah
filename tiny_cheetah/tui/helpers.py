from __future__ import annotations
import os
import time
from typing import Any

import tinygrad as tg
from transformers import AutoTokenizer

from textual.widgets import RichLog

from tiny_cheetah.models.llm.model import Model
from tiny_cheetah.models.shard import Shard
from tiny_cheetah.models.llm.helpers import sample
from tiny_cheetah.models.llm.quantize import is_quantized_model_config
from tiny_cheetah.orchestration.model_engine import ModelEngine
from tiny_cheetah.logging_utils import get_logger
logger = get_logger(__name__)


def detect_quantization_mode(model_config: Any) -> tuple[bool, str]:
    if not isinstance(model_config, dict):
        return False, "standard"
    if not is_quantized_model_config(model_config):
        return False, "standard"

    quantization_config = model_config.get("quantization_config")
    if not isinstance(quantization_config, dict):
        return True, "quantized"

    quant_method = str(quantization_config.get("quant_method", "quantized")).lower()
    quant_bits = "4-bit" if quantization_config.get("load_in_4bit") or quantization_config.get("_load_in_4bit") else (
        "8-bit" if quantization_config.get("load_in_8bit") or quantization_config.get("_load_in_8bit") else ""
    )
    quant_type = str(quantization_config.get("bnb_4bit_quant_type", "")).lower()

    parts = [quant_method]
    if quant_bits:
        parts.append(quant_bits)
    if quant_type:
        parts.append(quant_type)
    return True, " ".join(parts)


def streaming_generate(
    model: Model,
    input_ids: tg.Tensor,
    attention_mask: tg.Tensor,
    tokenizer: AutoTokenizer,
    max_new_tokens: int = 512,
    temp: float = 1.0,
    top_k: int = 0,
    top_p: float = 0.8,
    alpha_f: float = 0.0,
    alpha_p: float = 0.0,
    verbose: bool = False
) -> tuple[list[int], float]:
    device = input_ids.device
    input_ids = input_ids.to(device)
    attention_mask = attention_mask.to(device)

    out_tokens: list[int] = []
    curr_pos = attention_mask.shape[1]
    generated = 0
    start_time = time.time()

    # initial prefill
    position_ids = ((attention_mask.cumsum(axis=1) - 1) * attention_mask).to(device)
    logits = model(input_ids, attention_mask=attention_mask, position_ids=position_ids)
    next_logit = logits[:, -1, :].flatten()
    tok = sample(next_logit, temp=temp, k=top_k, p=top_p, af=alpha_f, ap=alpha_p).item()
    out_tokens.append(tok)
    generated += 1
    curr_pos += 1

    eos_hit = tok == tokenizer.eos_token_id
    limit = max_new_tokens - 1 if max_new_tokens > 0 else None

    while not eos_hit:
        if limit is not None and generated >= limit:
            break

        next_tok = tg.Tensor([[tok]], device=device)
        attention_mask = attention_mask.cat(
            tg.Tensor.ones((attention_mask.shape[0], 1), device=device), dim=1
        )
        position_ids = tg.Tensor([curr_pos], device=device)

        logits = model(
            next_tok,
            attention_mask=attention_mask,
            position_ids=position_ids,
        )
        next_logit = logits[:, -1, :].flatten()
        tok = sample(next_logit, temp=temp, k=top_k, p=top_p, af=alpha_f, ap=alpha_p).item()
        out_tokens.append(tok)
        generated += 1
        curr_pos += 1

        if tok == tokenizer.eos_token_id:
            eos_hit = True

    elapsed = time.time() - start_time

    if verbose:
        tok_s = generated / elapsed if elapsed > 0 else float("inf")
        print(f"[stream] {generated} tokens in {elapsed:.3f}s -> {tok_s:.2f} tok/s")

    return out_tokens, elapsed


def streaming_generate_with_peers(
    peer_client: Any,
    model: Model,
    input_ids: tg.Tensor,
    attention_mask: tg.Tensor,
    tokenizer: AutoTokenizer,
    max_new_tokens: int = 512,
    temp: float = 1.0,
    top_k: int = 0,
    top_p: float = 0.8,
    alpha_f: float = 0.0,
    alpha_p: float = 0.0,
    verbose: bool = False,
) -> tuple[list[int], float]:
    peers = peer_client.get_peers(include_self=True) if peer_client is not None else []
    if not peers or len(peers) <= 1:
        return streaming_generate(
            model,
            input_ids,
            attention_mask,
            tokenizer,
            max_new_tokens,
            temp,
            top_k,
            top_p,
            alpha_f,
            alpha_p,
            verbose,
        )

    engine = _local_engine(model)
    _plan_peer_shards(engine, peer_client, model)

    input_list = _tensor_to_list(input_ids)
    mask_list = _tensor_to_list(attention_mask)
    position_list = [[0]]
    hidden_state_list = []
    
    out_tokens: list[int] = []
    start_time = time.time()
    failed = False

    for step in range(max_new_tokens):
        logger.debug(f"[step: {step} Starting distributed generation with {len(peers)} peers for up to {max_new_tokens} tokens")
        logger.debug(f"Initial input_ids: {input_list}, attention_mask: {mask_list}")
        input_ids = tg.Tensor(input_list)
        attention_mask = tg.Tensor(mask_list)
        otoken_data = engine.get_tokens(
            model,
            input_ids,
            attention_mask,
            tokenizer,
            temp=temp,
            top_k=top_k,
            top_p=top_p,
            alpha_f=alpha_f,
            alpha_p=alpha_p,
        )

        mask_list, position_list, hidden_state_list, end_token = _apply_token_data(
            engine,
            tokenizer,
            otoken_data,
            input_list,
            mask_list,
            position_list,
            hidden_state_list,
            out_tokens,
        )

        for peer in peers:
            payload = {
                "command": "generate_token",
                "payload": {
                    "input_ids": input_list,
                    "attention_mask": mask_list,
                    "position_ids": position_list,
                    "hidden_state": hidden_state_list, 
                    "temp": temp,
                    "top_k": top_k,
                    "top_p": top_p,
                    "alpha_f": alpha_f,
                    "alpha_p": alpha_p,
                    "shard": _peer_shard_payload(peer),
                },
            }
            try:
                logger.debug(f"Sending payload to peer {getattr(peer, 'peer_client_id', 'unknown')}: {payload}")
                host = getattr(peer, "address", "0.0.0.0")
                port = int(getattr(peer, "port", os.getenv("TC_TENSOR_PORT", 1045)))
                address = (host, port)
                resp = peer_client.send_payload(
                    payload,
                    expect_reply=True,
                    address=address,
                )
                otoken_data = engine.recv_tokens(resp, tokenizer)
                logger.debug(f"Received response from peer {getattr(peer, 'peer_client_id', 'unknown')}: {otoken_data}")
            except Exception as err:
                logger.error(f"Error communicating with peer {getattr(peer, 'peer_client_id', 'unknown')}: {err}")
                break

            mask_list, position_list, hidden_state_list, end_token = _apply_token_data(
                engine,
                tokenizer,
                otoken_data,
                input_list,
                mask_list,
                position_list,
                hidden_state_list,
                out_tokens,
            )
            
            if end_token:
                break

    elapsed = time.time() - start_time
    return out_tokens, elapsed


def _tensor_to_list(tensor: Any) -> list[list[Any]]:
    if tensor is None:
        return [[]]
    if isinstance(tensor, list):
        if not tensor:
            return [[]]
        if isinstance(tensor[0], list):
            return tensor
        return [tensor]
    data = tensor.numpy().tolist()
    if not data:
        return [[]]
    if isinstance(data[0], list):
        return data
    return [data]


def _extract_token(engine: ModelEngine, response: Any, tokenizer: AutoTokenizer) -> tuple[int | None, bool]:
    msg = engine.recv_tokens(response, tokenizer)
    token = msg.get("token")
    if token is None:
        return None, False
    end_token = bool(msg.get("end_token", token == tokenizer.eos_token_id))
    return int(token), end_token


def _plan_peer_shards(model_engine: ModelEngine, peer_client: Any, model: Model) -> None:
    if peer_client is None:
        return
    peers = peer_client.get_peers(include_self=True)
    if len(peers) <= 1:
        return
    total_layers = _infer_total_layers(model)
    if total_layers <= 0:
        return
    model_name = str(getattr(model, "model_name", "") or getattr(model, "name", "") or "model")
    model_engine.plan_shards(peers, model_name, total_layers)


def _apply_token_data(
    engine: ModelEngine,
    tokenizer: AutoTokenizer,
    otoken_data: dict,
    input_list: list[list[int]],
    mask_list: list[list[int]],
    position_list: list[list[int]],
    hidden_state_list: list[list[Any]],
    out_tokens: list[int],
) -> tuple[list[list[int]], list[list[int]], list[list[Any]], bool]:
    msg = engine.recv_tokens(otoken_data, tokenizer)
    attention_mask = msg.get("attention_mask")
    position_ids = msg.get("position_ids")
    hidden_state = msg.get("hidden_state")
    token = msg.get("token")

    if attention_mask is not None:
        mask_list = _tensor_to_list(attention_mask)
    if position_ids is not None:
        position_list = _tensor_to_list(position_ids)

    if token is not None:
        tok = int(token)
        out_tokens.append(tok)
        input_list[0].append(tok)
        if mask_list:
            mask_list[0].append(1)
        else:
            mask_list = [[1]]
        hidden_state_list = []
    elif hidden_state is not None:
        hidden_state_list = _tensor_to_list(hidden_state)

    end_token = bool(msg.get("end_token", False))
    return mask_list, position_list, hidden_state_list, end_token


def _infer_total_layers(model: Model) -> int:
    config = getattr(model, "config", {}) or {}
    num_layers = config.get("num_layers")
    if num_layers is None:
        try:
            num_layers = len(getattr(model, "layers", []))
        except Exception:
            num_layers = 0
    try:
        num_layers_int = int(num_layers) if num_layers else 0
    except (TypeError, ValueError):
        num_layers_int = 0
    if num_layers_int <= 0:
        return 0
    return num_layers_int + 1


def _local_engine(model: Model) -> ModelEngine:
    shard = getattr(model, "shard", None)
    if shard is not None:
        return ModelEngine(shard=shard)
    total_layers = _infer_total_layers(model)
    if total_layers <= 0:
        return ModelEngine()
    return ModelEngine(shard=Shard("local", 0, total_layers - 1, total_layers))


def _peer_shard_payload(peer: Any) -> dict[str, int | str]:
    shard = getattr(peer, "shard", None)
    if shard is None:
        return {}
    try:
        return {
            "model_name": shard.model_name,
            "start_layer": int(shard.start_layer),
            "end_layer": int(shard.end_layer),
            "total_layers": int(shard.total_layers),
        }
    except Exception:
        return {}
