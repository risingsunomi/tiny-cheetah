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
from tiny_cheetah.orchestration.model_engine import ModelEngine


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
    
    out_tokens: list[int] = []
    start_time = time.time()
    failed = False

    for step in range(max_new_tokens):
        peer = peers[step % len(peers)]
        if getattr(peer, "peer_client_id", None) == getattr(peer_client, "peer_client_id", None):
            token, end_token = engine.get_tokens(
                engine,
                model,
                input_list,
                mask_list,
                tokenizer,
                temp=temp,
                top_k=top_k,
                top_p=top_p,
                alpha_f=alpha_f,
                alpha_p=alpha_p,
            )
        else:
            payload = {
                "command": "generate_token",
                "payload": {
                    "input_ids": input_list,
                    "attention_mask": mask_list,
                    "temp": temp,
                    "top_k": top_k,
                    "top_p": top_p,
                    "alpha_f": alpha_f,
                    "alpha_p": alpha_p,
                    "shard": _peer_shard_payload(peer),
                },
            }
            try:
                address = _peer_address(peer)
                response = peer_client.send_payload(
                    payload,
                    expect_reply=True,
                    address=address,
                )
            except Exception:
                failed = True
                break

            token, end_token = _extract_token(engine, response, tokenizer)
        if token is None:
            failed = True
            break

        out_tokens.append(token)
        input_list[0].append(token)
        mask_list[0].append(1)
        if end_token:
            break

    if failed and max_new_tokens > len(out_tokens):
        remaining = max_new_tokens - len(out_tokens)
        local_input = tg.Tensor(input_list)
        local_mask = tg.Tensor(mask_list)
        local_tokens, _ = streaming_generate(
            model,
            local_input,
            local_mask,
            tokenizer,
            remaining,
            temp,
            top_k,
            top_p,
            alpha_f,
            alpha_p,
            verbose,
        )
        out_tokens.extend(local_tokens)

    elapsed = time.time() - start_time
    return out_tokens, elapsed


def _tensor_to_list(tensor: tg.Tensor) -> list[list[int]]:
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


def _generate_next_token_local(
    engine: ModelEngine,
    model: Model,
    input_list: list[list[int]],
    mask_list: list[list[int]],
    tokenizer: AutoTokenizer,
    *,
    temp: float,
    top_k: int,
    top_p: float,
    alpha_f: float,
    alpha_p: float,
) -> tuple[int | None, bool]:
    input_tensor = tg.Tensor(input_list)
    mask_tensor = tg.Tensor(mask_list)
    payload = engine.get_tokens(
        model,
        input_tensor,
        mask_tensor,
        tokenizer,
        temp=temp,
        top_k=top_k,
        top_p=top_p,
        alpha_f=alpha_f,
        alpha_p=alpha_p,
    )
    msg = engine.recv_tokens(payload, tokenizer)
    token = msg.get("token")
    if token is None:
        return None, False
    end_token = bool(msg.get("end_token", token == tokenizer.eos_token_id))
    return int(token), end_token


def _peer_address(peer: Any) -> tuple[str, int]:
    host = getattr(peer, "address", "0.0.0.0")
    port = int(getattr(peer, "port", os.getenv("TC_TENSOR_PORT", 1045)))
    return host, port


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
