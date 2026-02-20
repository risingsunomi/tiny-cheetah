from __future__ import annotations

import json
import base64
import struct
from typing import Any, Dict, List, Sequence

import numpy as np
import tinygrad as tg

from tiny_cheetah.models.llm.backend import backend_helpers_module
from tiny_cheetah.models.shard import Shard


class ModelEngine:
    """Lightweight token generation and shard planning."""
    def __init__(
        self,
        shard: Shard | None = None,
    ) -> None:
        self.shard = shard or Shard("local", 0, 0, 0)

    def get_tokens(
        self,
        model: Any,
        input_ids: Any,
        attention_mask: Any,
        tokenizer: Any,
        hidden_state: Any | None = None,
        *,
        temp: float = 1.0,
        top_k: int = 0,
        top_p: float = 0.8,
        alpha_f: float = 0.0,
        alpha_p: float = 0.0,
    ) -> Dict[str, Any]:
        """Generate the next token and return a JSON-serializable payload."""
        device = input_ids.device
        curr_pos = attention_mask.shape[1] - 1

        if hidden_state is not None:
            prev_token = input_ids[:, -1].item()
            position_ids = tg.Tensor([curr_pos], device=device)
            model_output = model(
                None,
                attention_mask=attention_mask,
                position_ids=position_ids,
                hidden_state=hidden_state,
            )
        else:
            prev_token = input_ids[:, -1].item()
            next_tok = tg.Tensor([[prev_token]], device=device)
            attention_mask = attention_mask.cat(
                tg.Tensor.ones((attention_mask.shape[0], 1), device=device), dim=1
            )
            position_ids = tg.Tensor([curr_pos], device=device)
            model_output = model(next_tok, attention_mask=attention_mask, position_ids=position_ids)

        is_final = self.shard.end_layer == self.shard.total_layers - 1
        if not is_final:
            return {
                "hidden_state": _encode_tensor(model_output),
                "attention_mask": _encode_tensor(attention_mask),
                "position_ids": _encode_tensor(position_ids),
                "shard": _shard_payload(self.shard),
                "end_token": False,
            }

        next_logit = model_output[:, -1, :].flatten()
        tok = _sample_with_backend(next_logit, temp=temp, k=top_k, p=top_p, af=alpha_f, ap=alpha_p).item()
        end_token = bool(getattr(tokenizer, "eos_token_id", None) == tok)

        return {
            "token": _encode_token_tensor(tok),
            "attention_mask": _encode_tensor(attention_mask),
            "position_ids": _encode_tensor(position_ids),
            "shard": _shard_payload(self.shard),
            "end_token": end_token,
        }

    def recv_tokens(self, payload: Any, tokenizer: Any | None = None) -> Dict[str, Any]:
        """Normalize a token payload for chat/training consumers."""
        if isinstance(payload, (bytes, bytearray)):
            try:
                payload = json.loads(payload.decode("utf-8"))
            except Exception:
                return {}
        elif isinstance(payload, str):
            try:
                payload = json.loads(payload)
            except Exception:
                return {}

        if not isinstance(payload, dict):
            return {}

        msg = payload.get("payload", payload)
        if not isinstance(msg, dict):
            return {}

        token = msg.get("token")
        if token is None:
            token = _decode_tensor(msg.get("tensor"))
            if token is not None:
                msg["token"] = token

        hidden_state = _decode_tensor(msg.get("hidden_state"))
        if hidden_state is not None:
            msg["hidden_state"] = hidden_state

        attention_mask = _decode_tensor(msg.get("attention_mask"))
        if attention_mask is not None:
            msg["attention_mask"] = attention_mask

        position_ids = _decode_tensor(msg.get("position_ids"))
        if position_ids is not None:
            msg["position_ids"] = position_ids

        if "shard" not in msg:
            msg["shard"] = _shard_payload(self.shard)

        if tokenizer is not None and token is not None:
            eos_id = getattr(tokenizer, "eos_token_id", None)
            msg["end_token"] = bool(msg.get("end_token", False) or token == eos_id)
        else:
            msg.setdefault("end_token", False)

        return msg

    @staticmethod
    def plan_shards(peers: Sequence[Any], model_name: str, total_layers: int) -> List[Shard]:
        capacities = []
        for peer in peers:
            vram = _to_float(getattr(peer, "gpu_vram", 0.0))
            ram = _to_float(getattr(peer, "cpu_ram", 0.0))
            flops = getattr(peer, "gpu_flops", 0.0) if hasattr(peer, "gpu_flops") else 0.0
            capacity = max(vram, ram, flops, 1.0)
            capacities.append((peer, capacity))

        total_cap = sum(cap for _, cap in capacities) or 1.0
        shards: List[Shard] = []
        start = 0
        for peer, cap in capacities:
            fraction = cap / total_cap
            span = max(int(total_layers * fraction), 1)
            end = min(start + span, total_layers)
            shards.append(Shard(model_name=model_name, start_layer=start, end_layer=end, total_layers=total_layers))
            try:
                peer.shard = shards[-1]
            except Exception:
                pass
            start = end
        if shards:
            shards[-1].end_layer = total_layers
        return shards


def _to_float(val: Any) -> float:
    try:
        if isinstance(val, str):
            return float(val.lower().replace("gb", "").strip() or 0.0)
        return float(val)
    except Exception:
        return 0.0


def _sample_with_backend(*args: Any, **kwargs: Any):
    # Distributed generation is tinygrad-based today, but resolve through backend utility
    # so we don't hardcode tinygrad module paths in callers.
    try:
        return backend_helpers_module().sample(*args, **kwargs)
    except Exception:
        return backend_helpers_module("tinygrad").sample(*args, **kwargs)


def _encode_token_tensor(token: int) -> Dict[str, Any]:
    buf = struct.pack("<i", int(token))
    return {
        "buffer": base64.b64encode(buf).decode("ascii"),
        "shape": [1, 1],
        "dtype": "int32",
    }

def _encode_tensor(tensor: Any) -> Dict[str, Any]:
    if hasattr(tensor, "numpy"):
        arr = tensor.numpy()
    else:
        arr = np.asarray(tensor)
    return {
        "buffer": base64.b64encode(arr.tobytes()).decode("ascii"),
        "shape": list(arr.shape),
        "dtype": str(arr.dtype),
    }


def _decode_tensor(tensor_payload: Any) -> tg.Tensor | None:
    if not isinstance(tensor_payload, dict):
        return None
    buf = tensor_payload.get("buffer")
    if not buf:
        return None
    try:
        raw = base64.b64decode(buf)
    
        dtype = _normalize_dtype(str(tensor_payload.get("dtype", "float32")))
        arr = np.frombuffer(raw, dtype=np.dtype(dtype))
        shape = tensor_payload.get("shape")

        if shape:
            arr = arr.reshape(shape)

        return tg.Tensor(arr)
    except Exception:
        return None


def _normalize_dtype(dtype: str) -> str:
    lower = dtype.lower()
    for candidate in (
        "float16",
        "bfloat16",
        "float32",
        "float64",
        "int64",
        "int32",
        "int16",
        "int8",
        "uint8",
    ):
        if candidate in lower:
            return candidate
    return "float32"


def _shard_payload(shard: Shard) -> Dict[str, Any]:
    return {
        "model_name": shard.model_name,
        "start_layer": shard.start_layer,
        "end_layer": shard.end_layer,
        "total_layers": shard.total_layers,
    }
