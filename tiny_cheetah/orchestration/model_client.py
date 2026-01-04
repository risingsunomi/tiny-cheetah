from __future__ import annotations

import json
import socket
import base64
import asyncio
from typing import Any, Dict, List, Sequence
from pathlib import Path

import tinygrad as tg

from transformers import AutoTokenizer

from tiny_cheetah.models.shard import Shard

from tiny_cheetah.logging_utils import get_logger
from tiny_cheetah.orchestration.server import ServerProfile
from tiny_cheetah.models.llm.helpers import (
    load_model,
    sample
)

logger = get_logger(__name__)


class ModelClient:
    """Handles tensor RPCs and simple shard planning based on RAM/VRAM."""

    def __init__(
        self,
        server: ServerProfile,
        shard: Shard | None = None,
        peer_id: str | None = None,
        node_index: int = 0,
        ring: Sequence[Any] = (),
    ) -> None:
        self.server = server
        self.shard = shard or Shard("local", 0, 0, 0)
        self.peer_id = peer_id or server.server_id
        self.node_index = node_index
        self.ring = ring

    def dispatch_tensor(self, payload: dict) -> Dict[str, Any]:
        """Send a tensor payload to the remote peer for execution."""
        message = {
            "command": "tensor_payload",
            "payload": payload,
        }
        return self._send(message)

    def send_tensor_bytes(self, tensor_bytes: bytes) -> Dict[str, Any]:
        """Send raw tensor bytes; server will echo or process."""
        encoded = base64.b64encode(tensor_bytes).decode("ascii")
        message = {"command": "tensor_bytes", "payload": {"buffer": encoded}}
        return self._send(message)

    def request_tensor_bytes(self) -> bytes:
        """Request tensor bytes from server (server echoes back if available)."""
        reply = self._send({"command": "request_tensor"})
        buf = reply.get("buffer", "")
        try:
            return base64.b64decode(buf)
        except Exception:
            return b""

    def recv_tensor_bytes(self, timeout: float = 5.0) -> bytes:
        """
        Blocking receive helper for peer-to-peer exchange.
        In this simplified flow we reuse request_tensor; callers can poll.
        """
        try:
            return self.request_tensor_bytes()
        except Exception:
            return b""

    def ping(self) -> Dict[str, Any]:
        return self._send({"command": "ping"})

    def compute_resp_bytes(
        self,
        shard: Shard,
        tensor_bytes: bytes,
        ring: Sequence[Any],
        node_index: int,
    ) -> Dict[str, Any]:
        encoded = base64.b64encode(tensor_bytes or b"").decode("ascii") if tensor_bytes else ""
        resp: Dict[str, Any] = {
            "peer_id": self.server.server_id,
            "from_peer_id": self.peer_id,
            "shard": shard.__dict__ if hasattr(shard, "__dict__") else {"model_name": getattr(shard, "model_name", "model")},
            "tensor": encoded,
        }
        # No direct peer-to-peer routing; rely on the server to coordinate.
        resp["ring"] = list(ring)
        resp["node_index"] = node_index
        return resp

    # ------------------------------------------------------------------ compute
    async def compute_resp(
        self,
        input_ids: tg.Tensor,
        attention_mask: tg.Tensor,
        temp: float = 1.0,
        top_k: int = 0,
        top_p: float = 0.8,
        alpha_f: float = 0.0,
        alpha_p: float = 0.0
    ) -> Dict[str, Any]:
        try:
            # load model shard
            sanitized = self.shard.model_name.replace("/", "__")
            cache_path = (Path.home() / ".cache" / "tiny_cheetah_models") / sanitized
            candidate_path = Path(self.shard.model_name).expanduser()
            model, model_config, tokenizer, model_path = await load_model(
                self.shard.model_name,
                self.shard,
                self.input_ids.device
            )

            # generate output tensor
            device = input_ids.device
            input_ids = input_ids.to(device)
            attention_mask = attention_mask.to(device)

            out_tokens: list[int] = []
            curr_pos = attention_mask.shape[1]

            position_ids = ((attention_mask.cumsum(axis=1) - 1) * attention_mask).to(device)
            logits = model(input_ids, attention_mask=attention_mask, position_ids=position_ids)
            next_logit = logits[:, -1, :].flatten()
            tok = sample(next_logit, temp=temp, k=top_k, p=top_p, af=alpha_f, ap=alpha_p).item()
            out_tokens.append(tok)
            out_bytes = tg.Tensor(out_tokens, device="CPU").numpy().tobytes()

            compute_resp = {
                "peer_id": self.server.server_id,
                "from_peer_id": self.peer_id,
                "tensor": base64.b64encode(out_bytes).decode("ascii")
            }
            # Server will orchestrate forwarding; send to server (ring[0])
            if not self.ring:
                return compute_resp
            next_index = self.node_index + 1 if self.node_index + 1 < len(self.ring) else 0
            host, port = self.server.address
            if host and port:
                try:
                    self._send_to_peer(
                        host,
                        int(port),
                        {
                            "command": "tensor_forward",
                            "payload": {**compute_resp, "next_index": next_index},
                        },
                    )
                    compute_resp["forwarded_via_server"] = self.server.server_id
                except Exception as exc:
                    compute_resp["forward_error"] = str(exc)
            else:
                compute_resp["forward_error"] = "missing server address/port"
            return compute_resp
        except Exception as exc:
            return {"error": str(exc), "peer_id": self.server.server_id}

    def _send(self, message: dict) -> Dict[str, Any]:
        data = json.dumps(message).encode("utf-8")
        host, port = self.server.address
        with socket.create_connection((host, port), timeout=5) as sock:
            sock.sendall(data)
            sock.shutdown(socket.SHUT_WR)
            response = sock.recv(65536)
        try:
            return json.loads(response.decode("utf-8"))
        except Exception as err:
            logger.error("Invalid response from server %s: %s", self.server.server_id, err)
            raise

    def _send_to_peer(self, host: str, port: int, message: dict) -> Dict[str, Any]:
        data = json.dumps(message).encode("utf-8")
        with socket.create_connection((host, port), timeout=5) as sock:
            sock.sendall(data)
            sock.shutdown(socket.SHUT_WR)
            response = sock.recv(65536)
        try:
            return json.loads(response.decode("utf-8"))
        except Exception:
            return {}

    @staticmethod
    def plan_shards(peers: Sequence[Any], model_name: str, total_layers: int) -> List[Shard]:
        """
        Create shard assignments across peers based on available VRAM/RAM.
        Returns a list of Shard objects in peer order.
        """
        capacities = []
        for peer in peers:
            hw = peer.device_report or {}
            devices = hw.get("devices", []) or hw.get("gpus", []) or []
            gpu_entry = next((d for d in devices if isinstance(d, dict) and d.get("device") == "GPU"), None)
            vram = 0.0
            if gpu_entry:
                try:
                    vram = float(gpu_entry.get("ram_gb", gpu_entry.get("total_mem_gb", 0.0)) or 0.0)
                except Exception:
                    vram = 0.0
            ram = 0.0
            if isinstance(hw, dict):
                try:
                    ram = float(hw.get("ram_gb", 0.0))
                except Exception:
                    ram = 0.0
            if ram == 0.0 and isinstance(devices, list):
                cpu_entry = next((d for d in devices if isinstance(d, dict) and d.get("device") == "CPU"), None)
                if cpu_entry:
                    try:
                        ram = float(cpu_entry.get("ram_gb", 0.0))
                    except Exception:
                        ram = 0.0
            flops = getattr(peer, "flops_gflops", 0.0) if hasattr(peer, "flops_gflops") else 0.0
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
        # ensure coverage
        if shards:
            shards[-1].end_layer = total_layers
        return shards
