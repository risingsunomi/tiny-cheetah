from __future__ import annotations

import json
import socket
import base64
import asyncio
from typing import Any, Dict, List, Sequence

import numpy as np
try:
    from tinygrad import Tensor
except Exception:  # pragma: no cover - optional import
    Tensor = None  # type: ignore

from tiny_cheetah.models.shard import Shard

from tiny_cheetah.logging_utils import get_logger
from tiny_cheetah.orchestration.server import ServerProfile

logger = get_logger(__name__)


class ModelClient:
    """Handles tensor RPCs and simple shard planning based on RAM/VRAM."""

    def __init__(self, server: ServerProfile) -> None:
        self.server = server

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

    # ------------------------------------------------------------------ compute
    def compute_resp(
        self,
        shard: Shard,
        tensor_bytes: bytes,
        ring: List[Dict[str, Any]],
        index: int,
    ) -> Dict[str, Any]:
        """
        Run a tinygrad operation over the provided tensor bytes for the given shard.
        This is a minimal placeholder that applies tanh to the incoming tensor slice
        and forwards or returns the result based on the provided ring.
        """
        if shard is None:
            return {"error": "missing shard", "peer_id": self.server.server_id}
        if Tensor is None:
            return {"error": "tinygrad not available", "peer_id": self.server.server_id}
        try:
            arr = np.frombuffer(tensor_bytes, dtype=np.float32)
            t = Tensor(arr)
            out = t.tanh()
            out_np = out.numpy().astype(np.float32)
            out_bytes = out_np.tobytes()
            compute_resp = {
                "peer_id": self.server.server_id,
                "ring_index": index,
                "shard": {
                    "model_name": shard.model_name,
                    "start_layer": shard.start_layer,
                    "end_layer": shard.end_layer,
                    "total_layers": shard.total_layers,
                },
                "tensor": base64.b64encode(out_bytes).decode("ascii"),
                "ring": ring,
            }
            # Server will orchestrate forwarding; send to server (ring[0])
            if not ring:
                return compute_resp
            next_index = index + 1 if index + 1 < len(ring) else 0
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
            gpus = hw.get("gpus", []) or []
            vram = 0.0
            if gpus and isinstance(gpus[0], dict):
                try:
                    vram = float(gpus[0].get("total_mem_gb", 0.0))
                except Exception:
                    vram = 0.0
            ram = 0.0
            try:
                ram = float(hw.get("ram_gb", 0.0))
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
