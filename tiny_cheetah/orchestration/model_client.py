from __future__ import annotations

import json
import socket
from typing import Any, Dict, List

from tiny_cheetah.logging_utils import get_logger
from tiny_cheetah.orchestration.peer import PeerInfo

logger = get_logger(__name__)


class ModelClient:
    """Handles tensor RPCs and simple shard planning based on RAM/VRAM."""

    def __init__(self, peer: PeerInfo, password: str = "") -> None:
        self.peer = peer
        self.password = password

    def dispatch_tensor(self, payload: dict) -> Dict[str, Any]:
        """Send a tensor payload to the remote peer for execution."""
        message = {
            "command": "tensor_payload",
            "payload": payload,
            "password": self.password,
        }
        return self._send(message)

    def ping(self) -> Dict[str, Any]:
        return self._send({"command": "ping"})

    def _send(self, message: dict) -> Dict[str, Any]:
        data = json.dumps(message).encode("utf-8")
        with socket.create_connection((self.peer.address, self.peer.port), timeout=5) as sock:
            sock.sendall(data)
            sock.shutdown(socket.SHUT_WR)
            response = sock.recv(65536)
        try:
            return json.loads(response.decode("utf-8"))
        except Exception as err:
            logger.error("Invalid response from peer %s: %s", self.peer.peer_id, err)
            raise

    @staticmethod
    def plan_shards(peers: List[PeerInfo], payload_mem_gb: float = 1.0) -> List[dict]:
        """Return a simple shard plan proportional to reported VRAM/RAM."""
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
            capacity = vram if vram > 0 else ram
            if capacity <= 0:
                capacity = 1.0  # minimal weight
            capacities.append((peer, capacity))

        total = sum(cap for _, cap in capacities) or 1.0
        plan: List[dict] = []
        for peer, cap in capacities:
            fraction = cap / total
            plan.append(
                {
                    "peer_id": peer.peer_id,
                    "address": peer.address,
                    "port": peer.port,
                    "fraction": fraction,
                    "expected_mem_gb": max(payload_mem_gb * fraction, 0.0),
                }
            )
        return plan
