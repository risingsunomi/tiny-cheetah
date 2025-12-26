from __future__ import annotations

import json
import socket
from typing import Any, Dict

import tinygrad as tg

from tiny_cheetah.logging_utils import get_logger
from tiny_cheetah.orchestration.peer import PeerInfo

logger = get_logger(__name__)
class TensorClient:
    def __init__(self, peer_info: PeerInfo) -> None:
        self.peer_info = peer_info

    def _send(self, payload: tg.Tensor) -> Dict[str, Any]:
        sock = socket.create_connection((self.peer_info.address, self.peer_info.port), timeout=5)
        sock.sendall(payload.to_bytes())
        sock.shutdown(socket.SHUT_WR)
        response = sock.recv(65536)
        sock.close()
        try:
            return json.loads(response.decode("utf-8"))
        except json.JSONDecodeError:
            raise ValueError("Invalid JSON response from peer")
        except Exception as err:
            logger.error(f"Error occurred while sending payload: {err}")
            raise

    def send_tensor(self, payload: dict) -> Dict[str, Any]:
        return self._send({"command": "tensor", "payload": payload, "peer_id": self.peer_info.peer_id})
    
    def recv_tensor(self) -> tg.Tensor:
        sock = socket.create_connection((self.peer_info.address, self.peer_info.port), timeout=5)
        sock.sendall(b'{"command":"recv_tensor","peer_id":"' + self.peer_info.peer_id.encode("utf-8") + b'"}')
        sock.shutdown(socket.SHUT_WR)
        data = sock.recv(65536)
        sock.close()
        return tg.Tensor.from_bytes(data)

