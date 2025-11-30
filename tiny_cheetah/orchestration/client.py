from __future__ import annotations

import json
import socket
from typing import Any, Dict, Optional


class PeerClient:
    def __init__(self, address: str, port: int, identity: dict, password: str = "") -> None:
        self.address = address
        self.port = port
        self.identity = identity
        self.password = password

    def _send(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        sock = socket.create_connection((self.address, self.port), timeout=5)
        payload = dict(payload)
        if self.password:
            payload["password"] = self.password
        sock.sendall(json.dumps(payload).encode("utf-8"))
        sock.shutdown(socket.SHUT_WR)
        response = sock.recv(65536)
        sock.close()
        try:
            return json.loads(response.decode("utf-8"))
        except json.JSONDecodeError:
            return {"error": "invalid response"}

    def handshake(self) -> Dict[str, Any]:
        response = self._send({"command": "ping", "identity": self.identity})
        if "error" in response:
            raise RuntimeError(response["error"])
        identity = response.get("identity", {})
        return identity

    def dispatch_tensor(self, payload: dict) -> Dict[str, Any]:
        return self._send({"command": "tensor", "payload": payload, "identity": self.identity})
