from __future__ import annotations

from dataclasses import dataclass
import json
import socketserver
import threading
from typing import Callable, Optional


class PeerRequestHandler(socketserver.BaseRequestHandler):
    def handle(self) -> None:  # type: ignore[override]
        data = self.request.recv(8192)
        try:
            message = json.loads(data.decode("utf-8"))
        except Exception:
            self.request.sendall(b'{"error":"invalid json"}')
            return
        command = message.get("command")
        expected_password = getattr(self.server, "access_password", "")  # type: ignore[attr-defined]
        if expected_password and message.get("password") != expected_password:
            self.request.sendall(b'{"error":"unauthorized"}')
            return
        if command == "ping":
            response = {"status": "ok", "identity": self.server.identity}  # type: ignore[attr-defined]
        elif command == "tensor":
            callback = self.server.tensor_callback  # type: ignore[attr-defined]
            payload = message.get("payload", {})
            try:
                result = callback(payload)
            except Exception as exc:  # pragma: no cover
                result = {"error": str(exc)}
            response = {"status": "ok", "result": result}
        else:
            response = {"error": f"unknown command {command!r}"}
        self.request.sendall(json.dumps(response).encode("utf-8"))


class PeerServer(socketserver.ThreadingTCPServer):
    allow_reuse_address = True

    def __init__(
        self,
        address: tuple[str, int],
        identity: dict,
        tensor_callback: Callable[[dict], dict],
        access_password: str = "",
    ) -> None:
        super().__init__(address, PeerRequestHandler)
        self.identity = identity
        self.tensor_callback = tensor_callback
        self.access_password = access_password
        self._thread: Optional[threading.Thread] = None

    def start(self) -> None:
        if self._thread and self._thread.is_alive():
            return
        self._thread = threading.Thread(target=self.serve_forever, name="peer-server", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self.shutdown()
        if self._thread:
            self._thread.join(timeout=1.0)

@dataclass
class ServerProfile:
    description: str = ""
    flops_gflops: float = 0.0
    gpu_description: str = ""
    ping_ms: float = 0.0
    motd: str = ""