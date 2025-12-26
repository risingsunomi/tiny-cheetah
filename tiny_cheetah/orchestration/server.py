from __future__ import annotations

import asyncio
import json
import threading
from dataclasses import dataclass, field
from typing import Callable, Optional, Tuple, Dict

from tiny_cheetah.orchestration.peer import PeerInfo
from tiny_cheetah.logging_utils import get_logger

logger = get_logger(__name__)


@dataclass
class ServerProfile:
    server_id: str = ""
    address: Tuple[str, int] | Tuple[str, str] | tuple = ("0.0.0.0", 8765)
    access_password: str = ""
    description: str = "General compute"
    flops_gflops: float = 0.0
    gpu_description: str = ""
    ping_ms: float = 0.0
    motd: str = ""


class PeerServer:
    """Asyncio-based TCP server for peer communication."""

    def __init__(
        self,
        address: Tuple[str, int] | tuple,
        profile: ServerProfile,
        hardware: dict,
        devices: list[str],
        tensor_callback: Callable[[dict], dict],
        access_password: str = "",
    ) -> None:
        self.address = address
        self.server_profile = profile
        self.hardware = hardware
        self.devices = devices
        self.tensor_callback = tensor_callback
        self.access_password = access_password
        self._connected_peers: Dict[str, PeerInfo] = {}

        self._loop: Optional[asyncio.AbstractEventLoop] = None
        self._server: Optional[asyncio.AbstractServer] = None
        self._thread: Optional[threading.Thread] = None

    # ------------------------------------------------------------------ server
    def start(self) -> None:
        if self._thread and self._thread.is_alive():
            return
        self._thread = threading.Thread(target=self._run_loop, name=self.server_profile.server_id, daemon=True)
        self._thread.start()
        logger.info("Peer server starting on %s:%s", *self.address)

    def stop(self) -> None:
        if self._loop is None:
            return

        async def _shutdown() -> None:
            if self._server is not None:
                self._server.close()
                await self._server.wait_closed()
            tasks = asyncio.all_tasks(self._loop)
            for task in tasks:
                task.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)
            self._loop.stop()

        asyncio.run_coroutine_threadsafe(_shutdown(), self._loop).result(timeout=2.0)
        if self._thread:
            self._thread.join(timeout=2.0)
        self._server = None
        self._loop = None
        self._thread = None
        logger.info("Peer server stopped.")

    # ---------------------------------------------------------------- handlers
    async def _handle_client(self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
        peer = writer.get_extra_info("peername")
        try:
            data = await reader.read(8192)
            message = json.loads(data.decode("utf-8"))
        except Exception:
            writer.write(b'{"error":"invalid json"}')
            await writer.drain()
            writer.close()
            await writer.wait_closed()
            return

        if self.access_password and message.get("password") != self.access_password:
            writer.write(b'{"error":"unauthorized"}')
            await writer.drain()
            writer.close()
            await writer.wait_closed()
            return

        command = message.get("command")
        response: dict
        if command == "ping":
            response = {"status": "pong"}
        elif command == "list_peers":
            response = {
                "status": "ok",
                "peer_id": self.server_profile.server_id,
                "peers": [
                    {
                        "peer_id": p.peer_id,
                        "devices": p.devices,
                        "hardware": p.device_report,
                    }
                    for p in self._connected_peers.values()
                ],
            }
        elif command == "tensor_payload":
            payload = message.get("payload", {})
            try:
                result = self.tensor_callback(payload)
                response = {"status": "ok", "result": result}
            except Exception as exc:
                logger.warning("Tensor callback failed: %s", exc)
                response = {"error": str(exc)}
        else:
            response = {"error": f"unknown command {command!r}"}

        try:
            writer.write(json.dumps(response).encode("utf-8"))
            await writer.drain()
        finally:
            writer.close()
            await writer.wait_closed()
            logger.info("Handled request from %s", peer)
            remote_id = message.get("peer_id") or f"{peer[0]}:{peer[1]}"
            self._connected_peers[remote_id] = PeerInfo(
                peer_id=remote_id,
                address="hidden",
                port=0,
                devices=message.get("devices", []),
                device_report=message.get("hardware", {}),
                metadata={"last_seen": "ping" if command == "ping" else command},
                peer_type="client",
            )

    def get_connected_peers(self) -> list[PeerInfo]:
        return list(self._connected_peers.values())

    # ----------------------------------------------------------------- internals
    def _run_loop(self) -> None:
        self._loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self._loop)
        coro = asyncio.start_server(self._handle_client, *self.address)
        self._server = self._loop.run_until_complete(coro)
        try:
            self._loop.run_forever()
        finally:
            self._loop.close()
