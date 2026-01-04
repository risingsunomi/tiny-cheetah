from __future__ import annotations

import asyncio
import json
import threading
from dataclasses import dataclass, field
from typing import Callable, Optional, Tuple, Dict
from collections import deque
import socket

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
        self._caller_queue: deque[dict] = deque(maxlen=100)

        self._loop: Optional[asyncio.AbstractEventLoop] = None
        self._server: Optional[asyncio.AbstractServer] = None
        self._thread: Optional[threading.Thread] = None
        self._udp_transport: Optional[asyncio.DatagramTransport] = None

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
            if self._udp_transport is not None:
                self._udp_transport.close()
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

        profile_meta = {
            "description": self.server_profile.description,
            "flops_gflops": self.server_profile.flops_gflops,
            "gpu_description": self.server_profile.gpu_description,
            "motd": self.server_profile.motd,
            "password": bool(self.access_password),
        }

        command = message.get("command")
        response: dict
        if command == "ping":
            response = {
                "status": "pong",
                "server_id": self.server_profile.server_id,
                "hardware": self.hardware,
                "devices": self.devices,
                **profile_meta,
            }
        elif command == "tensor_forward":
            payload = message.get("payload", {}) or {}
            next_index = payload.get("next_index")
            ring = payload.get("ring", [])
            if next_index is None or not isinstance(ring, list):
                response = {"error": "invalid forward request"}
            elif next_index < len(ring):
                target = ring[next_index]
                host = target.get("address")
                port = target.get("port")
                if host and port:
                    try:
                        data = json.dumps(
                            {
                                "command": "tensor_forward",
                                "payload": {**payload, "next_index": next_index + 1},
                            }
                        ).encode("utf-8")
                        with socket.create_connection((host, int(port)), timeout=5) as sock:
                            sock.sendall(data)
                        response = {"status": "forwarded", "peer_id": target.get("peer_id", "")}
                    except Exception as exc:
                        response = {"error": str(exc)}
                else:
                    response = {"error": "missing address/port"}
            else:
                response = {"status": "complete"}
        elif command == "list_peers":
            response = {
                "status": "ok",
                "server_id": self.server_profile.server_id,
                "hardware": self.hardware,
                "devices": self.devices,
                **profile_meta,
                "peers": [
                    {
                        "peer_id": p.peer_id,
                        "devices": p.devices,
                        "hardware": p.device_report,
                        "address": p.address,
                        "port": p.port,
                    }
                    for p in self._connected_peers.values()
                ],
            }
        elif command == "return_tensor":
            if len(self._caller_queue) >= self._caller_queue.maxlen:  # type: ignore[arg-type]
                response = {"error": "server busy"}
            else:
                entry = {
                    "caller_id": message.get("caller_id") or message.get("peer_id") or f"{peer[0]}:{peer[1]}",
                    "payload": message.get("payload", {}),
                }
                self._caller_queue.append(entry)
                response = {"status": "ok"}
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
                address=peer[0],
                port=peer[1],
                devices=message.get("devices", []),
                device_report=message.get("hardware", {}),
                metadata={"last_seen": "ping" if command == "ping" else command},
                peer_type="client",
            )

    def _run_loop(self) -> None:
        """Event loop runner for the TCP server and UDP probe responder."""
        self._loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self._loop)
        coro = asyncio.start_server(self._handle_client, *self.address)
        self._server = self._loop.run_until_complete(coro)
        udp_coro = self._loop.create_datagram_endpoint(
            lambda: _UDPProbeResponder(self.server_profile.server_id, self.hardware),
            local_addr=("0.0.0.0", self.address[1]),
        )
        try:
            self._udp_transport, _ = self._loop.run_until_complete(udp_coro)
        except Exception:
            self._udp_transport = None
        try:
            self._loop.run_forever()
        finally:
            self._loop.close()

    def get_connected_peers(self) -> list[PeerInfo]:
        return list(self._connected_peers.values())


class _UDPProbeResponder(asyncio.DatagramProtocol):
    def __init__(self, server_id: str, hardware: dict) -> None:
        super().__init__()
        self.server_id = server_id
        self.hardware = hardware
        self.transport: Optional[asyncio.DatagramTransport] = None

    def datagram_received(self, data: bytes, addr) -> None:  # type: ignore[override]
        try:
            msg = json.loads(data.decode("utf-8"))
        except Exception:
            return
        if msg.get("command") != "ping":
            return
        response = json.dumps(
            {
                "server_id": self.server_id,
                "hardware": self.hardware,
            }
        ).encode("utf-8")
        if self.transport is not None:
            try:
                self.transport.sendto(response, addr)
            except Exception:
                return
