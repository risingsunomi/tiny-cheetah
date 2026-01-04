from __future__ import annotations

import json
import os
import socket
import threading
import time
import uuid
from typing import Any, Dict, List, Optional, Tuple

from tiny_cheetah.logging_utils import get_logger
from tiny_cheetah.orchestration.peer import PeerInfo
from tiny_cheetah.orchestration.server import ServerProfile
from tiny_cheetah.orchestration.model_client import ModelClient
from tiny_cheetah.orchestration.server_manager import ServerManager
from tiny_cheetah.orchestration.device_info import collect_host_info
from tiny_cheetah.models.shard import Shard

logger = get_logger(__name__)


class PeerManager:
    """Coordinates discovery, remote connections, and model dispatch."""

    def __init__(self) -> None:
        self.peer_id = os.getenv("TC_PEER_ID") or f"cheetah-{uuid.uuid4().hex[:6]}"
        self.address = "0.0.0.0"
        self.port = int(os.getenv("TC_PORT", "8765"))
        self.server_manager = ServerManager(self.peer_id, self.address, self.port)
        self.peer_info = PeerInfo(peer_id=self.peer_id, address=self.address, port=self.port, peer_type="self")
        self._tensor_callback = lambda payload: {"echo": payload}
        self._apply_local_host_info()
        self.peers: Dict[str, PeerInfo] = {self.peer_id: self.peer_info}
        self.connected_server: Optional[ServerProfile] = None
        self._known_servers: Dict[str, ServerProfile] = {}
        self._model_clients: Dict[str, ModelClient] = {}
        self._rr_index = 0
        self._lock = threading.RLock()
        self._server_targets = self._parse_server_env()
        # Seed a local model client so the caller is represented as a peer.
        self._get_model_client(ring=[], node_index=0, shard=Shard("local", 0, 0, 0))

    # Discovery -----------------------------------------------------------
    def discover_servers(self) -> None:
        with self._lock:
            self.peers = {self.peer_id: self.peer_info}

        # UDP local discovery
        for info in self.server_manager.discover_local_udp(self.port):
            server_id = info.get("server_id") or info.get("peer_id") or f"{info.get('address')}:{info.get('port')}"
            if server_id == self.peer_id:
                continue
            ping_ms = float(info.get("rtt_ms", 0.0))
            profile = ServerProfile(server_id=server_id, address=(info.get("address", ""), int(info.get("port", self.port))), ping_ms=ping_ms)
            self.connected_server = self.connected_server or profile
            self._ensure_server_peer(server_id, info.get("address", ""), int(info.get("port", self.port)))
            self._apply_server_peers(
                server_id,
                info.get("address", ""),
                int(info.get("port", self.port)),
                server_info=info,
                ping_ms=ping_ms,
            )

        # Static/env servers
        for host, port in self._server_targets:
            try:
                info = self._ping_server(host, port)
            except Exception as exc:
                logger.debug("Server %s:%s unreachable: %s", host, port, exc)
                continue
            server_id = info.get("server_id") or info.get("peer_id") or f"{host}:{port}"
            if server_id == self.peer_id:
                continue
            ping_ms = float(info.get("rtt_ms", 0.0))
            profile = ServerProfile(server_id=server_id, address=(host, port), ping_ms=ping_ms)
            self.connected_server = self.connected_server or profile
            self._ensure_server_peer(server_id, host, port)
            self._apply_server_peers(server_id, host, port, server_info=info, ping_ms=ping_ms)

        # If hosting, include connected clients
        for hosted in self.server_manager.hosted_clients():
            if hosted.peer_id == self.peer_id:
                continue
            self.peers[hosted.peer_id] = hosted

    def discover_peers(self) -> None:
        """Backward-compatible wrapper."""
        self.discover_servers()

    # Connections ---------------------------------------------------------
    def connect_to_server(self, host: str, port: int, password: str = "") -> ServerProfile:
        profile, _ = self._connect_and_sync(host, port, password=password)
        return profile

    def connect_to_peer(self, host: str, port: int, password: str = "") -> PeerInfo:
        _, peer = self._connect_and_sync(host, port, password=password)
        if peer is None:
            raise RuntimeError("Unable to connect to peer")
        return peer

    def list_peers(self, include_self: bool = False) -> List[PeerInfo]:
        with self._lock:
            if include_self:
                return list(self.peers.values())
            return [p for pid, p in self.peers.items() if pid != self.peer_id]

    def peer_count(self) -> int:
        with self._lock:
            return len(self.peers)

    def hosted_clients(self) -> List[PeerInfo]:
        return self.server_manager.hosted_clients()

    def aggregate_devices(self) -> List[str]:
        devices: List[str] = []
        with self._lock:
            for pid, peer in self.peers.items():
                if pid == self.peer_id:
                    local_devices = peer.devices or []
                    if not local_devices and self.server_manager.is_hosting():
                        local_devices = ["local-host"]
                    devices.extend(local_devices)
                    continue
                devices.extend(peer.devices)
        return devices

    def assign_shards(self, model_name: str, total_layers: int) -> None:
        ModelClient.plan_shards(self._ordered_peers(), model_name, total_layers)

    # Info proxies --------------------------------------------------------
    def get_gpu_inventory(self) -> List[dict]:
        return self.server_manager.get_gpu_inventory()

    def get_host_info(self) -> dict:
        self._apply_local_host_info()
        return self.server_manager.get_host_info()

    def is_hosting(self) -> bool:
        return self.server_manager.is_hosting()

    def access_password_required(self) -> bool:
        return self.server_manager.access_password_required()

    @property
    def server_profile(self) -> ServerProfile:
        return self.connected_server or self.server_manager.server_profile

    def update_server_profile(
        self,
        description: str,
        flops_gflops: float,
        gpu_description: str,
        ping_ms: float,
        motd: str,
    ) -> None:
        with self._lock:
            profile = self.server_manager.server_profile
            profile.description = description
            profile.flops_gflops = flops_gflops
            profile.gpu_description = gpu_description
            profile.ping_ms = ping_ms
            profile.motd = motd
            if self.connected_server and self.connected_server.server_id == profile.server_id:
                self.connected_server.description = description
                self.connected_server.flops_gflops = flops_gflops
                self.connected_server.gpu_description = gpu_description
                self.connected_server.ping_ms = ping_ms
                self.connected_server.motd = motd
            local_peer = self.peers.get(profile.server_id)
            if local_peer:
                local_peer.gpu_description = gpu_description
                local_peer.flops_gflops = flops_gflops
                local_peer.metadata.update({"motd": motd, "description": description})

    def disconnect_peer(self, peer_id: str) -> None:
        with self._lock:
            if peer_id == self.peer_id:
                return
            self.peers.pop(peer_id, None)
            if self.connected_server and self.connected_server.server_id == peer_id:
                self.connected_server = None
            self._model_clients.pop(peer_id, None)

    # Compute -------------------------------------------------------------
    def request_remote_compute(self, node_id: str, payload: dict) -> dict:
        peer = self.peers.get(node_id)
        if peer is None:
            raise RuntimeError("Unknown peer")
        ring = [
            {
                "peer_id": self.server_profile.server_id,
                "address": self.server_profile.address[0],
                "port": self.server_profile.address[1],
                "devices": ["server"],
            }
        ]
        ring.extend(
            {
                "peer_id": p.peer_id,
                "address": p.address,
                "port": p.port,
                "devices": p.devices,
            }
            for p in self._ordered_peers()
        )
        try:
            idx = [p["peer_id"] for p in ring].index(peer.peer_id)
        except ValueError:
            idx = 0
        shard = payload.get("shard")
        if isinstance(shard, dict):
            shard = Shard(
                shard.get("model_name", "model"),
                shard.get("start_layer", 0),
                shard.get("end_layer", 0),
                shard.get("total_layers", 0),
            )
        if shard is None or not isinstance(shard, Shard):
            shard = Shard("model", 0, 0, 0)
        tensor_bytes = payload.get("tensor_bytes", b"")
        client = self._get_model_client(shard=shard, ring=ring, node_index=idx)
        return client.compute_resp_bytes(shard, tensor_bytes, ring, idx)

    # Internal ------------------------------------------------------------
    def _ordered_peers(self) -> List[PeerInfo]:
        return ServerManager.ordered_peers(self.peers.values(), self.peer_id)

    def _apply_local_host_info(self) -> None:
        host_info = collect_host_info()
        device_entries = host_info.get("devices", []) or []
        device_names = [d.get("name", "device") for d in device_entries if isinstance(d, dict) and d.get("name")]
        if not device_names:
            device_names.append("cpu")
        self.peer_info.device_report = {"devices": device_entries}
        self.peer_info.gpu_description = ""
        self.peer_info.devices = device_names
        cpu_entry = next((d for d in device_entries if d.get("device") == "CPU"), {})
        gpu_entry = next((d for d in device_entries if d.get("device") == "GPU"), {})
        vram_gb = 0.0
        try:
            vram_gb = float(gpu_entry.get("ram_gb", 0.0) or 0.0)
        except Exception:
            vram_gb = 0.0
        meta = {
            "hostname": host_info.get("hostname", ""),
            "platform": host_info.get("platform", ""),
            "cpu_name": cpu_entry.get("name", ""),
            "cpu_cores": cpu_entry.get("cores", 0),
            "ram_gb": cpu_entry.get("ram_gb", 0.0),
            "gpu": gpu_entry.get("name", ""),
            "vram_gb": vram_gb,
        }
        self.peer_info.metadata.update(meta)

    def _connect_and_sync(self, host: str, port: int, password: str = "") -> Tuple[ServerProfile, Optional[PeerInfo]]:
        info = self._ping_server(host, port, password=password)
        peers_on_server = self._list_peers(host, port, password=password)
        server_id = info.get("server_id") or info.get("peer_id") or f"{host}:{port}"
        ping_ms = float(info.get("rtt_ms", 0.0))
        profile = ServerProfile(
            server_id=server_id,
            address=(host, port),
            ping_ms=ping_ms,
            description=info.get("description", "General compute"),
            flops_gflops=float(info.get("flops_gflops", 0.0) or 0.0),
            gpu_description=info.get("gpu_description", ""),
            motd=info.get("motd", ""),
            access_password=password or "",
        )
        with self._lock:
            self.connected_server = profile
            self._ensure_server_peer(server_id, host, port)
        self._apply_server_peers(server_id, host, port, peers_on_server, server_info=info, ping_ms=ping_ms)
        with self._lock:
            peer = self.peers.get(server_id)
        return profile, peer

    def _ensure_server_peer(self, server_id: str, host: str, port: int) -> None:
        with self._lock:
            if server_id in self.peers:
                return
            self.peers[server_id] = PeerInfo(
                peer_id=server_id,
                address=host,
                port=port,
                devices=[],
                metadata={"server_id": server_id},
                peer_type="server",
            )

    def _apply_server_peers(
        self,
        server_id: str,
        host: str,
        port: int,
        cached: Optional[list] = None,
        server_info: Optional[dict] = None,
        ping_ms: Optional[float] = None,
    ) -> None:
        nodes = cached if cached is not None else self._list_peers(host, port)
        with self._lock:
            server_peer = self.peers.get(server_id)
            if server_peer is not None:
                if ping_ms is not None:
                    server_peer.ping_ms = ping_ms
                if server_info:
                    server_peer.device_report = server_info.get("hardware", server_peer.device_report)
                    server_peer.devices = server_info.get("devices", server_peer.devices)
                    try:
                        server_peer.flops_gflops = float(server_info.get("flops_gflops", server_peer.flops_gflops) or 0.0)
                    except Exception:
                        pass
                    server_peer.gpu_description = server_info.get("gpu_description", server_peer.gpu_description)
            for entry in nodes:
                pid = entry.get("peer_id") or f"{server_id}:{len(self.peers)}"
                if pid == self.peer_id:
                    continue
                hw = entry.get("hardware", {}) if isinstance(entry, dict) else {}
                devices = entry.get("devices", []) if isinstance(entry, dict) else []
                server_port = int(entry.get("port", port)) if isinstance(entry, dict) else port
                self.peers[pid] = PeerInfo(
                    peer_id=pid,
                    address=entry.get("address", host) if isinstance(entry, dict) else host,
                    port=0,  # peer port not used; routing via server
                    devices=devices,
                    device_report=hw,
                    metadata={"server_id": server_id, "server_port": server_port},
                    peer_type="client",
                )
            if server_peer is not None:
                meta = server_peer.metadata or {}
                meta.update(
                    {
                        "server_id": server_id,
                        "motd": (server_info or {}).get("motd", meta.get("motd", "")),
                        "description": (server_info or {}).get("description", meta.get("description", "")),
                        "password": bool((server_info or {}).get("password", meta.get("password", False))),
                        "peers": nodes,
                    }
                )
                server_peer.metadata = meta

    def _get_model_client(
        self,
        shard: Optional[Shard] = None,
        ring: Optional[List[dict]] = None,
        node_index: int = 0,
    ) -> ModelClient:
        profile = self.connected_server or self.server_manager.server_profile
        key = profile.server_id
        client = self._model_clients.get(key)
        if client is None:
            client = ModelClient(
                profile,
                shard=shard or Shard("local", 0, 0, 0),
                peer_id=self.peer_info.peer_id,
                node_index=node_index,
                ring=ring or [],
            )
            self._model_clients[key] = client
        else:
            if shard is not None:
                client.shard = shard
            if ring is not None:
                client.ring = ring
            client.node_index = node_index
        return client

    def _parse_server_env(self) -> List[Tuple[str, int]]:
        raw = os.getenv("TC_SERVER", "").strip()
        if not raw:
            return []
        targets: List[Tuple[str, int]] = []
        for item in raw.split(","):
            part = item.strip()
            if not part:
                continue
            if ":" in part:
                host, port_text = part.split(":", 1)
                try:
                    targets.append((host, int(port_text)))
                except ValueError:
                    continue
            else:
                targets.append((part, self.port))
        return targets

    def _ping_server(self, host: str, port: int, password: str = "") -> dict:
        payload = {"command": "ping"}
        if password:
            payload["password"] = password
        data = json.dumps(payload).encode("utf-8")
        start = time.time()
        with socket.create_connection((host, port), timeout=3) as sock:
            sock.sendall(data)
            sock.shutdown(socket.SHUT_WR)
            response = sock.recv(4096)
        rtt_ms = max((time.time() - start) * 1000.0, 0.0)
        try:
            data = json.loads(response.decode("utf-8"))
        except Exception as exc:
            raise RuntimeError(f"Invalid ping response from {host}:{port}: {exc}")
        data["rtt_ms"] = rtt_ms
        return data

    def _list_peers(self, host: str, port: int, password: str = "") -> list:
        payload = {"command": "list_peers"}
        if password:
            payload["password"] = password
        data = json.dumps(payload).encode("utf-8")
        with socket.create_connection((host, port), timeout=3) as sock:
            sock.sendall(data)
            sock.shutdown(socket.SHUT_WR)
            response = sock.recv(8192)
        try:
            data = json.loads(response.decode("utf-8"))
            return data.get("peers", [])
        except Exception as exc:
            logger.debug("list_peers failed from %s:%s: %s", host, port, exc)
            return []


_MANAGER: Optional[PeerManager] = None


def get_peer_manager() -> PeerManager:
    global _MANAGER
    if _MANAGER is None:
        _MANAGER = PeerManager()
    return _MANAGER
