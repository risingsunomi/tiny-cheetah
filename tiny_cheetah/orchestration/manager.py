from __future__ import annotations

import socket
import json
import threading
import uuid
import time
import asyncio
from typing import Callable, Dict, List, Optional, Tuple

try:
    import torch  # type: ignore
except Exception:  # pragma: no cover - optional
    torch = None


from tiny_cheetah.logging_utils import get_logger
import os

from .device_info import collect_host_info
from tiny_cheetah.orchestration.peer import PeerInfo
from tiny_cheetah.orchestration.server import PeerServer, ServerProfile
from tiny_cheetah.orchestration.model_client import ModelClient

logger = get_logger(__name__)


class PeerManager:
    """Coordinates hosting, discovery, and remote connections."""
    def __init__(self) -> None:
        self.peer_id = os.getenv("TC_PEER_ID") or f"cheetah-{uuid.uuid4().hex[:6]}"
        self.address = "0.0.0.0"
        self.port = int(os.getenv("TC_PORT", "8765"))
        self.peer_info = PeerInfo(peer_id=self.peer_id, address=self.address, port=self.port, peer_type="self")
        self.peers: Dict[str, PeerInfo] = {self.peer_id: self.peer_info}
        self.server: Optional[ServerProfile] = None
        self._model_clients: Dict[str, ModelClient] = {}
        self._rr_index = 0
        self._lock = threading.RLock()
        self._host_lock = threading.Condition(self._lock)
        self._host_busy = False
        self._server: Optional[PeerServer] = None
        self._access_password: str = ""
        self._server_targets = self._parse_server_env()
        self._host_info = collect_host_info()
        self._gpu_info = self._detect_gpus()
        self.server_profile = ServerProfile()

    # Server Discovery -----------------------------------------------------------
    def discover_servers(self) -> None:
        with self._lock:
            self.peers = {self.peer_id: self.peer_info}
        # UDP local discovery
        for info in self._discover_local_udp():
            server_id = info.get("server_id") or info.get("peer_id") or f"{info.get('address')}:{info.get('port')}"
            if server_id == self.peer_id:
                continue
            ping_ms = float(info.get("rtt_ms", 0.0))
            with self._lock:
                profile = ServerProfile(
                    server_id=server_id,
                    address=(info.get("address", ""), int(info.get("port", self.port))),
                    ping_ms=ping_ms,
                )
                self.server = self.server or profile
                self._ensure_server_peer(server_id, info.get("address", ""), int(info.get("port", self.port)))
            self._apply_server_peers(server_id, info.get("address", ""), int(info.get("port", self.port)))

        # Static/env servers
        for host, port in self._server_targets:
            try:
                info = self._ping_peer(host, port)
            except Exception as exc:
                logger.debug("Server %s:%s unreachable: %s", host, port, exc)
                continue
            server_id = info.get("server_id") or info.get("peer_id") or f"{host}:{port}"
            if server_id == self.peer_id:
                continue
            ping_ms = float(info.get("rtt_ms", 0.0))
            with self._lock:
                profile = ServerProfile(
                    server_id=server_id,
                    address=(host, port),
                    ping_ms=ping_ms,
                )
                self.server = self.server or profile
                self._ensure_server_peer(server_id, host, port)
            self._apply_server_peers(server_id, host, port)
        if self._server is not None:
            for hosted in self._server.get_connected_peers():
                if hosted.peer_id == self.peer_id:
                    continue
                self.peers[hosted.peer_id] = hosted

    def discover_peers(self) -> None:
        """Backward-compatible wrapper."""
        self.discover_servers()
        
    # Hosting ------------------------------------------------------------------
    def start_hosting(self, password: str = "") -> None:
        if self._server is not None:
            return
        self._access_password = password.strip()
        self.server_profile.server_id = self.peer_id
        self.server_profile.address = (self.address, self.port)
        self._server = PeerServer(
            (self.address, self.port),
            self.server_profile,
            self._host_info,
            self._local_devices(),
            self._tensor_callback,
            self._access_password,
        )
        self.server = self.server_profile
        self._server.start()
        

    def stop_hosting(self) -> None:
        if self._server is not None:
            self._server.stop()
            self._server = None
        self.server = None
        with self._lock:
            self.peers = {self.peer_id: self.peer_info}
        

    # Connections ---------------------------------------------------------------
    def list_peers(self, include_self: bool = False) -> List[PeerInfo]:
        with self._lock:
            if include_self:
                return list(self.peers.values())
            return [p for pid, p in self.peers.items() if pid != self.peer_id]

    def peer_count(self) -> int:
        """Return total known nodes (includes self)."""
        with self._lock:
            return len(self.peers)

    def hosted_clients(self) -> List[PeerInfo]:
        if self._server is None:
            return []
        return self._server.get_connected_peers()

    def assign_shards(self, model_name: str, total_layers: int) -> None:
        """Compute shard assignments across peers."""
        ModelClient.plan_shards(self._ordered_peers(), model_name, total_layers)

    def aggregate_devices(self) -> List[str]:
        devices: List[str] = []
        with self._lock:
            for pid, peer in self.peers.items():
                if pid == self.peer_id and self._server is not None:
                    devices.append("local-host")
                devices.extend(peer.devices)
        return devices

    def _local_devices(self) -> List[str]:
        """Return only devices attached to this host."""
        return ["local-host"] if self._server is not None else []

    def get_gpu_inventory(self) -> List[dict]:
        return list(self._gpu_info)

    def get_host_info(self) -> dict:
        """Return CPU/RAM/GPU/TC_DEVICE info for this host."""
        return dict(self._host_info)

    def is_hosting(self) -> bool:
        with self._lock:
            return self._server is not None
    
    def update_server_profile(
        self,
        *,
        description: str,
        flops_gflops: float,
        gpu_description: str,
        ping_ms: float,
        motd: str,
    ) -> None:
        self.server_profile.description = description
        self.server_profile.flops_gflops = flops_gflops
        self.server_profile.gpu_description = gpu_description
        self.server_profile.ping_ms = ping_ms
        self.server_profile.motd = motd
        

    def connect_to_server(self, host: str, port: int, password: str = "") -> ServerProfile:
        """Explicitly connect to a server and record it."""
        info = self._ping_peer(host, port)
        server_id = info.get("server_id") or info.get("peer_id") or f"{host}:{port}"
        peers_on_server = self._list_peers(host, port)
        ping_ms = float(info.get("rtt_ms", 0.0))
        profile = ServerProfile(
            server_id=server_id,
            address=(host, port),
            ping_ms=ping_ms,
        )
        with self._lock:
            self.server = profile
            self._ensure_server_peer(server_id, host, port)
        self._apply_server_peers(server_id, host, port, peers_on_server)
        return profile

    def request_remote_compute(self, node_id: str, payload: dict) -> dict:
        peer = self.peers.get(node_id)
        if peer is None:
            raise RuntimeError("Unknown peer")
        client = self._get_model_client(peer)
        # Build ring with orchestrating server first, followed by ordered peers
        server_entry = {
            "peer_id": (self.server.server_id if self.server else self.peer_id),
            "address": (self.server.address[0] if self.server else self.address),
            "port": (self.server.address[1] if self.server else self.port),
            "devices": ["server"],
        }
        ring = [server_entry]
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
        shard_data = payload.get("shard")
        if isinstance(shard_data, dict):
            from tiny_cheetah.models.shard import Shard
            shard = Shard(
                shard_data.get("model_name", "model"),
                shard_data.get("start_layer", 0),
                shard_data.get("end_layer", 0),
                shard_data.get("total_layers", 0),
            )
        else:
            shard = payload.get("shard")
        tensor_bytes = payload.get("tensor_bytes", b"")
        # Run compute synchronously
        return client.compute_resp(shard, tensor_bytes, ring, idx)

    def schedule_tensor(self, payload: dict, prefer_peer_id: Optional[str] = None) -> dict:
        """Route a tensor job to an available peer, otherwise serialize on the local host."""
        with self._lock:
            ordered = self._ordered_peers()
            if prefer_peer_id in [p.peer_id for p in ordered]:
                target_id = prefer_peer_id
            elif ordered:
                target_id = ordered[self._rr_index % len(ordered)].peer_id
                self._rr_index += 1
            else:
                target_id = None
        if target_id:
            try:
                return self.request_remote_compute(target_id, payload)
            except Exception as exc:  # pragma: no cover - best-effort logging
                logger.warning("Remote dispatch failed for %s: %s", target_id, exc)
        with self._host_lock:
            while self._host_busy:
                self._host_lock.wait()
            self._host_busy = True
        try:
            return self._tensor_callback(payload)
        finally:
            with self._host_lock:
                self._host_busy = False
                self._host_lock.notify_all()

    def access_password_required(self) -> bool:
        return bool(self._access_password)

    def _detect_gpus(self) -> List[dict]:
        """Best-effort GPU inventory using torch if available."""
        gpus: List[dict] = []
        if torch is not None and hasattr(torch, "cuda") and torch.cuda.is_available():  # type: ignore[attr-defined]
            try:
                count = torch.cuda.device_count()
                for idx in range(count):
                    props = torch.cuda.get_device_properties(idx)
                    gpus.append(
                        {
                            "name": getattr(props, "name", f"cuda:{idx}"),
                            "total_mem_gb": round(getattr(props, "total_memory", 0) / (1024 ** 3), 2),
                            "compute": f"{getattr(props, 'multi_processor_count', 0)} SM",
                        }
                    )
            except Exception:
                pass
        if not gpus:
            gpus.append({"name": "CPU-only", "total_mem_gb": 0, "compute": "N/A"})
        return gpus

    # Misc ----------------------------------------------------------------------
    def _tensor_callback(self, payload: dict) -> dict:
        # Placeholder for actual tensor execution. schedule_tensor handles locking.
        return {
            "echo": payload,
            "notice": "Remote execution not implemented in this build.",
        }

    # Utilities -----------------------------------------------------------------
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

    def _ping_peer(self, host: str, port: int) -> dict:
        payload = json.dumps({"command": "ping"}).encode("utf-8")
        start = time.time()
        with socket.create_connection((host, port), timeout=3) as sock:
            sock.sendall(payload)
            sock.shutdown(socket.SHUT_WR)
            response = sock.recv(4096)
        rtt_ms = max((time.time() - start) * 1000.0, 0.0)
        try:
            data = json.loads(response.decode("utf-8"))
        except Exception as exc:
            raise RuntimeError(f"Invalid ping response from {host}:{port}: {exc}")
        data["rtt_ms"] = rtt_ms
        return data

    def _discover_local_udp(self) -> List[dict]:
        """Broadcast a ping over UDP and collect responding servers on LAN."""
        results: List[dict] = []
        payload = json.dumps({"command": "ping"}).encode("utf-8")
        try:
            sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_BROADCAST, 1)
            sock.settimeout(0.5)
            sock.sendto(payload, ("<broadcast>", self.port))
            while True:
                try:
                    data, addr = sock.recvfrom(4096)
                except socket.timeout:
                    break
                try:
                    info = json.loads(data.decode("utf-8"))
                    info["address"] = addr[0]
                    info.setdefault("port", self.port)
                    results.append(info)
                except Exception:
                    continue
        except Exception:
            return results
        finally:
            try:
                sock.close()
            except Exception:
                pass
        return results

    def _apply_server_peers(self, server_id: str, host: str, port: int, cached: Optional[list] = None) -> None:
        nodes = cached if cached is not None else self._list_peers(host, port)
        with self._lock:
            for entry in nodes:
                pid = entry.get("peer_id") or f"{server_id}:{len(self.peers)}"
                if pid == self.peer_id:
                    continue
                hw = entry.get("hardware", {}) if isinstance(entry, dict) else {}
                devices = entry.get("devices", []) if isinstance(entry, dict) else []
                self.peers[pid] = PeerInfo(
                    peer_id=pid,
                    address=host,
                    port=port,
                    devices=devices,
                    device_report=hw,
                    metadata={"server_id": server_id},
                    peer_type="client",
                )
                self._get_model_client(self.peers[pid])

    def _ensure_server_peer(self, server_id: str, host: str, port: int) -> None:
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
        self._get_model_client(self.peers[server_id])

    def _ordered_peers(self) -> List[PeerInfo]:
        """Return peers excluding self, sorted by capacity (VRAM/RAM/compute)."""
        peers = [p for pid, p in self.peers.items() if pid != self.peer_id]
        def capacity(peer: PeerInfo) -> float:
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
            return max(vram, ram, flops, 1.0)
        return sorted(peers, key=capacity, reverse=True)

    def _get_model_client(self, peer: PeerInfo) -> ModelClient:
        server_id = peer.metadata.get("server_id") if isinstance(peer.metadata, dict) else self.peer_id
        server_profile = self.server if self.server is not None else self.server_profile
        key = server_profile.server_id
        client = self._model_clients.get(key)
        if client is None:
            client = ModelClient(server_profile)
            self._model_clients[key] = client
        return client

    def _list_peers(self, host: str, port: int) -> list:
        payload = json.dumps({"command": "list_peers"}).encode("utf-8")
        with socket.create_connection((host, port), timeout=3) as sock:
            sock.sendall(payload)
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
