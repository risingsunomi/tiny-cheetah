from __future__ import annotations

import socket
import json
import threading
import uuid
import time
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
        self._lock = threading.RLock()
        self._host_lock = threading.Condition(self._lock)
        self._host_busy = False
        self._server: Optional[PeerServer] = None
        self._access_password: str = ""
        self._server_targets = self._parse_server_env()
        self._host_info = collect_host_info()
        self._gpu_info = self._detect_gpus()
        self.server_profile = ServerProfile()

    # Peer Discovery -----------------------------------------------------------
    def discover_peers(self) -> None:
        with self._lock:
            self.peers.setdefault(self.peer_id, self.peer_info)
        for host, port in self._server_targets:
            try:
                info = self._ping_peer(host, port)
            except Exception as exc:
                logger.debug("Server %s:%s unreachable: %s", host, port, exc)
                continue
            peer_id = info.get("peer_id") or f"{host}:{port}"
            if peer_id == self.peer_id:
                continue
            ping_ms = float(info.get("rtt_ms", 0.0))
            with self._lock:
                self.peers[peer_id] = PeerInfo(
                    peer_id=peer_id,
                    address=host,
                    port=port,
                    devices=[],
                    ping_ms=ping_ms,
                    metadata={"server": "1"},
                    device_report={},
                    peer_type="server",
                )
        if self._server is not None:
            for hosted in self._server.get_connected_peers():
                if hosted.peer_id == self.peer_id:
                    continue
                self.peers[hosted.peer_id] = hosted
        
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
        self._server.start()
        

    def stop_hosting(self) -> None:
        if self._server is not None:
            self._server.stop()
            self._server = None
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
        

    def connect_to_peer(self, host: str, port: int, password: str = "") -> PeerInfo:
        """Explicitly connect to a server/peer and record it."""
        info = self._ping_peer(host, port)
        peer_id = info.get("peer_id") or f"{host}:{port}"
        peers_on_server = self._list_peers(host, port)
        ping_ms = float(info.get("rtt_ms", 0.0))
        peer = PeerInfo(
            peer_id=peer_id,
            address=host,
            port=port,
            devices=[],
            ping_ms=ping_ms,
            metadata={"password": password, "peers": peers_on_server},
            peer_type="server",
        )
        with self._lock:
            self.peers[peer_id] = peer
        return peer

    def request_remote_compute(self, node_id: str, payload: dict) -> dict:
        peer = self.peers.get(node_id)
        if peer is None:
            raise RuntimeError("Unknown peer")
        client = ModelClient(peer, password=peer.metadata.get("password", ""))
        return client.dispatch_tensor(payload)

    def schedule_tensor(self, payload: dict, prefer_peer_id: Optional[str] = None) -> dict:
        """Route a tensor job to an available peer, otherwise serialize on the local host."""
        with self._lock:
            candidates = [pid for pid in self.peers if pid != self.peer_id]
            target = prefer_peer_id if prefer_peer_id in candidates else (candidates[0] if candidates else None)
        if target:
            try:
                return self.request_remote_compute(target, payload)
            except Exception as exc:  # pragma: no cover - best-effort logging
                logger.warning("Remote dispatch failed for %s: %s", target, exc)
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
