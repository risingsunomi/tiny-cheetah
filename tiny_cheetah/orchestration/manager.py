from __future__ import annotations

import threading
import time
import uuid
from typing import Callable, Dict, List, Optional

try:
    import torch  # type: ignore
except Exception:  # pragma: no cover - optional
    torch = None

from tiny_cheetah.billing import BillingProfile
from tiny_cheetah.user_management.identity import generate_identity
from .device_info import collect_host_info
from .discovery import DiscoveryService
from .peer import PeerInfo
from .server import PeerServer
from .client import PeerClient


class PeerManager:
    """Coordinates hosting, discovery, and remote connections."""

    def __init__(self) -> None:
        self.identity = generate_identity(username=f"cheetah-{uuid.uuid4().hex[:6]}")
        self.billing_profile = BillingProfile()
        self._peers: Dict[str, PeerInfo] = {}
        self._listeners: List[Callable[[], None]] = []
        self._lock = threading.RLock()
        self._host_lock = threading.Condition(self._lock)
        self._host_busy = False
        self._server: Optional[PeerServer] = None
        self._access_password: str = ""
        self._discovery = DiscoveryService(
            self.identity["username"],
            self.identity["fingerprint"],
        )
        self._discovery.start(self._on_discovered_peer)
        self._host_info = collect_host_info()
        self._gpu_info = self._detect_gpus()

    # Listener hooks -------------------------------------------------
    def add_listener(self, callback: Callable[[], None]) -> None:
        self._listeners.append(callback)

    def _notify(self) -> None:
        for callback in list(self._listeners):
            try:
                callback()
            except Exception:
                continue

    # Hosting ------------------------------------------------------------------
    def start_hosting(self, host: str = "0.0.0.0", port: int = 8765, password: str = "") -> None:
        if self._server is not None:
            return
        self._access_password = password.strip()
        self._server = PeerServer((host, port), self._identity_payload(), self._tensor_callback, self._access_password)
        self._server.start()
        self._notify()

    def stop_hosting(self) -> None:
        if self._server is not None:
            self._server.stop()
            self._server = None
            self._notify()

    # Connections ---------------------------------------------------------------
    def connect_to_peer(self, address: str, port: int, devices: Optional[List[str]] = None, password: str = "") -> PeerInfo:
        client = PeerClient(address, port, self.identity, password=password)
        identity = client.handshake()
        offer = identity.get("offer", {})
        hardware = identity.get("hardware", {})
        with self._lock:
            peer = PeerInfo(
                node_id=identity.get("fingerprint", uuid.uuid4().hex),
                username=identity.get("username", "unknown"),
                pgp_key=identity.get("pgp_public", ""),
                address=address,
                port=port,
                devices=devices or identity.get("devices", []),
                flops_gflops=float(offer.get("flops_gflops", 0.0)),
                gpu_description=offer.get("gpu_description", ""),
                ping_ms=float(offer.get("ping_ms", 0.0)),
                offer_description=offer.get("description", ""),
                motd=offer.get("motd", ""),
                device_report=hardware,
            )
            if password:
                peer.metadata["password"] = password
            self._peers[peer.node_id] = peer
        self._notify()
        return peer

    def disconnect_peer(self, node_id: str) -> None:
        with self._lock:
            self._peers.pop(node_id, None)
        self._notify()

    def list_peers(self) -> List[PeerInfo]:
        with self._lock:
            return list(self._peers.values())

    def peer_count(self) -> int:
        with self._lock:
            return len(self._peers) + (1 if self._server else 0)

    def aggregate_devices(self) -> List[str]:
        devices: List[str] = []
        with self._lock:
            if self._server is not None:
                devices.append("local-host")
            for peer in self._peers.values():
                devices.extend(peer.devices)
        return devices

    def _local_devices(self) -> List[str]:
        """Return only devices attached to this host."""
        return ["local-host"] if self._server is not None else []

    def get_billing_profile(self) -> BillingProfile:
        return self.billing_profile

    def get_gpu_inventory(self) -> List[dict]:
        return list(self._gpu_info)

    def get_host_info(self) -> dict:
        """Return CPU/RAM/GPU/TC_DEVICE info for this host."""
        return dict(self._host_info)

    def update_billing_profile(
        self,
        *,
        description: str,
        flops_gflops: float,
        gpu_description: str,
        ping_ms: float,
        motd: str,
    ) -> None:
        self.billing_profile.description = description
        self.billing_profile.flops_gflops = flops_gflops
        self.billing_profile.gpu_description = gpu_description
        self.billing_profile.ping_ms = ping_ms
        self.billing_profile.motd = motd
        if self._server is not None:
            self._server.identity = self._identity_payload()
        self._notify()

    def request_remote_compute(self, node_id: str, payload: dict) -> dict:
        peer = self._peers.get(node_id)
        if peer is None:
            raise RuntimeError("Unknown peer")
        password = peer.metadata.get("password", "")
        client = PeerClient(peer.address, peer.port, self.identity, password=password)
        return client.dispatch_tensor(payload)

    def schedule_tensor(self, payload: dict, prefer_peer_id: Optional[str] = None) -> dict:
        """Route a tensor job to an available peer, otherwise wait for local availability."""
        target = prefer_peer_id or next(iter(self._peers.keys()), None)
        if target:
            try:
                return self.request_remote_compute(target, payload)
            except Exception:
                pass
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

    def _identity_payload(self) -> dict:
        payload = dict(self.identity)
        payload["offer"] = {
            "description": self.billing_profile.description,
            "flops_gflops": self.billing_profile.flops_gflops,
            "gpu_description": self.billing_profile.gpu_description,
            "ping_ms": self.billing_profile.ping_ms,
            "motd": self.billing_profile.motd,
        }
        payload["devices"] = self._local_devices()
        payload["hardware"] = self._host_info
        return payload

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

    # Discovery -----------------------------------------------------------------
    def _on_discovered_peer(self, info: dict) -> None:
        address = info.get("address")
        if not address:
            return
        fingerprint = info.get("fingerprint")
        if fingerprint in self._peers:
            return
        peer = PeerInfo(
            node_id=fingerprint,
            username=info.get("username", "lan-peer"),
            pgp_key="",
            address=address,
            port=info.get("port", 8765),
            devices=["LAN CPU"],
            metadata={"discovered": "1"},
        )
        with self._lock:
            self._peers.setdefault(peer.node_id, peer)
        self._notify()

    # Misc ----------------------------------------------------------------------
    def _tensor_callback(self, payload: dict) -> dict:
        with self._host_lock:
            while self._host_busy:
                self._host_lock.wait()
            self._host_busy = True
        try:
            # Placeholder for actual tensor execution.
            return {
                "echo": payload,
                "notice": "Remote execution not implemented in this build.",
            }
        finally:
            with self._host_lock:
                self._host_busy = False
                self._host_lock.notify_all()


_MANAGER: Optional[PeerManager] = None


def get_peer_manager() -> PeerManager:
    global _MANAGER
    if _MANAGER is None:
        _MANAGER = PeerManager()
    return _MANAGER
