from __future__ import annotations

import json
import socket
import threading
from typing import Dict, List, Optional, Sequence, Tuple

from tiny_cheetah.orchestration.peer import PeerInfo
from tiny_cheetah.orchestration.server import PeerServer, ServerProfile
from tiny_cheetah.orchestration.device_info import collect_host_info


class ServerManager:
    """Owns local hosting lifecycle, discovery helpers, and capacity ordering."""

    def __init__(self, peer_id: str, address: str, port: int) -> None:
        self.peer_id = peer_id
        self.address = address
        self.port = port
        self.server_profile = ServerProfile(server_id=peer_id, address=(address, port))
        self._host_info = collect_host_info()
        self._gpu_info = [
            d for d in self._host_info.get("devices", []) if isinstance(d, dict) and d.get("device") == "GPU"
        ]
        self._server: Optional[PeerServer] = None
        self._access_password: str = ""
        self._lock = threading.RLock()

    # Hosting -------------------------------------------------------------
    def start_hosting(self, host: str, port: int, password: str = "", tensor_callback=None) -> None:
        if self._server is not None:
            return
        self._access_password = password.strip()
        self.server_profile.server_id = self.peer_id
        self.server_profile.address = (host, port)
        self._server = PeerServer(
            (host, port),
            self.server_profile,
            self._host_info,
            self._local_devices(),
            tensor_callback or (lambda payload: {"echo": payload}),
            self._access_password,
        )
        self._server.start()

    def stop_hosting(self) -> None:
        if self._server is not None:
            self._server.stop()
            self._server = None

    def is_hosting(self) -> bool:
        return self._server is not None

    def access_password_required(self) -> bool:
        return bool(self._access_password)

    # Discovery -----------------------------------------------------------
    def discover_local_udp(self, port: int) -> List[dict]:
        """Broadcast a ping over UDP and collect responding servers on LAN."""
        results: List[dict] = []
        payload = json.dumps({"command": "ping"}).encode("utf-8")
        try:
            sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_BROADCAST, 1)
            sock.settimeout(0.5)
            sock.sendto(payload, ("<broadcast>", port))
            while True:
                try:
                    data, addr = sock.recvfrom(4096)
                except socket.timeout:
                    break
                try:
                    info = json.loads(data.decode("utf-8"))
                    info["address"] = addr[0]
                    info.setdefault("port", port)
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

    # Ordering ------------------------------------------------------------
    @staticmethod
    def ordered_peers(peers: Sequence[PeerInfo], self_id: str) -> List[PeerInfo]:
        """Return peers excluding self, sorted by capacity (VRAM/RAM/FLOPS)."""
        filtered = [p for p in peers if p.peer_id != self_id]

        def capacity(peer: PeerInfo) -> float:
            hw = peer.device_report or {}
            devices = hw.get("devices", []) or hw.get("gpus", []) or []
            gpu_entry = next((d for d in devices if isinstance(d, dict) and d.get("device") == "GPU"), None)
            vram = 0.0
            if gpu_entry:
                try:
                    vram = float(gpu_entry.get("ram_gb", gpu_entry.get("total_mem_gb", 0.0)) or 0.0)
                except Exception:
                    vram = 0.0
            ram = 0.0
            if isinstance(hw, dict):
                try:
                    ram = float(hw.get("ram_gb", 0.0))
                except Exception:
                    ram = 0.0
            if ram == 0.0 and isinstance(devices, list):
                cpu_entry = next((d for d in devices if isinstance(d, dict) and d.get("device") == "CPU"), None)
                if cpu_entry:
                    try:
                        ram = float(cpu_entry.get("ram_gb", 0.0))
                    except Exception:
                        ram = 0.0
            flops = getattr(peer, "flops_gflops", 0.0) if hasattr(peer, "flops_gflops") else 0.0
            return max(vram, ram, flops, 1.0)

        return sorted(filtered, key=capacity, reverse=True)

    # Info ---------------------------------------------------------------
    def get_host_info(self) -> dict:
        return dict(self._host_info)

    def get_gpu_inventory(self) -> List[dict]:
        return list(self._gpu_info)

    def hosted_clients(self) -> List[PeerInfo]:
        if self._server is None:
            return []
        return self._server.get_connected_peers()

    # Internal -----------------------------------------------------------
    def _local_devices(self) -> List[str]:
        if not isinstance(self._host_info, dict):
            return ["local-host"]
        devices = self._host_info.get("devices", []) or []
        names = [d.get("name", "device") for d in devices if isinstance(d, dict) and d.get("name")]
        return names or ["local-host"]
