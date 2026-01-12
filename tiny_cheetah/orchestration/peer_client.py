from __future__ import annotations

import base64
import json
import os
import socket
import threading
import time
import uuid
from typing import Any, Dict, List, Optional, Tuple

from tiny_cheetah.logging_utils import get_logger
from tiny_cheetah.models.shard import Shard
from tiny_cheetah.orchestration.model_engine import ModelEngine
from tiny_cheetah.orchestration.cdevice import CDevice

logger = get_logger(__name__)


class PeerClient:
    """Network client for peer discovery and tensor exchange."""

    def __init__(self) -> None:
        self.peer_client_id = os.getenv("TC_PEER_ID") or f"cheetah-{uuid.uuid4().hex[:6]}"
        self.address = "0.0.0.0"
        self.port = int(os.getenv("TC_PORT", "8765"))
        self.peer_info = CDevice(
            self.peer_client_id,
            self.address,
            self.port,
            os.getenv("TC_DEVICE", "CPU")
        )
        self.in_use = False
        self.peers: Dict[str, CDevice] = {self.peer_client_id: self.peer_info}
        self._lock = threading.RLock()
        self._udp_thread: Optional[threading.Thread] = None
        self._udp_stop = threading.Event()
        self._start_udp_responder()

    # Networking ---------------------------------------------------------
    def send_tensor_bytes(
        self,
        tensor_bytes: bytes,
        address: Tuple[str, int] | None = None
    ) -> None:
        """Send raw tensor bytes to a peer; no response expected."""
        payload = {
            "command": "tensor_bytes",
            "payload": {"buffer": base64.b64encode(tensor_bytes).decode("ascii")}
        }
        logger.info(f"Sending tensor bytes from peer '{self.peer_client_id}' to {address or (self.address, self.port)}")
        self._send(payload, expect_reply=False, address=address)

    def recv_tensor_bytes(
        self,
        timeout: float = 5.0,
        bind_address: Tuple[str, int] | None = None,
    ) -> bytes:
        """Blocking receive helper; expects peer to initiate a send."""
        try:
            if not self.in_use:
                with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as sock:
                    sock.bind(bind_address or ("0.0.0.0", 0))
                    sock.settimeout(timeout)
                    data, _ = sock.recvfrom(65536)
                    msg = json.loads(data.decode("utf-8"))
                buf = msg.get("payload", {}).get("buffer", "")
                self.in_use = True
                return base64.b64decode(buf)
            else:
                logger.warning(f"recv_tensor_bytes called '{self.peer_client_id}' already in use")
                logger.info("Try again later")
                return b""
        except Exception:
            return b""

    def ping(self, address: Tuple[str, int] | None = None) -> Dict[str, Any]:
        try:
            return self._send({"command": "ping"}, expect_reply=True, address=address)
        except Exception as exc:
            return {"error": str(exc)}

    def send_payload(self, message: dict, *, expect_reply: bool = True, address: Tuple[str, int] | None = None) -> Dict[str, Any]:
        return self._send(message, expect_reply=expect_reply, address=address)

    def _send(self, message: dict, expect_reply: bool = True, address: Tuple[str, int] | None = None) -> Dict[str, Any]:
        data = json.dumps(message).encode("utf-8")
        if address is not None:
            host, port = address
        else:
            host, port = self.address, self.port
            logger.warning(f"Using default address {host}:{port}")

        with socket.create_connection((host, port), timeout=3) as sock:
            sock.sendall(data)
            if not expect_reply:
                return {}
            sock.shutdown(socket.SHUT_WR)
            response = sock.recv(65536)
        try:
            return json.loads(response.decode("utf-8"))
        except Exception as err:
            logger.error("Invalid response from peer %s: %s", self.peer_client_id, err)
            raise

    def _resolve_address(self, address: Tuple[str, int] | None) -> Tuple[str, int]:
        if address is not None:
            return address
        return self.address, self.port

    # Discovery -----------------------------------------------------------
    def discover_peers(self) -> None:
        self.peers = {self.peer_client_id: self.peer_info}

        # UDP broadcast
        for info in self._discover_local_udp():
            peer = self._build_peer_from_info(info)
            if peer is None or peer.peer_client_id == self.peer_client_id:
                continue
            self.peers[peer.peer_client_id] = peer

    # Connections ---------------------------------------------------------
    def get_peers(self, include_self: bool = False) -> List[CDevice]:
        if include_self:
            return list(self.peers.values())
        return [p for pid, p in self.peers.items() if pid != self.peer_client_id]

    def peer_count(self) -> int:
        return len(self.peers)

    def assign_shards(self, model_name: str, total_layers: int) -> None:
        ModelEngine.plan_shards(self._ordered_peers(), model_name, total_layers)

    # Internal ------------------------------------------------------------
    def _ordered_peers(self) -> List[CDevice]:
        peers = [p for p in self.peers.values() if p.peer_client_id != self.peer_client_id]

        def capacity(peer: CDevice) -> float:
            vram = _to_float(peer.gpu_vram)
            ram = _to_float(peer.cpu_ram)
            return max(vram, ram, 1.0)

        return sorted(peers, key=capacity, reverse=True)

    def _build_peer_from_info(self, info: dict) -> Optional[CDevice]:
        peer_client_id = info.get("peer_client_id")
        if not peer_client_id:
            return None

        shard_info = info.get("shard", {}) if isinstance(info.get("shard"), dict) else {}
        shard = Shard(
            shard_info.get("model_name", ""),
            int(shard_info.get("start_layer", 0) or 0),
            int(shard_info.get("end_layer", 0) or 0),
            int(shard_info.get("total_layers", shard_info.get("end_layer", 0)) or 0),
        )
        peer = CDevice(
            peer_client_id,
            info.get("ip_address", info.get("address", "")),
            int(info.get("port", self.port)),
            shard,
            in_use=False,
            tg_device=str(info.get("tg_device", "CPU")),
        )
        peer.cpu_model = str(info.get("cpu_model", ""))
        peer.gpu_model = str(info.get("gpu_model", ""))
        peer.gpu_vram = str(info.get("gpu_vram", ""))
        peer.gpu_flops = float(info.get("gpu_flops", 0.0) or 0.0)
        return peer

    def _discover_local_udp(self) -> List[dict]:
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
                    peer_info = json.loads(data.decode("utf-8"))
                    self._build_peer_from_info(peer_info)
                    results.append(peer_info)
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

    def _ping_peer_udp(self, peer: CDevice) -> dict:
        payload = json.dumps({"command": "ping"}).encode("utf-8")
        start = time.time()
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as sock:
            sock.settimeout(1.0)
            sock.sendto(payload, (peer.ip_address, peer.port))
            data, _ = sock.recvfrom(4096)
        ping_ms = max((time.time() - start) * 1000.0, 0.0)
        try:
            data_msg = json.loads(data.decode("utf-8"))
            if data_msg.get("command") == "pong":
                peer.ping_ms = ping_ms
        except Exception as exc:
            raise RuntimeError(f"Invalid ping response from {peer.ip_address}:{peer.port}: {exc}")

    # UDP responder -------------------------------------------------------
    def _start_udp_responder(self) -> None:
        if self._udp_thread and self._udp_thread.is_alive():
            return
        self._udp_stop.clear()
        self._udp_thread = threading.Thread(target=self._udp_loop, name="udp-responder", daemon=True)
        self._udp_thread.start()

    def _udp_loop(self) -> None:
        try:
            sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            sock.bind(("0.0.0.0", self.port))
        except Exception:
            return
        while not self._udp_stop.is_set():
            try:
                sock.settimeout(0.5)
                data, addr = sock.recvfrom(4096)
            except socket.timeout:
                continue
            except Exception:
                break
            try:
                msg = json.loads(data.decode("utf-8"))
            except Exception:
                continue
            if msg.get("command") != "ping":
                continue
            response = json.dumps(self.peer_info.as_dict()).encode("utf-8")
            try:
                sock.sendto(response, addr)
            except Exception:
                continue
        try:
            sock.close()
        except Exception:
            pass

def _to_float(val: Any) -> float:
    try:
        if isinstance(val, str):
            txt = val.lower().replace("gb", "").strip()
            return float(txt)
        return float(val)
    except Exception:
        return 0.0
