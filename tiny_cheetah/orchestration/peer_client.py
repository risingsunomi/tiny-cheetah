from __future__ import annotations

import base64
import json
import os
import socket
import threading
import time
import uuid
from typing import Any, Dict, List, Optional, Tuple
from contextlib import closing
import asyncio

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
        self.shard = Shard("", 0, 0, 0)
        self.in_use = False
        self.stop_ping: bool = False
        self.stop_udp_discovery = False
        self.stop_udp_broadcast = False
        self._peers: Dict[str, PeerClient] = {self.peer_client_id: self.peer_info}
        self._lock = threading.RLock()
        self._thread_ping: Optional[threading.Thread] = None        
        self._thread_udp_discovery: Optional[threading.Thread] = None        
        self._thread_udp_brodcast: Optional[threading.Thread] = None
        

        self._run_udp_response()
        self._run_udp_discover()

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
        self.in_use = True
        try:
            if not self.in_use:
                with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as sock:
                    sock.bind(bind_address or ("0.0.0.0", 0))
                    sock.settimeout(timeout)
                    data, _ = sock.recvfrom(65536)
                    msg = json.loads(data.decode("utf-8"))
                buf = msg.get("payload", {}).get("buffer", "")
                self.in_use = False
                return base64.b64decode(buf)
            else:
                logger.warning(f"recv_tensor_bytes called '{self.peer_client_id}' already in use")
                logger.info("Try again later")        
        except Exception:
            logger.exception("Error receiving tensor bytes")
        
        self.in_use = False
        return b""
    
    def ping(self, address: Tuple[str, int] | None = None) -> Dict[str, Any]:
        try:
            return self._send({"command": "ping"}, expect_reply=True, address=address)
        except Exception as exc:
            logger.error(f"Ping failed: {exc}")

        return {}

    def send_payload(self, message: dict, *, expect_reply: bool = True, address: Tuple[str, int] | None = None) -> Dict[str, Any]:
        return self._send(message, expect_reply=expect_reply, address=address)

    def _send(self, message: dict, expect_reply: bool = True, address: Tuple[str, int] | None = None) -> Dict[str, Any]:
        self.in_use = True
        try:
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
            
                self.in_use = False
                return json.loads(response.decode("utf-8"))
        except Exception as err:
            self.in_use = False
            logger.error("Invalid response from peer %s: %s", self.peer_client_id, err)
            raise

    def _resolve_address(self, address: Tuple[str, int] | None) -> Tuple[str, int]:
        if address is not None:
            return address
        return self.address, self.port

    # Discovery -----------------------------------------------------------
    def discover_peers(self) -> None:
        self._peers = {self.peer_client_id: self.peer_info}

        # UDP broadcast
        for info in self._udp_discover():
            peer = self._build_peer_from_info(info)
            if peer is None or peer.peer_client_id == self.peer_client_id:
                continue
            self._peers[peer.peer_client_id] = peer

    # Connections ---------------------------------------------------------
    def get_peers(self, include_self: bool = False) -> List[CDevice]:
        if include_self:
            return list(self._peers.values())
        return [p for pid, p in self._peers.items() if pid != self.peer_client_id]

    def peer_count(self) -> int:
        return len(self._peers)

    def assign_shards(self, model_name: str, total_layers: int) -> None:
        ModelEngine.plan_shards(self._ordered_peers(), model_name, total_layers)

    # Internal ------------------------------------------------------------
    def _ordered_peers(self) -> List[CDevice]:
        peers = [p for p in self._peers.values() if p.peer_client_id != self.peer_client_id]

        def capacity(peer: CDevice) -> float:
            vram = _to_float(peer.gpu_vram)
            ram = _to_float(peer.cpu_ram)
            return max(vram, ram, 1.0)

        return sorted(peers, key=capacity, reverse=True)

    def _build_peer_from_info(self, info: dict) -> Optional[PeerClient]:
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

        return PeerClient(peer)

    # Ping handling
    # --------------------------------
    def _ping_peer(self, peer_client: PeerClient) -> dict:
        missed_pong = {}
        while not self.stop_ping:
            try:
                for _, peer_client in self._peers:
                    payload = json.dumps({"command": "ping"}).encode("utf-8")
                    start = time.time()
                    with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as sock:
                        sock.settimeout(1.0)
                        sock.sendto(payload, (peer_client.ip_address, peer_client.port))
                        data, _ = sock.recvfrom(4096)
                    ping_ms = max((time.time() - start) * 1000.0, 0.0)

                    data_msg = json.loads(data.decode("utf-8"))
                    if data_msg.get("command") == "pong":
                        peer_client.ping_ms = ping_ms
                    else:
                        peer_client_id = peer_client.peer_client_id
                        if peer_client_id not in missed_pong.keys():
                            missed_pong[peer_client_id] = 0
                        else:
                            missed_pong[peer_client_id] += 1

                        missed_pong_limit = os.getenv("TC_PONG_MISS_LIMIT", 5)
                        if missed_pong[peer_client_id] == missed_pong_limit:
                            logger.info(f"Missing pong from {peer_client_id} {missed_pong_limit} times, dropping from peer list")
                            self._peers = {key: value for key, value in self._peers.items() if key != peer_client_id}
                            missed_pong = {key: value for key, value in self._peers.items() if key != peer_client_id}

            except Exception:
                continue
    
    def _run_ping(self)-> None:
        if self._thread_ping and self._thread_ping.is_alive():
            return
        self._thread_ping = threading.Thread(target=self._ping_loop, name="ping-peer", daemon=True)
        self._thread_ping.start()


    # UDP handling
    #-------------------------------------------------------
    def _udp_discover(self):
        payload = json.dumps(
            {
                "command": "D001",
                "peer_client_id": self.peer_client_id
        }).encode("utf-8")
        unicast_targets = os.getenv("TC_PEER_UNICAST_TARGETS", "")
        targets: list[tuple[str, int]] = []
        if unicast_targets:
            for entry in unicast_targets.split(","):
                entry = entry.strip()
                if not entry:
                    continue
                if ":" in entry:
                    host, port_str = entry.rsplit(":", 1)
                    try:
                        port = int(port_str)
                    except ValueError:
                        logger.warning("Invalid unicast target port: %s", entry)
                        continue
                else:
                    host = entry
                    port = self.port
                targets.append((host, port))
        try:
            sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_BROADCAST, 1)
            sock.settimeout(0.1)
        except Exception as err:
            logger.error(f"Error discovering local UDP peers {err}")
            raise

        with closing(sock):
            while not self.stop_udp_discovery:
                logger.info(f"Discovering peers @ {self.port}")
                try:
                    sock.sendto(payload, ("<broadcast>", self.port))
                    for host, port in targets:
                        sock.sendto(payload, (host, port))
                except Exception as err:
                    logger.error(f"Error broadcasting discovery packet: {err}")
                    time.sleep(1.0)
                    continue

                while not self.stop_udp_discovery:
                    try:
                        data, addr = sock.recvfrom(4096)
                    except socket.timeout:
                        break
                    
                    try:
                        udp_peer_info = json.loads(data.decode("utf-8"))
                        if udp_peer_info["command"] == "D002":
                            udp_client_id = udp_peer_info.get("peer_client_id")
                            if udp_client_id is not None:
                                if udp_client_id != self.peer_client_id and udp_client_id not in self._peers.keys():
                                    logger.info("UDP discovery response from %s: %s", addr, udp_peer_info)
                                    logger.info(f"New peer discovered @ {addr}.")
                                    logger.info(f"Adding peer {udp_client_id}")
                                    self.add_peer(udp_peer_info)
                    except Exception:
                        continue
                
                time.sleep(1.0)

    def _run_udp_discover(self) -> None:
        if self._thread_udp_discovery and self._thread_udp_discovery.is_alive():
            return
        self._thread_udp_discovery = threading.Thread(target=self._udp_discover, name="udp-discover", daemon=True)
        self._thread_udp_discovery.start()

    def _udp_response(self) -> None:
        try:
            sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            sock.bind(("0.0.0.0", self.port))
            sock.settimeout(0.5)
        except Exception as err:
            logger.error(f"Failed UDP response: {err}")
            raise

        with closing(sock):
            while not self.stop_udp_discovery:
                logger.info(f"Broadcasting client {self.peer_client_id} @ {self.address}:{self.port}")
                try:
                    data, addr = sock.recvfrom(4096)
                    msg = json.loads(data.decode("utf-8"))
                    if msg.get("command") == "D001" and msg.get("peer_client_id") != self.peer_client_id:
                        logger.info("UDP discovery request from %s: %s", addr, msg)
                        try:
                            response = self.as_dict()
                            response["command"] = "D002"
                            resp_data = json.dumps(response).encode("utf-8")
                            logger.info(f"Responding to D001 from {addr}")
                            sock.sendto(resp_data, (addr[0], self.port))
                        except Exception as err:
                            logger.error(f"Error responding to request: {err}")
                            continue
                        
                except Exception as err:
                    continue

    def _run_udp_response(self) -> None:
        if self._thread_udp_brodcast and self._thread_udp_brodcast.is_alive():
            return
        self._thread_udp_brodcast = threading.Thread(target=self._udp_response, name="udp-responder", daemon=True)
        self._thread_udp_brodcast.start()

    def add_peer(self, peer_data: dict) -> None:
        cdvice = CDevice(
            peer_data.get("peer_client_id", None),
            peer_data.get("ip_address", "0.0.0.0"),
            peer_data.get("port", self.port),
            peer_data.get("tg_device", "CPU")
        )

        peer_client = PeerClient(
            peer_data.get("peer_client_id"),
            peer_data.get("ip_address", "0.0.0.0"),
            peer_data.get("port", self.port),
            cdvice,
            Shard(
                peer_data["shard"].get("model_name"),
                peer_data["shard"].get("start_layer"),
                peer_data["shard"].get("end_layer")
            ),
        )
        self._peers[peer_client.peer_client_id] = peer_client

    def as_dict(self) -> dict:
        return {
            "peer_client_id": self.peer_client_id,
            "ip_address": self.ip_address,
            "port": self.port,
            "tg_device": self.tg_device,
            "shard": {
                "model_name": self.shard.model_name,
                "start_layer": self.shard.start_layer,
                "end_layer": self.shard.end_layer,
            },
            "peer_info": self.peer_info.as_dict()
        }

def _to_float(val: Any) -> float:
    try:
        if isinstance(val, str):
            txt = val.lower().replace("gb", "").strip()
            return float(txt)
        return float(val)
    except Exception:
        return 0.0
