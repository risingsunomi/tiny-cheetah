from __future__ import annotations

import json
import socket
import threading
import time
from typing import Callable, Optional


class DiscoveryService:
    """Broadcasts and listens for LAN peers via UDP."""

    def __init__(
        self,
        username: str,
        fingerprint: str,
        port: int = 5252,
        interval: float = 5.0,
    ) -> None:
        self.username = username
        self.fingerprint = fingerprint
        self.port = port
        self.interval = interval
        self._listener: Optional[threading.Thread] = None
        self._announcer: Optional[threading.Thread] = None
        self._stop = threading.Event()
        self._callback: Optional[Callable[[dict], None]] = None

    def start(self, callback: Callable[[dict], None]) -> None:
        self._callback = callback
        self._stop.clear()
        self._listener = threading.Thread(target=self._listen_loop, name="lan-discovery", daemon=True)
        self._listener.start()
        self._announcer = threading.Thread(target=self._announce_loop, name="lan-announce", daemon=True)
        self._announcer.start()

    def stop(self) -> None:
        self._stop.set()

    def _listen_loop(self) -> None:
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        sock.bind(("", self.port))
        sock.settimeout(1.0)
        while not self._stop.is_set():
            try:
                data, addr = sock.recvfrom(4096)
            except socket.timeout:
                continue
            try:
                info = json.loads(data.decode("utf-8"))
                info["address"] = addr[0]
            except json.JSONDecodeError:
                continue
            if info.get("fingerprint") == self.fingerprint:
                continue
            if self._callback is not None:
                self._callback(info)
        sock.close()

    def _announce_loop(self) -> None:
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_BROADCAST, 1)
        payload = json.dumps(
            {
                "fingerprint": self.fingerprint,
                "username": self.username,
            }
        ).encode("utf-8")
        while not self._stop.is_set():
            sock.sendto(payload, ("<broadcast>", self.port))
            time.sleep(self.interval)
        sock.close()
