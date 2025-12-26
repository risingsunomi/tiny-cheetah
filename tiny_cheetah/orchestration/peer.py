from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Dict


@dataclass
class PeerInfo:
    peer_id: str
    address: str
    port: int
    devices: List[str] = field(default_factory=list)
    flops_gflops: float = 0.0
    gpu_description: str = ""
    ping_ms: float = 0.0
    available: bool = True
    latency_ms: float = 0.0
    metadata: Dict[str, str] = field(default_factory=dict)
    device_report: Dict[str, object] = field(default_factory=dict)
    peer_type: str = "local"  # e.g., "local", "server"

    @property
    def as_dict(self) -> dict:
        return {
            "peer_id": self.peer_id,
            "address": self.address,
            "port": self.port,
            "devices": self.devices,
            "flops_gflops": self.flops_gflops,
            "gpu_description": self.gpu_description,
            "ping_ms": self.ping_ms,
            "available": self.available,
            "latency_ms": self.latency_ms,
            "metadata": self.metadata,
            "device_report": self.device_report,
        }
