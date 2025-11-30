from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Dict


@dataclass
class PeerInfo:
    node_id: str
    username: str
    pgp_key: str
    address: str
    port: int
    devices: List[str] = field(default_factory=list)
    flops_gflops: float = 0.0
    gpu_description: str = ""
    ping_ms: float = 0.0
    offer_description: str = ""
    payment_instructions: str = ""
    motd: str = ""
    available: bool = True
    latency_ms: float = 0.0
    metadata: Dict[str, str] = field(default_factory=dict)
    device_report: Dict[str, object] = field(default_factory=dict)
