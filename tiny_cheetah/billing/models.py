from __future__ import annotations

import time
from dataclasses import dataclass


@dataclass
class BillingProfile:
    description: str = "General compute"
    flops_gflops: float = 0.0
    gpu_description: str = ""
    ping_ms: float = 0.0
    motd: str = "Welcome to my host"

