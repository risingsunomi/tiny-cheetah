from __future__ import annotations

from dataclasses import dataclass
from typing import Dict


@dataclass
class TrainingNode:
    name: str
    status: str = "pending"
    repeated: bool = False


NODE_STATUS_STYLES: Dict[str, str] = {
    "pending": "#5c7080",
    "running": "#3bd6ee",
    "complete": "#00ff00",
    "stopped": "#ffaa00",
}

NODE_STATUS_SYMBOLS: Dict[str, str] = {
    "pending": "○",
    "running": "●",
    "complete": "✔",
    "stopped": "■",
}
