from __future__ import annotations

import logging
import os
from datetime import datetime
from pathlib import Path

_configured = False
_log_file: Path | None = None


def _log_path() -> Path:
    base_env = os.getenv("TC_LOG_DIR", "").strip()
    base_dir = Path(base_env) if base_env else Path(__file__).resolve().parent.parent / "logs"
    base_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    return base_dir / f"tiny_cheetah_{ts}.log"


def configure_logging() -> Path:
    """Configure a shared file handler for the process."""
    global _configured, _log_file
    if _configured and _log_file is not None:
        return _log_file
    _log_file = _log_path()
    handler = logging.FileHandler(_log_file)
    handler.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s"))
    root = logging.getLogger()
    root.addHandler(handler)
    # Keep console verbosity as-is; ensure file captures info and above.
    if root.level == logging.NOTSET or root.level > logging.INFO:
        root.setLevel(logging.INFO)
    _configured = True
    return _log_file


def get_logger(name: str) -> logging.Logger:
    configure_logging()
    return logging.getLogger(name)
