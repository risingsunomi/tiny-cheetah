from __future__ import annotations

import os
import platform
import sys
import subprocess
from typing import Dict, List

try:
    import psutil  # type: ignore
except Exception:  # pragma: no cover - optional
    psutil = None

try:
    import torch  # type: ignore
except Exception:  # pragma: no cover - optional
    torch = None


def _cpu_count() -> int:
    if psutil is not None:
        try:
            return psutil.cpu_count(logical=True) or 0
        except Exception:
            pass
    return os.cpu_count() or 0


def _ram_gb() -> float:
    if psutil is not None:
        try:
            mem = psutil.virtual_memory()
            return round(mem.total / (1024 ** 3), 2)
        except Exception:
            pass
    if hasattr(os, "sysconf") and sys.platform != "win32":  # pragma: no cover - platform guard
        try:
            pages = os.sysconf("SC_PHYS_PAGES")
            page_size = os.sysconf("SC_PAGE_SIZE")
            return round((pages * page_size) / (1024 ** 3), 2)
        except Exception:
            return 0.0
    return 0.0


def _gpus() -> List[Dict[str, object]]:
    gpus: List[Dict[str, object]] = []
    if torch is not None and hasattr(torch, "cuda") and torch.cuda.is_available():  # type: ignore[attr-defined]
        try:
            for idx in range(torch.cuda.device_count()):
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
    if not gpus and platform.system() == "Darwin":
        # Rough Apple GPU hint; Metal APIs not used to keep dependencies light.
        try:
            output = subprocess.check_output([
                "/usr/sbin/system_profiler",
                "SPDisplaysDataType",
            ], text=True, timeout=2)
            for line in output.splitlines():
                line = line.strip()
                if line.lower().startswith("chipset model:"):
                    name = line.split(":", 1)[1].strip()
                    gpus.append({"name": name, "total_mem_gb": 0.0, "compute": "Metal"})
                    break
        except Exception:
            gpus.append({"name": "Apple GPU", "total_mem_gb": 0.0, "compute": "Metal"})
    return gpus


def collect_host_info() -> Dict[str, object]:
    return {
        "hostname": platform.node(),
        "platform": platform.platform(),
        "cpu_count": _cpu_count(),
        "ram_gb": _ram_gb(),
        "tc_device": os.environ.get("TC_DEVICE", ""),
        "gpus": _gpus(),
    }
