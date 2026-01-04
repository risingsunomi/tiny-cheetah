from __future__ import annotations

import os
import platform
import sys
import subprocess
import re
from typing import Dict, List

from tiny_cheetah.logging_utils import get_logger

logger = get_logger(__name__)

try:
    import psutil  # type: ignore
except Exception:  # pragma: no cover - optional
    psutil = None


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
    system = platform.system()

    # Linux/Windows: try CUDA then ROCm, then a light WMI hint on Windows.
    if system in {"Linux", "Windows"}:
        gpus.extend(_cuda_gpus())
        if not gpus:
            gpus.extend(_rocm_gpus())
        if not gpus and system == "Windows":
            gpus.extend(_windows_gpu_hint())

    # macOS fallback using system_profiler (minimal, no heavy deps).
    if not gpus and system == "Darwin":
        try:
            graphics_output = subprocess.check_output(
                ["/usr/sbin/system_profiler", "SPDisplaysDataType"],
                text=True,
                timeout=2,
            )
            current: Dict[str, object] = {}
            in_graphics = False
            mem_from_graphics = False
            hardware_mem: float | None = None
            for raw in graphics_output.splitlines():
                line = raw.strip()
                lower = line.lower()
                if not line:
                    continue
                if "graphics/displays" in lower:
                    in_graphics = True
                    continue
                if line.startswith("Hardware:"):
                    in_graphics = False
                    continue
                if "chipset model" in lower and in_graphics:
                    name = line.split(":", 1)[1].strip()
                    current = {"name": name, "total_mem_gb": 0.0, "compute": "Metal"}
                    continue
                if in_graphics and "memory" in lower and current:
                    mem_gb = _parse_mac_mem_gb(line)
                    if mem_gb > 0:
                        current["total_mem_gb"] = mem_gb
                        mem_from_graphics = True
                    continue
                if not mem_from_graphics and not in_graphics and "memory" in lower and "chipset" not in lower:
                    hardware_mem = hardware_mem or _parse_mac_mem_gb(line)
            if current:
                if current.get("total_mem_gb", 0) == 0 and hardware_mem:
                    current["total_mem_gb"] = hardware_mem
                elif current.get("total_mem_gb", 0) == 0:
                    current["total_mem_gb"] = _ram_gb()
                # Apple Silicon uses unified memory shared with CPU.
                current["ram_gb"] = current.get("total_mem_gb", _ram_gb())
                current["unified_memory"] = True
                gpus.append(current)
        except Exception as exc:
            logger.debug("system_profiler GPU probe failed: %s", exc)
            gpus.append({"name": "Apple GPU", "total_mem_gb": 0.0, "compute": "Metal"})

    # Normalize ram/compute fields for consumers.
    for gpu in gpus:
        if "ram_gb" not in gpu:
            gpu["ram_gb"] = gpu.get("total_mem_gb", 0.0)
        if "compute" not in gpu:
            gpu["compute"] = ""

    return gpus


def _cuda_gpus() -> List[Dict[str, object]]:
    """Detect NVIDIA GPUs via nvidia-smi, if present."""
    cmd = [
        "nvidia-smi",
        "--query-gpu=name,memory.total",
        "--format=csv,noheader,nounits",
    ]
    try:
        output = subprocess.check_output(cmd, text=True, timeout=2)
    except Exception:
        return []
    gpus: List[Dict[str, object]] = []
    for line in output.splitlines():
        parts = [p.strip() for p in line.split(",") if p.strip()]
        if not parts:
            continue
        name = parts[0]
        mem_gb = 0.0
        if len(parts) > 1:
            try:
                mem_gb = float(parts[1])
                # nvidia-smi reports MiB; convert to GiB if it looks large.
                if mem_gb > 64:
                    mem_gb = round(mem_gb / 1024.0, 2)
            except Exception:
                mem_gb = 0.0
        gpus.append({"name": name, "total_mem_gb": mem_gb, "compute": "CUDA"})
    return gpus


def _rocm_gpus() -> List[Dict[str, object]]:
    """Detect AMD GPUs via rocm-smi, if present."""
    cmd = ["rocm-smi", "--showproductname", "--showmeminfo", "vram"]
    try:
        output = subprocess.check_output(cmd, text=True, timeout=2)
    except Exception:
        return []
    gpus: List[Dict[str, object]] = []
    current: Dict[str, object] = {}
    for raw in output.splitlines():
        line = raw.strip()
        if not line or line.startswith(("#", "=")):
            continue
        lower = line.lower()
        if lower.startswith("gpu"):
            if current:
                current.setdefault("compute", "ROCm")
                gpus.append(current)
            current = {}
            continue
        if "product name" in lower:
            try:
                _, val = line.split(":", 1)
                current["name"] = val.strip()
            except Exception:
                continue
        if "vram total memory" in lower:
            try:
                _, val = line.split(":", 1)
                bytes_val = float(val.strip().split()[0])
                current["total_mem_gb"] = round(bytes_val / (1024 ** 3), 2)
            except Exception:
                continue
    if current:
        current.setdefault("compute", "ROCm")
        gpus.append(current)
    return gpus


def _windows_gpu_hint() -> List[Dict[str, object]]:
    """Lightweight GPU hint on Windows via PowerShell, tolerant to failure."""
    cmd = [
        "powershell",
        "-Command",
        "Get-CimInstance Win32_VideoController | Select-Object Name,AdapterRAM",
    ]
    try:
        output = subprocess.check_output(cmd, text=True, timeout=2)
    except Exception:
        return []
    gpus: List[Dict[str, object]] = []
    for line in output.splitlines():
        line = line.strip()
        if not line or line.lower().startswith("name"):
            continue
        parts = line.split(None, 1)
        name = parts[0] if parts else "GPU"
        mem_gb = 0.0
        if len(parts) > 1:
            try:
                mem_gb = round(float(parts[1]) / (1024 ** 3), 2)
            except Exception:
                mem_gb = 0.0
        gpus.append({"name": name, "total_mem_gb": mem_gb})
    return gpus


def _parse_mac_mem_gb(line: str) -> float:
    """Extract memory size in GB from a line such as 'Memory: 8 GB'."""
    try:
        _, val = line.split(":", 1)
        text = val.strip()
    except Exception:
        text = line.strip()
    match = re.search(r"([0-9]+(?:\\.[0-9]+)?)", text)
    if not match:
        return 0.0
    num = float(match.group(1))
    lower = text.lower()
    if "tb" in lower:
        num *= 1024.0
    elif "mb" in lower:
        num = round(num / 1024.0, 2)
    return round(num, 2)


def _cpu_name() -> str:
    system = platform.system()
    if system == "Darwin":
        try:
            out = subprocess.check_output(
                ["/usr/sbin/sysctl", "-n", "machdep.cpu.brand_string"], text=True, timeout=2
            )
            if out.strip():
                return out.strip()
        except Exception:
            pass
    elif system == "Linux":
        try:
            with open("/proc/cpuinfo", "r", encoding="utf-8") as fh:
                for line in fh:
                    if "model name" in line.lower():
                        return line.split(":", 1)[1].strip()
        except Exception:
            pass
        try:
            out = subprocess.check_output(["lscpu"], text=True, timeout=2)
            for line in out.splitlines():
                if line.lower().startswith("model name"):
                    return line.split(":", 1)[1].strip()
        except Exception:
            pass
    elif system == "Windows":
        try:
            out = subprocess.check_output(
                ["powershell", "-Command", "(Get-CimInstance Win32_Processor).Name | Select-Object -First 1"],
                text=True,
                timeout=2,
            )
            if out.strip():
                return out.strip()
        except Exception:
            try:
                out = subprocess.check_output(["wmic", "cpu", "get", "Name"], text=True, timeout=2)
                for line in out.splitlines():
                    line = line.strip()
                    if line and line.lower() != "name":
                        return line
            except Exception:
                pass
    return platform.processor() or "CPU"


def _cpu_device() -> Dict[str, object]:
    return {
        "device": "CPU",
        "name": _cpu_name(),
        "cores": _cpu_count(),
        "ram_gb": _ram_gb(),
        "compute": "CPU",
    }


def collect_host_info() -> Dict[str, object]:
    gpus = _gpus()
    cpu = _cpu_device()
    devices: List[Dict[str, object]] = [cpu]
    devices.extend(
        {
            "device": "GPU",
            "name": gpu.get("name", "GPU"),
            "ram_gb": gpu.get("total_mem_gb", 0.0),
            "compute": gpu.get("compute", ""),
        }
        for gpu in gpus
    )
    return {
        "hostname": platform.node(),
        "platform": platform.platform(),
        "devices": devices,
    }
