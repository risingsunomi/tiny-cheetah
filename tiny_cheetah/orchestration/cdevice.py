from __future__ import annotations

from tiny_cheetah.models.shard import Shard
from tiny_cheetah.orchestration.device_info import collect_host_info


class CDevice:
    def __init__(
        self,
        peer_client_id: str,
        ip_address: str,
        port: int,
        shard: Shard | None = None,
        *,
        in_use: bool = False,
        tg_device: str = "CPU",
        load_host_info: bool = False,
    ) -> None:
        self.peer_client_id = peer_client_id
        self.ip_address = ip_address
        self.port = port
        self.in_use = in_use
        self.tg_device = tg_device
        self.shard = shard or Shard("", 0, 0, 0)
        self.cpu_make = ""
        self.cpu_model = ""
        self.cpu_proc_speed = ""
        self.cpu_cores = 0
        self.cpu_ram = ""
        self.gpu_make = ""
        self.gpu_model = ""
        self.gpu_vram = ""
        self.gpu_flops = 0.0
        self.ping_ms = 0.0
        if load_host_info:
            self._populate_from_host_info(collect_host_info())

    def as_dict(self) -> dict:
        return {
            "peer_client_id": self.peer_client_id,
            "ip_address": self.ip_address,
            "port": self.port,
            "in_use": self.in_use,
            "tg_device": self.tg_device,
            "shard": {
                "model_name": self.shard.model_name,
                "start_layer": self.shard.start_layer,
                "end_layer": self.shard.end_layer,
            },
            "cpu_make": self.cpu_make,
            "cpu_model": self.cpu_model,
            "cpu_proc_speed": self.cpu_proc_speed,
            "cpu_cores": self.cpu_cores,
            "cpu_ram": self.cpu_ram,
            "gpu_make": self.gpu_make,
            "gpu_model": self.gpu_model,
            "gpu_vram": self.gpu_vram,
            "gpu_flops": self.gpu_flops,
        }

    def _populate_from_host_info(self, host_info: dict) -> None:
        devices = host_info.get("devices", [])
        cpu_info = next((d for d in devices if d.get("kind") == "CPU"), {})
        gpu_info = next((d for d in devices if d.get("kind") == "GPU"), {})
        self.cpu_make = str(cpu_info.get("vendor", ""))
        self.cpu_model = str(cpu_info.get("name", ""))
        self.cpu_proc_speed = str(cpu_info.get("speed", ""))
        self.cpu_cores = int(cpu_info.get("cores", 0) or 0)
        self.cpu_ram = str(cpu_info.get("ram_gb", ""))
        self.gpu_make = str(gpu_info.get("vendor", ""))
        self.gpu_model = str(gpu_info.get("name", ""))
        self.gpu_vram = str(gpu_info.get("vram_gb", gpu_info.get("ram_gb", "")))
        self.gpu_flops = float(gpu_info.get("flops", 0.0) or 0.0)
