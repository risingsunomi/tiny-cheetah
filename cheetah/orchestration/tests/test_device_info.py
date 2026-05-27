# Testing device info
import unittest
from unittest import mock

from cheetah.orchestration import device_info
from cheetah.orchestration.device_info import collect_host_info
from cheetah.logging_utils import get_logger

logger = get_logger(__name__)

class TestDeviceInfo(unittest.TestCase):
    def test_collect_host_info(self):
        info = collect_host_info()
        logger.info("Collected host info: %s", info)
        self.assertIsNotNone(info)

    def test_rocm_gpus_parses_rocm_smi_output(self):
        rocm_smi = """
GPU[0] : Product Name: AMD Radeon RX 7900 XTX
GPU[0] : VRAM Total Memory (B): 2147483648
GPU[0] : VRAM Total Used Memory (B): 536870912
"""

        with mock.patch.object(device_info.subprocess, "check_output", return_value=rocm_smi):
            gpus = device_info._rocm_gpus()

        self.assertEqual(len(gpus), 1)
        self.assertEqual(gpus[0]["device"], "ROCM")
        self.assertEqual(gpus[0]["name"], "AMD Radeon RX 7900 XTX")
        self.assertEqual(gpus[0]["total_mem_gb"], 2.0)
        self.assertEqual(gpus[0]["available_vram_gb"], 1.5)

    def test_gpus_includes_cuda_and_rocm_devices(self):
        cuda_gpu = {"name": "NVIDIA RTX 4090", "total_mem_gb": 24.0, "device": "CUDA"}
        rocm_gpu = {"name": "AMD Radeon RX 7900 XTX", "total_mem_gb": 24.0, "device": "ROCM"}

        with (
            mock.patch.object(device_info.platform, "system", return_value="Linux"),
            mock.patch.object(device_info, "_cuda_gpus", return_value=[cuda_gpu]),
            mock.patch.object(device_info, "_rocm_gpus", return_value=[rocm_gpu]),
            mock.patch.object(device_info, "_match_flops", return_value=0.0),
        ):
            gpus = device_info._gpus()

        self.assertEqual([gpu["device"] for gpu in gpus], ["CUDA", "ROCM"])

if __name__ == "__main__":
    unittest.main()
