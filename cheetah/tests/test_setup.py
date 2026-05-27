from __future__ import annotations

import importlib.util
import os
import unittest
from pathlib import Path
from unittest import mock


def _load_setup_module():
    setup_path = Path(__file__).resolve().parents[2] / "setup.py"
    spec = importlib.util.spec_from_file_location("cheetah_setup", setup_path)
    if spec is None or spec.loader is None:
        raise RuntimeError("Unable to load setup.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


setup_module = _load_setup_module()


class TestSetupTorchVariants(unittest.TestCase):
    def test_normalize_rocm_tag_accepts_common_spellings(self) -> None:
        self.assertEqual(setup_module._normalize_torch_variant_tag("rocm7.1"), "rocm7.1")
        self.assertEqual(setup_module._normalize_torch_variant_tag("hip6.4"), "rocm6.4")
        self.assertEqual(setup_module._normalize_torch_variant_tag("amd624"), "rocm6.2.4")

    def test_torch_requirement_supports_rocm_variant(self) -> None:
        env = {
            "TC_TORCH_VARIANT": "rocm6.4",
            "TC_TORCH_VERSION": "2.9.1",
        }
        with mock.patch.dict(os.environ, env, clear=False):
            self.assertEqual(
                setup_module._torch_requirement(),
                "torch==2.9.1+rocm6.4",
            )

    def test_torch_requirement_keeps_cuda_variant(self) -> None:
        env = {
            "TC_TORCH_VARIANT": "cuda12.8",
            "TC_TORCH_VERSION": "2.10.0",
        }
        with mock.patch.dict(os.environ, env, clear=False):
            self.assertEqual(
                setup_module._torch_requirement(),
                "torch==2.10.0+cu128",
            )


if __name__ == "__main__":
    unittest.main()
