from __future__ import annotations

import os
import unittest
from types import SimpleNamespace
from unittest.mock import patch

from tiny_cheetah.tui import helpers


def _gib(value: float) -> int:
    return int(value * (1024 ** 3))


class _FakePsutil:
    def __init__(self, *, ram_percent: float, available_gb: float, swap_percent: float) -> None:
        self._virtual_memory = SimpleNamespace(percent=ram_percent, available=_gib(available_gb))
        self._swap_memory = SimpleNamespace(percent=swap_percent)

    def virtual_memory(self):
        return self._virtual_memory

    def swap_memory(self):
        return self._swap_memory


class TestMemoryAbortReason(unittest.TestCase):
    def test_swap_alone_does_not_abort_when_ram_and_available_are_healthy(self) -> None:
        fake_psutil = _FakePsutil(ram_percent=79.0, available_gb=5.03, swap_percent=93.3)
        with patch.dict(
            os.environ,
            {
                "TC_MEM_MAX_PERCENT": "92",
                "TC_SWAP_MAX_PERCENT": "90",
                "TC_SWAP_ONLY_ABORT_PERCENT": "99",
                "TC_MEM_MIN_AVAILABLE_GB": "0.75",
            },
            clear=False,
        ):
            with patch.object(helpers, "psutil", fake_psutil):
                self.assertIsNone(helpers.memory_abort_reason("agent loop"))

    def test_swap_and_low_available_abort(self) -> None:
        fake_psutil = _FakePsutil(ram_percent=79.0, available_gb=0.5, swap_percent=93.3)
        with patch.dict(
            os.environ,
            {
                "TC_MEM_MAX_PERCENT": "92",
                "TC_SWAP_MAX_PERCENT": "90",
                "TC_SWAP_ONLY_ABORT_PERCENT": "99",
                "TC_MEM_MIN_AVAILABLE_GB": "0.75",
            },
            clear=False,
        ):
            with patch.object(helpers, "psutil", fake_psutil):
                reason = helpers.memory_abort_reason("agent loop")
        self.assertIsNotNone(reason)
        assert reason is not None
        self.assertIn("Swap usage 93.3% >= 90.0%", reason)
        self.assertIn("Available RAM 0.50 GiB <= 0.75 GiB", reason)

    def test_critical_swap_alone_can_still_abort(self) -> None:
        fake_psutil = _FakePsutil(ram_percent=79.0, available_gb=5.03, swap_percent=99.2)
        with patch.dict(
            os.environ,
            {
                "TC_MEM_MAX_PERCENT": "92",
                "TC_SWAP_MAX_PERCENT": "90",
                "TC_SWAP_ONLY_ABORT_PERCENT": "99",
                "TC_MEM_MIN_AVAILABLE_GB": "0.75",
            },
            clear=False,
        ):
            with patch.object(helpers, "psutil", fake_psutil):
                reason = helpers.memory_abort_reason("agent loop")
        self.assertIsNotNone(reason)
        assert reason is not None
        self.assertIn("critical swap-only threshold", reason)


if __name__ == "__main__":
    unittest.main()
