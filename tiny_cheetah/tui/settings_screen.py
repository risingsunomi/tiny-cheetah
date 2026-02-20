from __future__ import annotations

import os
from pathlib import Path
from typing import List, Dict

from textual.app import ComposeResult
from textual.containers import Container, VerticalScroll
from textual.screen import Screen
from textual.widgets import Button, Checkbox, Footer, Header, Label, Static

from tiny_cheetah.models.llm.backend import (
    LLM_BACKEND_ENV,
    get_llm_backend,
    normalize_llm_backend,
    set_llm_backend,
)
from tiny_cheetah.orchestration.device_info import collect_host_info


class SettingsScreen(Screen[None]):
    """Device selection and environment configuration."""

    CSS_PATH = Path(__file__).with_name("settings_screen.tcss")
    BINDINGS = [("escape", "pop_screen", "Back"), ("b", "pop_screen", "Back")]

    def __init__(self) -> None:
        super().__init__()
        self._device_checks: List[tuple[Checkbox, str]] = []
        self._backend_checks: List[tuple[Checkbox, str]] = []
        self._status: Label | None = None
        self._device_render_nonce: int = 0

    def compose(self) -> ComposeResult:
        yield Header(show_clock=True)
        with Container(id="settings-root"):
            yield Label("Runtime Settings", id="settings-title")
            yield Static(
                "Configure compute device (TC_DEVICE) and model backend (TC_LLM_BACKEND).",
                id="settings-help",
            )
            with VerticalScroll(id="settings-scroll"):
                yield Label("Device", id="settings-device-title")
                device_container = Container(id="settings-devices")
                self._device_container = device_container
                yield device_container
                yield Label("LLM Backend", id="settings-backend-title")
                yield Static(
                    "Select which backend the chat UI uses for model loading and generation.",
                    id="settings-backend-help",
                )
                backend_container = Container(id="settings-backends")
                self._backend_container = backend_container
                yield backend_container
            with Container(id="settings-actions"):
                yield Button("Save", id="settings-save", variant="primary")
                yield Button("Cancel", id="settings-cancel", variant="warning")
            status = Label("", id="settings-status")
            self._status = status
            yield status
        yield Footer()

    def on_mount(self) -> None:
        self._load_devices()
        self._load_backends()

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "settings-save":
            self._save_selection()
        elif event.button.id == "settings-cancel":
            self.app.pop_screen()

    def action_pop_screen(self) -> None:
        self.app.pop_screen()

    def _load_devices(self) -> None:
        container = getattr(self, "_device_container", None)
        if container is None:
            return
        self._device_render_nonce += 1
        render_id = self._device_render_nonce
        for child in list(container.children):
            child.remove()
        self._device_checks = []
        host_info = collect_host_info()
        devices: List[Dict[str, object]] = host_info.get("devices", []) or []
        backend = self._selected_backend()
        env_target = self._normalize_device_for_backend(
            (os.getenv("TC_DEVICE") or "").split(",")[0].strip(),
            backend,
        )
        selected_any = False
        for idx, device in enumerate(devices):
            name = str(device.get("name", f"device-{idx}"))
            kind = str(device.get("kind", "device")).upper()
            compute = self._device_value_for_backend(device, backend)
            ram = device.get("ram_gb", "")
            label = f"{kind}: {name}"
            if ram not in ("", None):
                label += f" ({ram} GB)"
            label += f" [{compute}]"
            checkbox = Checkbox(label, id=f"dev-{render_id}-{idx}")
            if env_target and self._normalize_device_for_backend(compute, backend) == env_target and not selected_any:
                checkbox.value = True
                selected_any = True
            self._device_checks.append((checkbox, compute))
            container.mount(checkbox)
        if env_target:
            if selected_any:
                self._set_status(f"Loaded {LLM_BACKEND_ENV}={backend}, TC_DEVICE={env_target}")
            else:
                self._set_status(
                    f"TC_DEVICE={env_target} not found for backend '{backend}'; please select a device."
                )

    def _load_backends(self) -> None:
        container = getattr(self, "_backend_container", None)
        if container is None:
            return
        for child in list(container.children):
            child.remove()
        self._backend_checks = []

        selected_backend = get_llm_backend()
        for backend in ("tinygrad", "torch"):
            checkbox = Checkbox(backend, id=f"backend-{backend}")
            checkbox.value = backend == selected_backend
            self._backend_checks.append((checkbox, backend))
            container.mount(checkbox)

        self._set_status(
            f"Loaded TC_DEVICE={(os.getenv('TC_DEVICE') or '(unset)').strip() or '(unset)'}, "
            f"{LLM_BACKEND_ENV}={selected_backend}"
        )

    def on_checkbox_changed(self, event: Checkbox.Changed) -> None:
        if event.value is False:
            return

        checkbox_id = event.checkbox.id or ""
        if checkbox_id.startswith("dev-"):
            self._enforce_single_selection(self._device_checks, event.checkbox)
            for cb, compute in self._device_checks:
                if cb is event.checkbox and cb.value:
                    self._set_status(f"Selected device {compute}")
            return

        if checkbox_id.startswith("backend-"):
            self._enforce_single_selection(self._backend_checks, event.checkbox)
            for cb, backend in self._backend_checks:
                if cb is event.checkbox and cb.value:
                    self._set_status(f"Selected backend {backend}")
                    # Device identifiers differ by backend (e.g. tinygrad METAL vs torch mps),
                    # and we refresh after this event cycle to avoid mount/remove races.
                    self.call_after_refresh(self._load_devices)
            return

    @staticmethod
    def _enforce_single_selection(options: List[tuple[Checkbox, str]], selected: Checkbox) -> None:
        for checkbox, _ in options:
            if checkbox is not selected:
                checkbox.value = False

    def _save_selection(self) -> None:
        chosen_device = next((compute for cb, compute in self._device_checks if cb.value), "")
        if not chosen_device:
            self._set_status("Select exactly one device.")
            return
        chosen_backend = next((backend for cb, backend in self._backend_checks if cb.value), "")
        if not chosen_backend:
            self._set_status("Select exactly one backend.")
            return

        os.environ["TC_DEVICE"] = chosen_device
        backend = set_llm_backend(chosen_backend)

        try:
            self._persist_env_setting("TC_DEVICE", chosen_device)
            self._persist_env_setting(LLM_BACKEND_ENV, backend)
        except Exception as exc:
            self._set_status(
                f"Saved for this session, but failed to update .env: {exc}"
            )
            return

        self._set_status(
            f"Saved TC_DEVICE={os.environ['TC_DEVICE']} and {LLM_BACKEND_ENV}={normalize_llm_backend(backend)}"
        )

    def _selected_backend(self) -> str:
        selected = next((backend for cb, backend in self._backend_checks if cb.value), None)
        return normalize_llm_backend(selected or get_llm_backend())

    @staticmethod
    def _normalize_device_for_backend(value: str, backend: str) -> str:
        if backend == "torch":
            return value.strip().lower()
        return value.strip().upper()

    def _device_value_for_backend(self, device: Dict[str, object], backend: str) -> str:
        kind = str(device.get("kind", "device")).upper()
        raw_device = str(device.get("device", kind)).strip()
        device_upper = raw_device.upper()
        name = str(device.get("name", "")).lower()

        if backend == "torch":
            if kind == "CPU":
                return "cpu"
            if device_upper in {"METAL", "MPS"} or "apple" in name:
                return "mps"
            if device_upper in {"CUDA", "GPU"}:
                return "cuda"
            if device_upper in {"AMD", "ROCM", "HIP"}:
                # ROCm builds typically expose devices through torch.cuda.
                return "cuda"
            return "cpu"

        # tinygrad backend values
        if kind == "CPU":
            return "CPU"
        if device_upper in {"MPS", "METAL"} or "apple" in name:
            return "METAL"
        if device_upper in {"CUDA", "AMD"}:
            return device_upper
        return device_upper or "CPU"

    def _persist_env_setting(self, key: str, value: str) -> None:
        env_path = Path(__file__).resolve().parents[2] / ".env"
        new_line = f"{key}={value}"

        lines: List[str] = []
        if env_path.exists():
            lines = env_path.read_text().splitlines()

        replaced = False
        for idx, line in enumerate(lines):
            stripped = line.strip()
            if not stripped or stripped.startswith("#"):
                continue
            if stripped.startswith(f"{key}="):
                lines[idx] = new_line
                replaced = True
                break

        if not replaced:
            lines.append(new_line)

        env_path.write_text("\n".join(lines).rstrip() + "\n")

    def _set_status(self, message: str) -> None:
        if self._status is not None:
            self._status.update(message)
