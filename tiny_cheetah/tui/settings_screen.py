from __future__ import annotations

import os
from pathlib import Path
from typing import List, Dict

from textual.app import ComposeResult
from textual.containers import Container, VerticalScroll
from textual.screen import Screen
from textual.widgets import Button, Checkbox, Footer, Header, Label, Static

from tiny_cheetah.orchestration.device_info import collect_host_info


class SettingsScreen(Screen[None]):
    """Device selection and environment configuration."""

    CSS_PATH = Path(__file__).with_name("settings_screen.tcss")
    BINDINGS = [("escape", "pop_screen", "Back"), ("b", "pop_screen", "Back")]

    def __init__(self) -> None:
        super().__init__()
        self._device_checks: List[tuple[Checkbox, str]] = []
        self._status: Label | None = None

    def compose(self) -> ComposeResult:
        yield Header(show_clock=True)
        with Container(id="settings-root"):
            yield Label("Device Selection", id="settings-title")
            yield Static("Select the CPU/GPU devices to advertise and use (TC_DEVICE).", id="settings-help")
            with VerticalScroll(id="settings-scroll"):
                container = Container(id="settings-devices")
                self._device_container = container
                yield container
            with Container(id="settings-actions"):
                yield Button("Save", id="settings-save", variant="primary")
                yield Button("Cancel", id="settings-cancel", variant="warning")
            status = Label("", id="settings-status")
            self._status = status
            yield status
        yield Footer()

    def on_mount(self) -> None:
        self._load_devices()

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
        for child in list(container.children):
            child.remove()
        self._device_checks = []
        host_info = collect_host_info()
        devices: List[Dict[str, object]] = host_info.get("devices", []) or []
        env_target = (os.getenv("TC_DEVICE") or "").split(",")[0].strip().upper()
        selected_any = False
        for idx, device in enumerate(devices):
            name = str(device.get("name", f"device-{idx}"))
            kind = str(device.get("kind", "device")).upper()
            compute = str(device.get("device", kind)).upper() or kind
            ram = device.get("ram_gb", "")
            label = f"{kind}: {name}"
            if ram not in ("", None):
                label += f" ({ram} GB)"
            checkbox = Checkbox(label, id=f"dev-{idx}")
            if env_target and compute == env_target and not selected_any:
                checkbox.value = True
                selected_any = True
            self._device_checks.append((checkbox, compute))
            container.mount(checkbox)
        if env_target:
            if selected_any:
                self._set_status(f"Loaded TC_DEVICE={env_target}")
            else:
                self._set_status(f"TC_DEVICE={env_target} not found; please select a device.")

    def on_checkbox_changed(self, event: Checkbox.Changed) -> None:
        # Enforce single selection by clearing other checkboxes when one is set.
        if event.value is False:
            return
        for cb, _ in self._device_checks:
            if cb is not event.checkbox:
                cb.value = False
        for cb, compute in self._device_checks:
            if cb is event.checkbox and cb.value:
                self._set_status(f"Selected {compute}")

    def _save_selection(self) -> None:
        chosen = next((compute for cb, compute in self._device_checks if cb.value), "")
        if not chosen:
            self._set_status("Select exactly one device.")
            return
        os.environ["TC_DEVICE"] = chosen
        self._set_status(f"Saved TC_DEVICE={os.environ['TC_DEVICE']}")

    def _set_status(self, message: str) -> None:
        if self._status is not None:
            self._status.update(message)
