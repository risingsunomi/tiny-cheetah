from __future__ import annotations

from pathlib import Path

from textual.app import ComposeResult
from textual.containers import Container
from textual.screen import Screen
from textual.widgets import Button, Footer, Header, Input, Label, Log

from tiny_cheetah.orchestration import get_peer_manager


class HostingScreen(Screen[None]):
    CSS_PATH = Path(__file__).with_name("hosting_screen.tcss")
    BINDINGS = [("escape", "pop_screen", "Back"), ("b", "pop_screen", "Back")]

    def __init__(self) -> None:
        super().__init__()
        self._manager = get_peer_manager()
        self._status: Label | None = None
        self._host_input = Input(value="0.0.0.0", placeholder="bind host", id="host-address")
        self._port_input = Input(value="8765", placeholder="port", id="host-port")
        self._password_input = Input(value="", placeholder="access password (optional)", password=True, id="host-pass")
        profile = self._manager.server_profile
        host_info = self._manager.get_host_info()
        default_gpu = ""
        if host_info.get("gpus"):
            default_gpu = str(host_info["gpus"][0].get("name", ""))
        self._desc_input = Input(value=profile.description, placeholder="Server description", id="host-desc")
        self._flops_input = Input(value=f"{profile.flops_gflops}", placeholder="GFLOPS", id="host-flops")
        self._gpu_input = Input(value=profile.gpu_description or default_gpu, placeholder="GPU description", id="host-gpu")
        self._motd_input = Input(value=profile.motd, placeholder="MOTD shown to clients", id="host-motd")
        self._log: Log | None = None

    def compose(self) -> ComposeResult:
        yield Header(show_clock=True)
        with Container(id="hosting-root"):
            yield Label("Server Details", id="hosting-offer-title")
            with Container(id="hosting-metadata"):
                with Container(id="hosting-offer"):
                    
                    yield self._host_input
                    yield self._port_input
                    yield self._password_input
                    yield self._gpu_input
                    yield self._flops_input
                    yield self._desc_input
                    yield self._motd_input
                with Container(id="hosting-gpus"):
                    yield Label("Local GPUs", id="hosting-gpu-title")
                    gpu_table = self._gpu_table()
                    yield gpu_table
            with Container(id="hosting-buttons"):
                yield Button("Start Hosting", id="hosting-start", variant="success")
                yield Button("Stop Hosting", id="hosting-stop", variant="warning")
            status = Label("", id="hosting-status")
            self._status = status
            yield status
            log = Log(id="hosting-log", highlight=False, auto_scroll=True)
            self._log = log
            yield log
        yield Footer()

    def on_mount(self) -> None:
        if self._log is not None:
            # Defer log seeding until the app is active.
            self.call_after_refresh(
                lambda: self._log.write_line("Host activity will appear here once you start or stop the server.")
            )

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "hosting-start":
            self._start_hosting()
        elif event.button.id == "hosting-stop":
            self._manager.stop_hosting()
            self._set_status("Hosting stopped.")
            self._append_log("Stopped hosting server.")
        elif event.button.id == "hosting-save":
            self._save_details()

    def action_pop_screen(self) -> None:
        self.app.pop_screen()

    def _start_hosting(self) -> None:
        host = self._host_input.value.strip() or "0.0.0.0"
        try:
            port = int(self._port_input.value.strip() or "8765")
        except ValueError:
            self._set_status("Invalid port.")
            self._append_log("Invalid port specified; hosting not started.")
            return
        try:
            self._manager.start_hosting(host, port, password=self._password_input.value)
        except Exception as exc:  # pragma: no cover - defensive
            self._set_status(f"Failed to host: {exc}")
            self._append_log(f"Failed to start hosting on {host}:{port}: {exc}")
            return
        self._set_status(f"Hosting on {host}:{port}")
        self._append_log(f"Listening on {host}:{port}")

    def _save_details(self) -> None:
        try:
            flops = float(self._flops_input.value.strip())
        except ValueError:
            flops = self._manager.server_profile.flops_gflops
        try:
            ping = float(self._ping_input.value.strip())
        except ValueError:
            ping = self._manager.server_profile.ping_ms
        self._manager.update_server_profile(
            description=self._desc_input.value.strip() or "General compute",
            flops_gflops=flops,
            gpu_description=self._gpu_input.value.strip(),
            ping_ms=ping,
            motd=self._motd_input.value.strip() or "Welcome to my host",
        )
        self._append_log("Server details saved.")
        self._set_status("Server details updated.")

    def _set_status(self, message: str) -> None:
        if self._status is not None:
            self._status.update(message)

    def _append_log(self, message: str) -> None:
        if self._log is not None:
            self._log.write_line(message)

    def _gpu_table(self) -> Label:
        info = self._manager.get_gpu_inventory()
        lines = ["Name                Memory(GB)  Compute"]
        for gpu in info:
            lines.append(f"{gpu.get('name','')[:16]:<18} {gpu.get('total_mem_gb',0):>5}      {gpu.get('compute','')}")
        return Label("\\n".join(lines), id="hosting-gpu-table")
