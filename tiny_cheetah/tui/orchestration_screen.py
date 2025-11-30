from __future__ import annotations

import time
from pathlib import Path
from typing import Optional

from textual.app import ComposeResult
from textual.containers import Container
from textual.screen import Screen
from textual.widgets import Button, Footer, Header, Label, Static

from tiny_cheetah.orchestration import get_peer_manager
from tiny_cheetah.tui.connect_peer_screen import ConnectPeerScreen
from tiny_cheetah.tui.hosting_screen import HostingScreen
from tiny_cheetah.tui.peer_directory_screen import PeerDirectoryScreen


class OrchestrationScreen(Screen[None]):
    """Host dashboard for remote compute orchestration."""

    CSS_PATH = Path(__file__).with_name("orchestration_screen.tcss")

    BINDINGS = [
        ("escape", "pop_screen", "Back"),
        ("b", "pop_screen", "Back"),
        ("r", "refresh_panels", "Refresh data"),
        ("h", "open_hosting", "Manage server"),
        ("n", "open_connect", "Connect to peer"),
        ("p", "open_peers", "Peer directory"),
    ]

    def __init__(self) -> None:
        super().__init__()
        self._manager = get_peer_manager()
        self._summary_panel: Optional[Static] = None
        self._identity_panel: Optional[Static] = None
        self._offer_panel: Optional[Static] = None
        self._status_panel: Optional[Static] = None
        self._peer_panel: Optional[Static] = None
        self._policy_panel: Optional[Static] = None
        self._hardware_panel: Optional[Static] = None
        self._refresh_label: Optional[Label] = None
        self._hosting_button: Optional[Button] = None
        self._connect_button: Optional[Button] = None
        self._peers_button: Optional[Button] = None
        self._last_refresh = time.time()

    def compose(self) -> ComposeResult:
        yield Header(show_clock=True)
        with Container(id="orch-root"):
            with Container(id="orch-left"):
                summary = Static(self._summary_text(), id="orch-summary")
                self._summary_panel = summary
                yield summary
                with Container(id="identity-card"):
                    identity = Static(self._identity_text(), id="orch-identity")
                    self._identity_panel = identity
                    yield identity
                offer = Static(self._offer_text(), id="orch-offer")
                self._offer_panel = offer
                yield offer
                status = Static(self._status_text(), id="orch-status")
                self._status_panel = status
                yield status
                with Container(id="orch-actions"):
                    hosting_button = Button("[h] Manage Server", id="nav-hosting")
                    self._hosting_button = hosting_button
                    yield hosting_button
                    connect_button = Button("[n] Connect to Peer", id="nav-connect")
                    self._connect_button = connect_button
                    yield connect_button
                    peers_button = Button("[p] Peer Directory", id="nav-peers")
                    self._peers_button = peers_button
                    yield peers_button
                    yield Button("[esc] Main Menu", id="nav-back", variant="error")
                help_text = Static("Last refreshed: --  (press [r] refresh)", id="orch-help")
                self._refresh_label = help_text
                yield help_text
            with Container(id="orch-right"):
                with Container(id="peer-section"):
                    yield Label("Active Peers", classes="panel-title")
                    peer_panel = Static(self._peer_listing(), id="peer-details")
                    self._peer_panel = peer_panel
                    yield peer_panel
                with Container(id="policy-section"):
                    yield Label("Connection Policy", classes="panel-title")
                    policy_panel = Static(self._policy_listing(), id="policy-detail")
                    self._policy_panel = policy_panel
                    yield policy_panel
                with Container(id="hardware-section"):
                    yield Label("Host Hardware", classes="panel-title")
                    hw_panel = Static(self._hardware_listing(), id="hardware-detail")
                    self._hardware_panel = hw_panel
                    yield hw_panel
        yield Footer()

    def on_mount(self) -> None:
        if self.app is not None:
            self.app.sub_title = "Host Dashboard · Active Peers"
        self.set_interval(2.0, self._refresh_panels)
        self._refresh_panels()

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "nav-back":
            self.app.pop_screen()
            return
        handlers = {
            "nav-hosting": self.action_open_hosting,
            "nav-connect": self.action_open_connect,
            "nav-peers": self.action_open_peers,
        }
        handler = handlers.get(event.button.id)
        if handler:
            handler()

    def action_pop_screen(self) -> None:
        """Allow Esc/b to return to main menu."""
        self.app.pop_screen()

    def _refresh_panels(self) -> None:
        self._last_refresh = time.time()
        if self._summary_panel is not None:
            self._summary_panel.update(self._summary_text())
        if self._identity_panel is not None:
            self._identity_panel.update(self._identity_text())
        if self._offer_panel is not None:
            self._offer_panel.update(self._offer_text())
        if self._status_panel is not None:
            self._status_panel.update(self._status_text())
        if self._peer_panel is not None:
            self._peer_panel.update(self._peer_listing())
        if self._policy_panel is not None:
            self._policy_panel.update(self._policy_listing())
        if self._hardware_panel is not None:
            self._hardware_panel.update(self._hardware_listing())
        if self._refresh_label is not None:
            self._refresh_label.update(self._refresh_text())
        self._update_action_variants()

    def _identity_text(self) -> str:
        return (
            f"User: [bold]{self._manager.identity['username']}[/]\n"
            f"Access: {'Password required' if self._manager.access_password_required() else 'Open'}"
        )

    def _offer_text(self) -> str:
        profile = self._manager.get_billing_profile()
        ready = bool(profile.gpu_description or profile.flops_gflops)
        status = "[green]LIVE[/]" if ready else "[yellow]DRAFT[/]"
        flops_line = f"Compute: {profile.flops_gflops:.1f} GFLOPS" if profile.flops_gflops else "Compute: specify FLOPS"
        gpu_line = f"GPU: {profile.gpu_description or 'Describe your GPU'}"
        ping_line = f"Ping: {profile.ping_ms:.0f} ms" if profile.ping_ms else "Ping: --"
        motd_line = f"MOTD: {profile.motd or 'Set a welcome message'}"
        return "\n".join(
            [
                f"Server: {status}",
                flops_line,
                gpu_line,
                ping_line,
                motd_line,
            ]
        )

    def _status_text(self) -> str:
        peers = len(self._manager.list_peers())
        devices = ", ".join(self._manager.aggregate_devices()) or "local host"
        hosting = "Online" if self._manager.peer_count() > 0 else "Offline"
        return f"Host status: {hosting}\nConnected peers: {peers}\nDevice: {devices}"

    def _summary_text(self) -> str:
        profile = self._manager.get_billing_profile()
        ready = bool(profile.gpu_description or profile.flops_gflops)
        status = "live" if ready else "draft"
        peers = len(self._manager.list_peers())
        host_state = "Online" if self._manager.peer_count() > 0 else "Offline"
        return f"{host_state} · {peers} peers · offer {status}"

    def _peer_listing(self) -> str:
        peers = self._manager.list_peers()
        if not peers:
            return (
                "[bold]No peers connected yet.[/]\n"
                "You haven't discovered or connected to any peers.\n\n"
                "[p] Open Peer Directory\n"
                "[n] Connect to peer by address\n"
                "LAN discovery runs automatically."
            )
        lines = ["ID            Capability", "────────────────────────"]
        for peer in peers[:6]:
            flops = f"{peer.flops_gflops:.1f} GFLOPS" if peer.flops_gflops else "unspecified"
            lines.append(f"{peer.username[:12]:<12}  {flops}")
        if len(peers) > 6:
            lines.append(f"… {len(peers) - 6} more (use [p] Peer Directory)")
        else:
            lines.append("Use [p] Peer Directory for peer actions.")
        return "\n".join(lines)

    def _policy_listing(self) -> str:
        profile = self._manager.get_billing_profile()
        flops = f"Compute: {profile.flops_gflops:.1f} GFLOPS" if profile.flops_gflops else "Compute: specify FLOPS"
        gpu = f"GPU: {profile.gpu_description or 'Describe your GPU'}"
        ping = f"Ping: {profile.ping_ms:.0f} ms" if profile.ping_ms else "Ping: --"
        motd = profile.motd or "Set a welcome message so renters know what to expect."
        return (
            f"{flops}\n"
            f"{gpu}\n"
            f"{ping}\n"
            "MOTD preview:\n"
            f"{motd}\n\n"
            "Renters see this information immediately after connecting."
        )

    def _hardware_listing(self) -> str:
        info = self._manager.get_host_info()
        cpu = info.get("cpu_count", "--")
        ram = info.get("ram_gb", "--")
        tc = info.get("tc_device", "")
        gpus = info.get("gpus", []) or []
        gpu_lines = []
        for gpu in gpus:
            gpu_lines.append(
                f"- {gpu.get('name','GPU')} ({gpu.get('total_mem_gb','?')} GB, {gpu.get('compute','?')})"
            )
        if not gpu_lines:
            gpu_lines.append("- No GPUs detected")
        return "\n".join(
            [
                f"CPU cores: {cpu}",
                f"RAM: {ram} GB",
                f"TC_DEVICE: {tc or 'not set'}",
                "GPUs:",
                *gpu_lines,
            ]
        )

    def _refresh_text(self) -> str:
        return time.strftime("Last refreshed: %H:%M", time.localtime(self._last_refresh))

    # Textual actions ---------------------------------------------------------
    def action_refresh_panels(self) -> None:
        self._refresh_panels()

    def action_open_hosting(self) -> None:
        self.app.push_screen(HostingScreen())

    def action_open_connect(self) -> None:
        self.app.push_screen(ConnectPeerScreen())

    def action_open_peers(self) -> None:
        self.app.push_screen(PeerDirectoryScreen())

    def _copy_to_clipboard(self, value: str, toast: str) -> None:
        self.app.copy_to_clipboard(value)
        if self._refresh_label is not None:
            self._refresh_label.update(f"{self._refresh_text()} • {toast}")

    def _update_action_variants(self) -> None:
        profile = self._manager.get_billing_profile()
        offer_ready = bool(profile.gpu_description or profile.flops_gflops)
        if self._connect_button is not None:
            self._connect_button.variant = "primary" if offer_ready else "default"
        if self._peers_button is not None:
            peers = len(self._manager.list_peers())
            self._peers_button.variant = "primary" if peers else "default"
        if self._hosting_button is not None:
            self._hosting_button.variant = "primary" if self._manager.peer_count() == 0 else "default"
