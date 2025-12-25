from __future__ import annotations

import time
from pathlib import Path
from typing import Optional

from textual.app import ComposeResult
from textual.containers import Container
from textual.screen import Screen
from textual.widgets import Footer, Header, Label, Static

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
        self._peer_panel: Optional[Static] = None
        self._hostmap_panel: Optional[Static] = None
        self._refresh_label: Optional[Label] = None
        self._last_refresh = time.time()

    def compose(self) -> ComposeResult:
        yield Header(show_clock=True)
        with Container(id="orch-root"):
            with Container(id="orch-left"):
                local_panel = Static(self._local_node_text(), id="orch-local")
                self._summary_panel = local_panel
                yield local_panel
            with Container(id="orch-right"):
                network_panel = Static(self._network_map_text(), id="orch-network")
                self._peer_panel = network_panel
                yield network_panel
                hostmap_panel = Static(self._host_map_text(), id="orch-hostmap")
                self._hostmap_panel = hostmap_panel
                yield hostmap_panel
        yield Footer()

    def on_mount(self) -> None:
        if self.app is not None:
            self.app.sub_title = "Host Dashboard · Active Peers"
        self.set_interval(2.0, self._refresh_panels)
        self._refresh_panels()

    def action_pop_screen(self) -> None:
        """Allow Esc/b to return to main menu."""
        self.app.pop_screen()

    def _refresh_panels(self) -> None:
        self._last_refresh = time.time()
        if self._summary_panel is not None:
            self._summary_panel.update(self._local_node_text())
        if self._peer_panel is not None:
            self._peer_panel.update(self._network_map_text())
        if self._hostmap_panel is not None:
            self._hostmap_panel.update(self._host_map_text())
        if self._refresh_label is not None:
            self._refresh_label.update("")
        self._update_action_variants()

    def _local_node_text(self) -> str:
        profile = self._manager.server_profile
        host_info = self._manager.get_host_info()
        status = "[green]● Online[/]" if self._manager.peer_count() > 0 else "[red]● Offline[/]"
        cpu = host_info.get("cpu_count", "--")
        ram = host_info.get("ram_gb", "--")
        tc = host_info.get("tc_device", "")
        gpus = host_info.get("gpus", []) or []
        gpu_line = ", ".join(g.get("name", "GPU") for g in gpus) if gpus else "No GPUs"
        config_line = f"GPU: {profile.gpu_description or gpu_line} | Ping: {profile.ping_ms or '--'} ms"
        return "\n".join(
            [
                "[b]Local Node Info[/]",
                "Identity:",
                f"- User: {self._manager.identity['username']}",
                f"- Access: {'Password required' if self._manager.access_password_required() else 'Open'}",
                f"- Status: {status}",
                "",
                "Hardware:",
                f"- CPU cores: {cpu}",
                f"- RAM: {ram} GB",
                f"- GPUs: {gpu_line}",
                f"- TC_DEVICE: {tc or 'not set'}",
                "",
                "Config:",
                f"- {config_line}",
                f"- MOTD: {profile.motd or 'Set a welcome message'}",
            ]
        )

    def _network_map_text(self) -> str:
        peers = self._manager.list_peers()
        if not peers:
            return "\n".join(
                [
                    "[b]Network / Peer Map[/]",
                    "No peers connected.",
                    "Press n to connect or p to open peer directory.",
                ]
            )
        lines = ["[b]Network / Peer Map[/]", "This Device"]
        for index, peer in enumerate(peers):
            connector = "└──" if index == len(peers) - 1 else "├──"
            status = "[green]●[/]" if peer.available else "[red]●[/]"
            ping = f"{peer.ping_ms:.0f}ms" if getattr(peer, "ping_ms", 0) or peer.ping_ms == 0 else "--"
            hw = peer.device_report or {}
            gpu_list = hw.get("gpus", [])
            gpu_name = gpu_list[0].get("name", peer.gpu_description or "GPU") if gpu_list else (peer.gpu_description or "GPU")
            flops = getattr(peer, "flops_gflops", 0) or hw.get("flops_gflops", "")
            try:
                flops_val = float(flops)
                flops_text = f"{flops_val:.1f} GFLOPS" if flops_val else ""
            except Exception:
                flops_text = ""
            hw_summary = f"{gpu_name}"
            if flops_text:
                hw_summary += f" / {flops_text}"
            line = f"{connector} {status} {peer.username} [{hw_summary}] ({ping})"
            lines.append(line)
        return "\n".join(lines)

    def _host_map_text(self) -> str:
        peers = self._manager.list_peers()
        lines = ["[b]Host Map[/]"]
        local_status = "[green]■[/]" if self._manager.peer_count() > 0 else "[red]■[/]"
        lines.append(f"{local_status} {self._manager.identity['username']}")
        for peer in peers:
            status = "[green]■[/]" if peer.available else "[red]■[/]"
            ping = f"{peer.ping_ms:.0f}ms" if getattr(peer, "ping_ms", 0) or peer.ping_ms == 0 else "--"
            lines.append(f" ├─ {status} {peer.username} ({ping})")
        if len(lines) == 2:
            lines.append(" └─ No peers connected")
        return "\n".join(lines)
    
    # Textual actions ---------------------------------------------------------
    def action_refresh_panels(self) -> None:
        self._refresh_panels()

    def action_open_hosting(self) -> None:
        self.app.push_screen(HostingScreen())

    def action_open_connect(self) -> None:
        self.app.push_screen(ConnectPeerScreen())

    def action_open_peers(self) -> None:
        self.app.push_screen(PeerDirectoryScreen())

    def _update_action_variants(self) -> None:
        return
