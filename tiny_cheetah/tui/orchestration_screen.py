from __future__ import annotations

import time
import asyncio
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
from tiny_cheetah.tui.server_directory_screen import ServerDirectoryScreen


class OrchestrationScreen(Screen[None]):
    """Host dashboard for remote compute orchestration."""

    CSS_PATH = Path(__file__).with_name("orchestration_screen.tcss")

    BINDINGS = [
        ("escape", "pop_screen", "Back"),
        ("b", "pop_screen", "Back"),
        ("r", "refresh_panels", "Refresh data"),
        ("h", "open_hosting", "Manage server"),
        ("n", "open_connect", "Connect to server"),
        ("p", "open_peers", "Peer directory"),
        ("s", "open_servers", "Server directory"),
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

    async def on_mount(self) -> None:
        await asyncio.to_thread(self._discover_peers)
        self.set_interval(5.0, self._discover_peers)

    # Textual actions ---------------------------------------------------------
    def action_refresh_panels(self) -> None:
        self._refresh_panels()

    def action_open_hosting(self) -> None:
        self.app.push_screen(HostingScreen())

    def action_open_connect(self) -> None:
        self.app.push_screen(ConnectPeerScreen())

    def action_open_peers(self) -> None:
        self.app.push_screen(PeerDirectoryScreen())

    def action_open_servers(self) -> None:
        self.app.push_screen(ServerDirectoryScreen())
    
    def action_pop_screen(self) -> None:
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

    def _discover_peers(self) -> None:
        if self.app is None:
            return
        self._manager.discover_peers()
        count = self._manager.peer_count()
        current = getattr(self.app, "sub_title", "")
        new_title = f"Host Dashboard · Active Nodes {count}"
        if new_title != current:
            self.app.sub_title = new_title

    def _local_node_text(self) -> str:
        profile = self._manager.server_profile
        host_info = self._manager.get_host_info()
        online = self._manager.is_hosting() or bool(self._manager.list_peers())
        status = "[green]● Online[/]" if online else "[red]● Offline[/]"
        devices = host_info.get("devices", []) or []
        cpu_entry = next((d for d in devices if d.get("device") == "CPU"), {})
        gpu_entries = [d for d in devices if d.get("device") == "GPU"]
        cpu_cores = cpu_entry.get("cores", "--")
        ram = cpu_entry.get("ram_gb", "--")
        gpu_line = ", ".join(d.get("name", "GPU") for d in gpu_entries) if gpu_entries else "No GPUs"
        config_line = f"GPU: {profile.gpu_description or gpu_line} | Ping: {profile.ping_ms or '--'} ms"
        return "\n".join(
            [
                "[b]Local Node Info[/]",
                "Identity:",
                f"- Node: {self._manager.peer_id}",
                f"- Access: {'Password required' if self._manager.access_password_required() else 'Open'}",
                f"- Status: {status}",
                "",
                "Hardware:",
                f"- CPU cores: {cpu_cores}",
                f"- RAM: {ram} GB",
                f"- GPUs: {gpu_line}",
                "",
                "Config:",
                f"- {config_line}",
                f"- MOTD: {profile.motd or 'Set a welcome message'}",
            ]
        )

    def _network_map_text(self) -> str:
        peers = self._manager.list_peers(include_self=True)
        if len(peers) <= 1:
            return "\n".join(
                [
                    "[b]Peer Ring[/]",
                    "No remote peers in the ring.",
                    "Press n to connect to a server or p to open peer directory.",
                ]
            )

        def _status(peer) -> str:
            return "[green]●[/]" if peer.available else "[red]●[/]"

        ring_lines = ["[b]Peer Ring[/]"]
        ring_lines.append("Order:")
        ordered = [p for p in peers if p.peer_id != self._manager.peer_id]
        ordered.insert(0, self._manager.peer_info)  # self first in ring

        for idx, peer in enumerate(ordered):
            ping = f"{peer.ping_ms:.0f}ms" if getattr(peer, "ping_ms", 0) or peer.ping_ms == 0 else "--"
            hw = peer.device_report or {}
            devices = hw.get("devices", []) or hw.get("gpus", [])
            gpu_entry = next((d for d in devices if isinstance(d, dict) and d.get("device") == "GPU"), None)
            gpu_name = gpu_entry.get("name", peer.gpu_description or "GPU") if gpu_entry else (peer.gpu_description or "GPU")
            shard_obj = getattr(peer, "shard", None)
            shard = {}
            if shard_obj:
                shard = {
                    "start_layer": getattr(shard_obj, "start_layer", 0),
                    "end_layer": getattr(shard_obj, "end_layer", 0),
                }
            shard_span = ""
            if shard:
                shard_span = f" L{shard.get('start_layer', 0)}-{shard.get('end_layer', 0)}"
            line = f"{idx+1:>2}) {_status(peer)} {peer.peer_id} [{gpu_name}] ({ping}){shard_span}"
            ring_lines.append(line)

        # visual wrap indicator
        if len(ordered) > 1:
            ring_lines.append(" └─ wrap → back to 1")
        return "\n".join(ring_lines)

    def _host_map_text(self) -> str:
        peers = self._manager.list_peers(include_self=True)
        lines = ["[b]Capacity Tree[/]"]
        online = self._manager.is_hosting() or len(peers) > 1
        local_status = "[green]■[/]" if online else "[red]■[/]"
        lines.append(f"{local_status} {self._manager.peer_id}")
        for peer in peers:
            if peer.peer_id == self._manager.peer_id:
                continue
            status = "[green]■[/]" if peer.available else "[red]■[/]"
            hw = peer.device_report or {}
            devices = hw.get("devices", []) or hw.get("gpus", []) or []
            gpu_entry = next((d for d in devices if isinstance(d, dict) and d.get("device") == "GPU"), None)
            cpu_entry = next((d for d in devices if isinstance(d, dict) and d.get("device") == "CPU"), None)
            gpu = gpu_entry.get("name", peer.gpu_description or "GPU") if gpu_entry else (peer.gpu_description or "GPU")
            ram = cpu_entry.get("ram_gb", "--") if cpu_entry else hw.get("ram_gb", "--")
            shard_obj = getattr(peer, "shard", None)
            shard_span = ""
            if shard_obj:
                shard_span = f" | L{getattr(shard_obj,'start_layer',0)}-{getattr(shard_obj,'end_layer',0)}"
            lines.append(f" ├─ {status} {peer.peer_id} [{gpu}, {ram}GB]{shard_span}")
        if len(lines) == 2:
            lines.append(" └─ No peers connected")
        return "\n".join(lines)
    
    

    def _update_action_variants(self) -> None:
        return
