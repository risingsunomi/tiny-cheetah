from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Optional

from textual.app import ComposeResult
from textual.containers import Container
from textual.screen import Screen
from textual.widgets import Button, Footer, Header, Label, Static, DataTable

from tiny_cheetah.orchestration import get_peer_manager
from tiny_cheetah.orchestration.peer import PeerInfo


class PeerDirectoryScreen(Screen[None]):
    CSS_PATH = Path(__file__).with_name("peer_directory_screen.tcss")
    BINDINGS = [("escape", "pop_screen", "Back"), ("b", "pop_screen", "Back")]

    def __init__(self) -> None:
        super().__init__()
        self._manager = get_peer_manager()
        self._peer_table: Optional[DataTable] = None
        self._detail_panel: Optional[Static] = None
        self._selected_peer: Optional[str] = None

    def compose(self) -> ComposeResult:
        yield Header(show_clock=True)
        with Container(id="peer-root"):
            with Container(id="peer-list-column"):
                yield Label("Available Peers", id="peer-title")
                table = DataTable(id="peer-table", show_header=True, zebra_stripes=True)
                self._peer_table = table
                yield table
            with Container(id="peer-detail-column"):
                detail = Static("Select a peer to view details.", id="peer-detail")
                self._detail_panel = detail
                yield detail
                with Container(id="peer-buttons"):
                    yield Button("Refresh", id="peer-refresh", variant="primary")
                    self._connect_button = Button("Connect", id="peer-connect", variant="warning")
                    yield self._connect_button
        yield Footer()

    async def on_mount(self) -> None:
        if self._peer_table is not None:
            self._peer_table.add_columns("", "Name", "CPU/RAM", "GPU", "Ping")
        self._refresh_peers()
        self.set_interval(3.0, self._refresh_peers)
        await asyncio.to_thread(self._discover_peers)
        self.set_interval(1.0, self._discover_peers)

    def on_data_table_row_highlighted(self, event: DataTable.RowHighlighted) -> None:
        if event.data_table is not self._peer_table:
            return
        row_key = event.row_key
        if isinstance(row_key, str) and row_key.startswith("peer-"):
            self._selected_peer = row_key.split("peer-", 1)[1]
            self._update_detail()

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "peer-refresh":
            self._refresh_peers()
            return
        if event.button.id == "peer-connect" and self._selected_peer:
            # Disconnect if already connected, otherwise instruct to connect via network screen.
            if self._selected_peer in {p.peer_id for p in self._manager.list_peers()}:
                self._manager.disconnect_peer(self._selected_peer)
                self._selected_peer = None
                self._refresh_peers()
                self._set_detail("Peer disconnected.")
            else:
                self._set_detail("Use [n] Connect on the network screen to join this peer.")

    def action_pop_screen(self) -> None:
        self.app.pop_screen()

    def _refresh_peers(self) -> None:
        if self._peer_table is None:
            return
        selected = self._selected_peer
        self._peer_table.clear()
        peers = self._manager.list_peers()
        for peer in peers:
            row_key = f"peer-{peer.peer_id}"
            self._peer_table.add_row(*self._peer_row(peer), key=row_key)
            if selected == peer.peer_id:
                try:
                    self._peer_table.cursor_coordinate = (row_key, 0)  # type: ignore[arg-type]
                except Exception:
                    pass
        self._update_detail()
        self._update_connect_label()

    def _update_detail(self) -> None:
        if self._detail_panel is None:
            return
        if not self._selected_peer:
            self._detail_panel.update("Select a peer to view details.")
            return
        peer = next((p for p in self._manager.list_peers() if p.peer_id == self._selected_peer), None)
        if peer is None:
            self._detail_panel.update("Peer unavailable.")
            return
        hw = peer.device_report or {}
        devices = hw.get("devices", []) or hw.get("gpus", []) or []
        cpu_entry = next((d for d in devices if isinstance(d, dict) and d.get("device") == "CPU"), {})
        cpu = cpu_entry.get("cores", "--")
        ram = cpu_entry.get("ram_gb", "--")
        gpus = [d for d in devices if isinstance(d, dict) and d.get("device") == "GPU"]
        gpu_line = ", ".join(g.get("name", "GPU") for g in gpus) if gpus else "No GPUs reported"
        peers_on_server = peer.metadata.get("peers", []) if isinstance(peer.metadata, dict) else []
        if peers_on_server:
            peer_lines = "\n".join(f"- {p.get('peer_id', 'peer')}" for p in peers_on_server)
            text = (
                f"[bold]{peer.peer_id}[/] (server)\n"
                f"Connected peers:\n{peer_lines}\n"
                f"Hardware: CPUs {cpu} | RAM {ram} GB | GPUs: {gpu_line}"
            )
        else:
            text = (
                f"[bold]{peer.peer_id}[/]\n"
                f"Devices: {', '.join(peer.devices) or 'unknown'}\n"
                f"Hardware: CPUs {cpu} | RAM {ram} GB\n"
                f"GPUs: {gpu_line}"
            )
        self._detail_panel.update(text)
        self._update_connect_label()

    def _peer_row(self, peer: PeerInfo) -> list[str]:
        lock = "🔒" if peer.metadata.get("password") else ""
        ping = f"{peer.ping_ms:.0f}ms" if getattr(peer, "ping_ms", 0) or peer.ping_ms == 0 else "--"
        hw = peer.device_report or {}
        devices = hw.get("devices", []) or hw.get("gpus", []) or []
        cpu_entry = next((d for d in devices if isinstance(d, dict) and d.get("device") == "CPU"), {})
        cpu = cpu_entry.get("cores", "--")
        ram = cpu_entry.get("ram_gb", "--")
        gpu_entry = next((d for d in devices if isinstance(d, dict) and d.get("device") == "GPU"), None)
        gpu = gpu_entry.get("name", peer.gpu_description or "GPU") if gpu_entry else (peer.gpu_description or "GPU")
        cpu_ram = f"{cpu}c/{ram}GB"
        return [lock, peer.peer_id, cpu_ram, gpu, ping]

    def _set_detail(self, message: str) -> None:
        if self._detail_panel is not None:
            self._detail_panel.update(message)

    def _update_connect_label(self) -> None:
        if self._connect_button is None:
            return
        if self._selected_peer and self._selected_peer in {p.peer_id for p in self._manager.list_peers()}:
            self._connect_button.label = "Disconnect"
            self._connect_button.variant = "warning"
        else:
            self._connect_button.label = "Connect"
            self._connect_button.variant = "primary"

    def _discover_peers(self) -> None:
        if self.app is None:
            return
        self._manager.discover_peers()
        count = self._manager.peer_count()
        self.app.sub_title = f"Active Nodes {count}"
