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


class ServerDirectoryScreen(Screen[None]):
    """List local/known servers and allow connecting to one."""

    CSS_PATH = Path(__file__).with_name("peer_directory_screen.tcss")
    BINDINGS = [("escape", "pop_screen", "Back"), ("b", "pop_screen", "Back")]

    def __init__(self) -> None:
        super().__init__()
        self._manager = get_peer_manager()
        self._server_table: Optional[DataTable] = None
        self._detail_panel: Optional[Static] = None
        self._selected_server: Optional[str] = None
        self._status: Optional[Label] = None

    def compose(self) -> ComposeResult:
        yield Header(show_clock=True)
        with Container(id="peer-root"):
            with Container(id="peer-list-column"):
                yield Label("Local Servers", id="peer-title")
                table = DataTable(id="peer-table", show_header=True, zebra_stripes=True)
                self._server_table = table
                yield table
            with Container(id="peer-detail-column"):
                detail = Static("Select a server to view details.", id="peer-detail")
                self._detail_panel = detail
                yield detail
                with Container(id="peer-buttons"):
                    yield Button("Refresh", id="peer-refresh", variant="primary")
                    self._connect_button = Button("Connect", id="peer-connect", variant="warning")
                    yield self._connect_button
                    status = Label("", id="peer-status")
                    self._status = status
                    yield status
        yield Footer()

    async def on_mount(self) -> None:
        if self._server_table is not None:
            self._server_table.add_columns("", "Name", "Address", "GPU", "Ping")
        self._refresh_servers()
        self.set_interval(3.0, self._refresh_servers)
        await asyncio.to_thread(self._discover_servers)
        self.set_interval(2.0, self._discover_servers)

    def on_data_table_row_highlighted(self, event: DataTable.RowHighlighted) -> None:
        if event.data_table is not self._server_table:
            return
        row_key = event.row_key
        if isinstance(row_key, str) and row_key.startswith("srv-"):
            self._selected_server = row_key.split("srv-", 1)[1]
            self._update_detail()

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "peer-refresh":
            self._refresh_servers()
            return
        if event.button.id == "peer-connect" and self._selected_server:
            self._connect_selected()

    def action_pop_screen(self) -> None:
        self.app.pop_screen()

    def _refresh_servers(self) -> None:
        if self._server_table is None:
            return
        # Update discovery before rendering.
        self._discover_servers()
        selected = self._selected_server
        self._server_table.clear()
        peers = self._manager.list_peers(include_self=False)
        servers = [
            p
            for p in peers
            if getattr(p, "peer_type", "") == "server"
        ]
        for srv in servers:
            row_key = f"srv-{srv.peer_id}"
            self._server_table.add_row(*self._server_row(srv), key=row_key)
            if selected == srv.peer_id:
                try:
                    self._server_table.cursor_coordinate = (row_key, 0)  # type: ignore[arg-type]
                except Exception:
                    pass
        self._update_detail()
        self._update_connect_label()

    def _connect_selected(self) -> None:
        target = next((p for p in self._manager.list_peers(include_self=False) if p.peer_id == self._selected_server), None)
        if target is None:
            self._set_status("Server unavailable.")
            return
        try:
            self._manager.connect_to_server(target.address, target.port or 8765)
        except Exception as exc:
            self._set_status(str(exc))
            return
        self._set_status(f"Connected to {target.peer_id}")

    def _update_detail(self) -> None:
        if self._detail_panel is None:
            return
        if not self._selected_server:
            self._detail_panel.update("Select a server to view details.")
            return
        srv = next((p for p in self._manager.list_peers(include_self=False) if p.peer_id == self._selected_server), None)
        if srv is None:
            self._detail_panel.update("Server unavailable.")
            return
        hw = srv.device_report or {}
        devices = hw.get("devices", []) or hw.get("gpus", []) or []
        gpu_entry = next((d for d in devices if isinstance(d, dict) and d.get("device") == "GPU"), {})
        motd = srv.metadata.get("motd", "") if isinstance(srv.metadata, dict) else ""
        desc = srv.metadata.get("description", "") if isinstance(srv.metadata, dict) else ""
        gpu_line = gpu_entry.get("name", srv.gpu_description or "GPU")
        address = srv.address
        port = srv.port
        if srv.peer_id == self._manager.peer_id:
            profile = self._manager.server_profile
            address = profile.address[0]
            port = profile.address[1]
        text = "\n".join(
            [
                f"[bold]{srv.peer_id}[/]",
                f"Address: {address}:{port}",
                f"GPU: {gpu_line}",
                f"MOTD: {motd or 'No message'}",
                f"Description: {desc or 'No description'}",
            ]
        )
        self._detail_panel.update(text)
        self._update_connect_label()

    def _server_row(self, server: PeerInfo) -> list[str]:
        lock = "🔒" if isinstance(server.metadata, dict) and server.metadata.get("password") else ""
        addr = f"{server.address}:{server.port}"
        hw = server.device_report or {}
        devices = hw.get("devices", []) or hw.get("gpus", []) or []
        gpu_entry = next((d for d in devices if isinstance(d, dict) and d.get("device") == "GPU"), None)
        gpu = gpu_entry.get("name", server.gpu_description or "GPU") if gpu_entry else (server.gpu_description or "GPU")
        ping = f"{server.ping_ms:.0f}ms" if getattr(server, "ping_ms", 0) or server.ping_ms == 0 else "--"
        return [lock, server.peer_id, addr, gpu, ping]

    def _set_status(self, message: str) -> None:
        if self._status is not None:
            self._status.update(message)

    def _update_connect_label(self) -> None:
        if getattr(self, "_connect_button", None) is None:
            return
        if self._selected_server:
            self._connect_button.label = "Connect"
            self._connect_button.variant = "primary"
        else:
            self._connect_button.label = "Connect"
            self._connect_button.variant = "warning"

    def _discover_servers(self) -> None:
        if self.app is None:
            return
        self._manager.discover_servers()
        count = self._manager.peer_count()
        self.app.sub_title = f"Active Nodes {count}"
