from __future__ import annotations

import asyncio
from pathlib import Path

from textual.app import ComposeResult
from textual.containers import Container
from textual.screen import Screen
from textual.widgets import Button, Footer, Header, Input, Label, Static

from tiny_cheetah.orchestration import get_peer_manager
from tiny_cheetah.orchestration.peer import PeerInfo


class ConnectPeerScreen(Screen[None]):
    CSS_PATH = Path(__file__).with_name("connect_peer_screen.tcss")
    BINDINGS = [("escape", "pop_screen", "Back"), ("b", "pop_screen", "Back")]

    def __init__(self) -> None:
        super().__init__()
        self._manager = get_peer_manager()
        self._host_input = Input(placeholder="host", id="connect-host")
        self._port_input = Input(placeholder="port", id="connect-port")
        self._password = Input(placeholder="password (if required)", password=True, id="connect-pass")
        self._status: Label | None = None
        self._host_info: Static | None = None

    def compose(self) -> ComposeResult:
        yield Header(show_clock=True)
        with Container(id="connect-root"):
            yield Label("Connect to Remote Peer", id="connect-title")
            yield self._host_input
            yield self._port_input
            yield self._password
            with Container(id="connect-actions"):
                yield Button("Connect", id="connect-btn", variant="primary")
            status = Label("", id="connect-status")
            self._status = status
            yield status
            info = Static("Host MOTD and terms will appear here after connecting.", id="connect-info")
            self._host_info = info
            yield info
        yield Footer()

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "connect-btn":
            self._connect_peer()
        self._discover_peers()

    def action_pop_screen(self) -> None:
        self.app.pop_screen()

    def _connect_peer(self) -> None:
        host = self._host_input.value.strip()
        port_text = self._port_input.value.strip() or "8765"
        if not host:
            self._set_status("Specify host.")
            return
        try:
            port = int(port_text)
        except ValueError:
            self._set_status("Invalid port.")
            return
        try:
            peer = self._manager.connect_to_peer(host, port, password=self._password.value)
        except Exception as exc:
            self._set_status(str(exc))
            return
        self._set_status(f"Connected to {peer.peer_id}")
        self._show_host_info(peer)

    def _set_status(self, message: str) -> None:
        if self._status is not None:
            self._status.update(message)

    def _show_host_info(self, peer: PeerInfo) -> None:
        motd = peer.metadata.get("motd", "No welcome message provided.") if hasattr(peer, "metadata") else "No welcome message provided."
        flops = f"{peer.flops_gflops:.1f} GFLOPS" if getattr(peer, "flops_gflops", 0) else "unspecified"
        policy = f"Compute: {flops}"
        lines = [
            f"Host: [bold]{peer.peer_id}[/]",
            f"Devices: {', '.join(peer.devices) or 'unknown'}",
            policy,
        ]
        lines.append(f"MOTD: {motd}")
        if self._host_info is not None:
            self._host_info.update("\n".join(lines))

    async def on_mount(self) -> None:
        await asyncio.to_thread(self._discover_peers)
        self.set_interval(1.0, self._discover_peers)

    def _discover_peers(self) -> None:
        if self.app is None:
            return
        self._manager.discover_peers()
        count = self._manager.peer_count()
        self.app.sub_title = f"Active Nodes {count}"
