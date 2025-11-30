from __future__ import annotations

from pathlib import Path
from typing import Optional

from textual.app import ComposeResult
from textual.containers import Container
from textual.screen import Screen
from textual.widgets import Button, Footer, Header, Label, ListItem, ListView, Static

from tiny_cheetah.orchestration import get_peer_manager
from tiny_cheetah.orchestration.peer import PeerInfo


class PeerDirectoryScreen(Screen[None]):
    CSS_PATH = Path(__file__).with_name("peer_directory_screen.tcss")
    BINDINGS = [("escape", "pop_screen", "Back"), ("b", "pop_screen", "Back")]

    def __init__(self) -> None:
        super().__init__()
        self._manager = get_peer_manager()
        self._peer_list: Optional[ListView] = None
        self._detail_panel: Optional[Static] = None
        self._selected_peer: Optional[str] = None

    def compose(self) -> ComposeResult:
        yield Header(show_clock=True)
        with Container(id="peer-root"):
            with Container(id="peer-list-column"):
                yield Label("Connected Peers", id="peer-title")
                peer_list = ListView(id="peer-list")
                self._peer_list = peer_list
                yield peer_list
            with Container(id="peer-detail-column"):
                detail = Static("Select a peer to view details.", id="peer-detail")
                self._detail_panel = detail
                yield detail
                with Container(id="peer-buttons"):
                    yield Button("Refresh", id="peer-refresh", variant="primary")
                    yield Button("Disconnect", id="peer-disconnect", variant="warning")
                    yield Button("Back", id="peer-back")
        yield Footer()

    def on_mount(self) -> None:
        self._refresh_peers()
        self.set_interval(3.0, self._refresh_peers)

    def on_list_view_highlighted(self, event: ListView.Highlighted) -> None:
        if event.list_view is not self._peer_list:
            return
        item = event.item
        if item.id is None or not item.id.startswith("peer-"):
            return
        self._selected_peer = item.id.split("peer-", 1)[1]
        self._update_detail()

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "peer-refresh":
            self._refresh_peers()
            return
        if event.button.id == "peer-disconnect" and self._selected_peer:
            self._manager.disconnect_peer(self._selected_peer)
            self._selected_peer = None
            self._refresh_peers()
            self._set_detail("Peer disconnected.")
        elif event.button.id == "peer-back":
            self.app.pop_screen()

    def action_pop_screen(self) -> None:
        self.app.pop_screen()

    def _refresh_peers(self) -> None:
        if self._peer_list is None:
            return
        selected = self._selected_peer
        self._peer_list.clear()
        peers = self._manager.list_peers()
        target_index: Optional[int] = None
        for idx, peer in enumerate(peers):
            label = Label(self._peer_line(peer))
            self._peer_list.append(ListItem(label, id=f"peer-{peer.node_id}"))
            if selected == peer.node_id:
                target_index = idx
        if target_index is not None:
            self._peer_list.index = target_index
        self._update_detail()

    def _update_detail(self) -> None:
        if self._detail_panel is None:
            return
        if not self._selected_peer:
            self._detail_panel.update("Select a peer to view details.")
            return
        peer = next((p for p in self._manager.list_peers() if p.node_id == self._selected_peer), None)
        if peer is None:
            self._detail_panel.update("Peer unavailable.")
            return
        desc = peer.offer_description or "General compute"
        rate_line = f"GFLOPS: {peer.flops_gflops:.1f}" if getattr(peer, "flops_gflops", 0) else "GFLOPS: unspecified"
        motd = peer.motd or "No welcome message provided."
        hw = peer.device_report or {}
        cpu = hw.get("cpu_count", "--")
        ram = hw.get("ram_gb", "--")
        tc = hw.get("tc_device", "")
        gpus = hw.get("gpus", []) or []
        gpu_line = ", ".join(g.get("name", "GPU") for g in gpus) if gpus else "No GPUs reported"
        text = (
            f"[bold]{peer.username}[/]\n"
            f"Devices: {', '.join(peer.devices) or 'unknown'}\n"
            f"Offer: {desc}\n"
            f"{rate_line}\n"
            f"MOTD: {motd}\n"
            f"CPU cores: {cpu} | RAM: {ram} GB | TC_DEVICE: {tc or 'n/a'}\n"
            f"GPUs: {gpu_line}"
        )
        self._detail_panel.update(text)

    def _peer_line(self, peer: PeerInfo) -> str:
        lock = "🔒 " if peer.metadata.get("password") else ""
        devices = ", ".join(peer.devices) or "unknown device"
        ping = f"{peer.ping_ms:.0f}ms" if getattr(peer, "ping_ms", 0) else "--"
        flops = f"{getattr(peer, 'flops_gflops', 0):.1f} GFLOPS" if getattr(peer, "flops_gflops", 0) else "--"
        return f"{lock}{peer.username:<12} | {ping:<6} | {flops:<10} | {devices}"

    def _set_detail(self, message: str) -> None:
        if self._detail_panel is not None:
            self._detail_panel.update(message)
