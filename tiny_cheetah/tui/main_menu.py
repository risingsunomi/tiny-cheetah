from pathlib import Path
from typing import Optional

from textual.app import App, ComposeResult
from textual.widgets import Header, Button, Label
from textual.containers import Container, Center

from .chat_menu import ChatScreen
from .train_menu import TrainScreen
from .orchestration_screen import OrchestrationScreen
from .settings_screen import SettingsScreen
from tiny_cheetah.orchestration.peer_client import PeerClient


class MainMenu(App):
    CSS_PATH = Path(__file__).with_name("main_menu.tcss")
    BINDINGS = [("escape", "pop_screen", "Back")]

    def __init__(
        self,
        training_defaults: Optional[dict] = None,
        chat_default: Optional[str] = None,
        offline_mode: bool = False,
    ) -> None:
        super().__init__()
        self.training_defaults = training_defaults or {}
        self.chat_default = chat_default
        self.offline_mode = offline_mode
        self._peer_client = PeerClient()

    def compose(self) -> ComposeResult:
        yield Header(show_clock=True)
        with Container(id="main-menu"):
            yield Label(r"""
░░      ░░░  ░░░░  ░░        ░░        ░░        ░░░      ░░░  ░░░░  ░
▒  ▒▒▒▒  ▒▒  ▒▒▒▒  ▒▒  ▒▒▒▒▒▒▒▒  ▒▒▒▒▒▒▒▒▒▒▒  ▒▒▒▒▒  ▒▒▒▒  ▒▒  ▒▒▒▒  ▒
▓  ▓▓▓▓▓▓▓▓        ▓▓      ▓▓▓▓      ▓▓▓▓▓▓▓  ▓▓▓▓▓  ▓▓▓▓  ▓▓        ▓
█  ████  ██  ████  ██  ████████  ███████████  █████        ██  ████  █
██      ███  ████  ██        ██        █████  █████  ████  ██  ████  █
                """, id="title-txt")

        with Container(id="menu-btns"):
            yield Button("Chat", id="chat-btn")
            yield Button("Train", id="train-btn")
            yield Button("Network", id="network-btn")
            yield Button("Settings", id="settings-btn")
            yield Button("Quit", id="quit-btn")

    def on_mount(self) -> None:
        self.title="[tiny-cheetah] v0.1"
        if self.offline_mode:
            self.title += " [offline]"
        self._update_subtitle()
        self.set_interval(2.0, self._update_subtitle)
    
    def action_pop_screen(self):
        exit()

    def on_button_pressed(self, event: Button.Pressed) -> None:
        button_id = event.button.id
        if button_id == "chat-btn":
            defaults = getattr(self, "training_defaults", {}) or {}
            default_model = self.chat_default or defaults.get("model-id") or defaults.get("custom-model-id")
            self.push_screen(ChatScreen(default_model=default_model, offline=self.offline_mode))
        elif button_id == "train-btn":
            screen = TrainScreen(peer_manager=self._peer_client, offline_mode=self.offline_mode)
            defaults = getattr(self, "training_defaults", None)
            if defaults:
                screen.apply_default_settings(defaults)
            self.push_screen(screen)
        elif button_id == "network-btn":
            self.push_screen(OrchestrationScreen(self._peer_client))
        elif button_id == "settings-btn":
            self.push_screen(SettingsScreen())
        elif button_id == "quit-btn":
            self.exit()

    def set_training_defaults(self, settings: Optional[dict]) -> None:
        self.training_defaults = settings or {}

    def _update_subtitle(self) -> None:
        count = self._peer_client.peer_count()
        self.sub_title = f"Active Nodes {count}"
