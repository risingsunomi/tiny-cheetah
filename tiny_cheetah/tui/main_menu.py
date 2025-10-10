from pathlib import Path
from typing import Optional

from textual.app import App, ComposeResult
from textual.widgets import Header, Button, Label
from textual.containers import Container, Center

from .chat_menu import ChatScreen
from .train_menu import TrainScreen


class MainMenu(App):
    CSS_PATH = Path(__file__).with_name("main_menu.tcss")

    def __init__(self, training_defaults: Optional[dict] = None) -> None:
        super().__init__()
        self.training_defaults = training_defaults or {}

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
            yield Button("Quit", id="quit-btn")

    def on_mount(self) -> None:
        self.title="[tiny-cheetah] v0.1"
        self.sub_title="Active Nodes 1"

    def on_button_pressed(self, event: Button.Pressed) -> None:
        button_id = event.button.id
        if button_id == "chat-btn":
            defaults = getattr(self, "training_defaults", {}) or {}
            default_model = defaults.get("model-id") or defaults.get("custom-model-id")
            self.push_screen(ChatScreen(default_model=default_model))
        elif button_id == "train-btn":
            screen = TrainScreen()
            defaults = getattr(self, "training_defaults", None)
            if defaults:
                screen.apply_default_settings(defaults)
            self.push_screen(screen)
        elif button_id == "quit-btn":
            self.exit()

    def set_training_defaults(self, settings: Optional[dict]) -> None:
        self.training_defaults = settings or {}
