from textual.app import App, ComposeResult
from textual.widgets import Header, Button, Label
from textual.containers import Container, Horizontal


class MainMenu(App):
    CSS_PATH = "main_menu.tcss"

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
            
            yield Button("Start")
            yield Button("Settings")
            yield Button("Quit")

    def on_mount(self) -> None:
        self.title="[tiny-cheetah] v0.1"
