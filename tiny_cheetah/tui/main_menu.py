from textual.app import App, ComposeResult
from textual.widgets import Header, Button, Label, Static
from textual.containers import Container, Center


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

        with Container(id="menu-btns"):
            yield Button("Chat")
            yield Button("Train", disabled=True)
            yield Button("Quit")

    def on_mount(self) -> None:
        self.title="[tiny-cheetah] v0.1"
        self.sub_title="Active Nodes 1"