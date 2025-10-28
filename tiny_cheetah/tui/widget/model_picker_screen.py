
from pathlib import Path
from typing import Optional

from textual.binding import Binding
from textual.screen import ModalScreen
from textual.app import ComposeResult
from textual.containers import Container
from textual.widgets import Input, Button, Label, Static
from textual.events import Mount

try:
    import huggingface_hub
except Exception:
    huggingface_hub = None


class ModelPickerScreen(ModalScreen[str | None]):
    """Modal popup allowing the user to enter a model identifier or path."""

    CSS_PATH = Path(__file__).with_name("model_picker_screen.tcss")

    BINDINGS = [
        Binding("escape", "dismiss(None)", "Cancel"),
    ]

    def __init__(self, initial_value: str) -> None:
        super().__init__(id="model-picker")
        self._initial = initial_value
        self._input: Optional[Input] = None
        self._cache_options = self._discover_cached_models()

    def compose(self) -> ComposeResult:
        with Container(id="model-picker-container"):
            yield Static("Select Model", id="model-picker-title")
            yield Label("Enter a Hugging Face repo or choose a cached model:", id="model-picker-help")
            if self._cache_options:
                with Container(id="model-picker-cache"):
                    yield Label("Cached Models", id="model-picker-cache-title")
                    for option in self._cache_options:
                        yield Button(option, classes="model-cache-option")
            input_field = Input(id="model-picker-input", placeholder="Huggingface model ID or path...")
            self._input = input_field
            yield input_field
            with Container(id="model-picker-buttons"):
                yield Button("Cancel", id="model-picker-cancel")
                yield Button("Apply", id="model-picker-apply", variant="primary")

    def on_mount(self, _: Mount) -> None:
        if self._input is not None:
            self._input.value = self._initial
            self.call_after_refresh(self._input.focus)

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "model-picker-cancel":
            self.dismiss(None)
        elif event.button.id == "model-picker-apply":
            self.dismiss(self._input.value.strip() if self._input else None)
        elif "model-cache-option" in event.button.classes:
            self.dismiss(event.button.label.plain)

    def _discover_cached_models(self) -> list[str]:
        if huggingface_hub is None:
            return []
        cache_dir = Path(huggingface_hub.constants.HF_HUB_CACHE).expanduser()
        if not cache_dir.exists():
            return []
        options: set[str] = set()
        for repo_dir in cache_dir.glob("models--*"):
            resolved = repo_dir.name.replace("models--", "").replace("--", "/")
            options.add(resolved)
        return sorted(options)
