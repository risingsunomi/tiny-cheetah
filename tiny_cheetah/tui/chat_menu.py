from __future__ import annotations

from datetime import datetime
import os
import time
from pathlib import Path
from typing import Dict, List, Optional

import tinygrad as tg
from transformers import AutoTokenizer
from rich.markup import escape

# Some macOS Python builds hit `bad value(s) in fds_to_keep` when hf_transfer is enabled.
# Disable it here so the downloader falls back to the pure-python implementation.
os.environ.setdefault("HF_HUB_ENABLE_HF_TRANSFER", "0")

from tiny_cheetah.models.llm.helpers import generate, load_safetensors
from tiny_cheetah.models.llm.model import Model
from tiny_cheetah.models.llm.model_config import ModelConfig
from tiny_cheetah.models.llm.shard import Shard

try:  # pragma: no cover - optional dependency
    import huggingface_hub
    from tiny_cheetah.repos import RepoCustom
except Exception:  # pragma: no cover - missing huggingface_hub
    huggingface_hub = None  # type: ignore[assignment]
    RepoCustom = None  # type: ignore[assignment]

from textual.app import ComposeResult
from textual.binding import Binding
from textual.containers import Container, Vertical
from textual.events import Mount
from textual.message import Message
from textual.message_pump import MessagePump
from textual.screen import ModalScreen, Screen
from textual.widgets import Button, Footer, Header, Input, Label, RichLog, Static


class ChatModelSelected(Message):
    """Message emitted when the user selects a new model."""

    def __init__(self, sender: MessagePump, model_id: str) -> None:
        super().__init__(sender)
        self.model_id = model_id


class ChatScreen(Screen[None]):
    """Primary chat interface."""

    CSS_PATH = Path(__file__).with_name("chat_menu.tcss")

    BINDINGS = [
        Binding("escape", "pop_screen", "Back"),
        Binding("ctrl+s", "open_model_picker", "Select Model"),
        Binding("ctrl+b", "pop_screen", "Menu"),
    ]

    def __init__(self, default_model: str | None = None) -> None:
        super().__init__()
        self._chat_log: Optional[RichLog] = None
        self._input: Optional[Input] = None
        self._model_id = default_model or ""
        self._tok_stats = 0.0
        self._stats_label: Optional[Label] = None
        self._model_label: Optional[Label] = None
        self._model: Optional[Model] = None
        self._model_config: Optional[object] = None
        self._tokenizer: Optional[AutoTokenizer] = None
        self._model_cache_path: Optional[Path] = None
        self._history: List[Dict[str, str]] = []
        self._generation_thread_in_progress = False

    def compose(self) -> ComposeResult:
        yield Header(show_clock=True)
        with Container(id="chat-root"):
            with Container(id="chat-body"):
                yield RichLog(id="chat-log", markup=True, auto_scroll=True, wrap=True)
                with Container(id="chat-side"):
                    with Static(id="model-panel"):
                        yield Label("Model", classes="panel-title")
                        model_value = Label(self._model_id or "<select>", id="model-value")
                        self._model_label = model_value
                        yield model_value
                        yield Button("Select Model", id="open-model-picker")
                        yield Button("Load Model", id="load-model")
                    with Static(id="nodes-panel"):
                        yield Label("Nodes", classes="panel-title")
                        yield Label("Self", id="nodes-value")
                    with Static(id="stats-panel"):
                        yield Label("Processing", classes="panel-title")
                        stats_value = Label("Tokens/sec: --", id="stats-value")
                        self._stats_label = stats_value
                        yield stats_value
                    with Static(id="actions-panel"):
                        yield Label("Actions", classes="panel-title")
                        yield Button("Clear Model", id="clear-model", variant="error")
                        yield Button("Back to Menu", id="chat-back")
            yield Input(placeholder="Type a prompt and press Enter…", id="chat-input")
        yield Footer()

    def on_mount(self, _: Mount) -> None:
        self._chat_log = self.query_one("#chat-log", RichLog)
        self._input = self.query_one("#chat-input", Input)
        if self._input is not None:
            self.call_after_refresh(self._input.focus)
        if self._model_label is not None:
            self._model_label.update(self._model_id or "<select>")

    def action_open_model_picker(self) -> None:
        self._open_model_picker()

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "open-model-picker":
            self._open_model_picker()
        elif event.button.id == "load-model":
            if not self._model_id:
                self._append_system("Select a model first.")
                return
            if self._model is not None and self._tokenizer is not None:
                self._append_system("Model already loaded.")
                return
            self._append_system(f"Loading model '{self._model_id}'...")
            time.sleep(1)  # allow UI to update
            try:
                self._load_model_blocking()
            except Exception as exc:  # pragma: no cover
                self._append_system(f"Failed to load model: {exc}")
        elif event.button.id == "chat-back":
            self.app.pop_screen()
        elif event.button.id == "clear-model":
            self._clear_model()

    def _open_model_picker(self) -> None:
        self.app.push_screen(ModelPickerScreen(self._model_id or ""), self._handle_model_selected)

    def _handle_model_selected(self, result: Optional[str]) -> None:
        if not result:
            return
        result = result.strip()
        if not result:
            return
        if result != self._model_id:
            self._model = None
            self._model_config = None
            self._tokenizer = None
            self._model_cache_path = None
            self._history.clear()
        self._model_id = result
        if self._model_label is not None:
            self._model_label.update(result)
        self._append_system(f"Model set to '{result}'.")
        self._append_system("Click 'Load Model' to download weights into memory.")

    def on_input_submitted(self, event: Input.Submitted) -> None:
        if event.input.id != "chat-input":
            return
        content = event.value.strip()
        if not content:
            return
        event.input.value = ""
        if not self._model_id:
            self._append_system("Select a model before chatting (Ctrl+S).")
            return
        if self._model is None or self._tokenizer is None:
            self._append_system("Loading model now...")
            try:
                self._load_model_blocking()
            except Exception as exc:  # pragma: no cover
                self._append_system(f"Failed to load model: {exc}")
                return
        if self._generation_thread_in_progress:
            self._append_system("Model is generating a response; please wait...")
            return
        self._append_user(content)
        self._history.append({"role": "user", "content": content})
        try:
            self._generate_response()
        except Exception as exc:  # pragma: no cover - runtime safety
            self._append_system(f"Error: {exc}")


    def _append_user(self, content: str) -> None:
        if self._chat_log is not None:
            text = escape(content)
            self._chat_log.write(
                f"({datetime.now().strftime('%Y-%m-%d %H:%M:%S')}) [bold][#3bd6ee]User[/][/bold]: {text}\n"
            )

    def _append_system(self, content: str) -> None:
        if self._chat_log is not None:
            text = escape(content)
            self._chat_log.write(
                f"({datetime.now().strftime('%Y-%m-%d %H:%M:%S')}) [bold][#00FF00]System[/][/bold]: {text}\n"
            )

    def _append_model(self, content: str) -> None:
        if self._chat_log is not None:
            label = escape(self._model_id or "Model")
            text = escape(content)
            self._chat_log.write(
                f"({datetime.now().strftime('%Y-%m-%d %H:%M:%S')}) [bold][#ff0000]{label}[/][/bold]: {text}\n"
            )

    def _load_model_blocking(self) -> None:
        if not self._model_id:
            raise RuntimeError("No model selected")
        start = time.time()
        model, model_config, tokenizer, model_path, messages = self._load_model_worker()
        for message in messages:
            self._append_system(message)
        elapsed = time.time() - start
        self._model = model
        self._model_config = model_config
        self._tokenizer = tokenizer
        self._model_cache_path = model_path
        self._append_system(f"Model ready in {elapsed:.1f}s.")

    @staticmethod
    def _config_get(config: object, key: str, default=None):
        if config is None:
            return default
        if isinstance(config, dict):
            return config.get(key, default)
        if hasattr(config, "__getitem__"):
            value = config[key]  # type: ignore[index]
            return default if value is None else value
        return default

    def _load_local_config(self, directory: Path) -> tuple[Path, ModelConfig]:
        config = ModelConfig()
        config_file = directory / "config.json"
        if not config_file.exists():
            raise FileNotFoundError(f"config.json not found under {directory}")
        config.load(config_file)
        gen_config = directory / "generation_config.json"
        if gen_config.exists():
            config.load_generation_config(gen_config)
        return directory, config

    def _generate_response(self, max_new_tokens: int = 128) -> None:
        if self._model is None or self._tokenizer is None:
            return
        assert self._tokenizer is not None
        assert self._model is not None
        assert self._model_config is not None

        template = self._tokenizer.apply_chat_template(
            self._history,
            add_generation_prompt=True,
            tokenize=False,
        )
        enc = self._tokenizer(template, return_tensors="np")
        input_ids = tg.Tensor(enc["input_ids"])
        attention_mask = tg.Tensor(enc["attention_mask"])

        max_new = max_new_tokens
        temp = self._config_get(self._model_config, "temperature", 0.7) or 0.7
        top_k = self._config_get(self._model_config, "top_k", 0) or 0
        top_p = self._config_get(self._model_config, "top_p", 0.8) or 0.8

        if self._generation_thread_in_progress:
            self._append_system("Model is already generating a response.")
            return

        self._generation_thread_in_progress = True
        try:
            out_tokens, elapsed = self._generate_tokens_worker(
                self._model,
                self._tokenizer,
                input_ids,
                attention_mask,
                max_new,
                temp,
                top_k,
                top_p,
            )
        except Exception as exc:
            self._generation_thread_in_progress = False
            self._append_system(f"Error: {exc}")
            return
        self._generation_thread_in_progress = False
        self._on_generation_complete(out_tokens, elapsed)

    def _on_generation_complete(self, out_tokens: list[int], elapsed: float) -> None:
        token_count = len(out_tokens)
        tok_rate = (token_count / elapsed) if elapsed > 0 else float("inf")
        self._tok_stats = tok_rate if token_count else 0.0
        if self._stats_label is not None:
            display_rate = self._tok_stats if token_count else 0.0
            self._stats_label.update(f"Tokens/sec: {display_rate:.1f}")

        reply = self._tokenizer.decode(out_tokens, skip_special_tokens=True).strip()
        if not reply:
            reply = "(empty response)"
        self._append_model(reply)
        self._history.append({"role": "assistant", "content": reply})
        if len(self._history) > 6:
            self._history = self._history[-6:]

    def _clear_model(self) -> None:
        self._append_system("Clearing loaded model.")
        self._model = None
        self._model_config = None
        self._tokenizer = None
        self._model_cache_path = None
        self._history.clear()
        self._generation_thread_in_progress = False
        if self._model_label is not None:
            self._model_label.update("<select>")
        self._append_system("Model cleared. Select and load a model to continue.")


    def _load_model_worker(self):
        log_messages: List[str] = []
        candidate_path = Path(self._model_id).expanduser()
        if candidate_path.exists():
            log_messages.append(f"Loading local model from {candidate_path}")
            model_path, model_config = self._load_local_config(candidate_path)
            tokenizer_source = str(model_path)
            tokenizer_local = True
        else:
            log_messages.append(f"Downloading model from Hugging Face: {self._model_id}")
            if RepoCustom is None:
                raise RuntimeError("huggingface_hub is required to download models.")
            repo = RepoCustom(self._model_id)
            model_path, model_config, repo_messages = repo.download()
            log_messages.extend(f"[download] {msg}" for msg in repo_messages)
            log_messages.append(f"Model cached at {model_path}")
            tokenizer_source = str(model_path)
            tokenizer_local = True

        num_layers = int(self._config_get(model_config, "num_layers", 0))
        shard = Shard(
            self._model_id,
            start_layer=0,
            end_layer=num_layers,
            total_layers=num_layers + 1,
        )
        log_messages.append(f"Instantiating shard {shard}")
        model = Model(model_config, shard)
        load_safetensors(
            model,
            model_path,
            model_config,
            weight_device=os.getenv("TC_DEVICE", "CPU"),
            use_tied=bool(self._config_get(model_config, "tie_word_embeddings", False)),
        )
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_source, local_files_only=tokenizer_local)
        return model, model_config, tokenizer, model_path, log_messages

    def _generate_tokens_worker(
        self,
        model: Model,
        tokenizer: AutoTokenizer,
        input_ids: tg.Tensor,
        attention_mask: tg.Tensor,
        max_new: int,
        temp: float,
        top_k: int,
        top_p: float,
    ):
        start = time.time()
        out_tokens = generate(
            model=model,
            input_ids=input_ids,
            attention_mask=attention_mask,
            tokenizer=tokenizer,
            max_new_tokens=max_new,
            temp=temp,
            top_k=top_k,
            top_p=top_p,
            alpha_f=0.0,
            alpha_p=0.0,
        )
        elapsed = time.time() - start
        return out_tokens, elapsed


class ModelPickerScreen(ModalScreen[str | None]):
    """Modal popup allowing the user to enter a model identifier or path."""

    CSS_PATH = Path(__file__).with_name("chat_menu.tcss")

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
            input_field = Input(id="model-picker-input")
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
