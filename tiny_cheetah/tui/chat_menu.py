# Chat menu UI
# written with openai codex assistance
from __future__ import annotations

import asyncio
from datetime import datetime
import inspect
import os
import time
from pathlib import Path
from typing import Awaitable, Callable, Dict, List, Optional
import traceback

import tinygrad as tg
from transformers import AutoTokenizer
from rich.markup import escape

from tiny_cheetah.models.llm.helpers import load_safetensors
from tiny_cheetah.tui.helpers import streaming_generate
from tiny_cheetah.models.llm.model import Model
from tiny_cheetah.models.llm.model_config import ModelConfig
from tiny_cheetah.models.llm.shard import Shard
from tiny_cheetah.tui.widget.model_picker_screen import ModelPickerScreen
from tiny_cheetah.repos import RepoCustom

from textual.app import ComposeResult
from textual.binding import Binding
from textual.containers import Container
from textual.events import Mount
from textual.message import Message
from textual.message_pump import MessagePump
from textual.screen import Screen
from textual.widgets import Button, Footer, Header, Input, Label, RichLog, Static


class ChatModelSelected(Message):
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

    def __init__(self, default_model: str | None = None, offline: bool = False) -> None:
        super().__init__()
        self._offline = offline
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
        self._generating_resp = False
        self.out_tokens: List[int] = []
        self._loading_in_progress = False
        self._loading_task: Optional[asyncio.Task] = None
        self._generation_task: Optional[asyncio.Task] = None
        self._post_load_callbacks: List[Callable[[], Optional[Awaitable[None]]]] = []
        self._pending_generation = False
        self._load_button: Optional[Button] = None

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
                        load_button = Button("Load Model", id="load-model")
                        self._load_button = load_button
                        yield load_button
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

    async def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "open-model-picker":
            self._open_model_picker()
        elif event.button.id == "load-model":
            if not self._model_id:
                self._append_system("Select a model first.")
                return
            if self._model is not None and self._tokenizer is not None:
                self._append_system("Model already loaded.")
                return
            await self._start_model_load()
        elif event.button.id == "chat-back":
            self._clear_model()
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

    async def on_input_submitted(self, event: Input.Submitted) -> None:
        if event.input.id == "chat-input":
            content = event.value.strip()
            if not content:
                return
            event.input.value = ""
            if not self._model_id:
                self._append_system("Select a model before chatting (Ctrl+S).")
                return
            if self._generating_resp:
                self._append_system("Model is generating a response; please wait...")
                return
            if self._loading_in_progress:
                self._append_system("Model load in progress. Please wait…")
                return

            self._append_user(content)
            self._history.append({"role": "user", "content": content})

            if self._model is None or self._tokenizer is None:
                self._append_system("Loading model now...")
                self._pending_generation = True
                await self._start_model_load(self._continue_generation)
                return

            self._pending_generation = False
            self._schedule_generation()


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

    def _set_load_button_enabled(self, enabled: bool) -> None:
        if self._load_button is not None:
            self._load_button.disabled = not enabled

    def _schedule_generation(self) -> None:
        if self._generation_task is not None and not self._generation_task.done():
            return
        if self._model is None or self._tokenizer is None:
            return
        if self._generating_resp:
            return
        task = asyncio.create_task(self._generate_response())
        self._generation_task = task
        task.add_done_callback(self._on_generation_task_done)

    def _on_generation_task_done(self, task: asyncio.Future) -> None:
        self._generation_task = None
        if task.cancelled():
            return
        try:
            task.result()
        except Exception as exc:
            traceback.print_exception(type(exc), exc, exc.__traceback__)
            self._append_system(f"Error: {exc}")

    def _append_model(self, content: str) -> None:
        if self._chat_log is not None:
            label = escape(self._model_id or "Model")
            text = escape(content)
            self._chat_log.write(
                f"({datetime.now().strftime('%Y-%m-%d %H:%M:%S')}) [bold][#ff0000]{label}[/][/bold]: {text}\n"
            )

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

    async def _start_model_load(
        self,
        on_complete: Optional[Callable[[], Optional[Awaitable[None]]]] = None
    ) -> None:
        if self._loading_in_progress:
            self._append_system("Model load already in progress…")
            if on_complete is not None:
                self._post_load_callbacks.append(on_complete)
            return
        if not self._model_id:
            self._append_system("No model selected to load.")
            return

        if on_complete is not None:
            self._post_load_callbacks.append(on_complete)

        self._loading_in_progress = True
        self._set_load_button_enabled(False)
        self._append_system(f"Loading model '{self._model_id}'...")
        self._log_sys_msg("Preparing model load…")

        async def load_task() -> None:
            try:
                model, model_config, tokenizer, model_path, elapsed = await self._load_model_async()
            except Exception as exc:  # pragma: no cover - runtime safety
                await self._handle_load_failure(str(exc))
                return
            await self._handle_load_success(model, model_config, tokenizer, model_path, elapsed)

        self._loading_task = asyncio.create_task(load_task())
        self._loading_task.add_done_callback(self._on_load_task_finished)

    async def _handle_load_failure(self, message: str) -> None:
        self._log_sys_msg(f"Error loading model: {message}")
        self._loading_in_progress = False
        self._set_load_button_enabled(True)
        self._post_load_callbacks.clear()
        self._pending_generation = False
        if self._input is not None:
            self._input.focus()

    async def _handle_load_success(
        self,
        model: Model,
        model_config: object,
        tokenizer: AutoTokenizer,
        model_path: Path,
        elapsed: float
    ) -> None:
        ready_msg = f"Model ready in {elapsed:.1f}s."
        self._log_sys_msg(ready_msg)
        self._loading_in_progress = False
        self._set_load_button_enabled(True)
        self._model = model
        self._model_config = model_config
        self._tokenizer = tokenizer
        self._model_cache_path = model_path
        if self._input is not None:
            self._input.focus()
        callbacks = list(self._post_load_callbacks)
        self._post_load_callbacks.clear()
        for callback in callbacks:
            result = callback()
            if inspect.isawaitable(result):
                await result

    def _on_load_task_finished(self, task: asyncio.Future) -> None:
        self._loading_task = None
        try:
            task.result()
        except Exception:
            # Exceptions are reported in the load task itself.
            pass

    def _log_sys_msg(self, message: str) -> None:
        self._append_system(message)
    
    async def _log_sys_msg_async(self, message: str) -> Awaitable[None]:
        await asyncio.to_thread(self._append_system, message)

    def _continue_generation(self) -> Optional[Awaitable[None]]:
        if not self._pending_generation:
            return None
        if self._model is None or self._tokenizer is None:
            return None
        if self._generating_resp:
            return None
        self._pending_generation = False
        self._schedule_generation()
        return None

    async def _load_model_async(self) -> tuple[Model, object, AutoTokenizer, Path, float]:
        start = time.time()
        sanitized = self._model_id.replace("/", "__")
        cache_path = (Path.home() / ".cache" / "tiny_cheetah_models") / sanitized
        candidate_path = Path(self._model_id).expanduser()

        tokenizer_source: str
        tokenizer_local: bool
        self._log_sys_msg(f"Resolving model '{self._model_id}'")

        resolved_path: Path | None = None
        if candidate_path.exists():
            resolved_path = candidate_path
        elif cache_path.exists():
            resolved_path = cache_path

        if resolved_path is not None:
            self._log_sys_msg_async(f"Loading local model from {resolved_path}")
            # model_path, model_config = await asyncio.to_thread(self._load_local_config, resolved_path)
            model_path, model_config = self._load_local_config(resolved_path)
            tokenizer_source = str(model_path)
            tokenizer_local = True
        elif self._offline:
            raise RuntimeError("Offline mode requires cached model weights.")
        else:
            self._log_sys_msg(f"Downloading model from Hugging Face: {self._model_id}")
            if RepoCustom is None:
                raise RuntimeError("huggingface_hub is required to download models.")
            repo = RepoCustom(self._model_id)
            model_path, model_config, repo_messages = repo.download()
            for msg in repo_messages:
                self._log_sys_msg(f"[download] {msg}")
            self._log_sys_msg(f"Model cached at {model_path}")
            tokenizer_source = str(model_path)
            tokenizer_local = True

        num_layers = int(self._config_get(model_config, "num_layers", 0))
        shard = Shard(
            self._model_id,
            start_layer=0,
            end_layer=num_layers,
            total_layers=num_layers + 1,
        )
        await self._log_sys_msg_async(f"Instantiating model wtih shard {shard}")
        # causing segmentation fault on macosx METAL
        # model = await asyncio.to_thread(Model, model_config, shard)
        model = Model(model_config, shard)
        await self._log_sys_msg_async(f"Model instantiated. {model}")
        await self._log_sys_msg_async("Loading weights…")
        weight_device = os.getenv("TC_DEVICE", "CPU")
        use_tied = bool(self._config_get(model_config, "tie_word_embeddings", False))
        # await asyncio.to_thread(
        #     load_safetensors,
        #     model,
        #     model_path,
        #     model_config,
        #     weight_device=weight_device,
        #     use_tied=use_tied,
        # )
        load_safetensors(
            model,
            model_path,
            model_config,
            weight_device=weight_device,
            use_tied=use_tied,
        )
        await self._log_sys_msg_async("Weights loaded.")
        await self._log_sys_msg_async("Loading tokenizer…")
        # tokenizer = await asyncio.to_thread(
        #     AutoTokenizer.from_pretrained,
        #     tokenizer_source,
        #     local_files_only=tokenizer_local,
        # )
        tokenizer = AutoTokenizer.from_pretrained(
            tokenizer_source,
            local_files_only=tokenizer_local,
        )
        await self._log_sys_msg_async("Tokenizer ready.")
        elapsed = time.time() - start
        return model, model_config, tokenizer, model_path, elapsed

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

    async def _generate_response(self, max_new_tokens: int = 4096) -> None:
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

        if hasattr(self._model, "reset_kv_cache"):
            self._model.reset_kv_cache()

        max_new = max_new_tokens
        temp = self._config_get(self._model_config, "temperature", 0.7) or 0.7
        top_k = self._config_get(self._model_config, "top_k", 0) or 0
        top_p = self._config_get(self._model_config, "top_p", 0.8) or 0.8

        if self._generating_resp:
            self._append_system("Model is already generating a response.")
            return

        self._generating_resp = True
        try:
            await asyncio.to_thread(self._append_model, "Thinking...")
            result = streaming_generate(
                self._model,
                input_ids,
                attention_mask,
                self._tokenizer,
                max_new_tokens=max_new,
                temp=temp,
                top_k=top_k,
                top_p=top_p,
            )
        finally:
            self._generating_resp = False
        out_tokens, elapsed = result
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
        if hasattr(self, "_model") and self._model is not None and hasattr(self._model, "reset_kv_cache"):
            self._model.reset_kv_cache()
        self._model = None
        self._model_config = None
        self._tokenizer = None
        self._model_cache_path = None
        self._history.clear()
        self._generating_resp = False
        if self._generation_task is not None and not self._generation_task.done():
            self._generation_task.cancel()
        self._generation_task = None
        self._pending_generation = False
        if not self._loading_in_progress:
            self._set_load_button_enabled(True)
        if self._model_label is not None:
            self._model_label.update("<select>")
        self._append_system("Model cleared. Select and load a model to continue.")
