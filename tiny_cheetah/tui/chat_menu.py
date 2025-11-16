# Chat menu UI
# written with openai codex assistance
from __future__ import annotations

import asyncio
from datetime import datetime
import inspect
import os
import time
from pathlib import Path
from typing import Awaitable, Callable, List, Optional
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
from tiny_cheetah.tui.chat_log_storage import ChatLogStorage, ChatLogSummary, ChatMessage

from textual.app import ComposeResult
from textual.binding import Binding
from textual.containers import Container
from textual.events import Mount
from textual.message import Message
from textual.message_pump import MessagePump
from textual.screen import Screen, ModalScreen
from textual.widgets import Button, Footer, Header, Input, Label, ListItem, ListView, RichLog, Static

import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
MAX_RESTORED_MESSAGES = 20

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
        self._history: List[dict[str, str]] = []
        self._generating_resp = False
        self.out_tokens: List[int] = []
        self._loading_in_progress = False
        self._loading_task: Optional[asyncio.Task] = None
        self._generation_task: Optional[asyncio.Task] = None
        self._post_load_callbacks: List[Callable[[], Optional[Awaitable[None]]]] = []
        self._pending_generation = False
        self._load_button: Optional[Button] = None
        self._chat_log_list: Optional[ListView] = None
        self._log_storage = ChatLogStorage()
        self._current_log_id: Optional[int] = None

    def compose(self) -> ComposeResult:
        yield Header(show_clock=True)
        with Container(id="chat-root"):
            with Container(id="chat-body"):
                with Container(id="chat-history"):
                    yield Label("Chat Logs", classes="panel-title")
                    history_list = ListView(id="chat-log-list")
                    self._chat_log_list = history_list
                    yield history_list
                    with Container(id="chat-log-actions"):
                        yield Button("New Log", id="new-chat-log", variant="primary")
                        yield Button("Load Log", id="load-chat-log")
                        yield Button("Rename Log", id="rename-chat-log")
                        yield Button("Delete Log", id="delete-chat-log", variant="error")
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

    async def on_mount(self, _: Mount) -> None:
        self._chat_log = self.query_one("#chat-log", RichLog)
        self._input = self.query_one("#chat-input", Input)
        if self._input is not None:
            self.call_after_refresh(self._input.focus)
        if self._model_label is not None:
            self._model_label.update(self._model_id or "<select>")
        await self._initialize_chat_logs()

    def action_open_model_picker(self) -> None:
        self._open_model_picker()

    async def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "open-model-picker":
            self._open_model_picker()
        elif event.button.id == "load-model":
            if not self._model_id:
                self._log_sys_msg("Select a model first.")
                return
            if self._model is not None and self._tokenizer is not None:
                self._log_sys_msg("Model already loaded.")
                return
            await self._start_model_load()
        elif event.button.id == "chat-back":
            self._clear_model()
            self.app.pop_screen()
        elif event.button.id == "clear-model":
            self._clear_model()
        elif event.button.id == "new-chat-log":
            await self._create_new_log()
        elif event.button.id == "load-chat-log":
            await self._load_selected_chat_log()
        elif event.button.id == "rename-chat-log":
            self._prompt_rename_selected_log()
        elif event.button.id == "delete-chat-log":
            self._confirm_delete_selected_log()

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
        asyncio.create_task(self._update_current_log_model_async())
        self._log_sys_msg(f"Model set to '{result}'.")
        self._log_sys_msg("Click 'Load Model' to download weights into memory.")

    async def on_input_submitted(self, event: Input.Submitted) -> None:
        if event.input.id == "chat-input":
            content = event.value.strip()
            if not content:
                return
            event.input.value = ""
            if self._current_log_id is None:
                self._log_sys_msg("Create or load a chat log before chatting.")
                return
            if not self._model_id:
                self._log_sys_msg("Select a model before chatting (Ctrl+S).")
                return
            if self._generating_resp:
                self._log_sys_msg("Model is generating a response; please wait...")
                return
            if self._loading_in_progress:
                self._log_sys_msg("Model load in progress. Please wait…")
                return

            self._append_user(content)
            self._history.append({"role": "user", "content": content})

            if self._model is None or self._tokenizer is None:
                self._log_sys_msg("Loading model now...")
                self._pending_generation = True
                await self._start_model_load(self._continue_generation)
                return

            self._pending_generation = False
            self._schedule_generation()


    def _append_user(
        self,
        content: str,
        persist: bool = True,
        timestamp: Optional[str] = None
    ) -> None:
        if self._chat_log is None:
            return
        entry_time = timestamp or datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        text = escape(content)
        self._write_to_chat_log(
            f"({entry_time}) [bold][#3bd6ee]User[/][/bold]: {text}\n"
        )
        if persist:
            self._record_message("user", content, entry_time)

    def _append_system(
        self,
        content: str,
        persist: bool = True,
        timestamp: Optional[str] = None
    ) -> None:
        if self._chat_log is None:
            return
        entry_time = timestamp or datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        text = escape(content)
        self._write_to_chat_log(
            f"({entry_time}) [bold][#00FF00]System[/][/bold]: {text}\n"
        )
        if persist:
            self._record_message("system", content, entry_time)

    def _append_model(
        self,
        content: str,
        persist: bool = True,
        timestamp: Optional[str] = None
    ) -> None:
        if self._chat_log is None:
            return
        entry_time = timestamp or datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        label = escape(self._model_id or "Model")
        text = escape(content)
        self._write_to_chat_log(
            f"({entry_time}) [bold][#ff0000]{label}[/][/bold]: {text}\n"
        )
        if persist:
            self._record_message("assistant", content, entry_time)
            
    def _write_to_chat_log(self, message: str) -> None:
        if self._chat_log is None:
            return
        self._chat_log.write(message)
        scroll_end = getattr(self._chat_log, "scroll_end", None)
        if callable(scroll_end):
            scroll_end(animate=False)
            return
        action_scroll_end = getattr(self._chat_log, "action_scroll_end", None)
        if callable(action_scroll_end):
            action_scroll_end()

    def _record_message(self, role: str, content: str, timestamp: str) -> None:
        if self._current_log_id is None:
            return
        try:
            self._log_storage.append_message(self._current_log_id, role, content, timestamp)
        except Exception as exc:  # pragma: no cover - logging defensive path
            logger.exception("Failed to record chat message: %s", exc)

    def _set_load_button_enabled(self, enabled: bool) -> None:
        if self._load_button is not None:
            self._load_button.disabled = not enabled

    async def _initialize_chat_logs(self) -> None:
        summaries = self._log_storage.list_logs()
        if not summaries:
            default_name = datetime.now().strftime("Session %Y-%m-%d %H:%M:%S")
            new_id = self._log_storage.create_log(default_name, self._model_id or "")
            summaries = self._log_storage.list_logs()
            self._current_log_id = new_id
        else:
            self._current_log_id = summaries[0]["id"]
        summaries = await self._refresh_chat_log_list(select_id=self._current_log_id, summaries=summaries)
        if self._current_log_id is None:
            return
        current_log = self._log_storage.get_log(self._current_log_id)
        if current_log is not None:
            stored_model = current_log["model_id"] or ""
            if stored_model:
                self._model_id = stored_model
                if self._model_label is not None:
                    self._model_label.update(self._model_id or "<select>")
        messages = self._log_storage.get_messages(self._current_log_id, limit=MAX_RESTORED_MESSAGES)
        if self._chat_log is not None:
            self._chat_log.clear()
        self._history.clear()
        self._restore_messages_to_view(messages, persist=False)

    async def _refresh_chat_log_list(
        self,
        select_id: Optional[int] = None,
        summaries: Optional[List[ChatLogSummary]] = None
    ) -> List[ChatLogSummary]:
        if summaries is None:
            summaries = self._log_storage.list_logs()
        list_view = self._chat_log_list
        if list_view is None:
            return summaries
        await list_view.clear()
        for summary in summaries:
            item = self._build_log_item(summary)
            await list_view.mount(item)
        target_id = select_id
        if target_id is None and summaries:
            target_id = summaries[0]["id"]
        if target_id is not None:
            self._highlight_log(target_id)
        return summaries

    def _build_log_item(self, summary: ChatLogSummary) -> ListItem:
        label_text = self._format_log_item_label(summary)
        label = Label(label_text)
        return ListItem(label, id=f"log-{summary['id']}")

    @staticmethod
    def _format_log_item_label(summary: ChatLogSummary) -> str:
        name = summary["name"]
        model = summary["model_id"] or "no model"
        stamp = summary["updated_at"]
        stamp = stamp.split(".")[0]
        return f"{name} ({model}) [{stamp}]"

    def _highlight_log(self, log_id: int) -> None:
        if self._chat_log_list is None:
            return
        for index, child in enumerate(self._chat_log_list.children):
            if getattr(child, "id", "") == f"log-{log_id}":
                try:
                    self._chat_log_list.index = index  # type: ignore[attr-defined]
                except AttributeError:
                    pass
                break

    def _get_selected_log_id(self) -> Optional[int]:
        if self._chat_log_list is None:
            return None
        index = getattr(self._chat_log_list, "index", None)
        if index is None:
            return None
        try:
            child = self._chat_log_list.children[index]
        except IndexError:
            return None
        return self._log_id_from_widget(child)

    async def _create_new_log(self) -> None:
        name = datetime.now().strftime("Session %Y-%m-%d %H:%M:%S")
        try:
            new_id = self._log_storage.create_log(name, self._model_id or "")
        except Exception as exc:  # pragma: no cover - defensive logging
            self._log_sys_msg(f"Failed to create new log: {exc}")
            return
        self._current_log_id = new_id
        await self._refresh_chat_log_list(select_id=new_id)
        if self._chat_log is not None:
            self._chat_log.clear()
        self._history.clear()
        self._log_sys_msg(f"Created log '{name}'.")
        self.call_after_refresh(lambda: self._open_log_name_modal(new_id, "Name Chat Log", name))

    def _prompt_rename_selected_log(self) -> None:
        log_id = self._get_selected_log_id()
        if log_id is None:
            log_id = self._current_log_id
        if log_id is None:
            self._log_sys_msg("Select a chat log to rename.")
            return
        summary = self._log_storage.get_log(log_id)
        if summary is None:
            self._log_sys_msg("Chat log not found.")
            return
        self._open_log_name_modal(log_id, "Rename Chat Log", summary["name"])

    def _open_log_name_modal(self, log_id: int, title: str, initial: str) -> None:
        modal = ChatLogNameModal(title, initial)
        self.app.push_screen(modal, lambda value: self._apply_log_name(log_id, value))

    def _apply_log_name(self, log_id: int, result: Optional[str]) -> None:
        if result is None:
            return
        name = result.strip()
        if not name:
            self._log_sys_msg("Log name cannot be blank.")
            return
        try:
            self._log_storage.rename_log(log_id, name)
        except Exception as exc:  # pragma: no cover - defensive logging
            self._log_sys_msg(f"Failed to rename chat log: {exc}")
            return
        self._log_sys_msg(f"Log renamed to '{name}'.")
        self._current_log_id = log_id
        asyncio.create_task(self._refresh_chat_log_list(select_id=log_id))

    def _confirm_delete_selected_log(self) -> None:
        log_id = self._get_selected_log_id()
        if log_id is None:
            log_id = self._current_log_id
        if log_id is None:
            self._log_sys_msg("Select a chat log to delete.")
            return
        summary = self._log_storage.get_log(log_id)
        if summary is None:
            self._log_sys_msg("Chat log not found.")
            return
        modal = ConfirmModal(
            "Delete Chat Log",
            f"Delete '{summary['name']}'? This cannot be undone.",
            confirm_label="Delete",
            confirm_variant="error",
        )
        self.app.push_screen(modal, lambda confirmed: self._finalize_delete_log(log_id, confirmed))

    def _finalize_delete_log(self, log_id: int, confirmed: Optional[bool]) -> None:
        if not confirmed:
            return
        try:
            self._log_storage.delete_log(log_id)
        except Exception as exc:  # pragma: no cover - defensive logging
            self._log_sys_msg(f"Failed to delete chat log: {exc}")
            return
        removed_current = self._current_log_id == log_id
        if removed_current:
            self._current_log_id = None
            self._history.clear()
            if self._chat_log is not None:
                self._chat_log.clear()
            self._clear_model(persist=False)
        self._log_sys_msg("Chat log deleted.")
        asyncio.create_task(self._post_delete_refresh(removed_current))

    async def _post_delete_refresh(self, reload_current: bool) -> None:
        summaries = self._log_storage.list_logs()
        if not summaries:
            default_name = datetime.now().strftime("Session %Y-%m-%d %H:%M:%S")
            new_id = self._log_storage.create_log(default_name, self._model_id or "")
            summaries = self._log_storage.list_logs()
            self._current_log_id = new_id
            reload_current = True
        valid_ids = {summary["id"] for summary in summaries}
        if self._current_log_id not in valid_ids:
            self._current_log_id = summaries[0]["id"] if summaries else None
            reload_current = True
        await self._refresh_chat_log_list(select_id=self._current_log_id, summaries=summaries)
        if reload_current and self._current_log_id is not None:
            summary = self._log_storage.get_log(self._current_log_id)
            if summary is not None:
                self._model_id = summary["model_id"] or ""
                if self._model_label is not None:
                    self._model_label.update(self._model_id or "<select>")
            messages = self._log_storage.get_messages(self._current_log_id, limit=MAX_RESTORED_MESSAGES)
            if self._chat_log is not None:
                self._chat_log.clear()
            self._history.clear()
            self._restore_messages_to_view(messages, persist=False)

    def on_list_view_highlighted(self, event: ListView.Highlighted) -> None:
        log_id = self._log_id_from_widget(event.item)
        if log_id is not None:
            self._current_log_id = log_id

    async def on_list_view_selected(self, event: ListView.Selected) -> None:
        log_id = self._log_id_from_widget(event.item)
        if log_id is None:
            return
        event.stop()
        await self._handle_chat_log_load(log_id)

    async def _load_selected_chat_log(self) -> None:
        log_id = self._get_selected_log_id()
        if log_id is None:
            self._log_sys_msg("Select a chat log to load.")
            return
        await self._handle_chat_log_load(log_id)

    async def _handle_chat_log_load(self, log_id: int) -> None:
        summary = self._log_storage.get_log(log_id)
        if summary is None:
            self._log_sys_msg("Chat log not found.")
            return
        messages = self._log_storage.get_messages(log_id, limit=MAX_RESTORED_MESSAGES)
        self._current_log_id = log_id
        self._highlight_log(log_id)
        self._clear_model(persist=False)
        if self._chat_log is not None:
            self._chat_log.clear()
        self._history.clear()
        self._model_id = summary["model_id"] or ""
        if self._model_label is not None:
            self._model_label.update(self._model_id or "<select>")
        self._restore_messages_to_view(messages, persist=False)
        if self._model_id:
            await self._start_model_load()
        else:
            self._log_sys_msg("No model associated with this log. Select one to continue.")

    def _restore_messages_to_view(self, messages: List[ChatMessage], persist: bool) -> None:
        for message in messages:
            role = message["role"]
            content = message["content"]
            timestamp = message["timestamp"]
            if role == "user":
                self._append_user(content, persist=persist, timestamp=timestamp)
                self._history.append({"role": "user", "content": content})
            elif role == "assistant":
                self._append_model(content, persist=persist, timestamp=timestamp)
                self._history.append({"role": "assistant", "content": content})
            else:
                self._append_system(content=content, persist=persist, timestamp=timestamp)
        if len(self._history) > 6:
            self._history = self._history[-6:]

    async def _update_current_log_model_async(self) -> None:
        if self._current_log_id is None:
            return
        try:
            self._log_storage.set_log_model(self._current_log_id, self._model_id or "")
            await self._refresh_chat_log_list(select_id=self._current_log_id)
        except Exception as exc:  # pragma: no cover - defensive logging
            logger.exception("Failed to update log model mapping: %s", exc)

    @staticmethod
    def _parse_log_item_id(item_id: Optional[str]) -> Optional[int]:
        if not item_id:
            return None
        if not item_id.startswith("log-"):
            return None
        try:
            return int(item_id.split("-", 1)[1])
        except (IndexError, ValueError):
            return None

    def _log_id_from_widget(self, widget: Optional[ListItem]) -> Optional[int]:
        if widget is None:
            return None
        return self._parse_log_item_id(getattr(widget, "id", None))

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
            self._log_sys_msg(f"Error: {exc}")

    async def _start_model_load(
        self,
        on_complete: Optional[Callable[[], Optional[Awaitable[None]]]] = None
    ) -> None:
        if self._loading_in_progress:
            self._log_sys_msg("Model load already in progress…")
            if on_complete is not None:
                self._post_load_callbacks.append(on_complete)
            return
        if not self._model_id:
            self._log_sys_msg("No model selected to load.")
            return

        if on_complete is not None:
            self._post_load_callbacks.append(on_complete)

        self._loading_in_progress = True
        self._set_load_button_enabled(False)
        await self._log_sys_msg_async(f"Loading model '{self._model_id}'...")
        await self._log_sys_msg_async("Preparing model load…")

        # loaded_model = self._load_model()
        try:
            model, model_config, tokenizer, model_path, elapsed = await self._load_model_async()
        except Exception as exc:
            return await self._h_model_load_failure(f"Model load failed: {exc}\n traceback: {traceback.format_exc()}")
        else:
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

            # handle post-load callbacks
            callbacks = list(self._post_load_callbacks)
            self._post_load_callbacks.clear()
            for callback in callbacks:
                result = callback()
                if inspect.isawaitable(result):
                    await result
    
    async def _h_model_load_failure(self, message: str) -> None:
        self._log_sys_msg(message)
        self._loading_in_progress = False
        self._set_load_button_enabled(True)
        if self._input is not None:
            self._input.focus()

    def _log_sys_msg(self, message: str, *, persist: bool = False) -> None:
        self._append_system(message, persist=persist)
    
    async def _log_sys_msg_async(self, message: str) -> Awaitable[None]:
        await asyncio.to_thread(self._append_system, message, persist=False)

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

        local_model = False
        if resolved_path is not None and any(resolved_path.glob("*.*")):
            self._log_sys_msg_async(f"Loading local model from {resolved_path}")
            local_model = True
            # model_path, model_config = await asyncio.to_thread(self._load_local_config, resolved_path)
            model_path, model_config = self._load_local_config(resolved_path)
            tokenizer_source = str(model_path)
            tokenizer_local = True
        elif self._offline:
            return await self._h_model_load_failure("Model not found in cache for offline mode.")
        
        if not local_model:
            logger.info(f"not local model downloading {self._model_id}")
            self._log_sys_msg(f"Downloading model from Hugging Face: {self._model_id}")
            if RepoCustom is None:
                return await self._h_model_load_failure("Custom repo handler not available. Please use huggingface_hub library.")
            repo = RepoCustom(self._model_id)
            async def download_progress(message: str) -> None:
                await self._log_sys_msg_async(f"[download] {message}")
            model_path, model_config, _ = await repo.download(progress_callback=download_progress)
            await self._log_sys_msg_async(f"repo download done: {model_path}, {model_config}")
            self._log_sys_msg(f"Model cached at {model_path}")
            tokenizer_source = str(model_path)
            tokenizer_local = True

        num_layers = model_config["num_layers"]
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
        use_tied = model_config["tie_word_embeddings"]
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
        return directory, config.config

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
        temp = self._model_config["temperature"]
        top_k = self._model_config["top_k"]
        top_p = self._model_config["top_p"]

        if self._generating_resp:
            self._log_sys_msg("Model is already generating a response.")
            return

        self._generating_resp = True
        try:
            await asyncio.to_thread(self._append_model, "Thinking...", persist=False)
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

            # causes sql thread issue on mac
            # result = await asyncio.to_thread(
            #     streaming_generate,
            #     self._model,
            #     input_ids,
            #     attention_mask,
            #     self._tokenizer,
            #     max_new_tokens=max_new,
            #     temp=temp,
            #     top_k=top_k,
            #     top_p=top_p,
            # )
        except Exception as exc:
            self._generating_resp = False
            self._log_sys_msg(f"Error during generation: {exc}")
            self._log_sys_msg(f"Traceback: {traceback.format_exc()}")
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

    def _clear_model(self, *, persist: bool = True) -> None:
        self._log_sys_msg("Clearing loaded model.", persist=persist)
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
        self._log_sys_msg("Model cleared. Select and load a model to continue.", persist=persist)


class ChatLogNameModal(ModalScreen[Optional[str]]):
    """Modal dialog to capture a chat log name."""

    def __init__(self, title: str, initial: str = "") -> None:
        super().__init__(id="chat-log-name-modal")
        self._title = title
        self._initial = initial
        self._input: Optional[Input] = None

    def compose(self) -> ComposeResult:
        with Container(id="chat-log-name-modal-container"):
            yield Label(self._title, id="chat-log-name-title")
            self._input = Input(id="chat-log-name-field", placeholder="Enter log name…")
            yield self._input
            with Container(id="chat-log-name-buttons"):
                yield Button("Cancel", id="chat-log-name-cancel")
                yield Button("Save", id="chat-log-name-save", variant="primary")

    def on_mount(self) -> None:
        if self._input is not None:
            self._input.value = self._initial
            self.call_after_refresh(self._input.focus)

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "chat-log-name-cancel":
            self.dismiss(None)
        elif event.button.id == "chat-log-name-save":
            value = self._input.value.strip() if self._input else ""
            self.dismiss(value or None)


class ConfirmModal(ModalScreen[Optional[bool]]):
    """Generic confirmation modal."""

    def __init__(
        self,
        title: str,
        message: str,
        *,
        confirm_label: str = "OK",
        confirm_variant: str = "primary",
    ) -> None:
        super().__init__(id="chat-log-confirm-modal")
        self._title = title
        self._message = message
        self._confirm_label = confirm_label
        self._confirm_variant = confirm_variant

    def compose(self) -> ComposeResult:
        with Container(id="chat-log-confirm-modal-container"):
            yield Label(self._title, id="chat-log-confirm-title")
            yield Static(self._message, id="chat-log-confirm-message")
            with Container(id="chat-log-confirm-buttons"):
                yield Button("Cancel", id="chat-log-confirm-cancel")
                yield Button(self._confirm_label, id="chat-log-confirm-accept", variant=self._confirm_variant)

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "chat-log-confirm-cancel":
            self.dismiss(False)
        elif event.button.id == "chat-log-confirm-accept":
            self.dismiss(True)
