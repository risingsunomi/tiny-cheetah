from __future__ import annotations

import math
import os
import re
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from queue import Empty, Queue
from threading import Event, Thread
from typing import Dict, Iterable, List, Optional

from textual.app import ComposeResult
from textual.binding import Binding
from textual.containers import Container, Vertical, VerticalScroll
from textual.screen import ModalScreen, Screen
from textual.widgets import Button, Checkbox, Header, Input, Label, Log, Static

try:
    import psutil  # type: ignore
except Exception:  # pragma: no cover - psutil is optional at runtime
    psutil = None


@dataclass
class TrainingStats:
    status: str = "Idle"
    step: int = 0
    epoch: int = 0
    total_epochs: Optional[int] = None
    loss: Optional[float] = None
    mean_loss: Optional[float] = None
    tokens: int = 0
    tok_rate: Optional[float] = None


def chunked(sequence: List[Dict[str, object]], size: int) -> Iterable[List[Dict[str, object]]]:
    """Yield successive chunks from a list."""
    for index in range(0, len(sequence), size):
        yield sequence[index:index + size]


SETTINGS_FIELDS: List[Dict[str, object]] = [
    {"name": "model-id", "label": "Model ID", "placeholder": "Hugging Face repo", "default": ""},
    {"name": "custom-model-id", "label": "Custom Model ID", "placeholder": "Local model name", "default": ""},
    {"name": "tokenizer-id", "label": "Tokenizer ID", "placeholder": "Tokenizer repo", "default": ""},
    {"name": "tokenizer-file", "label": "Tokenizer File", "placeholder": "Path to tokenizer.json", "default": ""},
    {"name": "config-path", "label": "Config Path", "placeholder": "config.json path", "default": ""},
    {"name": "generation-config-path", "label": "Generation Config", "placeholder": "generation_config.json path", "default": ""},
    {"name": "weights-dir", "label": "Weights Dir", "placeholder": "Safetensors directory", "default": ""},
    {"name": "data-path", "label": "Data Path", "placeholder": "Local dataset path", "default": ""},
    {"name": "dataset-id", "label": "Dataset ID", "placeholder": "HF dataset identifier", "default": ""},
    {"name": "dataset-cache-dir", "label": "Dataset Cache Dir", "placeholder": "Cache directory", "default": ""},
    {"name": "max-dataset-entries", "label": "Max Entries", "placeholder": "Limit dataset size", "default": ""},
    {"name": "seq-length", "label": "Seq Length", "placeholder": "Default 256", "default": "256"},
    {"name": "batch-size", "label": "Batch Size", "placeholder": "Default 2", "default": "2"},
    {"name": "epochs", "label": "Epochs", "placeholder": "Default 1", "default": "1"},
    {"name": "lr", "label": "Learning Rate", "placeholder": "Default 1e-4", "default": "1e-4"},
    {"name": "device", "label": "Device", "placeholder": "Default CPU", "default": os.environ.get("TC_DEVICE", "CPU")},
    {"name": "gradient-accumulation", "label": "Grad Accum", "placeholder": "Default 1", "default": "1"},
    {"name": "save-dir", "label": "Save Dir", "placeholder": "Checkpoint output", "default": ""},
    {"name": "offline", "label": "Offline Mode", "type": "checkbox", "default": False},
    {"name": "from-scratch", "label": "From Scratch", "type": "checkbox", "default": False},
]


class TrainingProcess:
    """Lightweight wrapper around the training subprocess."""

    def __init__(self, script_path: Path, args: List[str]) -> None:
        self.script_path = script_path
        self.args = args
        self._queue: "Queue[Optional[str]]" = Queue()
        self._thread: Optional[Thread] = None
        self._process: Optional[subprocess.Popen[str]] = None
        self._stop = Event()

    def start(self) -> None:
        if self._thread is not None:
            raise RuntimeError("Training process already started.")

        command = [sys.executable, "-u", str(self.script_path), *self.args]

        def worker() -> None:
            try:
                self._process = subprocess.Popen(
                    command,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    text=True,
                    bufsize=1,
                )
                assert self._process.stdout is not None
                for raw_line in self._process.stdout:
                    if self._stop.is_set():
                        break
                    segments = raw_line.replace("\r", "\n").splitlines()
                    if not segments:
                        self._queue.put("")
                    else:
                        for segment in segments:
                            self._queue.put(segment)
                if self._process.poll() is None:
                    self._process.wait()
                exit_code = self._process.returncode
                if exit_code not in (0, None):
                    self._queue.put(f"[error] training exited with status {exit_code}")
            except FileNotFoundError as exc:
                self._queue.put(f"[error] Unable to launch training: {exc}")
            except Exception as exc:  # pragma: no cover - defensive
                self._queue.put(f"[error] {exc}")
            finally:
                self._queue.put(None)

        self._thread = Thread(target=worker, name="train-runner", daemon=True)
        self._thread.start()

    def terminate(self) -> None:
        self._stop.set()
        if self._process and self._process.poll() is None:
            try:
                self._process.terminate()
            except Exception:
                pass

    def is_running(self) -> bool:
        return self._process is not None and self._process.poll() is None

    def drain(self) -> Iterable[Optional[str]]:
        while True:
            try:
                yield self._queue.get_nowait()
            except Empty:
                break


class TrainScreen(Screen[None]):
    """Textual screen that presents training progress and controls."""

    CSS_PATH = Path(__file__).with_name("train_menu.tcss")

    BINDINGS = [
        Binding("escape", "pop_screen", "Back"),
    ]

    @staticmethod
    def _normalize_bool(value: object) -> bool:
        if isinstance(value, str):
            return value.strip().lower() in {"1", "true", "yes", "on", "y"}
        return bool(value)

    def __init__(self) -> None:
        super().__init__()
        self._stats = TrainingStats()
        self._training: Optional[TrainingProcess] = None
        self._poll_timer = None
        self._resource_timer = None
        self._stat_labels: Dict[str, Label] = {}
        self._resource_labels: Dict[str, Label] = {}
        self._active_args_label: Optional[Label] = None
        self._settings_summary: Dict[str, Label] = {}
        self._log: Optional[Log] = None
        self._settings: Dict[str, object] = {}
        for field in SETTINGS_FIELDS:
            name = str(field["name"])
            default = field.get("default")
            if field.get("type") == "checkbox":
                self._settings[name] = bool(default)
            else:
                self._settings[name] = "" if default is None else str(default)

    def compose(self) -> ComposeResult:
        yield Header(show_clock=True)
        with Container(id="train-root"):
            with Container(id="train-left"):
                yield Static("Training Console", id="model-banner")
                yield Log(id="train-log", highlight=False, auto_scroll=True)
                with Container(id="control-row"):
                    yield Button("Start", id="start-training", variant="primary")
                    yield Button("Stop", id="stop-training", disabled=True)
                    yield Button("Back", id="back-to-menu")
            with Container(id="train-right"):
                with VerticalScroll(id="train-right-scroll"):
                    with Container(id="train-right-content"):
                        with Static(id="stats-panel"):
                            yield Label("Status: Idle", id="stat-status")
                            yield Label("Step: 0", id="stat-step")
                            yield Label("Epoch: 0", id="stat-epoch")
                            yield Label("Loss: --", id="stat-loss")
                            yield Label("Tokens: 0", id="stat-tokens")
                            yield Label("Tok/s: --", id="stat-tok-rate")
                        with Static(id="settings-panel"):
                            yield Label("Settings", id="settings-title")
                            yield Label("Model: --", id="settings-model")
                            yield Label("Dataset: --", id="settings-data")
                            yield Button("Edit Settings", id="open-settings")
                            self._active_args_label = Label("Active Args: --", id="active-args")
                            yield self._active_args_label
                        with Static(id="resource-panel"):
                            yield Label("CPU: --", id="resource-cpu")
                            yield Label("Memory: --", id="resource-ram")
                            yield Label("GPU: --", id="resource-gpu")

    def apply_default_settings(self, defaults: Dict[str, object]) -> None:
        for key, value in defaults.items():
            if key not in self._settings:
                continue
            current = self._settings[key]
            if isinstance(current, bool):
                self._settings[key] = self._normalize_bool(value)
            else:
                if value is None:
                    self._settings[key] = ""
                else:
                    self._settings[key] = str(value).strip()
        if getattr(self, "_settings_summary", None):
            self._update_settings_summary()

    def on_mount(self) -> None:
        self._log = self.query_one("#train-log", Log)
        if self._log is not None:
            self._log.clear()
        self._stat_labels = {
            "status": self.query_one("#stat-status", Label),
            "step": self.query_one("#stat-step", Label),
            "epoch": self.query_one("#stat-epoch", Label),
            "loss": self.query_one("#stat-loss", Label),
            "tokens": self.query_one("#stat-tokens", Label),
            "tok_rate": self.query_one("#stat-tok-rate", Label),
        }
        self._resource_labels = {
            "cpu": self.query_one("#resource-cpu", Label),
            "ram": self.query_one("#resource-ram", Label),
            "gpu": self.query_one("#resource-gpu", Label),
        }
        if self._active_args_label is None:
            self._active_args_label = self.query_one("#active-args", Label)
        self._settings_summary = {
            "model": self.query_one("#settings-model", Label),
            "data": self.query_one("#settings-data", Label),
        }
        self._update_settings_summary()

        self._poll_timer = self.set_interval(0.25, self._poll_training_output, pause=True)
        self._resource_timer = self.set_interval(1.0, self._update_resource_usage)

        # Prime CPU stats if psutil is available to avoid the initial 0.0 reading.
        if psutil is not None:
            _ = psutil.cpu_percent(interval=None)

    def on_unmount(self) -> None:
        if self._poll_timer is not None:
            self._poll_timer.stop()
            self._poll_timer = None
        if self._resource_timer is not None:
            self._resource_timer.stop()
            self._resource_timer = None
        if self._training is not None and self._training.is_running():
            self._training.terminate()

    def on_button_pressed(self, event: Button.Pressed) -> None:
        button_id = event.button.id
        if button_id == "start-training":
            self._start_training()
        elif button_id == "stop-training":
            self._stop_training()
        elif button_id == "back-to-menu":
            if self._training and self._training.is_running():
                self._append_log("[warn] Stop training before returning to the menu.")
            else:
                self.app.pop_screen()
        elif button_id == "open-settings":
            self._open_settings()

    def _open_settings(self) -> None:
        screen = TrainSettingsScreen(dict(self._settings))
        self.app.push_screen(screen, self._on_settings_result)

    def _on_settings_result(self, result: Optional[Dict[str, object]]) -> None:
        if not result:
            return
        for key, value in result.items():
            if key not in self._settings:
                continue
            if key in {"offline", "from-scratch"}:
                self._settings[key] = self._normalize_bool(value)
            else:
                self._settings[key] = "" if value is None else str(value).strip()
        self._update_settings_summary()

    def _start_training(self) -> None:
        if self._training is not None and self._training.is_running():
            self._append_log("[warn] Training is already running.")
            return

        args = self._collect_args()
        if args is None:
            return

        script_path = Path(__file__).resolve().parents[2] / "train.py"
        if not script_path.exists():
            self._append_log(f"[error] Could not locate train.py at {script_path}")
            return

        self._training = TrainingProcess(script_path, args)
        self._training.start()

        self._stats = TrainingStats(status="Running")
        self._update_stats_display()
        self._set_buttons(running=True)
        if self._poll_timer is not None:
            self._poll_timer.resume()

        if self._active_args_label is not None:
            joined = " ".join(args) if args else "--"
            self._active_args_label.update(f"Active Args: {joined}")

        self._append_log(f"[info] Launched training: {' '.join(args) if args else '(defaults)'}")

    def _stop_training(self) -> None:
        if self._training is None or not self._training.is_running():
            self._append_log("[warn] No active training process.")
            return
        self._training.terminate()
        self._append_log("[info] Terminating training process...")

    def _collect_args(self) -> Optional[List[str]]:
        args: List[str] = []

        def push(flag: str, value: str) -> None:
            args.extend([flag, value])

        settings = self._settings

        def get_str(key: str) -> str:
            value = settings.get(key, "")
            if isinstance(value, bool):
                return ""
            return str(value).strip()

        model_id = get_str("model-id")
        config_path = get_str("config-path")
        if not model_id and not config_path:
            self._append_log("[error] Provide either a --model-id or a --config-path.")
            return None

        if model_id:
            push("--model-id", model_id)
        if config_path:
            push("--config-path", config_path)

        text_fields = [
            ("custom-model-id", "--custom-model-id"),
            ("tokenizer-id", "--tokenizer-id"),
            ("tokenizer-file", "--tokenizer-file"),
            ("generation-config-path", "--generation-config-path"),
            ("weights-dir", "--weights-dir"),
            ("data-path", "--data-path"),
            ("dataset-id", "--dataset-id"),
            ("dataset-cache-dir", "--dataset-cache-dir"),
            ("max-dataset-entries", "--max-dataset-entries"),
            ("seq-length", "--seq-length"),
            ("batch-size", "--batch-size"),
            ("epochs", "--epochs"),
            ("lr", "--lr"),
            ("device", "--device"),
            ("gradient-accumulation", "--gradient-accumulation"),
            ("save-dir", "--save-dir"),
        ]

        for key, flag in text_fields:
            value = get_str(key)
            if value:
                push(flag, value)

        bool_fields = [
            ("offline", "--offline"),
            ("from-scratch", "--from-scratch"),
        ]
        for key, flag in bool_fields:
            if self._normalize_bool(settings.get(key, False)):
                args.append(flag)

        data_path = get_str("data-path")
        dataset_id = get_str("dataset-id")

        if not data_path and not dataset_id:
            self._append_log("[warn] No dataset provided; training will likely fail.")

        return args

    def _poll_training_output(self) -> None:
        if self._training is None:
            if self._poll_timer is not None:
                self._poll_timer.pause()
            return

        for line in self._training.drain():
            if line is None:
                self._handle_training_complete()
                return
            self._append_log(line)
            self._parse_training_line(line)

    def _handle_training_complete(self) -> None:
        self._append_log("[info] Training finished.")
        self._set_buttons(running=False)
        self._stats.status = "Idle"
        self._update_stats_display()
        if self._poll_timer is not None:
            self._poll_timer.pause()

    def _append_log(self, line: str) -> None:
        if self._log is None:
            return
        fragments = line.replace("\r", "\n").splitlines() or [""]
        for fragment in fragments:
            self._log.write_line(fragment)

    def _update_settings_summary(self) -> None:
        model = str(self._settings.get("model-id") or self._settings.get("config-path") or "--")
        data = str(self._settings.get("dataset-id") or self._settings.get("data-path") or "--")
        if self._active_args_label is not None:
            args_preview = []
            for key, value in self._settings.items():
                if key in {"offline", "from-scratch"}:
                    if self._normalize_bool(value):
                        args_preview.append(f"--{key}")
                    continue
                text_value = str(value).strip()
                if text_value:
                    args_preview.append(f"--{key}={text_value}")
            preview = ", ".join(args_preview) or "--"
            if len(preview) > 120:
                preview = preview[:117] + "..."
            self._active_args_label.update(f"Active Args: {preview}")
        if self._settings_summary:
            self._settings_summary["model"].update(f"Model: {model}")
            self._settings_summary["data"].update(f"Dataset: {data}")

    def _set_buttons(self, *, running: bool) -> None:
        start_btn = self.query_one("#start-training", Button)
        stop_btn = self.query_one("#stop-training", Button)
        start_btn.disabled = running
        stop_btn.disabled = not running

    def _parse_training_line(self, line: str) -> None:
        step_match = re.search(r"step=(\d+)", line)
        loss_match = re.search(r"loss=([0-9]*\.?[0-9]+)", line)
        token_match = re.search(r"total_tok=(\d+)", line)
        tok_rate_match = re.search(r"tok/s=([0-9]*\.?[0-9]+)", line)
        epoch_header = re.search(r"=== Epoch (\d+)/(\d+) ===", line)
        epoch_summary = re.search(r"\[epoch\]\s+(\d+)\s+mean loss = ([0-9]*\.?[0-9]+)", line)
        done_summary = re.search(r"done steps=(\d+)\s+mean_loss=([0-9]*\.?[0-9]+)", line)

        if epoch_header:
            self._stats.epoch = int(epoch_header.group(1))
            self._stats.total_epochs = int(epoch_header.group(2))
            self._stats.status = "Epoch Running"
        if epoch_summary:
            self._stats.epoch = int(epoch_summary.group(1))
            self._stats.mean_loss = float(epoch_summary.group(2))
        if step_match:
            self._stats.step = int(step_match.group(1))
        if loss_match:
            self._stats.loss = float(loss_match.group(1))
        if token_match:
            self._stats.tokens = int(token_match.group(1))
        if tok_rate_match:
            self._stats.tok_rate = float(tok_rate_match.group(1))
        if done_summary:
            self._stats.step = int(done_summary.group(1))
            self._stats.mean_loss = float(done_summary.group(2))
            self._stats.status = "Completed"

        self._update_stats_display()

    def _update_stats_display(self) -> None:
        status = self._stats.status
        epoch_text = (
            f"{self._stats.epoch}/{self._stats.total_epochs}"
            if self._stats.total_epochs
            else str(self._stats.epoch)
        )
        loss_text = f"{self._stats.loss:.4f}" if self._stats.loss is not None else "--"
        tok_rate = f"{self._stats.tok_rate:.1f}" if self._stats.tok_rate is not None else "--"
        mean_loss_text = (
            f"{self._stats.mean_loss:.4f}"
            if self._stats.mean_loss is not None
            else "--"
        )

        self._stat_labels["status"].update(f"Status: {status}")
        self._stat_labels["step"].update(f"Step: {self._stats.step}")
        self._stat_labels["epoch"].update(f"Epoch: {epoch_text}")
        self._stat_labels["loss"].update(f"Loss: {loss_text} (mean {mean_loss_text})")
        self._stat_labels["tokens"].update(f"Tokens: {self._stats.tokens}")
        self._stat_labels["tok_rate"].update(f"Tok/s: {tok_rate}")

    def _update_resource_usage(self) -> None:
        if psutil is not None:
            cpu = psutil.cpu_percent(interval=None)
            mem = psutil.virtual_memory()
            self._resource_labels["cpu"].update(f"CPU: {cpu:.1f}%")
            mem_used_gb = mem.used / (1024 ** 3)
            mem_total_gb = mem.total / (1024 ** 3)
            self._resource_labels["ram"].update(
                f"Memory: {mem.percent:.1f}% ({mem_used_gb:.1f} / {mem_total_gb:.1f} GiB)"
            )
        else:
            load_avg = os.getloadavg()
            self._resource_labels["cpu"].update(
                f"Load Avg: {load_avg[0]:.2f}, {load_avg[1]:.2f}, {load_avg[2]:.2f}"
            )
            self._resource_labels["ram"].update("Memory: --")

        # GPU metrics placeholder (extend in future when integrations are available)
        self._resource_labels["gpu"].update("GPU: N/A")


class TrainSettingsScreen(ModalScreen[Dict[str, object]]):
    """Modal for editing training settings."""

    CSS_PATH = Path(__file__).with_name("train_menu.tcss")

    BINDINGS = [
        Binding("escape", "dismiss", "Cancel"),
    ]

    def __init__(self, values: Dict[str, object]) -> None:
        super().__init__(id="train-settings")
        self._initial = dict(values)
        self._inputs: Dict[str, Input | Checkbox] = {}

    def compose(self) -> ComposeResult:
        columns = 3 if len(SETTINGS_FIELDS) >= 9 else 2
        per_column = math.ceil(len(SETTINGS_FIELDS) / columns)
        with Container(id="settings-modal-container"):
            yield Static("Training Settings", id="settings-modal-title")
            with VerticalScroll(id="settings-scroll"):
                with Container(id="settings-columns"):
                    for chunk in chunked(SETTINGS_FIELDS, per_column):
                        with Vertical(classes="settings-column"):
                            for field in chunk:
                                name = str(field["name"])
                                label = str(field["label"])
                                field_type = field.get("type", "text")
                                yield Label(label, classes="settings-field-label")
                                if field_type == "checkbox":
                                    widget = Checkbox(id=f"settings-{name}")
                                else:
                                    placeholder = str(field.get("placeholder", ""))
                                    widget = Input(id=f"settings-{name}", placeholder=placeholder)
                                widget.add_class("settings-field-input")
                                self._inputs[name] = widget
                                yield widget
            with Container(id="settings-modal-buttons"):
                yield Button("Cancel", id="settings-cancel")
                yield Button("Apply", id="settings-apply", variant="primary")

    def on_mount(self) -> None:
        for name, widget in self._inputs.items():
            value = self._initial.get(name)
            if isinstance(widget, Checkbox):
                widget.value = bool(value)
            elif value is not None:
                widget.value = str(value)
        first = self._inputs.get("model-id")
        if isinstance(first, Input):
            self.call_after_refresh(first.focus)

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "settings-cancel":
            self.dismiss(None)
        elif event.button.id == "settings-apply":
            self.dismiss(self._gather_values())

    def _gather_values(self) -> Dict[str, object]:
        result: Dict[str, object] = {}
        for name, widget in self._inputs.items():
            if isinstance(widget, Checkbox):
                result[name] = widget.value
            else:
                result[name] = widget.value.strip()
        return result
