from __future__ import annotations

import asyncio
import json
import os
import re
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional

from rich.markup import escape
from transformers import AutoTokenizer

from textual.app import ComposeResult
from textual.containers import Container, VerticalScroll
from textual.events import Mount
from textual.screen import ModalScreen, Screen
from textual.widgets import Button, Checkbox, Footer, Header, Input, Label, RichLog, Static

from tiny_cheetah.logging_utils import get_logger
from tiny_cheetah.agent.functions import AgentFunctions
from tiny_cheetah.models.llm.backend import (
    detect_quantization_mode,
    get_llm_backend,
    load_model_for_backend,
)
from tiny_cheetah.orchestration.peer_client import PeerClient
from tiny_cheetah.tui.chat_menu import GenerationConfigModal
from tiny_cheetah.tui.help_screen import HelpScreen
from tiny_cheetah.tui.helpers import (
    MemoryPressureError,
    memory_abort_reason,
    streaming_generate_with_peers,
)
from tiny_cheetah.tui.widget.model_picker_screen import ModelPickerScreen

logger = get_logger(__name__)


class AgentScreen(Screen[None]):
    """Config-driven CoT agent control screen."""

    CSS_PATH = Path(__file__).with_name("agent_screen.tcss")
    BINDINGS = [
        ("escape", "pop_screen", "Back"),
        ("b", "pop_screen", "Back"),
        ("h", "open_help", "Help"),
    ]

    def __init__(
        self,
        peer_client: PeerClient,
        default_model: str | None = None,
        offline: bool = False,
    ) -> None:
        super().__init__()
        self._peer_client = peer_client
        self._offline = offline
        self._llm_backend: str = get_llm_backend()
        self._model_id: str = default_model or ""

        self._model: Optional[Any] = None
        self._model_config: Optional[dict[str, Any]] = None
        self._tokenizer: Optional[AutoTokenizer] = None
        self._model_cache_path: Optional[Path] = None
        self._model_loaded: bool = False
        self._model_is_quantized: bool = False
        self._agent_running: bool = False
        self._function_format: str = "tools"
        self._agent_functions: list[dict[str, Any]] = []
        self._agent_messages: list[dict[str, str]] = []
        self._agent_task: Optional[asyncio.Task[str]] = None
        self._agent_runtime = AgentFunctions()
        self._endless_mode: bool = self._env_flag("TC_AGENT_ENDLESS_MODE", False)
        self._last_agent_response: str = ""

        self._model_label: Optional[Label] = None
        self._backend_label: Optional[Label] = None
        self._state_label: Optional[Label] = None
        self._functions_summary: Optional[Static] = None
        self._load_button: Optional[Button] = None
        self._start_button: Optional[Button] = None
        self._stop_button: Optional[Button] = None
        self._functions_button: Optional[Button] = None
        self._cli_button: Optional[Button] = None
        self._name_input: Optional[Input] = None
        self._instructions_input: Optional[Input] = None
        self._endless_checkbox: Optional[Checkbox] = None
        self._agent_log: Optional[RichLog] = None

        self._gen_overrides: Dict[str, float | int] = {}

    def compose(self) -> ComposeResult:
        yield Header(show_clock=True)
        with Container(id="agent-root"):
            with Container(id="agent-body"):
                with Container(id="agent-main"):
                    yield RichLog(id="agent-log", markup=True, auto_scroll=True, wrap=True, highlight=True)
                    with Static(id="agent-run-panel"):
                        with Container(id="agent-run-actions"):
                            start_button = Button("Start", id="agent-start", variant="primary")
                            self._start_button = start_button
                            yield start_button
                            stop_button = Button("Stop", id="agent-stop", variant="error")
                            stop_button.disabled = True
                            self._stop_button = stop_button
                            yield stop_button
                with VerticalScroll(id="agent-side"):
                    with Static(id="agent-model-panel"):
                        yield Label("Model", classes="panel-title")
                        model_value = Label(self._model_id or "<select>", id="agent-model-value")
                        self._model_label = model_value
                        yield model_value
                        backend_value = Label(self._llm_backend, id="agent-backend-value")
                        self._backend_label = backend_value
                        yield backend_value
                        with Container(id="agent-model-actions-top"):
                            yield Button("Select Model", id="agent-open-model-picker")
                            yield Button("Gen Config", id="agent-open-gen-config")
                        with Container(id="agent-model-actions-bottom"):
                            load_button = Button("Load Model", id="agent-load-model")
                            self._load_button = load_button
                            yield load_button
                            yield Button("Clear Model", id="agent-clear-model", variant="error")
                    with Static(id="agent-config-panel"):
                        yield Label("Agent Config", classes="panel-title")
                        yield Label("Agent Name", classes="agent-field-label")
                        name_input = Input(id="agent-name-input", placeholder="e.g. research-assistant")
                        self._name_input = name_input
                        yield name_input
                        yield Label("Agent Instructions", classes="agent-field-label")
                        instructions_input = Input(
                            id="agent-instructions-input",
                            placeholder="Describe goals, reasoning style, and constraints...",
                        )
                        self._instructions_input = instructions_input
                        yield instructions_input
                        endless_checkbox = Checkbox("Endless Mode (ignore end_run)", id="agent-endless-mode")
                        self._endless_checkbox = endless_checkbox
                        yield endless_checkbox
                        with Container(id="agent-config-actions"):
                            functions_button = Button("Functions", id="agent-open-functions")
                            self._functions_button = functions_button
                            yield functions_button
                            cli_button = Button("CLI Access", id="agent-open-cli")
                            self._cli_button = cli_button
                            yield cli_button
                    with Static(id="agent-status-panel"):
                        yield Label("Status", classes="panel-title")
                        state_value = Label("Idle", id="agent-state-value")
                        self._state_label = state_value
                        yield state_value
                        functions_summary = Static("", id="agent-functions-summary")
                        self._functions_summary = functions_summary
                        yield functions_summary
        yield Footer()

    async def on_mount(self, _: Mount) -> None:
        self._agent_log = self.query_one("#agent-log", RichLog)
        if self._name_input is not None:
            self._name_input.value = "cot-agent"
        if self._endless_checkbox is not None:
            self._endless_checkbox.value = self._endless_mode
        self._sync_functions_with_runtime()
        self._refresh_backend_label()
        self._refresh_state_label()
        self._refresh_functions_summary()
        self._log("Agent screen ready.")

    def action_pop_screen(self) -> None:
        self.app.pop_screen()

    def action_open_help(self) -> None:
        self.app.push_screen(HelpScreen("Agent Help", self._help_text()))

    async def on_button_pressed(self, event: Button.Pressed) -> None:
        button_id = event.button.id
        if button_id == "agent-open-model-picker":
            self._open_model_picker()
        elif button_id == "agent-open-gen-config":
            self._open_gen_config()
        elif button_id == "agent-load-model":
            await self._start_model_load()
        elif button_id == "agent-clear-model":
            self._clear_model()
        elif button_id == "agent-open-functions":
            self._open_functions_menu()
        elif button_id == "agent-open-cli":
            self._open_cli_menu()
        elif button_id == "agent-start":
            await self._start_agent()
        elif button_id == "agent-stop":
            self._stop_agent()

    def on_checkbox_changed(self, event: Checkbox.Changed) -> None:
        if event.checkbox.id != "agent-endless-mode":
            return
        self._endless_mode = bool(event.value)
        self._log(
            "Endless mode enabled; run will only stop on manual Stop."
            if self._endless_mode
            else "Endless mode disabled; agent will stop when end_run is called."
        )

    def _open_model_picker(self) -> None:
        self.app.push_screen(ModelPickerScreen(self._model_id or ""), self._handle_model_selected)

    @staticmethod
    def _help_text() -> str:
        return "\n".join(
            [
                "Agent Screen",
                "- Start/Stop controls run looping agent execution.",
                "- Functions opens OpenAI-style tool schema editor.",
                "- CLI Access runs one-off shell commands.",
                "- Endless Mode ignores end_run and loops until manual Stop.",
                "- h opens this help screen.",
                "- b / Esc returns to previous menu.",
            ]
        )

    def _handle_model_selected(self, result: Optional[str]) -> None:
        if not result:
            return
        selected = result.strip()
        if not selected:
            return
        if selected != self._model_id:
            self._clear_model(update_log=False)
        self._model_id = selected
        if self._model_label is not None:
            self._model_label.update(selected)
        self._log(f"Model set to '{selected}'.")

    def _open_gen_config(self) -> None:
        effective = self._effective_gen_config()
        self.app.push_screen(
            GenerationConfigModal(
                temperature=effective["temperature"],
                top_k=int(effective["top_k"]),
                top_p=effective["top_p"],
                alpha_f=effective["alpha_f"],
                alpha_p=effective["alpha_p"],
            ),
            self._apply_gen_config,
        )

    def _open_functions_menu(self) -> None:
        self._sync_functions_with_runtime()
        self._refresh_functions_summary()
        self.app.push_screen(
            AgentFunctionsModal(format_mode=self._function_format, functions=self._agent_functions),
            self._apply_functions_menu,
        )

    def _apply_functions_menu(self, result: Optional[dict[str, Any]]) -> None:
        if not result:
            return
        functions = result.get("functions")
        format_mode = str(result.get("format", self._function_format))
        if not isinstance(functions, list):
            return
        self._function_format = "functions" if format_mode == "functions" else "tools"
        self._agent_functions = list(functions)
        self._sync_functions_with_runtime()
        self._refresh_functions_summary()
        self._log(f"Functions updated: {len(self._agent_functions)} entries ({self._function_format} format).")

    def _open_cli_menu(self) -> None:
        self.app.push_screen(AgentCLIModal(), self._handle_cli_command)

    def _handle_cli_command(self, command: Optional[str]) -> None:
        if command is None:
            return
        cmd = command.strip()
        if not cmd:
            self._log("CLI command was empty.")
            return
        asyncio.create_task(self._run_cli_command(cmd))

    async def _run_cli_command(self, command: str) -> None:
        self._log(f"[CLI] $ {command}")
        try:
            proc = await asyncio.create_subprocess_shell(
                command,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            stdout, stderr = await proc.communicate()
            rc = int(proc.returncode or 0)
        except Exception as exc:
            self._log(f"[CLI] Failed: {exc}")
            return

        out_text = stdout.decode("utf-8", errors="replace").strip()
        err_text = stderr.decode("utf-8", errors="replace").strip()
        if out_text:
            self._log(f"[CLI][stdout] {out_text[:3000]}")
        if err_text:
            self._log(f"[CLI][stderr] {err_text[:3000]}")
        self._log(f"[CLI] Exit code: {rc}")

    def _apply_gen_config(self, result: Optional[dict[str, float | int]]) -> None:
        if result is None:
            return
        self._gen_overrides.update(result)
        self._log(
            "Generation config updated: "
            f"temp={self._gen_overrides.get('temperature')}, "
            f"top_k={self._gen_overrides.get('top_k')}, "
            f"top_p={self._gen_overrides.get('top_p')}, "
            f"alpha_f={self._gen_overrides.get('alpha_f')}, "
            f"alpha_p={self._gen_overrides.get('alpha_p')}"
        )

    def _effective_gen_config(self) -> dict[str, float | int]:
        config = self._model_config if isinstance(self._model_config, dict) else {}

        def _as_float(value: Any, default: float) -> float:
            try:
                return default if value is None else float(value)
            except (TypeError, ValueError):
                return default

        def _as_int(value: Any, default: int) -> int:
            try:
                return default if value is None else int(value)
            except (TypeError, ValueError):
                return default

        return {
            "temperature": _as_float(self._gen_overrides.get("temperature", config.get("temperature")), 1.0),
            "top_k": _as_int(self._gen_overrides.get("top_k", config.get("top_k")), 0),
            "top_p": _as_float(self._gen_overrides.get("top_p", config.get("top_p")), 0.8),
            "alpha_f": _as_float(self._gen_overrides.get("alpha_f", 0.0), 0.0),
            "alpha_p": _as_float(self._gen_overrides.get("alpha_p", 0.0), 0.0),
        }

    async def _start_model_load(self) -> None:
        if not self._model_id:
            self._log("Select a model first.")
            return
        if self._model is not None and self._tokenizer is not None:
            self._log("Model already loaded.")
            return
        self._set_load_button_enabled(False)
        self._llm_backend = get_llm_backend()
        self._refresh_backend_label()
        self._log(f"Loading model '{self._model_id}' with backend '{self._llm_backend}'...")
        try:
            self._model, self._model_config, self._tokenizer, self._model_cache_path, elapsed = await self._load_model()
        except Exception as exc:
            self._log(f"Model load failed: {exc}")
            logger.exception("Agent model load failed")
        else:
            self._model_is_quantized, quant_mode = detect_quantization_mode(
                self._model_config,
                backend=self._llm_backend,
            )
            mode_label = f"quantized ({quant_mode})" if self._model_is_quantized else "standard"
            self._model_loaded = True
            self._log(f"Model ready in {elapsed:.1f}s. Backend: {self._llm_backend}. Mode: {mode_label}.")
        finally:
            self._set_load_button_enabled(True)
            self._refresh_state_label()

    async def _load_model(self) -> tuple[Any, dict[str, Any], AutoTokenizer, Path, float]:
        self._llm_backend = get_llm_backend()
        start = time.time()
        model, model_config, tokenizer, model_path = await load_model_for_backend(
            model_id=self._model_id,
            shard=None,
            weight_device=None,
            offline_mode=self._offline,
            backend=self._llm_backend,
        )
        elapsed = time.time() - start
        return model, model_config, tokenizer, model_path, elapsed

    async def _start_agent(self) -> None:
        if self._agent_running:
            return
        if not self._model_loaded:
            await self._start_model_load()
            if not self._model_loaded:
                return

        name = (self._name_input.value.strip() if self._name_input is not None else "") or "cot-agent"
        instructions = self._instructions_input.value.strip() if self._instructions_input is not None else ""
        if not instructions:
            self._log("Add agent instructions before starting.")
            return

        self._agent_running = True
        self._set_agent_controls_running(True)
        self._refresh_state_label()

        gen_cfg = self._effective_gen_config()
        self._log(f"Agent '{name}' started.")
        self._log(f"Instructions: {instructions}")
        self._log(
            "Generation settings: "
            f"temp={gen_cfg['temperature']}, "
            f"top_k={gen_cfg['top_k']}, "
            f"top_p={gen_cfg['top_p']}, "
            f"alpha_f={gen_cfg['alpha_f']}, "
            f"alpha_p={gen_cfg['alpha_p']}"
        )
        self._log(
            f"Function mode: {self._function_format}; "
            f"functions configured: {len(self._agent_functions)}"
        )
        self._log(
            "Endless mode: enabled (ignores end_run)."
            if self._endless_mode
            else "Endless mode: disabled (end_run stops the loop)."
        )
        self._agent_messages = self._build_initial_messages(name=name, instructions=instructions)
        self._last_agent_response = ""
        self._agent_task = asyncio.create_task(self._agent_loop())

    def _stop_agent(self) -> None:
        if not self._agent_running:
            return
        self._agent_running = False
        if self._agent_task is not None and not self._agent_task.done():
            self._agent_task.cancel()
        self._agent_task = None
        self._set_agent_controls_running(False)
        self._refresh_state_label()
        self._log("Agent stopped.")

    def _build_initial_messages(self, *, name: str, instructions: str) -> list[dict[str, str]]:
        enabled_functions = sorted(self._enabled_function_names())
        tool_payload = json.dumps(self._agent_functions, ensure_ascii=True)
        end_behavior = (
            "Endless mode is enabled: continue running until user presses Stop; ignore end_run."
            if self._endless_mode
            else "When the task is complete, call end_run with an optional summary."
        )
        system_prompt = (
            f"You are '{name}', an autonomous reasoning agent.\n"
            "You can either think and answer directly, or request a function call.\n"
            "If you need a function call, output JSON only in one of these forms:\n"
            '1) {"function_call":{"name":"...","arguments":{...}}}\n'
            '2) {"tool_calls":[{"function":{"name":"...","arguments":{...}}}]}\n'
            '3) {"name":"...","arguments":{...}}\n'
            f"{end_behavior}\n"
            f"Enabled function names: {', '.join(enabled_functions) if enabled_functions else '(none)'}\n"
            f"Function schema payload ({self._function_format} format): {tool_payload}\n"
            "Keep function arguments strictly valid JSON."
        )
        return [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": instructions},
        ]

    async def _agent_loop(self) -> str:
        max_steps_raw = os.getenv("TC_AGENT_MAX_STEPS", "10")
        try:
            max_steps = max(1, int(max_steps_raw))
        except ValueError:
            max_steps = 10

        final_reply = ""
        try:
            step = 0
            while self._agent_running:
                reason = self._memory_abort_reason()
                if reason:
                    self._log(reason)
                    break
                step += 1
                if not self._endless_mode and step > max_steps:
                    self._log(f"Reached max steps ({max_steps}); stopping loop.")
                    break

                try:
                    reply = await asyncio.to_thread(self._generate_agent_reply, self._agent_messages)
                except MemoryPressureError as exc:
                    self._log(str(exc))
                    break
                if not reply:
                    self._log(f"[agent][step {step}] Empty response; stopping loop.")
                    break

                final_reply = reply
                self._agent_messages.append({"role": "assistant", "content": reply})
                self._log(f"[agent][step {step}] {reply}")

                function_call = self._extract_function_call(reply)
                if function_call is None:
                    self._log(f"[agent][step {step}] No function call found; continuing loop.")
                    if reply.lstrip().upper().startswith("FINAL:"):
                        nudge = (
                            "You returned FINAL text. To stop this run, call end_run "
                            "with optional {'summary': '...'} arguments."
                        )
                    else:
                        nudge = (
                            "Continue the task. Use function-call JSON when needed. "
                            "When finished, call end_run."
                        )
                    self._agent_messages.append({"role": "user", "content": nudge})
                    await asyncio.sleep(0)
                    continue

                function_name, arguments = function_call
                call_json = json.dumps(
                    {"name": function_name, "arguments": arguments},
                    ensure_ascii=True,
                    indent=2,
                )
                self._log(f"[function.call]\n{call_json}")
                if function_name not in self._enabled_function_names():
                    result = {
                        "ok": False,
                        "error": f"Function '{function_name}' is not enabled.",
                        "enabled_functions": sorted(self._enabled_function_names()),
                    }
                else:
                    result = await asyncio.to_thread(
                        self._agent_runtime.execute_agent_function,
                        function_name,
                        arguments,
                    )

                result_json = json.dumps(result, ensure_ascii=True, indent=2)
                self._log(f"[function.result]\n{result_json}")

                is_end_run = function_name == "end_run" and bool(result.get("ok")) and bool(result.get("end_run"))
                if is_end_run and not self._endless_mode:
                    self._log("end_run received; stopping loop.")
                    break
                if is_end_run and self._endless_mode:
                    self._log("end_run received but ignored because Endless Mode is enabled.")

                self._agent_messages.append(
                    {
                        "role": "user",
                        "content": (
                            f"Function result for {function_name}: {result_json}\n"
                            "Continue."
                        ),
                    }
                )
                await asyncio.sleep(0)

            if self._agent_running:
                self._log("Agent loop completed.")
            self._last_agent_response = final_reply
        except asyncio.CancelledError:
            self._log("Agent loop canceled.")
            raise
        except Exception as exc:  # pragma: no cover - defensive runtime path
            self._log(f"Agent loop failed: {exc}")
            logger.exception("Agent loop failed")
            self._last_agent_response = final_reply
        finally:
            self._agent_running = False
            self._agent_task = None
            self._set_agent_controls_running(False)
            self._refresh_state_label()
        return final_reply

    def _enabled_function_names(self) -> set[str]:
        names: set[str] = set()
        for entry in self._agent_functions:
            if not isinstance(entry, dict):
                continue
            if "function" in entry and isinstance(entry.get("function"), dict):
                name = entry["function"].get("name")
            else:
                name = entry.get("name")
            if isinstance(name, str) and name.strip():
                names.add(name.strip())
        if not names:
            names = set(self._agent_runtime.list_builtin_functions())
        return names

    def _context_window_tokens(self) -> int:
        config = self._model_config if isinstance(self._model_config, dict) else {}
        configured = config.get("max_seq_len", 2048)
        try:
            context_window = int(configured)
        except (TypeError, ValueError):
            context_window = 2048
        return max(256, context_window)

    def _response_reserve_tokens(self, context_window: int) -> int:
        reserve = os.getenv("TC_AGENT_MAX_RESP_LEN", "256")
        try:
            reserve_int = int(reserve)
        except (TypeError, ValueError):
            reserve_int = 256
        return max(32, min(reserve_int, context_window - 1))

    def _token_count_for_messages(self, messages: list[dict[str, str]]) -> int:
        if self._tokenizer is None:
            return 0
        template = self._tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=False,
        )
        enc = self._tokenizer(template, return_tensors="np")
        return int(enc["input_ids"].shape[1])

    def _prepare_agent_prompt(self, messages: list[dict[str, str]]) -> tuple[dict[str, Any], int, list[dict[str, str]]]:
        if self._tokenizer is None:
            raise RuntimeError("Tokenizer not loaded")
        context_window = self._context_window_tokens()
        reserve = self._response_reserve_tokens(context_window)
        input_budget = max(1, context_window - reserve)

        selected: list[dict[str, str]] = []
        for message in reversed(messages):
            candidate = [message, *selected]
            token_count = self._token_count_for_messages(candidate)
            if token_count <= input_budget or not selected:
                selected = candidate
            else:
                break

        template = self._tokenizer.apply_chat_template(
            selected,
            add_generation_prompt=True,
            tokenize=False,
        )
        enc = self._tokenizer(template, return_tensors="np")
        input_ids = enc["input_ids"]
        attention_mask = enc["attention_mask"]
        prompt_tokens = int(input_ids.shape[1])
        if prompt_tokens > input_budget:
            input_ids = input_ids[:, -input_budget:]
            attention_mask = attention_mask[:, -input_budget:]
            prompt_tokens = input_budget

        max_new_tokens = max(1, min(reserve, context_window - prompt_tokens))
        return {"input_ids": input_ids, "attention_mask": attention_mask}, max_new_tokens, selected

    def _generate_agent_reply(self, messages: list[dict[str, str]]) -> str:
        if self._model is None or self._tokenizer is None:
            raise RuntimeError("Model/tokenizer not loaded")

        enc, max_new_tokens, selected = self._prepare_agent_prompt(messages)
        if selected and selected != messages:
            messages[:] = selected

        gen_cfg = self._effective_gen_config()
        temp = float(gen_cfg["temperature"])
        top_k = int(gen_cfg["top_k"])
        top_p = float(gen_cfg["top_p"])
        alpha_f = float(gen_cfg["alpha_f"])
        alpha_p = float(gen_cfg["alpha_p"])

        if hasattr(self._model, "reset_kv_cache"):
            self._model.reset_kv_cache()

        if self._llm_backend == "torch":
            import torch

            device = self._torch_runtime_device()
            input_ids = torch.tensor(enc["input_ids"], dtype=torch.long, device=device)
            attention_mask = torch.tensor(enc["attention_mask"], dtype=torch.long, device=device)
        else:
            import tinygrad as tg

            input_ids = tg.Tensor(enc["input_ids"])
            attention_mask = tg.Tensor(enc["attention_mask"])

        out_tokens, _ = streaming_generate_with_peers(
            self._peer_client,
            self._model,
            input_ids,
            attention_mask,
            self._tokenizer,
            max_new_tokens=max_new_tokens,
            temp=temp,
            top_k=top_k,
            top_p=top_p,
            alpha_f=alpha_f,
            alpha_p=alpha_p,
            verbose=False,
            on_token=None,
            abort_check=self._memory_abort_reason,
        )
        if not out_tokens:
            return ""
        return self._tokenizer.decode(out_tokens, skip_special_tokens=True).strip()

    @staticmethod
    def _torch_runtime_device() -> str:
        device = str(os.getenv("TC_DEVICE", "cpu")).strip().lower()
        if device in {"metal", "mps"}:
            return "mps"
        if device.startswith("cuda"):
            return device
        return "cpu"

    def _memory_abort_reason(self) -> str | None:
        return memory_abort_reason("agent loop")

    def _extract_function_call(self, text: str) -> tuple[str, dict[str, Any] | str] | None:
        for candidate in self._json_candidates(text):
            try:
                payload = json.loads(candidate)
            except json.JSONDecodeError:
                continue
            if not isinstance(payload, dict):
                continue

            name = None
            arguments: dict[str, Any] | str = {}

            function_call = payload.get("function_call")
            if isinstance(function_call, dict):
                name = function_call.get("name")
                arguments = function_call.get("arguments", {})

            if name is None:
                tool_calls = payload.get("tool_calls")
                if isinstance(tool_calls, list) and tool_calls:
                    first = tool_calls[0]
                    if isinstance(first, dict):
                        fn = first.get("function", {})
                        if isinstance(fn, dict):
                            name = fn.get("name")
                            arguments = fn.get("arguments", {})

            if name is None and isinstance(payload.get("name"), str):
                name = payload.get("name")
                arguments = payload.get("arguments", {})

            if isinstance(name, str) and name.strip():
                return name.strip(), arguments
        return None

    @staticmethod
    def _json_candidates(text: str) -> list[str]:
        candidates: list[str] = []
        for block in re.findall(r"```(?:json)?\s*([\s\S]*?)```", text, flags=re.IGNORECASE):
            if block.strip():
                candidates.append(block.strip())

        stripped = text.strip()
        if stripped.startswith("{") and stripped.endswith("}"):
            candidates.append(stripped)

        brace_match = re.search(r"\{[\s\S]*\}", text)
        if brace_match:
            candidates.append(brace_match.group(0).strip())

        # Preserve order while removing duplicates.
        deduped: list[str] = []
        seen: set[str] = set()
        for item in candidates:
            if item in seen:
                continue
            seen.add(item)
            deduped.append(item)
        return deduped

    def _set_agent_controls_running(self, running: bool) -> None:
        if self._start_button is not None:
            self._start_button.disabled = running
        if self._stop_button is not None:
            self._stop_button.disabled = not running
        if self._functions_button is not None:
            self._functions_button.disabled = running
        if self._cli_button is not None:
            self._cli_button.disabled = running
        if self._name_input is not None:
            self._name_input.disabled = running
        if self._instructions_input is not None:
            self._instructions_input.disabled = running
        if self._endless_checkbox is not None:
            self._endless_checkbox.disabled = running

    @staticmethod
    def _env_flag(name: str, default: bool) -> bool:
        raw = os.getenv(name)
        if raw is None:
            return default
        return str(raw).strip().lower() in {"1", "true", "yes", "on"}

    def _sync_functions_with_runtime(self) -> None:
        builtin = self._agent_runtime.get_agent_functions(format_mode=self._function_format)
        if not self._agent_functions:
            self._agent_functions = builtin
            return
        self._agent_functions = self._merge_functions(self._agent_functions, builtin)

    def _merge_functions(
        self,
        preferred: list[dict[str, Any]],
        defaults: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        merged: dict[str, dict[str, Any]] = {}
        extras: list[dict[str, Any]] = []

        for entry in defaults:
            name = self._function_entry_name(entry)
            if name is None:
                extras.append(entry)
                continue
            merged[name] = entry

        for entry in preferred:
            name = self._function_entry_name(entry)
            if name is None:
                extras.append(entry)
                continue
            merged[name] = entry

        return list(merged.values()) + extras

    @staticmethod
    def _function_entry_name(entry: dict[str, Any]) -> str | None:
        if not isinstance(entry, dict):
            return None
        if isinstance(entry.get("function"), dict):
            name = entry["function"].get("name")
        else:
            name = entry.get("name")
        if isinstance(name, str):
            stripped = name.strip()
            return stripped or None
        return None

    def _refresh_functions_summary(self) -> None:
        if self._functions_summary is None:
            return
        builtin_names = self._agent_runtime.list_builtin_functions()
        enabled_names = sorted(self._enabled_function_names())
        missing_builtin = [name for name in builtin_names if name not in enabled_names]
        custom_names = [name for name in enabled_names if name not in builtin_names]

        lines: list[str] = [f"Built-ins ({len(builtin_names)}):"]
        if builtin_names:
            lines.extend([f"- {name}" for name in builtin_names])
        else:
            lines.append("- none")

        lines.append(f"Enabled ({len(enabled_names)}):")
        if enabled_names:
            lines.extend([f"- {name}" for name in enabled_names])
        else:
            lines.append("- none")

        if missing_builtin:
            lines.append("Missing Built-ins:")
            lines.extend([f"- {name}" for name in missing_builtin])
        if custom_names:
            lines.append("Custom Functions:")
            lines.extend([f"- {name}" for name in custom_names])
        self._functions_summary.update("\n".join(lines))

    def _set_load_button_enabled(self, enabled: bool) -> None:
        if self._load_button is not None:
            self._load_button.disabled = not enabled

    def _clear_model(self, *, update_log: bool = True) -> None:
        if self._agent_running:
            self._stop_agent()
        self._model = None
        self._model_config = None
        self._tokenizer = None
        self._model_cache_path = None
        self._model_loaded = False
        self._model_is_quantized = False
        if update_log:
            self._log("Model cleared.")
        self._refresh_state_label()

    def _refresh_backend_label(self) -> None:
        if self._backend_label is not None:
            self._backend_label.update(f"Backend: {self._llm_backend}")

    def _refresh_state_label(self) -> None:
        if self._state_label is None:
            return
        if self._agent_running:
            state = "Running"
        elif self._model_loaded:
            state = "Ready"
        else:
            state = "Idle"
        self._state_label.update(state)

    def _log(self, message: str) -> None:
        if self._agent_log is None:
            return
        stamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        self._agent_log.write(f"({stamp}) {escape(message)}")
        self._agent_log.scroll_end(animate=False, force=True)


class AgentCLIModal(ModalScreen[Optional[str]]):
    """Simple modal for one-off local CLI command execution."""

    def __init__(self) -> None:
        super().__init__(id="agent-cli-modal")
        self._command_input: Optional[Input] = None

    def compose(self) -> ComposeResult:
        with Container(id="agent-cli-modal-container"):
            yield Label("CLI Access", id="agent-cli-title")
            yield Label("Enter a shell command to run on this machine:", id="agent-cli-help")
            command_input = Input(id="agent-cli-command", placeholder="e.g. ls -la")
            self._command_input = command_input
            yield command_input
            with Container(id="agent-cli-buttons"):
                yield Button("Cancel", id="agent-cli-cancel")
                yield Button("Run", id="agent-cli-run", variant="primary")

    def on_mount(self, _: Mount) -> None:
        if self._command_input is not None:
            self.call_after_refresh(self._command_input.focus)

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "agent-cli-cancel":
            self.dismiss(None)
        elif event.button.id == "agent-cli-run":
            value = self._command_input.value.strip() if self._command_input is not None else ""
            self.dismiss(value or None)

    def on_input_submitted(self, event: Input.Submitted) -> None:
        if event.input.id != "agent-cli-command":
            return
        value = event.value.strip()
        self.dismiss(value or None)


class AgentFunctionsModal(ModalScreen[Optional[dict[str, Any]]]):
    """Modal for defining OpenAI-style function specs."""

    def __init__(self, *, format_mode: str, functions: list[dict[str, Any]]) -> None:
        super().__init__(id="agent-functions-modal")
        self._format_mode = "functions" if format_mode == "functions" else "tools"
        self._functions = list(functions)

        self._format_value: Optional[Label] = None
        self._name_input: Optional[Input] = None
        self._description_input: Optional[Input] = None
        self._params_input: Optional[Input] = None
        self._status: Optional[Label] = None
        self._preview: Optional[Static] = None
        self._tools_button: Optional[Button] = None
        self._functions_button: Optional[Button] = None

    def compose(self) -> ComposeResult:
        with Container(id="agent-functions-modal-container"):
            yield Label("Agent Functions", id="agent-functions-title")
            yield Label("Format", classes="agent-functions-label")
            format_value = Label("", id="agent-functions-format")
            self._format_value = format_value
            yield format_value
            with Container(id="agent-functions-mode-buttons"):
                tools_button = Button("OpenAI Tools", id="agent-functions-mode-tools")
                self._tools_button = tools_button
                yield tools_button
                functions_button = Button("Legacy Functions", id="agent-functions-mode-functions")
                self._functions_button = functions_button
                yield functions_button

            yield Label("Function Name", classes="agent-functions-label")
            name_input = Input(id="agent-functions-name", placeholder="get_weather")
            self._name_input = name_input
            yield name_input

            yield Label("Description", classes="agent-functions-label")
            description_input = Input(id="agent-functions-description", placeholder="Describe what the function does")
            self._description_input = description_input
            yield description_input

            yield Label("Parameters JSON Schema", classes="agent-functions-label")
            params_input = Input(
                id="agent-functions-params",
                placeholder='{"type":"object","properties":{...},"required":[...]}',
            )
            self._params_input = params_input
            yield params_input

            status = Label("", id="agent-functions-status")
            self._status = status
            yield status

            preview = Static("", id="agent-functions-preview")
            self._preview = preview
            yield preview

            with Container(id="agent-functions-buttons"):
                yield Button("Add", id="agent-functions-add", variant="primary")
                yield Button("Remove Last", id="agent-functions-remove")
                yield Button("Cancel", id="agent-functions-cancel")
                yield Button("Save", id="agent-functions-save", variant="success")

    def on_mount(self, _: Mount) -> None:
        if self._params_input is not None:
            self._params_input.value = '{"type":"object","properties":{},"required":[]}'
        self._refresh_format_ui()
        self._refresh_preview()

    def on_button_pressed(self, event: Button.Pressed) -> None:
        button_id = event.button.id
        if button_id == "agent-functions-mode-tools":
            self._format_mode = "tools"
            self._refresh_format_ui()
            return
        if button_id == "agent-functions-mode-functions":
            self._format_mode = "functions"
            self._refresh_format_ui()
            return
        if button_id == "agent-functions-add":
            self._add_function()
            return
        if button_id == "agent-functions-remove":
            if self._functions:
                self._functions.pop()
                self._set_status("Removed last function.")
                self._refresh_preview()
            else:
                self._set_status("No functions to remove.")
            return
        if button_id == "agent-functions-cancel":
            self.dismiss(None)
            return
        if button_id == "agent-functions-save":
            self.dismiss({"format": self._format_mode, "functions": self._functions})

    def _add_function(self) -> None:
        name = self._name_input.value.strip() if self._name_input is not None else ""
        description = self._description_input.value.strip() if self._description_input is not None else ""
        raw_params = self._params_input.value.strip() if self._params_input is not None else ""

        if not name:
            self._set_status("Function name is required.")
            return
        if not description:
            self._set_status("Function description is required.")
            return
        if not raw_params:
            self._set_status("Parameters JSON is required.")
            return

        try:
            parameters = json.loads(raw_params)
        except json.JSONDecodeError as exc:
            self._set_status(f"Invalid JSON: {exc}")
            return
        if not isinstance(parameters, dict):
            self._set_status("Parameters must decode to a JSON object.")
            return

        if self._format_mode == "tools":
            entry: dict[str, Any] = {
                "type": "function",
                "function": {
                    "name": name,
                    "description": description,
                    "parameters": parameters,
                },
            }
        else:
            entry = {
                "name": name,
                "description": description,
                "parameters": parameters,
            }
        self._functions.append(entry)
        self._set_status(f"Added function '{name}'.")
        self._refresh_preview()

    def _refresh_format_ui(self) -> None:
        if self._format_value is not None:
            self._format_value.update(
                "OpenAI tools format" if self._format_mode == "tools" else "OpenAI legacy functions format"
            )
        if self._tools_button is not None:
            self._tools_button.variant = "primary" if self._format_mode == "tools" else "default"
        if self._functions_button is not None:
            self._functions_button.variant = "primary" if self._format_mode == "functions" else "default"

    def _refresh_preview(self) -> None:
        if self._preview is None:
            return
        if not self._functions:
            self._preview.update("No functions configured.")
            return
        rendered = json.dumps(self._functions, indent=2)
        self._preview.update(rendered)

    def _set_status(self, message: str) -> None:
        if self._status is not None:
            self._status.update(message)
