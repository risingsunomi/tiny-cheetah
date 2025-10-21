# tiny-cheetah distributed AI
from __future__ import annotations

import argparse
import atexit
import os
import sys
from typing import Dict, Optional

from tiny_cheetah.tui import main_menu
from tiny_cheetah.tui.train_menu import SETTINGS_FIELDS
from dotenv import load_dotenv

try:
    from tinygrad.device import Device
except Exception:  # pragma: no cover - tinygrad missing or import failure
    Device = None  # type: ignore[assignment]
else:
    def _cleanup_tinygrad_devices() -> None:
        if Device is None:
            return
        opened = getattr(Device, "_opened_devices", set())
        for dev_name in list(opened):
            try:
                device = Device[dev_name]
            except Exception:
                continue
            try:
                device.synchronize()
            except Exception:
                pass
            try:
                device.finalize()
            except Exception:
                pass

    atexit.register(_cleanup_tinygrad_devices)


class TinyCheetahApp(main_menu.MainMenu):
    """Main menu app with optional training and chat defaults."""

    def __init__(
        self,
        training_defaults: Dict[str, object] | None = None,
        chat_default: Optional[str] = None
    ) -> None:
        super().__init__(
            training_defaults=training_defaults,
            chat_default=chat_default
        )


def parse_cli_args(argv: list[str]) -> tuple[Dict[str, object], Optional[str]]:
    parser = argparse.ArgumentParser(
        description="Tiny Cheetah TUI launcher",
        add_help=True
    )
    parser.add_argument(
        "--chat__model",
        dest="chat_model",
        type=str,
        default=None,
        help="Default chat model identifier passed to the chat screen."
    )

    # Dynamically add training arguments
    def _dest_name(raw: str) -> str:
        return f"training__{raw.replace('-', '_')}"

    for field in SETTINGS_FIELDS:
        raw_name = str(field["name"])
        arg_name = f"--training__{raw_name}"
        dest_name = _dest_name(raw_name)
        help_text = field.get("placeholder") or f"Set training field '{raw_name}'"
        if field.get("type") == "checkbox":
            parser.add_argument(
                arg_name,
                dest=dest_name,
                nargs="?",
                const="true",
                default=None,
                metavar="VALUE",
                help=help_text
            )
        else:
            parser.add_argument(
                arg_name,
                dest=dest_name,
                type=str,
                default=None,
                help=help_text
            )

    parsed = parser.parse_args(argv)

    training_defaults: Dict[str, object] = {}

    for field in SETTINGS_FIELDS:
        raw_name = str(field["name"])
        dest_name = _dest_name(raw_name)
        value = getattr(parsed, dest_name, None)
        if value is not None:
            training_defaults[raw_name] = value

    return training_defaults, parsed.chat_model


def main():
    load_dotenv()

    training_defaults, chat_default = parse_cli_args(sys.argv[1:])

    def env_key(raw: str) -> str:
        return f"TRAINING__{raw.replace('-', '_').upper()}"

    env_training_defaults: Dict[str, object] = {}
    for field in SETTINGS_FIELDS:
        raw_name = str(field["name"])
        env_value = os.getenv(env_key(raw_name))
        print(f"env check %s=%s", env_key(raw_name), env_value)
        if env_value is not None and raw_name not in training_defaults:
            env_training_defaults[raw_name] = env_value

    env_chat_default = os.getenv("CHAT__MODEL") or os.getenv("chat__model")
    if chat_default is None:
        chat_default = env_chat_default

    merged_training_defaults: Dict[str, object] = {**env_training_defaults, **training_defaults}

    sys.argv = sys.argv[:1]
    app = TinyCheetahApp(
        training_defaults=merged_training_defaults,
        chat_default=chat_default
    )
    app.run()


if __name__ == "__main__":
    main()
