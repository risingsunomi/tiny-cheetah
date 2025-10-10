# tiny-cheetah distributed AI
from __future__ import annotations

import sys
from typing import Dict

from tiny_cheetah.tui import main_menu


class TinyCheetahApp(main_menu.MainMenu):
    """Main menu app with optional training configuration injection."""

    def __init__(self, training_defaults: Dict[str, object] | None = None) -> None:
        super().__init__(training_defaults=training_defaults)


def parse_training_defaults(argv: list[str]) -> Dict[str, object]:
    from tiny_cheetah.tui.train_menu import SETTINGS_FIELDS

    valid_keys = {str(field["name"]) for field in SETTINGS_FIELDS}
    defaults: Dict[str, object] = {}
    index = 0
    while index < len(argv):
        token = argv[index]
        if not token.startswith("--"):
            index += 1
            continue
        key_value = token[2:]
        if not key_value:
            index += 1
            continue
        if "=" in key_value:
            key, value = key_value.split("=", 1)
            if key in valid_keys:
                defaults[key] = value
        else:
            if index + 1 < len(argv) and not argv[index + 1].startswith("--"):
                if key_value in valid_keys:
                    defaults[key_value] = argv[index + 1]
                index += 1
            else:
                if key_value in valid_keys:
                    defaults[key_value] = True
        index += 1
    return defaults

def main():
    defaults = parse_training_defaults(sys.argv[1:])
    sys.argv = sys.argv[:1]
    app = TinyCheetahApp(training_defaults=defaults)
    app.run()

if __name__ == "__main__":
    main()
