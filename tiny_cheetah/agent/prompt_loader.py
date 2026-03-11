from __future__ import annotations

from functools import lru_cache
from typing import Any

from jinja2 import Environment, PackageLoader


PROMPT_TEMPLATE_NAME = "cot_agent_system_prompt.j2"


@lru_cache(maxsize=1)
def _prompt_environment() -> Environment:
    return Environment(
        loader=PackageLoader("tiny_cheetah.agent", "prompts"),
        autoescape=False,
        trim_blocks=True,
        lstrip_blocks=True,
    )

def render_agent_system_prompt(**context: Any) -> str:
    template = _prompt_environment().get_template(PROMPT_TEMPLATE_NAME)
    return template.render(**context).strip()
