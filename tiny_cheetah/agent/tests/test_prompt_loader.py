from __future__ import annotations

import unittest

from tiny_cheetah.agent.prompt_loader import render_agent_system_prompt


class TestPromptLoader(unittest.TestCase):
    def test_render_agent_system_prompt_uses_template(self) -> None:
        rendered = render_agent_system_prompt(
            name="cot-agent",
            endless_mode=False,
            enabled_functions=["write_file", "end_run"],
            function_format="tools",
            tool_summary="- write_file(path*, content*): Write text\n- end_run(summary?): Stop the run",
            agent_prompt="Create a file and stop.",
        )

        self.assertIn("You are 'cot-agent'", rendered)
        self.assertIn("Respond in this JSON format", rendered)
        self.assertIn('"thoughts"', rendered)
        self.assertIn('"ability"', rendered)
        self.assertIn("write_file, end_run", rendered)
        self.assertIn("Ability signatures:", rendered)
        self.assertNotIn("Function schema payload", rendered)


if __name__ == "__main__":
    unittest.main()
