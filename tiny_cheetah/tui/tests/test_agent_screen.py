import unittest

from textual.app import App
from textual.widgets import Static

from tiny_cheetah.tui.agent_screen import AgentScreen


class _AgentScreenHost(App[None]):
    def compose(self):
        yield Static("host")


class TestAgentScreen(unittest.IsolatedAsyncioTestCase):
    async def test_agent_screen_mounts_without_query_error(self) -> None:
        app = _AgentScreenHost()

        async with app.run_test() as pilot:
            app.push_screen(AgentScreen(peer_client=object()))
            await pilot.pause()

            screen = app.screen
            self.assertIsInstance(screen, AgentScreen)
            self.assertIsNotNone(screen._agent_log)
            self.assertIsNotNone(screen._config_button)
            self.assertIsNotNone(screen._functions_summary)


class TestAgentScreenPrompt(unittest.TestCase):
    def test_build_initial_messages_uses_structured_agent_loop_prompt(self) -> None:
        screen = AgentScreen(peer_client=object())

        messages = screen._build_initial_messages(
            name="cot-agent",
            instructions="Create a file and stop when done.",
        )

        self.assertEqual(messages[0]["role"], "system")
        system_prompt = messages[0]["content"]
        self.assertIn("Respond in this JSON format", system_prompt)
        self.assertIn('"thoughts"', system_prompt)
        self.assertIn('"ability"', system_prompt)
        self.assertIn('"step completed"', system_prompt)
        self.assertIn("set `ability.name` to `end_run`", system_prompt)
        self.assertIn("Ability signatures:", system_prompt)
        self.assertNotIn("Function schema payload", system_prompt)

    def test_extract_function_call_reads_ability_format(self) -> None:
        screen = AgentScreen(peer_client=object())

        result = screen._extract_function_call(
            """
            {
              "thoughts": {
                "text": "Need to create the file",
                "reasoning": "The task asks for a file write",
                "criticism": "None",
                "step completed": "Selected the correct action",
                "plan": "- write file\\n- end run",
                "speak": "Writing the file now"
              },
              "ability": {
                "name": "write_file",
                "args": {
                  "path": "notes.txt",
                  "content": "hello"
                }
              }
            }
            """
        )

        self.assertEqual(
            result,
            ("write_file", {"path": "notes.txt", "content": "hello"}),
        )

    def test_compact_agent_reply_for_memory_drops_verbose_thoughts(self) -> None:
        screen = AgentScreen(peer_client=object())

        compact = screen._compact_agent_reply_for_memory(
            {
                "thoughts": {
                    "text": "Long hidden thought",
                    "reasoning": "Long reasoning",
                    "criticism": "Long criticism",
                    "step completed": "Prepared the write action",
                    "plan": "- one\n- two",
                    "speak": "Writing the file now",
                },
                "ability": {
                    "name": "write_file",
                    "args": {"path": "notes.txt", "content": "hello"},
                },
            },
            function_name="write_file",
            arguments={"path": "notes.txt", "content": "hello"},
        )

        self.assertIsNotNone(compact)
        assert compact is not None
        self.assertEqual(compact["role"], "assistant")
        self.assertIn("ability=write_file", compact["content"])
        self.assertIn("speak=Writing the file now", compact["content"])
        self.assertNotIn("reasoning", compact["content"])


if __name__ == "__main__":
    unittest.main()
