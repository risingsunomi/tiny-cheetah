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


if __name__ == "__main__":
    unittest.main()
