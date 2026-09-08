from __future__ import annotations

import asyncio
import unittest
from unittest.mock import AsyncMock, Mock, patch

import flet as ft

from app.core.event_bus import EventBus
from app.ui.flet_app import OJOZApp
from app.ui.console_view import _on_stt_text


class ChatTranscriptionTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.bus = EventBus()
        with patch("app.core.event_bus.event_bus", self.bus):
            self.app = OJOZApp()
        self.app.chat_messages = ft.ListView()
        self.app.chat_messages.scroll_to = AsyncMock()
        self.tasks = []
        self.app.page = Mock()

        def run_task(handler):
            task = asyncio.create_task(handler())
            self.tasks.append(task)
            return task

        self.app.page.run_task.side_effect = run_task

    async def drain(self):
        while self.tasks:
            tasks, self.tasks = self.tasks, []
            await asyncio.gather(*tasks)

    async def test_transcription_creates_one_visible_user_bubble_before_response(self):
        self.bus.subscribe("stt:text", lambda **kw: self.bus.publish(
            "ui:print", role="app/tts", text="Te escucho."
        ))
        self.bus.publish("stt:text", text="Hola OJOZ, ya tengo una cuenta", confidence=None)
        await self.drain()
        rows = self.app.chat_messages.controls
        self.assertEqual(len(rows), 2)
        self.assertEqual(rows[0].alignment, ft.MainAxisAlignment.END)
        user = rows[0].controls[0]
        self.assertEqual(user.opacity, 1)
        self.assertEqual(user.bgcolor, "#0F1822BB")
        self.assertEqual(user.content.controls[1].value, "Hola OJOZ, ya tengo una cuenta")
        self.assertEqual(rows[1].alignment, ft.MainAxisAlignment.START)
        self.assertEqual(rows[1].controls[0].content.controls[1].value, "Te escucho.")

    async def test_repeated_phrases_are_separate_bubbles_and_empty_text_is_ignored(self):
        for text in ("si", "si", "   "):
            self.bus.publish("stt:text", text=text)
        await self.drain()
        self.assertEqual(len(self.app.chat_messages.controls), 2)

    async def test_scroll_failure_does_not_hide_long_transcription(self):
        self.app.chat_messages.scroll_to.side_effect = RuntimeError("scroll unavailable")
        message = "Quiero que leas este documento. " * 100
        self.bus.publish("stt:text", text=message)
        await self.drain()
        bubble = self.app.chat_messages.controls[0].controls[0]
        body = bubble.content.controls[1]
        self.assertEqual(bubble.opacity, 1)
        self.assertEqual(bubble.col, {"xs": 12, "md": 6})
        self.assertEqual(body.value, message.strip())
        self.assertIs(body.no_wrap, False)
        self.assertIsNone(body.max_lines)

    async def test_technical_message_stays_out_of_chat(self):
        self.bus.publish("ui:print", role="sys", text="Verificando si ya tienes una cuenta...")
        await self.drain()
        self.assertEqual(len(self.app.chat_messages.controls), 0)

    def test_console_keeps_showing_transcriptions(self):
        with patch("app.ui.console_view._on_ui_print") as display:
            _on_stt_text("hola", confidence=0.8)
        display.assert_called_once_with(role="user", text="hola")


if __name__ == "__main__":
    unittest.main()
