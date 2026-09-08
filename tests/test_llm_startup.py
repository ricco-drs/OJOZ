from __future__ import annotations

import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

from app.core.controller import Controller
from app.core.llm_agent import LLMAgent


class ControllerStartupTests(unittest.TestCase):
    def setUp(self):
        self.thread_patch = patch("app.core.controller.threading.Thread")
        self.thread = self.thread_patch.start()
        self.addCleanup(self.thread_patch.stop)
        events = patch("app.core.controller.event_bus.publish")
        events.start()
        self.addCleanup(events.stop)
        with patch.object(Controller, "_register_events"):
            self.controller = Controller(tts=Mock(), stt=Mock())
        self.controller.tts.is_speaking.return_value = False
        self.controller.stt.is_capturing.return_value = False
        self.controller._llm = Mock()
        self.controller._llm.start_conversation.return_value = "Mira hacia la camara."
        self.controller.speak = Mock()
        self.controller._enable_listening_after_delay = Mock()
        self.thread.reset_mock()
        self.addCleanup(self.controller.stop)

    def start(self):
        with patch.dict("os.environ", {"OJOZ_DEV_LOGIN": ""}):
            self.controller.start()

    def complete_intro(self):
        self.controller._on_tts_end()
        return self.thread.call_args.kwargs["target"]

    def test_llm_starts_after_intro_without_user_input_and_only_once(self):
        self.start()
        self.controller._llm.start_conversation.assert_not_called()
        worker = self.complete_intro()
        self.controller._on_tts_end()
        self.thread.assert_called_once()
        worker()
        self.controller._llm.start_conversation.assert_called_once_with(self.controller._introduction)
        self.controller.speak.assert_called_with("Mira hacia la camara.")
        self.controller._enable_listening_after_delay.reset_mock()
        self.controller._on_tts_end()
        self.thread.assert_called_once()
        self.controller._enable_listening_after_delay.assert_called_once()

    def test_tts_events_and_failsafe_do_not_interrupt_model_actions(self):
        self.start()
        worker = self.complete_intro()
        self.controller.stt.reset_mock()
        self.controller._on_tts_end()
        self.controller._force_enable_listening()
        self.controller.stt.enable_listening.assert_not_called()
        self.controller._enable_listening_after_delay.assert_called()
        worker()
        self.assertFalse(self.controller._llm_turn_active)

    def test_unavailable_model_rearms_listening(self):
        self.controller._llm.is_available.return_value = False
        self.start()
        worker = self.complete_intro()
        self.controller._enable_listening_after_delay.reset_mock()
        worker()
        self.controller._llm.start_conversation.assert_not_called()
        self.controller._enable_listening_after_delay.assert_called_once()

    def test_failed_initial_turn_rearms_listening(self):
        self.controller._llm.start_conversation.return_value = None
        self.start()
        worker = self.complete_intro()
        self.controller._enable_listening_after_delay.reset_mock()
        worker()
        self.assertFalse(self.controller._llm_turn_active)
        self.controller._enable_listening_after_delay.assert_called_once()

    def test_failsafe_also_starts_llm_when_end_event_is_missing(self):
        self.start()
        self.controller._force_enable_listening()
        self.thread.assert_called_once()
        self.thread.call_args.kwargs["target"]()
        self.controller._llm.start_conversation.assert_called_once()

    def test_shutdown_suppresses_late_model_response(self):
        self.start()
        worker = self.complete_intro()
        self.controller.stop()
        self.controller.speak.reset_mock()
        self.controller._enable_listening_after_delay.reset_mock()
        worker()
        self.controller._llm.start_conversation.assert_not_called()
        self.controller.speak.assert_not_called()
        self.controller._enable_listening_after_delay.assert_not_called()

    def test_initial_response_waits_for_llm_without_being_lost(self):
        self.start()
        worker = self.complete_intro()
        self.controller._process_stt_text = Mock()
        self.controller._on_stt_text("Quiero leer un documento")
        self.controller._process_stt_text.assert_not_called()
        self.assertEqual(list(self.controller._pending_user_texts), ["Quiero leer un documento"])
        worker()
        self.assertTrue(self.controller._dispatch_user_turn())
        self.controller._process_stt_text.assert_called_once_with("Quiero leer un documento")
        self.assertFalse(self.controller._pending_user_texts)

    def test_response_transcribed_during_tts_is_not_discarded(self):
        self.controller.tts.is_speaking.return_value = True
        self.controller._process_stt_text = Mock()
        self.controller._on_stt_text("Es mi primera vez")
        self.controller._process_stt_text.assert_not_called()
        self.controller.tts.is_speaking.return_value = False
        self.controller._dispatch_user_turn()
        self.controller._process_stt_text.assert_called_once_with("Es mi primera vez")

    def test_intro_processing_can_listen_without_running_second_llm_turn(self):
        self.start()
        self.complete_intro()
        self.controller._on_stt_text("hola")
        self.controller.stt.enable_listening.assert_called_with(True)
        self.controller._llm.handle.assert_not_called()
        with patch("app.core.controller.time.sleep"):
            Controller._enable_listening_after_delay(self.controller)
            self.thread.call_args.kwargs["target"]()
        self.controller.stt.enable_listening.assert_called_with(True)
        self.assertEqual(list(self.controller._pending_user_texts), ["hola"])

    def test_speech_waits_until_user_finishes_capturing(self):
        self.controller.stt.is_capturing.side_effect = [True, True, False]
        with patch("app.core.controller.time.sleep") as sleep, \
                patch.object(self.controller, "_schedule_fallback_rearm"):
            Controller.speak(self.controller, "Te escucho")
        self.assertEqual(sleep.call_count, 2)
        self.controller.tts.say.assert_called_once_with("Te escucho")


class AgentStartupTests(unittest.TestCase):
    def test_startup_passes_intro_as_internal_context_and_remembers_response(self):
        agent = LLMAgent(actions={}, speak=Mock(), estado=lambda: "Sesion verificada")
        response = SimpleNamespace(
            stop_reason="end_turn", content=[SimpleNamespace(type="text", text="Que necesitas?")],
        )
        client = Mock()
        client.messages.create.return_value = response
        with patch.object(agent, "_get_client", return_value=client):
            self.assertEqual(agent.start_conversation("Soy OJOZ"), "Que necesitas?")
        self.assertIn("Evento interno", agent._history[0]["content"])
        self.assertIn("Soy OJOZ", agent._history[0]["content"])
        self.assertEqual(agent._history[-1], {"role": "assistant", "content": response.content})
        agent._speak.assert_not_called()

    def test_explicitly_disabled_model_does_not_create_client(self):
        agent = LLMAgent(actions={}, speak=Mock(), estado=lambda: "")
        with patch("app.core.llm_agent.llm_config.enabled", False), \
                patch.object(agent, "_get_client") as get_client:
            self.assertFalse(agent.is_available())
        get_client.assert_not_called()


if __name__ == "__main__":
    unittest.main()
