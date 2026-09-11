from __future__ import annotations

import unittest
from datetime import datetime
from types import SimpleNamespace
from unittest.mock import Mock, patch

from app.config.settings import llm as llm_config
from app.core.controller import Controller
from app.core.llm_agent import LLMAgent, fecha_y_hora_actual


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

    def test_the_name_is_spoken_as_ojos_but_written_as_ojoz(self):
        # La voz debe decir "ojos"; el chat debe seguir mostrando la marca.
        with patch("app.core.controller.event_bus.publish") as publish, \
                patch.object(self.controller, "_schedule_fallback_rearm"):
            Controller.speak(self.controller, "Hola, soy OJOZ, tu asistente.")

        self.controller.tts.say.assert_called_once_with("Hola, soy ojos, tu asistente.")
        escrito = [c for c in publish.call_args_list if c.kwargs.get("role") == "app/tts"]
        self.assertEqual(escrito[0].kwargs["text"], "Hola, soy OJOZ, tu asistente.")

    def test_the_name_is_replaced_in_any_casing_without_touching_other_words(self):
        normalize = Controller._normalize_tts_text
        self.assertEqual(normalize(self.controller, "Ojoz y ojoz"), "ojos y ojos")
        # No debe morder palabras que solo lo contienen.
        self.assertEqual(normalize(self.controller, "OJOZILLA"), "OJOZILLA")


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

    def test_current_time_reaches_the_model_on_every_turn(self):
        # Preguntar la hora es de lo mas comun, y el reloj del equipo ya la
        # sabe: va en el contexto para responder sin una llamada extra.
        agent = LLMAgent(actions={}, speak=Mock(), estado=lambda: "Sesion verificada")
        response = SimpleNamespace(
            stop_reason="end_turn", content=[SimpleNamespace(type="text", text="Son las tres")],
        )
        client = Mock()
        client.messages.create.return_value = response
        with patch.object(agent, "_get_client", return_value=client), \
                patch("app.core.llm_agent.fecha_y_hora_actual", return_value="martes 10 de junio de 2025, 15:04"):
            agent.handle("¿Qué hora es?")

        system = client.messages.create.call_args.kwargs["system"]
        self.assertIn("martes 10 de junio de 2025, 15:04", system)
        self.assertIn("Sesion verificada", system)


class EntregarDocumentoTests(unittest.TestCase):
    """
    Tras enunciar un documento hay que poder repetirlo.

    OJOZ mismo ofrece "¿necesitas que te lea alguna parte completa?" despues
    del resumen; si el texto se descartara al primer uso, esa oferta obligaria
    a volver a escanear el mismo papel.
    """

    def setUp(self):
        events = patch("app.core.controller.event_bus.publish")
        events.start()
        self.addCleanup(events.stop)
        with patch.object(Controller, "_register_events"):
            self.controller = Controller(tts=Mock(), stt=Mock())
        self.addCleanup(self.controller.stop)
        self.controller.speak = Mock()
        self.controller._authenticated = True
        self.controller._user_name = "Ricco"

        acciones = self.controller._build_llm_actions()
        self.entregar = acciones["entregar_documento"]
        self.cerrar_sesion = acciones["cerrar_sesion"]
        self.controller._pending_ocr_text = "Factura de luz, vence el 20 de junio."

    def test_the_document_can_be_delivered_more_than_once(self):
        with patch("app.core.controller.summarize_document_text", return_value="Es una factura."):
            self.assertNotIn("No hay ningun documento", self.entregar("resumen"))
        # Despues del resumen pide el contenido completo, y luego repetirlo.
        self.assertNotIn("No hay ningun documento", self.entregar("completo"))
        self.assertNotIn("No hay ningun documento", self.entregar("completo"))

        hablado = " ".join(str(c.args[0]) for c in self.controller.speak.call_args_list)
        self.assertIn("Factura de luz", hablado)

    def test_reading_another_document_replaces_the_previous_one(self):
        self.controller._pending_ocr_text = "Receta medica."
        self.entregar("completo")
        self.assertIn("Receta medica", str(self.controller.speak.call_args.args[0]))

    def test_closing_the_session_forgets_the_document(self):
        # Quien entre despues no tiene por que poder pedir que se lo lean.
        self.cerrar_sesion()
        self.assertIsNone(self.controller._pending_ocr_text)
        self.assertIn("No hay ningun documento", self.entregar("completo"))


class TrimHistoryTests(unittest.TestCase):
    """El recorte del historial es lo unico que limita la memoria de OJOZ."""

    def turno_hablado(self, texto: str) -> list[dict]:
        return [
            {"role": "user", "content": texto},
            {"role": "assistant", "content": [{"type": "text"}]},
        ]

    def turno_con_herramientas(self, texto: str, llamadas: int = 5) -> list[dict]:
        # La persona pide algo, el modelo encadena herramientas y responde.
        return (
            [{"role": "user", "content": texto}]
            + [
                {"role": "assistant", "content": [{"type": "tool_use"}]},
                {"role": "user", "content": [{"type": "tool_result"}]},
            ]
            * llamadas
            + [{"role": "assistant", "content": [{"type": "text"}]}]
        )

    def test_short_conversation_is_kept_whole(self):
        historial = self.turno_hablado("hola") + self.turno_hablado("¿qué hora es?")
        self.assertEqual(LLMAgent._trim(historial), historial)

    def test_a_run_of_tool_calls_does_not_erase_the_memory(self):
        # Antes esto devolvia [] y OJOZ olvidaba la conversacion entera: dentro
        # del limite no habia ningun mensaje hablado, solo tool_result.
        historial = [{"role": "user", "content": "lee el documento"}] + [
            {"role": "assistant", "content": [{"type": "tool_use"}]},
            {"role": "user", "content": [{"type": "tool_result"}]},
        ] * 60

        recortado = LLMAgent._trim(historial)

        self.assertTrue(recortado, "se perdio toda la memoria de la conversacion")
        self.assertEqual(recortado[0], {"role": "user", "content": "lee el documento"})

    def test_trimmed_history_always_starts_on_a_spoken_turn(self):
        # Un historial que empiece con un tool_result suelto es invalido para
        # la API: el tool_use que lo justifica quedaria fuera.
        historial = []
        for i in range(30):
            historial += self.turno_con_herramientas(f"pedido {i}")

        recortado = LLMAgent._trim(historial)

        self.assertEqual(recortado[0]["role"], "user")
        self.assertIsInstance(recortado[0]["content"], str)

    def test_recent_exchanges_survive_the_trim(self):
        historial = []
        for i in range(60):
            historial += self.turno_hablado(f"mensaje {i}")

        recortado = LLMAgent._trim(historial)

        self.assertLessEqual(len(recortado), llm_config.max_history_messages)
        self.assertEqual(recortado[-2], {"role": "user", "content": "mensaje 59"})


class FechaYHoraTests(unittest.TestCase):
    def test_names_are_spanish_regardless_of_system_locale(self):
        # strftime devolveria "Tuesday" o "martes" segun el idioma de Windows.
        self.assertEqual(
            fecha_y_hora_actual(datetime(2025, 6, 10, 15, 4)),
            "martes 10 de junio de 2025, 15:04",
        )

    def test_midnight_and_new_year_are_formatted_as_spoken(self):
        self.assertEqual(
            fecha_y_hora_actual(datetime(2026, 1, 1, 0, 0)),
            "jueves 1 de enero de 2026, 00:00",
        )


if __name__ == "__main__":
    unittest.main()
