from __future__ import annotations
import os
import re
import shutil
import threading
import time
import uuid
from collections import deque
from pathlib import Path
from typing import Literal

from app.core.event_bus import event_bus
from app.audio.tts import TTS
from app.audio.stt import STT
from app.config.settings import config  # (tu import original, lo mantengo)
# ===== NUEVOS IMPORTS =====
from app.config.settings import vision  # rutas/umbrales de visión
from app.vision.face_capture import capture_faces
from app.vision.face_train import train_dataset
from app.vision.face_recognition import recognize_best_frame
from app.utils.fs import write_display_name
# ==== IMPORTS DE BASE DE DATOS ====
from app.db import dao
# ==========================
# === OCR (Opción 1) ===
from app.vision.ocr import read_text_best_frame
# === CURRENCY (Opción 2) ===
from app.vision.currency import detect_currency_best_frame
# === EXPIRY (Opción 3) ===
from app.vision.expiry import check_expiry_best_frame
# === ESCENA (descripción independiente) ===
from app.vision.scene import describe_scene_best_frame
from app.vision.camera_service import camera_service

from app.core.router import infer_intent
from app.core.llm_agent import LLMAgent

State = Literal["IDLE", "TTS_SPEAKING", "LISTENING", "PROCESSING"]

# Nombre hablado de cada divisa que puede detectar vision/currency.py
_CURR_TO_UNIT = {"PEN": "soles", "USD": "dólares"}

class Controller:
    """
    Orquestador de turnos TTS/STT con máquina de estados.
    - Usa eventos "tts:start"/"tts:end" para gatear el micrófono.
    - Incluye un WATCHDOG que reactiva STT si "tts:end" no llega.
    """

    def __init__(self, tts: TTS, stt: STT):
        self.tts = tts
        self.stt = stt
        self.state: State = "IDLE"
        self._lock = threading.RLock()
        self._fallback_timer: threading.Timer | None = None
        
        # Estado del flujo de conversación
        self._conversation_state: str | None = None  # "waiting_new_user_name", "waiting_returning_user_name", etc.
        self._user_name: str | None = None
        self._authenticated: bool = False  # Flag para saber si el usuario está autenticado

        # ===== NUEVOS FLAGS =====
        self._pending_enroll_after_tts: bool = False   # dispara captura tras tts:end
        self._pending_auth_after_tts: bool = False     # dispara reconocimiento tras tts:end
        # ========================
        # Para Opción 1 (OCR)
        self._pending_ocr_after_tts: bool = False
        self._ocr_capture_seconds: float | None = None
        # Para Opción 2 (Currency/Dinero)
        self._pending_currency_after_tts: bool = False
        # Para Opción 3 (Expiry/Vencimiento)
        self._pending_expiry_after_tts: bool = False
        self._expiry_capture_seconds: float | None = None
        # Para describir escena (funcion suelta, fuera del menu numerado)
        self._pending_describe_scene_after_tts: bool = False
        # Identificacion por rostro antes de preguntar nombre (evita choques
        # de nombres repetidos: el rostro decide si es alguien nuevo o no).
        self._pending_identify_after_tts: bool = False
        # Usuario nuevo: primero se captura el rostro, el apodo se pregunta
        # despues (para no atarse a un nombre antes de tener las fotos).
        self._pending_enroll_capture_after_tts: bool = False
        self._pending_enroll_temp_folder: Path | None = None
        self._pending_enroll_photo_count: int = 0

        # Opción B: conversación conducida por un modelo de lenguaje. Si está
        # desactivada o no puede atender, se usa el flujo por palabras clave.
        self._llm = LLMAgent(
            actions=self._build_llm_actions(),
            speak=self.speak,
            estado=self._llm_estado,
        )
        self._llm_reset_pending = False
        self._pending_llm_after_intro = False
        self._llm_turn_active = False
        self._user_turn_active = False
        self._pending_user_texts = deque()
        self._introduction = ""

        self._register_events()

        # Watchdog: si por algún driver no llega tts:end, reactivamos STT
        self._watch_stop = threading.Event()
        self._watcher = threading.Thread(target=self._watchdog_loop, daemon=True)
        self._watcher.start()

    # -----------------------------
    # Acciones disponibles para el modelo (opción B)
    # -----------------------------
    def _llm_estado(self) -> str:
        """Resume para el modelo en qué punto está la sesión."""
        if self._authenticated and self._user_name:
            return f"{self._user_name} tiene la identidad verificada y puede usar todas las funciones."
        return "Nadie ha iniciado sesión todavía. Primero hay que registrarse o verificar identidad."

    def _build_llm_actions(self) -> dict:
        """
        Funciones reales que el modelo puede pedir que se ejecuten.

        Cada una comprueba por su cuenta si la sesión está iniciada: es el código
        el que concede o niega el acceso, nunca el modelo. Antes de abrir la
        cámara se espera a que termine de hablar, para que la persona escuche la
        indicación antes de que empiece la captura.
        """

        def _sesion_iniciada() -> str | None:
            if not (self._authenticated and self._user_name):
                return (
                    "La persona todavía no tiene la identidad verificada, así que no "
                    "se ejecutó nada. Ofrécele registrarse o verificar su identidad."
                )
            return None

        def _segundos_validos(segundos) -> float:
            try:
                return float(max(5, min(int(segundos), 60)))
            except (TypeError, ValueError):
                return 10.0

        def identificar_usuario() -> str:
            self.tts.join()
            # Aviso obligatorio antes de encender la cámara: no depende de
            # que el modelo decida avisarlo por su cuenta.
            self.speak("Posiciónate bien frente a la cámara para poder reconocerte.")
            self.tts.join()
            try:
                ok, recognized_name, conf = recognize_best_frame(seconds=5.0, expected_name=None)
            except FileNotFoundError:
                ok, recognized_name, conf = False, None, None
            if ok and recognized_name:
                self._user_name = recognized_name
                self._mark_authenticated()
                return f"Se reconoció a {recognized_name}. Ya está identificado y puede usar las funciones."
            return (
                "No se reconoció ningún rostro conocido: es una persona nueva. "
                "Pregúntale su nombre y usa registrar_usuario para crearle una cuenta."
            )

        def registrar_usuario(nombre: str) -> str:
            nombre = self._extract_name(nombre) or nombre
            if not nombre:
                return "No se entendió el nombre. Pídeselo de nuevo."
            # No se comprueba si el nombre ya existe: la identidad la decide el
            # rostro (identificar_usuario), no el nombre, asi que dos personas
            # pueden llamarse igual sin chocar.
            self.tts.join()
            return self._enroll_workflow(nombre, llm_mode=True)

        def autenticar_usuario(nombre: str) -> str:
            nombre = self._extract_name(nombre) or nombre
            if not nombre:
                return "No se entendió el nombre. Pídeselo de nuevo."
            self._user_name = nombre
            self.tts.join()
            return self._auth_workflow(llm_mode=True)

        def leer_documento(segundos: int = 10) -> str:
            if (error := _sesion_iniciada()) is not None:
                return error
            self.tts.join()
            return self._ocr_workflow(capture_seconds=_segundos_validos(segundos), llm_mode=True)

        def identificar_dinero() -> str:
            if (error := _sesion_iniciada()) is not None:
                return error
            self.tts.join()
            return self._currency_workflow(llm_mode=True)

        def verificar_vencimiento(segundos: int = 10) -> str:
            if (error := _sesion_iniciada()) is not None:
                return error
            self.tts.join()
            return self._expiry_workflow(capture_seconds=_segundos_validos(segundos), llm_mode=True)

        def describir_escena() -> str:
            if (error := _sesion_iniciada()) is not None:
                return error
            self.tts.join()
            return self._describe_scene_workflow(llm_mode=True)

        def cerrar_sesion() -> str:
            nombre = self._user_name or "la persona"
            self._user_name = None
            self._authenticated = False
            self._conversation_state = None
            # El historial se limpia al terminar el turno, no en mitad de él.
            self._llm_reset_pending = True
            self._end_session_camera()
            return f"Sesión de {nombre} cerrada. Despídete brevemente."

        def cerrar_aplicacion() -> str:
            event_bus.publish("app:shutdown")

            def _apagar():
                self.tts.join()
                self.stt.stop()
                self.tts.shutdown()

            threading.Thread(target=_apagar, daemon=True).start()
            return "La aplicación se está cerrando. Despídete en una frase."

        return {
            "identificar_usuario": identificar_usuario,
            "registrar_usuario": registrar_usuario,
            "autenticar_usuario": autenticar_usuario,
            "leer_documento": leer_documento,
            "identificar_dinero": identificar_dinero,
            "verificar_vencimiento": verificar_vencimiento,
            "describir_escena": describir_escena,
            "cerrar_sesion": cerrar_sesion,
            "cerrar_aplicacion": cerrar_aplicacion,
        }

    # -----------------------------
    # Setup de eventos
    # -----------------------------
    def _register_events(self) -> None:
        event_bus.subscribe("tts:start", self._on_tts_start)
        event_bus.subscribe("tts:end", self._on_tts_end)
        event_bus.subscribe("stt:text", self._on_stt_text)

    def _set_state(self, new_state: State) -> None:
        with self._lock:
            old = self.state
            self.state = new_state
        event_bus.publish("ctrl:state", from_state=old, to=new_state)

    # -----------------------------
    # Camara de sesion
    # -----------------------------
    def _mark_authenticated(self) -> None:
        """
        Marca la sesion como autenticada y enciende la camara compartida.

        A partir de aqui, leer_documento/identificar_dinero/verificar_vencimiento/
        describir_escena ya no abren ni cierran la camara en cada peticion: la
        dejan encendida en un hilo secundario mientras dure la sesion.
        """
        self._authenticated = True
        camera_service.start()

    def _end_session_camera(self) -> None:
        camera_service.stop()

    # -----------------------------
    # Utilidades internas
    # -----------------------------
    def _extract_name(self, raw_text: str) -> str:
        """
        Extrae el nombre propio de una respuesta natural a "como quieres que
        te llame?", del estilo "quiero que me llames Juan", "dime Pedro" o
        "me llamo Maria". Usa Claude para entender la frase sin depender de
        una lista fija de patrones; si no hay clave configurada o la llamada
        falla, cae al recorte por patrones conocidos de siempre.
        """
        text = (raw_text or "").strip()
        if not text:
            return text

        via_claude = self._extract_name_with_claude(text)
        if via_claude:
            return via_claude

        return self._extract_name_by_pattern(text)

    def _extract_name_with_claude(self, raw_text: str) -> str | None:
        """Le pide a Claude que saque solo el nombre de la frase completa."""
        if not os.environ.get("ANTHROPIC_API_KEY", "").strip():
            return None

        try:
            import anthropic
        except ImportError:
            return None

        try:
            client = anthropic.Anthropic()
            response = client.messages.create(
                model="claude-haiku-4-5-20251001",
                max_tokens=20,
                messages=[
                    {
                        "role": "user",
                        "content": (
                            "Alguien respondio esto a la pregunta de como quiere "
                            f'que lo llamen: "{raw_text}". Responde UNICAMENTE '
                            "con el nombre propio que dijo, tal cual lo dijo, sin "
                            "nada mas alrededor. Si no dijo ningun nombre, "
                            "responde unicamente: NINGUNO"
                        ),
                    }
                ],
            )
        except Exception as exc:
            from app.utils.logger import logger

            logger.debug(f"Extraccion de nombre con Claude fallo, se usa el patron de siempre: {exc}")
            return None

        text = "".join(
            b.text for b in response.content if getattr(b, "type", None) == "text"
        ).strip()
        if not text or text.upper() == "NINGUNO":
            return None
        return text

    def _extract_name_by_pattern(self, raw_text: str) -> str:
        """
        Respaldo sin IA: recorta frases conocidas como "me llamo Rico" o "mi
        nombre es Rico". Si no reconoce ningun patron, devuelve el texto tal cual.
        """
        text = (raw_text or "").strip()
        lowered = text.lower()

        patterns = [
            "me llamo ",
            "mi nombre es ",
            "mi nombre es: ",
            "mi nombre ",
            "yo soy ",
            "soy ",
            "me dicen ",
        ]

        for p in patterns:
            idx = lowered.find(p)
            if idx != -1:
                candidate = text[idx + len(p) :].strip()
                if candidate:
                    text = candidate
                    break

        # Casos donde el STT antepone "es " al nombre (p.ej., "es ricco")
        if text.lower().startswith("es "):
            text = text[3:]

        # Otros casos: "nombre es ricco" o "nombre es: ricco"
        if text.lower().startswith("nombre es"):
            text = text[10:].lstrip(": ").strip()

        text = text.strip(" .,!?:;")
        return text

    def _extract_seconds(self, raw_text: str) -> int | None:
        """
        Busca un numero en el texto y lo devuelve como segundos.
        Acepta expresiones simples como "20", "20 segundos" o "20 seg".
        """
        if not raw_text:
            return None
        import re as _re
        m = _re.search(r"(\d{1,3})", raw_text)
        if not m:
            # Intentar reconocer el número escrito en palabras (hasta 60)
            normalized = raw_text.lower().strip()
            text_numbers = {
                "cinco": 5,
                "diez": 10,
                "quince": 15,
                "veinte": 20,
                "treinta": 30,
                "cuarenta": 40,
                "cincuenta": 50,
                "sesenta": 60,
            }
            for word, value in text_numbers.items():
                if word in normalized:
                    return value
            return None
        try:
            return int(m.group(1))
        except Exception:
            return None

    def _is_new_user_utterance(self, raw_text: str) -> bool:
        """
        Detecta frases tipo "soy usuario nuevo" / "me quiero registrar".
        """
        t = (raw_text or "").lower()
        return any(
            kw in t
            for kw in [
                "usuario nuevo",
                "soy nuevo",
                "soy nueva",
                "registr",
                "crear una cuenta",
                "crear mi cuenta",
                "quiero registrarme",
                "me voy a registrar",
                "me voy a registro",
                "me quiero registrar",
            ]
        )

    def _is_existing_user_utterance(self, raw_text: str) -> bool:
        """
        Detecta frases tipo "ya tengo cuenta" mientras estamos pidiendo nombre de registro.
        """
        t = (raw_text or "").lower()
        return any(
            kw in t
            for kw in [
                "ya tengo cuenta",
                "ya tengo una cuenta",
                "ya estoy registrado",
                "ya estoy registrada",
                "no soy nuevo",
                "no soy nueva",
                "tengo cuenta",
            ]
        )

    # -----------------------------
    # Ciclo de vida
    # -----------------------------
    @staticmethod
    def _number_to_words_es(num: int) -> str:
        if num < 0:
            return f"menos {Controller._number_to_words_es(abs(num))}"

        units = [
            "cero",
            "uno",
            "dos",
            "tres",
            "cuatro",
            "cinco",
            "seis",
            "siete",
            "ocho",
            "nueve",
        ]

        special = {
            10: "diez",
            11: "once",
            12: "doce",
            13: "trece",
            14: "catorce",
            15: "quince",
            16: "dieciseis",
            17: "diecisiete",
            18: "dieciocho",
            19: "diecinueve",
            20: "veinte",
        }
        for i in range(1, 10):
            special[20 + i] = f"veinti{units[i]}"

        tens = {
            30: "treinta",
            40: "cuarenta",
            50: "cincuenta",
            60: "sesenta",
            70: "setenta",
            80: "ochenta",
            90: "noventa",
        }

        hundreds = {
            2: "doscientos",
            3: "trescientos",
            4: "cuatrocientos",
            5: "quinientos",
            6: "seiscientos",
            7: "setecientos",
            8: "ochocientos",
            9: "novecientos",
        }

        if num < 10:
            return units[num]
        if num < 30:
            return special.get(num, "")
        if num < 100:
            ten = (num // 10) * 10
            rest = num % 10
            if rest == 0:
                return tens.get(ten, str(num))
            return f"{tens.get(ten, str(ten))} y {units[rest]}"
        if num < 1000:
            hundred = num // 100
            rest = num % 100
            if hundred == 1:
                prefix = "cien" if rest == 0 else "ciento"
            else:
                prefix = hundreds.get(hundred, f"{units[hundred]}cientos")
            if rest == 0:
                return prefix
            return f"{prefix} {Controller._number_to_words_es(rest)}"
        if num < 10000:
            thousand = num // 1000
            rest = num % 1000
            if thousand == 1:
                prefix = "mil"
            else:
                prefix = f"{units[thousand]} mil"
            if rest == 0:
                return prefix
            return f"{prefix} {Controller._number_to_words_es(rest)}"
        return str(num)

    def _normalize_tts_text(self, text: str) -> str:
        if not text:
            return text

        def _repl(match: re.Match[str]) -> str:
            value = int(match.group(0))
            if value > 9999:
                digits = " ".join(match.group(0))
                return digits
            return Controller._number_to_words_es(value)

        return re.sub(r"\d+", _repl, text)

    def start(self) -> None:
        """
        Arranca el reconocimiento y da un saludo inicial por TTS.

        Solo este primer saludo es automático: no se dispara
        _start_llm_after_intro aquí, para que OJOZ se quede esperando la
        respuesta real de la persona (si es nueva o ya tiene cuenta) en vez
        de continuar la conversación por su cuenta.
        """
        self.stt.start()

        dev_login = os.environ.get("OJOZ_DEV_LOGIN", "").strip()
        if dev_login:
            # Atajo solo para pruebas locales: entra autenticado sin pasar por
            # la camara, para probar el resto de funciones cuando el
            # enrolamiento facial no es viable (poca luz, sin camara, etc.).
            # No usar en la demo real: no verifica identidad.
            self._user_name = dev_login
            self._mark_authenticated()
            self._introduction = f"Modo de pruebas activo. Sesión iniciada como {dev_login}. ¿En qué puedo ayudarte?"
            self.speak(self._introduction)
            return

        self._introduction = "Hola, soy OJOZ, tu asistente. ¿Es tu primera vez aquí o ya tienes una cuenta registrada?"
        self.speak(self._introduction)

    def _start_llm_after_intro(self) -> bool:
        with self._lock:
            if self._llm_turn_active:
                return True
            if not self._pending_llm_after_intro or self._watch_stop.is_set():
                return False
            self._pending_llm_after_intro = False
            self._llm_turn_active = True
        self._cancel_fallback_rearm()
        self.stt.enable_listening(True)
        self._set_state("PROCESSING")

        def _continue_conversation():
            response = None
            try:
                if not self._watch_stop.is_set() and self._llm.is_available():
                    response = self._llm.start_conversation(self._introduction)
            except Exception:
                from app.utils.logger import logger
                logger.exception("No se pudo iniciar la conversacion tras el saludo")
            finally:
                with self._lock:
                    self._llm_turn_active = False
            if self._watch_stop.is_set():
                return
            if response:
                self.speak(response)
            else:
                self._enable_listening_after_delay()

        # Las acciones del LLM esperan tts.join(): nunca ejecutarlas en el hilo TTS.
        threading.Thread(target=_continue_conversation, daemon=True).start()
        return True

    def speak(self, text: str) -> None:
        """Envía texto a TTS (el gate de STT lo gestiona tts:start/tts:end o el watchdog)."""
        # No cortar una respuesta que empezo mientras el modelo pensaba.
        while self.stt.is_capturing() and not self._watch_stop.is_set():
            time.sleep(0.02)
        if self._watch_stop.is_set():
            return
        # Publicar PRIMERO al chat (antes de que el TTS empiece a hablar)
        event_bus.publish("ui:print", role="app/tts", text=text)
        # Luego enviar al TTS para que hable
        tts_text = self._normalize_tts_text(text)
        self.tts.say(tts_text)
        self._schedule_fallback_rearm()

    # -----------------------------
    # Handlers de eventos
    # -----------------------------
    def _on_tts_start(self) -> None:
        # Gate: deshabilita escucha mientras el TTS habla
        self.stt.enable_listening(False)
        self._set_state("TTS_SPEAKING")

    def _on_tts_end(self) -> None:
        """
        Cuando termina de hablar el TTS:
        - Si hay una acción pendiente (enrolamiento o autenticación), ejecútala ahora.
        - Si no, rearmar STT después de un pequeño delay.
        """
        if self._start_llm_after_intro():
            # Entre las indicaciones habladas, permite responder aunque la
            # identificacion o el LLM sigan trabajando. El turno queda en cola.
            self._enable_listening_after_delay()
            return

        # === Disparadores atados al fin del mensaje del TTS ===
        # Enrolamiento: "Bienvenido {name}, mire a la camara..."
        if self._pending_enroll_after_tts and self._user_name:
            self._pending_enroll_after_tts = False
            threading.Thread(target=self._enroll_workflow, args=(self._user_name,), daemon=True).start()
            return  # no rearmar escucha aquí; el workflow maneja el audio

        # Autenticación: "Por favor mire a la camara para verificar su identidad."
        if self._pending_auth_after_tts:
            self._pending_auth_after_tts = False
            threading.Thread(target=self._auth_workflow, daemon=True).start()
            return  # no rearmar escucha aquí; el workflow maneja el audio

        # OCR (Opción 1): "Apunte la camara al texto..."
        if self._pending_ocr_after_tts:
            self._pending_ocr_after_tts = False
            threading.Thread(target=self._ocr_workflow, daemon=True).start()
            return  # no rearmar escucha; el workflow maneja el audio

        # Currency (Opción 2): "Muestre el billete o moneda a la camara..."
        if self._pending_currency_after_tts:
            self._pending_currency_after_tts = False
            threading.Thread(target=self._currency_workflow, daemon=True).start()
            return  # no rearmar escucha; el workflow maneja el audio

        # Expiry (Opción 3): "Muestre la fecha de vencimiento a la camara..."
        if self._pending_expiry_after_tts:
            self._pending_expiry_after_tts = False
            capture_seconds = self._expiry_capture_seconds or 10.0
            self._expiry_capture_seconds = None
            threading.Thread(target=self._expiry_workflow, args=(capture_seconds,), daemon=True).start()
            return  # no rearmar escucha; el workflow maneja el audio

        # Describir escena: "¿hay personas?", "¿qué ves?"
        if self._pending_describe_scene_after_tts:
            self._pending_describe_scene_after_tts = False
            threading.Thread(target=self._describe_scene_workflow, daemon=True).start()
            return  # no rearmar escucha; el workflow maneja el audio

        # Identificación por rostro (arranque o reinicio de flujo)
        if self._pending_identify_after_tts:
            self._pending_identify_after_tts = False
            threading.Thread(target=self._identify_workflow, daemon=True).start()
            return  # no rearmar escucha; el workflow maneja el audio

        # Usuario nuevo: captura el rostro antes de preguntar el apodo
        if self._pending_enroll_capture_after_tts:
            self._pending_enroll_capture_after_tts = False
            threading.Thread(target=self._enroll_capture_workflow, daemon=True).start()
            return  # no rearmar escucha; el workflow maneja el audio

        # Si no hay workflows pendientes, reactivar escucha normalmente
        self._enable_listening_after_delay()

    def _on_stt_text(self, text: str, confidence=None) -> None:
        text = (text or "").strip()
        if not text:
            return
        with self._lock:
            self._pending_user_texts.append(text)
        self._dispatch_user_turn()

    def _dispatch_user_turn(self) -> bool:
        with self._lock:
            if (self._watch_stop.is_set() or self._llm_turn_active or self._user_turn_active
                    or self._pending_llm_after_intro or self.tts.is_speaking()
                    or not self._pending_user_texts):
                return False
            text = self._pending_user_texts.popleft()
            self._user_turn_active = True
        try:
            self._process_stt_text(text)
        finally:
            with self._lock:
                self._user_turn_active = False
        return True

    def _process_stt_text(self, text: str) -> None:
        # Bloquea escucha mientras procesa la intención
        self.stt.enable_listening(False)
        self._set_state("PROCESSING")

        # Opción B: si la conversación por modelo está activa, atiende ella el
        # turno. Si no está disponible o falla, se sigue con el flujo de siempre.
        if self._llm.is_available():
            with self._lock:
                self._llm_turn_active = True
            try:
                respuesta = self._llm.handle(text)
            finally:
                with self._lock:
                    self._llm_turn_active = False
            if self._llm_reset_pending:
                self._llm.reset()
                self._llm_reset_pending = False
            if respuesta:
                self.speak(respuesta)
                return

        # Verificar si estamos en un flujo de conversación específico
        reply = None


        if self._conversation_state == "waiting_new_user_nickname":
            # Ya se capturo el rostro (vease _enroll_capture_workflow); ahora
            # se guarda con el apodo que diga, sin comprobar si ya existe
            # alguien con el mismo nombre (la carpeta se guarda por ID de
            # usuario, nunca por nombre, asi que nunca chocan).
            nombre = self._extract_name(text)
            self._conversation_state = None
            reply = self._finalize_enrollment(
                nombre,
                self._pending_enroll_temp_folder,
                self._pending_enroll_photo_count,
            )

        elif self._conversation_state == "waiting_identify_name":
            # No reconocimos el rostro: quien sea, se le crea una cuenta nueva
            # con este nombre, sin comprobar si ya existe alguien con el mismo
            # nombre (la identidad la decide el rostro, no el nombre).
            self._user_name = self._extract_name(text)
            self._conversation_state = None
            self._pending_enroll_after_tts = True
            reply = f"Encantado, {self._user_name}."

        elif self._conversation_state == "waiting_new_user_name":
            lower = (text or "").lower()
            if self._is_existing_user_utterance(lower):
                # Corrige: dijo que ya tiene cuenta, cambiar al flujo de usuario recurrente
                self._conversation_state = "waiting_returning_user_name"
                self._user_name = None
                reply = "Entiendo, ya tienes cuenta. ¿Cuál es tu nombre para verificar tu identidad?"
            else:
                # El usuario acaba de decirnos su nombre para registro
                self._user_name = self._extract_name(text)
                self._conversation_state = None  # Salir del flujo

                # Verificar si el usuario ya está registrado en la BD
                if dao.user_exists(self._user_name):
                    # Usuario ya existe, redirigir a autenticación
                    self._pending_auth_after_tts = True
                    reply = "Tu ya tienes una cuenta creada."
                else:
                    # Usuario nuevo, proceder con captura y entrenamiento
                    self._pending_enroll_after_tts = True
                    reply = f"Encantado, {self._user_name}."
        
        elif self._conversation_state == "waiting_returning_user_name":
            lower = (text or "").lower()
            if self._is_new_user_utterance(lower):
                # Corrige: si dice que quiere registrarse, saltar al flujo de registro
                self._conversation_state = "waiting_new_user_name"
                self._user_name = None
                reply = "Entiendo, eres usuario nuevo. ¿Cómo te llamas para crear tu cuenta?"
            else:
                # El usuario que vuelve nos dijo su nombre
                self._user_name = self._extract_name(text)
                
                # Verificar si el usuario existe en la base de datos
                if dao.user_exists(self._user_name):
                    # Usuario encontrado, proceder con autenticación facial
                    self._conversation_state = None  # Salir del flujo
                    self._pending_auth_after_tts = True
                    reply = f"¡Un gusto tenerte de vuelta, {self._user_name}!"
                else:
                    # Usuario no encontrado: preguntar si desea registrarse
                    self._conversation_state = "confirm_register_for_returning"
                    reply = f"{self._user_name}, aun no tienes una cuenta conmigo. ¿Deseas registrarte?"
        
        elif self._conversation_state == "confirm_register_for_returning":
            lower = (text or "").lower()

            positives = [
                "si",
                "sí",
                "claro",
                "por supuesto",
                "me encantaria",
                "me encantaría",
                "me gustaria",
                "me gustaría",
                "me encantaría hacerlo",
                "dale",
                "ok",
            ]
            negatives = [
                "no",
                "ahora no",
                "no gracias",
                "prefiero que no",
                "despues",
                "después",
                "tal vez luego",
                "en otro momento",
            ]

            def _matches(words: list[str]) -> bool:
                return any(phrase in lower for phrase in words)

            if _matches(positives):
                self._conversation_state = "waiting_new_user_name"
                reply = "Perfecto, ¿como te llamas para crear tu cuenta?"
            elif _matches(negatives):
                nombre = self._user_name or "amigo"
                self._conversation_state = None
                self._user_name = None
                reply = f"Esta bien {nombre}, vuelve cuando desees hacerlo."
            else:
                self._conversation_state = "waiting_new_user_name"
                reply = "Para registrarte necesito tu nombre. ¿Como te llamas?"

        elif self._conversation_state == "waiting_new_user_name":
            lower = (text or "").lower()
            if self._is_existing_user_utterance(lower):
                self._conversation_state = "waiting_returning_user_name"
                self._user_name = None
                reply = "Entiendo, ya tienes cuenta. ¿Cual es tu nombre para verificar identidad?"
            else:
                self._user_name = self._extract_name(text)
                self._conversation_state = None
                self._pending_enroll_after_tts = True
                reply = f"Perfecto, {self._user_name}. Vamos a crearte una cuenta. Mira a la camara cuando te lo indique."
        
        elif self._conversation_state == "waiting_ocr_more_time":
            lower = (text or "").lower()
            secs = self._extract_seconds(lower)
            if secs is not None:
                secs = max(5, min(secs, 60))
                self._ocr_capture_seconds = float(secs)
                self._pending_ocr_after_tts = True
                self._conversation_state = None
                reply = f"Perfecto, esperare {secs} segundos antes de capturar."
            elif any(kw in lower for kw in ("ya esta", "ya esta,", "ya esta.", "toma la captura", "captura ahora")):
                self._conversation_state = None
                self._ocr_capture_seconds = None
                self._pending_ocr_after_tts = True
                reply = "De acuerdo, capturo ahora."
            elif any(kw in lower for kw in ("no", "ya no", "mejor ya no", "otra opcion", "luego")):
                self._conversation_state = None
                nombre = self._user_name or "amigo"
                reply = f"Esta bien, {nombre}. Necesitas ayuda en algo mas?"
            elif any(kw in lower for kw in ("si", "claro", "dale")):
                self._conversation_state = "waiting_ocr_seconds"
                reply = "Cuanto tiempo necesitas para enfocar bien el documento en la camara?"
            else:
                reply = "Cuanto tiempo necesitas para enfocar bien el documento en la camara? Dime por ejemplo 10 segundos, o di que no."

        elif self._conversation_state == "waiting_ocr_seconds":
            lower = (text or "").lower()
            if any(kw in lower for kw in ("ya esta", "ya esta,", "ya esta.", "toma la captura", "captura ahora")):
                self._conversation_state = None
                self._ocr_capture_seconds = None
                self._pending_ocr_after_tts = True
                reply = "Listo, capturo ahora."
            else:
                secs = self._extract_seconds(lower)
                if secs is not None:
                    secs = max(5, min(secs, 60))
                    self._ocr_capture_seconds = float(secs)
                    self._pending_ocr_after_tts = True
                    self._conversation_state = None
                    reply = f"Te doy {secs} segundos para enfocar. Si estas listo antes, di 'ya esta' y capturo de inmediato."
                else:
                    reply = "Cuanto tiempo necesitas para enfocar bien el documento en la camara?"
        elif self._conversation_state == "waiting_expiry_more_time":
            lower = (text or "").lower()
            secs = self._extract_seconds(lower)
            positives = ("si", "sí", "claro", "dale", "ok", "vale", "por favor", "porfa", "porfabor")
            negatives = ("no", "ya no", "otra opcion", "mejor no", "luego")
            if secs is not None:
                secs = max(5, min(secs, 60))
                self._expiry_capture_seconds = float(secs)
                self._pending_expiry_after_tts = True
                self._conversation_state = None
                reply = f"De acuerdo, te dare {secs} segundos adicionales para enfocar la fecha."
            elif any(kw in lower for kw in positives):
                # Respuesta positiva sin numero: usar un valor por defecto rapido
                default_secs = 10.0
                self._expiry_capture_seconds = default_secs
                self._pending_expiry_after_tts = True
                self._conversation_state = None
                reply = f"De acuerdo, usare {int(default_secs)} segundos para capturar la fecha."
            elif any(kw in lower for kw in negatives):
                self._conversation_state = None
                nombre = self._user_name or "amigo"
                reply = f"Esta bien {nombre}, ¿Necesitas ayuda en algo mas?"
            else:
                reply = "¿Deseas que te de mas tiempo? Dime 'si' o 'no'."
        elif self._conversation_state == "waiting_expiry_seconds":
            lower = (text or "").lower()
            secs = self._extract_seconds(lower)
            positives = ("si", "sí", "claro", "dale", "ok", "vale", "por favor", "porfa")
            if secs is not None:
                secs = max(5, min(secs, 60))
                self._expiry_capture_seconds = float(secs)
                self._pending_expiry_after_tts = True
                self._conversation_state = None
                reply = f"Perfecto, usare {secs} segundos para capturar la fecha."
            elif any(kw in lower for kw in ("ya esta", "captura ahora", "toma la captura")):
                self._conversation_state = None
                self._expiry_capture_seconds = None
                self._pending_expiry_after_tts = True
                reply = "Entendido, capturo ahora."
            elif any(kw in lower for kw in positives):
                default_secs = 10.0
                self._expiry_capture_seconds = default_secs
                self._pending_expiry_after_tts = True
                self._conversation_state = None
                reply = f"Perfecto, usare {int(default_secs)} segundos para capturar la fecha."
            elif any(kw in lower for kw in ("no", "ya no", "mejor no", "luego")):
                self._conversation_state = None
                nombre = self._user_name or "amigo"
                reply = f"Esta bien {nombre}, ¿Necesitas ayuda en algo mas?"
            else:
                reply = "¿Cuanto tiempo necesitas para mostrar la fecha? Por ejemplo 10 segundos."
        else:
            # Procesamiento normal de intenciones
            reply = self._handle_intent(text)

        if reply:
            self.speak(reply)
        else:
            # Mensaje amable cuando no se comprende la intención
            self.speak("No te logré escuchar muy bien, podrías repetirlo por favor.")

    # -----------------------------
    # Intents
    # -----------------------------
    def _handle_intent(self, text: str) -> str | None:
        intent = infer_intent(text)

        if intent == "exit":
            # Despedida y reinicio del flujo (no cerrar la app)
            farewell = f"Adios {self._user_name}, espero vuelvas pronto." if self._user_name else "Adios, espero vuelvas pronto."

            # Resetear el estado del controller
            self._user_name = None
            self._authenticated = False
            self._conversation_state = None
            self._pending_enroll_after_tts = False
            self._pending_auth_after_tts = False
            self._end_session_camera()

            # Programar reinicio del flujo tras la despedida
            import time

            def _restart_flow():
                time.sleep(1.5)  # Esperar un poco después de la despedida
                # Saludo corto: la presentacion completa solo se da una vez,
                # al arrancar la app (ver start()). Identifica por rostro antes
                # de preguntar nada, igual que al arrancar la app.
                self.speak("¡Hola! ¿Eres usuario nuevo o ya tienes una cuenta registrada?")

            threading.Thread(target=_restart_flow, daemon=True).start()
            # Devolver el mensaje de despedida (lo dirá _on_stt_text)
            return farewell

        if intent == "shutdown":
            # Cerrar la aplicación completamente
            event_bus.publish("app:shutdown")

            # Apagar audio con gracia en un hilo aparte
            def _shutdown():
                self.tts.join()
                self.stt.stop()
                self.tts.shutdown()
            threading.Thread(target=_shutdown, daemon=True).start()

            return "Ok, cerrando la aplicación. Hasta luego!"

        if intent == "greet":
            if self._authenticated and self._user_name:
                return f"Hola de nuevo, {self._user_name}. ¿Qué opción deseas?"
            return "¡Hola! ¿Eres usuario nuevo o ya tienes una cuenta registrada?"

        if intent == "show_menu":
            # Mostrar menú de opciones disponibles
            if self._authenticated and self._user_name:
                return (
                    "Puedo ayudarte con varias cosas:\n"
                    "1. Leer lo que aparece en la cámara.\n"
                    "2. Decirte el valor del dinero que me estés mostrando.\n"
                    "3. Revisar la fecha de vencimiento de un producto.\n"
                    "También puedo describirte lo que ve la cámara si me lo pides.\n"
                    f"¿Qué opción deseas, {self._user_name}?"
                )
            else:
                return "Primero debes autenticarte. Di opcion 2 para iniciar sesion."

        if intent == "new_user_option":
            # Si ya está autenticado, interpretar como Opción 1 del menú
            if self._authenticated and self._user_name:
                self._pending_ocr_after_tts = True
                return "Entendido, apunte la camara al documento para leerlo."
            # Usuario nuevo: primero se captura el rostro; el apodo se
            # pregunta despues, ya con las fotos listas (evita atar el
            # registro a un nombre que otra persona ya podria usar).
            self._pending_enroll_capture_after_tts = True
            return "Entendido."

        if intent == "returning_user_option":
            # Si ya está autenticado, interpretar como Opción 2 del menú
            if self._authenticated and self._user_name:
                self._pending_currency_after_tts = True
                return "Entendido, muestre el billete o moneda a la camara."
            # Ya tiene cuenta: se reconoce por rostro, sin pedir el nombre
            # (asi el nombre nunca decide la identidad).
            self._pending_identify_after_tts = True
            return "Que bueno tenerte de vuelta."

        if intent == "open_camera":
            # Mantengo tu intent original; ahora la cámara se usa en workflows.
            return "Abriendo cámara."
        # ===== Primera opción del menú: Leer documento (OCR) =====
        if intent == "read_image":
            if self._authenticated and self._user_name:
                # Lanza el OCR tras terminar el mensaje de TTS
                self._pending_ocr_after_tts = True
                return "Entendido, apunte la camara al documento para leerlo."
            else:
                return "Primero debes autenticarte. Di opcion 2 para iniciar sesion."

        # ===== Segunda opción del menú: Reconocer valor del dinero (Currency) =====
        # "recognize_currency" es el nombre historico del mismo intent.
        if intent in ("currency_value", "recognize_currency"):
            if self._authenticated and self._user_name:
                # Lanza el reconocimiento de dinero tras terminar el mensaje de TTS
                self._pending_currency_after_tts = True
                return "Entendido, muestre el billete o moneda a la camara."
            else:
                return "Primero debes autenticarte. Di opcion 2 para iniciar sesion."

        # ===== Tercera opción del menú: Verificar fecha de vencimiento (Expiry) =====
        if intent == "verify_expiry":
            if self._authenticated and self._user_name:
                # Lanza la verificación de vencimiento tras terminar el mensaje de TTS
                self._pending_expiry_after_tts = True
                return "Entendido, muestre la fecha de vencimiento del producto a la camara."
            else:
                return "Primero debes autenticarte. Di opcion 2 para iniciar sesion."

        # ===== Describir escena (fuera del menú numerado) =====
        if intent == "describe_scene":
            if self._authenticated and self._user_name:
                self._pending_describe_scene_after_tts = True
                return "Entendido, dejame ver lo que hay frente a la cámara."
            else:
                return "Primero debes autenticarte. Di opcion 2 para iniciar sesion."

        return None

    # -----------------------------
    # Helpers
    # -----------------------------
    def _enable_listening_after_delay(self):
        self._cancel_fallback_rearm()
        delay = getattr(config, "rearm_stt_delay_ms", 400) / 1000.0

        def _reactivate():
            time.sleep(delay)
            if self._watch_stop.is_set() or self.tts.is_speaking():
                return
            with self._lock:
                if self._llm_turn_active:
                    self.stt.enable_listening(True)
                    return
            if self._start_llm_after_intro():
                return
            if self._dispatch_user_turn():
                return
            # Solo reactivar si no estamos ya escuchando
            with self._lock:
                if self.state == "LISTENING":
                    return
            self.stt.enable_listening(True)
            self._set_state("LISTENING")

        # Reactivar en un hilo para no bloquear al publicador del evento
        threading.Thread(target=_reactivate, daemon=True).start()

    def _schedule_fallback_rearm(self) -> None:
        failsafe_ms = getattr(config, "rearm_stt_failsafe_ms", 8000)
        if failsafe_ms <= 0:
            return

        timer = threading.Timer(failsafe_ms / 1000.0, self._force_enable_listening)
        timer.daemon = True

        self._cancel_fallback_rearm()
        with self._lock:
            self._fallback_timer = timer
        timer.start()

    def _cancel_fallback_rearm(self) -> None:
        timer = None
        with self._lock:
            if self._fallback_timer is not None:
                timer = self._fallback_timer
                self._fallback_timer = None
        if timer is not None:
            timer.cancel()

    def _force_enable_listening(self) -> None:
        with self._lock:
            self._fallback_timer = None
            current_state = self.state
            if self._llm_turn_active or self._watch_stop.is_set():
                return

        if self.tts.is_speaking():
            # El TTS sigue hablando de verdad (frase larga): no forzar el
            # microfono a mitad de la frase, o captaria la propia voz de OJOZ
            # por el parlante. Se reintenta pasado el mismo plazo de seguridad.
            self._schedule_fallback_rearm()
            return

        if self._start_llm_after_intro():
            return

        if self._dispatch_user_turn():
            return

        # Solo reactivar si no estamos ya escuchando
        if current_state == "LISTENING":
            return

        self.stt.enable_listening(True)
        self._set_state("LISTENING")

    def _watchdog_loop(self):
        """
        Revisa periódicamente si el TTS ya no está hablando pero seguimos en TTS_SPEAKING.
        Si eso pasa (p.ej., faltó 'tts:end'), reactiva la escucha.
        """
        while not self._watch_stop.is_set():
            time.sleep(0.15)
            with self._lock:
                current = self.state
            if current == "TTS_SPEAKING" and not self.tts.is_speaking():
                # Fallback por watchdog
                self._enable_listening_after_delay()

    def stop(self):
        self._watch_stop.set()
        self._cancel_fallback_rearm()
        camera_service.stop()

    # =============================
    # NUEVOS WORKFLOWS (visión)
    # =============================
    def _enroll_workflow(self, name: str, llm_mode: bool = False) -> str:
        """
        Captura fotos del usuario y actualiza la galeria de embeddings ArcFace.

        Se captura en una carpeta temporal (sin nombre) y recien se asocia al
        nombre en `_finalize_enrollment`, guardando por ID de usuario y no por
        nombre: asi dos personas con el mismo nombre nunca comparten carpeta
        ni se pisan las fotos entre si.

        En `llm_mode` no encadena los mensajes de bienvenida (los da el modelo) y
        devuelve un resumen. El alta de la sesión la marca este código, no el modelo.
        """
        from app.utils.logger import logger

        try:
            # Bloquear escucha durante enrolamiento
            self.stt.enable_listening(False)
            # Aviso obligatorio antes de encender la cámara: nunca se hace
            # reconocimiento facial sin decirlo primero.
            self.speak(f"Perfecto, {name}. Mantente frente a la cámara mientras te capturo.")
            self.tts.join()
            event_bus.publish("camera.opened", index=vision.camera_index)
            event_bus.publish("ui:print", role="sys", text=f"Iniciando captura de {vision.capture_count} rostros para {name}...")

            logger.debug(f"=== INICIO ENROLAMIENTO: {name} ===")

            temp_folder = vision.fotos_dir / f"_pendiente_{uuid.uuid4().hex[:10]}"
            saved = capture_faces(count=vision.capture_count, show_preview=vision.show_preview, folder=temp_folder)
            logger.debug(f"Captura completada: {saved} fotos")

            if saved < vision.min_enrollment_photos:
                shutil.rmtree(temp_folder, ignore_errors=True)
                raise RuntimeError(
                    f"Se capturaron muy pocas fotos ({saved}). "
                    f"Se requieren al menos {vision.min_enrollment_photos}."
                )

            event_bus.publish("ui:print", role="sys", text=f"Se capturaron {saved} rostros. Creando galeria facial...")
            mensaje = self._finalize_enrollment(name, temp_folder, saved)
            event_bus.publish("camera.closed", index=vision.camera_index)

            if llm_mode:
                return mensaje

            if not self._authenticated:
                # _finalize_enrollment fallo (mensaje de error, sesion no marcada)
                self.speak(mensaje)
                return ""

            # Continuar el flujo sin regresar al saludo inicial
            self.speak(f"Listo, {name}. Su cuenta ha sido creada satisfactoriamente.")

            def _continue_after_enroll():
                time.sleep(1.5)  # Esperar un poco después del mensaje de confirmación
                self._pending_enroll_after_tts = False
                self._pending_auth_after_tts = False
                self.speak(f"Un gusto conocerte, {name}. ¿En qué puedo ayudarte?")

            threading.Thread(target=_continue_after_enroll, daemon=True).start()

        except Exception as e:
            logger.error(f"Error en enrolamiento: {e}", exc_info=True)
            event_bus.publish("ui:print", role="sys", text=f"[ERROR] {e}")
            if llm_mode:
                return f"El registro falló: {e}"
            self.speak("Hubo un problema durante el registro. Intente nuevamente.")
        finally:
            # La reactivación de STT se hará tras tts:end del mensaje final
            pass
        return ""

    def _enroll_capture_workflow(self) -> None:
        """
        Usuario nuevo: primero comprueba con una lectura rapida si ese rostro
        ya esta en la galeria (por si dijo "nuevo" por error o ya se habia
        registrado antes). Si ya tiene cuenta, lo avisa y lo deja identificado
        sin duplicar el registro. Si no, recien ahi captura el rostro; el
        apodo se pregunta despues, en `waiting_new_user_nickname`, y se llama
        a `_finalize_enrollment` con las fotos ya listas.
        """
        from app.utils.logger import logger

        try:
            self.stt.enable_listening(False)
            # Aviso obligatorio antes de encender la cámara: nunca se hace
            # reconocimiento facial sin decirlo primero.
            self.speak("Posiciónate bien frente a la cámara para poder reconocerte.")
            self.tts.join()
            event_bus.publish("camera.opened", index=vision.camera_index)
            event_bus.publish("ui:print", role="sys", text="Verificando si ya tienes una cuenta...")

            try:
                ya_existe, nombre_existente, _conf = recognize_best_frame(seconds=3.0, expected_name=None)
            except FileNotFoundError:
                ya_existe, nombre_existente = False, None

            if ya_existe and nombre_existente:
                event_bus.publish("camera.closed", index=vision.camera_index)
                self._user_name = nombre_existente
                self._mark_authenticated()
                event_bus.publish("ui:print", role="sys", text=f"[OK] Ya tenia cuenta: {nombre_existente}")
                self.speak(f"Ya tienes una cuenta, {nombre_existente}. ¿En qué puedo ayudarte?")
                return

            self.speak("Perfecto, vamos a crear tu cuenta. Mantente frente a la cámara mientras te capturo.")
            self.tts.join()
            event_bus.publish("ui:print", role="sys", text=f"Iniciando captura de {vision.capture_count} rostros...")

            temp_folder = vision.fotos_dir / f"_pendiente_{uuid.uuid4().hex[:10]}"
            saved = capture_faces(count=vision.capture_count, show_preview=vision.show_preview, folder=temp_folder)

            event_bus.publish("camera.closed", index=vision.camera_index)

            if saved < vision.min_enrollment_photos:
                shutil.rmtree(temp_folder, ignore_errors=True)
                self.speak(
                    f"Se capturaron muy pocas fotos, se requieren al menos "
                    f"{vision.min_enrollment_photos}. Intenta de nuevo diciendo "
                    "que eres usuario nuevo."
                )
                return

            self._pending_enroll_temp_folder = temp_folder
            self._pending_enroll_photo_count = saved
            self._conversation_state = "waiting_new_user_nickname"
            self.speak("Listo, ya te tengo. ¿Cómo quieres que te llame?")
        except Exception as e:
            logger.error(f"Error en captura de enrolamiento: {e}", exc_info=True)
            event_bus.publish("ui:print", role="sys", text=f"[ERROR] {e}")
            self.speak("Hubo un problema durante la captura. Intente nuevamente.")

    def _finalize_enrollment(self, name: str, temp_folder: Path | None, saved: int) -> str:
        """
        Crea el usuario (siempre nuevo, nunca reutiliza un id por coincidir el
        nombre), mueve las fotos ya capturadas a una carpeta por ID y entrena
        la galeria. Devuelve el mensaje a decir; marca la sesión como
        autenticada solo si todo salió bien.
        """
        from app.utils.logger import logger

        self._pending_enroll_temp_folder = None
        self._pending_enroll_photo_count = 0

        if not temp_folder or not temp_folder.exists():
            return "No encontré la captura de tu rostro. Vamos a intentarlo de nuevo diciendo que eres usuario nuevo."

        try:
            user_id = dao.create_user(name)
            logger.debug(f"Usuario en BD: id={user_id}, nombre={name}")

            final_folder = vision.fotos_dir / f"user_{user_id}"
            if final_folder.exists():
                shutil.rmtree(final_folder, ignore_errors=True)
            shutil.move(str(temp_folder), str(final_folder))
            write_display_name(final_folder, name)

            photo_files = sorted(final_folder.glob("rostro_*.jpg"))
            photo_paths = [str(p.relative_to(vision.base_dir)) for p in photo_files]
            if photo_paths:
                dao.replace_face_photos(user_id, photo_paths)

            dao.insert_enrollment(user_id, saved, notes=f"Captura automática de {saved} rostros")

            model_path = train_dataset()
            model_relative = str(Path(model_path).relative_to(vision.base_dir))
            dao.upsert_global_model(
                file_path=model_relative,
                threshold=vision.face_similarity_threshold,
                version=vision.face_model_name,
                model_type="ArcFace",
            )

            event_bus.publish("ui:print", role="sys", text=f"[OK] Usuario '{name}' registrado exitosamente en la base de datos")

            self._user_name = name
            self._mark_authenticated()
            self._conversation_state = None
            return f"Listo, {name}. Tu cuenta ha sido creada satisfactoriamente. ¿En qué puedo ayudarte?"
        except Exception as e:
            logger.error(f"Error al finalizar el registro: {e}", exc_info=True)
            event_bus.publish("ui:print", role="sys", text=f"[ERROR] {e}")
            shutil.rmtree(temp_folder, ignore_errors=True)
            return "Hubo un problema al crear tu cuenta. Intenta registrarte de nuevo."

    def _identify_workflow(self) -> None:
        """
        Reconocimiento facial abierto (sin nombre esperado): mira quién es
        antes de preguntar nada. Si reconoce a alguien en la galería, entra
        directo autenticado con su nombre guardado. Si no reconoce a nadie
        (o aún no existe la galería), recién ahí pregunta el nombre para
        registrar a una persona nueva.

        Así el nombre deja de ser la llave de identidad: dos personas pueden
        llamarse igual sin chocar, porque cada una es un rostro distinto.
        """
        from app.utils.logger import logger

        try:
            self.stt.enable_listening(False)
            # Aviso obligatorio antes de encender la cámara: nunca se hace
            # reconocimiento facial sin decirlo primero.
            self.speak("Posiciónate bien frente a la cámara para poder reconocerte.")
            self.tts.join()
            event_bus.publish("camera.opened", index=vision.camera_index)
            event_bus.publish("ui:print", role="sys", text="Verificando identidad...")

            try:
                ok, recognized_name, conf = recognize_best_frame(seconds=5.0, expected_name=None)
            except FileNotFoundError:
                ok, recognized_name, conf = False, None, None

            event_bus.publish("camera.closed", index=vision.camera_index)
            logger.debug(f"Identificación abierta: ok={ok}, nombre={recognized_name}, confianza={conf}")

            if ok and recognized_name:
                self._user_name = recognized_name
                self._mark_authenticated()
                event_bus.publish("ui:print", role="sys", text=f"[OK] Identidad reconocida: {recognized_name}")
                self.speak(f"¡Hola de nuevo, {recognized_name}! ¿En qué puedo ayudarte?")
            else:
                self._conversation_state = "waiting_identify_name"
                self.speak("No te reconocí. ¿Cómo quieres que te llame?")
        except Exception as e:
            logger.error(f"Error al identificar: {e}", exc_info=True)
            event_bus.publish("ui:print", role="sys", text=f"[ERROR IDENTIFICAR] {e}")
            self._conversation_state = "waiting_identify_name"
            self.speak("Tuve un problema para verificar tu identidad por ahora. ¿Cómo quieres que te llame?")

    def _auth_workflow(self, llm_mode: bool = False) -> str:
        """
        Verifica la identidad mediante embeddings ArcFace durante unos segundos.

        En `llm_mode` devuelve el veredicto en lugar de conducir la conversación.
        Quien decide si la identidad es válida es el reconocimiento facial: el
        modelo solo recibe el resultado y nunca puede alterar `_authenticated`.
        """
        from app.utils.logger import logger


        try:
            # Bloquear escucha durante autenticación
            self.stt.enable_listening(False)
            # Aviso obligatorio antes de encender la cámara: nunca se hace
            # reconocimiento facial sin decirlo primero.
            self.speak("Posiciónate bien frente a la cámara para poder reconocerte.")
            self.tts.join()
            event_bus.publish("camera.opened", index=vision.camera_index)
            event_bus.publish("ui:print", role="sys", text="Verificando identidad...")

            ok, recognized_name, conf = recognize_best_frame(
                seconds=5.0,
                expected_name=self._user_name,
            )

            event_bus.publish("camera.closed", index=vision.camera_index)
            
            logger.debug(f"Resultado autenticación: ok={ok}, nombre={recognized_name}, confianza={conf}, esperado={self._user_name}")

            if ok and recognized_name:
                # Se reconoció un rostro
                if self._user_name and recognized_name.lower() == self._user_name.lower():
                    # El nombre coincide con el esperado
                    self._mark_authenticated()  # Marcar como autenticado
                    event_bus.publish("ui:print", role="sys", text=f"[OK] Autenticación exitosa: {self._user_name}")
                    if llm_mode:
                        return f"Identidad verificada: es {self._user_name}. Ya puede usar las funciones."
                    self.speak(f"Un gusto conocerte, {self._user_name}. ¿En qué puedo ayudarte?")
                else:
                    # Se reconoció pero NO es la persona esperada
                    self._authenticated = False
                    # Solo registrar en logs; no mostrar mensaje técnico en el chat de la interfaz
                    logger.warning(f"Usuario reconocido como '{recognized_name}' pero se esperaba '{self._user_name}'")
                    esperado = self._user_name
                    self._user_name = None
                    if llm_mode:
                        return (
                            f"El rostro no coincide con {esperado}. No se concede acceso; "
                            "pide el nombre correcto o propón registrarse."
                        )
                    self._conversation_state = "waiting_returning_user_name"  # Volver a pedir nombre
                    self.speak(f"Tu no eres {esperado}, por favor digame correctamente su nombre.")
            else:
                # No se pudo reconocer con suficiente confianza
                self._authenticated = False
                event_bus.publish("ui:print", role="sys", text=f"[FALLO] Reconocimiento fallido (confianza: {conf})")
                self._user_name = None

                if llm_mode:
                    if conf is None:
                        return "No se detectó ningún rostro. Sugiere acomodarse frente a la cámara o registrarse."
                    return "No se reconoció el rostro con suficiente certeza. No se concede acceso."

                if conf is None:
                    # No se detectó ningún rostro - ofrecer registro
                    msg = "No detectó ningun rostro ¿Desea registraste?"
                    self._conversation_state = None  # Permitir que elija opción 1 o diga nombre
                else:
                    # Baja confianza - volver a pedir nombre
                    msg = "No se pudo reconocer de forma satisfactoria. Por favor digame su nombre de nuevo."
                    self._conversation_state = "waiting_returning_user_name"  # Volver a pedir nombre

                self.speak(msg)

        except FileNotFoundError:
            # Modelo inexistente
            event_bus.publish("ui:print", role="sys", text="[FALLO] Modelo de reconocimiento no encontrado")
            if llm_mode:
                return "Todavía no hay ningún usuario registrado, así que no se puede verificar identidad."
            self.speak("Aun no hay modelo de reconocimiento. Por favor registrese primero con la opcion uno.")
        except Exception as e:
            logger.error(f"Error en autenticación: {e}", exc_info=True)
            event_bus.publish("ui:print", role="sys", text=f"[ERROR] {e}")
            if llm_mode:
                return f"La verificación falló: {e}"
            self.speak("Ocurrio un error durante la autenticacion.")
        finally:
            # La reactivación de STT se hará tras tts:end del mensaje final
            pass
        return ""

    # ======== NUEVO: OCR (Opción 1) ========
    def _ocr_workflow(self, capture_seconds: float | None = None, llm_mode: bool = False) -> str:
        """
        Captura durante unos segundos, realiza OCR (Tesseract) y lee en voz alta el texto.

        En `llm_mode` no enuncia el aviso inicial (ya lo dijo el modelo) ni encadena
        preguntas de seguimiento, y devuelve un resumen del resultado. El texto del
        documento se lee siempre literal, nunca reformulado.
        """
        from app.utils.logger import logger

        sid = None
        try:
            # Registrar sesión en BD (si el DAO lo soporta)
            try:
                sid = dao.start_session("ocr", None)
            except Exception:
                sid = None

            # Bloquear escucha durante el OCR
            self.stt.enable_listening(False)
            event_bus.publish("ui:print", role="sys", text="Preparando cámara para capturar documento...")

            capture_seconds = capture_seconds or self._ocr_capture_seconds or 10.0
            self._ocr_capture_seconds = None
            if not llm_mode:
                self.speak(f"Muestre el documento frente a la camara. La captura se realizara en {int(capture_seconds)} segundos.")

            # Capturar foto y procesar (ventana configurable para enfocar)
            ok, text, conf = read_text_best_frame(seconds=capture_seconds, lang='spa')

            if ok and text:
                # Guardar en BD si está disponible
                try:
                    if sid is not None:
                        dao.insert_ocr_result(sid, text=text, language=None, confidence=conf)
                        dao.finish_session(sid, ok=True, details=f"conf={conf}")
                except Exception:
                    pass

                # Limitar TTS si el texto es muy largo
                MAX_TTS_CHARS = 1200  # Aumentado para documentos más largos
                spoken = text[:MAX_TTS_CHARS] + (" …" if len(text) > MAX_TTS_CHARS else "")

                event_bus.publish("ui:print", role="sys", text=f"[OK] OCR listo (conf={conf:.1f}%)" if conf else "[OK] OCR listo")
                event_bus.publish("ui:print", role="app/ocr", text=text)

                # Leer por voz con introducción
                word_count = len(text.split())
                self.speak(f"He detectado {word_count} palabras. Leyendo contenido:")
                self.speak(spoken)

                if llm_mode:
                    return f"Documento leído: {word_count} palabras. Ya se le leyó el texto a la persona."
            else:
                try:
                    if sid is not None:
                        dao.finish_session(sid, ok=False, details=f"conf={conf}")
                except Exception:
                    pass
                event_bus.publish("ui:print", role="sys", text="[FALLO] No encontré texto en el documento")
                if llm_mode:
                    return "No se encontró texto legible en el documento."
                self._conversation_state = "waiting_ocr_more_time"
                self.speak("No encontré texto que leer. ¿Necesitas más tiempo?")
        except Exception as e:
            logger.error(f"Error en OCR: {e}", exc_info=True)
            event_bus.publish("ui:print", role="sys", text=f"[ERROR OCR] {e}")
            try:
                if sid is not None:
                    dao.finish_session(sid, ok=False, details=repr(e))
            except Exception:
                pass
            if llm_mode:
                return f"La lectura falló: {e}"
            self.speak("Ocurrio un error leyendo el texto.")
        finally:
            # La reactivacion de STT ocurrira despues del tts:end del mensaje anterior
            pass
        return ""

    # ======== NUEVO: CURRENCY (Opción 2) ========
    def _currency_workflow(self, llm_mode: bool = False) -> str:
        """
        Detecta el valor de billetes (10, 20, 50, 100, 200 soles) o monedas.
        Usa OCR en región central del billete para leer el número grande.

        En `llm_mode` omite el aviso inicial y devuelve un resumen del resultado.
        El valor detectado se enuncia siempre tal cual lo devuelve la detección.
        """
        from app.utils.logger import logger

        sid = None
        try:
            # Registrar sesión en BD
            try:
                sid = dao.start_session("currency", None)
            except Exception:
                sid = None

            # Bloquear escucha durante la detección
            self.stt.enable_listening(False)
            event_bus.publish("ui:print", role="sys", text="Preparando cámara, por favor mantenga quieto el billete...")
            if not llm_mode:
                self.speak("Muestre el billete o moneda a la camara. Tendra 15 segundos para posicionar el billete.")


            # Detectar dinero (15 segundos de captura para mejor posicionamiento)
            ok, curr, value, conf = detect_currency_best_frame(seconds=15.0)

            if ok and value is not None:
                # Guardar en BD si está disponible
                try:
                    if sid is not None:
                        dao.insert_currency_detection(sid, currency=curr or "PEN", value=float(value), confidence=conf)
                        dao.finish_session(sid, ok=True, details=f"{curr} {value}, conf={conf}")
                except Exception:
                    pass

                # Formatear valor para TTS y UI usando la divisa detectada.
                # curr puede ser None si no se identifico la divisa; en ese caso
                # se habla de "unidades" y si es una divisa desconocida se usa su
                # propio codigo.
                moneda_natural = _CURR_TO_UNIT.get(curr, curr) if curr else "unidades"

                es_entero = abs(value - int(value)) < 1e-6
                value_str = str(int(value)) if es_entero else f"{value:.2f}"

                # Determinar si es billete o moneda (heurística simple)
                tipo = "un billete de" if value >= 10 else "una moneda de"

                # Responder por voz usando la moneda detectada
                self.speak(f"Detecté {tipo} {value_str} {moneda_natural}.")

                conf_str = f" (conf={conf:.1f}%)" if conf is not None else ""
                event_bus.publish("ui:print", role="sys", text=f"[OK] Detectado: {curr} {value_str}{conf_str}")
                event_bus.publish("ui:print", role="app/currency", text=f"Valor: {value_str} {moneda_natural}")

                if llm_mode:
                    return f"Detectado: {value_str} {moneda_natural}. Ya se le comunicó a la persona."
            else:
                try:
                    if sid is not None:
                        dao.finish_session(sid, ok=False, details=f"conf={conf}")
                except Exception:
                    pass
                
                event_bus.publish("ui:print", role="sys", text="[FALLO] No pude detectar el valor del dinero")
                if llm_mode:
                    return "No se pudo determinar el valor. Conviene más luz o acercar el billete."
                self.speak("No pude determinar el valor de ese dinero. Intente con mejor iluminacion o acercando mas el billete a la camara.")

        except Exception as e:
            logger.error(f"Error en currency workflow: {e}", exc_info=True)
            event_bus.publish("ui:print", role="sys", text=f"[ERROR CURRENCY] {e}")
            try:
                if sid is not None:
                    dao.finish_session(sid, ok=False, details=repr(e))
            except Exception:
                pass
            if llm_mode:
                return f"La identificación falló: {e}"
            self.speak("Ocurrio un error al identificar el dinero.")
        finally:
            # La reactivacion de STT ocurrira despues del tts:end del mensaje anterior
            pass
        return ""

    # ======== NUEVO: EXPIRY (Opción 3) ========
    def _expiry_workflow(self, capture_seconds: float = 10.0, llm_mode: bool = False) -> str:
        """
        Lee la fecha de vencimiento y determina si el producto está vencido.
        Busca palabras clave como "vencimiento", "caducidad", "exp", etc.

        En `llm_mode` omite el aviso inicial y devuelve un resumen del resultado.
        La fecha y el estado se enuncian siempre tal cual los devuelve la verificación.
        """
        from app.utils.logger import logger

        sid = None
        try:
            # Registrar sesión en BD
            try:
                sid = dao.start_session("expiry", None)
            except Exception:
                sid = None

            # Bloquear escucha durante la detección
            self.stt.enable_listening(False)
            event_bus.publish("ui:print", role="sys", text="Preparando cámara para verificar fecha de vencimiento...")
            if not llm_mode:
                self.speak("Muestre la fecha de vencimiento del producto a la camara. La verificacion comenzara ahora.")


            # Verificar vencimiento con ventana configurable
            ok, date_text, is_expired, conf = check_expiry_best_frame(seconds=capture_seconds)

            if ok and date_text:
                # Guardar en BD si está disponible
                try:
                    if sid is not None:
                        dao.insert_expiry_check(
                            session_id=sid,
                            product_name=None,
                            expiry_date=date_text,
                            is_expired=bool(is_expired),
                            confidence=conf if conf is not None else 0.0,
                            raw_text=None,
                        )
                        dao.finish_session(sid, ok=True, details=f"{date_text}, vencido={is_expired}, conf={conf}")
                except Exception:
                    pass

                # Responder por voz según el estado
                if is_expired:
                    self.speak(f"El producto esta vencido. La fecha de vencimiento era {date_text}.")
                    status_label = "VENCIDO"
                else:
                    self.speak(f"El producto no esta vencido. La fecha de vencimiento es {date_text}.")
                    status_label = "VIGENTE"

                detalle = f"{status_label}: {date_text}"
                if conf is not None:
                    detalle += f" (conf={conf:.1f}%)"
                event_bus.publish("ui:print", role="sys", text=detalle)
                
                event_bus.publish("ui:print", role="app/expiry", text=f"Fecha: {date_text} - Estado: {'VENCIDO' if is_expired else 'VIGENTE'}")

                if llm_mode:
                    estado = "vencido" if is_expired else "vigente"
                    return f"Fecha {date_text}, producto {estado}. Ya se le comunicó a la persona."
            else:
                try:
                    if sid is not None:
                        dao.finish_session(sid, ok=False, details=f"conf={conf}")
                except Exception:
                    pass
                
                event_bus.publish("ui:print", role="sys", text="[FALLO] No pude leer la fecha de vencimiento")
                if llm_mode:
                    return "No se pudo leer la fecha de vencimiento con claridad."
                self._conversation_state = "waiting_expiry_more_time"
                self._expiry_capture_seconds = None
                self.speak("No pude leer la fecha de vencimiento con claridad. ¿Deseas más tiempo?")

        except Exception as e:
            logger.error(f"Error en expiry workflow: {e}", exc_info=True)
            event_bus.publish("ui:print", role="sys", text=f"[ERROR EXPIRY] {e}")
            try:
                if sid is not None:
                    dao.finish_session(sid, ok=False, details=repr(e))
            except Exception:
                pass
            if llm_mode:
                return f"La verificación falló: {e}"
            self.speak("Ocurrio un error al verificar la fecha de vencimiento.")
        finally:
            # La reactivacion de STT ocurrira despues del tts:end del mensaje anterior
            pass
        return ""

    # ======== NUEVO: DESCRIBIR ESCENA ========
    def _describe_scene_workflow(self, llm_mode: bool = False) -> str:
        """
        Describe el entorno que ve la cámara (personas, obstáculos, objetos)
        con Claude Vision, como función suelta, sin pasar por documento/dinero/
        vencimiento. Pensada para preguntas naturales como "¿hay personas?"
        o "¿qué ves?".
        """
        from app.utils.logger import logger

        try:
            self.stt.enable_listening(False)
            event_bus.publish("ui:print", role="sys", text="Analizando el entorno...")
            if not llm_mode:
                self.speak("Voy a describir lo que ve la cámara.")

            ok, description = describe_scene_best_frame(seconds=5.0)

            if ok and description:
                self.speak(description)
                event_bus.publish("ui:print", role="app/scene", text=description)
                if llm_mode:
                    return "Ya se le describió el entorno a la persona con el resultado literal."
            else:
                event_bus.publish("ui:print", role="sys", text="[FALLO] No pude describir el entorno")
                if llm_mode:
                    return "No se pudo describir el entorno (sin conexión o sin clave configurada)."
                self.speak("No pude describir el entorno en este momento. Verifica tu conexión a internet.")
        except Exception as e:
            logger.error(f"Error al describir escena: {e}", exc_info=True)
            event_bus.publish("ui:print", role="sys", text=f"[ERROR ESCENA] {e}")
            if llm_mode:
                return f"La descripción falló: {e}"
            self.speak("Ocurrio un error al describir el entorno.")
        finally:
            # La reactivacion de STT ocurrira despues del tts:end del mensaje anterior
            pass
        return ""
