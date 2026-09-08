from __future__ import annotations

import os
import queue
import tempfile
import threading
import json
from urllib.error import HTTPError, URLError
from urllib.parse import quote
from urllib.request import Request, urlopen

import pyttsx3
from gtts import gTTS

from app.config.settings import config, tts as tts_config
from app.core.event_bus import event_bus
from app.utils.logger import logger

try:
    import pygame
except ImportError:  # pragma: no cover - dependencias opcionales
    pygame = None


PYGAME_TICK_HZ = 20


class ElevenLabsRequestError(RuntimeError):
    def __init__(self, status_code: int, detail: str) -> None:
        super().__init__(f"ElevenLabs HTTP {status_code}: {detail}")
        self.status_code = status_code


class TTS:
    """
    Motor TTS con cola y eventos:
      - ElevenLabs (online, principal si hay API key)
      - gTTS (online, respaldo)
      - pyttsx3 (voces del sistema, respaldo offline)
    Publica eventos "tts:start" / "tts:end" y "ui:print" para errores.
    """

    def __init__(self):
        self._q: queue.Queue = queue.Queue()
        self._speaking = threading.Event()
        self._shutdown = False
        self._stop_token = object()

        self._engine: pyttsx3.Engine | None = None
        self._engine_lock = threading.RLock()
        self._elevenlabs_disabled = False

        self._play_lock = threading.RLock()
        self._mixer_ready = threading.Event()

        self._worker = threading.Thread(target=self._loop, daemon=True)
        self._worker.start()

    # -----------------------------
    # Ciclo de vida
    # -----------------------------
    def warmup(self) -> None:
        """Permite inicializar recursos de audio antes del primer saludo."""
        if pygame is None:
            return
        try:
            self._ensure_mixer()
        except Exception as exc:  # pragma: no cover - solo log
            logger.debug(f"No se pudo calentar mixer: {exc}")

    def _loop(self) -> None:
        with self._engine_lock:
            self._engine = None

        while True:
            item = self._q.get()
            if item is None or item is self._stop_token:
                self._q.task_done()
                break

            try:
                self._handle_text(str(item))
            finally:
                self._q.task_done()

        if pygame and self._mixer_ready.is_set():
            try:
                pygame.mixer.music.stop()
                pygame.mixer.quit()
            except Exception:
                pass

    def _handle_text(self, text: str) -> None:
        try:
            logger.debug(f"TTS procesando: {text}")
            self._speaking.set()
            event_bus.publish("tts:start")
            elevenlabs_key = os.environ.get("ELEVENLABS_API_KEY", "").strip()
            if elevenlabs_key and not self._elevenlabs_disabled:
                try:
                    # 1) Voz neuronal, si el usuario configuro la API.
                    self._speak_with_elevenlabs(text)
                    return
                except Exception as e_elevenlabs:
                    if isinstance(e_elevenlabs, ElevenLabsRequestError) and e_elevenlabs.status_code in {
                        401,
                        402,
                        403,
                        404,
                    }:
                        # Estos estados no se resuelven reintentando cada frase.
                        # Se vuelve a comprobar al reiniciar OJOZ.
                        self._elevenlabs_disabled = True
                    logger.warning(
                        "ElevenLabs fallo, intentando gTTS: %s",
                        e_elevenlabs,
                    )

            try:
                # 2) gTTS online.
                self._speak_with_gtts(text)
                return
            except Exception as e_gtts:
                logger.warning("gTTS fallo, intentando pyttsx3: %s", e_gtts)

            try:
                # 3) pyttsx3 (offline, voces del sistema).
                self._speak_with_pyttsx3(text)
            except Exception as e_pyttsx3:
                logger.error(
                    "Todos los motores TTS fallaron. "
                    f"gTTS: {e_gtts}, pyttsx3: {e_pyttsx3}"
                )
                event_bus.publish(
                    "ui:print",
                    role="sys",
                    text="Error TTS: No se pudo reproducir audio",
                )
        finally:
            self._speaking.clear()
            event_bus.publish("tts:end")
            with self._engine_lock:
                self._engine = None

    # -----------------------------
    # Motores específicos
    # -----------------------------
    def _speak_with_elevenlabs(self, text: str) -> None:
        """Genera MP3 con ElevenLabs sin guardar la clave en el proyecto."""
        api_key = os.environ.get("ELEVENLABS_API_KEY", "").strip()
        voice_id = os.environ.get("ELEVENLABS_VOICE_ID", "").strip()
        if not api_key:
            raise RuntimeError("Falta ELEVENLABS_API_KEY")
        if not voice_id:
            raise RuntimeError("Falta ELEVENLABS_VOICE_ID")

        model_id = os.environ.get(
            "ELEVENLABS_MODEL",
            getattr(tts_config, "elevenlabs_model", "eleven_flash_v2_5"),
        ).strip()
        endpoint = (
            "https://api.elevenlabs.io/v1/text-to-speech/"
            f"{quote(voice_id, safe='')}?output_format=mp3_44100_128"
        )
        payload = json.dumps(
            {
                "text": text,
                "model_id": model_id,
                "language_code": "es",
            }
        ).encode("utf-8")
        request = Request(
            endpoint,
            data=payload,
            headers={
                "Accept": "audio/mpeg",
                "Content-Type": "application/json",
                "xi-api-key": api_key,
            },
            method="POST",
        )

        try:
            with urlopen(
                request,
                timeout=getattr(tts_config, "elevenlabs_timeout_seconds", 15.0),
            ) as response:
                audio = response.read()
        except HTTPError as exc:
            detail = f"HTTP {exc.code}"
            try:
                body = json.loads(exc.read().decode("utf-8", errors="replace"))
                api_detail = body.get("detail", {})
                if isinstance(api_detail, dict):
                    detail = api_detail.get("code") or api_detail.get("message") or detail
                elif api_detail:
                    detail = str(api_detail)
            except (ValueError, OSError):
                pass
            raise ElevenLabsRequestError(exc.code, detail) from exc
        except URLError as exc:
            raise RuntimeError("No se pudo conectar con ElevenLabs") from exc

        if not audio:
            raise RuntimeError("ElevenLabs devolvio audio vacio")

        tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".mp3")
        tmp_path = tmp.name
        try:
            tmp.write(audio)
            tmp.close()
            self._play_with_pygame(tmp_path)
        finally:
            try:
                tmp.close()
            except Exception:
                pass
            self._safe_remove(tmp_path)

    def _speak_with_gtts(self, text: str) -> None:
        """
        gTTS: requiere internet para cada texto.
        Genera un MP3 temporal y lo reproduce con pygame.
        """
        lang = getattr(tts_config, "language", None) or "es"

        tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".mp3")
        tmp_path = tmp.name
        tmp.close()

        tts_obj = gTTS(text=text, lang=lang, slow=False)
        tts_obj.save(tmp_path)

        try:
            self._play_with_pygame(tmp_path)
        finally:
            self._safe_remove(tmp_path)

    def _speak_with_pyttsx3(self, text: str) -> None:
        engine = pyttsx3.init()
        self._configure_engine(engine)

        with self._engine_lock:
            self._engine = engine

        logger.debug(f"TTS reproduciendo (pyttsx3 fallback): {text}")
        try:
            engine.say(text)
            engine.runAndWait()
        finally:
            try:
                del engine
            except Exception:
                pass

    def _configure_engine(self, engine: pyttsx3.Engine) -> None:
        try:
            engine.setProperty("rate", getattr(tts_config, "rate", 180))
            engine.setProperty("volume", getattr(tts_config, "volume", 1.0))
            voice_name = getattr(tts_config, "voice", None)
            if voice_name:
                for v in engine.getProperty("voices"):
                    if voice_name.lower() in v.name.lower():
                        engine.setProperty("voice", v.id)
                        break
        except Exception as exc:
            logger.debug(f"No se pudo configurar pyttsx3: {exc}")

    def _play_with_pygame(self, path: str) -> None:
        if pygame is None:
            raise RuntimeError("pygame no está instalado para reproducir audio")

        self._ensure_mixer()

        with self._play_lock:
            pygame.mixer.music.load(path)
            pygame.mixer.music.play()
            clock = pygame.time.Clock()
            while pygame.mixer.music.get_busy():
                clock.tick(PYGAME_TICK_HZ)
            pygame.mixer.music.stop()

    def _ensure_mixer(self) -> None:
        if pygame is None or self._mixer_ready.is_set():
            return
        try:
            pygame.mixer.init()
            self._mixer_ready.set()
            logger.debug("Mixer de pygame inicializado una sola vez.")
        except Exception as exc:
            self._mixer_ready.clear()
            raise RuntimeError(f"No se pudo inicializar pygame.mixer: {exc}") from exc

    @staticmethod
    def _safe_remove(path: str) -> None:
        try:
            os.remove(path)
        except OSError:
            pass

    # -----------------------------
    # API pública
    # -----------------------------
    def play_sound_file(self, path: str) -> bool:
        """Reproduce un archivo de audio externo (wav/mp3) usando pygame."""
        if pygame is None:
            logger.debug("pygame no está instalado; no se puede reproducir audio externo.")
            return False
        if not path:
            return False
        if not os.path.isfile(path):
            logger.warning(f"No se encontró archivo de sonido: {path}")
            return False
        try:
            self._play_with_pygame(path)
            return True
        except Exception as exc:
            logger.warning(f"No se pudo reproducir sonido '{path}': {exc}")
            return False

    def say(self, text: str) -> None:
        """Encola texto para ser hablado."""
        if self._shutdown:
            logger.debug("TTS apagado: ignorando say()")
            return
        self._q.put(text)

    def stop(self) -> None:
        """Intenta detener el habla actual (para barge-in)."""
        with self._engine_lock:
            if self._engine is not None:
                try:
                    self._engine.stop()
                except Exception:
                    pass
        if pygame and pygame.mixer.get_init():
            try:
                pygame.mixer.music.stop()
            except Exception:
                pass

    def is_speaking(self) -> bool:
        return self._speaking.is_set()

    def join(self) -> None:
        """Espera a que la cola se vacíe (todas las frases habladas)."""
        self._q.join()

    def shutdown(self) -> None:
        """Cierra el worker y libera recursos."""
        if not self._shutdown:
            self._shutdown = True
            self._q.put(self._stop_token)
            try:
                self._worker.join(timeout=2.0)
            except Exception:
                pass
