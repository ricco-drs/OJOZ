from __future__ import annotations
import audioop
import binascii
import json
import os
import re
import threading
import time
import traceback
import wave
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

import speech_recognition as sr

from app.audio.denoise import TARGET_SAMPLE_RATE, denoiser
from app.audio.microphone import Microphone, input_devices
from app.config.settings import stt as stt_config
from app.core.event_bus import event_bus
from app.utils.logger import logger

class _SilenceDetected(Exception):
    pass


class _ListeningInterrupted(Exception):
    pass


class ElevenLabsSTTError(RuntimeError):
    def __init__(self, status_code: int, detail: str) -> None:
        super().__init__(f"ElevenLabs STT HTTP {status_code}: {detail}")
        self.status_code = status_code


class _PhraseSource(sr.AudioSource):
    """Deja adaptar SR al ambiente, pero congela el umbral al empezar la voz."""

    def __init__(self, source, recognizer, cancelled=None, on_voice=None):
        self.source = source
        self.recognizer = recognizer
        self.cancelled = cancelled
        self.on_voice = on_voice
        self.SAMPLE_RATE = source.SAMPLE_RATE
        self.SAMPLE_WIDTH = source.SAMPLE_WIDTH
        self.CHUNK = source.CHUNK
        self.stream = self

    def read(self, frames):
        if self.cancelled and self.cancelled():
            raise _ListeningInterrupted()
        data = self.source.stream.read(frames)
        if data and audioop.rms(data, self.SAMPLE_WIDTH) > self.recognizer.energy_threshold:
            self.recognizer.dynamic_energy_threshold = False
            if self.on_voice:
                self.on_voice()
                self.on_voice = None
        return data


class STT:
    """
    Reconocimiento de voz online:
      - Google Speech (via speech_recognition)
    Publica:
      - stt:start / stt:end
      - stt:text (text=..., confidence=...)
      - ui:print (mensajes al usuario)
    """

    def __init__(self):
        self._rec = sr.Recognizer()
        self._rec.operation_timeout = 10.0
        self._language = getattr(stt_config, "language", None) or "es-ES"

        # Seleccion de micro
        self._mic_name = "desconocido"
        self._device_index = stt_config.device_index
        self._device_name_hint = (getattr(stt_config, "device_name_hint", None) or "").casefold().strip()
        self._configured_device_index = stt_config.device_index
        if self._device_index is None:
            self._device_index = self._auto_select_microphone()
        try:
            names = sr.Microphone.list_microphone_names() or []
            idx = self._device_index
            if idx is not None and 0 <= idx < len(names):
                self._mic_name = names[idx]
            elif names:
                self._mic_name = names[0]
        except Exception:
            pass

        # Parametros de SR
        self._rec.pause_threshold = stt_config.pause_threshold
        self._rec.phrase_threshold = stt_config.phrase_threshold
        self._rec.non_speaking_duration = stt_config.non_speaking_duration
        self._rec.dynamic_energy_threshold = stt_config.dynamic_energy_threshold
        if stt_config.energy_threshold is not None:
            self._rec.energy_threshold = stt_config.energy_threshold

        raw_phrase_limit = stt_config.phrase_time_limit
        self._phrase_time_limit = raw_phrase_limit if raw_phrase_limit and raw_phrase_limit > 0 else None
        raw_max = getattr(stt_config, "max_phrase_seconds", None)
        if (raw_max is None or raw_max <= 0) and self._phrase_time_limit:
            raw_max = self._phrase_time_limit * 2.0
        self._max_phrase_seconds = raw_max if raw_max and raw_max > 0 else None
        self._listen_timeout = getattr(stt_config, "listen_timeout", None)
        if self._listen_timeout is not None and self._listen_timeout <= 0:
            self._listen_timeout = None
        self._extend_phrase = bool(getattr(stt_config, "extend_phrase", False))
        self._continuation_timeout = stt_config.continuation_timeout
        self._calibration_duration = max(0.6, getattr(stt_config, "calibration_duration", 1.0) or 1.0)
        self._energy_boost = float(getattr(stt_config, "energy_boost", 1.0) or 1.0)
        self._min_energy_threshold = float(getattr(stt_config, "min_energy_threshold", 0.0) or 0.0)
        self._recalibrate_after_empty = max(1, int(getattr(stt_config, "recalibrate_after_empty", 4) or 4))
        self._strict_device_lock = bool(getattr(stt_config, "strict_device_lock", True))
        self._snr_min_ratio = max(1.5, float(getattr(stt_config, "snr_min_ratio", 3.0) or 3.0))
        self._min_rms = float(getattr(stt_config, "min_rms", 15.0) or 15.0)
        self._max_energy_threshold = float(getattr(stt_config, "max_energy_threshold", 0.0) or 0.0)
        self._max_required_rms = float(getattr(stt_config, "max_required_rms", 3500.0) or 3500.0)
        self._noise_floor_rms: float | None = None
        self._base_energy_threshold: float | None = None
        self._apply_energy_boost()

        self._running = threading.Event()
        self._capturing = threading.Event()
        self._listen_enabled = threading.Event()
        # Modo manual (empujar para hablar): el microfono empieza siempre
        # desactivado y solo la tecla de espacio (via set_push_to_talk) puede
        # encenderlo. Mientras este activo, cualquier otro intento de
        # reactivar el microfono desde el resto de la app se ignora, para que
        # no compita con el control manual.
        self._manual_mode = bool(getattr(stt_config, "push_to_talk", True))
        # Grabacion de empujar-para-hablar: mientras se mantiene la tecla se
        # acumula audio crudo sin ninguna deteccion de silencio; recien al
        # soltar se limpia el ruido y se transcribe todo de una sola vez.
        self._ptt_recording = False
        self._ptt_frames: list[bytes] = []
        self._ptt_sample_rate = 16000
        self._ptt_sample_width = 2
        self._calibrated = False
        self._empty_results = 0
        self._warned_google = False
        self._elevenlabs_stt_disabled = False
        self._last_logged_device_index: int | None = None
        self._has_logged_mic = False
        self._last_error_message = ""
        self._recognition_failed = False
        self._listen_generation = 0

        self._thread = threading.Thread(target=self._loop, daemon=True)

    # -----------------------------
    # Seleccion automatica de microfono
    # -----------------------------
    @staticmethod
    def _probe_microphone_rms(index: int) -> float | None:
        try:
            with Microphone(device_index=index) as source:
                # Algunos drivers entregan ceros durante el arranque del stream.
                for _ in range(max(1, int(0.2 * source.SAMPLE_RATE / source.CHUNK))):
                    source.stream.read(source.CHUNK)
                values = []
                for _ in range(max(1, int(0.5 * source.SAMPLE_RATE / source.CHUNK))):
                    buf = source.stream.read(source.CHUNK)
                    if buf:
                        buf = audioop.bias(buf, source.SAMPLE_WIDTH, -audioop.avg(buf, source.SAMPLE_WIDTH))
                        values.append(float(audioop.rms(buf, source.SAMPLE_WIDTH)))
            if not values:
                return None
            values.sort()
            return values[len(values) // 2]
        except Exception as exc:
            logger.debug(f"Microfono descartado idx={index}: {exc}")
            return None

    def _auto_select_microphone(self) -> int | None:
        """Respeta la entrada de Windows y descarta streams sin senal."""
        try:
            candidates = self._available_inputs()
        except Exception as exc:
            logger.debug(f"No se pudieron listar microfonos: {exc}")
            return None

        fallback = None
        for device in candidates:
            idx, name = device["index"], device["name"]
            rms = self._probe_microphone_rms(idx)
            if rms is None:
                continue
            if fallback is None:
                fallback = idx
            logger.debug(f"Entrada idx={idx}, nombre='{name}', rms={rms:.1f}")
            if rms > 0:
                return idx
        return fallback

    def _available_inputs(self) -> list[dict]:
        devices = input_devices()
        if self._device_name_hint:
            devices = [d for d in devices if self._device_name_hint in d["name"].casefold()]
        return devices

    def _microphone_candidates(self) -> list[int]:
        devices = self._available_inputs()
        indices = [d["index"] for d in devices]
        if not indices and self._device_name_hint:
            raise OSError(f"No hay una entrada conectada con el nombre '{self._device_name_hint}'")
        if self._strict_device_lock and self._configured_device_index is not None:
            return [self._configured_device_index]
        if self._device_index in indices:
            indices.remove(self._device_index)
            indices.insert(0, self._device_index)
        return indices

    # -----------------------------
    # Ciclo de vida
    # -----------------------------
    def start(self) -> None:
        # Controller.start emite el saludo despues de esta llamada: calibrar
        # aqui evita medir la propia voz de OJOZ o abrir dos streams a la vez.
        if not self._calibrated:
            self._calibrate_microphone()
        self._running.set()

        # El modelo de supresión de ruido se carga en paralelo para no demorar
        # el saludo inicial. Si aún no está listo cuando llegue la primera
        # frase, esa frase se transcribe sin filtrar.
        if getattr(stt_config, "denoise_enabled", False):
            threading.Thread(target=denoiser.warmup, daemon=True).start()

        if not self._thread.is_alive():
            if self._thread.ident is not None:
                self._thread = threading.Thread(target=self._loop, daemon=True)
            self._thread.start()

    def stop(self) -> None:
        self._running.clear()

    def is_capturing(self) -> bool:
        return self._capturing.is_set()

    # -----------------------------
    # Seleccion de microfono en vivo (panel de ajustes)
    # -----------------------------
    def list_input_devices(self) -> list[dict]:
        """Microfonos reales disponibles, ordenados con el actual primero si aplica."""
        try:
            devices = input_devices()
        except Exception as exc:
            logger.debug(f"No se pudieron listar microfonos: {exc}")
            return []
        return [
            {"index": d["index"], "name": d.get("name") or f"Dispositivo {d['index']}"}
            for d in devices
        ]

    def get_device_index(self) -> int | None:
        return self._device_index

    def set_device_index(self, index: int) -> None:
        """
        Cambia el microfono en caliente. Recalibra el ruido ambiente del
        nuevo dispositivo en un hilo aparte (tarda ~1.5s) para no bloquear la
        interfaz mientras se aplica el cambio.
        """
        if index == self._configured_device_index:
            return
        self._device_index = index
        self._configured_device_index = index
        self._calibrated = False
        self._has_logged_mic = False
        threading.Thread(target=self._calibrate_microphone, daemon=True).start()

    def enable_listening(self, value: bool) -> None:
        if value and self._manual_mode:
            # En modo manual, solo set_push_to_talk (la tecla de espacio)
            # enciende el microfono, con su propia grabacion; se ignora
            # cualquier reactivacion automatica del modo de escucha continua
            # (fin de TTS, workflows, etc.).
            return
        if value:
            self._listen_enabled.set()
            event_bus.publish("ui:print", role="sys", text="Micrófono activo (escuchando).")
        else:
            self._listen_generation += 1
            self._listen_enabled.clear()
            event_bus.publish("ui:print", role="sys", text="Micrófono en pausa (TTS hablando).")
        # Evento dedicado para indicadores visuales (badge de estado del mic),
        # separado del texto de arriba para no tener que interpretar strings.
        event_bus.publish("mic:state", active=value)

    def set_push_to_talk(self, active: bool) -> None:
        """
        Unico punto de entrada para el microfono en modo manual (mantener
        presionada la tecla de espacio).

        A diferencia de la escucha automatica (que corta la frase sola por
        deteccion de silencio, pensada para "siempre escuchando"), aqui se
        graba audio crudo sin ninguna deteccion mientras se mantiene la
        tecla; recien al soltarla se limpia el ruido y se transcribe todo lo
        grabado de una sola vez. Asi la persona decide exactamente cuando
        empieza y termina su frase, en vez de que lo decida el silencio.
        """
        if active:
            self._start_ptt_recording()
        else:
            self._stop_ptt_recording_and_transcribe()

    def _start_ptt_recording(self) -> None:
        if self._ptt_recording:
            return
        self._ptt_recording = True
        self._ptt_frames = []
        event_bus.publish("ui:print", role="sys", text="Micrófono activo (escuchando).")
        event_bus.publish("mic:state", active=True)
        event_bus.publish("stt:start")
        threading.Thread(target=self._ptt_record_loop, daemon=True).start()

    def _ptt_record_loop(self) -> None:
        """Acumula audio crudo del microfono mientras _ptt_recording sea True."""
        try:
            candidates = self._microphone_candidates()
            device_idx = candidates[0] if candidates else self._device_index
            with Microphone(device_index=device_idx) as source:
                self._ptt_sample_rate = source.SAMPLE_RATE
                self._ptt_sample_width = source.SAMPLE_WIDTH
                while self._ptt_recording:
                    buf = source.stream.read(source.CHUNK)
                    if buf:
                        self._ptt_frames.append(buf)
        except Exception as exc:
            logger.debug(f"Error grabando en modo empujar-para-hablar: {exc}")
            self._report_error("No se pudo grabar del microfono. Revisa el dispositivo de entrada.")

    def _stop_ptt_recording_and_transcribe(self) -> None:
        if not self._ptt_recording:
            return
        self._ptt_recording = False
        event_bus.publish("mic:state", active=False)
        event_bus.publish("ui:print", role="sys", text="Micrófono en pausa.")

        # Pequena espera para que el hilo de grabacion termine su ultima
        # lectura en curso antes de leer/vaciar la lista de fragmentos.
        time.sleep(0.1)
        frames, self._ptt_frames = self._ptt_frames, []
        event_bus.publish("stt:end")

        if not frames:
            return

        raw = b"".join(frames)
        try:
            audio = sr.AudioData(raw, self._ptt_sample_rate, self._ptt_sample_width)
        except Exception as exc:
            logger.debug(f"No se pudo armar el audio grabado: {exc}")
            return

        duration = self._estimate_duration(audio)
        rms = self._compute_rms(audio)
        logger.debug(f"PTT grabado: {duration:.2f}s, rms={rms:.1f}")
        self._save_ptt_debug_wav(audio, "ptt_original.wav")

        cleaned = self._denoise(audio)
        if cleaned is not audio:
            logger.debug(f"PTT tras supresion de ruido: rms={self._compute_rms(cleaned):.1f}")
            self._save_ptt_debug_wav(cleaned, "ptt_filtrado.wav")

        text, conf = self._recognize(cleaned)
        if not text and cleaned is not audio and not self._recognition_failed:
            text, conf = self._recognize(audio)

        text = (text or "").strip()
        if text:
            event_bus.publish("stt:text", text=text, confidence=conf)
        else:
            logger.debug("PTT: ningun motor pudo transcribir la grabacion.")

    @staticmethod
    def _save_ptt_debug_wav(audio: sr.AudioData, filename: str) -> None:
        """Guarda la ultima captura para poder escucharla y comparar (sobreescribe)."""
        try:
            from app.config.settings import vision
            out_dir = vision.data_dir / "audio_check"
            out_dir.mkdir(parents=True, exist_ok=True)
            with wave.open(str(out_dir / filename), "wb") as wf:
                wf.setnchannels(1)
                wf.setsampwidth(audio.sample_width)
                wf.setframerate(audio.sample_rate)
                wf.writeframes(audio.get_raw_data())
        except Exception as exc:
            logger.debug(f"No se pudo guardar wav de depuracion: {exc}")

    # -----------------------------
    # Calibracion y energia
    # -----------------------------
    def _calibrate_microphone(self) -> None:
        try:
            last_err = None
            candidates = self._microphone_candidates()

            for device_idx in candidates:
                try:
                    with Microphone(device_index=device_idx) as source:
                        self._rec.adjust_for_ambient_noise(source, duration=self._calibration_duration)
                        measured = self._measure_ambient_rms(source, duration=0.6)
                    if measured is None or measured == 0:
                        raise OSError("La entrada devuelve silencio digital; revisa si esta silenciada")
                    self._apply_energy_boost(measured)
                    self._update_noise_floor(measured)
                    self._device_index = device_idx
                    try:
                        names = sr.Microphone.list_microphone_names() or []
                        if device_idx is None:
                            self._mic_name = names[0] if names else "desconocido"
                        elif 0 <= device_idx < len(names):
                            self._mic_name = names[device_idx]
                    except Exception:
                        pass
                    self._calibrated = True
                    msg = f"Micrófono calibrado: idx={self._device_index}, nombre='{self._mic_name}'."
                    event_bus.publish("ui:print", role="sys", text=msg)
                    self._log_selected_microphone(prefix="Microfono calibrado")
                    return
                except Exception as e:
                    last_err = e
                    logger.debug(f"Microphone calibration failed for index {device_idx}: {e}")

            if self._configured_device_index is not None:
                event_bus.publish(
                    "ui:print",
                    role="sys",
                    text=(
                        f"No se pudo calibrar el microfono configurado "
                        f"(idx={self._configured_device_index}). Prueba otro indice "
                        "con OJOZ_MIC_INDEX."
                    ),
                )
                return

            err_text = str(last_err) if last_err else "error desconocido"
            event_bus.publish("ui:print", role="sys",
                              text=f"No se pudo calibrar el micrófono ({err_text}). Verifica permisos y dispositivo.")
        except Exception as e:
            err_text = str(e)
            event_bus.publish("ui:print", role="sys",
                              text=f"No se pudo calibrar el micrófono ({err_text}). Verifica permisos y dispositivo.")
            logger.warning(f"STT calibration failed: {e}")

    def _apply_energy_boost(self, ambient_rms: float | None = None) -> None:
        """
        Fija el umbral de energía a partir del ruido ambiente medido.

        El umbral se recalcula siempre desde la medición, nunca desde su propio
        valor anterior: multiplicarlo por el boost en cada calibración lo iba
        desplazando turno a turno (al alza con boost mayor que 1, hacia cero con
        boost menor que 1) hasta dejar de representar el entorno real.
        """
        try:
            if ambient_rms is not None and ambient_rms > 0:
                # Mismo margen sobre el ruido que aplica speech_recognition.
                margen = float(getattr(self._rec, "dynamic_energy_ratio", 1.5) or 1.5)
                base = ambient_rms * margen
            elif self._base_energy_threshold:
                base = self._base_energy_threshold
            else:
                base = float(getattr(self._rec, "energy_threshold", 0.0) or 0.0)

            if base <= 0:
                return

            self._base_energy_threshold = base
            target = base * self._energy_boost
            if self._min_energy_threshold:
                target = max(self._min_energy_threshold, target)
            if self._max_energy_threshold:
                # Un ruido puntual muy fuerte no debe dejar el umbral tan alto
                # que el asistente deje de oír durante el resto de la sesión.
                target = min(self._max_energy_threshold, target)

            if abs(target - float(getattr(self._rec, "energy_threshold", 0.0) or 0.0)) < 1e-3:
                return
            logger.debug(f"Ajustando energy_threshold de {self._rec.energy_threshold} a {target}")
            self._rec.energy_threshold = target
        except Exception as exc:
            logger.debug(f"No se pudo ajustar energy_threshold: {exc}")

    def _measure_ambient_rms(self, source: sr.Microphone, duration: float = 0.5) -> float | None:
        """
        Mide el RMS real del ruido ambiente leyendo directamente del micrófono.

        No se deriva de energy_threshold porque ese valor ya viene multiplicado
        por energy_boost, lo que distorsionaría la comparación de SNR. Se usa la
        mediana de las lecturas para que un golpe puntual no eleve el piso.
        """
        try:
            width = source.SAMPLE_WIDTH
            chunk = source.CHUNK
            rate = source.SAMPLE_RATE
            reads = max(int((rate / chunk) * duration), 1)

            values: list[float] = []
            for _ in range(reads):
                buf = source.stream.read(chunk)
                if not buf:
                    break
                buf = audioop.bias(buf, width, -audioop.avg(buf, width))
                values.append(float(audioop.rms(buf, width)))

            if not values:
                return None
            values.sort()
            return values[len(values) // 2]
        except Exception as exc:
            logger.debug(f"No se pudo medir el ruido ambiente: {exc}")
            return None

    def _update_noise_floor(self, measured: float | None) -> None:
        """
        Actualiza el piso de ruido suavizando el valor anterior.

        El suavizado evita que una medición tomada mientras alguien habla
        dispare el piso y deje al asistente sordo durante el resto de la sesión.
        """
        if measured is None or measured <= 0:
            if self._noise_floor_rms is None:
                self._capture_noise_floor()
            return

        if self._noise_floor_rms is None:
            self._noise_floor_rms = measured
        else:
            self._noise_floor_rms = 0.7 * self._noise_floor_rms + 0.3 * measured

    def _capture_noise_floor(self) -> None:
        """
        Respaldo cuando no se pudo leer el micrófono: estima el piso a partir
        del umbral calibrado por speech_recognition.
        """
        try:
            thr = getattr(self._rec, "energy_threshold", None)
            if thr is None:
                return
            boost = self._energy_boost if self._energy_boost > 0 else 1.0
            # Se descuenta el boost para recuperar una estimación del ambiente.
            self._noise_floor_rms = max(float(thr) / boost, self._min_rms)
        except Exception:
            self._noise_floor_rms = None

    def _compute_rms(self, audio: sr.AudioData) -> float:
        """Calcula RMS del audio crudo para estimar nivel de voz."""
        try:
            raw = audio.get_raw_data()
            width = max(getattr(audio, "sample_width", 2), 1)
            return float(audioop.rms(raw, width))
        except Exception:
            return 0.0

    def _passes_voice_gate(self, audio: sr.AudioData) -> bool:
        """
        Descarta el audio que no destaque lo suficiente sobre el ruido de fondo.

        La exigencia es relativa al ruido medido, no un valor fijo: en una sala
        en silencio basta con hablar con normalidad, mientras que en un lugar
        concurrido se pide una voz cercana al micrófono, que es justo lo que
        distingue al usuario de las conversaciones del entorno.
        """
        # Las pausas de una respuesta corta no deben diluir el nivel de voz.
        raw = audio.get_raw_data()
        window = max(1, int(audio.sample_rate * 0.03)) * audio.sample_width
        levels = sorted(audioop.rms(raw[i:i + window], audio.sample_width)
                        for i in range(0, len(raw), window))
        rms = float(levels[min(len(levels) - 1, int(len(levels) * 0.8))]) if levels else 0.0
        floor = self._noise_floor_rms or self._min_rms
        min_allowed = max(self._min_rms, floor * self._snr_min_ratio)
        # Aun con mucho ruido, el listón debe seguir siendo alcanzable por una
        # voz cercana; si no, el asistente dejaría de responder por completo.
        min_allowed = min(min_allowed, self._max_required_rms)
        if rms < min_allowed:
            logger.debug(f"Audio descartado por SNR: rms={rms:.1f}, piso={floor:.1f}, req={min_allowed:.1f}")
            return False
        return True

    def _refresh_noise_floor(self) -> bool:
        """Usa el ruido aprendido por SR mientras esperaba voz, sin grabar aparte."""
        if not self._rec.dynamic_energy_threshold:
            return False
        measured = self._rec.energy_threshold / self._rec.dynamic_energy_ratio
        self._noise_floor_rms = measured
        self._apply_energy_boost(measured)
        return True

    def _has_clean_voice(self, cleaned: sr.AudioData, original: sr.AudioData) -> bool:
        """Rescata voz que DTLN separo de un ambiente con mucha energia."""
        if cleaned is original:
            return False
        raw = cleaned.get_raw_data()
        window = max(1, int(cleaned.sample_rate * 0.03)) * cleaned.sample_width
        levels = sorted(audioop.rms(raw[i:i + window], cleaned.sample_width)
                        for i in range(0, len(raw), window))
        if not levels:
            return False
        floor = levels[int(len(levels) * 0.2)]
        required = max(self._min_rms, floor * self._snr_min_ratio)
        return sum(level > required for level in levels) * 0.03 >= self._rec.phrase_threshold

    # -----------------------------
    def _log_selected_microphone(self, prefix: str = "Microfono", only_if_changed: bool = False) -> None:
        """
        Registra en consola el microfono activo. Si only_if_changed es True,
        evita repetir el mismo mensaje para el mismo dispositivo.
        """
        try:
            names = sr.Microphone.list_microphone_names() or []
            if self._device_index is None:
                name = names[0] if names else "desconocido"
            elif 0 <= self._device_index < len(names):
                name = names[self._device_index]
            else:
                name = "desconocido"
        except Exception:
            name = "desconocido"

        changed = (
            not getattr(self, "_has_logged_mic", False)
            or self._device_index != getattr(self, "_last_logged_device_index", None)
            or name != getattr(self, "_mic_name", "")
        )
        self._mic_name = name
        if only_if_changed and not changed:
            return

        self._last_logged_device_index = self._device_index
        self._has_logged_mic = True
        logger.debug(f"{prefix}: idx={self._device_index}, nombre='{self._mic_name}'")

    # -----------------------------
    # Captura de audio extendida
    # -----------------------------
    def _record_phrase(self, source: sr.Microphone) -> sr.AudioData:
        timeout = self._listen_timeout
        chunk_limit = self._phrase_time_limit
        max_total = self._max_phrase_seconds
        extend = bool(self._extend_phrase and chunk_limit)

        segments: list[sr.AudioData] = []
        total_duration = 0.0

        generation = self._listen_generation
        cancelled = None
        if self._running.is_set():
            cancelled = lambda: not self._running.is_set() or generation != self._listen_generation
        tracked_source = _PhraseSource(source, self._rec, cancelled, self._capturing.set)
        dynamic = self._rec.dynamic_energy_threshold
        try:
            while True:
                try:
                    chunk = self._rec.listen(
                        tracked_source,
                        timeout=timeout if not segments else self._continuation_timeout,
                        phrase_time_limit=min(chunk_limit, max_total - total_duration)
                        if chunk_limit and max_total else chunk_limit,
                    )
                except sr.WaitTimeoutError:
                    if segments:
                        break
                    raise
                if not chunk.frame_data:
                    break
                segments.append(chunk)
                total_duration += self._estimate_duration(chunk)
                if not extend or (max_total and total_duration >= max_total):
                    break
        finally:
            self._rec.dynamic_energy_threshold = dynamic
            self._capturing.clear()

        if not segments:
            raise sr.WaitTimeoutError("No se recibio una frase")
        if len(segments) == 1:
            return segments[0]
        return self._combine_segments(segments)

    @staticmethod
    def _estimate_duration(audio: sr.AudioData) -> float:
        if not audio:
            return 0.0
        frame_count = len(audio.frame_data)
        sample_width = max(getattr(audio, "sample_width", 2), 1)
        sample_rate = max(getattr(audio, "sample_rate", 16000), 1)
        return frame_count / (sample_width * sample_rate)

    @staticmethod
    def _combine_segments(segments: list[sr.AudioData]) -> sr.AudioData:
        if not segments:
            raise ValueError("No hay segmentos para combinar.")
        base = segments[0]
        raw = b"".join(seg.frame_data for seg in segments)
        return sr.AudioData(raw, base.sample_rate, base.sample_width)

    # -----------------------------
    # Supresión de ruido
    # -----------------------------
    def _denoise(self, audio: sr.AudioData) -> sr.AudioData:
        """
        Limpia el audio con el modelo de IA antes de transcribir.

        Se convierte a 16 kHz mono, que es el formato con el que se entrenó el
        modelo y el que prefiere el reconocedor. Ante cualquier problema se
        devuelve el audio recibido sin modificar.
        """
        if not stt_config.denoise_enabled:
            return audio
        try:
            raw16 = audio.get_raw_data(convert_rate=TARGET_SAMPLE_RATE, convert_width=2)
            cleaned = denoiser.enhance_pcm16(raw16, TARGET_SAMPLE_RATE)
            if cleaned is None:
                return audio
            # No entregar una frase muda o casi borrada por el modelo.
            original_rms = audioop.rms(raw16, 2)
            filtered_rms = audioop.rms(cleaned, 2)
            if filtered_rms < max(1.0, original_rms * 0.05):
                logger.debug("DTLN atenuo demasiado la frase; se conserva el audio original.")
                return audio
            return sr.AudioData(cleaned, TARGET_SAMPLE_RATE, 2)
        except Exception as exc:
            logger.debug(f"No se pudo aplicar supresión de ruido: {exc}")
            return audio

    # -----------------------------
    # Reconocedores
    # -----------------------------
    def _recognize_with_google(self, audio: sr.AudioData) -> tuple[str, float | None]:
        self._recognition_failed = False
        try:
            text = self._rec.recognize_google(audio, language=self._language)
            text = (text or "").strip()
            if text:
                text = self._apply_custom_normalization(text)
            return text, None
        except sr.UnknownValueError:
            logger.debug("Google STT no entendió el audio.")
            return "", None
        except sr.RequestError as e:
            self._recognition_failed = True
            logger.debug(f"Google STT request error: {e}")
            self._report_error("El microfono capta audio, pero no se pudo conectar con el reconocimiento de voz. Revisa tu conexion a internet.")
            return "", None
        except Exception as e:
            self._recognition_failed = True
            logger.debug(f"Google STT error: {e}")
            self._report_error(f"No se pudo transcribir el audio: {e}")
            return "", None

    def _recognize_with_elevenlabs(self, audio: sr.AudioData) -> tuple[str, float | None]:
        """Transcribe con ElevenLabs Scribe, usando la misma clave que la voz de salida."""
        api_key = os.environ.get("ELEVENLABS_API_KEY", "").strip()
        if not api_key:
            raise RuntimeError("Falta ELEVENLABS_API_KEY")

        model_id = os.environ.get(
            "ELEVENLABS_STT_MODEL",
            getattr(stt_config, "elevenlabs_stt_model", "scribe_v1"),
        ).strip() or "scribe_v1"
        wav_bytes = audio.get_wav_data(convert_rate=16000, convert_width=2)

        boundary = binascii.hexlify(os.urandom(16)).decode()
        body = bytearray()
        body += (
            f'--{boundary}\r\nContent-Disposition: form-data; name="model_id"\r\n\r\n'
            f'{model_id}\r\n'
        ).encode()
        body += (
            f'--{boundary}\r\nContent-Disposition: form-data; name="file"; filename="audio.wav"\r\n'
            f'Content-Type: audio/wav\r\n\r\n'
        ).encode()
        body += wav_bytes
        body += f'\r\n--{boundary}--\r\n'.encode()

        request = Request(
            "https://api.elevenlabs.io/v1/speech-to-text",
            data=bytes(body),
            headers={
                "xi-api-key": api_key,
                "Content-Type": f"multipart/form-data; boundary={boundary}",
            },
            method="POST",
        )
        try:
            with urlopen(
                request,
                timeout=getattr(stt_config, "elevenlabs_timeout_seconds", 20.0),
            ) as response:
                result = json.loads(response.read().decode("utf-8"))
        except HTTPError as exc:
            detail = f"HTTP {exc.code}"
            try:
                payload = json.loads(exc.read().decode("utf-8", errors="replace"))
                api_detail = payload.get("detail", {})
                if isinstance(api_detail, dict):
                    detail = api_detail.get("message") or api_detail.get("code") or detail
                elif api_detail:
                    detail = str(api_detail)
            except (ValueError, OSError):
                pass
            raise ElevenLabsSTTError(exc.code, detail) from exc
        except URLError as exc:
            raise RuntimeError("No se pudo conectar con ElevenLabs") from exc

        return (result.get("text") or "").strip(), None

    def _recognize(self, audio: sr.AudioData) -> tuple[str, float | None]:
        """
        Motor principal de transcripción, con Google como respaldo gratuito.

        ElevenLabs Scribe usa la misma cuenta que la voz de salida y es mas
        robusto ante ruido de fondo. Si falla por credenciales o saldo se
        desactiva por el resto de la sesion (no tiene sentido reintentar en
        cada frase) y se sigue con Google sin interrumpir el servicio.
        """
        elevenlabs_key = os.environ.get("ELEVENLABS_API_KEY", "").strip()
        if elevenlabs_key and not self._elevenlabs_stt_disabled:
            try:
                text, _ = self._recognize_with_elevenlabs(audio)
                self._recognition_failed = False
                if text:
                    text = self._apply_custom_normalization(text)
                return text, None
            except Exception as exc:
                if isinstance(exc, ElevenLabsSTTError) and exc.status_code in {401, 402, 403, 404}:
                    self._elevenlabs_stt_disabled = True
                    self._report_error(
                        "ElevenLabs STT no disponible, se usa Google Speech como respaldo."
                    )
                logger.warning(f"ElevenLabs STT fallo, intentando Google: {exc}")

        return self._recognize_with_google(audio)

    def _report_error(self, message: str) -> None:
        if message != self._last_error_message:
            event_bus.publish("ui:print", role="sys", text=message)
            self._last_error_message = message

    # -----------------------------
    # Normalización de texto
    # -----------------------------
    def _apply_custom_normalization(self, text: str) -> str:
        if not text:
            return text
        if not getattr(stt_config, "normalize_ricco", True):
            return text
        # Sustituir siempre "rico" -> "ricco" respetando mayúsculas/minúsculas.
        def _repl(match: re.Match) -> str:
            original = match.group(0)
            target = "ricco"
            if original.isupper():
                return target.upper()
            if original[0].isupper():
                return target.capitalize()
            return target

        return re.sub(r"\brico\b", _repl, text, flags=re.IGNORECASE)

    # -----------------------------
    # Bucle principal
    # -----------------------------
    def _loop(self):
        while self._running.is_set():
            if not self._listen_enabled.is_set():
                time.sleep(0.02)
                continue

            generation = self._listen_generation
            event_bus.publish("stt:start")
            text, conf = "", None
            self._recognition_failed = False

            try:
                if not self._warned_google:
                    engine_name = (
                        "ElevenLabs Scribe"
                        if os.environ.get("ELEVENLABS_API_KEY", "").strip()
                        else "Google Speech"
                    )
                    event_bus.publish(
                        "ui:print",
                        role="sys",
                        text=f"Usando {engine_name} para transcribir (requiere internet).",
                    )
                    self._warned_google = True

                last_err = None
                candidates = self._microphone_candidates()

                audio = None
                timeout_silence = False
                for device_idx in candidates:
                    try:
                        logger.debug(f"Trying microphone for listen: device_index={device_idx}")
                        with Microphone(device_index=device_idx) as source:
                            audio = self._record_phrase(source)
                        self._device_index = device_idx
                        self._log_selected_microphone(prefix="Microfono en uso", only_if_changed=True)
                        break
                    except sr.WaitTimeoutError:
                        last_err = sr.WaitTimeoutError("timeout")
                        timeout_silence = True
                        logger.debug(f"No se detectó voz en el tiempo esperado (idx={device_idx}).")
                        self._device_index = device_idx
                        audio = None
                        self._refresh_noise_floor()
                        break
                    except Exception as e:
                        last_err = e
                        logger.debug(f"Microphone listen failed for index {device_idx}: {e}")

                if audio is None:
                    if timeout_silence:
                        raise _SilenceDetected("timeout")
                    raise RuntimeError(f"Unable to open any microphone: {last_err}")

                if not self._running.is_set() or generation != self._listen_generation or not self._listen_enabled.is_set():
                    continue

                cleaned = self._denoise(audio)
                if not self._passes_voice_gate(audio) and not self._has_clean_voice(cleaned, audio):
                    raise _SilenceDetected("noise_gate")

                # Solo se limpia lo que ya se considera voz: ahorra CPU y evita
                # procesar ruido que igualmente se iba a descartar.
                logger.debug(f"Voz capturada: {self._estimate_duration(audio):.1f}s, rms={self._compute_rms(audio):.1f}")
                text, conf = self._recognize(cleaned)
                if not text and cleaned is not audio and not self._recognition_failed:
                    text, conf = self._recognize(audio)

            except _ListeningInterrupted:
                continue
            except _SilenceDetected:
                logger.debug("Silencio detectado: reintentando escucha.")
                self._empty_results += 1
                if self._empty_results >= self._recalibrate_after_empty:
                    self._empty_results = 0
                    self._report_error("No se detecta tu voz. Comprueba que el microfono este activado y habla cerca de el.")
                time.sleep(0.2)
                continue
            except Exception:
                logger.debug("STT listen error:\n" + traceback.format_exc())
                self._report_error("No se pudo leer el microfono. Revisa el dispositivo de entrada y los permisos de microfono de Windows.")
                time.sleep(1.0)
                continue
            finally:
                event_bus.publish("stt:end")

            text = (text or "").strip()
            if text:
                # El audio ya se valido antes de transcribirlo. Que OJOZ empiece
                # a hablar durante la peticion de red no borra esa frase previa.
                if not self._running.is_set():
                    continue
                self._last_error_message = ""
                self._empty_results = 0
                event_bus.publish("stt:text", text=text, confidence=conf)
            elif not self._recognition_failed:
                self._empty_results += 1
                if self._empty_results >= self._recalibrate_after_empty:
                    self._empty_results = 0
                    self._report_error("Se recibe audio, pero no se entienden las palabras. Acerca el microfono y vuelve a hablar.")

            time.sleep(0.05)
