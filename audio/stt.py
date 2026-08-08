from __future__ import annotations
import audioop
import re
import threading
import time
import traceback
import speech_recognition as sr

from app.config.settings import stt as stt_config
from app.core.event_bus import event_bus
from app.utils.logger import logger

class _SilenceDetected(Exception):
    pass


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
        self._language = getattr(stt_config, "language", None) or "es-ES"

        # Seleccion de micro
        self._mic_name = "desconocido"
        self._device_index = stt_config.device_index
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
        self._calibration_duration = max(0.6, getattr(stt_config, "calibration_duration", 1.0) or 1.0)
        self._energy_boost = float(getattr(stt_config, "energy_boost", 1.0) or 1.0)
        self._min_energy_threshold = float(getattr(stt_config, "min_energy_threshold", 0.0) or 0.0)
        self._recalibrate_after_empty = max(1, int(getattr(stt_config, "recalibrate_after_empty", 4) or 4))
        self._strict_device_lock = bool(getattr(stt_config, "strict_device_lock", True))
        self._snr_min_ratio = max(1.5, float(getattr(stt_config, "snr_min_ratio", 3.0) or 3.0))
        self._min_rms = float(getattr(stt_config, "min_rms", 15.0) or 15.0)
        self._noise_floor_rms: float | None = None
        self._apply_energy_boost()

        self._running = threading.Event()
        self._listen_enabled = threading.Event()
        self._calibrated = False
        self._empty_results = 0
        self._recalibration_lock = threading.Lock()
        self._warned_google = False
        self._last_logged_device_index: int | None = None
        self._has_logged_mic = False

        self._thread = threading.Thread(target=self._loop, daemon=True)

    # -----------------------------
    # Ciclo de vida
    # -----------------------------
    def start(self) -> None:
        self._running.set()
        if not self._calibrated:
            self._calibrate_microphone()
        if not self._thread.is_alive():
            self._thread.start()

    def stop(self) -> None:
        self._running.clear()

    def enable_listening(self, value: bool) -> None:
        if value:
            self._listen_enabled.set()
            event_bus.publish("ui:print", role="sys", text="Micrófono activo (escuchando).")
        else:
            self._listen_enabled.clear()
            event_bus.publish("ui:print", role="sys", text="Micrófono en pausa (TTS hablando).")

    # -----------------------------
    # Calibracion y energia
    # -----------------------------
    def _calibrate_microphone(self) -> None:
        try:
            last_err = None
            candidates = []
            if self._device_index is not None:
                candidates.append(self._device_index)
            candidates.append(None)
            try:
                names = sr.Microphone.list_microphone_names() or []
                for i in range(len(names)):
                    if i not in candidates:
                        candidates.append(i)
            except Exception:
                pass

            for device_idx in candidates:
                try:
                    with sr.Microphone(device_index=device_idx) as source:
                        self._rec.adjust_for_ambient_noise(source, duration=self._calibration_duration)
                    self._apply_energy_boost()
                    self._capture_noise_floor()
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

            err_text = str(last_err) if last_err else "error desconocido"
            event_bus.publish("ui:print", role="sys",
                              text=f"No se pudo calibrar el micrófono ({err_text}). Verifica permisos y dispositivo.")
        except Exception as e:
            err_text = str(e)
            event_bus.publish("ui:print", role="sys",
                              text=f"No se pudo calibrar el micrófono ({err_text}). Verifica permisos y dispositivo.")
            logger.warning(f"STT calibration failed: {e}")

    def _quick_calibrate(self, source: sr.Microphone) -> None:
        try:
            self._rec.adjust_for_ambient_noise(source, duration=0.25)
            self._apply_energy_boost()
            self._capture_noise_floor()
        except Exception:
            pass

    def _apply_energy_boost(self) -> None:
        try:
            base = getattr(self._rec, "energy_threshold", None)
            if base is None or base <= 0:
                return
            target = base * self._energy_boost
            if self._min_energy_threshold:
                target = max(self._min_energy_threshold, target)
            if abs(target - base) < 1e-3:
                return
            logger.debug(f"Ajustando energy_threshold de {self._rec.energy_threshold} a {target}")
            self._rec.energy_threshold = target
        except Exception as exc:
            logger.debug(f"No se pudo ajustar energy_threshold: {exc}")

    def _capture_noise_floor(self) -> None:
        """
        Guarda una referencia del ruido ambiente medido por speech_recognition
        para filtrar audio de fondo luego (SNR).
        """
        try:
            thr = getattr(self._rec, "energy_threshold", None)
            if thr is None:
                return
            # energy_threshold es ya un promedio calibrado; lo usamos como piso.
            self._noise_floor_rms = max(float(thr), self._min_rms)
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
        Filtra audio que no supere una SNR mínima respecto al piso de ruido
        medido en calibración. Evita capturar ruido ambiente.
        """
        rms = self._compute_rms(audio)
        floor = self._noise_floor_rms or self._min_rms
        min_allowed = max(self._min_rms, floor * self._snr_min_ratio)
        if rms < min_allowed:
            logger.debug(f"Audio descartado por SNR: rms={rms:.1f}, piso={floor:.1f}, req={min_allowed:.1f}")
            return False
        return True

    def _recalibrate_async(self) -> None:
        # Desactivar recalibración automática para evitar bloqueos si no hay micrófono disponible.
        return

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
        logger.info(f"{prefix}: idx={self._device_index}, nombre='{self._mic_name}'")

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

        while True:
            chunk = self._rec.listen(
                source,
                timeout=timeout,
                phrase_time_limit=chunk_limit,
            )
            segments.append(chunk)
            duration = self._estimate_duration(chunk)
            total_duration += duration

            if not extend:
                break

            near_limit = chunk_limit and duration >= max(chunk_limit - 0.6, chunk_limit * 0.8)
            if not near_limit:
                break

            if max_total and total_duration >= max_total:
                break

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
    # Reconocedores
    # -----------------------------
    def _recognize_with_google(self, audio: sr.AudioData) -> tuple[str, float | None]:
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
            logger.debug(f"Google STT request error: {e}")
            return "", None
        except Exception as e:
            logger.debug(f"Google STT error: {e}")
            return "", None

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
        while True:
            if not self._running.is_set():
                time.sleep(0.05)
                continue
            if not self._listen_enabled.is_set():
                time.sleep(0.02)
                continue

            event_bus.publish("stt:start")
            text, conf = "", None

            try:
                if not self._warned_google:
                    event_bus.publish(
                        "ui:print",
                        role="sys",
                        text="Usando Google Speech para transcribir (requiere internet).",
                    )
                    self._warned_google = True

                last_err = None
                candidates = []
                if self._device_index is not None:
                    candidates.append(self._device_index)
                else:
                    candidates.append(None)
                # Si strict_device_lock es False, probamos otros mics si falla el principal.
                if not self._strict_device_lock:
                    try:
                        names = sr.Microphone.list_microphone_names() or []
                        for i in range(len(names)):
                            if i not in candidates:
                                candidates.append(i)
                    except Exception:
                        pass

                audio = None
                timeout_silence = False
                for device_idx in candidates:
                    try:
                        logger.debug(f"Trying microphone for listen: device_index={device_idx}")
                        with sr.Microphone(device_index=device_idx) as source:
                            self._quick_calibrate(source)
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
                        break
                    except Exception as e:
                        last_err = e
                        logger.debug(f"Microphone listen failed for index {device_idx}: {e}")

                if audio is None:
                    if timeout_silence:
                        raise _SilenceDetected("timeout")
                    raise RuntimeError(f"Unable to open any microphone: {last_err}")

                # Filtro anti-ruido / anti-voz-lejana
                if not self._passes_voice_gate(audio):
                    raise _SilenceDetected("noise_gate")

                text, conf = self._recognize_with_google(audio)

            except _SilenceDetected:
                logger.debug("Silencio detectado: reintentando escucha.")
                self._empty_results += 1
                if self._empty_results >= self._recalibrate_after_empty:
                    self._empty_results = 0
                    self._recalibrate_async()
                time.sleep(0.2)
                continue
            except Exception:
                logger.debug("STT listen error:\n" + traceback.format_exc())
            finally:
                event_bus.publish("stt:end")

            text = (text or "").strip()
            if text:
                self._empty_results = 0
                event_bus.publish("ui:print", role="user", text=text)
                event_bus.publish("stt:text", text=text, confidence=conf)
            else:
                self._empty_results += 1
                if self._empty_results >= self._recalibrate_after_empty:
                    self._empty_results = 0
                    self._recalibrate_async()

            time.sleep(0.05)
