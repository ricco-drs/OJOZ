from __future__ import annotations
import audioop
import re
import threading
import time
import traceback
import speech_recognition as sr

from app.audio.denoise import TARGET_SAMPLE_RATE, denoiser
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
        self._max_energy_threshold = float(getattr(stt_config, "max_energy_threshold", 0.0) or 0.0)
        self._max_required_rms = float(getattr(stt_config, "max_required_rms", 3500.0) or 3500.0)
        self._noise_floor_rms: float | None = None
        self._base_energy_threshold: float | None = None
        self._apply_energy_boost()

        # Recalibración periódica del ruido ambiente.
        self._recalibrate_cooldown_s = float(getattr(stt_config, "recalibrate_cooldown_s", 20.0) or 0.0)
        self._recalibrate_every_s = float(getattr(stt_config, "recalibrate_every_s", 180.0) or 0.0)
        self._last_recalibration = 0.0

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

        # El modelo de supresión de ruido se carga en paralelo para no demorar
        # el saludo inicial. Si aún no está listo cuando llegue la primera
        # frase, esa frase se transcribe sin filtrar.
        if getattr(stt_config, "denoise_enabled", False):
            threading.Thread(target=denoiser.warmup, daemon=True).start()

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
                        measured = self._measure_ambient_rms(source, duration=0.6)
                    self._apply_energy_boost(measured)
                    self._update_noise_floor(measured)
                    self._last_recalibration = time.monotonic()
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
            measured = self._measure_ambient_rms(source, duration=0.25)
            self._apply_energy_boost(measured)
            self._update_noise_floor(measured)
        except Exception:
            pass

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
        rms = self._compute_rms(audio)
        floor = self._noise_floor_rms or self._min_rms
        min_allowed = max(self._min_rms, floor * self._snr_min_ratio)
        # Aun con mucho ruido, el listón debe seguir siendo alcanzable por una
        # voz cercana; si no, el asistente dejaría de responder por completo.
        min_allowed = min(min_allowed, self._max_required_rms)
        if rms < min_allowed:
            logger.debug(f"Audio descartado por SNR: rms={rms:.1f}, piso={floor:.1f}, req={min_allowed:.1f}")
            return False
        return True

    def _recalibrate_noise_floor(self, force: bool = False) -> bool:
        """
        Vuelve a medir el ruido ambiente sobre el micrófono ya seleccionado.

        Se invoca desde el bucle de escucha con el micrófono cerrado, por lo que
        abrirlo aquí es seguro. Solo se prueba el dispositivo en uso (no se
        recorre la lista completa) para que un dispositivo ausente no bloquee la
        escucha, y cualquier fallo se ignora: es preferible seguir con el umbral
        anterior antes que interrumpir el reconocimiento.
        """
        now = time.monotonic()
        with self._recalibration_lock:
            if not force and (now - self._last_recalibration) < self._recalibrate_cooldown_s:
                return False
            self._last_recalibration = now

        try:
            with sr.Microphone(device_index=self._device_index) as source:
                self._rec.adjust_for_ambient_noise(source, duration=self._calibration_duration)
                measured = self._measure_ambient_rms(source, duration=0.5)
            self._apply_energy_boost(measured)
            self._update_noise_floor(measured)
            piso = f"{self._noise_floor_rms:.1f}" if self._noise_floor_rms else "sin medir"
            umbral = getattr(self._rec, "energy_threshold", 0.0) or 0.0
            logger.debug(
                f"Ruido ambiente recalibrado: energy_threshold={umbral:.1f}, piso={piso}"
            )
            return True
        except Exception as exc:
            logger.debug(f"No se pudo recalibrar el ruido ambiente: {exc}")
            return False

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
    # Supresión de ruido
    # -----------------------------
    def _denoise(self, audio: sr.AudioData) -> sr.AudioData:
        """
        Limpia el audio con el modelo de IA antes de transcribir.

        Se convierte a 16 kHz mono, que es el formato con el que se entrenó el
        modelo y el que prefiere el reconocedor. Ante cualquier problema se
        devuelve el audio recibido sin modificar.
        """
        try:
            raw16 = audio.get_raw_data(convert_rate=TARGET_SAMPLE_RATE, convert_width=2)
            cleaned = denoiser.enhance_pcm16(raw16, TARGET_SAMPLE_RATE)
            if cleaned is None:
                return audio
            return sr.AudioData(cleaned, TARGET_SAMPLE_RATE, 2)
        except Exception as exc:
            logger.debug(f"No se pudo aplicar supresión de ruido: {exc}")
            return audio

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

            # Recalibración periódica: el ruido de un stand cambia durante la
            # jornada y un piso medido al arrancar deja de ser representativo.
            if (
                self._recalibrate_every_s > 0
                and (time.monotonic() - self._last_recalibration) >= self._recalibrate_every_s
            ):
                self._recalibrate_noise_floor(force=True)

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

                # Filtro anti-ruido / anti-voz-lejana. Se evalúa sobre el audio
                # original, que es con el que se midió el piso de ruido.
                if not self._passes_voice_gate(audio):
                    raise _SilenceDetected("noise_gate")

                # Solo se limpia lo que ya se considera voz: ahorra CPU y evita
                # procesar ruido que igualmente se iba a descartar.
                audio = self._denoise(audio)

                text, conf = self._recognize_with_google(audio)

            except _SilenceDetected:
                logger.debug("Silencio detectado: reintentando escucha.")
                self._empty_results += 1
                if self._empty_results >= self._recalibrate_after_empty:
                    self._empty_results = 0
                    self._recalibrate_noise_floor()
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
                    self._recalibrate_noise_floor()

            time.sleep(0.05)
