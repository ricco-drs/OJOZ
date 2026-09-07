from __future__ import annotations

"""
Supresión de ruido por modelo de IA (DTLN sobre ONNX Runtime).

El audio capturado por el micrófono se limpia antes de enviarlo a reconocer, lo
que mejora la transcripción en entornos ruidosos (ferias, aulas, exposiciones).

DTLN (Dual-signal Transformation LSTM Network, Westhausen y Meyer, Interspeech
2020) es una red recurrente entrenada para separar voz de ruido. Trabaja en dos
etapas: la primera estima una máscara sobre el espectro de magnitud y la segunda
refina la señal en el dominio del tiempo.

Se eligió ONNX Runtime en lugar de PyTorch porque los pesos ocupan 4 MB, viajan
dentro del repositorio (funciona sin conexión y sin descargas el día de la
demostración) y no arrastra dependencias que compilar.

El módulo está pensado para no ser nunca un punto de fallo: si el modelo no está
disponible o falla al procesar, se devuelve el audio original y el asistente
continúa funcionando con normalidad.
"""

import threading
from pathlib import Path

import numpy as np

from app.config.settings import stt as stt_config, vision
from app.utils.logger import logger

# Frecuencia y tamaños de bloque con los que fue entrenado DTLN.
TARGET_SAMPLE_RATE = 16000
_BLOCK_LEN = 512
_BLOCK_SHIFT = 128
_STATE_SHAPE = (1, 2, 128, 2)

_MODELS_DIR = vision.base_dir / "assets" / "models" / "dtln"


class _Denoiser:
    """Carga perezosa y única de las dos etapas del modelo."""

    def __init__(self) -> None:
        self._stage1 = None
        self._stage2 = None
        self._io1: tuple[str, str] | None = None  # nombres (entrada_audio, entrada_estado)
        self._io2: tuple[str, str] | None = None
        self._lock = threading.RLock()
        self._load_attempted = False
        self._unavailable_reason: str | None = None

    # -----------------------------
    # Carga del modelo
    # -----------------------------
    def _load(self) -> bool:
        """Carga ambas etapas una sola vez. Devuelve True si quedaron listas."""
        with self._lock:
            if self._stage1 is not None:
                return True
            if self._load_attempted:
                return False

            self._load_attempted = True
            try:
                import onnxruntime as ort

                paths = [_MODELS_DIR / "model_1.onnx", _MODELS_DIR / "model_2.onnx"]
                faltantes = [p.name for p in paths if not p.is_file()]
                if faltantes:
                    raise FileNotFoundError(
                        f"faltan los pesos {', '.join(faltantes)} en {_MODELS_DIR}"
                    )

                # Un solo hilo por sesión: la inferencia corre en el hilo de
                # escucha y no debe competir con la cámara ni con la interfaz.
                opts = ort.SessionOptions()
                opts.intra_op_num_threads = 1
                opts.inter_op_num_threads = 1
                opts.log_severity_level = 3  # silencia avisos informativos

                logger.debug("Cargando modelo de supresión de ruido (DTLN)...")
                sessions = [
                    ort.InferenceSession(str(p), opts, providers=["CPUExecutionProvider"])
                    for p in paths
                ]

                self._stage1, self._stage2 = sessions
                # Los nombres de entrada se leen del propio modelo en lugar de
                # fijarlos, por si se sustituyen los pesos por otra versión.
                self._io1 = tuple(i.name for i in self._stage1.get_inputs())[:2]
                self._io2 = tuple(i.name for i in self._stage2.get_inputs())[:2]

                logger.debug("Modelo de supresión de ruido listo.")
                return True

            except ImportError as exc:
                self._unavailable_reason = (
                    f"onnxruntime no está instalado ({exc}). "
                    "Instala con: pip install onnxruntime"
                )
            except FileNotFoundError as exc:
                self._unavailable_reason = str(exc)
            except Exception as exc:
                self._unavailable_reason = f"no se pudo cargar el modelo ({exc})"

            logger.warning(
                f"Supresión de ruido desactivada: {self._unavailable_reason}. "
                "El audio se enviará sin filtrar."
            )
            return False

    def warmup(self) -> bool:
        """
        Precarga el modelo antes de la primera interacción para que el usuario
        no espere durante la conversación.
        """
        if not stt_config.denoise_enabled:
            logger.debug("Supresión de ruido deshabilitada por configuración.")
            return False
        if not self._load():
            return False

        try:
            # Inferencia en vacío: inicializa los buffers internos de ONNX.
            self._process(np.zeros(TARGET_SAMPLE_RATE // 2, dtype=np.float32))
            return True
        except Exception as exc:
            logger.debug(f"Warmup de supresión de ruido falló: {exc}")
            return False

    def is_available(self) -> bool:
        return self._stage1 is not None

    # -----------------------------
    # Inferencia
    # -----------------------------
    def _process(self, samples: np.ndarray) -> np.ndarray:
        """
        Aplica DTLN sobre un arreglo float32 mono en el rango [-1, 1].

        Se recorre la señal en bloques solapados: de cada bloque se toma el
        espectro, la primera etapa devuelve una máscara que atenúa las
        frecuencias dominadas por ruido, y la segunda refina el resultado en el
        tiempo. Los bloques se recomponen por solapamiento y suma.
        """
        name1_audio, name1_state = self._io1
        name2_audio, name2_state = self._io2

        # Relleno final para que las últimas muestras también se procesen.
        padded = np.concatenate([samples, np.zeros(_BLOCK_LEN, dtype=np.float32)])
        output = np.zeros(len(padded), dtype=np.float32)

        in_buffer = np.zeros(_BLOCK_LEN, dtype=np.float32)
        out_buffer = np.zeros(_BLOCK_LEN, dtype=np.float32)
        state1 = np.zeros(_STATE_SHAPE, dtype=np.float32)
        state2 = np.zeros(_STATE_SHAPE, dtype=np.float32)

        num_blocks = (len(padded) - (_BLOCK_LEN - _BLOCK_SHIFT)) // _BLOCK_SHIFT

        for idx in range(num_blocks):
            start = idx * _BLOCK_SHIFT
            in_buffer[:-_BLOCK_SHIFT] = in_buffer[_BLOCK_SHIFT:]
            in_buffer[-_BLOCK_SHIFT:] = padded[start : start + _BLOCK_SHIFT]

            spectrum = np.fft.rfft(in_buffer)
            magnitude = np.abs(spectrum).astype(np.float32).reshape(1, 1, -1)
            phase = np.angle(spectrum)

            # Etapa 1: máscara sobre el espectro de magnitud.
            mask, state1 = self._stage1.run(
                None, {name1_audio: magnitude, name1_state: state1}
            )

            estimated = (magnitude.reshape(-1) * mask.reshape(-1)) * np.exp(1j * phase)
            block = np.fft.irfft(estimated).astype(np.float32).reshape(1, 1, -1)

            # Etapa 2: refinado en el dominio del tiempo.
            enhanced_block, state2 = self._stage2.run(
                None, {name2_audio: block, name2_state: state2}
            )

            out_buffer[:-_BLOCK_SHIFT] = out_buffer[_BLOCK_SHIFT:]
            out_buffer[-_BLOCK_SHIFT:] = 0.0
            out_buffer += enhanced_block.reshape(-1)

            output[start : start + _BLOCK_SHIFT] = out_buffer[:_BLOCK_SHIFT]

        return output[: len(samples)]

    def enhance_pcm16(self, raw: bytes, sample_rate: int = TARGET_SAMPLE_RATE) -> bytes | None:
        """
        Limpia audio PCM de 16 bits mono muestreado a `sample_rate`.

        Devuelve el audio procesado en el mismo formato, o None si el modelo no
        está disponible o el audio no cumple los límites configurados; en ese
        caso quien llama debe seguir usando el audio original.
        """
        if not stt_config.denoise_enabled or not raw:
            return None
        if sample_rate != TARGET_SAMPLE_RATE:
            logger.debug(
                f"Audio a {sample_rate} Hz: se esperaba {TARGET_SAMPLE_RATE} Hz, se omite el filtrado."
            )
            return None
        if not self._load():
            return None

        try:
            samples = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
            duration = len(samples) / float(sample_rate)

            if duration < stt_config.denoise_min_seconds:
                return None
            if duration > stt_config.denoise_max_seconds:
                logger.debug(f"Audio de {duration:.1f}s supera el límite; se omite el filtrado.")
                return None

            enhanced = self._process(samples)

            # El modelo puede devolver la señal con otro nivel. Se iguala al RMS
            # original para no alterar el volumen que recibe el reconocedor.
            enhanced = self._match_level(samples, enhanced)

            enhanced = np.clip(enhanced, -1.0, 1.0)
            return (enhanced * 32767.0).astype(np.int16).tobytes()

        except Exception as exc:
            logger.warning(f"Fallo al filtrar ruido, se usa el audio original: {exc}")
            return None

    @staticmethod
    def _match_level(original: np.ndarray, enhanced: np.ndarray) -> np.ndarray:
        """Iguala el nivel de la señal procesada al de la original."""
        src_rms = float(np.sqrt(np.mean(np.square(original)))) if original.size else 0.0
        out_rms = float(np.sqrt(np.mean(np.square(enhanced)))) if enhanced.size else 0.0

        if src_rms <= 1e-6 or out_rms <= 1e-6:
            return enhanced

        # Se acota la ganancia para no amplificar ruido residual si el modelo
        # devolvió una señal muy atenuada.
        gain = min(max(src_rms / out_rms, 0.1), 10.0)
        return enhanced * gain


# Instancia compartida por la aplicación.
denoiser = _Denoiser()
