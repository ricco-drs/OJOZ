from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Optional


# ===========================
# AUDIO
# ===========================
@dataclass
class TTSConfig:
    # Nombre de voz (opcional). Ej.: "Microsoft Sabina Desktop" en Windows
    voice: Optional[str] = None
    rate: int = 180
    volume: float = 1.0
    edge_voice: str = "es-MX-DaliaNeural"


@dataclass
class STTConfig:
    # Selección de micrófono: usa None para el predeterminado, o un índice (int).
    # Para ver los índices disponibles, imprime sr.Microphone.list_microphone_names()
    device_index: Optional[int] = None

    # Idioma (para Google STT). Ejemplo: español de Perú.
    language: str = "es-PE"

    # Modelo offline Vosk (ruta opcional). Si se deja None, se usa la
    # variable de entorno VOSK_MODEL. Vosk está deshabilitado; se usa Google STT online.
    vosk_model_path: Optional[str] = None

    # Normalización personalizada: cuando el usuario dice "rico" como nombre propio
    # en frases tipo "me llamo rico" o "soy rico", se escribirá "ricco".
    normalize_ricco: bool = True

    # Parámetros de escucha.
    # Tiempo máximo de frase y tolerancia a pausas amplios para que el usuario pueda
    # decir "Hola OJOZ, es mi primera vez aquí" sin que se corte tras "Hola OJOZ".
    phrase_time_limit: float = 16.0             # permite frases más largas
    max_phrase_seconds: Optional[float] = 30.0  # extiende automáticamente en frases largas
    listen_timeout: Optional[float] = 4.5       # espera de inicio de habla antes de reintentar
    extend_phrase: bool = True                  # activar concatenación de fragmentos largos
    pause_threshold: float = 1.0                # espera antes de cortar por silencio
    calibration_duration: float = 2.0           # segundos para medir ruido ambiente
    energy_boost: float = 0.25                  # multiplica el umbral calculado para voces suaves
    min_energy_threshold: float = 3.0           # no bajar de este valor para evitar ruido extremo
    recalibrate_after_empty: int = 2            # frases vacías antes de recalibrar
    dynamic_energy_threshold: bool = True
    energy_threshold: Optional[int] = None      # usa valor fijo si lo seteas

    # Filtro / robustez frente a ruido
    strict_device_lock: bool = False  # permite probar otros micrófonos si falla el configurado
    snr_min_ratio: float = 3.5        # mínimo RMS vs piso de ruido para aceptar audio
    min_rms: float = 20.0             # RMS absoluto mínimo


@dataclass
class AppConfig:
    enable_barge_in: bool = False             # (reservado) barge-in
    rearm_stt_delay_ms: int = 400             # espera tras TTS antes de re-escuchar
    rearm_stt_failsafe_ms: int = 8000         # failsafe: fuerza reactivación si falta tts:end
    startup_sound_path: Optional[str] = None  # ruta al audio de arranque (mp3/wav)


# Instancias de configuración global
config = AppConfig()
tts = TTSConfig()
stt = STTConfig()


# ===========================
# VISIÓN
# ===========================
_HAAR_FILENAME = "haarcascade_frontalface_default.xml"


def _haar_candidates() -> list[Path]:
    """Posibles ubicaciones del clasificador Haar, en orden de preferencia."""
    candidates: list[Path] = []

    # 1) Ruta indicada explícitamente por el usuario.
    env_path = os.environ.get("HAARCASCADE_PATH")
    if env_path:
        candidates.append(Path(env_path))

    # 2) Copia local dentro del proyecto (assets/haarcascades/).
    candidates.append(Path(__file__).resolve().parents[1] / "assets" / "haarcascades" / _HAAR_FILENAME)

    # 3) La que trae OpenCV. Ojo: OpenCV 5 ya no distribuye estos XML,
    #    por lo que la carpeta puede existir pero estar vacía.
    try:
        import cv2 as _cv2

        candidates.append(Path(_cv2.data.haarcascades) / _HAAR_FILENAME)
    except Exception:
        pass

    return candidates


def _haar_default_path() -> str:
    """
    Devuelve la ruta del clasificador Haar frontal.

    Si no se encuentra en ninguna ubicación conocida se devuelve el nombre
    del archivo a secas; load_face_detector() se encargará de avisar con un
    error claro en vez de fallar en silencio.
    """
    for candidate in _haar_candidates():
        if candidate.is_file():
            return str(candidate)
    return _HAAR_FILENAME


def load_face_detector():
    """
    Crea el CascadeClassifier de rostros validando que se haya cargado.

    cv2.CascadeClassifier() no lanza excepción si el XML no existe: devuelve un
    clasificador vacío que nunca detecta nada, lo que hace que el enrolamiento y
    la autenticación fallen sin ningún mensaje. Aquí se comprueba de forma
    explícita y se explica cómo resolverlo.
    """
    import cv2

    detector = cv2.CascadeClassifier(vision.face_detector)
    if detector.empty():
        buscadas = "\n  - ".join(str(c) for c in _haar_candidates())
        raise RuntimeError(
            "No se pudo cargar el clasificador de rostros "
            f"'{_HAAR_FILENAME}'.\n"
            f"Rutas consultadas:\n  - {buscadas}\n"
            "OpenCV 5 ya no incluye estos archivos. Soluciones:\n"
            "  a) Copiar el XML en assets/haarcascades/ (se descarga del repo "
            "opencv/data/haarcascades).\n"
            "  b) Indicar su ruta con la variable de entorno HAARCASCADE_PATH.\n"
            "  c) Instalar OpenCV 4.x, que sí lo distribuye."
        )
    return detector


@dataclass
class VisionConfig:
    # Rutas base
    base_dir: Path
    data_dir: Path
    fotos_dir: Path
    modelos_dir: Path
    model_file: Path
    ocr_dir: Path       # carpeta para capturas OCR
    currency_dir: Path  # carpeta de referencias de billetes

    # Dispositivo de cámara y parámetros
    camera_index: int = 0
    capture_count: int = 300
    face_size: tuple[int, int] = (120, 120)

    # Detección y reconocimiento
    face_detector: str = _haar_default_path()
    lbph_threshold: int = 70

    # UI / depuración
    show_preview: bool = True  # ventanas (imshow) durante captura/autenticación


# Construcción de rutas a partir de la ubicación del proyecto (app/)
_BASE_DIR = Path(__file__).resolve().parents[1]
_DATA_DIR = _BASE_DIR / "data"
_FOTOS_DIR = _DATA_DIR / "fotos"
_MODELOS_DIR = _DATA_DIR / "modelos"
_OCR_DIR = _DATA_DIR / "ocr"
_CURRENCY_DIR = _DATA_DIR / "currency_refs"
_MODEL_FILE = _MODELOS_DIR / "modeloLBPHFace.xml"
_STARTUP_SOUND_DIR = _BASE_DIR / "assets" / "sounds"
_STARTUP_SOUND_WAV = _STARTUP_SOUND_DIR / "dog_bark.wav"
_STARTUP_SOUND_MP3 = _STARTUP_SOUND_DIR / "dog_bark.mp3"

# Crear las carpetas si no existen (idempotente)
os.makedirs(_FOTOS_DIR, exist_ok=True)
os.makedirs(_MODELOS_DIR, exist_ok=True)
os.makedirs(_OCR_DIR, exist_ok=True)
os.makedirs(_CURRENCY_DIR, exist_ok=True)
os.makedirs(_STARTUP_SOUND_DIR, exist_ok=True)

if config.startup_sound_path is None:
    if _STARTUP_SOUND_MP3.exists():
        config.startup_sound_path = str(_STARTUP_SOUND_MP3)
    elif _STARTUP_SOUND_WAV.exists():
        config.startup_sound_path = str(_STARTUP_SOUND_WAV)
    else:
        # Ruta por defecto si aún no hay archivo; se usará cuando exista.
        config.startup_sound_path = str(_STARTUP_SOUND_WAV)

# ===========================
# OCR / TESSERACT
# ===========================
# Ruta tipica de instalación en Windows. Se puede sobreescribir con las
# variables de entorno TESSERACT_CMD y TESSDATA_PREFIX.
_TESSERACT_DEFAULT = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
_TESSDATA_DEFAULT = r"C:\Program Files\Tesseract-OCR\tessdata"

TESSERACT_CMD = os.environ.get("TESSERACT_CMD", _TESSERACT_DEFAULT)
TESSDATA_PREFIX = os.environ.get("TESSDATA_PREFIX", _TESSDATA_DEFAULT)


def configure_tesseract() -> bool:
    """
    Apunta pytesseract al binario configurado. Devuelve True si la ruta existe.

    Si no existe se deja el valor por defecto de pytesseract, que buscará
    "tesseract" en el PATH. Llamar desde cada módulo de visión que use OCR.
    """
    try:
        import pytesseract

        if Path(TESSERACT_CMD).exists():
            pytesseract.pytesseract.tesseract_cmd = TESSERACT_CMD
            os.environ["TESSDATA_PREFIX"] = TESSDATA_PREFIX
            return True
        return False
    except Exception:
        return False


# Instancia global de visión
vision = VisionConfig(
    base_dir=_BASE_DIR,
    data_dir=_DATA_DIR,
    fotos_dir=_FOTOS_DIR,
    modelos_dir=_MODELOS_DIR,
    ocr_dir=_OCR_DIR,
    currency_dir=_CURRENCY_DIR,
    model_file=_MODEL_FILE,
    camera_index=0,
    capture_count=300,
    face_size=(120, 120),
    face_detector=_haar_default_path(),
    lbph_threshold=70,
    show_preview=True,
)
