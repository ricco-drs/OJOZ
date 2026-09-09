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
    language: str = "es"
    rate: int = 180
    volume: float = 1.0
    edge_voice: str = "es-MX-DaliaNeural"
    # eleven_flash_v2_5 prioriza latencia baja a costa de naturalidad: acorta
    # las pausas de puntuacion y suena apurado. eleven_multilingual_v2 tarda
    # un poco mas en generarse pero respeta comas y puntos, con una entonacion
    # mucho mas fluida.
    elevenlabs_model: str = "eleven_multilingual_v2"
    elevenlabs_timeout_seconds: float = 15.0
    # Ajustes de la voz: velocidad por debajo de 1.0 para un ritmo mas calmado,
    # y stability/similarity moderados para que suene expresiva sin volverse
    # inestable entre frases.
    elevenlabs_stability: float = 0.5
    elevenlabs_similarity_boost: float = 0.75
    elevenlabs_style: float = 0.35
    elevenlabs_speed: float = 0.92


def _env_float(name: str, default: float) -> float:
    """Lee un float de una variable de entorno; si no es válida, usa el default."""
    raw = os.environ.get(name)
    if raw is None or not raw.strip():
        return default
    try:
        return float(raw)
    except ValueError:
        return default


def _env_bool(name: str, default: bool) -> bool:
    """Lee un booleano de una variable de entorno ("1", "true", "si", "on")."""
    raw = os.environ.get(name)
    if raw is None or not raw.strip():
        return default
    return raw.strip().lower() in ("1", "true", "t", "yes", "y", "si", "sí", "on")


def _env_int(name: str, default: int) -> int:
    """Lee un entero de una variable de entorno; si no es válido, usa el default."""
    raw = os.environ.get(name)
    if raw is None or not raw.strip():
        return default
    try:
        return int(raw)
    except ValueError:
        return default


@dataclass
class STTConfig:
    # Selección de micrófono: usa None para el predeterminado, o un índice (int).
    # Para ver los índices disponibles, imprime sr.Microphone.list_microphone_names()
    device_index: Optional[int] = None
    device_name_hint: Optional[str] = None

    # Idioma (para Google STT). Ejemplo: español de Perú.
    language: str = "es-PE"

    # Modelo offline Vosk (ruta opcional). Si se deja None, se usa la
    # variable de entorno VOSK_MODEL. Vosk está deshabilitado; se usa Google STT online.
    vosk_model_path: Optional[str] = None

    # Normalización personalizada: cuando el usuario dice "rico" como nombre propio
    # en frases tipo "me llamo rico" o "soy rico", se escribirá "ricco".
    normalize_ricco: bool = True

    # Empujar para hablar: el microfono empieza siempre desactivado y solo se
    # enciende mientras se mantiene presionada la tecla de espacio.
    push_to_talk: bool = True

    # Parámetros de escucha.
    # Tiempo máximo de frase y tolerancia a pausas amplios para que el usuario pueda
    # decir "Hola OJOZ, es mi primera vez aquí" sin que se corte tras "Hola OJOZ".
    phrase_time_limit: float = 16.0             # permite frases más largas
    max_phrase_seconds: Optional[float] = 30.0  # extiende automáticamente en frases largas
    listen_timeout: Optional[float] = 4.5       # espera de inicio de habla antes de reintentar
    extend_phrase: bool = True                  # activar concatenación de fragmentos largos
    pause_threshold: float = 1.2                # tolera pausas dentro de una frase
    continuation_timeout: float = 0.7           # une la continuacion antes de transcribir
    phrase_threshold: float = 0.15              # admite respuestas breves como "si"
    non_speaking_duration: float = 0.5           # conserva el inicio y final de las palabras
    calibration_duration: float = 2.0           # segundos para medir ruido ambiente

    # Los umbrales no son valores fijos: se recalculan a partir del ruido que se
    # mide en el propio micrófono, de modo que el asistente se adapta solo tanto
    # a una habitación en silencio como a un pabellón lleno de gente.
    #   umbral = ruido_medido x 1.5 x energy_boost   (acotado por los límites)
    # energy_boost = 1.0 mantiene el margen estándar; por debajo de 1 el
    # asistente se vuelve más sensible y por encima, más exigente.
    energy_boost: float = 1.0
    min_energy_threshold: float = 50.0          # suelo: evita disparar con ruido electrónico
    max_energy_threshold: float = 4000.0        # techo: evita quedar sordo ante un ruido puntual
    recalibrate_after_empty: int = 3            # frases vacías antes de recalibrar
    dynamic_energy_threshold: bool = True       # adapta el umbral mientras espera voz
    energy_threshold: Optional[int] = None      # usa valor fijo si lo seteas

    # Filtro / robustez frente a ruido. Al comparar la voz contra el ruido
    # medido, el mismo valor sirve en silencio y en ambiente ruidoso: lo que
    # cambia es la referencia, no el criterio.
    strict_device_lock: bool = False  # permite probar otros micrófonos si falla el configurado
    snr_min_ratio: float = 1.5        # margen para admitir respuestas cortas y voz suave
    min_rms: float = 40.0             # RMS absoluto mínimo
    max_required_rms: float = 3500.0  # tope: por muy alto que sea el ruido, sigue siendo alcanzable

    # Supresión de ruido por modelo de IA (DTLN sobre ONNX Runtime).
    # El audio capturado se limpia antes de enviarlo a reconocer. Si el modelo
    # no está disponible se continúa con el audio original.
    # Los pesos viven en assets/models/dtln/ y se versionan con el proyecto.
    denoise_enabled: bool = True
    denoise_min_seconds: float = 0.3   # audios más cortos no se procesan
    denoise_max_seconds: float = 32.0  # cubre tambien las frases extendidas

    # Transcripción con ElevenLabs Scribe (misma cuenta/clave que la voz de
    # salida): mas robusta ante ruido de fondo que el endpoint gratuito de
    # Google. Se usa como motor principal si hay ELEVENLABS_API_KEY; si falla
    # por credenciales o saldo se desactiva por el resto de la sesion y se
    # sigue con Google, sin interrumpir el servicio.
    elevenlabs_stt_model: str = "scribe_v1"
    elevenlabs_timeout_seconds: float = 20.0


@dataclass
class LLMConfig:
    """
    Conversación gestionada por un modelo de lenguaje (opción B).

    Se inicia al terminar el saludo si hay una credencial configurada. El
    enrutador queda como respaldo si falta la credencial o falla la API.

    Requiere una credencial en la variable de entorno ANTHROPIC_API_KEY.
    """

    enabled: bool = True
    model: str = "claude-sonnet-5"
    max_tokens: int = 1024          # las respuestas son habladas, no hacen falta más
    effort: str = "low"             # prioriza la fluidez de la conversación
    timeout_seconds: float = 20.0   # antes que hacer esperar, se usa el respaldo
    max_retries: int = 1            # reintentar mucho añade silencios incómodos
    max_tool_iterations: int = 5    # tope de acciones encadenadas en un turno
    max_history_messages: int = 24  # memoria de la conversación

    # Si la API falla varias veces seguidas se hace una pausa y se atiende con el
    # enrutador, en lugar de reintentar en cada turno y demorar cada respuesta.
    failures_before_pause: int = 2
    pause_seconds: float = 120.0


@dataclass
class AppConfig:
    enable_barge_in: bool = False             # (reservado) barge-in
    rearm_stt_delay_ms: int = 400             # espera tras TTS antes de re-escuchar
    rearm_stt_failsafe_ms: int = 8000         # failsafe: fuerza reactivación si falta tts:end


# Instancias de configuración global
config = AppConfig()
tts = TTSConfig()
stt = STTConfig()
llm = LLMConfig()


def _apply_env_overrides() -> None:
    """
    Permite afinar la escucha sin editar código ni reinstalar nada.

    No hacen falta en condiciones normales —los umbrales se adaptan solos al
    ruido del lugar—, pero dejan margen para corregir un caso concreto sobre la
    marcha, por ejemplo un micrófono con poca ganancia.
    """
    stt.energy_boost = _env_float("OJOZ_ENERGY_BOOST", stt.energy_boost)
    stt.min_energy_threshold = _env_float("OJOZ_MIN_ENERGY", stt.min_energy_threshold)
    stt.snr_min_ratio = _env_float("OJOZ_SNR_RATIO", stt.snr_min_ratio)
    stt.min_rms = _env_float("OJOZ_MIN_RMS", stt.min_rms)
    stt.denoise_enabled = _env_bool("OJOZ_DENOISE", stt.denoise_enabled)
    stt.push_to_talk = _env_bool("OJOZ_PUSH_TO_TALK", stt.push_to_talk)
    mic_index = os.environ.get("OJOZ_MIC_INDEX", "").strip()
    if mic_index:
        try:
            stt.device_index = int(mic_index)
        except ValueError:
            stt.device_index = None
    stt.device_name_hint = os.environ.get("OJOZ_MIC_NAME", "").strip() or stt.device_name_hint

    # OJOZ_LLM=0 permite desactivar la conversacion por modelo.
    llm.enabled = _env_bool("OJOZ_LLM", llm.enabled)
    llm.model = os.environ.get("OJOZ_LLM_MODEL", llm.model).strip() or llm.model


_apply_env_overrides()


@dataclass
class VisionConfig:
    # Rutas base
    base_dir: Path
    data_dir: Path
    fotos_dir: Path
    modelos_dir: Path
    model_file: Path
    insightface_root: Path
    ocr_dir: Path       # carpeta para capturas OCR
    currency_dir: Path  # carpeta de referencias de billetes

    # Dispositivo de camara y enrolamiento
    camera_index: int = 0
    # Si se define, se busca una camara cuyo nombre contenga este texto y se
    # usa su indice en vez de camera_index (mas confiable que un numero fijo,
    # que puede cambiar segun que otras camaras/apps virtuales esten activas).
    camera_name_hint: Optional[str] = None
    # Gira el video si la camara entrega horizontal aunque se sostenga en
    # vertical (comun con el celular como webcam via apps tipo iVCam).
    # Valores: 0, 90, 180 o 270.
    camera_rotate_degrees: int = 0
    # Corrige el efecto espejo si la camara entrega el video reflejado
    # horizontalmente (comun en apps tipo iVCam en modo "selfie"): el texto
    # de un documento se veria al reves. Activar con OJOZ_CAMERA_FLIP=1.
    camera_flip_horizontal: bool = False
    capture_count: int = 20
    min_enrollment_photos: int = 10
    capture_frame_width: int = 960
    capture_interval_seconds: float = 0.25
    capture_timeout_seconds: float = 60.0

    # SCRFD detecta y alinea; ArcFace genera embeddings de 512 dimensiones.
    face_model_name: str = "buffalo_l"
    face_provider: str = "auto"
    face_detection_size: tuple[int, int] = (640, 640)
    face_detection_threshold: float = 0.65
    face_min_size: int = 120
    face_blur_threshold: float = 50.0
    face_similarity_threshold: float = 0.50
    face_ambiguity_margin: float = 0.05
    face_required_confirmations: int = 3

    # UI / depuración
    show_preview: bool = True  # ventanas (imshow) durante captura/autenticación


# Construcción de rutas a partir de la ubicación del proyecto (app/)
_BASE_DIR = Path(__file__).resolve().parents[1]
_DATA_DIR = _BASE_DIR / "data"
_FOTOS_DIR = _DATA_DIR / "fotos"
_MODELOS_DIR = _DATA_DIR / "modelos"
_OCR_DIR = _DATA_DIR / "ocr"
_CURRENCY_DIR = _DATA_DIR / "currency_refs"
_MODEL_FILE = _MODELOS_DIR / "arcface_gallery.npz"
_INSIGHTFACE_ROOT = _BASE_DIR / "assets" / "models" / "insightface"

# Crear las carpetas si no existen (idempotente)
os.makedirs(_FOTOS_DIR, exist_ok=True)
os.makedirs(_MODELOS_DIR, exist_ok=True)
os.makedirs(_OCR_DIR, exist_ok=True)
os.makedirs(_CURRENCY_DIR, exist_ok=True)
os.makedirs(_INSIGHTFACE_ROOT, exist_ok=True)

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
    insightface_root=_INSIGHTFACE_ROOT,
    camera_index=_env_int("OJOZ_CAMERA_INDEX", 0),
    # Por defecto usa el celular como camara via iVCam: se puede acercar a un
    # documento o billete, algo que una webcam fija del laptop (apuntando a
    # la cara) no permite bien. Vacio ("") vuelve a depender solo de
    # OJOZ_CAMERA_INDEX; cambiar a "UVC WebCam" para forzar la webcam fisica.
    camera_name_hint=os.environ.get("OJOZ_CAMERA_NAME", "iVCam").strip() or None,
    camera_rotate_degrees=_env_int("OJOZ_CAMERA_ROTATE", 0),
    camera_flip_horizontal=_env_bool("OJOZ_CAMERA_FLIP", False),
    capture_count=20,
    min_enrollment_photos=10,
    capture_frame_width=960,
    face_model_name=os.environ.get("OJOZ_FACE_MODEL", "buffalo_l"),
    face_provider=os.environ.get("OJOZ_FACE_PROVIDER", "auto"),
    face_similarity_threshold=_env_float("OJOZ_FACE_THRESHOLD", 0.50),
    show_preview=_env_bool("OJOZ_SHOW_PREVIEW", True),
)

if vision.camera_name_hint:
    try:
        from app.vision.camera import find_camera_index_by_name

        _resolved = find_camera_index_by_name(vision.camera_name_hint)
        if _resolved is not None:
            vision.camera_index = _resolved
    except Exception:
        pass  # Se sigue usando camera_index tal cual si la busqueda falla.
