from __future__ import annotations
from typing import Callable, Dict, List, Optional, TypedDict
from threading import RLock, local

from app.utils.logger import logger

# El logger reenvia sus mensajes al propio bus ("ui:print"). Si un handler de ese
# evento fallara, loguear el error volveria a publicar y entrariamos en recursion.
# Este flag por hilo corta la cadena: el segundo nivel de fallo se ignora.
_reporting = local()

class EventBus:
    """
    Bus de eventos simple (pub/sub) con bloqueo para uso en hilos.
    - subscribe(event, handler): registra un handler para un evento.
    - unsubscribe(event, handler): elimina el handler.
    - publish(event, **data): emite el evento con datos (kwargs).
    """
    def __init__(self):
        self._listeners: Dict[str, List[Callable]] = {}
        self._lock = RLock()

    def subscribe(self, event: str, handler: Callable):
        with self._lock:
            self._listeners.setdefault(event, []).append(handler)

    def unsubscribe(self, event: str, handler: Callable):
        with self._lock:
            if event in self._listeners and handler in self._listeners[event]:
                self._listeners[event].remove(handler)

    def publish(self, event: str, **data):
        # Copiamos la lista fuera del lock para evitar deadlocks si el handler publica a su vez
        with self._lock:
            listeners = list(self._listeners.get(event, []))
        for h in listeners:
            try:
                h(**data)
            except Exception:
                # Un handler con error no debe bloquear al resto, pero tampoco
                # desaparecer en silencio.
                if getattr(_reporting, "active", False):
                    continue
                _reporting.active = True
                try:
                    name = getattr(h, "__qualname__", repr(h))
                    logger.exception("Handler %r fallo al procesar el evento %r", name, event)
                finally:
                    _reporting.active = False

    # =========================
    # EXTENSIONES (NUEVO)
    # =========================

    def once(self, event: str, handler: Callable):
        """
        Registra un handler que se ejecuta solo una vez.
        Devuelve el wrapper interno por si necesitas desuscribir manualmente.
        """
        def _wrapper(**data):
            try:
                handler(**data)
            finally:
                self.unsubscribe(event, _wrapper)
        self.subscribe(event, _wrapper)
        return _wrapper

    def on(self, event: str):
        """
        Decorador para suscripción concisa:
        @event_bus.on(Events.USER_ENROLL_REQUEST)
        def handler(name: str): ...
        """
        def _decorator(func: Callable):
            self.subscribe(event, func)
            return func
        return _decorator

    def listeners(self, event: Optional[str] = None) -> Dict[str, List[Callable]] | List[Callable]:
        """
        Inspección: devuelve copia de los listeners.
        - Si event es None, devuelve dict completo {event: [handlers]}.
        - Si event es str, devuelve la lista de handlers para ese evento.
        """
        with self._lock:
            if event is None:
                return {k: list(v) for k, v in self._listeners.items()}
            return list(self._listeners.get(event, []))

    def clear(self, event: Optional[str] = None):
        """
        Borra suscripciones:
        - Si event es None, borra todas.
        - Si event es str, borra solo las de ese evento.
        """
        with self._lock:
            if event is None:
                self._listeners.clear()
            else:
                self._listeners.pop(event, None)


# =========================
# EVENTOS ESTÁNDAR (NUEVO)
# =========================
class Events:
    # Flujo de enrolamiento (opción 1)
    USER_ENROLL_REQUEST = "user.enroll.request"      # {name}
    CAPTURE_PROGRESS    = "capture.progress"         # {done, total}
    USER_ENROLL_DONE    = "user.enroll.done"         # {name, saved, model_path}

    # Flujo de autenticación (opción 2)
    USER_AUTH_REQUEST   = "user.auth.request"        # {}
    USER_AUTH_RESULT    = "user.auth.result"         # {ok, name, confidence}

    # Estado de cámara (útil para UI/logs)
    CAMERA_OPENED       = "camera.opened"            # {index}
    CAMERA_CLOSED       = "camera.closed"            # {index}

    # Mensajes TTS (para que UI muestre lo que se dijo por voz)
    TTS_SAID            = "tts.said"                 # {text}


# =========================
# TIPOS DE PAYLOAD (NUEVO)
# =========================
class EnrollRequestPayload(TypedDict):
    name: str

class CaptureProgressPayload(TypedDict):
    done: int
    total: int

class EnrollDonePayload(TypedDict):
    name: str
    saved: int
    model_path: str

class AuthRequestPayload(TypedDict, total=False):
    # reservado por si luego agregamos opciones de timeout o sensibilidad
    pass

class AuthResultPayload(TypedDict):
    ok: bool
    name: Optional[str]
    confidence: Optional[float]

class CameraStatePayload(TypedDict):
    index: int

class TTSSaidPayload(TypedDict):
    text: str


# Instancia global del bus de eventos
event_bus = EventBus()

# ==============
# HELPERS (NUEVO)
# ==============
# Estos helpers opcionales estandarizan la publicación con tipado más claro.

def publish_enroll_request(name: str):
    event_bus.publish(Events.USER_ENROLL_REQUEST, name=name)

def publish_capture_progress(done: int, total: int):
    event_bus.publish(Events.CAPTURE_PROGRESS, done=done, total=total)

def publish_enroll_done(name: str, saved: int, model_path: str):
    event_bus.publish(Events.USER_ENROLL_DONE, name=name, saved=saved, model_path=model_path)

def publish_auth_request():
    event_bus.publish(Events.USER_AUTH_REQUEST)

def publish_auth_result(ok: bool, name: Optional[str], confidence: Optional[float]):
    event_bus.publish(Events.USER_AUTH_RESULT, ok=ok, name=name, confidence=confidence)

def publish_camera_opened(index: int):
    event_bus.publish(Events.CAMERA_OPENED, index=index)

def publish_camera_closed(index: int):
    event_bus.publish(Events.CAMERA_CLOSED, index=index)

def publish_tts_said(text: str):
    event_bus.publish(Events.TTS_SAID, text=text)
