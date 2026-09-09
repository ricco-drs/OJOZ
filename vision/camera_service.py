# app/vision/camera_service.py
"""
Camara compartida de la sesion.

Antes, cada funcion de vision (OCR, dinero, vencimiento, escena) abria y
cerraba su propia camara en cada peticion, lo que se notaba como una pausa
al iniciar cada opcion. Aqui se abre una sola vez apenas la sesion queda
autenticada (ver Controller._mark_authenticated) y se mantiene leyendo
fotogramas en un hilo secundario mientras dure la sesion; las funciones de
vision solo piden el ultimo fotograma disponible.

El reconocimiento y enrolamiento facial ocurren ANTES de que exista sesion
(es como se decide quien es la persona), asi que siguen abriendo y cerrando
su propia camara con open_camera/release, sin pasar por aqui.
"""
from __future__ import annotations

import threading
import time

import cv2

from app.config.settings import vision
from app.core.event_bus import event_bus
from app.utils.logger import logger
from app.vision.camera import open_camera, read_frame, release

# Fotogramas con un brillo promedio por debajo de esto se consideran
# "negros" (glitch de la camara/driver, comun con webcams virtuales como
# iVCam tras un rato de uso): se descartan en vez de publicarse como el
# ultimo fotograma valido, para que OCR/dinero/vencimiento/escena nunca
# reciban una foto negra.
_BLACK_FRAME_MEAN_THRESHOLD = 5.0
# Fotogramas negros seguidos antes de reabrir la camara para recuperarla.
_BLACK_FRAMES_BEFORE_REOPEN = 30


class CameraService:
    def __init__(self) -> None:
        self._cap = None
        self._lock = threading.RLock()
        self._latest_frame = None
        self._running = threading.Event()
        self._thread: threading.Thread | None = None
        self._camera_index: int = 0

    def start(self, index: int | None = None) -> None:
        """Enciende la camara compartida si no lo estaba ya. Idempotente."""
        with self._lock:
            if self._running.is_set():
                return
            self._camera_index = index if index is not None else vision.camera_index
            try:
                cap = self._open_configured(self._camera_index)
            except Exception as exc:
                logger.warning(f"No se pudo encender la camara de sesion: {exc}")
                return
            self._cap = cap
            self._running.set()
            self._thread = threading.Thread(target=self._loop, daemon=True)
            self._thread.start()
        event_bus.publish("camera.opened", index=vision.camera_index)
        logger.debug("Camara de sesion encendida (hilo secundario).")

    @staticmethod
    def _open_configured(index: int):
        """
        Abre la camara exactamente igual que el reconocimiento facial
        (open_camera, sin forzar resolucion ni autoenfoque): forzar una
        resolucion como 1920x1080 es lo que hacia que camaras virtuales como
        iVCam entregaran video corrupto (estatico), aunque esa misma camara
        abre limpio cuando se usa con su formato nativo por defecto.
        """
        return open_camera(index)

    def _loop(self) -> None:
        black_streak = 0
        while self._running.is_set():
            try:
                frame = read_frame(self._cap)
            except Exception as exc:
                logger.debug(f"Error leyendo la camara de sesion: {exc}")
                time.sleep(0.1)
                continue

            if frame.mean() < _BLACK_FRAME_MEAN_THRESHOLD:
                black_streak += 1
                if black_streak >= _BLACK_FRAMES_BEFORE_REOPEN:
                    logger.warning(
                        "La camara de sesion entrego fotogramas negros seguidos; reabriendo..."
                    )
                    self._reopen()
                    black_streak = 0
                # No se publica un fotograma negro: se conserva el ultimo
                # fotograma valido en vez de reemplazarlo por uno inservible.
                continue
            black_streak = 0

            with self._lock:
                self._latest_frame = frame

    def _reopen(self) -> None:
        try:
            with self._lock:
                old_cap = self._cap
                self._cap = None
            if old_cap is not None:
                try:
                    old_cap.release()
                except Exception:
                    pass
            new_cap = self._open_configured(self._camera_index)
            with self._lock:
                self._cap = new_cap
        except Exception as exc:
            logger.warning(f"No se pudo reabrir la camara de sesion: {exc}")

    def stop(self) -> None:
        """Apaga la camara compartida. Idempotente."""
        with self._lock:
            if not self._running.is_set():
                return
            self._running.clear()
            thread, self._thread = self._thread, None
        if thread:
            thread.join(timeout=2.0)
        with self._lock:
            if self._cap is not None:
                release(self._cap)
                self._cap = None
            self._latest_frame = None
        event_bus.publish("camera.closed", index=vision.camera_index)
        logger.debug("Camara de sesion apagada.")

    def is_running(self) -> bool:
        return self._running.is_set()

    def get_frame(self):
        """Copia del ultimo fotograma disponible, o None si aun no hay ninguno."""
        with self._lock:
            return None if self._latest_frame is None else self._latest_frame.copy()


camera_service = CameraService()


def frames_for(seconds: float, configure_capture=None):
    """
    Generador de fotogramas durante `seconds`.

    Si la camara de sesion esta encendida, lee de ahi sin abrir ni cerrar
    nada. Si no (herramientas de diagnostico, o esta funcion se llama fuera
    de una sesion activa), abre y cierra su propia camara temporal, igual
    que se hacia antes en cada funcion de vision por separado.
    """
    if camera_service.is_running():
        t0 = time.time()
        while time.time() - t0 < seconds:
            frame = camera_service.get_frame()
            if frame is None:
                time.sleep(0.02)
                continue
            yield frame
        return

    cap = cv2.VideoCapture(getattr(vision, "camera_index", 0), cv2.CAP_DSHOW)
    if not cap.isOpened():
        return
    if configure_capture:
        configure_capture(cap)
    try:
        t0 = time.time()
        while time.time() - t0 < seconds:
            ret, current = cap.read()
            if not ret:
                break
            from app.vision.camera import flip_frame, rotate_frame

            yield flip_frame(rotate_frame(current))
    finally:
        cap.release()
        cv2.destroyAllWindows()
