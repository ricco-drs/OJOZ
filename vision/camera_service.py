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


class CameraService:
    def __init__(self) -> None:
        self._cap = None
        self._lock = threading.RLock()
        self._latest_frame = None
        self._running = threading.Event()
        self._thread: threading.Thread | None = None

    def start(self, index: int | None = None) -> None:
        """Enciende la camara compartida si no lo estaba ya. Idempotente."""
        with self._lock:
            if self._running.is_set():
                return
            try:
                cap = open_camera(index if index is not None else vision.camera_index)
                # Resolucion mas alta: la usa OCR y no perjudica a dinero,
                # vencimiento ni escena (Claude Vision trabaja bien con ella).
                cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
                cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
                cap.set(cv2.CAP_PROP_AUTOFOCUS, 1)
            except Exception as exc:
                logger.warning(f"No se pudo encender la camara de sesion: {exc}")
                return
            self._cap = cap
            self._running.set()
            self._thread = threading.Thread(target=self._loop, daemon=True)
            self._thread.start()
        event_bus.publish("camera.opened", index=vision.camera_index)
        logger.debug("Camara de sesion encendida (hilo secundario).")

    def _loop(self) -> None:
        while self._running.is_set():
            try:
                frame = read_frame(self._cap)
            except Exception as exc:
                logger.debug(f"Error leyendo la camara de sesion: {exc}")
                time.sleep(0.1)
                continue
            with self._lock:
                self._latest_frame = frame

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
            from app.vision.camera import rotate_frame

            yield rotate_frame(current)
    finally:
        cap.release()
        cv2.destroyAllWindows()
