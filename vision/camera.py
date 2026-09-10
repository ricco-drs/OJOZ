# vision/camera.py
from __future__ import annotations

import threading
import time

import cv2

from app.config.settings import vision

_ROTATIONS = {
    90: cv2.ROTATE_90_CLOCKWISE,
    180: cv2.ROTATE_180,
    270: cv2.ROTATE_90_COUNTERCLOCKWISE,
}


def find_camera_index_by_name(name_hint: str) -> int | None:
    """
    Busca el indice de una camara por parte de su nombre (p. ej. "USB2.0 HD
    UVC WebCam", la webcam fisica del laptop).

    El indice de DirectShow no es estable entre sesiones: cambia segun que
    otras camaras/apps virtuales esten activas en ese momento (iVCam, OBS,
    NVIDIA Broadcast, etc.), asi que fijar un numero a mano es fragil. Sin
    pygrabber instalado, o si no encuentra el nombre, devuelve None y el
    llamador cae al indice configurado por numero.
    """
    try:
        from pygrabber.dshow_graph import FilterGraph

        devices = FilterGraph().get_input_devices()
    except Exception:
        return None

    hint = name_hint.casefold()
    for i, name in enumerate(devices):
        if hint in name.casefold():
            return i
    return None


def open_camera(index: int = 0):
    cap = cv2.VideoCapture(index, cv2.CAP_DSHOW)
    if not cap or not cap.isOpened():
        raise RuntimeError("No se pudo abrir la cámara")
    return cap

def rotate_frame(frame):
    """
    Gira el fotograma si OJOZ_CAMERA_ROTATE esta configurado (90/180/270).

    Util cuando la camara (p. ej. el celular como webcam via iVCam) entrega
    el video en horizontal aunque se sostenga en vertical.
    """
    rotation = _ROTATIONS.get(getattr(vision, "camera_rotate_degrees", 0))
    if rotation is None:
        return frame
    return cv2.rotate(frame, rotation)

def flip_frame(frame):
    """
    Corrige el efecto espejo si OJOZ_CAMERA_FLIP esta activado.

    Algunas camaras (p. ej. el celular como webcam via iVCam, en modo
    "selfie") entregan el video reflejado horizontalmente: el texto de un
    documento se ve al reves y no coincide con lo que la persona ve al
    sostenerlo frente a la camara.
    """
    if not getattr(vision, "camera_flip_horizontal", False):
        return frame
    return cv2.flip(frame, 1)

def show_preview(window_name: str, frame) -> None:
    """
    Muestra el fotograma en una ventana de vista previa, forzandola por
    encima de las demas ventanas abiertas.

    En una demo en vivo, quien expone necesita que el publico vea que la
    camara realmente se abrio (enrolamiento, autenticacion, OCR, dinero,
    vencimiento, escena); si la ventana de OpenCV queda detras de otra
    aplicacion (el navegador, la presentacion, etc.) eso no se nota.

    El "topmost" solo se aplica al crear la ventana, no en cada fotograma:
    reafirmarlo en cada frame (varias veces por segundo, en bucles como el
    de reconocimiento facial) obliga a Windows a recalcular el orden de
    ventanas todo el tiempo, lo que se nota como lag en la vista previa.
    """
    exists = cv2.getWindowProperty(window_name, cv2.WND_PROP_VISIBLE) >= 1
    if not exists:
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        try:
            cv2.setWindowProperty(window_name, cv2.WND_PROP_TOPMOST, 1)
        except cv2.error:
            pass
    cv2.imshow(window_name, frame)


def read_frame(cap):
    ok, frame = cap.read()
    if not ok:
        raise RuntimeError("No se pudo leer frame de la cámara")
    return flip_frame(rotate_frame(frame))

def release(cap):
    try:
        cap.release()
    finally:
        cv2.destroyAllWindows()


class FrameGrabber:
    """
    Lee la camara en un hilo aparte y conserva solo el ultimo fotograma.

    Leer la camara dentro del mismo bucle que hace la inferencia arrastra dos
    problemas. Primero, la vista previa solo se refresca al ritmo del modelo,
    asi que con InsightFace en CPU la imagen avanza a tirones. Segundo,
    DirectShow va encolando los fotogramas que nadie lee, de modo que lo que
    se muestra es cada vez mas viejo que lo que esta pasando de verdad.

    Este hilo vacia esa cola todo el tiempo y descarta lo que no alcanza a
    consumirse, asi que `latest()` siempre devuelve lo mas reciente que vio
    la camara, sin importar cuanto tarde quien lo pida.
    """

    def __init__(self, index: int | None = None) -> None:
        self._index = vision.camera_index if index is None else index
        self._cap = None
        self._frame = None
        self._sequence = 0
        self._lock = threading.Lock()
        self._running = threading.Event()
        self._has_frame = threading.Event()
        self._thread: threading.Thread | None = None

    def start(self) -> "FrameGrabber":
        self._cap = open_camera(self._index)
        self._running.set()
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()
        return self

    def _loop(self) -> None:
        while self._running.is_set():
            try:
                frame = read_frame(self._cap)
            except RuntimeError:
                # Un fotograma suelto que no llega no justifica cerrar la
                # camara: se reintenta, que es lo que hacia el bucle de antes.
                time.sleep(0.01)
                continue
            with self._lock:
                self._frame = frame
                self._sequence += 1
            self._has_frame.set()

    def latest(self, timeout: float = 2.0):
        """
        Copia del ultimo fotograma. Espera hasta `timeout` por el primero
        (abrir la camara tarda) y devuelve None si nunca llega.

        Se copia porque quien dibuja el recuadro del rostro lo hace sobre la
        imagen recibida, y esa escritura corromperia el fotograma compartido.
        """
        if not self._has_frame.wait(timeout):
            return None
        with self._lock:
            return None if self._frame is None else self._frame.copy()

    def latest_after(self, sequence: int):
        """
        (numero, copia) si hay un fotograma mas nuevo que `sequence`; None si no.

        Sirve para no repetir trabajo sobre una imagen que no cambio: el bucle
        de vista previa gira mucho mas rapido de lo que la camara entrega, y
        volver a dibujarla y copiarla cientos de veces por segundo gasta
        justamente lo que se quiere dedicar a que se vea fluida.
        """
        with self._lock:
            if self._sequence <= sequence or self._frame is None:
                return None
            return self._sequence, self._frame.copy()

    def stop(self) -> None:
        if not self._running.is_set():
            return
        self._running.clear()
        thread, self._thread = self._thread, None
        if thread is not None:
            thread.join(timeout=2.0)
        if self._cap is not None:
            release(self._cap)
            self._cap = None
        with self._lock:
            self._frame = None
            self._sequence = 0
        self._has_frame.clear()

    def __enter__(self) -> "FrameGrabber":
        return self.start()

    def __exit__(self, *_exc) -> None:
        self.stop()
