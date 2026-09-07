# vision/camera.py
import cv2

from app.config.settings import vision

_ROTATIONS = {
    90: cv2.ROTATE_90_CLOCKWISE,
    180: cv2.ROTATE_180,
    270: cv2.ROTATE_90_COUNTERCLOCKWISE,
}


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

def read_frame(cap):
    ok, frame = cap.read()
    if not ok:
        raise RuntimeError("No se pudo leer frame de la cámara")
    return rotate_frame(frame)

def release(cap):
    try:
        cap.release()
    finally:
        cv2.destroyAllWindows()
