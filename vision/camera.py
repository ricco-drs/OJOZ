# vision/camera.py
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
