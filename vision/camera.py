# vision/camera.py
import cv2

def open_camera(index: int = 0):
    cap = cv2.VideoCapture(index, cv2.CAP_DSHOW)
    if not cap or not cap.isOpened():
        raise RuntimeError("No se pudo abrir la cámara")
    return cap

def read_frame(cap):
    ok, frame = cap.read()
    if not ok:
        raise RuntimeError("No se pudo leer frame de la cámara")
    return frame

def release(cap):
    try:
        cap.release()
    finally:
        cv2.destroyAllWindows()
