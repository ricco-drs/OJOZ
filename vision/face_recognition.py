# vision/face_recognition.py
import time

import cv2

from app.config.settings import load_face_detector, vision
from app.utils.fs import iter_user_folders
from app.vision.camera import open_camera, read_frame, release

# Se carga de forma perezosa: si el XML del clasificador falta, el error debe
# aparecer al usar el reconocimiento, no al importar el modulo.
_detector = None


def _get_detector():
    global _detector
    if _detector is None:
        _detector = load_face_detector()
    return _detector


def load_recognizer():
    if not vision.model_file.exists():
        raise FileNotFoundError("Modelo LBPH no encontrado. Entrena primero.")
    recognizer = cv2.face.LBPHFaceRecognizer_create()
    recognizer.read(str(vision.model_file))
    labels = [display for _, display in iter_user_folders()]
    return recognizer, labels


def recognize_best_frame(seconds: float = 5.0):
    detector = _get_detector()
    rec, labels = load_recognizer()
    cap = open_camera(vision.camera_index)
    name, conf = None, None
    t0 = time.time()

    try:
        while time.time() - t0 < seconds:
            frame = read_frame(cap)
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = detector.detectMultiScale(gray, 1.3, 5)
            for (x, y, w, h) in faces:
                face = gray[y : y + h, x : x + w]
                face = cv2.resize(face, vision.face_size, interpolation=cv2.INTER_CUBIC)
                pred = rec.predict(face)
                if conf is None or pred[1] < conf:
                    conf = pred[1]
                    name = labels[pred[0]] if pred[1] < vision.lbph_threshold else None
            if vision.show_preview:
                cv2.imshow("auth", frame)
                if cv2.waitKey(1) & 0xFF == 27:
                    break
    finally:
        release(cap)

    return (True, name, conf) if name else (False, None, conf)
