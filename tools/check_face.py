"""Prepara InsightFace y muestra el proveedor real usado por OJOZ."""

from __future__ import annotations

import argparse
import sys

import cv2

from app.config.settings import vision
from app.vision.camera import open_camera, read_frame, release
from app.vision.face_engine import get_face_engine


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--camera",
        action="store_true",
        help="abre la camara y comprueba que se detecte un rostro",
    )
    args = parser.parse_args()

    print("Python:", sys.version.split()[0])
    print("Modelo:", vision.face_model_name)
    print("Directorio:", vision.insightface_root)
    print("Preparando InsightFace (la primera vez puede descargar el modelo)...")

    engine = get_face_engine()
    provider = engine.warm_up()
    print("Proveedor activo:", provider)

    if not args.camera:
        print("OK: modelo listo. Usa --camera para probar deteccion.")
        return 0

    cap = open_camera(vision.camera_index)
    try:
        frame = read_frame(cap)
        faces = engine.analyze(frame)
    finally:
        release(cap)

    print("Rostros detectados:", len(faces))
    if not faces:
        print("No se detecto un rostro; revisa iluminacion, distancia y camara.")
        return 1
    print("OK: deteccion y embedding ArcFace funcionando.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
