"""
Lista las camaras que OpenCV es capaz de abrir y guarda un fotograma de prueba.

Ejecutar con: python -m app.tools.check_camera
"""
import sys

import cv2


def main() -> None:
    print("Python:", sys.version.split()[0])
    print("OpenCV:", cv2.__version__)

    found = False
    for i in range(6):
        try:
            # Usar DirectShow backend en Windows para mejor compatibilidad
            cap = cv2.VideoCapture(i, cv2.CAP_DSHOW)
        except Exception:
            cap = cv2.VideoCapture(i)

        opened = cap.isOpened()
        ok = False
        frame = None
        if opened:
            ok, frame = cap.read()
        print(f"index={i} opened={opened} read={ok}")

        if ok and frame is not None:
            fname = f"test_cam_{i}.jpg"
            cv2.imwrite(fname, frame)
            print("Guardado:", fname)
            found = True

        try:
            cap.release()
        except Exception:
            pass

    if not found:
        print("OpenCV no detecto ninguna camara (indices 0..5).")
    else:
        print("Camara detectada. Revisa los archivos test_cam_*.jpg generados.")


if __name__ == "__main__":
    main()
