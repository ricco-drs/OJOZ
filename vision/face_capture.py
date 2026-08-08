# vision/face_capture.py
from typing import Optional

import cv2, imutils

from app.config.settings import load_face_detector, vision
from app.utils.fs import ensure_user_folder
from app.vision.camera import open_camera, read_frame, release

def capture_faces(name: str, count: Optional[int] = None, show_preview: Optional[bool] = None) -> int:
    from app.utils.logger import logger
    
    count = count if count is not None else vision.capture_count
    show_preview = vision.show_preview if show_preview is None else show_preview

    detector = load_face_detector()
    folder = ensure_user_folder(name)
    logger.info(f"Carpeta de captura: {folder}")

    cap = open_camera(vision.camera_index)
    saved = 0

    try:
        while saved < count:
            frame = read_frame(cap)
            frame = imutils.resize(frame, width=840)
            gray  = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            aux   = frame.copy()

            faces = detector.detectMultiScale(gray, 1.3, 5)
            for (x,y,w,h) in faces:
                rostro = aux[y:y+h, x:x+w]
                rostro = cv2.resize(rostro, vision.face_size, interpolation=cv2.INTER_CUBIC)
                out = folder / f"rostro_{saved:04d}.jpg"
                success = cv2.imwrite(str(out), rostro)
                
                if not success:
                    logger.error(f"No se pudo guardar la imagen: {out}")
                else:
                    saved += 1
                    if saved % 50 == 0:  # Log cada 50 fotos
                        logger.info(f"Capturadas {saved}/{count} fotos")

                if show_preview:
                    cv2.rectangle(frame, (x,y), (x+w,y+h), (0,255,0), 2)
                    cv2.putText(frame, f"{saved}/{count}", (10, 30), 
                               cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

                if saved >= count:
                    break

            if show_preview:
                cv2.imshow("captura", frame)
                if cv2.waitKey(1) & 0xFF == 27:
                    break
    finally:
        release(cap)
        logger.info(f"Captura finalizada: {saved} fotos guardadas en {folder}")
        try:
            cv2.destroyAllWindows()
        except Exception:
            pass

    return saved


