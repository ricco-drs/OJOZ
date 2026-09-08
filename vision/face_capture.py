from __future__ import annotations

import json
import shutil
import time
from pathlib import Path
from typing import Optional

import cv2
import imutils

from app.config.settings import vision
from app.utils.fs import ensure_user_folder
from app.vision.camera import open_camera, read_frame, release
from app.vision.face_engine import DetectedFace, get_face_engine


def _face_quality(frame, face: DetectedFace) -> tuple[bool, str]:
    height, width = frame.shape[:2]
    x1, y1, x2, y2 = face.bbox.astype(int)
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(width, x2), min(height, y2)
    face_width, face_height = x2 - x1, y2 - y1

    if min(face_width, face_height) < vision.face_min_size:
        return False, "Acerquese a la camara"

    crop = frame[y1:y2, x1:x2]
    if crop.size == 0:
        return False, "Rostro fuera de cuadro"

    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    sharpness = float(cv2.Laplacian(gray, cv2.CV_64F).var())
    if sharpness < vision.face_blur_threshold:
        return False, "Mantengase quieto"

    return True, "Mueva lentamente el rostro"


def _promote_capture(temp_folder: Path, user_folder: Path, count: int) -> None:
    for old_photo in user_folder.glob("rostro_*.jpg"):
        old_photo.unlink()
    for photo in sorted(temp_folder.glob("rostro_*.jpg")):
        photo.replace(user_folder / photo.name)

    metadata = {
        "format": "full_frame",
        "model": vision.face_model_name,
        "photos": count,
    }
    (user_folder / ".arcface_enrollment.json").write_text(
        json.dumps(metadata, ensure_ascii=True, indent=2),
        encoding="utf-8",
    )


def capture_faces(
    name: Optional[str] = None,
    count: Optional[int] = None,
    show_preview: Optional[bool] = None,
    folder: Optional[Path] = None,
) -> int:
    """
    Captura fotos de un rostro. Si se pasa `folder` explicito, se captura ahi
    directamente (uso: capturar antes de saber el nombre/apodo de la persona).
    Si no, se resuelve la carpeta a partir de `name` como siempre.
    """
    from app.utils.logger import logger

    count = count if count is not None else vision.capture_count
    show_preview = vision.show_preview if show_preview is None else show_preview
    if folder is None:
        folder = ensure_user_folder(name or "")
    else:
        folder.mkdir(parents=True, exist_ok=True)
    temp_folder = folder / ".capture_tmp"
    logger.debug("Carpeta de captura: %s", folder)

    if temp_folder.exists():
        shutil.rmtree(temp_folder)
    temp_folder.mkdir(parents=True)

    engine = get_face_engine()
    engine.warm_up()
    cap = open_camera(vision.camera_index)
    saved = 0
    last_saved_at = 0.0
    started_at = time.monotonic()

    try:
        while saved < count and time.monotonic() - started_at < vision.capture_timeout_seconds:
            frame = imutils.resize(read_frame(cap), width=vision.capture_frame_width)
            faces = engine.detect(frame, max_faces=2)
            status = "Coloque un solo rostro frente a la camara"
            selected = faces[0] if len(faces) == 1 else None

            if selected is not None:
                quality_ok, status = _face_quality(frame, selected)
                now = time.monotonic()
                if quality_ok and now - last_saved_at >= vision.capture_interval_seconds:
                    output = temp_folder / f"rostro_{saved:04d}.jpg"
                    if cv2.imwrite(str(output), frame, [cv2.IMWRITE_JPEG_QUALITY, 92]):
                        saved += 1
                        last_saved_at = now
                        if saved % 5 == 0 or saved == count:
                            logger.debug("Capturadas %s/%s fotos validas", saved, count)
                    else:
                        logger.error("No se pudo guardar la imagen: %s", output)

            if show_preview:
                for face in faces:
                    x1, y1, x2, y2 = face.bbox.astype(int)
                    color = (0, 200, 0) if len(faces) == 1 else (0, 165, 255)
                    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                cv2.putText(
                    frame,
                    f"{saved}/{count} - {status}",
                    (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 255, 0),
                    2,
                )
                cv2.imshow("OJOZ - enrolamiento ArcFace", frame)
                if cv2.waitKey(1) & 0xFF == 27:
                    break
    finally:
        release(cap)

    try:
        if saved >= vision.min_enrollment_photos:
            _promote_capture(temp_folder, folder, saved)
        else:
            logger.warning(
                "Captura descartada: %s fotos; se requieren al menos %s",
                saved,
                vision.min_enrollment_photos,
            )
    finally:
        shutil.rmtree(temp_folder, ignore_errors=True)

    logger.debug("Captura finalizada: %s fotos validas en %s", saved, folder)
    return saved
