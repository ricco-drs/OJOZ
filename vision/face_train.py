from __future__ import annotations

import os
import tempfile
from pathlib import Path

import cv2
import numpy as np

from app.config.settings import vision
from app.utils.fs import iter_user_folders
from app.vision.face_engine import get_face_engine, normalize_embedding


def _robust_centroid(embeddings: list[np.ndarray]) -> tuple[np.ndarray, int]:
    matrix = np.vstack(embeddings).astype(np.float32)
    initial = normalize_embedding(matrix.mean(axis=0))

    if len(matrix) >= 5:
        similarities = matrix @ initial
        median = float(np.median(similarities))
        mad = float(np.median(np.abs(similarities - median)))
        cutoff = max(0.25, median - max(0.05, 3.0 * mad))
        filtered = matrix[similarities >= cutoff]
        if len(filtered) >= 3:
            matrix = filtered

    return normalize_embedding(matrix.mean(axis=0)), len(matrix)


def _save_gallery(
    path: Path,
    names: list[str],
    embeddings: list[np.ndarray],
    sample_counts: list[int],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    handle, temp_name = tempfile.mkstemp(prefix="arcface_", suffix=".npz", dir=path.parent)
    try:
        with os.fdopen(handle, "wb") as temp_file:
            np.savez_compressed(
                temp_file,
                schema_version=np.asarray([1], dtype=np.int32),
                model_name=np.asarray([vision.face_model_name]),
                names=np.asarray(names),
                embeddings=np.vstack(embeddings).astype(np.float32),
                sample_counts=np.asarray(sample_counts, dtype=np.int32),
            )
        os.replace(temp_name, path)
    except Exception:
        try:
            os.unlink(temp_name)
        except OSError:
            pass
        raise


def train_dataset() -> str:
    """Genera la galeria ArcFace a partir de las capturas de cada usuario."""
    from app.utils.logger import logger

    if not vision.fotos_dir.exists():
        raise RuntimeError(f"Directorio de fotos no existe: {vision.fotos_dir}")

    people = list(iter_user_folders())
    if not people:
        raise RuntimeError(f"No hay carpetas de usuarios en {vision.fotos_dir}")

    logger.debug("Personas encontradas para enrolar: %s", [name for _, name in people])
    engine = get_face_engine()
    engine.warm_up()
    names: list[str] = []
    centroids: list[np.ndarray] = []
    sample_counts: list[int] = []

    for folder, display_name in people:
        files = sorted(
            path
            for path in folder.iterdir()
            if path.is_file() and path.suffix.lower() in {".jpg", ".jpeg", ".png"}
        )
        embeddings: list[np.ndarray] = []

        for file_path in files:
            image = cv2.imread(str(file_path))
            if image is None:
                logger.warning("No se pudo leer imagen: %s", file_path)
                continue
            # Las capturas antiguas eran recortes de 120x120 sin alinear.
            # No son plantillas fiables para ArcFace y deben volver a capturarse.
            if min(image.shape[:2]) < 240:
                continue
            faces = engine.analyze(image, max_faces=1)
            if faces:
                embeddings.append(faces[0].embedding)

        if len(embeddings) < vision.min_enrollment_photos:
            logger.warning(
                "Se omite %s: %s capturas ArcFace validas de %s requeridas",
                display_name,
                len(embeddings),
                vision.min_enrollment_photos,
            )
            continue

        centroid, retained = _robust_centroid(embeddings)
        names.append(display_name)
        centroids.append(centroid)
        sample_counts.append(retained)
        logger.debug("Galeria: %s con %s muestras validas", display_name, retained)

    if not centroids:
        raise RuntimeError(
            "No hay capturas validas para ArcFace. Registra nuevamente al usuario."
        )

    _save_gallery(vision.model_file, names, centroids, sample_counts)
    logger.debug("Galeria ArcFace guardada en: %s", vision.model_file)
    return str(vision.model_file)
