# vision/face_train.py
import cv2
import numpy as np

from app.config.settings import vision
from app.utils.fs import iter_user_folders


def train_dataset() -> str:
    from app.utils.logger import logger

    # Verificar que OpenCV tenga el módulo "face" (se requiere opencv-contrib)
    if not hasattr(cv2, "face"):
        raise RuntimeError(
            "OpenCV no tiene el módulo 'face'. Instala opencv-contrib-python para entrenar el modelo LBPH."
        )

    if not vision.fotos_dir.exists():
        raise RuntimeError(f"Directorio de fotos no existe: {vision.fotos_dir}")

    people = list(iter_user_folders())
    logger.info(f"Personas encontradas para entrenar: {[display for _, display in people]}")

    if not people:
        raise RuntimeError(f"No hay carpetas de usuarios en {vision.fotos_dir}")

    labels, faces = [], []
    label = 0

    for folder_path, display_name in people:
        files = [
            f
            for f in folder_path.iterdir()
            if f.is_file() and f.suffix.lower() in {".jpg", ".jpeg", ".png"}
        ]
        logger.info(f"Procesando {len(files)} imágenes de {display_name}")

        images_loaded = 0
        for file_path in files:
            img = cv2.imread(str(file_path), 0)
            if img is None:
                logger.warning(f"No se pudo leer imagen: {file_path}")
                continue
            faces.append(img)
            labels.append(label)
            images_loaded += 1

        logger.info(f"Cargadas {images_loaded} imágenes válidas de {display_name}")
        label += 1

    if not faces:
        raise RuntimeError("No hay imágenes de rostros válidas para entrenar")

    logger.info(f"Entrenando modelo con {len(faces)} rostros de {len(people)} personas...")
    recognizer = cv2.face.LBPHFaceRecognizer_create()
    recognizer.train(faces, np.array(labels))

    vision.model_file.parent.mkdir(parents=True, exist_ok=True)
    recognizer.write(str(vision.model_file))
    logger.info(f"Modelo guardado exitosamente en: {vision.model_file}")

    if not vision.model_file.exists():
        raise RuntimeError(f"El modelo no se creó en {vision.model_file}")

    return str(vision.model_file)
