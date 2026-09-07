from __future__ import annotations

import time
from dataclasses import dataclass

import cv2
import numpy as np

from app.config.settings import vision
from app.vision.camera import open_camera, read_frame, release
from app.vision.face_engine import get_face_engine, normalize_embedding


@dataclass(frozen=True)
class FaceGallery:
    names: tuple[str, ...]
    embeddings: np.ndarray
    sample_counts: np.ndarray
    model_name: str


def load_recognizer() -> FaceGallery:
    if not vision.model_file.exists():
        raise FileNotFoundError("Galeria ArcFace no encontrada. Registra un usuario primero.")

    try:
        with np.load(vision.model_file, allow_pickle=False) as data:
            schema_version = int(data["schema_version"][0])
            model_name = str(data["model_name"][0])
            names = tuple(str(name) for name in data["names"].tolist())
            embeddings = np.asarray(data["embeddings"], dtype=np.float32)
            sample_counts = np.asarray(data["sample_counts"], dtype=np.int32)
    except (KeyError, OSError, ValueError) as exc:
        raise RuntimeError("La galeria ArcFace esta danada o es incompatible") from exc

    if schema_version != 1 or model_name != vision.face_model_name:
        raise RuntimeError(
            "La galeria facial fue creada con otro modelo. Registra nuevamente los rostros."
        )
    if embeddings.ndim != 2 or len(names) != len(embeddings) or not names:
        raise RuntimeError("La galeria ArcFace no contiene identidades validas")

    normalized = np.vstack([normalize_embedding(row) for row in embeddings])
    normalized.setflags(write=False)
    return FaceGallery(names, normalized, sample_counts, model_name)


def find_best_match(
    query_embedding: np.ndarray,
    gallery: FaceGallery,
    expected_name: str | None = None,
) -> tuple[str | None, float]:
    query = normalize_embedding(query_embedding)
    scores = gallery.embeddings @ query

    if expected_name:
        expected = expected_name.casefold()
        index = next(
            (i for i, name in enumerate(gallery.names) if name.casefold() == expected),
            None,
        )
        if index is None:
            return None, float(scores.max())
        score = float(scores[index])
        return (gallery.names[index] if score >= vision.face_similarity_threshold else None, score)

    order = np.argsort(scores)[::-1]
    best_index = int(order[0])
    best_score = float(scores[best_index])
    second_score = float(scores[int(order[1])]) if len(order) > 1 else -1.0
    is_clear = best_score - second_score >= vision.face_ambiguity_margin

    if best_score >= vision.face_similarity_threshold and is_clear:
        return gallery.names[best_index], best_score
    return None, best_score


def recognize_best_frame(
    seconds: float = 5.0,
    expected_name: str | None = None,
) -> tuple[bool, str | None, float | None]:
    gallery = load_recognizer()
    engine = get_face_engine()
    engine.warm_up()
    cap = open_camera(vision.camera_index)
    confirmations: dict[str, list[float]] = {}
    best_score: float | None = None
    started_at = time.monotonic()

    try:
        while time.monotonic() - started_at < seconds:
            frame = read_frame(cap)
            faces = engine.analyze(frame, max_faces=1)
            match_name: str | None = None
            score: float | None = None

            if faces:
                match_name, score = find_best_match(
                    faces[0].embedding,
                    gallery,
                    expected_name=expected_name,
                )
                best_score = score if best_score is None else max(best_score, score)
                if match_name is not None:
                    matches = confirmations.setdefault(match_name, [])
                    matches.append(score)
                    if len(matches) >= vision.face_required_confirmations:
                        confirmed_score = float(
                            np.median(matches[-vision.face_required_confirmations :])
                        )
                        if confirmed_score >= vision.face_similarity_threshold:
                            return True, match_name, confirmed_score

            if vision.show_preview:
                if faces:
                    x1, y1, x2, y2 = faces[0].bbox.astype(int)
                    color = (0, 200, 0) if match_name else (0, 165, 255)
                    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                label = "Buscando rostro" if score is None else f"Similitud: {score:.3f}"
                cv2.putText(
                    frame,
                    label,
                    (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (0, 255, 0),
                    2,
                )
                cv2.imshow("OJOZ - autenticacion ArcFace", frame)
                if cv2.waitKey(1) & 0xFF == 27:
                    break
    finally:
        release(cap)

    return False, None, best_score
