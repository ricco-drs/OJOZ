from __future__ import annotations

import threading
from dataclasses import dataclass
from functools import lru_cache
from typing import Any

import numpy as np

from app.config.settings import vision


@dataclass(frozen=True)
class AnalyzedFace:
    bbox: np.ndarray
    detection_score: float
    embedding: np.ndarray


@dataclass(frozen=True)
class DetectedFace:
    bbox: np.ndarray
    detection_score: float


def normalize_embedding(embedding: np.ndarray) -> np.ndarray:
    vector = np.asarray(embedding, dtype=np.float32).reshape(-1)
    norm = float(np.linalg.norm(vector))
    if not np.isfinite(norm) or norm <= 1e-12:
        raise ValueError("InsightFace devolvio un embedding facial invalido")
    return vector / norm


class InsightFaceEngine:
    """Carga SCRFD + ArcFace una sola vez y serializa las inferencias."""

    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._analysis: Any = None
        self.provider = ""

    def _load(self) -> Any:
        if self._analysis is not None:
            return self._analysis

        with self._lock:
            if self._analysis is not None:
                return self._analysis

            try:
                import onnxruntime as ort
                from insightface.app import FaceAnalysis
            except ImportError as exc:
                raise RuntimeError(
                    "InsightFace no esta instalado. Ejecuta: "
                    "python -m pip install -r requirements.txt"
                ) from exc

            # ORT GPU puede cargar las DLL de CUDA/cuDNN instaladas por pip.
            if hasattr(ort, "preload_dlls"):
                try:
                    ort.preload_dlls(directory="")
                except Exception:
                    pass

            available = set(ort.get_available_providers())
            requested = vision.face_provider.strip().lower()
            if requested not in {"auto", "cuda", "cpu"}:
                raise ValueError("OJOZ_FACE_PROVIDER debe ser auto, cuda o cpu")

            use_cuda = requested != "cpu" and "CUDAExecutionProvider" in available
            if requested == "cuda" and not use_cuda:
                raise RuntimeError(
                    "Se solicito CUDA, pero ONNX Runtime no ofrece CUDAExecutionProvider"
                )

            providers = (
                ["CUDAExecutionProvider", "CPUExecutionProvider"]
                if use_cuda
                else ["CPUExecutionProvider"]
            )
            ctx_id = 0 if use_cuda else -1

            try:
                analysis = FaceAnalysis(
                    name=vision.face_model_name,
                    root=str(vision.insightface_root),
                    allowed_modules=["detection", "recognition"],
                    providers=providers,
                )
                analysis.prepare(
                    ctx_id=ctx_id,
                    det_thresh=vision.face_detection_threshold,
                    det_size=vision.face_detection_size,
                )
            except Exception:
                if not use_cuda or requested == "cuda":
                    raise

                # En modo auto, una instalacion incompleta de CUDA no debe dejar
                # inutilizable la demo: se reconstruyen las sesiones sobre CPU.
                analysis = FaceAnalysis(
                    name=vision.face_model_name,
                    root=str(vision.insightface_root),
                    allowed_modules=["detection", "recognition"],
                    providers=["CPUExecutionProvider"],
                )
                analysis.prepare(
                    ctx_id=-1,
                    det_thresh=vision.face_detection_threshold,
                    det_size=vision.face_detection_size,
                )
                use_cuda = False

            self._analysis = analysis
            self.provider = "CUDA" if use_cuda else "CPU"

            from app.utils.logger import logger

            logger.debug(
                "InsightFace listo: modelo=%s, proveedor=%s, deteccion=%s",
                vision.face_model_name,
                self.provider,
                vision.face_detection_size,
            )
            return analysis

    def analyze(self, frame: np.ndarray, max_faces: int = 0) -> list[AnalyzedFace]:
        if frame is None or frame.size == 0:
            return []

        analysis = self._load()
        with self._lock:
            raw_faces = analysis.get(frame, max_num=max_faces)

        result: list[AnalyzedFace] = []
        for face in raw_faces:
            embedding = getattr(face, "normed_embedding", None)
            if embedding is None:
                embedding = getattr(face, "embedding", None)
            if embedding is None:
                continue
            result.append(
                AnalyzedFace(
                    bbox=np.asarray(face.bbox, dtype=np.float32),
                    detection_score=float(face.det_score),
                    embedding=normalize_embedding(embedding),
                )
            )
        return result

    def detect(self, frame: np.ndarray, max_faces: int = 0) -> list[DetectedFace]:
        """Solo deteccion (SCRFD), sin calcular el embedding ArcFace.

        Mucho mas liviano que analyze(): para vistas previas en vivo donde
        todavia no hace falta el embedding, como el bucle de captura de
        enrolamiento (el embedding se calcula despues, sobre las fotos ya
        guardadas, en face_train.py).
        """
        if frame is None or frame.size == 0:
            return []

        analysis = self._load()
        with self._lock:
            bboxes, _ = analysis.det_model.detect(frame, max_num=max_faces, metric="default")

        return [
            DetectedFace(
                bbox=np.asarray(bboxes[i, :4], dtype=np.float32),
                detection_score=float(bboxes[i, 4]),
            )
            for i in range(bboxes.shape[0])
        ]

    def warm_up(self) -> str:
        self._load()
        return self.provider


@lru_cache(maxsize=1)
def get_face_engine() -> InsightFaceEngine:
    return InsightFaceEngine()
