from __future__ import annotations

import threading
import time
from dataclasses import dataclass
from typing import Callable

import cv2
import numpy as np

from app.config.settings import vision
from app.vision.camera import FrameGrabber, show_preview
from app.vision.face_engine import AnalyzedFace, get_face_engine, normalize_embedding

# Guia hablada mientras la persona se acomoda frente a la camara. Quien usa
# OJOZ no ve la vista previa ni el recuadro del rostro, asi que sin esto no
# tiene forma de saber si esta bien puesta, si se salio del cuadro o si la
# camara simplemente no la esta viendo.
#
# Las frases son deliberadamente cortas: se escuchan a mitad de un ajuste de
# postura y se repiten, asi que una frase larga estorba mas de lo que ayuda.
_GUIDANCE_INTERVAL_SECONDS = 5.0
# Tope de espera extra mientras se le guia: sin esto, alguien que no llega a
# acomodarse dejaria el reconocimiento abierto indefinidamente.
_GUIDANCE_MAX_SECONDS = 20.0
# Margen libre exigido en cada borde para dar el rostro por completo: si el
# recuadro toca el borde de la imagen, lo normal es que parte de la cara ya
# haya quedado fuera.
_EDGE_MARGIN_RATIO = 0.02

_NO_FACE_MESSAGE = "No te veo, acomódate."
_CUT_OFF_MESSAGE = "No entras completo, acomódate."
_TOO_FAR_MESSAGE = "Acércate un poco."
_READY_MESSAGE = "Te veo, quédate quieto."


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


def _framing_message(frame: np.ndarray, face: AnalyzedFace) -> str | None:
    """Correccion hablada si el rostro no esta bien encuadrado, o None si lo esta."""
    height, width = frame.shape[:2]
    x1, y1, x2, y2 = (float(value) for value in face.bbox)
    margin_x = width * _EDGE_MARGIN_RATIO
    margin_y = height * _EDGE_MARGIN_RATIO

    if x1 < margin_x or y1 < margin_y or x2 > width - margin_x or y2 > height - margin_y:
        return _CUT_OFF_MESSAGE
    if min(x2 - x1, y2 - y1) < vision.face_min_size:
        return _TOO_FAR_MESSAGE
    return None


class _PositionGuide:
    """
    Decide que decirle a la persona para que se acomode, y cada cuanto.

    Se mantiene aparte del bucle de camara para poder razonarla (y probarla)
    sin hilos ni camara de por medio.
    """

    def __init__(
        self,
        on_feedback: Callable[[str], None],
        interval: float = _GUIDANCE_INTERVAL_SECONDS,
    ) -> None:
        self._on_feedback = on_feedback
        self._interval = interval
        self._last_at: float | None = None
        self._announced_ready = False

    def update(self, frame: np.ndarray, faces: list[AnalyzedFace], now: float) -> bool:
        """Avisa como acomodarse si hace falta; devuelve True si ya esta bien puesta."""
        if self._last_at is None:
            # La primera indicacion sale al instante, sin esperar un ciclo.
            self._last_at = now - self._interval

        issue = _NO_FACE_MESSAGE if not faces else _framing_message(frame, faces[0])

        if issue is None:
            if not self._announced_ready:
                self._announced_ready = True
                self._last_at = now
                self._on_feedback(_READY_MESSAGE)
            return True

        if now - self._last_at >= self._interval:
            # Solo al avisar de verdad se olvida el "te veo": asi un parpadeo
            # suelto del detector no lo hace repetirse.
            self._last_at = now
            self._announced_ready = False
            self._on_feedback(issue)
        return False


class _BackgroundAnalyzer:
    """
    Corre la deteccion y el embedding de InsightFace en su propio hilo.

    Una pasada del modelo tarda bastante mas que un fotograma (mucho mas en
    CPU), y hacerla dentro del bucle de la vista previa congelaba la imagen
    entre pasada y pasada. Aqui el bucle solo recoge el ultimo resultado
    disponible y sigue dibujando, asi que la camara se ve fluida aunque el
    modelo vaya lento.

    Cada resultado se numera para que quien consume distinga uno nuevo de
    uno ya visto: contar dos veces el mismo analisis inflaria las
    confirmaciones y daria por reconocida a una persona con un solo
    fotograma.
    """

    def __init__(self, engine, grabber: FrameGrabber) -> None:
        self._engine = engine
        self._grabber = grabber
        self._lock = threading.Lock()
        self._running = threading.Event()
        self._thread: threading.Thread | None = None
        self._faces: list[AnalyzedFace] = []
        self._sequence = 0

    def start(self) -> "_BackgroundAnalyzer":
        self._running.set()
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()
        return self

    def _loop(self) -> None:
        from app.utils.logger import logger

        # Espera a que la camara entregue el primero antes de empezar.
        if self._grabber.latest() is None:
            return

        frame_sequence = 0
        while self._running.is_set():
            fresh = self._grabber.latest_after(frame_sequence)
            if fresh is None:
                # Ya se analizo el fotograma mas reciente: volver a pasarlo por
                # el modelo daria el mismo resultado y solo gastaria CPU/GPU.
                time.sleep(0.002)
                continue
            frame_sequence, frame = fresh
            try:
                faces = self._engine.analyze(frame, max_faces=1)
            except Exception:
                logger.exception("Fallo el analisis facial de un fotograma")
                continue
            with self._lock:
                self._faces = faces
                self._sequence += 1

    def result_after(self, sequence: int) -> tuple[int, list[AnalyzedFace]] | None:
        """Ultimo analisis si es posterior a `sequence`; None si no hay uno nuevo."""
        with self._lock:
            if self._sequence <= sequence:
                return None
            return self._sequence, self._faces

    def faces(self) -> list[AnalyzedFace]:
        """Ultimo analisis conocido, para dibujar la vista previa."""
        with self._lock:
            return self._faces

    def stop(self) -> None:
        self._running.clear()
        thread, self._thread = self._thread, None
        if thread is not None:
            thread.join(timeout=2.0)

    def __enter__(self) -> "_BackgroundAnalyzer":
        return self.start()

    def __exit__(self, *_exc) -> None:
        self.stop()


def recognize_best_frame(
    seconds: float = 5.0,
    expected_name: str | None = None,
    on_feedback: Callable[[str], None] | None = None,
) -> tuple[bool, str | None, float | None]:
    """
    Reconoce a la persona frente a la camara durante `seconds`.

    Con `on_feedback` se le va diciendo en voz alta como acomodarse (cada
    _GUIDANCE_INTERVAL_SECONDS mientras no se le vea bien, y una vez cuando
    por fin se le ve). Mientras esta mal puesta el tiempo de reconocimiento
    no corre: no tiene sentido gastarlo mirando un cuadro vacio o media cara,
    y quien no ve necesita ese margen para corregir la postura. La espera
    extra esta acotada por _GUIDANCE_MAX_SECONDS. Sin `on_feedback` nadie
    podria oir la guia, asi que se mantiene el comportamiento de siempre.
    """
    gallery = load_recognizer()
    engine = get_face_engine()
    engine.warm_up()
    confirmations: dict[str, list[float]] = {}
    best_score: float | None = None
    guide = _PositionGuide(on_feedback) if on_feedback is not None else None
    started_at = time.monotonic()
    deadline = started_at + seconds
    hard_deadline = deadline + (_GUIDANCE_MAX_SECONDS if on_feedback else 0.0)
    last_sequence = 0
    score: float | None = None
    match_name: str | None = None

    with FrameGrabber(vision.camera_index) as grabber, _BackgroundAnalyzer(engine, grabber) as analyzer:
        frame = grabber.latest()
        if frame is None:
            return False, None, None

        frame_sequence = 0
        drawn_sequence = -1

        while True:
            now = time.monotonic()
            if now >= deadline or now >= hard_deadline:
                break

            fresh_frame = grabber.latest_after(frame_sequence)
            if fresh_frame is not None:
                frame_sequence, frame = fresh_frame

            # Cada analisis se evalua una sola vez; entre uno y otro el bucle
            # sigue refrescando la imagen para que se vea fluida.
            fresh = analyzer.result_after(last_sequence)
            if fresh is not None:
                last_sequence, faces = fresh
                match_name, score = None, None

                if guide is not None and not guide.update(frame, faces, now):
                    # El tiempo de reconocimiento no corre mientras este mal
                    # puesta: quien no ve necesita ese margen para corregir.
                    deadline = min(now + seconds, hard_deadline)

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
                # Solo se redibuja cuando la camara entrego algo nuevo; el
                # waitKey se llama igual en cada vuelta para que la ventana
                # siga respondiendo y para marcar el ritmo del bucle.
                if frame_sequence != drawn_sequence:
                    drawn_sequence = frame_sequence
                    known_faces = analyzer.faces()
                    if known_faces:
                        x1, y1, x2, y2 = known_faces[0].bbox.astype(int)
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
                    show_preview("OJOZ - autenticacion ArcFace", frame)
                if cv2.waitKey(1) & 0xFF == 27:
                    break
            else:
                # Sin ventana no hay waitKey que marque el ritmo: sin esta
                # pausa el bucle giraria en vacio comiendose un nucleo.
                time.sleep(0.005)

    return False, None, best_score
