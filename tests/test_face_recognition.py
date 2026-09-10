from __future__ import annotations

import time
import unittest
from unittest.mock import Mock

import numpy as np

from app.config.settings import vision
from app.vision.face_engine import AnalyzedFace
from app.vision.face_recognition import (
    _CUT_OFF_MESSAGE,
    _NO_FACE_MESSAGE,
    _READY_MESSAGE,
    _TOO_FAR_MESSAGE,
    FaceGallery,
    _BackgroundAnalyzer,
    _framing_message,
    _PositionGuide,
    find_best_match,
)


class FindBestMatchTests(unittest.TestCase):
    def setUp(self) -> None:
        self.gallery = FaceGallery(
            names=("Ricco", "Ana"),
            embeddings=np.asarray(
                [
                    [1.0, 0.0, 0.0],
                    [0.0, 1.0, 0.0],
                ],
                dtype=np.float32,
            ),
            sample_counts=np.asarray([20, 20], dtype=np.int32),
            model_name="buffalo_l",
        )

    def test_identifies_clear_best_match(self) -> None:
        name, score = find_best_match(np.asarray([1.0, 0.0, 0.0]), self.gallery)
        self.assertEqual(name, "Ricco")
        self.assertAlmostEqual(score, 1.0)

    def test_verifies_only_claimed_identity(self) -> None:
        name, score = find_best_match(
            np.asarray([1.0, 0.0, 0.0]),
            self.gallery,
            expected_name="Ana",
        )
        self.assertIsNone(name)
        self.assertAlmostEqual(score, 0.0)

    def test_rejects_ambiguous_identification(self) -> None:
        name, score = find_best_match(np.asarray([1.0, 1.0, 0.0]), self.gallery)
        self.assertIsNone(name)
        self.assertGreater(score, 0.5)


def face(x1: float, y1: float, x2: float, y2: float) -> AnalyzedFace:
    return AnalyzedFace(
        bbox=np.asarray([x1, y1, x2, y2], dtype=np.float32),
        detection_score=0.9,
        embedding=np.asarray([1.0, 0.0, 0.0], dtype=np.float32),
    )


class FramingMessageTests(unittest.TestCase):
    def setUp(self) -> None:
        self.frame = np.zeros((480, 640, 3), dtype=np.uint8)
        self.size = vision.face_min_size

    def test_centered_face_needs_no_correction(self) -> None:
        centered = face(200, 100, 200 + self.size, 100 + self.size)
        self.assertIsNone(_framing_message(self.frame, centered))

    def test_face_touching_the_edge_is_reported_as_cut_off(self) -> None:
        for cut in (
            face(0, 100, self.size, 100 + self.size),          # pegado a la izquierda
            face(200, 0, 200 + self.size, self.size),          # pegado arriba
            face(640 - self.size, 100, 640, 100 + self.size),  # pegado a la derecha
            face(200, 480 - self.size, 200 + self.size, 480),  # pegado abajo
        ):
            self.assertEqual(_framing_message(self.frame, cut), _CUT_OFF_MESSAGE)

    def test_small_face_is_reported_as_too_far(self) -> None:
        tiny = face(300, 200, 300 + self.size - 1, 200 + self.size - 1)
        self.assertEqual(_framing_message(self.frame, tiny), _TOO_FAR_MESSAGE)


class PositionGuideTests(unittest.TestCase):
    """La guia hablada es lo unico que le dice a una persona ciega como acomodarse."""

    def setUp(self) -> None:
        self.frame = np.zeros((480, 640, 3), dtype=np.uint8)
        size = vision.face_min_size
        self.well_framed = [face(200, 100, 200 + size, 100 + size)]
        self.cut_off = [face(0, 100, size, 100 + size)]
        self.spoken: list[str] = []
        self.guide = _PositionGuide(self.spoken.append)

    def update(self, faces: list, at: float) -> bool:
        return self.guide.update(self.frame, faces, at)

    def test_missing_face_is_announced_at_once_and_then_every_five_seconds(self) -> None:
        for second in range(12):
            self.assertFalse(self.update([], 1000.0 + second))
        # Inmediato, y luego uno cada cinco segundos.
        self.assertEqual(self.spoken, [_NO_FACE_MESSAGE] * 3)

    def test_partially_visible_face_is_reported_as_cut_off(self) -> None:
        self.assertFalse(self.update(self.cut_off, 1000.0))
        self.assertEqual(self.spoken, [_CUT_OFF_MESSAGE])

    def test_good_position_is_confirmed_once(self) -> None:
        for second in range(12):
            self.assertTrue(self.update(self.well_framed, 1000.0 + second))
        self.assertEqual(self.spoken, [_READY_MESSAGE])

    def test_confirmation_comes_after_correcting_the_position(self) -> None:
        self.update([], 1000.0)
        self.update([], 1001.0)
        self.assertTrue(self.update(self.well_framed, 1002.0))
        self.assertEqual(self.spoken, [_NO_FACE_MESSAGE, _READY_MESSAGE])

    def test_detector_flicker_does_not_repeat_the_confirmation(self) -> None:
        # Un fotograma suelto sin deteccion, entre varios buenos, no debe
        # provocar un segundo "te veo".
        self.update(self.well_framed, 1000.0)
        self.update([], 1000.5)
        self.update(self.well_framed, 1001.0)
        self.assertEqual(self.spoken, [_READY_MESSAGE])

    def test_losing_the_face_for_a_while_warns_and_confirms_again(self) -> None:
        self.update(self.well_framed, 1000.0)
        for second in range(1, 8):
            self.update([], 1000.0 + second)
        self.update(self.well_framed, 1008.0)
        self.assertEqual(self.spoken, [_READY_MESSAGE, _NO_FACE_MESSAGE, _READY_MESSAGE])


class FakeGrabber:
    """Camara simulada que entrega un fotograma nuevo en cada consulta."""

    def __init__(self) -> None:
        self.frame = np.zeros((480, 640, 3), dtype=np.uint8)
        self.sequence = 0

    def latest(self, timeout: float = 2.0):
        return self.frame

    def latest_after(self, sequence: int):
        self.sequence += 1
        return self.sequence, self.frame


class BackgroundAnalyzerTests(unittest.TestCase):
    def setUp(self) -> None:
        size = vision.face_min_size
        self.faces = [face(200, 100, 200 + size, 100 + size)]
        self.engine = Mock()
        self.grabber = FakeGrabber()

    def analyzer(self) -> _BackgroundAnalyzer:
        instance = _BackgroundAnalyzer(self.engine, self.grabber)
        self.addCleanup(instance.stop)
        return instance

    def test_a_frame_already_analyzed_is_not_analyzed_again(self) -> None:
        # Volver a pasar el mismo fotograma por el modelo daria el mismo
        # resultado y solo gastaria CPU y GPU.
        still = FakeGrabber()
        still.latest_after = lambda sequence: None
        instance = _BackgroundAnalyzer(self.engine, still)
        self.addCleanup(instance.stop)

        instance.start()
        time.sleep(0.05)
        instance.stop()

        self.engine.analyze.assert_not_called()

    def test_same_analysis_is_delivered_only_once(self) -> None:
        # Releer el mismo analisis inflaria las confirmaciones y daria por
        # reconocida a una persona con un solo fotograma.
        instance = self.analyzer()
        with instance._lock:
            instance._faces = self.faces
            instance._sequence = 1

        sequence, faces = instance.result_after(0)
        self.assertEqual((sequence, faces), (1, self.faces))
        self.assertIsNone(instance.result_after(sequence))

    def test_failed_inference_does_not_kill_the_thread(self) -> None:
        attempts = []

        def analyze(*_args, **_kwargs):
            attempts.append(1)
            if len(attempts) == 1:
                raise RuntimeError("modelo ocupado")
            return self.faces

        self.engine.analyze.side_effect = analyze
        instance = self.analyzer().start()
        deadline = time.monotonic() + 2.0
        while instance.result_after(0) is None and time.monotonic() < deadline:
            time.sleep(0.01)
        instance.stop()

        fresh = instance.result_after(0)
        self.assertIsNotNone(fresh)
        self.assertEqual(fresh[1], self.faces)


if __name__ == "__main__":
    unittest.main()
