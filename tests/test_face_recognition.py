from __future__ import annotations

import unittest

import numpy as np

from app.vision.face_recognition import FaceGallery, find_best_match


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


if __name__ == "__main__":
    unittest.main()
