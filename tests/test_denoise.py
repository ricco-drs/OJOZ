from __future__ import annotations

import unittest
from unittest.mock import Mock

import numpy as np

from app.audio.denoise import _Denoiser


class DenoiseBoundaryTests(unittest.TestCase):
    def test_overlap_add_preserves_first_and_last_samples(self):
        denoiser = _Denoiser()
        denoiser._io1 = denoiser._io2 = ("audio", "state")
        denoiser._stage1 = Mock()
        denoiser._stage2 = Mock()
        # Etapas identidad: cuatro bloques solapados deben reconstruir la senal.
        denoiser._stage1.run.side_effect = lambda _, inputs: (
            np.ones_like(inputs["audio"]), inputs["state"],
        )
        denoiser._stage2.run.side_effect = lambda _, inputs: (
            inputs["audio"] / 4, inputs["state"],
        )
        for length in (1, 128, 511, 16000, 16037):
            with self.subTest(length=length):
                samples = np.random.default_rng(7).uniform(-0.5, 0.5, length).astype(np.float32)
                result = denoiser._process(samples)
                self.assertEqual(len(result), length)
                np.testing.assert_allclose(result, samples, atol=1e-6)


if __name__ == "__main__":
    unittest.main()
