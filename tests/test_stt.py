from __future__ import annotations

import io
import struct
import unittest
from unittest.mock import Mock, patch

import speech_recognition as sr

from app.audio.microphone import Microphone, input_devices
from app.audio.stt import STT
from app.config.settings import STTConfig


def pcm(level: int, seconds: float) -> bytes:
    return struct.pack("<hh", level, -level) * int(16000 * seconds / 2)


class MemoryMicrophone(sr.AudioSource):
    SAMPLE_RATE = 16000
    SAMPLE_WIDTH = 2
    CHUNK = 480

    def __init__(self, data: bytes):
        self.data = io.BytesIO(data)
        self.stream = self

    def read(self, frames):
        return self.data.read(frames * self.SAMPLE_WIDTH)

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.data.close()


class STTTests(unittest.TestCase):
    def setUp(self):
        self.devices = [{"index": 1, "name": "Realtek"}, {"index": 6, "name": "Realtek"},
                        {"index": 7, "name": "iVCam"}]
        self.config = STTConfig(device_index=1, denoise_enabled=False)
        for target, value in (
            ("app.audio.stt.stt_config", self.config),
            ("app.audio.stt.input_devices", Mock(return_value=self.devices)),
            ("speech_recognition.Microphone.list_microphone_names", Mock(return_value=["", "Realtek"])),
        ):
            patcher = patch(target, value)
            patcher.start()
            self.addCleanup(patcher.stop)
        self.stt = STT()
        self.stt._noise_floor_rms = 100
        self.audio = sr.AudioData(pcm(200, 0.36) + pcm(0, 0.6), 16000, 2)

    def test_quiet_short_answer_survives_trailing_silence(self):
        self.assertLess(self.stt._compute_rms(self.audio), 150)
        self.assertTrue(self.stt._passes_voice_gate(self.audio))

    def test_background_noise_and_silence_are_rejected(self):
        for level in (0, 100):
            self.assertFalse(self.stt._passes_voice_gate(sr.AudioData(pcm(level, 1), 16000, 2)))

    def test_selects_default_with_signal_before_louder_alternative(self):
        with patch.object(self.stt, "_probe_microphone_rms", side_effect=[80, 2000]) as probe:
            self.assertEqual(self.stt._auto_select_microphone(), 1)
        probe.assert_called_once_with(1)

    def test_skips_default_returning_zeros(self):
        with patch.object(self.stt, "_probe_microphone_rms", side_effect=[0, 80]):
            self.assertEqual(self.stt._auto_select_microphone(), 6)

    def test_name_constraint_applies_to_fallback(self):
        self.stt._device_name_hint = "realtek"
        self.assertEqual(self.stt._microphone_candidates(), [1, 6])
        self.stt._device_name_hint = "missing"
        with self.assertRaisesRegex(OSError, "missing"):
            self.stt._microphone_candidates()

    def test_obsolete_output_index_falls_back_to_real_inputs(self):
        self.stt._device_index = self.stt._configured_device_index = 8
        self.assertEqual(self.stt._microphone_candidates(), [1, 6, 7])
        self.stt._strict_device_lock = True
        self.assertEqual(self.stt._microphone_candidates(), [8])

    def test_calibration_finishes_before_worker_starts(self):
        order = []
        self.stt._thread = Mock()
        self.stt._thread.is_alive.return_value = False
        self.stt._thread.ident = None
        self.stt._thread.start.side_effect = lambda: order.append("listen")
        with patch.object(self.stt, "_calibrate_microphone", side_effect=lambda: order.append("calibrate")):
            self.stt.start()
        self.stt.stop()
        self.assertEqual(order, ["calibrate", "listen"])

    def test_silent_calibration_tries_next_input(self):
        with patch("app.audio.stt.Microphone", side_effect=lambda **kw: MemoryMicrophone(pcm(80, 4))), \
                patch.object(self.stt, "_measure_ambient_rms", side_effect=[0, 80]):
            self.stt._calibrate_microphone()
        self.assertTrue(self.stt._calibrated)
        self.assertEqual(self.stt._device_index, 6)

    def test_noise_refresh_does_not_open_or_consume_microphone(self):
        self.stt._rec.energy_threshold = 120
        with patch("app.audio.stt.Microphone") as microphone:
            self.stt._refresh_noise_floor()
        microphone.assert_not_called()
        self.assertEqual(self.stt._noise_floor_rms, 80)

    def test_recorded_long_phrase_survives_timeout_on_extension(self):
        first = sr.AudioData(pcm(200, 16), 16000, 2)
        with patch.object(self.stt._rec, "listen", side_effect=[first, sr.WaitTimeoutError()]):
            self.assertIs(self.stt._record_phrase(Mock()), first)

    def test_pause_inside_request_keeps_both_parts(self):
        first, second = pcm(600, 0.5), pcm(250, 0.5)
        self.stt._rec.energy_threshold = 150
        source = MemoryMicrophone(pcm(40, 0.3) + first + pcm(0, 1.5) + second + pcm(0, 2))
        result = self.stt._record_phrase(source)
        self.assertIn(first, result.frame_data)
        self.assertIn(second, result.frame_data)
        self.assertTrue(self.stt._rec.dynamic_energy_threshold)
        self.assertFalse(self.stt.is_capturing())

    def test_loud_start_does_not_hide_softer_end_of_phrase(self):
        self.stt._rec.energy_threshold = 150
        loud, soft = pcm(3000, 2), pcm(300, 2)
        source = MemoryMicrophone(pcm(40, 0.3) + loud + soft + pcm(0, 2))
        result = self.stt._record_phrase(source)
        self.assertIn(soft, result.frame_data)
        self.assertLess(self.stt._rec.energy_threshold, 300)

    def test_denoised_voice_can_pass_when_raw_noise_floor_was_too_high(self):
        self.stt._noise_floor_rms = 1000
        clean = sr.AudioData(pcm(0, 0.5) + pcm(150, 0.4) + pcm(0, 0.5), 16000, 2)
        self.assertFalse(self.stt._passes_voice_gate(self.audio))
        self.assertTrue(self.stt._has_clean_voice(clean, self.audio))
        self.assertFalse(self.stt._has_clean_voice(sr.AudioData(pcm(50, 1), 16000, 2), self.audio))

    def test_denoiser_does_not_replace_voice_with_silence(self):
        self.config.denoise_enabled = True
        with patch("app.audio.stt.denoiser.enhance_pcm16", return_value=pcm(0, 0.96)):
            self.assertIs(self.stt._denoise(self.audio), self.audio)

    def test_tts_during_transcription_keeps_the_already_recorded_message(self):
        def recognize(*args, **kwargs):
            self.stt.enable_listening(False)
            return "Quiero leer mi documento"

        events = []

        def publish(event, **data):
            events.append((event, data))
            if event == "stt:text":
                self.stt.stop()

        self.stt._running.set()
        self.stt._listen_enabled.set()
        with patch("app.audio.stt.Microphone"), \
                patch("app.audio.stt.event_bus.publish", side_effect=publish), \
                patch.object(self.stt, "_record_phrase", return_value=self.audio), \
                patch.object(self.stt._rec, "recognize_google", side_effect=recognize):
            self.stt._loop()
        self.assertIn(("stt:text", {"text": "Quiero leer mi documento", "confidence": None}), events)

    def test_short_answer_reaches_text_event_without_recalibration(self):
        events = []

        def publish(event, **data):
            events.append((event, data))
            if event == "stt:text":
                self.stt.stop()

        source = MemoryMicrophone(pcm(50, 0.3) + pcm(300, 0.36) + pcm(0, 1.2))
        self.stt._rec.energy_threshold = 150
        self.stt._running.set()
        self.stt._listen_enabled.set()
        with patch("app.audio.stt.Microphone", return_value=source), \
                patch("app.audio.stt.event_bus.publish", side_effect=publish), \
                patch.object(self.stt._rec, "adjust_for_ambient_noise") as calibrate, \
                patch.object(self.stt._rec, "recognize_google", return_value="si") as google:
            self.stt._loop()
        calibrate.assert_not_called()
        google.assert_called_once()
        self.assertIn(("stt:text", {"text": "si", "confidence": None}), events)

    def test_tts_pause_discards_inflight_audio(self):
        def record(source):
            self.stt.enable_listening(False)
            self.stt.enable_listening(True)
            self.stt.stop()
            return self.audio

        self.stt._running.set()
        self.stt._listen_enabled.set()
        with patch("app.audio.stt.Microphone"), \
                patch.object(self.stt, "_record_phrase", side_effect=record), \
                patch.object(self.stt._rec, "recognize_google") as google:
            self.stt._loop()
        google.assert_not_called()

    def test_network_error_is_reported_to_ui(self):
        with patch.object(self.stt._rec, "recognize_google", side_effect=sr.RequestError("offline")), \
                patch("app.audio.stt.event_bus.publish") as publish:
            self.assertEqual(self.stt._recognize_with_google(self.audio), ("", None))
        messages = [call.kwargs.get("text", "") for call in publish.call_args_list]
        self.assertTrue(any("conexion a internet" in message for message in messages))
        self.assertTrue(self.stt._recognition_failed)
        self.assertEqual(self.stt._rec.operation_timeout, 10)

    def test_unrecognized_filtered_audio_retries_original_even_after_prior_warning(self):
        self.stt._last_error_message = "No se detecta tu voz"
        self.stt._running.set()
        self.stt._listen_enabled.set()
        cleaned = sr.AudioData(pcm(1, 1), 16000, 2)

        def publish(event, **data):
            if event == "stt:text":
                self.stt.stop()

        with patch("app.audio.stt.Microphone"), \
                patch("app.audio.stt.event_bus.publish", side_effect=publish), \
                patch.object(self.stt, "_record_phrase", return_value=self.audio), \
                patch.object(self.stt, "_denoise", return_value=cleaned), \
                patch.object(self.stt._rec, "recognize_google", side_effect=[sr.UnknownValueError(), "hola"]) as google:
            self.stt._loop()
        self.assertEqual([call.args[0] for call in google.call_args_list], [cleaned, self.audio])


class MicrophoneTests(unittest.TestCase):
    def test_open_failure_preserves_original_error_and_releases_driver(self):
        microphone = Microphone.__new__(Microphone)
        microphone.stream = None
        microphone.device_index = 1
        microphone.format = 8
        microphone.SAMPLE_RATE = 16000
        microphone.CHUNK = 1024
        microphone.pyaudio_module = Mock()
        driver = microphone.pyaudio_module.PyAudio.return_value
        driver.get_device_info_by_index.return_value = {"maxInputChannels": 1}
        driver.open.side_effect = OSError("device unavailable")
        with self.assertRaisesRegex(OSError, "device unavailable"):
            with microphone:
                self.fail("No se debe entrar con un stream vacio")
        driver.terminate.assert_called_once()

    def test_device_list_filters_output_and_loopback_and_prioritizes_default(self):
        devices = [
            {"name": "USB Audio", "maxInputChannels": 1, "hostApi": 0},
            {"name": "Speakers", "maxInputChannels": 0, "hostApi": 0},
            {"name": "Stereo Mix", "maxInputChannels": 2, "hostApi": 0},
            {"name": "Realtek", "maxInputChannels": 2, "hostApi": 0},
        ]
        pa = Mock(paWASAPI=13, paMME=2)
        driver = pa.PyAudio.return_value
        driver.get_default_input_device_info.return_value = {"index": 3}
        driver.get_device_count.return_value = len(devices)
        driver.get_device_info_by_index.side_effect = devices.__getitem__
        driver.get_host_api_info_by_index.return_value = {"type": 2}
        with patch("speech_recognition.Microphone.get_pyaudio", return_value=pa):
            self.assertEqual([d["index"] for d in input_devices()], [3, 0])
        driver.terminate.assert_called_once()


if __name__ == "__main__":
    unittest.main()
