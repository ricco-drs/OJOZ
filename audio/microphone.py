from __future__ import annotations

import speech_recognition as sr


class Microphone(sr.Microphone):
    """Conserva el error de PortAudio si no se puede abrir la entrada."""

    def __enter__(self):
        assert self.stream is None, "El microfono ya esta abierto"
        self.audio = self.pyaudio_module.PyAudio()
        try:
            info = (self.audio.get_default_input_device_info() if self.device_index is None
                    else self.audio.get_device_info_by_index(self.device_index))
            if info["maxInputChannels"] < 1:
                raise OSError(f"El dispositivo {self.device_index} no es una entrada de audio")
            self.stream = self.MicrophoneStream(self.audio.open(
                input_device_index=self.device_index, channels=1, format=self.format,
                rate=self.SAMPLE_RATE, frames_per_buffer=self.CHUNK, input=True,
            ))
        except Exception:
            self.audio.terminate()
            self.audio = None
            raise
        return self


def input_devices() -> list[dict]:
    """Entradas reales, con el dispositivo predeterminado primero."""
    pa = sr.Microphone.get_pyaudio()
    audio = pa.PyAudio()
    try:
        try:
            default_index = int(audio.get_default_input_device_info()["index"])
        except OSError:
            default_index = None
        devices = []
        for index in range(audio.get_device_count()):
            info = dict(audio.get_device_info_by_index(index))
            name = info.get("name", "").casefold()
            if info["maxInputChannels"] < 1:
                continue
            if any(word in name for word in (
                "mapper", "asignador", "controlador primario", "primary sound",
                "mezcla", "stereo mix", "output", "loopback", "altavoz", "speaker",
            )):
                continue
            info["index"] = index
            info["host_type"] = audio.get_host_api_info_by_index(info["hostApi"])["type"]
            devices.append(info)
        # WASAPI y MME antes de DirectSound y WDM-KS, que pueden abrir sin senal.
        order = {getattr(pa, "paWASAPI", 13): 0, getattr(pa, "paMME", 2): 1}
        devices.sort(key=lambda d: (
            d["index"] != default_index, order.get(d["host_type"], 2), d["index"],
        ))
        return devices
    finally:
        audio.terminate()
