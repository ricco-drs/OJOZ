from __future__ import annotations

import audioop

import speech_recognition as sr


class _DownmixedStream:
    """
    Envuelve el stream de PyAudio y mezcla a mono a mano cuando el
    dispositivo entrega estereo.

    Pedirle directamente 1 canal a un microfono cuyo formato nativo en
    Windows es estereo obliga al driver a reducir los canales por su cuenta,
    y en varios drivers WASAPI esa reduccion viene mal hecha: en vez de
    promediar izquierda y derecha, entrega las muestras intercaladas como si
    fueran una sola señal continua, lo que suena distorsionado sin importar
    el volumen o las mejoras de audio configuradas en Windows. Pedir el
    dispositivo en su cantidad de canales nativa y mezclar aqui evita esa
    conversion implicita.
    """

    def __init__(self, pyaudio_stream, channels: int, sample_width: int):
        self.pyaudio_stream = pyaudio_stream
        self.channels = channels
        self.sample_width = sample_width

    def read(self, size):
        data = self.pyaudio_stream.read(size, exception_on_overflow=False)
        if self.channels <= 1 or not data:
            return data
        return audioop.tomono(data, self.sample_width, 0.5, 0.5)

    def close(self):
        try:
            if not self.pyaudio_stream.is_stopped():
                self.pyaudio_stream.stop_stream()
        finally:
            self.pyaudio_stream.close()


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

            # audioop.tomono solo mezcla estereo; con mas canales (arreglos
            # de mas de 2 microfonos) se pide mono directo, como antes.
            native_channels = int(info.get("maxInputChannels") or 1)
            channels = native_channels if native_channels in (1, 2) else 1

            raw_stream = self.audio.open(
                input_device_index=self.device_index, channels=channels, format=self.format,
                rate=self.SAMPLE_RATE, frames_per_buffer=self.CHUNK, input=True,
            )
            self.stream = _DownmixedStream(raw_stream, channels, self.SAMPLE_WIDTH)
        except Exception:
            self.audio.terminate()
            self.audio = None
            raise
        return self


def input_devices() -> list[dict]:
    """
    Entradas reales, con el dispositivo predeterminado primero.

    Windows expone cada microfono fisico una vez POR cada API de audio
    (MME, DirectSound, WASAPI, WDM-KS), asi que sin filtrar aparece
    duplicado 3-4 veces. Ademas MME trunca los nombres a 31 caracteres, y
    WDM-KS (la mas antigua, de bajo nivel) suele exponer entradas basura:
    nombres vacios, referencias de recursos de Windows sin resolver, o el
    canal de manos-libres de un parlante Bluetooth registrado como si fuera
    un microfono. Aqui se descarta WDM-KS cuando hay otra via disponible, y
    se deduplica por nombre para quedarse con una sola entrada por microfono
    real, preferida por WASAPI > DirectSound > MME.
    """
    pa = sr.Microphone.get_pyaudio()
    audio = pa.PyAudio()
    try:
        try:
            default_index = int(audio.get_default_input_device_info()["index"])
        except OSError:
            default_index = None

        wasapi = getattr(pa, "paWASAPI", 13)
        directsound = getattr(pa, "paDirectSound", 1)
        mme = getattr(pa, "paMME", 2)
        wdm_ks = getattr(pa, "paWDMKS", 11)
        host_priority = {wasapi: 0, directsound: 1, mme: 2}

        devices = []
        for index in range(audio.get_device_count()):
            info = dict(audio.get_device_info_by_index(index))
            raw_name = (info.get("name") or "").strip()
            name = raw_name.casefold()
            if info["maxInputChannels"] < 1:
                continue
            if any(word in name for word in (
                "mapper", "asignador", "controlador primario", "primary sound",
                "mezcla", "stereo mix", "output", "loopback", "altavoz", "speaker",
            )):
                continue
            # Nombre vacio ("Input ()") o referencia de recurso de Windows
            # sin resolver ("Input (@System32\drivers\...;(JBL Flip 6))").
            bare = raw_name.split("(", 1)[-1].rstrip(")").strip()
            if not bare or raw_name.startswith("@") or "@system32" in name or "\\drivers\\" in name:
                continue
            info["index"] = index
            info["name"] = raw_name
            info["host_type"] = audio.get_host_api_info_by_index(info["hostApi"])["type"]
            devices.append(info)

        # WDM-KS solo se conserva si es la unica via disponible.
        if any(d["host_type"] != wdm_ks for d in devices):
            devices = [d for d in devices if d["host_type"] != wdm_ks]

        # Primero calidad de la API (asi sobrevive el nombre completo de
        # WASAPI/DirectSound, no el truncado de MME), y solo como
        # desempate dentro de la misma calidad, el predeterminado.
        devices.sort(key=lambda d: (
            host_priority.get(d["host_type"], 3),
            d["index"] != default_index,
            d["index"],
        ))

        # Deduplicar por nombre normalizado. MME trunca a 31 caracteres, asi
        # que una version es prefijo literal de la otra; se compara en ambos
        # sentidos para no depender de cual llego primero.
        seen: list[str] = []
        deduped = []
        for d in devices:
            key = d["name"].strip().casefold()
            if any(key == s or key.startswith(s) or s.startswith(key) for s in seen):
                continue
            seen.append(key)
            deduped.append(d)

        return deduped
    finally:
        audio.terminate()
