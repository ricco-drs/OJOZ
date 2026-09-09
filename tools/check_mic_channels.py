"""
Diagnostico puntual: compara pedir el microfono en mono directo (como hace
hoy la app) contra pedirlo en estereo nativo y mezclar los canales a mano.

Si el driver de Windows no reduce bien 2 canales a 1, la version "mono
directo" sale distorsionada (zumbido, cruces por cero muy altos) mientras que
la version "estereo mezclado a mano" suena limpia. Eso confirma un problema
de negociacion de canales en vez de ruido de fondo o ganancia.

Ejecutar desde el directorio padre de app/ con:
    python -m app.tools.check_mic_channels
"""
from __future__ import annotations

import audioop
import time
import wave
from pathlib import Path

from app.audio.stt import STT
from app.config.settings import vision

_SALIDA = vision.data_dir / "audio_check"


def _grabar(pa, device_index: int, channels: int, rate: int, seconds: float) -> bytes:
    chunk = 1024
    stream = pa.open(
        input_device_index=device_index,
        channels=channels,
        format=pa.get_format_from_width(2),
        rate=rate,
        frames_per_buffer=chunk,
        input=True,
    )
    frames = []
    try:
        for _ in range(int(rate / chunk * seconds)):
            frames.append(stream.read(chunk, exception_on_overflow=False))
    finally:
        stream.stop_stream()
        stream.close()
    return b"".join(frames)


def _guardar(nombre: str, raw: bytes, rate: int) -> None:
    _SALIDA.mkdir(parents=True, exist_ok=True)
    with wave.open(str(_SALIDA / nombre), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(rate)
        wf.writeframes(raw)


def _stats(raw: bytes, rate: int) -> str:
    rms = audioop.rms(raw, 2)
    peak = audioop.max(raw, 2)
    import struct
    samples = struct.unpack("<%dh" % (len(raw) // 2), raw)
    zcr = sum(1 for i in range(1, len(samples)) if (samples[i - 1] < 0) != (samples[i] < 0))
    dur = len(samples) / rate
    return f"rms={rms} peak={peak} zcr={zcr/dur:.0f}/s dur={dur:.2f}s"


def main() -> None:
    stt = STT()
    device_index = stt._device_index
    print(f"Usando microfono idx={device_index}, nombre='{stt._mic_name}'")

    import speech_recognition as sr
    pa = sr.Microphone.get_pyaudio().PyAudio()
    try:
        info = pa.get_device_info_by_index(device_index)
        rate = int(info["defaultSampleRate"])
        max_ch = int(info["maxInputChannels"])
        print(f"Canales maximos del dispositivo: {max_ch}, sample rate: {rate}")

        input("\nPresiona ENTER y luego habla una frase normal durante la grabacion (mono directo)...")
        print("Grabando 3s en MONO DIRECTO (como hace la app hoy)...")
        mono = _grabar(pa, device_index, 1, rate, 3.0)
        _guardar("mono_directo.wav", mono, rate)
        print(f"  mono_directo.wav -> {_stats(mono, rate)}")

        if max_ch >= 2:
            time.sleep(0.5)
            input("\nPresiona ENTER y repite la MISMA frase (esteroo mezclado a mano)...")
            print("Grabando 3s en ESTEREO NATIVO...")
            stereo = _grabar(pa, device_index, 2, rate, 3.0)
            # Downmix manual: promedio de canal izquierdo y derecho.
            downmixed = audioop.tomono(stereo, 2, 0.5, 0.5)
            _guardar("mono_downmixed.wav", downmixed, rate)
            print(f"  mono_downmixed.wav -> {_stats(downmixed, rate)}")
        else:
            print("El dispositivo solo soporta 1 canal; no aplica la prueba estereo.")

        print(f"\nArchivos guardados en: {_SALIDA}")
        print("Escucha mono_directo.wav y mono_downmixed.wav y compara cual suena limpio.")
    finally:
        pa.terminate()


if __name__ == "__main__":
    main()
