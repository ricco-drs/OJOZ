"""
Diagnostico del canal de audio: microfono, ruido ambiente y supresion de ruido.

Pensado para calibrar el asistente en el lugar donde se va a usar. Mide el ruido
de fondo real, calcula si la voz supera el umbral exigido y deja una grabacion
comparativa (antes y despues del filtrado) para escuchar el resultado.

Ejecutar desde el directorio padre de app/ con:
    python -m app.tools.check_audio
"""
from __future__ import annotations

import audioop
import time
import wave
from pathlib import Path

import speech_recognition as sr

from app.audio.denoise import TARGET_SAMPLE_RATE, denoiser
from app.config.settings import stt as cfg, vision

_SALIDA = vision.data_dir / "audio_check"


def _titulo(texto: str) -> None:
    print(f"\n{texto}")
    print("-" * len(texto))


def _listar_microfonos() -> None:
    _titulo("Microfonos detectados")
    try:
        nombres = sr.Microphone.list_microphone_names() or []
    except Exception as exc:
        print(f"  No se pudieron listar: {exc}")
        return

    if not nombres:
        print("  Ninguno. Revisa que el microfono este conectado y habilitado.")
        return

    for i, nombre in enumerate(nombres):
        marca = "  <- configurado" if i == cfg.device_index else ""
        print(f"  [{i}] {nombre}{marca}")

    if cfg.device_index is None:
        print("\n  device_index=None: se usa el predeterminado del sistema.")
        print("  Para fijar uno, edita STTConfig.device_index en config/settings.py.")


def _medir_ruido(rec: sr.Recognizer) -> float | None:
    """Mide el nivel de ruido de fondo con el micrografo en silencio."""
    _titulo("Ruido ambiente")
    print("  Guarda silencio durante 3 segundos...")
    time.sleep(1.0)

    try:
        with sr.Microphone(device_index=cfg.device_index) as source:
            rec.adjust_for_ambient_noise(source, duration=3.0)
            lecturas = []
            for _ in range(int((source.SAMPLE_RATE / source.CHUNK) * 1.5)):
                buf = source.stream.read(source.CHUNK)
                if not buf:
                    break
                lecturas.append(float(audioop.rms(buf, source.SAMPLE_WIDTH)))
    except Exception as exc:
        print(f"  ERROR al abrir el microfono: {exc}")
        return None

    if not lecturas:
        print("  No se pudo leer audio del microfono.")
        return None

    lecturas.sort()
    piso = lecturas[len(lecturas) // 2]
    print(f"  RMS del ruido de fondo : {piso:.1f}")
    print(f"  energy_threshold        : {rec.energy_threshold:.1f}")

    exigido = min(max(cfg.min_rms, piso * cfg.snr_min_ratio), cfg.max_required_rms)
    print(f"  Tu voz debe superar RMS : {exigido:.1f}  (ajustado solo a este ruido)")

    if piso < 60:
        print("  Entorno silencioso.")
    elif piso < 200:
        print("  Ruido moderado: el asistente ya subio el liston por su cuenta.")
    else:
        print("  MUCHO RUIDO: se recomienda un microfono de diadema.")
    return piso


def _probar_voz(rec: sr.Recognizer, piso: float | None) -> sr.AudioData | None:
    """Graba una frase y comprueba si superaria el filtro anti-ruido."""
    _titulo("Prueba de voz")
    print("  Di una frase normal, por ejemplo: 'Hola OJOZ, lee este documento'.")
    print("  Escuchando...")

    try:
        with sr.Microphone(device_index=cfg.device_index) as source:
            audio = rec.listen(source, timeout=8.0, phrase_time_limit=6.0)
    except sr.WaitTimeoutError:
        print("  No se detecto voz. Habla mas fuerte o acerca el microfono.")
        return None
    except Exception as exc:
        print(f"  ERROR al grabar: {exc}")
        return None

    rms = float(audioop.rms(audio.get_raw_data(), audio.sample_width))
    exigido = min(
        max(cfg.min_rms, (piso or cfg.min_rms) * cfg.snr_min_ratio),
        cfg.max_required_rms,
    )

    print(f"  RMS de tu voz : {rms:.1f}")
    print(f"  Minimo exigido: {exigido:.1f}")

    if rms >= exigido:
        margen = rms / exigido
        print(f"  ACEPTADA (margen x{margen:.1f}).")
    else:
        print("  RECHAZADA: el asistente la descartaria como ruido.")
        print("  Soluciones: acercar el microfono, hablar mas fuerte, o bajar")
        print(f'  el umbral con  $env:OJOZ_SNR_RATIO = "{max(1.5, cfg.snr_min_ratio - 1):.1f}"')

    return audio


def _probar_denoise(audio: sr.AudioData | None) -> None:
    """Aplica el modelo de IA y guarda las dos versiones para compararlas."""
    _titulo("Supresion de ruido (DTLN)")

    if not denoiser.warmup():
        print("  El modelo NO esta disponible. El audio se enviaria sin filtrar.")
        print("  Revisa que exista assets/models/dtln/ y que onnxruntime este instalado.")
        return
    print("  Modelo cargado correctamente.")

    if audio is None:
        print("  Sin grabacion que procesar.")
        return

    raw = audio.get_raw_data(convert_rate=TARGET_SAMPLE_RATE, convert_width=2)
    inicio = time.time()
    limpio = denoiser.enhance_pcm16(raw, TARGET_SAMPLE_RATE)
    tardanza = time.time() - inicio

    if limpio is None:
        print("  El audio no se proceso (demasiado corto o largo).")
        return

    duracion = len(raw) / 2 / TARGET_SAMPLE_RATE
    print(f"  Procesado en {tardanza:.2f}s  ({duracion / max(tardanza, 1e-6):.0f}x tiempo real)")

    antes = float(audioop.rms(raw, 2))
    despues = float(audioop.rms(limpio, 2))
    print(f"  RMS antes={antes:.1f}  despues={despues:.1f}")

    _SALIDA.mkdir(parents=True, exist_ok=True)
    for nombre, datos in (("original.wav", raw), ("filtrado.wav", limpio)):
        ruta = _SALIDA / nombre
        with wave.open(str(ruta), "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(TARGET_SAMPLE_RATE)
            wf.writeframes(datos)

    print(f"\n  Grabaciones guardadas en: {_SALIDA}")
    print("  Escucha original.wav y filtrado.wav para comparar.")


def main() -> None:
    print("=" * 60)
    print("  OJOZ - Diagnostico de audio")
    print("=" * 60)

    _titulo("Configuracion de escucha")
    print(f"  energy_boost={cfg.energy_boost}  (margen sobre el ruido medido)")
    print(f"  snr_min_ratio={cfg.snr_min_ratio}  (cuanto debe superar la voz al ruido)")
    print(f"  limites: energia [{cfg.min_energy_threshold}, {cfg.max_energy_threshold}]"
          f"  rms [{cfg.min_rms}, {cfg.max_required_rms}]")
    print(f"  supresion de ruido: {'activada' if cfg.denoise_enabled else 'desactivada'}")
    print(f"  el ruido se vuelve a medir cada {cfg.recalibrate_every_s:.0f}s durante el uso")

    _listar_microfonos()

    rec = sr.Recognizer()
    rec.pause_threshold = cfg.pause_threshold
    rec.dynamic_energy_threshold = cfg.dynamic_energy_threshold

    piso = _medir_ruido(rec)
    audio = _probar_voz(rec, piso)
    _probar_denoise(audio)

    print("\n" + "=" * 60)
    print("  Fin del diagnostico")
    print("=" * 60)


if __name__ == "__main__":
    main()
