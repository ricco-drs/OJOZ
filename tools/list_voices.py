"""
Muestra en JSON las voces de pyttsx3 disponibles en el sistema.

Util para rellenar TTSConfig.voice en config/settings.py.
Ejecutar con: python -m app.tools.list_voices
"""
import json

import pyttsx3


def main() -> None:
    engine = pyttsx3.init()
    try:
        voices = engine.getProperty("voices")
        info = []
        for i, v in enumerate(voices):
            try:
                langs = getattr(v, "languages", None)
            except Exception:
                langs = None
            info.append(
                {
                    "index": i,
                    "id": getattr(v, "id", None),
                    "name": getattr(v, "name", None),
                    "languages": langs,
                }
            )
        print(json.dumps(info, ensure_ascii=False, indent=2))
    finally:
        engine.stop()


if __name__ == "__main__":
    main()
