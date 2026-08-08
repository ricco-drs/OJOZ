"""
Punto de entrada para OJOZ con interfaz Flet.

Ejecutar desde la carpeta padre del proyecto:
    python -m app.main_flet
"""
from __future__ import annotations

import sys
import threading
from pathlib import Path

# Permite ejecutar también como script suelto (python app/main_flet.py):
# añade la carpeta padre al path para que los imports 'app.*' resuelvan.
_PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

try:
    import flet as ft
except ImportError:
    print("Error: flet no está instalado.")
    print("Instala las dependencias con: pip install -r requirements.txt")
    sys.exit(1)

from app.audio.stt import STT
from app.audio.tts import TTS
from app.core.controller import Controller
from app.db.engine import init_db
from app.ui.flet_app import OJOZApp
from app.utils.logger import logger


def main(page: ft.Page):
    """Inicializa la interfaz y carga servicios en background para mejorar el arranque."""
    app = OJOZApp(controller=None)
    app.build(page)
    app.update_status_badge("Iniciando servicios...", "#F2B33D")
    app.show_bootstrap_message("Preparando OJOZ, por favor espera un momento...")

    def _bootstrap():
        try:
            app.update_status_badge("Inicializando base de datos...", "#F2B33D")
            init_db()
            app.show_bootstrap_message("Base de datos lista.")

            app.update_status_badge("Cargando motor de voz...", "#F2B33D")
            tts = TTS()
            tts.warmup()
            app.show_bootstrap_message("Motor de voz preparado.")

            app.update_status_badge("Configurando micrófono...", "#F2B33D")
            stt = STT()

            controller = Controller(tts=tts, stt=stt)
            app.attach_controller(controller)
            app.update_status_badge(app.current_user_name())
            app.show_bootstrap_message("Servicios listos, OJOZ te hablará en un instante.")

            controller.start()
        except Exception:
            logger.exception("Error al iniciar servicios")
            app.update_status_badge("Error al iniciar servicios", "#ff6b6b")
            app.show_bootstrap_message("Ocurrió un error durante el arranque. Revisa la consola.")

    threading.Thread(target=_bootstrap, daemon=True).start()


if __name__ == "__main__":
    # ft.app() quedo obsoleta en flet 0.80; requirements.txt fija 0.86.5.
    #
    # assets_dir se pasa absoluto a proposito: por defecto flet lo resuelve
    # contra sys.argv[0], asi que dependeria de desde donde se lance la app.
    # Las rutas de ft.Image(src=...) son relativas a esta carpeta.
    ft.run(main, assets_dir=str(Path(__file__).resolve().parent / "assets"))
