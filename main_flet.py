"""
Punto de entrada para OJOZ con interfaz Flet.

Ejecutar desde la carpeta padre del proyecto:
    python -m app.main_flet
"""
from __future__ import annotations

import os
import signal
import sys
import threading
from pathlib import Path

# Permite ejecutar también como script suelto (python app/main_flet.py):
# añade la carpeta padre al path para que los imports 'app.*' resuelvan.
_PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

# Carga las claves de API y variables opcionales (ELEVENLABS_*, ANTHROPIC_API_KEY,
# GOOGLE_VISION_API_KEY, OJOZ_*) desde un .env junto a este archivo, para no
# tener que volver a escribirlas en cada terminal nueva. Tiene que ejecutarse
# ANTES de importar app.config.settings (mas abajo), que lee estas variables
# al importarse. Sin el archivo, o sin python-dotenv instalado, la app sigue
# funcionando igual, solo que las variables deben venir ya seteadas a mano.
try:
    from dotenv import load_dotenv

    load_dotenv(Path(__file__).resolve().parent / ".env")
except ImportError:
    pass

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

    def _on_disconnect(_e) -> None:
        # Se dispara cuando la ventana de escritorio se cierra (el cliente
        # Flutter se desconecta del servidor Python): en modo app de
        # escritorio, ft.run() a veces se queda esperando indefinidamente a
        # que el subproceso de la ventana termine (ver comentario mas abajo
        # sobre Ctrl+C), dejando la terminal colgada aunque la ventana ya se
        # haya cerrado. Forzar la salida aqui no depende de que ese await
        # alguna vez regrese.
        print("\nVentana cerrada, cerrando OJOZ...")
        os._exit(0)

    page.on_disconnect = _on_disconnect

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


def _force_exit(signum, frame) -> None:
    print("\nCerrando OJOZ...")
    os._exit(0)


if __name__ == "__main__":
    # ft.app() quedo obsoleta en flet 0.80; requirements.txt fija 0.86.5.
    #
    # assets_dir se pasa absoluto a proposito: por defecto flet lo resuelve
    # contra sys.argv[0], asi que dependeria de desde donde se lance la app.
    # Las rutas de ft.Image(src=...) son relativas a esta carpeta.
    #
    # ft.run() instala su propio manejador de SIGINT (Ctrl+C) por dentro,
    # pero en modo app de escritorio nunca llega a cerrar nada con la primera
    # pulsacion: la señal de terminacion que arma no se consulta en ese modo,
    # solo se restaura el comportamiento por defecto de Windows, asi que se
    # necesita una SEGUNDA pulsacion para que el sistema operativo mate el
    # proceso de golpe (sin limpieza). Como via confiable de una sola
    # pulsacion, se usa Ctrl+Break (SIGBREAK): es una señal distinta que Flet
    # no toca, asi que este manejador siempre queda activo.
    if hasattr(signal, "SIGBREAK"):
        signal.signal(signal.SIGBREAK, _force_exit)

    try:
        ft.run(main, assets_dir=str(Path(__file__).resolve().parent / "assets"))
    except KeyboardInterrupt:
        print("\nCerrando OJOZ...")
    finally:
        os._exit(0)
