import time
from pathlib import Path
from threading import Event

# Carga las claves de API y variables opcionales desde un .env junto a este
# archivo (ver main_flet.py para el detalle); tiene que ir antes de importar
# app.config.settings (via los imports de abajo), que las lee al importarse.
try:
    from dotenv import load_dotenv

    load_dotenv(Path(__file__).resolve().parent / ".env")
except ImportError:
    pass

from app.ui.console_view import init_console_view
from app.audio.tts import TTS
from app.audio.stt import STT
from app.core.controller import Controller
from app.core.event_bus import event_bus
from app.utils.logger import logger

# === NUEVO (DB): inicialización de SQLite ===
from app.db.engine import init_db


def main():
    # 1) Inicializa la base de datos (crea app/data/db/app.sqlite3 y tablas si no existen)
    try:
        init_db()
        logger.info("SQLite inicializada correctamente.")
    except Exception as e:
        logger.error(f"Error al inicializar SQLite: {e!r}")

    # 2) Inicializa la vista de consola (suscrita al bus de eventos)
    init_console_view()

    # 3) Instancia motores de audio y controlador
    tts = TTS()
    tts.warmup()
    stt = STT()
    ctrl = Controller(tts, stt)

    # Evento para terminar el bucle principal cuando llegue "app:shutdown"
    stop = Event()

    def _on_shutdown():
        stop.set()

    # Suscripción al evento de apagado
    event_bus.subscribe("app:shutdown", _on_shutdown)

    # Arranca el flujo (saludo + activar STT con gate)
    ctrl.start()

    # Bucle principal controlado por evento
    try:
        while not stop.is_set():
            time.sleep(0.2)
    except KeyboardInterrupt:
        logger.info("Interrumpido por usuario. Saliendo...")
        stt.stop()
        tts.shutdown()


if __name__ == "__main__":
    main()
