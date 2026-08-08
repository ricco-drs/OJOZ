import logging
import sys


class EventBusHandler(logging.Handler):
    """Handler que envía logs al event_bus para mostrarlos en la UI"""
    def __init__(self):
        super().__init__()
        # Importación lazy para evitar circular imports
        self.event_bus = None
    
    def emit(self, record):
        try:
            if self.event_bus is None:
                from app.core.event_bus import event_bus
                self.event_bus = event_bus
            
            msg = self.format(record)
            # Publicar solo logs INFO y superiores al chat
            if record.levelno >= logging.INFO:
                self.event_bus.publish("ui:print", role="sys", text=f"{msg}")
        except Exception:
            pass  # No queremos que el logger rompa la app


def setup_logging(level=logging.DEBUG):
    logger = logging.getLogger("app")
    if logger.handlers:
        return logger  # evitar handlers duplicados si se llama dos veces

    logger.setLevel(level)
    
    # Handler para terminal
    console_handler = logging.StreamHandler(sys.stdout)
    console_formatter = logging.Formatter("[%(asctime)s] [%(levelname)s] %(message)s", "%H:%M:%S")
    console_handler.setFormatter(console_formatter)
    logger.addHandler(console_handler)
    
    # Handler para UI (event_bus)
    ui_handler = EventBusHandler()
    ui_formatter = logging.Formatter("%(message)s")  # Más simple para el chat
    ui_handler.setFormatter(ui_formatter)
    ui_handler.setLevel(logging.INFO)  # Solo INFO y superiores van al chat
    logger.addHandler(ui_handler)
    
    return logger

logger = setup_logging()
