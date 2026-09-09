import logging
import sys


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

    # No se reenvian logs al chat (event_bus): son mensajes tecnicos para
    # diagnostico, y reenviarlos hacia el chat/voz los hacia aparecer como si
    # OJOZ los hubiera dicho, salvo que coincidieran por casualidad con la
    # lista de textos a ocultar en la UI. Los avisos que si deben llegar al
    # chat ya se publican explicitamente con event_bus.publish("ui:print", ...)
    # en el punto exacto donde tienen sentido.

    return logger

logger = setup_logging()
