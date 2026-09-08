from __future__ import annotations
import datetime
from colorama import Fore, Style, init
from app.core.event_bus import event_bus

# Inicializa color en consola
init(autoreset=True)

def _stamp() -> str:
    return datetime.datetime.now().strftime("%H:%M:%S")

def _on_ui_print(role: str, text: str):
    if role == "app/tts":
        prefix = Fore.CYAN + "[APP]" + Style.RESET_ALL
    elif role == "user":
        prefix = Fore.GREEN + "[TÚ]" + Style.RESET_ALL
    else:
        prefix = Fore.MAGENTA + "[SYS]" + Style.RESET_ALL
    print(f"{Style.DIM}{_stamp()}{Style.RESET_ALL} {prefix} {text}", flush=True)

def _on_state(from_state: str, to: str):
    print(Style.DIM + f"{_stamp()} (estado: {from_state} → {to})" + Style.RESET_ALL, flush=True)

def _on_tts_start():
    print(Style.DIM + f"{_stamp()} [TTS] start" + Style.RESET_ALL, flush=True)

def _on_tts_end():
    print(Style.DIM + f"{_stamp()} [TTS] end" + Style.RESET_ALL, flush=True)

def _on_stt_start():
    print(Style.DIM + f"{_stamp()} [STT] escuchando..." + Style.RESET_ALL, flush=True)

def _on_stt_end():
    print(Style.DIM + f"{_stamp()} [STT] fin de escucha" + Style.RESET_ALL, flush=True)


def _on_stt_text(text: str, confidence=None):
    _on_ui_print(role="user", text=text)

def init_console_view():
    # Suscripciones a eventos del bus
    event_bus.subscribe("ui:print", _on_ui_print)
    event_bus.subscribe("ctrl:state", _on_state)
    event_bus.subscribe("tts:start", lambda: _on_tts_start())
    event_bus.subscribe("tts:end",   lambda: _on_tts_end())
    event_bus.subscribe("stt:start", lambda: _on_stt_start())
    event_bus.subscribe("stt:end",   lambda: _on_stt_end())
    event_bus.subscribe("stt:text", _on_stt_text)
