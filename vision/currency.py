# app/vision/currency.py
from __future__ import annotations

import base64
import os
import time
from typing import List, Optional, Tuple

import cv2
import numpy as np

from app.config.settings import llm as llm_config, vision
from app.vision.camera import show_preview
from app.vision.camera_service import frames_for

# Denominaciones reales del sol peruano (PEN).
# Monedas: 10, 20 y 50 centimos (0.10/0.20/0.50 soles) y 1, 2, 5 soles.
# Billetes: 10, 20, 50, 100 y 200 soles.
KNOWN_BILLS = {0.10, 0.20, 0.50, 1, 2, 5, 10, 20, 50, 100, 200}

# Un item detectado: ("billete"|"moneda", valor)
CurrencyItem = Tuple[str, float]


def _claude_vision_currency(frame: np.ndarray) -> Tuple[bool, List[CurrencyItem]]:
    """
    Identifica TODOS los billetes y monedas de sol peruano visibles en la
    imagen a la vez (puede haber varios), pidiendole a Claude que reconozca
    el diseno completo de cada uno (color, tamano, motivos) en vez de leer
    un numero impreso con OCR o comparar contra plantillas.
    """
    from app.utils.logger import logger

    try:
        import anthropic
    except ImportError:
        return False, []

    ok, buffer = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 90])
    if not ok:
        return False, []

    try:
        client = anthropic.Anthropic()
        response = client.messages.create(
            model=llm_config.model,
            max_tokens=200,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image",
                            "source": {
                                "type": "base64",
                                "media_type": "image/jpeg",
                                "data": base64.b64encode(buffer.tobytes()).decode("ascii"),
                            },
                        },
                        {
                            "type": "text",
                            "text": (
                                "Esta foto puede mostrar uno o varios billetes y monedas de "
                                "sol peruano (PEN) a la vez, incluso varios del mismo valor. "
                                "Primero cuenta cuantos objetos fisicos distintos (billetes o "
                                "monedas) hay realmente en la foto, contando cada uno una sola "
                                "vez aunque se vea parcialmente superpuesto o reflejado; no "
                                "inventes copias de un mismo objeto ni lo cuentes dos veces. Si "
                                "dos monedas del mismo valor estan una al lado de la otra, son "
                                "dos objetos distintos y las dos cuentan. Luego, para cada "
                                "objeto, fijate primero en el numero o texto impreso en el (por "
                                "ejemplo '10 SOLES', '50 CENTIMOS', '2 SOLES') antes de guiarte "
                                "solo por el color o el tamano, ya que varias monedas peruanas "
                                "se parecen entre si. Las denominaciones validas son: monedas "
                                "de 10 centimos, 20 centimos, 50 centimos, 1 sol, 2 soles y 5 "
                                "soles; billetes de 10, 20, 50, 100 y 200 soles. Responde "
                                "UNICAMENTE con una linea por cada objeto fisico, en este "
                                "formato exacto: tipo|VALOR (tipo es billete o moneda; VALOR en "
                                "soles, usa 0.1, 0.2 o 0.5 para centimos). Por ejemplo, si hay "
                                "un billete de 20 soles y dos monedas de 50 centimos:\n"
                                "billete|20\nmoneda|0.5\nmoneda|0.5\n"
                                "No numeres las lineas, no agregues explicaciones ni texto "
                                "adicional. Si algun objeto no es claramente un billete o "
                                "moneda peruano, o no puedes leer su valor con certeza, no lo "
                                "incluyas: es mejor omitirlo que adivinar. Si no identificas "
                                "ninguno con certeza, responde unicamente: NINGUNO"
                            ),
                        },
                    ],
                }
            ],
        )
    except Exception as exc:
        logger.warning(f"Claude Vision fallo para dinero: {exc}")
        return False, []

    raw = "".join(b.text for b in response.content if getattr(b, "type", None) == "text").strip()
    if not raw or raw.upper() == "NINGUNO":
        return False, []

    items: List[CurrencyItem] = []
    for line in raw.splitlines():
        line = line.strip().lstrip("-•").strip()
        if not line or "|" not in line:
            continue
        tipo, valor_str = line.split("|", 1)
        try:
            valor = float("".join(ch for ch in valor_str if ch.isdigit() or ch == "."))
        except ValueError:
            continue
        if valor not in KNOWN_BILLS:
            continue
        tipo = tipo.strip().lower()
        if tipo not in ("billete", "moneda"):
            tipo = "billete" if valor >= 10 else "moneda"
        items.append((tipo, valor))

    if not items:
        logger.debug(f"Respuesta de Claude Vision no reconocida: {raw!r}")
        return False, []

    return True, items


def _should_show_debug_preview() -> bool:
    return bool(getattr(vision, "show_preview", False))


def detect_currency_best_frame(seconds: float = 8.0) -> Tuple[bool, List[CurrencyItem]]:
    """
    Abre la camara, muestra una vista previa limpia (sin recuadros guia) y
    manda la mejor foto a Claude Vision, que identifica todos los billetes y
    monedas visibles a la vez. Sin ANTHROPIC_API_KEY, o si Claude Vision no
    logra identificar nada con certeza, devuelve una lista vacia.
    """
    if not os.environ.get("ANTHROPIC_API_KEY", "").strip():
        return False, []

    frame = None
    t0 = time.time()
    for current in frames_for(min(seconds, 15.0)):
        frame = current
        if _should_show_debug_preview():
            show = current.copy()
            time_left = int(seconds - (time.time() - t0)) + 1
            cv2.putText(
                show,
                f"Muestre el billete o moneda - {time_left}s",
                (40, 60),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.1,
                (0, 255, 0),
                3,
                cv2.LINE_AA,
            )
            show_preview("Deteccion de Dinero", show)
            if cv2.waitKey(1) & 0xFF == 27:
                break
    if _should_show_debug_preview():
        cv2.destroyAllWindows()

    if frame is None:
        return False, []

    return _claude_vision_currency(frame)
