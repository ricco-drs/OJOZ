# app/vision/scene.py
"""
Descripcion del entorno usando Claude Vision.

A diferencia de OCR/dinero/vencimiento, describir una escena completa no es
una tarea de extraer un dato puntual sino de "entender" la imagen en general,
por eso no tiene un respaldo local: sin ANTHROPIC_API_KEY, simplemente no esta
disponible (se avisa por voz en el llamador).
"""
from __future__ import annotations

import base64
import os
import time
from typing import Optional, Tuple

import cv2

from app.config.settings import llm as llm_config, vision
from app.vision.camera_service import frames_for

_PROMPT = (
    "Describe brevemente en espanol, en dos o tres frases, lo que se ve en "
    "esta imagen, como si se lo explicaras a una persona con discapacidad "
    "visual que necesita desenvolverse en el lugar. Prioriza siempre: 1) si "
    "hay personas presentes y donde estan ubicadas respecto a la camara, 2) "
    "cualquier obstaculo cercano que pueda representar un riesgo al caminar "
    "(escalones, objetos en el piso, muebles, puertas), y 3) el resto de "
    "objetos relevantes de la escena. No agregues nada mas alla de la "
    "descripcion."
)


def _claude_vision_scene(frame) -> Optional[str]:
    from app.utils.logger import logger

    try:
        import anthropic
    except ImportError:
        return None

    ok, buffer = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 90])
    if not ok:
        return None

    try:
        client = anthropic.Anthropic()
        response = client.messages.create(
            model=llm_config.model,
            max_tokens=300,
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
                        {"type": "text", "text": _PROMPT},
                    ],
                }
            ],
        )
    except Exception as exc:
        logger.warning(f"Claude Vision fallo al describir la escena: {exc}")
        return None

    text = "".join(
        b.text for b in response.content if getattr(b, "type", None) == "text"
    ).strip()
    return text or None


def describe_scene_best_frame(seconds: float = 5.0) -> Tuple[bool, Optional[str]]:
    """
    Abre la camara, deja que se estabilice el enfoque y describe lo que ve.

    Sin ANTHROPIC_API_KEY no abre la camara: devuelve (False, None) de una vez.
    """
    if not os.environ.get("ANTHROPIC_API_KEY", "").strip():
        return False, None

    frame = None
    t0 = time.time()
    for current in frames_for(seconds):
        frame = current
        if getattr(vision, "show_preview", False):
            show = frame.copy()
            time_left = int(seconds - (time.time() - t0)) + 1
            cv2.putText(
                show,
                f"Describiendo el entorno - {time_left}s",
                (40, 60),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.0,
                (0, 255, 0),
                3,
                cv2.LINE_AA,
            )
            cv2.imshow("Descripcion de escena", show)
            if cv2.waitKey(1) & 0xFF == 27:
                break
    if getattr(vision, "show_preview", False):
        cv2.destroyAllWindows()

    if frame is None:
        return False, None

    text = _claude_vision_scene(frame)
    if not text:
        return False, None
    return True, text
