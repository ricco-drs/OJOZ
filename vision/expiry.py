# app/vision/expiry.py
from __future__ import annotations

import base64
import os
import re
import time
from datetime import date, datetime
from typing import Dict, Optional, Tuple

import cv2
import numpy as np
import pytesseract

from app.config.settings import configure_tesseract, llm as llm_config, vision
from app.vision.camera import show_preview
from app.vision.camera_service import frames_for

configure_tesseract()


def _claude_vision_expiry(frame: np.ndarray) -> Tuple[bool, Optional[str], Optional[bool], Optional[float]]:
    """
    Le pide a Claude que encuentre la fecha de vencimiento en el empaque.

    A diferencia de Tesseract + regex, Claude puede decidir cual fecha es la
    de vencimiento (no la de fabricacion ni un codigo de lote) y leer texto
    en relieve o formatos no previstos. El calculo de vencido/vigente se hace
    en Python contra la fecha de hoy, nunca lo decide el modelo.
    """
    from app.utils.logger import logger

    try:
        import anthropic
    except ImportError:
        return False, None, None, None

    ok, buffer = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 90])
    if not ok:
        return False, None, None, None

    try:
        client = anthropic.Anthropic()
        response = client.messages.create(
            model=llm_config.model,
            max_tokens=20,
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
                                "Esta foto muestra el empaque de un producto peruano. Busca la "
                                "fecha de vencimiento o caducidad, identificada con etiquetas "
                                "como VENC, VTO, F.V., FEC. VENC., CAD, CADUCIDAD, EXP, "
                                "'Consumir preferentemente antes de', Best Before o Use By. "
                                "Ignora otras fechas o codigos cercanos que NO son la de "
                                "vencimiento, como la fecha de fabricacion/elaboracion (ELAB, "
                                "FAB, FEC. FAB.) o el codigo de lote (LOTE, L:).\n"
                                "Los productos peruanos usan formatos variados; reconocelos "
                                "todos: DD/MM/AAAA, DD/MM/AA, DD-MM-AAAA, DD.MM.AAAA, sin "
                                "separadores (DDMMAA), con el mes en texto o abreviado "
                                "('15 MAR 2026', '15 DE MARZO DE 2026', '15MAR26'), o solo "
                                "mes y anio ('03/2026', 'MAR 2026') cuando el empaque no "
                                "imprime el dia exacto.\n"
                                "Responde UNICAMENTE en este formato, sin nada mas alrededor: "
                                "DD/MM/AAAA (por ejemplo 15/03/2026). Si la fecha encontrada "
                                "solo trae mes y anio (sin dia), usa el ULTIMO dia de ese mes "
                                "(el producto se considera vigente hasta el final del mes "
                                "impreso). Si no encuentras una fecha de vencimiento clara, "
                                "responde unicamente: NINGUNO"
                            ),
                        },
                    ],
                }
            ],
        )
    except Exception as exc:
        logger.warning(f"Claude Vision fallo para vencimiento: {exc}")
        return False, None, None, None

    raw = "".join(b.text for b in response.content if getattr(b, "type", None) == "text").strip()

    if not raw or raw.upper() == "NINGUNO":
        return False, None, None, None

    try:
        parsed = datetime.strptime(raw, "%d/%m/%Y").date()
    except ValueError:
        logger.debug(f"Fecha de Claude Vision no reconocida: {raw!r}")
        return False, None, None, None

    is_expired = parsed < date.today()
    return True, parsed.strftime("%d/%m/%Y"), is_expired, 92.0


# =============================================================
# Respaldo local (sin internet): OCR con Tesseract + patrones de
# fecha por regex. Mas limitado que Claude Vision (no puede "razonar"
# cual fecha es la de vencimiento si hay varias), pero es lo unico
# que sigue funcionando sin conexion.
# =============================================================

_KW = [
    "venc", "vence", "vencimiento", "vto", "cad", "caduca", "caducidad",
    "exp", "expira", "expiry", "best before", "use by", "consumir",
]

_MONTHS_ES = {
    "enero": 1, "ene": 1,
    "febrero": 2, "feb": 2,
    "marzo": 3, "mar": 3,
    "abril": 4, "abr": 4,
    "mayo": 5, "may": 5,
    "junio": 6, "jun": 6,
    "julio": 7, "jul": 7,
    "agosto": 8, "ago": 8,
    "septiembre": 9, "sept": 9, "sep": 9, "set": 9,
    "octubre": 10, "oct": 10,
    "noviembre": 11, "nov": 11,
    "diciembre": 12, "dic": 12,
}

_RE_NUMERIC = re.compile(r"(\d{1,2})[\/\-.](\d{1,2})[\/\-.](\d{2,4})")
_RE_PREFIX_NUMERIC = re.compile(
    r"(?:f\.?\s*v\.?\.?|venc\.?|vto\.?|cad\.?)\s*[:\-]?\s*(\d{1,2})[\/\-.](\d{1,2})[\/\-.](\d{2,4})",
    re.IGNORECASE,
)
_RE_COMPACT = re.compile(r"\b(\d{2})(\d{2})(\d{2})\b")  # DDMMAA sin separadores
_RE_MONTH_TEXT = re.compile(
    r"(\d{1,2})\s*(?:de\s*)?(" + "|".join(_MONTHS_ES) + r")\.?\s*(?:de\s*)?(\d{2,4})",
    re.IGNORECASE,
)
_RE_MONTH_YEAR = re.compile(
    r"\b(" + "|".join(_MONTHS_ES) + r")\.?\s*(\d{4})\b", re.IGNORECASE
)
_RE_MM_YYYY = re.compile(r"\b(\d{1,2})[\/\-.](\d{4})\b")


def _year_fix(y: int) -> int:
    if y < 100:
        return 2000 + y if y < 70 else 1900 + y
    return y


def _safe_date(d: int, m: int, y: int) -> Optional[date]:
    try:
        return date(y, m, d)
    except Exception:
        return None


def _last_day_of_month(m: int, y: int) -> Optional[date]:
    for d in (31, 30, 29, 28):
        dt = _safe_date(d, m, y)
        if dt:
            return dt
    return None


def _parse_date_from_text(raw_text: str) -> Optional[date]:
    """Prueba varios formatos de fecha, del mas especifico al mas ambiguo."""
    t = raw_text.lower()

    m = _RE_PREFIX_NUMERIC.search(t)
    if m:
        d, mo, y = m.groups()
        return _safe_date(int(d), int(mo), _year_fix(int(y)))

    m = _RE_MONTH_TEXT.search(t)
    if m:
        d, month_s, y = m.groups()
        month = _MONTHS_ES.get(month_s.lower())
        if month:
            return _safe_date(int(d), month, _year_fix(int(y)))

    # Con varias fechas numericas seguidas, la de vencimiento suele ser la
    # segunda (la primera suele ser fabricacion/elaboracion).
    matches = list(_RE_NUMERIC.finditer(t))
    if len(matches) >= 2:
        d, mo, y = matches[1].groups()
        dt = _safe_date(int(d), int(mo), _year_fix(int(y)))
        if dt:
            return dt
    if matches:
        d, mo, y = matches[0].groups()
        dt = _safe_date(int(d), int(mo), _year_fix(int(y)))
        if dt:
            return dt

    m = _RE_MONTH_YEAR.search(t)
    if m:
        month_s, y = m.groups()
        month = _MONTHS_ES.get(month_s.lower())
        if month:
            return _last_day_of_month(month, _year_fix(int(y)))

    m = _RE_MM_YYYY.search(t)
    if m:
        mo, y = m.groups()
        if 1 <= int(mo) <= 12:
            return _last_day_of_month(int(mo), _year_fix(int(y)))

    for m in _RE_COMPACT.finditer(t):
        d, mo, y = m.groups()
        dt = _safe_date(int(d), int(mo), _year_fix(int(y)))
        if dt:
            return dt

    return None


def _preprocess_for_ocr(frame: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if frame.ndim == 3 else frame
    h, w = gray.shape[:2]
    if max(h, w) < 1200:
        scale = 1.5
        gray = cv2.resize(gray, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_CUBIC)
    thr = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 31, 9)
    return cv2.morphologyEx(thr, cv2.MORPH_CLOSE, np.ones((2, 2), np.uint8), iterations=1)


def _ocr_expiry_from_frame(frame: np.ndarray) -> Tuple[Optional[date], float]:
    """Corre OCR sobre un fotograma y busca una fecha cerca de una palabra clave."""
    processed = _preprocess_for_ocr(frame)
    data: Dict = pytesseract.image_to_data(
        processed, lang="spa", config="--oem 3 --psm 6", output_type=pytesseract.Output.DICT
    )
    texts = data.get("text", []) or []
    confs = data.get("conf", []) or []

    joined = " ".join(t for t in texts if t and str(t).strip())
    if not any(k in joined.lower() for k in _KW):
        return None, 0.0

    dt = _parse_date_from_text(joined)
    if not dt:
        return None, 0.0

    valid_confs = [float(c) for c in confs if str(c) not in ("-1", -1)]
    return dt, (sum(valid_confs) / len(valid_confs) if valid_confs else 65.0)


def _check_expiry_offline(seconds: float) -> Tuple[bool, Optional[str], Optional[bool], Optional[float]]:
    """
    Respaldo sin internet: OCR local con Tesseract + patrones de fecha por
    regex. Usa la camara de sesion compartida, igual que las demas opciones
    (nunca abre su propia camara aparte). Menos preciso que Claude Vision
    porque no puede "razonar" cual fecha es la de vencimiento si el empaque
    tiene varias, pero es lo unico que funciona sin conexion.
    """
    best_dt: Optional[date] = None
    best_conf = -1.0
    last_process = 0.0

    for frame in frames_for(seconds):
        now = time.time()
        if now - last_process < 0.2:
            continue
        last_process = now

        try:
            dt, conf = _ocr_expiry_from_frame(frame)
        except Exception:
            continue

        if dt is not None and conf > best_conf:
            best_dt, best_conf = dt, conf
            if best_conf >= 80.0:
                break

    if best_dt is None:
        return False, None, None, None

    is_expired = best_dt < date.today()
    return True, best_dt.strftime("%d/%m/%Y"), is_expired, best_conf


def _should_show_debug_preview() -> bool:
    return bool(getattr(vision, "show_preview", False))


def check_expiry_best_frame(seconds: float = 6.0) -> Tuple[bool, Optional[str], Optional[bool], Optional[float]]:
    """
    Abre la camara de sesion (compartida con las demas opciones), muestra una
    vista previa limpia y manda la mejor foto a Claude Vision, que decide
    cual fecha del empaque es la de vencimiento. Si no hay ANTHROPIC_API_KEY
    o falla por falta de conexion, cae al respaldo local con OCR (funciona
    sin internet, aunque es menos preciso).
    """
    frame = None
    t0 = time.time()
    for current in frames_for(seconds):
        frame = current
        if _should_show_debug_preview():
            show = current.copy()
            time_left = int(seconds - (time.time() - t0)) + 1
            cv2.putText(
                show,
                f"Muestre la fecha de vencimiento - {time_left}s",
                (40, 60),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.0,
                (0, 255, 0),
                3,
                cv2.LINE_AA,
            )
            show_preview("Verificacion de Vencimiento", show)
            if cv2.waitKey(1) & 0xFF == 27:
                break
    if _should_show_debug_preview():
        cv2.destroyAllWindows()

    if frame is not None and os.environ.get("ANTHROPIC_API_KEY", "").strip():
        claude_result = _claude_vision_expiry(frame)
        if claude_result[0]:
            return claude_result

    return _check_expiry_offline(seconds)
