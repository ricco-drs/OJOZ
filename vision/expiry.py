# app/vision/expiry.py
from __future__ import annotations
import base64, os, time, re, threading, queue
from typing import Optional, Tuple, List, Dict
import cv2
import numpy as np
import pytesseract
from datetime import date, datetime

# Config de vision (indice de camara) y ruta de Tesseract, ambas centralizadas.
from app.config.settings import configure_tesseract, llm as llm_config, vision
from app.vision.camera import rotate_frame

configure_tesseract()

KW = [
    "venc", "vence", "vencimiento", "cad", "caduca", "caducidad",
    "exp", "expira", "expiry", "best before", "use by"
]

MONTHS_ES = {
    "enero":1, "ene":1,
    "febrero":2, "feb":2,
    "marzo":3, "mar":3,
    "abril":4, "abr":4,
    "mayo":5, "may":5,
    "junio":6, "jun":6,
    "julio":7, "jul":7,
    "agosto":8, "ago":8,
    "septiembre":9, "sept":9, "sep":9, "set":9,
    "octubre":10, "oct":10,
    "noviembre":11, "nov":11,
    "diciembre":12, "dic":12
}

RE_NUMERIC = re.compile(r"(\d{1,2})[\/\-.](\d{1,2})[\/\-.](\d{2,4})")
RE_FV = re.compile(r"\bfv\s*(\d{1,2})\s+(\d{1,2})\s+(\d{2,4})", re.IGNORECASE)
RE_MONTH_TEXT = re.compile(
    r"(\d{1,2})\s*(?:de\s*)?(enero|ene|febrero|feb|marzo|mar|abril|abr|mayo|may|junio|jun|julio|jul|agosto|ago|septiembre|sept|sep|set|octubre|oct|noviembre|nov|diciembre|dic)\s*(?:de\s*)?(\d{2,4})",
    re.IGNORECASE
)
RE_MMYYYY = re.compile(r"(\d{1,2})[\/\-.](\d{2,4})")  # mm/yyyy o dd/yyyy (ambiguo)
RE_PREFIX_MMYYYY = re.compile(r"(?:f\.?\s*v\.?\.?)\s*(\d{1,2})[\/\-.](\d{1,2})[\/\-.](\d{2,4})", re.IGNORECASE)
RE_COMPACT_MONTH = re.compile(
    r"(\d{1,2})\s*(enero|ene|febrero|feb|marzo|mar|abril|abr|mayo|may|junio|jun|"
    r"julio|jul|agosto|ago|septiembre|sept|sep|set|octubre|oct|noviembre|nov|diciembre|dic)"
    r"\s*(\d{2,4})",
    re.IGNORECASE,
)
RE_PREFIX_SPACE = re.compile(r"(?:f\.?\s*v\.?\.?)\s*(\d{1,2})\s+(\d{1,2})\s+(\d{2,4})", re.IGNORECASE)

def _normalize(s: str) -> str:
    return s.lower()

def _year_fix(y: int) -> int:
    # 2 dígitos → heurística
    if y < 100:
        return 2000 + y if y < 70 else 1900 + y
    return y

def _safe_date(d: int, m: int, y: int) -> Optional[date]:
    try:
        return date(y, m, d)
    except Exception:
        return None

def _parse_date_from_text(txt: str) -> Optional[date]:
    t = _normalize(txt)

    m = RE_FV.search(t)
    if m:
        d_s, m_s, y_s = m.groups()
        D = int(d_s)
        M = int(m_s)
        Y = _year_fix(int(y_s))
        return _safe_date(D, M, Y)

    # Caso con dos fechas (fabricacion y luego vencimiento): usar la segunda
    matches = list(RE_NUMERIC.finditer(t))
    if len(matches) >= 2:
        d2, m2, y2 = matches[1].groups()
        return _safe_date(int(d2), int(m2), _year_fix(int(y2)))

    m = RE_NUMERIC.search(t)
    if m:
        d, mm, yy = m.groups()
        D = int(d); M = int(mm); Y = _year_fix(int(yy))
        return _safe_date(D, M, Y)

    m = RE_MONTH_TEXT.search(t)
    if m:
        d_s, month_s, y_s = m.groups()
        D = int(d_s)
        M = MONTHS_ES.get(_normalize(month_s), 0)
        Y = _year_fix(int(y_s))
        return _safe_date(D, M, Y)

    m = RE_COMPACT_MONTH.search(t.replace("/", "").replace("-", "").replace(".", ""))
    if m:
        d_s, month_s, y_s = m.groups()
        D = int(d_s)
        M = MONTHS_ES.get(_normalize(month_s), 0)
        Y = _year_fix(int(y_s))
        return _safe_date(D, M, Y)

    m = RE_PREFIX_SPACE.search(t)
    if m:
        d_s, m_s, y_s = m.groups()
        D = int(d_s)
        M = int(m_s)
        Y = _year_fix(int(y_s))
        return _safe_date(D, M, Y)

    m = RE_PREFIX_MMYYYY.search(t)
    if m:
        d_s, m_s, y_s = m.groups()
        D = int(d_s)
        M = int(m_s)
        Y = _year_fix(int(y_s))
        return _safe_date(D, M, Y)

    # Intento mm/yyyy si el primer número <= 12
    m = RE_MMYYYY.search(t)
    if m:
        a, b = m.groups()
        A = int(a); B = int(b)
        if A <= 12:
            # mm/yyyy -> asumimos día 1
            return _safe_date(1, A, _year_fix(B))
        else:
            # dd/yyyy -> no sabemos mes, descartamos
            return None

    return None

def _preprocess(img: np.ndarray) -> np.ndarray:
    if img.ndim == 3:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        gray = img
    # Aumentar un poco tamaño si chico
    h, w = gray.shape[:2]
    scale = 1.5 if max(h, w) < 1200 else 1.0
    if scale != 1.0:
        gray = cv2.resize(gray, (int(w*scale), int(h*scale)), interpolation=cv2.INTER_CUBIC)
    # Adaptive threshold + ligero cierre
    thr = cv2.adaptiveThreshold(gray,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY,31,9)
    thr = cv2.morphologyEx(thr, cv2.MORPH_CLOSE, np.ones((2,2), np.uint8), iterations=1)
    return thr

def _ocr_data(img: np.ndarray, lang: str = "spa") -> Dict:
    config = "--oem 3 --psm 6"
    return pytesseract.image_to_data(img, lang=lang, config=config, output_type=pytesseract.Output.DICT)

def _find_best_expiry_from_data(data: Dict) -> Tuple[Optional[date], Optional[float], Optional[Tuple[int,int,int,int]]]:
    texts = data.get("text", []) or []
    confs = data.get("conf", []) or []
    xs, ys, ws, hs = (data.get(k, []) or [] for k in ("left","top","width","height"))

    # 1) buscar tokens con keywords y extraer ventana alrededor
    def box_union(boxes: List[Tuple[int,int,int,int]]) -> Tuple[int,int,int,int]:
        x0 = min(b[0] for b in boxes); y0 = min(b[1] for b in boxes)
        x1 = max(b[0]+b[2] for b in boxes); y1 = max(b[1]+b[3] for b in boxes)
        return (x0, y0, x1-x0, y1-y0)

    kw_boxes = []
    for i, t in enumerate(texts):
        if not t or str(confs[i]) in ("-1", -1):
            continue
        tt = _normalize(t)
        if any(k in tt for k in KW):
            kw_boxes.append((xs[i], ys[i], ws[i], hs[i]))

    if kw_boxes:
        # Unimos áreas de keywords
        main = box_union(kw_boxes)
        return None, None, main  # señal: recorta alrededor de este box y re-evalúa
    else:
        # 2) sin keyword: unir todos los tokens en un solo texto y buscar la fecha ahi
        joined = " ".join([t for t in texts if t and str(t).strip()])
        dt = _parse_date_from_text(joined)
        if dt:
            # confianza media simple
            valid_confs = [float(c) for c in confs if str(c) not in ("-1", -1)]
            mean_conf = sum(valid_confs)/len(valid_confs) if valid_confs else 70.0
            return dt, mean_conf, None

    return None, None, None

def _extract_roi(img: np.ndarray, box: Tuple[int,int,int,int]) -> np.ndarray:
    H, W = img.shape[:2]
    x, y, w, h = box
    # expandimos un margen para capturar la fecha cercana a la palabra clave
    pad = int(0.6 * max(w, h))
    x0 = max(0, x - pad); y0 = max(0, y - pad)
    x1 = min(W, x + w + pad); y1 = min(H, y + h + pad)
    return img[y0:y1, x0:x1]

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
                                "Esta foto muestra el empaque de un producto. Busca la fecha "
                                "de vencimiento o caducidad (etiquetas como VENC, EXP, F.V., "
                                "CAD, Best Before, Use By) y no otra fecha del empaque, como "
                                "la de fabricacion o un codigo de lote. Responde UNICAMENTE en "
                                "este formato, sin nada mas alrededor: DD/MM/AAAA (por ejemplo "
                                "15/03/2026). Si no encuentras una fecha de vencimiento clara, "
                                "responde unicamente: NINGUNO"
                            ),
                        },
                    ],
                }
            ],
        )
    except Exception as exc:
        logger.warning(f"Claude Vision fallo para vencimiento, se usa OCR local: {exc}")
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


def _detect_expiry_with_claude(seconds: float) -> Tuple[bool, Optional[str], Optional[bool], Optional[float]]:
    """
    Abre la camara, muestra el preview y manda la mejor foto a Claude Vision.
    Sin ANTHROPIC_API_KEY no abre la camara: se cae al flujo de OCR+regex.
    """
    if not os.environ.get("ANTHROPIC_API_KEY", "").strip():
        return False, None, None, None

    cap = cv2.VideoCapture(getattr(vision, "camera_index", 0), cv2.CAP_DSHOW)
    if not cap.isOpened():
        return False, None, None, None

    frame = None
    try:
        t0 = time.time()
        while time.time() - t0 < seconds:
            ret, current = cap.read()
            if not ret:
                break
            current = rotate_frame(current)
            frame = current
            if getattr(vision, "show_preview", False):
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
                cv2.imshow("Verificacion de Vencimiento", show)
                if cv2.waitKey(1) & 0xFF == 27:
                    break
    finally:
        cap.release()
        cv2.destroyAllWindows()

    if frame is None:
        return False, None, None, None

    return _claude_vision_expiry(frame)


def check_expiry_best_frame(seconds: float = 6.0) -> Tuple[bool, Optional[str], Optional[bool], Optional[float]]:
    # Primero Claude Vision (decide cual fecha es la de vencimiento); si no
    # hay clave configurada o falla, se sigue con OCR + regex de siempre.
    claude_result = _detect_expiry_with_claude(seconds=min(seconds, 8.0))
    if claude_result[0]:
        return claude_result
    return _check_expiry_ocr(seconds=seconds)


def _check_expiry_ocr(seconds: float = 6.0) -> Tuple[bool, Optional[str], Optional[bool], Optional[float]]:
    """
    Captura frames durante `seconds` y trata de leer la fecha de vencimiento.
    El preview corre fluido y el OCR se procesa en un hilo aparte.
    """
    cap = cv2.VideoCapture(getattr(vision, "camera_index", 0), cv2.CAP_DSHOW)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)

    if not cap.isOpened():
        print("No se pudo abrir la camara para verificar fecha de vencimiento.")
        return False, None, None, None

    shared = {"best_dt": None, "best_conf": -1.0, "best_dt_text": None}
    done = threading.Event()
    stop = threading.Event()
    frame_queue: "queue.Queue[np.ndarray]" = queue.Queue(maxsize=1)

    def _update_result(dt_value: Optional[date], conf: Optional[float]):
        if dt_value is None or conf is None:
            return
        if conf > shared["best_conf"]:
            shared["best_dt"] = dt_value
            shared["best_conf"] = float(conf)
            shared["best_dt_text"] = dt_value.strftime("%d/%m/%Y")
            print(f"  [OK] Fecha detectada: {shared['best_dt_text']} (conf={conf:.1f}%)")
            if conf >= 82.0:
                done.set()

    def _process_frame(frame: np.ndarray):
        proc = _preprocess(frame)
        data = _ocr_data(proc, lang="spa")
        dt, conf, kw_box = _find_best_expiry_from_data(data)

        if kw_box is not None and dt is None:
            roi = _extract_roi(frame, kw_box)
            proc2 = _preprocess(roi)
            data2 = _ocr_data(proc2, lang="spa")
            txt = " ".join([t for t in data2.get("text", []) if t and str(t).strip()])
            dt2 = _parse_date_from_text(txt)
            if dt2:
                confs2 = [float(c) for c in data2.get("conf", []) if str(c) not in ("-1", -1)]
                conf2 = sum(confs2)/len(confs2) if confs2 else 72.0
                dt, conf = dt2, conf2

        if dt:
            _update_result(dt, conf or 70.0)

    def _worker():
        last_process = 0.0
        interval = 0.12
        while not stop.is_set() or not frame_queue.empty():
            try:
                frame = frame_queue.get(timeout=0.1)
            except queue.Empty:
                continue
            now = time.time()
            if now - last_process < interval:
                continue
            last_process = now
            _process_frame(frame)

    worker = threading.Thread(target=_worker, daemon=True)
    worker.start()

    t0 = time.time()
    best_dt_text = None
    best_conf = -1.0

    try:
        while time.time() - t0 < seconds and not done.is_set():
            ret, frame = cap.read()
            if not ret:
                break
            frame = rotate_frame(frame)

            if frame_queue.full():
                try:
                    frame_queue.get_nowait()
                except queue.Empty:
                    pass
            try:
                frame_queue.put_nowait(frame.copy())
            except queue.Full:
                pass

            if getattr(vision, "show_preview", False):
                show = frame.copy()
                time_left = int(seconds - (time.time() - t0)) + 1
                cv2.putText(show, f"Tiempo: {time_left}s", (50, 80), cv2.FONT_HERSHEY_SIMPLEX, 2.0, (0, 255, 0), 4, cv2.LINE_AA)

                preview_text = shared["best_dt_text"]
                preview_conf = shared["best_conf"]
                preview_dt = shared["best_dt"]

                if preview_text:
                    cv2.putText(show, f"Fecha: {preview_text}", (50, 150), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 3, cv2.LINE_AA)
                    cv2.putText(show, f"Confianza: {preview_conf:.1f}%", (50, 200), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 2, cv2.LINE_AA)
                    today = date.today()
                    if preview_dt and preview_dt < today:
                        status = "VENCIDO"
                        color = (0, 0, 255)
                    else:
                        status = "VIGENTE"
                        color = (0, 255, 0)
                    cv2.putText(show, status, (50, 250), cv2.FONT_HERSHEY_SIMPLEX, 1.5, color, 4, cv2.LINE_AA)
                else:
                    cv2.putText(show, "Muestre la fecha de vencimiento", (50, 150), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 2, cv2.LINE_AA)

                cv2.imshow("Verificacion de Vencimiento", show)
                if cv2.waitKey(1) & 0xFF == 27:
                    break

        best_dt_text = shared["best_dt_text"]
        best_conf = shared["best_conf"]
    finally:
        stop.set()
        cap.release()
        cv2.destroyAllWindows()
        worker.join(timeout=1.0)

    best_dt = shared["best_dt"]
    if best_dt is None:
        return False, None, None, None

    today = date.today()
    is_expired = best_dt < today
    return True, best_dt_text, is_expired, best_conf
