# app/vision/currency.py
from __future__ import annotations

import time
import threading
import queue
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import pytesseract

# Config de vision (indice de camara, rutas) y ruta de Tesseract, ambas centralizadas.
from app.config.settings import configure_tesseract, vision

configure_tesseract()

# Solo soles (incluye monedas 1,2,5)
KNOWN_BILLS = {
    "PEN": {1, 2, 5, 10, 20, 50, 100, 200},
}
ALL_BILL_VALUES = {val for vals in KNOWN_BILLS.values() for val in vals}

PEN_INDICATORS = ["s/", "pen", "sol", "soles", "nuevo sol", "peru", "banco central"]
PRIORITY_BILLS = {("PEN", 1), ("PEN", 2), ("PEN", 5), ("PEN", 10)}  # monedas y billete comun


def _guess_currency_from_text(text: str) -> Optional[str]:
    if any(ind in text for ind in PEN_INDICATORS):
        return "PEN"
    return "PEN"


def _textual_value_hint(text: str) -> Tuple[Optional[str], Optional[int], float]:
    """
    Reglas rapidas por texto para los valores esperados en soles.
    """
    if not text:
        return None, None, 0.0
    txt = text.lower().replace("\n", " ")
    has_sol = any(k in txt for k in ["sol", "soles", "banco central", "peru", "s/"])
    if has_sol:
        for val in sorted(KNOWN_BILLS["PEN"]):
            if str(val) in txt:
                return "PEN", val, 80.0
    return None, None, 0.0


def _parse_template_metadata(path: Path) -> tuple[Optional[str], Optional[int]]:
    name = path.stem.lower()
    parts = name.replace("-", "_").split("_")
    currency = None
    value = None
    for part in parts:
        part_upper = part.upper()
        if part_upper in KNOWN_BILLS:
            currency = part_upper
        else:
            digits = "".join(ch for ch in part if ch.isdigit())
            if digits:
                try:
                    value = int(digits)
                except ValueError:
                    continue
    if currency and value in KNOWN_BILLS.get(currency, set()):
        return currency, value
    return None, None


def _load_reference_templates():
    templates = []
    ref_dir = Path(getattr(vision, "currency_dir", "data/currency_refs"))
    if not ref_dir.exists():
        return templates
    orb = cv2.ORB_create(1000)
    for path in ref_dir.glob("*.*"):
        if path.suffix.lower() not in {".jpg", ".jpeg", ".png"}:
            continue
        currency, value = _parse_template_metadata(path)
        if not currency or value is None:
            continue
        img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
        if img is None:
            continue
        kp, desc = orb.detectAndCompute(img, None)
        if desc is None or len(kp) < 10:
            continue
        templates.append({"currency": currency, "value": value, "kp": kp, "desc": desc})
        print(f"  -> Referencia cargada: {path.name}")
    return templates


_TEMPLATE_FEATURES = _load_reference_templates()
_ORB = cv2.ORB_create(800)
_MATCHER = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)


def _preprocess_for_digits(img: np.ndarray) -> np.ndarray:
    if img.ndim == 3:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        gray = img
    h, w = gray.shape[:2]
    scale = 1.4 if max(h, w) < 1100 else 1.0
    if scale != 1.0:
        gray = cv2.resize(gray, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_CUBIC)
    gray = cv2.GaussianBlur(gray, (3, 3), 0)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    gray = clahe.apply(gray)
    gray = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 15, 3)
    return gray


def _ocr_digits(img: np.ndarray) -> Dict:
    config = "--oem 3 --psm 6 -c tessedit_char_whitelist=0123456789"
    return pytesseract.image_to_data(img, lang="eng", config=config, output_type=pytesseract.Output.DICT)


def _ocr_text(img: np.ndarray) -> str:
    try:
        config = "--oem 3 --psm 6"
        txt = pytesseract.image_to_string(img, lang="eng", config=config)
        return (txt or "").lower()
    except Exception:
        return ""


def _pick_best_denomination(data: Dict) -> Tuple[Optional[int], Optional[float]]:
    texts: List[str] = data.get("text", []) or []
    confs_raw = data.get("conf", []) or []
    tokens: List[Tuple[str, float]] = []
    for i, t in enumerate(texts):
        if not t:
            continue
        conf_raw = confs_raw[i] if i < len(confs_raw) else "50"
        if str(conf_raw) in ("-1", -1):
            continue
        token = "".join(ch for ch in t if ch.isdigit())
        if not token:
            continue
        try:
            conf_val = float(conf_raw)
        except Exception:
            conf_val = 60.0
        tokens.append((token, conf_val))
    if not tokens:
        return None, None

    value_conf: Dict[int, List[float]] = {}

    def _push(val: int, conf: float) -> None:
        if val in ALL_BILL_VALUES:
            value_conf.setdefault(val, []).append(conf)

    for token, conf in tokens:
        try:
            val = int(token)
        except ValueError:
            continue
        _push(val, conf)

    for i in range(len(tokens) - 1):
        merged = tokens[i][0] + tokens[i + 1][0]
        if 2 <= len(merged) <= 4:
            try:
                val = int(merged)
            except ValueError:
                continue
            avg_conf = (tokens[i][1] + tokens[i + 1][1]) / 2.0
            _push(val, avg_conf)

    if not value_conf:
        return None, None

    has_high = any(v >= 10 for v in value_conf.keys())

    best_val, best_conf = None, -1.0
    for v, confs in value_conf.items():
        avg = sum(confs) / len(confs)
        # Ajustes heurísticos: favorecer billetes completos, penalizar falsos 1/2/5 si hay opciones altas
        if v >= 10:
            avg += 6.0
        if has_high and v in {1, 2, 5}:
            avg -= 12.0
        if avg > best_conf:
            best_val, best_conf = v, avg

    return best_val, best_conf if best_val is not None else (None, None)


def _match_with_templates(gray_img: np.ndarray) -> Tuple[Optional[str], Optional[int], float]:
    if not _TEMPLATE_FEATURES:
        return None, None, 0.0
    kp, desc = _ORB.detectAndCompute(gray_img, None)
    if desc is None or len(kp) < 10:
        return None, None, 0.0
    best_currency = None
    best_value = None
    best_score = 0.0
    for tpl in _TEMPLATE_FEATURES:
        matches = _MATCHER.knnMatch(desc, tpl["desc"], k=2)
        good = [m for m, n in matches if n and m.distance < 0.75 * n.distance]
        if not good:
            continue
        score = 100.0 * len(good) / max(len(tpl["kp"]), 1)
        if score > best_score:
            best_score = score
            best_currency = tpl["currency"]
            best_value = tpl["value"]
    return best_currency, best_value, best_score


def detect_currency_best_frame(seconds: float = 8.0) -> Tuple[bool, Optional[str], Optional[float], Optional[float]]:
    cap = cv2.VideoCapture(getattr(vision, "camera_index", 0), cv2.CAP_DSHOW)
    if not cap.isOpened():
        print("No se pudo abrir la camara para detectar billetes.")
        return False, None, None, None

    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 960)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    cap.set(cv2.CAP_PROP_FPS, 30)

    shared = {"best_val": None, "best_conf": -1.0, "best_currency": None, "done": False}
    lock = threading.Lock()
    frame_queue: "queue.Queue[np.ndarray]" = queue.Queue(maxsize=1)

    def _update_result(val: Optional[int], conf: Optional[float], currency: Optional[str], priority: bool = False) -> None:
        if val is None or conf is None:
            return
        if priority:
            conf = min(99.9, conf + 8.0)
        with lock:
            if conf > shared["best_conf"]:
                shared["best_val"] = val
                shared["best_conf"] = conf
                shared["best_currency"] = currency or "PEN"
                print(f"  Detectado: {shared['best_currency']} {val} (conf={conf:.1f}%)")
                threshold = 70.0 if priority else 80.0
                if conf >= threshold:
                    shared["done"] = True

    def _processor() -> None:
        last_process = 0.0
        last_full_scan = 0.0
        frame_idx = 0
        process_interval = 0.15

        while not shared["done"]:
            try:
                frame = frame_queue.get(timeout=0.1)
            except queue.Empty:
                continue

            frame_idx += 1
            now = time.time()
            if now - last_process < process_interval:
                continue
            last_process = now

            h, w = frame.shape[:2]
            x0, y0 = int(w * 0.18), int(h * 0.18)
            x1, y1 = int(w * 0.82), int(h * 0.82)
            roi = frame[y0:y1, x0:x1]
            regions: List[Tuple[str, np.ndarray]] = [("roi", roi)]

            with lock:
                current_conf = shared["best_conf"]

            if current_conf < 70 and (now - last_full_scan) > 0.8:
                regions.append(("full", frame))
                last_full_scan = now

            if current_conf < 80:
                corner = int(min(h, w) * 0.36)
                corners = {
                    "top_left": frame[0:corner, 0:corner],
                    "top_right": frame[0:corner, w - corner : w],
                    "bottom_left": frame[h - corner : h, 0:corner],
                    "bottom_right": frame[h - corner : h, w - corner : w],
                }
                for name, region in corners.items():
                    if region.size > 0:
                        regions.append((name, region))

            template_currency = None
            template_value = None
            template_score = 0.0
            if _TEMPLATE_FEATURES and frame_idx % 2 == 0 and current_conf < 75:
                gray_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
                target_tpl_w = 720
                gray_roi = cv2.resize(
                    gray_roi,
                    (target_tpl_w, int(target_tpl_w * gray_roi.shape[0] / gray_roi.shape[1])),
                    interpolation=cv2.INTER_AREA,
                )
                template_currency, template_value, template_score = _match_with_templates(gray_roi)

            for region_name, region in regions:
                proc = _preprocess_for_digits(region)
                target_w = 720 if region_name == "roi" else 640
                proc = cv2.resize(
                    proc,
                    (target_w, int(target_w * proc.shape[0] / proc.shape[1])),
                    interpolation=cv2.INTER_AREA,
                )
                data = _ocr_digits(proc)
                val, conf = _pick_best_denomination(data)

                currency_guess = "PEN"
                txt = None
                if frame_idx % 3 == 0:
                    txt = _ocr_text(proc)
                    currency_guess = _guess_currency_from_text(txt)
                    hint_cur, hint_val, hint_conf = _textual_value_hint(txt)
                    if hint_val is not None:
                        val = hint_val
                        conf = max(conf or 0, hint_conf)
                        currency_guess = hint_cur or currency_guess

                if template_value is not None and template_score > 25:
                    val = template_value
                    conf = max(conf or 0, template_score)
                    currency_guess = template_currency or currency_guess or "PEN"

                if val is not None:
                    conf_val = float(conf or 0)
                    possibles = [cur for cur, s in KNOWN_BILLS.items() if val in s]
                    if currency_guess and val not in KNOWN_BILLS.get(currency_guess, set()):
                        currency_guess = None
                    chosen_currency = currency_guess or (possibles[0] if possibles else "PEN")
                    is_priority = (chosen_currency, val) in PRIORITY_BILLS
                    _update_result(val, conf_val, chosen_currency, priority=is_priority)

            if shared["done"]:
                break

    worker = threading.Thread(target=_processor, daemon=True)
    worker.start()

    t0 = time.time()
    try:
        while time.time() - t0 < seconds and not shared["done"]:
            ret, frame = cap.read()
            if not ret:
                break

            if frame_queue.full():
                try:
                    frame_queue.get_nowait()
                except queue.Empty:
                    pass
            frame_queue.put_nowait(frame.copy())

            if getattr(vision, "show_preview", False):
                show = frame.copy()
                h, w = show.shape[:2]
                x0, y0 = int(w * 0.18), int(h * 0.18)
                x1, y1 = int(w * 0.82), int(h * 0.82)
                time_left = int(seconds - (time.time() - t0)) + 1
                cv2.rectangle(show, (x0, y0), (x1, y1), (0, 255, 255), 3)
                cv2.putText(
                    show,
                    f"Tiempo: {time_left}s",
                    (50, 80),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    2.0,
                    (0, 255, 0),
                    4,
                    cv2.LINE_AA,
                )
                with lock:
                    preview_val = shared["best_val"]
                    preview_conf = shared["best_conf"]
                if preview_val is not None:
                    cv2.putText(
                        show,
                        f"Detectado: PEN {preview_val}",
                        (50, 150),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1.5,
                        (0, 255, 0),
                        3,
                        cv2.LINE_AA,
                    )
                    cv2.putText(
                        show,
                        f"Confianza: {preview_conf:.1f}%",
                        (50, 200),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1.2,
                        (0, 255, 0),
                        2,
                        cv2.LINE_AA,
                    )
                else:
                    cv2.putText(
                        show,
                        "Muestre el numero del billete",
                        (50, 150),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1.2,
                        (255, 255, 255),
                        2,
                        cv2.LINE_AA,
                    )
                cv2.imshow("Deteccion de Dinero", show)
                if cv2.waitKey(1) & 0xFF == 27:
                    break

        shared["done"] = True
    finally:
        cap.release()
        cv2.destroyAllWindows()
        worker.join(timeout=1.0)

    best_val = shared["best_val"]
    best_conf = shared["best_conf"]

    if best_val is None:
        return False, None, None, None

    final_currency = "PEN"
    return True, final_currency, float(best_val), float(best_conf)
