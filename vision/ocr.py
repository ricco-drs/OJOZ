# app/vision/ocr.py
import base64
import json
import os
import cv2
import pytesseract
import numpy as np
import time
import re
import unicodedata
from datetime import datetime
from typing import Optional, Tuple, List
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from app.config.settings import configure_tesseract, vision
from app.vision.camera_service import frames_for

# Configurar Tesseract. La ruta esta centralizada en config.settings y se puede
# sobreescribir con la variable de entorno TESSERACT_CMD. Si no se encuentra el
# binario, pytesseract intentara usar el del PATH.
configure_tesseract()



def _apply_gamma(gray: np.ndarray) -> np.ndarray:
    """Corrige gamma para escenas oscuras o subexpuestas."""
    mean = float(np.mean(gray))
    if mean <= 0:
        return gray
    if mean < 110:
        gamma = min(2.2, max(1.2, 140.0 / mean))
        inv_gamma = 1.0 / gamma
        table = np.array([(i / 255.0) ** inv_gamma * 255 for i in np.arange(0, 256)]).astype("uint8")
        return cv2.LUT(gray, table)
    return gray


def _preprocess_for_ocr(gray: np.ndarray) -> dict:
    """Genera multiples representaciones de la imagen para OCR."""
    height, width = gray.shape
    target_w = 1900
    if width < target_w:
        scale = target_w / width
        new_width = int(width * scale)
        new_height = int(height * scale)
        gray = cv2.resize(gray, (new_width, new_height), interpolation=cv2.INTER_LANCZOS4)

    gray = _apply_gamma(gray)

    clahe = cv2.createCLAHE(clipLimit=2.6, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray)
    denoised = cv2.fastNlMeansDenoising(enhanced, None, 12, 7, 21)

    # Enfocar ligeramente para mejorar bordes de caracteres finos
    sharpen = cv2.addWeighted(denoised, 1.5, cv2.GaussianBlur(denoised, (0, 0), 1.2), -0.5, 0)

    blur = cv2.GaussianBlur(sharpen, (3, 3), 0)
    binary = cv2.adaptiveThreshold(
        blur,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        41,
        8,
    )
    kernel = np.ones((2, 2), np.uint8)
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
    binary_inv = cv2.bitwise_not(binary)
    thicker = cv2.dilate(binary, np.ones((2, 2), np.uint8), iterations=1)
    thinner = cv2.erode(binary, np.ones((2, 2), np.uint8), iterations=1)

    _, otsu = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    otsu_inv = cv2.bitwise_not(otsu)

    return {
        "enhanced": denoised,
        "sharpen": sharpen,
        "binary": binary,
        "binary_inv": binary_inv,
        "thick": thicker,
        "thin": thinner,
        "otsu": otsu,
        "otsu_inv": otsu_inv,
        "original": gray,
    }


def _analyze_lighting(gray: np.ndarray) -> tuple[float, float]:
    """Retorna promedio y desviación estándar para decidir el pipeline."""
    return float(np.mean(gray)), float(np.std(gray))




def _frame_quality_score(frame: np.ndarray) -> tuple[float, float, float, float]:
    """Mide nitidez y brillo para elegir el mejor frame antes de correr OCR."""
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    focus = float(cv2.Laplacian(gray, cv2.CV_64F).var())
    brightness = float(np.mean(gray))
    contrast = float(np.std(gray))
    penalty = abs(brightness - 140.0) * 1.2  # penaliza escenas muy oscuras o muy brillantes
    score = focus + (contrast * 3.0) - penalty
    return score, focus, brightness, contrast


def _rotate_image(image, angle: int):
    """Rota la imagen en múltiplos de 90 grados."""
    if angle == 0:
        return image
    if angle == 90:
        return cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)
    if angle == -90 or angle == 270:
        return cv2.rotate(image, cv2.ROTATE_90_COUNTERCLOCKWISE)
    if angle == 180:
        return cv2.rotate(image, cv2.ROTATE_180)
    return image


def _sanitize_text(text: str) -> str:
    if not text:
        return ""
    normalized = unicodedata.normalize("NFKD", text)
    ascii_text = normalized.encode("ascii", "ignore").decode("ascii")
    ascii_text = ascii_text.replace("\r", "\n")
    allowed_pattern = r"[^A-Za-z0-9 \n]"
    cleaned = re.sub(allowed_pattern, " ", ascii_text)
    cleaned = re.sub(r"[ \t]+", " ", cleaned)
    cleaned = re.sub(r"\n{2,}", "\n", cleaned)
    cleaned = cleaned.strip()
    # Si el saneado queda vacio pero el texto original tenia algo, devuelvo el original
    if not cleaned and text.strip():
        return text.strip()
    return cleaned


def _google_vision_ocr(frame: np.ndarray, lang: str) -> Tuple[str, float]:
    """
    OCR con Google Cloud Vision, si hay GOOGLE_VISION_API_KEY configurada.

    Suele leer mejor que Tesseract fotos reales (torcidas, con sombra, mala
    letra). Sin la clave no hace nada: el llamador sigue con Tesseract como
    respaldo local, igual que el TTS cae a gTTS si falta ELEVENLABS_API_KEY.
    """
    from app.utils.logger import logger

    api_key = os.environ.get("GOOGLE_VISION_API_KEY", "").strip()
    if not api_key:
        return "", 0.0

    ok, buffer = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 92])
    if not ok:
        return "", 0.0

    payload = json.dumps(
        {
            "requests": [
                {
                    "image": {"content": base64.b64encode(buffer.tobytes()).decode("ascii")},
                    "features": [{"type": "DOCUMENT_TEXT_DETECTION"}],
                    "imageContext": {"languageHints": [lang or "es"]},
                }
            ]
        }
    ).encode("utf-8")

    endpoint = f"https://vision.googleapis.com/v1/images:annotate?key={api_key}"
    request = Request(
        endpoint,
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )

    try:
        with urlopen(request, timeout=15.0) as response:
            body = json.loads(response.read().decode("utf-8"))
    except (HTTPError, URLError) as exc:
        logger.warning(f"Google Cloud Vision fallo, se usa Tesseract: {exc}")
        return "", 0.0
    except Exception as exc:
        logger.warning(f"Google Cloud Vision fallo, se usa Tesseract: {exc}")
        return "", 0.0

    try:
        first_response = body.get("responses", [{}])[0]
    except (IndexError, AttributeError):
        return "", 0.0

    if "error" in first_response:
        logger.warning(f"Google Cloud Vision error: {first_response['error']}")
        return "", 0.0

    raw_text = first_response.get("fullTextAnnotation", {}).get("text", "")
    text = _sanitize_text(raw_text)
    if not text:
        return "", 0.0
    return text, 95.0


def _clean_ocr_text_with_claude(raw_text: str) -> str:
    """
    Limpia el texto extraido por OCR (Google Vision o Tesseract) descartando
    ruido: palabras sin sentido, caracteres sueltos o simbolos que no son
    palabras reales en espanol ni ingles, tipicos de errores de reconocimiento.

    No traduce ni reformula, solo quita el ruido. Sin ANTHROPIC_API_KEY, o si
    la llamada falla, se devuelve el texto original sin tocar.
    """
    if not raw_text or not raw_text.strip():
        return raw_text

    from app.utils.logger import logger

    try:
        import anthropic
    except ImportError:
        return raw_text

    try:
        client = anthropic.Anthropic()
        response = client.messages.create(
            model="claude-haiku-4-5-20251001",
            max_tokens=1024,
            messages=[
                {
                    "role": "user",
                    "content": (
                        "El siguiente texto se extrajo con OCR de un documento y puede "
                        "tener ruido: palabras sin sentido, caracteres sueltos o simbolos "
                        "que no son palabras reales en espanol ni en ingles, producto de "
                        "errores de reconocimiento. Devuelve el mismo texto pero quitando "
                        "unicamente ese ruido. No traduzcas, no reformules, no agregues "
                        "nada nuevo, no corrijas ortografia de palabras validas: solo "
                        "elimina lo que claramente no es una palabra real. Si una linea "
                        "queda vacia al quitar el ruido, omite esa linea. Responde "
                        "unicamente con el texto limpio, sin comentarios ni explicaciones.\n\n"
                        f"Texto:\n{raw_text}"
                    ),
                }
            ],
        )
    except Exception as exc:
        logger.warning(f"Limpieza de OCR con Claude fallo, se usa el texto sin limpiar: {exc}")
        return raw_text

    cleaned = "".join(b.text for b in response.content if getattr(b, "type", None) == "text").strip()
    return cleaned or raw_text


def _extract_text(image, lang='spa'):
    """Extrae texto usando Tesseract con configuraciones estrictas."""
    best_text = None
    best_conf = 0.0

    # Combinaciones reducidas para rapidez
    psm_modes = [6]
    # Probar idioma principal, combinado y fallback a ingles si no esta presente
    languages = []
    if lang:
        languages.append(lang)
        if "eng" not in lang:
            languages.append(f"{lang}+eng")
    languages.append("eng")

    for lang_code in languages:
        for psm in psm_modes:
            try:
                config = f"--oem 3 --psm {psm}"

                text_result = pytesseract.image_to_string(image, lang=lang_code, config=config)
                print(f"  {lang_code} PSM {psm}: {len(text_result.split()) if text_result else 0} palabras")

                if not text_result or not text_result.strip():
                    continue

                data = pytesseract.image_to_data(
                    image,
                    lang=lang_code,
                    config=config,
                    output_type=pytesseract.Output.DICT,
                )
                confidences = [
                    int(c) for c in data["conf"] if str(c).isdigit() and int(c) > 0
                ]
                avg_conf = sum(confidences) / len(confidences) if confidences else 0.0
                word_count = len(text_result.split())
                current_best_words = len(best_text.split()) if best_text else 0

                is_better = False
                if word_count > current_best_words:
                    is_better = True
                elif word_count == current_best_words and avg_conf > best_conf:
                    is_better = True
                elif not best_text and text_result.strip():
                    is_better = True

                if is_better:
                    best_text = text_result.strip()
                    best_conf = avg_conf
                print(f"    V Mejor resultado hasta ahora (conf={avg_conf:.1f}%)")
            except Exception as e:
                print(f"  {lang_code} PSM {psm}: Error - {e}")
                continue

    return best_text, best_conf


def _fallback_ocr(gray: np.ndarray, lang: str) -> tuple[str, float]:
    """
    Fallback simple para casos donde el pipeline principal no devuelve texto.
    Usa reescalado fuerte y PSM lineal.
    """
    h, w = gray.shape
    if w < 2100:
        scale = 2100 / w
        gray = cv2.resize(gray, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_LANCZOS4)
    sharpen = cv2.addWeighted(gray, 1.6, cv2.GaussianBlur(gray, (0, 0), 1.3), -0.6, 0)
    _, thresh = cv2.threshold(sharpen, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    langs = [lang, f"{lang}+eng"] if "eng" not in lang else [lang]
    psm_list = [6, 13]
    best_text = ""
    best_conf = 0.0
    for lg in langs:
        for psm in psm_list:
            try:
                config = f"--oem 3 --psm {psm}"
                text = pytesseract.image_to_string(thresh, lang=lg, config=config)
                if not text or not text.strip():
                    continue
                data = pytesseract.image_to_data(
                    thresh,
                    lang=lg,
                    config=config,
                    output_type=pytesseract.Output.DICT,
                )
                confidences = [int(c) for c in data["conf"] if str(c).isdigit() and int(c) > 0]
                avg_conf = sum(confidences) / len(confidences) if confidences else 0.0
                if len(text.split()) > len(best_text.split()) or (len(text.split()) == len(best_text.split()) and avg_conf > best_conf):
                    best_text = text.strip()
                    best_conf = avg_conf
            except Exception:
                continue
    return best_text, best_conf


def _quick_simple_ocr(gray: np.ndarray, lang: str) -> tuple[str, float]:
    """
    Ultimo recurso: OCR directo sin tanto preprocesado para evitar devolver texto vacio.
    """
    variants: list[tuple[str, np.ndarray]] = [("gray", gray)]
    try:
        _, otsu_img = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        variants.append(("otsu", otsu_img))
    except Exception:
        pass

    best_text, best_conf = "", 0.0
    configs = [
        ("--oem 3 --psm 6", "psm6"),
        ("--oem 3 --psm 11", "psm11"),
        ("--oem 3 --psm 4", "psm4"),
    ]

    for _tag, img in variants:
        for cfg, _cfg_name in configs:
            try:
                text = pytesseract.image_to_string(img, lang=lang, config=cfg)
                if not text or not text.strip():
                    continue
                data = pytesseract.image_to_data(
                    img,
                    lang=lang,
                    config=cfg,
                    output_type=pytesseract.Output.DICT,
                )
                confs = [int(c) for c in data.get("conf", []) if str(c).isdigit() and int(c) > 0]
                avg_conf = sum(confs) / len(confs) if confs else 0.0
                cleaned = _sanitize_text(text)
                if not cleaned:
                    continue
                # Prioriza mas palabras; a igualdad, mayor confianza.
                if len(cleaned.split()) > len(best_text.split()) or (
                    len(cleaned.split()) == len(best_text.split()) and avg_conf > best_conf
                ):
                    best_text, best_conf = cleaned, avg_conf
            except Exception:
                continue

    return best_text, best_conf


def read_text_best_frame(seconds: float = 10.0, lang: str = 'spa') -> Tuple[bool, Optional[str], Optional[float]]:
    """
    Muestra preview de la cámara, elige el frame más nítido y procesa OCR.
    """
    def _configure(cap):
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
        cap.set(cv2.CAP_PROP_AUTOFOCUS, 1)

    captured_frame = None
    best_frame = None
    best_score = -1e9
    best_focus = 0.0
    best_brightness = 0.0
    best_contrast = 0.0
    last_frame = None
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    try:
        # Dar tiempo para que el usuario posicione el documento (8s minimo para agilizar)
        preview_time = max(seconds, 8.0)
        t0 = time.time()

        print("?? Posicione el documento frente a la camara...")

        for frame in frames_for(preview_time, configure_capture=_configure):
            last_frame = frame.copy()

            score, focus, brightness, contrast = _frame_quality_score(frame)
            if score > best_score:
                best_score = score
                best_frame = frame.copy()
                best_focus = focus
                best_brightness = brightness
                best_contrast = contrast

            if vision.show_preview:
                display = frame.copy()
                time_left = int(preview_time - (time.time() - t0)) + 1

                # Mostrar cuenta regresiva en esquina superior izquierda
                text = f"Capturando en: {time_left}s"
                font = cv2.FONT_HERSHEY_SIMPLEX
                font_scale = 1.8
                thickness = 4

                (text_width, text_height), _ = cv2.getTextSize(text, font, font_scale, thickness)
                x = 40
                y = 40 + text_height

                overlay = display.copy()
                cv2.rectangle(
                    overlay,
                    (x - 20, y - text_height - 20),
                    (x + text_width + 20, y + 20),
                    (0, 0, 0),
                    -1,
                )
                cv2.addWeighted(overlay, 0.7, display, 0.3, 0, display)

                cv2.putText(display, text, (x, y), font, font_scale, (0, 255, 0), thickness)

                cv2.imshow("OCR - Preparando captura", display)
                if cv2.waitKey(1) & 0xFF == 27:
                    return False, None, None

        # Usar el frame final (segundo cero) como prioridad; si falla, usar el mejor frame
        if last_frame is not None:
            captured_frame = last_frame
            _, best_focus, best_brightness, best_contrast = _frame_quality_score(captured_frame)
        elif best_frame is not None:
            captured_frame = best_frame

        if captured_frame is not None:
            print(f"?? Usando frame con nitidez={best_focus:.1f}, brillo={best_brightness:.1f}, contraste={best_contrast:.1f}")

        # Mostrar la foto capturada por un instante
        if vision.show_preview and captured_frame is not None:
            display = captured_frame.copy()
            cv2.putText(display, "FOTO CAPTURADA - Procesando...", (50, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 3)
            cv2.imshow("OCR - Foto capturada", display)
            cv2.waitKey(800)

    finally:
        try:
            cv2.destroyAllWindows()
        except Exception:
            pass

    # Procesar la foto capturada
    if captured_frame is None:
        return False, None, None

    print("?? Procesando texto del documento...")

    # Convertir a escala de grises
    gray_base = cv2.cvtColor(captured_frame, cv2.COLOR_BGR2GRAY)

    best_candidate = None

    # Si hay GOOGLE_VISION_API_KEY configurada, probar primero: suele leer
    # mejor fotos reales (torcidas, con sombra) que Tesseract.
    google_text, google_conf = _google_vision_ocr(captured_frame, lang)
    if google_text:
        best_candidate = {
            "text": google_text,
            "conf": google_conf,
            "label": "google-vision",
            "angle": 0,
            "processed": {"enhanced": gray_base, "binary": gray_base, "binary_inv": gray_base},
            "gray": gray_base,
        }

    angles = [] if best_candidate else [0]  # saltar Tesseract si Google ya funciono

    for angle in angles:
        rotated_gray = _rotate_image(gray_base, angle)
        crops: List[tuple[str, np.ndarray]] = [("full", rotated_gray)]

        for crop_label, crop_gray in crops:
            processed = _preprocess_for_ocr(crop_gray)

            mean, std = _analyze_lighting(processed["original"])
            pipelines = [("nitida", processed["sharpen"]), ("binaria", processed["binary"])]

            candidate_text = ""
            candidate_conf = 0.0
            candidate_label = ""

            for label, image in pipelines:
                raw_text, conf = _extract_text(image, lang)
                cleaned = _sanitize_text(raw_text)
                if not cleaned:
                    continue
                word_count = len(cleaned.split())
                best_word_count = len(candidate_text.split()) if candidate_text else 0
                if word_count > best_word_count or (word_count == best_word_count and conf > candidate_conf):
                    candidate_text, candidate_conf, candidate_label = cleaned, conf, f"{label}-{crop_label}"
                    # Salida temprana si ya hay texto suficiente
                    if word_count >= 10:
                        best_candidate = {
                            "text": candidate_text,
                            "conf": float(candidate_conf),
                            "label": candidate_label,
                            "angle": angle,
                            "processed": processed,
                            "gray": processed["original"],
                        }
                        break

            if best_candidate:
                break
        if best_candidate:
            break

    # Fallback de ultimo recurso si no encontramos texto
    if not best_candidate or not best_candidate.get("text"):
        fb_text, fb_conf = _fallback_ocr(gray_base, lang)
        if fb_text:
            best_candidate = {
                "text": _sanitize_text(fb_text),
                "conf": float(fb_conf),
                "label": "fallback",
                "angle": 0,
                "processed": {"enhanced": gray_base, "binary": gray_base, "binary_inv": gray_base},
                "gray": gray_base,
            }

    # Intento directo simplificado para no regresar vacio
    if (not best_candidate or not best_candidate.get("text")) and gray_base is not None:
        simple_text, simple_conf = _quick_simple_ocr(gray_base, lang if lang else "spa")
        if simple_text:
            best_candidate = {
                "text": simple_text,
                "conf": float(simple_conf),
                "label": "simple",
                "angle": 0,
                "processed": {"enhanced": gray_base, "binary": gray_base, "binary_inv": gray_base},
                "gray": gray_base,
            }

    if not best_candidate or not best_candidate.get("text"):
        cv2.destroyAllWindows()
        return False, None, None

    text = best_candidate["text"]
    conf = best_candidate.get("conf", 0.0)
    best_angle = best_candidate["angle"]
    selected_processed = best_candidate["processed"]
    selected_gray = best_candidate["gray"]
    word_count = len(text.split())

    # Guardar imágenes con el mejor ángulo
    suffix = "" if best_angle == 0 else f"_rot{best_angle}"
    rotated_original = _rotate_image(captured_frame, best_angle)
    original_path = vision.ocr_dir / f"captura_original_{timestamp}{suffix}.jpg"
    gray_path = vision.ocr_dir / f"captura_gray_{timestamp}{suffix}.jpg"
    enhanced_path = vision.ocr_dir / f"captura_enhanced_{timestamp}{suffix}.jpg"
    binary_path = vision.ocr_dir / f"captura_procesada_{timestamp}{suffix}.jpg"
    binary_inv_path = vision.ocr_dir / f"captura_procesada_inv_{timestamp}{suffix}.jpg"
    sharpen_path = vision.ocr_dir / f"captura_sharp_{timestamp}{suffix}.jpg"
    otsu_path = vision.ocr_dir / f"captura_otsu_{timestamp}{suffix}.jpg"
    otsu_inv_path = vision.ocr_dir / f"captura_otsu_inv_{timestamp}{suffix}.jpg"

    cv2.imwrite(str(original_path), rotated_original)
    cv2.imwrite(str(gray_path), selected_gray)
    cv2.imwrite(str(enhanced_path), selected_processed["enhanced"])
    cv2.imwrite(str(binary_path), selected_processed["binary"])
    cv2.imwrite(str(binary_inv_path), selected_processed["binary_inv"])
    if "sharpen" in selected_processed:
        cv2.imwrite(str(sharpen_path), selected_processed["sharpen"])
    if "otsu" in selected_processed and "otsu_inv" in selected_processed:
        cv2.imwrite(str(otsu_path), selected_processed["otsu"])
        cv2.imwrite(str(otsu_inv_path), selected_processed["otsu_inv"])
    print(f"V OCR listo ({word_count} palabras, conf={conf:.1f}%). Imagen procesada: {binary_path}")

    if vision.show_preview:
        cv2.imshow("OCR - Imagen mejorada", selected_processed["enhanced"])
        cv2.imshow("OCR - Imagen binaria", selected_processed["binary"])
        cv2.imshow("OCR - Imagen binaria invertida", selected_processed["binary_inv"])
        if "otsu" in selected_processed:
            cv2.imshow("OCR - Imagen otsu", selected_processed["otsu"])
        cv2.waitKey(1500)

    result_path = vision.ocr_dir / f"resultado_{timestamp}.txt"
    with open(result_path, 'w', encoding='utf-8') as f:
        f.write(f"Frame (focus={best_focus:.1f}, brillo={best_brightness:.1f}, contraste={best_contrast:.1f})\n")
        f.write(f"Mejor ángulo: {best_angle}°\n")
        f.write(f"Pipeline elegido: {best_candidate['label'] or 'sin resultado'}\n")
        f.write(f"Confianza: {conf:.2f}%\n")
        f.write(f"Palabras detectadas: {word_count}\n")
        f.write("-" * 50 + "\n")
        f.write(text if text else "No se detectó texto")
    print(f"[OK] Resultado guardado en: {result_path}")

    cv2.destroyAllWindows()

    # Limpiar ruido de OCR (palabras sin sentido) antes de enunciarlo. El
    # archivo de resultado de arriba ya guardo el texto crudo para depurar.
    text = _clean_ocr_text_with_claude(text)

    # Considerar exitoso si hay texto, aunque la confianza sea baja (para no devolver vacío)
    ok = bool(text)
    return ok, text, conf


def leer_texto_desde_camara():
    """Activa la cámara, lee texto en tiempo real y lo muestra en consola."""
    cam = cv2.VideoCapture(0)
    if not cam.isOpened():
        print("[ERROR] No se pudo abrir la cámara.")
        return

    print("Camara encendida. Presiona 'q' para salir.")

    while True:
        ret, frame = cam.read()
        if not ret:
            print("Error al capturar frame.")
            break

        # Conversión a escala de grises y mejora de contraste
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        text = pytesseract.image_to_string(gray, lang='spa')

        # Mostrar en consola
        if text.strip():
            print("\nTexto detectado:")
            print(text.strip())

        # Mostrar vista previa
        cv2.imshow('Vista en tiempo real (OCR)', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cam.release()
    cv2.destroyAllWindows()
