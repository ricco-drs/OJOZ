#!/usr/bin/env python3
"""
Script para probar OCR en imágenes ya guardadas.
Útil para ajustar configuraciones sin recapturar.
"""

import cv2
import pytesseract
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from app.config.settings import configure_tesseract

# Configurar Tesseract (ruta centralizada en config.settings)
configure_tesseract()


def _ocr_string(img, lang: str, config: str) -> str:
    """
    Devuelve el texto OCR como str.

    pytesseract.image_to_string() decide su tipo de retorno segun output_type
    (str, bytes o dict). Con el valor por defecto Output.STRING siempre es str,
    pero su firma declara la union completa, asi que la acotamos aqui en un
    unico sitio en vez de repetir comprobaciones en cada llamada.
    """
    result = pytesseract.image_to_string(img, lang=lang, config=config)
    return result if isinstance(result, str) else str(result)


def test_image(image_path, lang='spa'):
    """
    Prueba OCR en una imagen específica con múltiples configuraciones.
    """
    print(f"\n{'='*60}")
    print(f"Analizando: {image_path}")
    print(f"{'='*60}\n")
    
    # Cargar imagen
    img = cv2.imread(str(image_path))
    if img is None:
        print(f"[ERROR] No se pudo cargar la imagen {image_path}")
        return
    
    # Mostrar info de la imagen
    height, width = img.shape[:2]
    print(f"Dimensiones: {width}x{height} píxeles\n")
    
    # Probar diferentes PSM modes
    psm_modes = {
        3: "Detección automática completa",
        4: "Una columna de texto",
        6: "Un bloque uniforme de texto",
        11: "Sparse text (detecta todo)",
        12: "Sparse text con OSD"
    }
    
    results = []
    
    for psm, description in psm_modes.items():
        print(f"PSM {psm} - {description}")
        try:
            config = f'--oem 3 --psm {psm}'

            # Obtener texto
            text = _ocr_string(img, lang, config)
            
            # Obtener confianza
            data = pytesseract.image_to_data(img, lang=lang, config=config, 
                                            output_type=pytesseract.Output.DICT)
            confidences = [int(c) for c in data['conf'] if str(c).isdigit() and int(c) > 0]
            avg_conf = sum(confidences) / len(confidences) if confidences else 0
            
            word_count = len(text.split()) if text else 0
            
            print(f"   Palabras: {word_count}")
            print(f"   Confianza: {avg_conf:.2f}%")
            
            if text and text.strip():
                results.append({
                    'psm': psm,
                    'text': text.strip(),
                    'conf': avg_conf,
                    'words': word_count
                })
                print("   [OK] Texto detectado")
            else:
                print("   [--] No se detectó texto")
            print()
            
        except Exception as e:
            print(f"   [ERROR] {e}\n")
    
    # Mostrar mejor resultado
    if results:
        best = max(results, key=lambda x: x['words'])
        print(f"\n{'='*60}")
        print(f"MEJOR RESULTADO (PSM {best['psm']})")
        print(f"{'='*60}")
        print(f"Palabras: {best['words']}")
        print(f"Confianza: {best['conf']:.2f}%")
        print("\nTexto detectado:\n")
        print(best['text'])
        print(f"\n{'='*60}\n")
    else:
        print("\n[ERROR] No se pudo detectar texto con ninguna configuración\n")


if __name__ == "__main__":
    if len(sys.argv) > 1:
        # Usar imagen especificada por argumento
        image_path = Path(sys.argv[1])
    else:
        # Buscar la última imagen procesada
        from app.config.settings import vision
        ocr_dir = vision.ocr_dir
        
        processed_images = sorted(ocr_dir.glob("captura_procesada_*.jpg"))
        
        if not processed_images:
            print("[ERROR] No se encontraron imágenes procesadas en:", ocr_dir)
            print("\nUso: python -m app.vision.test_ocr_image [ruta_imagen]")
            sys.exit(1)
        
        image_path = processed_images[-1]
        print("Usando ultima imagen capturada")
    
    test_image(image_path)
