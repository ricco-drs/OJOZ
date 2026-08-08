from __future__ import annotations
from typing import Optional
import unicodedata


def _normalize(s: str) -> str:
    """
    Normaliza a minusculas y quita acentos para comparar de forma robusta.
    """
    s = s.lower()
    return "".join(
        c for c in unicodedata.normalize("NFKD", s) if not unicodedata.combining(c)
    )


def infer_intent(text: str) -> Optional[str]:
    """
    Devuelve una intencion simple a partir del texto del usuario.
    Posibles retornos:
    - "exit", "shutdown"
    - "greet", "show_menu"
    - "new_user_option", "returning_user_option"
    - "open_camera"
    - "read_image"        (Opcion 1 - OCR, menu autenticado)
    - "currency_value"    (Opcion 2 - dinero)
    - "verify_expiry"     (Opcion 3 - vencimiento)
    - "describe_everything" (Opcion 4 - descripcion completa / enlace)
    """
    t = _normalize(text)

    # ------------------- Salida / cierre -------------------
    if "salir" in t or "exit" in t or "cerrar" in t:
        # Distinguir entre "cerrar sesion" y "cerrar aplicacion"
        if "aplicacion" in t or "programa" in t or "app" in t:
            return "shutdown"
        return "exit"

    # ------------------- Saludos -------------------
    # Nota: si el usuario responde al saludo inicial con algo como
    # "hola ojos, es mi primera vez aqui", queremos que pese mas
    # la intencion de registro/autenticacion que el simple saludo.
    # Por eso el manejo de "primera vez"/"tengo cuenta" se hace
    # mas abajo, antes de devolver greet definitivamente.
    if any(w in t for w in ["hola", "buenos dias", "buenas tardes", "buenas noches"]):
        possible_greet = True
    else:
        possible_greet = False

    # =========================================================
    # Opciones del MENU DESPUES de autenticacion
    #   primera opcion -> leer documento (OCR)
    #   segunda opcion -> valor del dinero
    #   tercera opcion -> fecha de vencimiento
    #   cuarta opcion  -> describir todo lo que ve (enlace externo)
    # (se usan normalmente DESPUES de que el usuario ya se autentico)
    # =========================================================
    if "primera" in t and "opcion" in t:
        return "read_image"

    # Segunda opcion = Reconocer dinero (mapeamos a currency_value)
    if "segunda" in t and "opcion" in t:
        return "currency_value"

    # Tercera opcion = Verificar fecha de vencimiento
    if "tercera" in t and "opcion" in t:
        return "verify_expiry"
    if "opcion 3" in t or "opcion tres" in t or "opcion numero tres" in t:
        return "verify_expiry"

    # Cuarta opcion = Describir todo lo que ve (provee enlace)
    if "cuarta" in t and "opcion" in t:
        return "describe_everything"
    if "opcion 4" in t or "opcion cuatro" in t or "opcion numero cuatro" in t:
        return "describe_everything"
    if any(phrase in t for phrase in ["modo pro", "modo unificado", "modo completo", "modo avanzado"]):
        return "describe_everything"

    # Frases alternativas para OCR
    if "leer" in t and any(k in t for k in ["documento", "texto", "imagen"]):
        return "read_image"
    if any(
        phrase in t
        for phrase in [
            "quisiera que leas por mi",
            "quiero que leas por mi",
            "quisiera que leas por mí",
            "quiero que leas por mí",
            "lees por mi un documento",
            "lees por mí un documento",
            "leer por mi un documento",
            "leer por mí un documento",
            "podrias leer por mi",
            "podrías leer por mi",
            "podrias leer por mí",
            "podrías leer por mí",
        ]
    ):
        return "read_image"

    # ======== Frases alternativas para Opcion 2 (dinero/billete/moneda) ========
    # Dispara sin necesidad de decir "segunda opcion"
    if any(
        phrase in t
        for phrase in [
            "valor del dinero",
            "valor del billete",
            "valor de la moneda",
            "cuanto vale",
            "que valor tiene",
            "decirte el valor del dinero",
            "reconocer dinero",
            "reconocer billete",
        ]
    ):
        return "currency_value"

    # Tambien si menciona dinero/billetes/monedas y pide valor/precio
    if (
        any(k in t for k in ["dinero", "billete", "billetes", "moneda", "monedas", "sol", "soles"])
        and any(k in t for k in ["valor", "vale", "precio", "monto"])
    ):
        return "currency_value"

    # ======== Frases alternativas para Opcion 3 (fecha de vencimiento) ========
    # Dispara sin necesidad de decir "tercera opcion"
    if any(
        phrase in t
        for phrase in [
            "vencimiento",
            "fecha de vencimiento",
            "fecha de caducidad",
            "caducidad",
            "esta vencido",
            "vence el",
            "expira el",
            "best before",
            "use by",
            "verificar fecha",
            "revisar fecha",
        ]
    ):
        return "verify_expiry"

    # Tambien si menciona producto/alimento y fecha/vencimiento
    if any(k in t for k in ["producto", "alimento", "comida", "medicina", "medicamento"]) and any(
        k in t for k in ["vencido", "vence", "caducado", "caduca", "fecha"]
    ):
        return "verify_expiry"

    # =========================================================
    # Flujo INICIAL (registro / autenticacion):
    #   "es mi primera vez aqui"  -> new_user_option
    #   "ya tengo una cuenta"     -> returning_user_option
    #   "opcion 1"/"opcion 2" o "1"/"2"
    # =========================================================

    # Respuestas naturales que indican que YA tiene cuenta (no es la primera vez)
    if any(
        phrase in t
        for phrase in [
            "no es mi primera vez",
            "ya he venido",
            "ya e venido",
            "ya he estado aqui",
            "no soy nuevo",
            "no soy nueva",
            "ya tengo cuenta",
            "ya tengo una cuenta",
            "ya tengo cuenta registrada",
            "ya estoy registrado",
            "ya estoy registrada",
            "tengo una cuenta registrada",
            "tengo cuenta registrada",
            "tengo cuenta",
        ]
    ):
        return "returning_user_option"

    # Respuestas naturales que indican que SI es la primera vez / no tiene cuenta
    first_time = (
        any(
            phrase in t
            for phrase in [
                "es mi primera vez",
                "mi primera vez aqui",
                "mi primera vez aca",
                "primera vez aqui",
                "primera vez aca",
                "soy nuevo",
                "soy nueva",
                "usuario nuevo",
                "soy usuario nuevo",
                "soy un usuario nuevo",
                "nunca he venido",
                "nunca e venido",
                "no tengo cuenta",
                "no tengo una cuenta",
                "no tengo cuenta registrada",
                "no estoy registrado",
                "no estoy registrada",
            ]
        )
        or ("primera vez" in t and "no es" not in t)
    )
    if first_time:
        return "new_user_option"

    # Respuestas numericas/directas al flujo inicial
    if "opcion" in t:
        if "opcion 1" in t or "opcion uno" in t:
            return "new_user_option"
        if "opcion 2" in t or "opcion dos" in t:
            return "returning_user_option"

    trimmed = t.strip()
    if trimmed in {"1", "uno"}:
        return "new_user_option"
    if trimmed in {"2", "dos"}:
        return "returning_user_option"

    # =========================================================
    # Otros intents globales
    # =========================================================

    # Solicitud de descripcion general de la escena
    if any(
        phrase in t
        for phrase in [
            "describir todo lo que veo",
            "describeme lo que ves",
            "que estas viendo",
            "que ves en la camara",
            "que ves en la cámara",
            "describir lo que ves",
            "que ves ahi",
            "que ves alli",
        ]
    ):
        return "describe_everything"

    # Solicitar menu de opciones despues de autenticacion
    pedir_menu = any(
        palabra in t for palabra in ["opciones", "menu", "que puedes hacer", "que puedo hacer", "disponibles"]
    )
    pedir_ayuda = (
        any(k in t for k in ["ayudar", "ayudas", "ayudarme", "ayuda"])
        and any(k in t for k in ["puedes", "podrias", "podrias", "puedo", "ayudas"])
    )
    if pedir_menu or pedir_ayuda:
        return "show_menu"

    # Si solo detectamos un saludo pero no hay ninguna senal
    # de flujo inicial ni de menu, tratamos el mensaje como greet.
    if possible_greet:
        return "greet"

    # Abrir camara (soporta "camara")
    if "abrir" in t and "camara" in t:
        return "open_camera"

    return None
