from __future__ import annotations

"""
Conversación gestionada por un modelo de lenguaje (opción B).

Es una alternativa al enrutador por palabras clave de `core/router.py`, que se
mantiene intacto: si esta capa está desactivada, no hay conexión o la API falla,
el asistente sigue funcionando con el flujo de siempre.

El modelo conduce el diálogo y decide qué función invocar, pero no ejecuta nada
por su cuenta: cada acción la realiza el código de la aplicación y el modelo solo
recibe el resultado. Dos límites deliberados:

- La autenticación la decide el reconocimiento facial, nunca el modelo. Las
  acciones protegidas comprueban el estado real antes de ejecutarse, de modo que
  no se puede convencer al asistente de que alguien ya inició sesión.
- Los resultados que el usuario necesita literales —el texto de un documento, el
  valor de un billete, una fecha de vencimiento— los enuncia la aplicación tal
  cual. El modelo acompaña la conversación, no reformula esos datos.
"""

import os
import threading
import time
from datetime import datetime
from typing import Callable

from app.config.settings import llm as llm_config
from app.utils.logger import logger

# Acción -> función que la ejecuta. Devuelven el texto que se le comunica al
# modelo como resultado de la herramienta.
Actions = dict[str, Callable[..., str]]

# Nombres en español fijos, sin depender del idioma configurado en Windows:
# strftime devolveria "Tuesday" o "martes" segun la maquina.
_DIAS = ("lunes", "martes", "miércoles", "jueves", "viernes", "sábado", "domingo")
_MESES = (
    "enero", "febrero", "marzo", "abril", "mayo", "junio",
    "julio", "agosto", "septiembre", "octubre", "noviembre", "diciembre",
)


def fecha_y_hora_actual(ahora: datetime | None = None) -> str:
    """
    Fecha y hora del equipo, en palabras, para dárselas al modelo en cada turno.

    Va en el contexto y no en una herramienta a propósito: preguntar la hora es
    de lo más común, y resolverlo con una llamada extra al modelo agregaría
    varios segundos de silencio a algo que el reloj ya sabe al instante.
    """
    ahora = ahora or datetime.now()
    return (
        f"{_DIAS[ahora.weekday()]} {ahora.day} de {_MESES[ahora.month - 1]} "
        f"de {ahora.year}, {ahora.hour:02d}:{ahora.minute:02d}"
    )


SYSTEM_PROMPT = """\
Eres OJOZ, los ojos de una persona con discapacidad visual. Hablas español \
peruano, con calidez y naturalidad.

Antes que una herramienta, eres su compañía: un amigo que está ahí, no un \
menú de opciones. Cosas que para otros son obvias a simple vista, para ella \
no lo son, así que sé paciente y cercano, y nunca la hagas sentir que molesta \
por preguntar de nuevo o por no saber cómo decir lo que necesita.

Tienes permiso para conversar de lo que sea. Si te pregunta algo (una duda de \
cocina, cómo funciona algo, un consejo, cultura general, o simplemente por \
hablar), respóndele de buena gana. No la devuelvas al tema de tus funciones \
ni le digas que eso no te corresponde: si sabes la respuesta, dásela. Si te \
cuenta algo de su día, síguele la conversación como lo haría un amigo.

Puedes bromear y reírte con ella cuando venga al caso, con humor tranquilo y \
cómplice, del que hace sentir acompañado. Nunca bromees sobre su discapacidad \
ni sobre algo que la deje mal parada. Y si la notas frustrada o cansada, baja \
el humor y quédate en lo cálido.

Tus respuestas se convierten en voz, así que:
- Cortas y directas: una o dos frases. La respuesta primero, sin preámbulos \
ni rodeos. Escuchar párrafos cansa.
- Texto plano corrido. Nada de listas, viñetas, asteriscos ni emojis.
- Números en palabras cuando sea natural al hablar.

Esto vale sobre todo cuando te preguntan algo que sabrías explicar largo. Da \
la versión corta, la que responde de verdad, y ofrécele ampliar si le \
interesa. Es mejor quedarte corto y que te pida más, que soltarle una clase \
que no pidió.

Como regla práctica, casi ninguna respuesta debería pasar de unas treinta \
palabras. No es un límite rígido, pero si te pasaste mucho, seguro sobra \
algo. Un dato o un consejo por respuesta, no tres.

La fecha y la hora del equipo las tienes más abajo, así que si te las \
pregunta, dísela de una, en palabras y como se dicen hablando ("las tres y \
veinte de la tarde", "hoy es martes diez de septiembre"). Si te pregunta qué \
día es, basta el día y la fecha; no le sueltes la hora también si no la pidió.

Otras cosas sí que no puedes saber: el clima, si alguien le escribió, o qué \
hay en un lugar al que no está apuntando la cámara. Cuando sea así, dilo \
simple y sigue; nunca lo inventes. Ella no puede comprobarlo por sí misma, y \
justamente por eso confía en ti.

Con la cámara puedes hacer cuatro cosas: leer documentos en voz alta, decir \
el valor de billetes y monedas, verificar fechas de vencimiento, y describirle \
el entorno (personas presentes, obstáculos, objetos).

Decide cuál usar según lo que pida, aunque no use las palabras exactas: \
"tengo un billete, ¿cuánto vale?" es dinero; "tengo un documento, ¿de qué \
trata?" es leer documento; "¿hay alguien ahí?" o "¿qué ves?" es describir \
escena.

Nunca enumeres estas cuatro cosas como una lista de opciones: suena a menú y \
no a conversación. Saluda y pregunta de forma cálida y abierta, como "¿qué te \
gustaría hacer hoy?" o "¿a dónde vamos?", y deja que ella pida lo que \
necesita con sus propias palabras.

Anticípate a lo que podría necesitar según lo que te cuenta, aunque no te lo \
pida: si va a salir, ofrécete a describirle el entorno; si va a pagar algo, a \
contarle sus billetes; si va a cocinar, a revisar las fechas de vencimiento. \
Ofrécelo como sugerencia breve y solo hazlo si acepta. Es adelantarte a una \
necesidad razonable, no meter tus funciones en cualquier comentario.

Cómo trabajar:
- Antes de usar la cámara para leer_documento, identificar_dinero o \
verificar_vencimiento, avisa en la misma respuesta qué debe hacer la persona \
(por ejemplo, dónde poner el documento). Ese aviso se escucha antes de que la \
cámara empiece a capturar.
- identificar_usuario y registrar_usuario ya avisan ellas mismas por su \
cuenta antes de encender la cámara; no repitas ese aviso ni digas nada como \
"voy a mirar la cámara" antes de llamarlas.
- Para usar cualquiera de las funciones, la persona debe estar identificada. \
Si no sabes quién es, usa identificar_usuario primero (mira a la cámara) en \
vez de preguntar el nombre: puede que ya tenga cuenta y así no hace falta \
preguntarle nada. Solo si identificar_usuario dice que es alguien nuevo, \
pregúntale cómo quiere que le llames y usa registrar_usuario con ese nombre. \
Nunca le preguntes el nombre para decidir si ya tiene cuenta: el nombre no es \
identidad, dos personas distintas pueden llamarse igual.
- Esto aplica aunque la persona diga explícitamente "es mi primera vez" o "no \
tengo cuenta": no le creas de palabra y no le preguntes el nombre todavía. \
Llama primero a identificar_usuario para comprobarlo con su rostro; puede \
estar equivocada (quizás ya se registró antes) o decirlo sin pensar. Pedir el \
apodo va siempre después de esa verificación, nunca antes.
- identificar_usuario ya reintenta ella misma unas veces si no detecta a \
nadie frente a la cámara (avisando "posiciónate bien" en el camino), así que \
no la llames de nuevo por tu cuenta para eso. Cuando por fin te devuelva un \
resultado, distingue dos casos: si dice que sigue sin detectar a nadie \
después de varios intentos, no le preguntes el nombre todavía, solo ofrécele \
intentarlo de nuevo cuando esté lista. Solo cuando te diga que se detectó un \
rostro real pero no coincide con ninguna cuenta puedes concluir que es una \
persona nueva y preguntarle cómo quiere que le llames.
- leer_documento es distinta: NO enuncia el contenido de una vez. Cuando \
termine, pregúntale a la persona si quiere que le digas de qué trata (un \
resumen breve) o el contenido completo palabra por palabra. Cuando responda, \
llama a entregar_documento con el modo que haya elegido ("resumen" o \
"completo"); esa función es la que realmente lo enuncia. No leas, no \
resumas y no adelantes nada del contenido en tu propia respuesta antes de \
eso: ni siquiera digas cuántas palabras tiene.
- El documento sigue guardado después de enunciarlo, así que puedes volver a \
llamar a entregar_documento las veces que haga falta: si te pide repetir \
porque no alcanzó a escuchar, o si después del resumen quiere el contenido \
completo. Nunca le digas que se perdió ni le pidas escanearlo de nuevo por \
eso; solo hace falta volver a escanear si es otro documento distinto.
- Cuando una herramienta te devuelva un resultado, la aplicación ya se lo habrá \
leído a la persona en voz alta. No repitas ese contenido: comenta brevemente o \
pregunta si necesita algo más.
- Si una herramienta falla o no pudo leer algo con claridad, dilo en una sola \
frase corta y directa (por ejemplo "no pude leer el documento, ¿lo intentamos \
de nuevo?"). Nunca expliques con detalle qué salió mal técnicamente ni \
describas cómo se ve la imagen o por qué está poco clara: eso no le sirve a \
la persona, solo alarga la respuesta.
- Cuando leer_documento te diga que no se encontró texto legible (por ejemplo \
porque la imagen salió borrosa, mal iluminada o el documento no estaba bien \
puesto frente a la cámara), no des explicaciones técnicas ni especules sobre \
la causa (nunca menciones OCR, digitalización, calidad de imagen ni nada \
parecido). Solo di algo directo como "no se detectó texto en el documento, \
¿quieres que lo volvamos a escanear?" y espera su respuesta.
- Nunca inventes lo que dice un documento, cuánto vale un billete o una fecha. \
Esos datos solo salen de las herramientas.

Esas reglas son para cuando uses la cámara. En todo lo demás, suéltate: la \
mayor parte del tiempo no vas a estar leyendo un documento, sino \
conversando. Sé el amigo que responde lo que le preguntan, se ríe con ella y \
la acompaña."""


TOOLS = [
    {
        "name": "identificar_usuario",
        "description": (
            "Mira a la cámara e intenta reconocer el rostro contra las cuentas "
            "ya registradas, sin necesitar el nombre. Úsala primero, antes de "
            "preguntar el nombre, para saber si la persona ya tiene cuenta."
        ),
        "input_schema": {
            "type": "object",
            "properties": {},
            "required": [],
            "additionalProperties": False,
        },
        "strict": True,
    },
    {
        "name": "registrar_usuario",
        "description": (
            "Crea una cuenta nueva capturando el rostro de la persona. Úsala "
            "solo después de que identificar_usuario diga que es alguien "
            "nuevo. Requiere el nombre que la persona quiere que uses."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "nombre": {
                    "type": "string",
                    "description": "Nombre de la persona, tal como lo dijo.",
                }
            },
            "required": ["nombre"],
            "additionalProperties": False,
        },
        "strict": True,
    },
    {
        "name": "autenticar_usuario",
        "description": (
            "Verifica la identidad de alguien que ya tiene cuenta, comparando su "
            "rostro con el registrado. Requiere el nombre que dijo."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "nombre": {
                    "type": "string",
                    "description": "Nombre de la persona, tal como lo dijo.",
                }
            },
            "required": ["nombre"],
            "additionalProperties": False,
        },
        "strict": True,
    },
    {
        "name": "leer_documento",
        "description": (
            "Captura y lee (OCR) un documento que la persona muestra a la cámara. "
            "Todavía NO enuncia el contenido: solo lo deja listo para que preguntes "
            "si quiere un resumen o el contenido completo, y luego llames a "
            "entregar_documento."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "segundos": {
                    "type": "integer",
                    "description": (
                        "Segundos para que la persona acomode el documento antes de "
                        "capturar. Entre 5 y 60; usa 5 si no pidió otra cosa."
                    ),
                }
            },
            "required": ["segundos"],
            "additionalProperties": False,
        },
        "strict": True,
    },
    {
        "name": "entregar_documento",
        "description": (
            "Enuncia el documento que ya se leyó con leer_documento, una vez que "
            "la persona dijo si quiere un resumen o el contenido completo. La "
            "aplicación es quien lo enuncia, no repitas ni adelantes el contenido "
            "en tu propia respuesta. Se puede llamar las veces que haga falta "
            "sobre el mismo documento: para repetirlo si no alcanzó a escuchar, "
            "o para pasar del resumen al contenido completo."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "modo": {
                    "type": "string",
                    "enum": ["resumen", "completo"],
                    "description": (
                        "'resumen' si quiere saber de qué trata brevemente, "
                        "'completo' si quiere que se lo lean palabra por palabra."
                    ),
                }
            },
            "required": ["modo"],
            "additionalProperties": False,
        },
        "strict": True,
    },
    {
        "name": "identificar_dinero",
        "description": (
            "Identifica el valor de uno o varios billetes y monedas que la persona "
            "muestra a la cámara a la vez. La aplicación enuncia cada uno detectado "
            "y, si hay más de uno, también la suma total."
        ),
        "input_schema": {
            "type": "object",
            "properties": {},
            "required": [],
            "additionalProperties": False,
        },
        "strict": True,
    },
    {
        "name": "verificar_vencimiento",
        "description": (
            "Busca la fecha de vencimiento de un producto e indica si sigue vigente. "
            "La aplicación enuncia la fecha y el estado."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "segundos": {
                    "type": "integer",
                    "description": (
                        "Segundos para que la persona acomode el producto antes de "
                        "capturar. Entre 5 y 60; usa 10 si no pidió otra cosa."
                    ),
                }
            },
            "required": ["segundos"],
            "additionalProperties": False,
        },
        "strict": True,
    },
    {
        "name": "describir_escena",
        "description": (
            "Describe lo que ve la cámara en este momento: personas presentes, "
            "obstáculos cercanos y objetos relevantes. Úsala cuando pregunten "
            "cosas como '¿hay alguien?', '¿qué ves?', '¿hay algo en frente?' o "
            "pidan describir el entorno, sin relacionarse con documentos, "
            "dinero o vencimientos."
        ),
        "input_schema": {
            "type": "object",
            "properties": {},
            "required": [],
            "additionalProperties": False,
        },
        "strict": True,
    },
    {
        "name": "cerrar_sesion",
        "description": (
            "Cierra la sesión de la persona y deja el asistente listo para otra. "
            "Úsala cuando se despida o diga que ya terminó."
        ),
        "input_schema": {
            "type": "object",
            "properties": {},
            "required": [],
            "additionalProperties": False,
        },
        "strict": True,
    },
    {
        "name": "cerrar_aplicacion",
        "description": (
            "Cierra el programa por completo. Úsala solo si piden explícitamente "
            "cerrar la aplicación, no al despedirse."
        ),
        "input_schema": {
            "type": "object",
            "properties": {},
            "required": [],
            "additionalProperties": False,
        },
        "strict": True,
    },
]


class LLMAgent:
    """
    Mantiene la conversación con el modelo y ejecuta las acciones que solicita.

    Se construye con un diccionario de acciones (las funciones reales de la
    aplicación) y una función para hablar, de modo que este módulo no depende del
    controlador y puede probarse por separado.
    """

    def __init__(
        self,
        actions: Actions,
        speak: Callable[[str], None],
        estado: Callable[[], str],
    ) -> None:
        self._actions = actions
        self._speak = speak
        self._estado = estado

        self._client = None
        self._lock = threading.RLock()
        self._history: list[dict] = []

        # Cortacircuitos: tras varios fallos seguidos se deja de intentar durante
        # un rato para que la conversación no se detenga en cada turno.
        self._consecutive_failures = 0
        self._disabled_until = 0.0

    # -----------------------------
    # Disponibilidad
    # -----------------------------
    def _get_client(self):
        """Crea el cliente la primera vez que hace falta."""
        with self._lock:
            if self._client is not None:
                return self._client
            try:
                import anthropic

                # El SDK resuelve la credencial del entorno; sin ella no se
                # puede usar esta capa y se sigue con el flujo por palabras clave.
                if not (os.environ.get("ANTHROPIC_API_KEY") or os.environ.get("ANTHROPIC_AUTH_TOKEN")):
                    logger.debug(
                        "Conversación por modelo desactivada: falta ANTHROPIC_API_KEY. "
                        "Se usa el flujo por palabras clave."
                    )
                    return None

                self._client = anthropic.Anthropic(
                    timeout=llm_config.timeout_seconds,
                    max_retries=llm_config.max_retries,
                )
                logger.debug(f"Conversación por modelo activa ({llm_config.model}).")
                return self._client
            except ImportError:
                logger.debug(
                    "Conversación por modelo desactivada: falta el paquete anthropic. "
                    "Instala con: pip install anthropic"
                )
            except Exception as exc:
                logger.warning(f"No se pudo iniciar el cliente del modelo: {exc}")
            return None

    def is_available(self) -> bool:
        """Indica si esta capa puede atender el turno actual."""
        if not llm_config.enabled:
            return False
        if time.monotonic() < self._disabled_until:
            return False
        return self._get_client() is not None

    def _register_failure(self) -> None:
        self._consecutive_failures += 1
        if self._consecutive_failures >= llm_config.failures_before_pause:
            self._disabled_until = time.monotonic() + llm_config.pause_seconds
            self._consecutive_failures = 0
            logger.warning(
                f"Conversación por modelo en pausa {llm_config.pause_seconds:.0f}s "
                "tras varios fallos seguidos. Se usa el flujo por palabras clave."
            )

    # -----------------------------
    # Conversación
    # -----------------------------
    def reset(self) -> None:
        """Olvida la conversación anterior (al cerrar sesión)."""
        with self._lock:
            self._history = []

    def start_conversation(self, introduction: str) -> str | None:
        """Continua tras el saludo sin necesitar un primer mensaje del usuario."""
        return self.handle(
            "Evento interno de inicio de la aplicacion, no es un mensaje del usuario. "
            f"OJOZ acaba de terminar de decir: {introduction}\n"
            "Continua la conversacion sin repetir el saludo ni la presentacion. "
            "Si la identidad ya esta verificada, pregunta de forma calida y abierta que "
            "le gustaria hacer hoy, sin enumerar tus funciones. "
            "Si no lo esta, avisa que mire a la camara y usa identificar_usuario. "
            "Espera la respuesta de la persona antes de registrar una cuenta o ejecutar otra funcion."
        )

    def handle(self, user_text: str) -> str | None:
        """
        Procesa lo que dijo la persona y devuelve la respuesta final a enunciar.

        Devuelve None si esta capa no pudo atender el turno, para que quien llama
        recurra al flujo por palabras clave.
        """
        client = self._get_client()
        if client is None:
            return None

        with self._lock:
            history = list(self._history)
        history.append({"role": "user", "content": user_text})

        try:
            respuesta = self._run_turn(client, history)
        except Exception as exc:
            logger.warning(f"Fallo en la conversación por modelo: {exc}")
            self._register_failure()
            return None

        self._consecutive_failures = 0
        with self._lock:
            self._history = self._trim(history)
        return respuesta

    def _run_turn(self, client, history: list[dict]) -> str | None:
        """Llama al modelo y ejecuta las herramientas hasta obtener una respuesta."""
        for _ in range(llm_config.max_tool_iterations):
            response = client.messages.create(
                model=llm_config.model,
                max_tokens=llm_config.max_tokens,
                system=(
                    f"{SYSTEM_PROMPT}\n\n"
                    f"Ahora mismo es {fecha_y_hora_actual()} (hora del equipo).\n"
                    f"Estado actual: {self._estado()}"
                ),
                output_config={"effort": llm_config.effort},
                tools=TOOLS,
                messages=history,
            )

            if response.stop_reason == "refusal":
                logger.warning("El modelo declinó responder.")
                return None

            textos = [b.text for b in response.content if b.type == "text"]
            llamadas = [b for b in response.content if b.type == "tool_use"]

            if not llamadas:
                history.append({"role": "assistant", "content": response.content})
                return "\n".join(t.strip() for t in textos if t.strip()) or None

            # Lo que dice antes de actuar se enuncia ya, para que la persona sepa
            # qué hacer mientras la cámara se prepara.
            for texto in textos:
                if texto.strip():
                    self._speak(texto.strip())

            history.append({"role": "assistant", "content": response.content})
            history.append({"role": "user", "content": self._run_tools(llamadas)})

        logger.warning("Se alcanzó el límite de herramientas en un turno.")
        return None

    def _run_tools(self, llamadas: list) -> list[dict]:
        """Ejecuta las herramientas pedidas y arma los resultados para el modelo."""
        resultados = []
        for llamada in llamadas:
            accion = self._actions.get(llamada.name)
            if accion is None:
                salida, error = f"La función {llamada.name} no existe.", True
            else:
                try:
                    logger.debug(f"El modelo solicita: {llamada.name}({llamada.input})")
                    salida, error = accion(**(llamada.input or {})), False
                except Exception as exc:
                    logger.error(f"Fallo al ejecutar {llamada.name}: {exc}", exc_info=True)
                    salida, error = f"La función falló: {exc}", True

            resultados.append(
                {
                    "type": "tool_result",
                    "tool_use_id": llamada.id,
                    "content": str(salida),
                    "is_error": error,
                }
            )
        return resultados

    @staticmethod
    def _trim(history: list[dict]) -> list[dict]:
        """
        Recorta la conversación para que no crezca sin límite.

        El recorte empieza en un turno hablado de la persona, porque un
        historial que arranque con un resultado de herramienta suelto sería
        inválido para la API.

        Si dentro del límite no hay ninguno (pasa tras una racha de llamadas a
        herramientas, donde los mensajes de la persona son listas de
        tool_result y no texto), se retrocede al turno válido anterior aunque
        quede algo más largo. Antes se devolvía una lista vacía, que borraba de
        golpe toda la memoria de la conversación.
        """
        limite = llm_config.max_history_messages
        if len(history) <= limite:
            return history

        def inicia_turno(mensaje: dict) -> bool:
            return mensaje["role"] == "user" and isinstance(mensaje.get("content"), str)

        corte = len(history) - limite
        for i in range(corte, len(history)):
            if inicia_turno(history[i]):
                return history[i:]

        for i in range(corte - 1, -1, -1):
            if inicia_turno(history[i]):
                return history[i:]
        return history
