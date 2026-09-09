"""
Interfaz Ultra Moderna con Flet para OJOZ
Diseño Glassmorphism con animaciones y efectos premium
"""
import flet as ft
from datetime import datetime
from threading import Thread
import cv2
import base64
import time
import unicodedata
import asyncio


class OJOZApp:
    def __init__(self, controller=None):
        self.controller = controller
        self.page = None
        self.chat_messages = None
        self.camera_preview = None
        self.camera_container = None
        self.status_text = None
        self.status_badge = None
        self.camera_active = False
        self.preview_thread = None
        self.header_container = None
        self.animate_header = True
        # Empujar para hablar (tecla espacio)
        self._space_ptt_active = False
        self._start_push_to_talk_listener()

        # Importar event_bus GLOBAL
        from app.core.event_bus import event_bus
        self.event_bus = event_bus
        
        # Suscribirse a eventos
        self.event_bus.subscribe("ui:print", self._on_ui_print)
        self.event_bus.subscribe("tts:start", self._on_tts_start)
        self.event_bus.subscribe("tts:end", self._on_tts_end)
        self.event_bus.subscribe("stt:text", self._on_stt_text)
        # La camara se muestra en una ventana de OpenCV aparte (con deteccion
        # de personas y objetos en vivo), no dentro de la interfaz de Flet:
        # _on_camera_start/_on_camera_stop quedan sin usar a proposito.

        print("[UI DEBUG] Event bus subscriptions registered!")
    
    # -----------------------------
    # Helpers para el arranque
    # -----------------------------
    def attach_controller(self, controller) -> None:
        """Permite asignar el controller real una vez que termina el bootstrap."""
        self.controller = controller
        self.update_status_badge(self._status_label())

    # -----------------------------
    # Empujar para hablar (tecla espacio)
    # -----------------------------
    def _start_push_to_talk_listener(self) -> None:
        """
        Escucha la tecla espacio a nivel de sistema operativo con pynput, en
        vez de Flet.Page.on_keyboard_event: Flutter (el motor detras de Flet)
        separa la repeticion de una tecla mantenida en un evento aparte que
        Flet no reenvia, asi que no hay forma confiable de saber cuando se
        suelta usando solo eventos de Flet. pynput sí distingue presionar de
        soltar de verdad, sin depender de repeticiones.

        Nota: al ser un listener global, la tecla espacio activa el
        microfono aunque la ventana de OJOZ no tenga el foco en ese momento.
        """
        try:
            from pynput import keyboard
        except ImportError:
            print("[UI WARNING] pynput no esta instalado; empujar-para-hablar no funcionara. Instala con: pip install pynput")
            return

        def _on_press(key) -> None:
            if key != keyboard.Key.space or self._space_ptt_active:
                return
            self._space_ptt_active = True
            if self.controller:
                self.controller.stt.set_push_to_talk(True)

        def _on_release(key) -> None:
            if key != keyboard.Key.space or not self._space_ptt_active:
                return
            self._space_ptt_active = False
            if self.controller:
                self.controller.stt.set_push_to_talk(False)

        listener = keyboard.Listener(on_press=_on_press, on_release=_on_release)
        listener.daemon = True
        listener.start()

    def _run_on_ui(self, fn) -> None:
        """Serializa los cambios de controles en el bucle de la pagina."""
        if not self.page:
            return

        async def update():
            try:
                fn()
            except Exception as e:
                print(f"[UI ERROR] No se pudo actualizar la interfaz: {e}")

        try:
            self.page.run_task(update)
        except Exception as e:
            print(f"[UI WARNING] No se pudo programar la actualizacion de UI: {e}")

    def update_status_badge(self, text: str, color: str = "#ffffff") -> None:
        """Actualiza el badge superior (thread-safe)."""
        if not self.status_text or not self.page:
            return

        def _update():
            self.status_text.value = text
            self.status_text.color = color
            self.page.update()

        self._run_on_ui(_update)

    def show_bootstrap_message(self, text: str) -> None:
        """
        Muestra mensajes informativos del arranque sin llenar el chat.
        Se imprime solo en la consola para mantener limpia la interfaz.
        """
        print(f"[BOOTSTRAP] {text}")

    def current_user_name(self) -> str:
        if self.controller:
            return getattr(self.controller, "_user_name", "Usuario")
        return "Iniciando..."

    # Alias historico mantenido por compatibilidad.
    _current_user_name = current_user_name

    def _status_label(self) -> str:
        """Texto amigable para el badge superior."""
        name = self.current_user_name()
        if self.controller and getattr(self.controller, "_authenticated", False) and name and name not in ("Usuario", "Iniciando..."):
            return f"Usuario verificado: {name}"
        if name and name not in ("Usuario", "Iniciando..."):
            return f"Usuario: {name}"
        return "Esperando usuario..."
    
    def _on_ui_print(self, **kwargs):
        """Agregar mensaje al chat cuando el sistema habla"""
        text = kwargs.get("text", "")
        role = kwargs.get("role", "sys")

        def _norm(s: str) -> str:
            if not s:
                return ""
            return "".join(c for c in unicodedata.normalize("NFKD", s) if not unicodedata.combining(c)).lower()
        
        # Lista de palabras clave para OCULTAR del chat (solo mensajes técnicos)
        FILTROS_OCULTAR = [
            # Mensajes técnicos de micrófono
            "Microfono calibrado",
            "Vosk no configurado",
            "gTTS",
            "Vosk model cargado",
            "Microfono en pausa",
            "Microfono activo (escuchando",
            "idx=",
            "nombre='",
            "device_index",
            "VOSK_MODEL",
            "Google/Sphinx",
            "Usando Google Speech para transcribir",
            
            # Mensajes técnicos de autenticación (solo los muy técnicos)
            "Resultado autenticacion:",
            "ok=",
            "confianza=",
            "esperado=",
            "Verificando identidad...",
            "Verificando si ya tienes una cuenta...",
            "V Autenticacion exitosa:",
            "Usuario en BD:",
            "Usuario '",
            "registrado exitosamente en la base de datos",
            "Modelo actualizado:",
            "Autenticacion exitosa",
            
            # Mensajes técnicos de enrolamiento
            "=== INICIO ENROLAMIENTO:",
            "Carpeta de captura:",
            "Capturadas",
            "fotos validas",
            "Enrollment registrado con id=",
            "Se capturaron",
            "rostros. Creando galeria facial",
            "Generando embeddings ArcFace",
            "Procesando",
            "Iniciando captura de 20 rostros",
            "Captura finalizada:",
            "Captura completada:",
            "fotos en la BD",
            "imagenes de",
            "Cargadas",
            "imagenes validas de",
            "Galeria ArcFace",
            "rostros de",
            "personas",
            "Modelo guardado exitosamente en:",
            "Modelo registrado en BD con id=",
            "Modelo entrenado exitosamente:",
            "=== FIN ENROLAMIENTO EXITOSO:",
            
            # Mensajes técnicos de OCR
            "?? Preparando camara",
            "Preparando camara para capturar documento",
            "Preparando cámara para capturar documento",
            "Preparando camara para verificar fecha de vencimiento",
            "Preparando cámara para verificar fecha de vencimiento",
            "No encontre texto en el documento",
            "conf=",
            "OCR - imagen procesada",
            "(No responde)",
            # Mensajes de errores/estado de audio que no deben ir al chat
            "reconocimiento fallido (confianza",
            "recalibrando microfono para escucharte mejor",
            "microfono listo, intenta nuevamente",
            "No pude leer la fecha de vencimiento",
            "No pude leer la fecha de vencimiento con claridad",
            "No se pudo calibrar el microfono",
            "No se pudo grabar del microfono",
            "No se pudo leer el microfono",
            "El microfono capta audio, pero no se pudo conectar",
            "No se detecta tu voz. Comprueba que el microfono",
            "Se recibe audio, pero no se entienden las palabras",
            "No se pudo transcribir el audio",
            # Regla general: cualquier aviso tecnico que mencione ElevenLabs
            # (fallas de la voz o de la transcripcion) es solo para consola,
            # sin importar como cambie la redaccion exacta del mensaje.
            "elevenlabs",
        ]

        # Verificar si el mensaje debe ocultarse del chat (solo roles técnicos)
        if role not in ("app/tts", "user"):
            # Regla genérica: cualquier mensaje de estado con prefijo tipo
            # "[OK] ...", "[FALLO] ..." o "[ERROR ...] ..." es informativo/
            # técnico, nunca para el chat (evita tener que listar cada frase
            # nueva una por una).
            stripped = (text or "").lstrip()
            if stripped.startswith(("[OK]", "[FALLO]", "[ERROR")):
                print(f"[TERMINAL ONLY] {text}")
                return

            norm_text = _norm(text)
            debe_ocultar = any(_norm(filtro) in norm_text for filtro in FILTROS_OCULTAR)
            if debe_ocultar:
                # Solo mostrar en terminal, NO en chat
                print(f"[TERMINAL ONLY] {text}")
                return

        # Mostrar mensajes según el rol usando método thread-safe
        if role == "user":
            # Normalizar nombre del asistente cuando el usuario dice "ojos"
            if text:
                text = (
                    text.replace("ojos", "OJOZ")
                        .replace("Ojos", "OJOZ")
                        .replace("OJOS", "OJOZ")
                )
            # Mensaje del USUARIO (verde, derecha)
            self._safe_add_message(text, is_user=True)
        elif role in ("sys", "app/tts"):
            # Mensaje del SISTEMA/TTS (azul, izquierda)
            self._safe_add_message(text, is_user=False)
        else:
            # Cualquier otro rol: mostrar como sistema para no perder mensajes
            self._safe_add_message(text, is_user=False)

    def _safe_add_message(self, text: str, is_user: bool):
        """Añadir mensaje de forma thread-safe"""
        if not self.page:
            return
        
        self._run_on_ui(lambda: self.add_message(text, is_user))
    
    def _on_tts_start(self, **kwargs):
        """Indicar que el sistema está hablando con animación y ondas de sonido"""
        if self.status_text and self.page:
            def update():
                self.status_text.value = "Reproduciendo..."
                self.status_text.color = "#00f5a0"
                if self.status_badge:
                    self.status_badge.border = ft.Border.all(2, "#00f5a080")
                    self.status_badge.bgcolor = "#00f5a020"
                    self.status_badge.shadow = ft.BoxShadow(
                        spread_radius=5,
                        blur_radius=25,
                        color="#00f5a060",
                        offset=ft.Offset(0, 5),
                    )
                    self.status_badge.animate = ft.Animation(500, ft.AnimationCurve.EASE_IN_OUT)
                self.page.update()
            
            self._run_on_ui(update)
    
    def _on_tts_end(self, **kwargs):
        """Volver al estado normal con transici¢n suave"""
        if self.status_text and self.page:
            def update():
                self.status_text.value = self._status_label()
                self.status_text.color = "#ffffff"
                if self.status_badge:
                    self.status_badge.border = ft.Border.all(1, "#0D1F2330")
                    self.status_badge.bgcolor = "#1a1f3a80"
                    self.status_badge.shadow = ft.BoxShadow(
                        spread_radius=0,
                        blur_radius=15,
                        color="#0D1F2320",
                        offset=ft.Offset(0, 5),
                    )
                self.page.update()
            
            self._run_on_ui(update)

    def _on_camera_start(self, **kwargs):
        """Activar preview de cámara"""
        self.camera_active = True
        if self.camera_preview and self.page:
            def update():
                self.camera_preview.visible = True
                if self.camera_container:
                    self.camera_container.visible = True
                self.page.update()

            self._run_on_ui(update)

            # Iniciar thread de actualización de preview
            if not self.preview_thread or not self.preview_thread.is_alive():
                self.preview_thread = Thread(target=self._update_camera_preview, daemon=True)
                self.preview_thread.start()

    def _on_camera_stop(self, **kwargs):
        """Desactivar preview de cámara"""
        self.camera_active = False
        if self.camera_preview and self.page:
            def update():
                self.camera_preview.visible = False
                if self.camera_container:
                    self.camera_container.visible = False
                self.page.update()
            
            self._run_on_ui(update)
    
    def _update_camera_preview(self):
        """
        Actualizar preview de cámara en tiempo real.

        Lee de la cámara compartida de sesión (camera_service) en vez de abrir
        su propio cv2.VideoCapture: Windows solo deja que un proceso tenga el
        dispositivo abierto a la vez, así que un segundo "open" aparte fallaba
        en silencio y la previsualización nunca aparecía.
        """
        from app.vision.camera_service import camera_service

        while self.camera_active:
            frame = camera_service.get_frame()
            if frame is not None:
                frame = cv2.resize(frame, (320, 240))
                _, buffer = cv2.imencode('.jpg', frame)
                img_base64 = base64.b64encode(buffer).decode()

                if self.camera_preview and self.page:
                    def update_preview():
                        self.camera_preview.src = f"data:image/jpeg;base64,{img_base64}"
                        self.page.update()

                    self._run_on_ui(update_preview)
            time.sleep(0.1)  # 10 FPS
    
    def add_message(self, text, is_user=False):
        """Agregar mensaje al chat con diseño premium y animaciones"""
        if not self.chat_messages or not self.page:
            return
        
        timestamp = datetime.now().strftime("%H:%M")
        
        # Mismo estilo para ambos roles (uniforme): solo cambian la alineación
        # (ver mas abajo) y la etiqueta/avatar.
        text_color = "#ffffff"
        timestamp_color = "#AFB3B7"
        shadow_color = "#00000035"
        avatar_label = "Tu" if is_user else "OJOZ"
        
        # Contenedor del mensaje con glassmorphism y animación
        message_container = ft.Container(
            col={"xs": 12, "md": 6},
            content=ft.Column([
                ft.Row([
                    ft.Container(
                        content=ft.Image(
                            src="images/PerroOjoz.png",
                            width=28,
                            height=28,
                            fit=ft.BoxFit.CONTAIN,
                        ) if not is_user else ft.Text(avatar_label, size=11, color=text_color),
                        bgcolor="#0D1F2340",
                        border_radius=20,
                        padding=2,
                        width=32,
                        height=32,
                        alignment=ft.Alignment.CENTER,
                    ),
                    ft.Text(
                        "OJOZ" if not is_user else "Tú",
                        size=13,
                        color=timestamp_color,
                        weight=ft.FontWeight.BOLD,
                        font_family="Poppins",
                    ),
                ], spacing=8),
                ft.Text(
                    text,
                    color=text_color,
                    size=18,
                    weight=ft.FontWeight.W_400,
                    font_family="Poppins",
                    selectable=True,
                    no_wrap=False,
                ),
                ft.Row([
                    ft.Text(
                        timestamp,
                        color=timestamp_color,
                        size=12,
                        font_family="Poppins",
                    ),
                ], alignment=ft.MainAxisAlignment.END),
            ], spacing=10, horizontal_alignment=ft.CrossAxisAlignment.STRETCH),
            gradient=None,
            bgcolor="#0F1822BB",
            border_radius=20,
            padding=22,
            opacity=1,
            margin=ft.Margin.only(bottom=10),
            border=None,
            shadow=ft.BoxShadow(
                spread_radius=0,
                blur_radius=15,
                color=shadow_color,
                offset=ft.Offset(0, 5),
            ),
            animate=ft.Animation(300, ft.AnimationCurve.EASE_OUT),
        )
        
        # Alinear según quién envía
        row = ft.ResponsiveRow(
            [message_container],
            alignment=ft.MainAxisAlignment.START if not is_user else ft.MainAxisAlignment.END,
            spacing=0,
            run_spacing=0,
        )
        
        self.chat_messages.controls.append(row)
        
        # Actualizar inmediatamente para que el control aparezca en el DOM
        try:
            self.page.update()
        except Exception as e:
            print(f"[UI ERROR] No se pudo mostrar el mensaje: {e}")
        
        # El desplazamiento es opcional; la burbuja ya debe ser visible.
        async def scroll_to_message():
            # Espera a que Flutter mida la nueva burbuja, especialmente si ocupa
            # varias lineas, antes de calcular el extremo inferior del chat.
            await asyncio.sleep(0.1)
            try:
                await self.chat_messages.scroll_to(offset=-1, duration=100)
                await asyncio.sleep(0.15)
                await self.chat_messages.scroll_to(offset=-1, duration=0)
            except Exception as e:
                print(f"[SCROLL ERROR] {e}")
        try:
            self.page.run_task(scroll_to_message)
        except Exception as e:
            print(f"[UI WARNING] No se pudo desplazar el chat: {e}")
    
    def _on_stt_text(self, **kwargs):
        """Cada transcripcion confirmada se muestra como un mensaje del usuario."""
        text = (kwargs.get("text") or "").strip()
        if text:
            self._on_ui_print(role="user", text=text)
    
    def build(self, page: ft.Page):
        """Construir la interfaz ultra moderna"""
        self.page = page
        page.title = "OJOZ - AI Vision Assistant"
        page.theme_mode = ft.ThemeMode.DARK
        page.fonts = {
            "Poppins": "https://raw.githubusercontent.com/google/fonts/master/ofl/poppins/Poppins-Regular.ttf",
        }
        page.padding = 0
        page.window.resizable = True
        page.bgcolor = "#9CA0A5"  # Fondo azul oscuro premium

        # Configurar ventana maximizada
        page.window.maximized = True
        page.window.always_on_top = False
        
        # Header moderno con gradiente animado
        self.header_container = ft.Container(
            content=ft.Column([
                ft.Row([
                    ft.Icon(ft.Icons.VISIBILITY, color="#C9762E", size=40),
                    ft.Column([
                        ft.Text(
                            "OJOZ",
                            size=42,
                            weight=ft.FontWeight.W_900,
                            color="#F28D35",
                            font_family="Poppins",
                            text_align=ft.TextAlign.CENTER,
                            style=ft.TextStyle(
                                letter_spacing=2,
                            ),
                        ),
                        ft.Text(
                            "Asistente de Visión Artificial",
                            size=20,
                            color="#D3833C",
                            weight=ft.FontWeight.W_500,
                            font_family="Poppins",
                            text_align=ft.TextAlign.CENTER,
                            italic=True,
                            style=ft.TextStyle(
                                letter_spacing=1,
                            ),
                        ),
                    ], spacing=0),
                ], alignment=ft.MainAxisAlignment.CENTER, spacing=15),
            ]),
            # Fondo liso semi-transparente (sin degradado)
            bgcolor="#132E3580",
            padding=ft.Padding.only(left=20, right=20, top=15, bottom=12),
            # Esquinas simétricas (arriba y abajo)
            border_radius=ft.BorderRadius.all(25),
            # Separar el header de los bordes de la ventana
            margin=ft.Margin.only(left=10, right=10, top=10),
            shadow=ft.BoxShadow(
                spread_radius=0,
                blur_radius=30,
                color="#00000080",
                offset=ft.Offset(0, 10),
            ),
            animate=ft.Animation(1000, ft.AnimationCurve.EASE_IN_OUT),
        )
        
        # Franja sin animación de gradiente (estática)
        self.animate_header = False
        
        header = self.header_container
        
        # Funciones disponibles en horizontal con indicador de usuario
        self.status_text = ft.Text(
            self._status_label(),
            size=14,
            color="#ffffff",
            weight=ft.FontWeight.W_500,
        )
        self.status_badge = ft.Container(
            content=self.status_text,
            bgcolor="#1a1f3a80",
            border_radius=20,
            padding=ft.Padding.symmetric(horizontal=20, vertical=10),
            border=ft.Border.all(1, "#0D1F2330"),
            shadow=ft.BoxShadow(
                spread_radius=0,
                blur_radius=15,
                color="#0D1F2320",
                offset=ft.Offset(0, 5),
            ),
        )
        
        status_bar = ft.Container(
            content=ft.Row([
                # Funciones horizontales
                ft.Container(
                    content=ft.Row([
                        ft.Text(
                            "Funciones Disponibles",
                            size=16,
                            color="#ffffff",
                            weight=ft.FontWeight.BOLD,
                        ),
                        self._create_compact_option("1", "Leer\nDocumento", ft.Icons.DESCRIPTION, "#0D1F23"),
                        self._create_compact_option("2", "Identificar\nDinero", ft.Icons.PAYMENTS, "#0D1F23"),
                        self._create_compact_option("3", "Verificar\nVencimiento", ft.Icons.EVENT, "#0D1F23"),
                    ], spacing=15, alignment=ft.MainAxisAlignment.START),
                    expand=True,
                ),
                # Indicador de usuario
                self.status_badge,
            ], alignment=ft.MainAxisAlignment.SPACE_BETWEEN),
            bgcolor="transparent",
            padding=15,
        )
        
        # Chat messages con scroll automático - mensajes nuevos siempre visibles abajo
        self.chat_messages = ft.ListView(
            spacing=12,
            padding=ft.Padding.only(left=10, right=10, top=0, bottom=10),
            auto_scroll=True,  # Auto-scroll al agregar nuevos mensajes
            expand=True,
        )
        
        chat_container = ft.Container(
            content=self.chat_messages,
            bgcolor="transparent",
            expand=True,
        )
        
        # Camera preview con diseño moderno (inicialmente oculto)
        self.camera_preview = ft.Image(
            src="",
            width=340,
            height=255,
            fit=ft.BoxFit.COVER,
            visible=False,
            border_radius=20,
        )
        
        camera_container = ft.Container(
            content=ft.Column([
                ft.Container(
                    content=ft.Row([
                        ft.Container(
                            content=ft.Icon(ft.Icons.FIBER_MANUAL_RECORD, color="#ff6b6b", size=12),
                            animate_opacity=ft.Animation(800, ft.AnimationCurve.EASE_IN_OUT),
                        ),
                        ft.Icon(ft.Icons.CAMERA_ALT, color="#ff6b6b", size=20),
                        ft.Text(
                            "Vista en Vivo",
                            size=15,
                            color="#ffffff",
                            weight=ft.FontWeight.BOLD,
                        ),
                    ], spacing=8),
                    bgcolor="#1a1f3a80",
                    border_radius=15,
                    padding=10,
                    border=ft.Border.all(1, "#ff6b6b30"),
                    shadow=ft.BoxShadow(
                        spread_radius=0,
                        blur_radius=15,
                        color="#ff6b6b40",
                        offset=ft.Offset(0, 3),
                    ),
                ),
                ft.Container(
                    content=self.camera_preview,
                    bgcolor="#000000",
                    border_radius=20,
                    border=ft.Border.all(2, "#ff6b6b40"),
                    shadow=ft.BoxShadow(
                        spread_radius=0,
                        blur_radius=20,
                        color="#ff6b6b30",
                        offset=ft.Offset(0, 5),
                    ),
                    animate_scale=ft.Animation(300, ft.AnimationCurve.EASE_OUT),
                ),
            ], horizontal_alignment=ft.CrossAxisAlignment.CENTER, spacing=10),
            bgcolor="transparent",
            padding=15,
            visible=False,  # Oculto por defecto hasta que se active la cámara
        )
        self.camera_container = camera_container
        

        
        # Carta de presentación de OJOZ con imagen del personaje y globo de diálogo
        presentation_card = ft.Container(
            content=ft.Row([
                # Imagen del personaje OJOZ con fondo transparente
                ft.Container(
                    content=ft.Image(
                        src="images/PerroOjoz.png",
                        width=360,
                        height=360,
                        fit=ft.BoxFit.CONTAIN,
                    ),
                    bgcolor="transparent",
                    padding=ft.Padding.only(left=10, right=10, top=0, bottom=10),
                    margin=ft.Margin.only(left=20, right=20, top=0, bottom=20),
                ),
                # Texto de presentación
                ft.Container(
                        content=ft.Column([
                            ft.Text(
                                "¡Hola! Soy OJOZ",
                                size=28,
                                weight=ft.FontWeight.BOLD,
                                color="#8C311C",
                                font_family="Poppins",
                            ),
                            ft.Container(height=10),
                            ft.Text(
                                "Tu asistente de visión artificial",
                                size=18,
                                color="#F2B33D",
                                weight=ft.FontWeight.W_500,
                                italic=True,
                                font_family="Poppins",
                            ),
                            ft.Container(height=15),
                            ft.Text(
                                "Estoy aquí para ayudarte con:",
                                size=16,
                                color="#F2D43D",
                                weight=ft.FontWeight.W_600,
                                font_family="Poppins",
                            ),
                            ft.Container(height=10),
                            ft.Column([
                                ft.Text("Lectura de documentos", size=14, color="#ffffff", font_family="Poppins"),
                                ft.Text("Identificación de billetes y monedas", size=14, color="#ffffff", font_family="Poppins"),
                                ft.Text("Verificación de fechas de vencimiento", size=14, color="#ffffff", font_family="Poppins"),
                                ft.Text("Descripción del entorno", size=14, color="#ffffff", font_family="Poppins"),
                            ], spacing=8),
                        ], spacing=0),
                        bgcolor="transparent",
                        border_radius=25,
                        padding=ft.Padding.only(left=10, right=10, top=0, bottom=10),
                    ),
            ], alignment=ft.MainAxisAlignment.START, vertical_alignment=ft.CrossAxisAlignment.CENTER, spacing=10),
            bgcolor="transparent",
            padding=ft.Padding.only(left=10, right=10, top=0, bottom=10),
        )

        # Agregar la carta de presentación al chat
        if self.chat_messages:
            self.chat_messages.controls.append(presentation_card)
            # Actualizar para que aparezca la carta
            try:
                page.update()
            except:
                pass

        # Layout principal con degradado de fondo (solo los 3 colores solicitados)
        main_container = ft.Container(
            content=ft.Column([
                header,
                chat_container,  # Ya tiene expand=True dentro
                camera_container,
            ], spacing=0, expand=True),
            gradient=ft.LinearGradient(
                begin=ft.Alignment.TOP_LEFT,
                end=ft.Alignment.BOTTOM_RIGHT,
                colors=["#0B1220", "#011126", "#8C311C"],
                stops=[0.0, 0.6, 1.0],
            ),
            expand=True,
        )
        
        # Indicador flotante del estado del microfono (esquina superior derecha)
        self.mic_icon = ft.Icon(ft.Icons.MIC_OFF, color="#ff6b6b", size=18)
        self.mic_label = ft.Text(
            "Silenciado",
            size=13,
            color="#ffffff",
            weight=ft.FontWeight.W_600,
            font_family="Poppins",
        )
        mic_pill = ft.Container(
            content=ft.Row([self.mic_icon, self.mic_label], spacing=8, tight=True),
            bgcolor="#1a1f3aE0",
            border_radius=20,
            padding=ft.Padding.symmetric(horizontal=16, vertical=10),
            shadow=ft.BoxShadow(
                spread_radius=0,
                blur_radius=15,
                color="#00000050",
                offset=ft.Offset(0, 4),
            ),
        )
        settings_button = ft.Container(
            content=ft.Icon(ft.Icons.SETTINGS, color="#AFB3B7", size=20),
            bgcolor="#1a1f3aE0",
            border_radius=20,
            width=40,
            height=40,
            alignment=ft.Alignment.CENTER,
            shadow=ft.BoxShadow(
                spread_radius=0,
                blur_radius=15,
                color="#00000050",
                offset=ft.Offset(0, 4),
            ),
            on_click=self._open_audio_settings,
            tooltip="Configurar micrófono y altavoz",
            ink=True,
        )
        self.mic_status_badge = ft.Container(
            content=ft.Row([settings_button, mic_pill], spacing=10),
            top=130,
            right=20,
            animate=ft.Animation(200, ft.AnimationCurve.EASE_OUT),
        )
        self.event_bus.subscribe("mic:state", self._on_mic_state)

        self._build_audio_settings_dialog()
        page.overlay.append(self.audio_settings_dialog)

        page.add(ft.Stack([main_container, self.mic_status_badge], expand=True))

        # Forzar actualización y maximizar después de agregar contenido
        page.update()

    # -----------------------------
    # Panel de ajustes de audio (micrófono / altavoz)
    # -----------------------------
    def _build_audio_settings_dialog(self) -> None:
        self.mic_dropdown = ft.Dropdown(
            label="Micrófono",
            options=[],
            border_color="#0D1F2330",
            focused_border_color="#F2B33D",
        )
        self.speaker_dropdown = ft.Dropdown(
            label="Altavoz",
            options=[],
            border_color="#0D1F2330",
            focused_border_color="#F2B33D",
        )
        self.audio_settings_status = ft.Text("", size=12, color="#00f5a0")

        self.audio_settings_dialog = ft.AlertDialog(
            modal=True,
            bgcolor="#132E35",
            title=ft.Text(
                "Configuración de audio",
                font_family="Poppins",
                weight=ft.FontWeight.BOLD,
                color="#ffffff",
            ),
            content=ft.Column(
                [
                    ft.Text(
                        "Elige qué micrófono y qué altavoz debe usar OJOZ.",
                        size=13,
                        color="#AFB3B7",
                        font_family="Poppins",
                    ),
                    ft.Container(height=14),
                    self.mic_dropdown,
                    ft.Container(height=14),
                    self.speaker_dropdown,
                    ft.Container(height=8),
                    self.audio_settings_status,
                ],
                tight=True,
                width=360,
            ),
            actions=[
                ft.TextButton("Cerrar", on_click=self._close_audio_settings),
                ft.FilledButton(
                    "Aplicar",
                    icon=ft.Icons.CHECK,
                    on_click=self._apply_audio_settings,
                ),
            ],
            actions_alignment=ft.MainAxisAlignment.END,
        )

    def _open_audio_settings(self, e: ft.ControlEvent) -> None:
        if not self.controller:
            return

        mic_options = []
        current_mic = self.controller.stt.get_device_index()
        for device in self.controller.stt.list_input_devices():
            mic_options.append(ft.DropdownOption(key=str(device["index"]), text=device["name"]))
        self.mic_dropdown.options = mic_options
        self.mic_dropdown.value = str(current_mic) if current_mic is not None else None

        speaker_options = [ft.DropdownOption(key="", text="Predeterminado del sistema")]
        current_speaker = self.controller.tts.get_output_device()
        for name in self.controller.tts.list_output_devices():
            speaker_options.append(ft.DropdownOption(key=name, text=name))
        self.speaker_dropdown.options = speaker_options
        self.speaker_dropdown.value = current_speaker or ""

        self.audio_settings_status.value = ""
        self.audio_settings_dialog.open = True
        self.page.update()

    def _close_audio_settings(self, e: ft.ControlEvent) -> None:
        self.audio_settings_dialog.open = False
        self.page.update()

    def _apply_audio_settings(self, e: ft.ControlEvent) -> None:
        if not self.controller:
            return

        if self.mic_dropdown.value is not None:
            try:
                self.controller.stt.set_device_index(int(self.mic_dropdown.value))
            except (TypeError, ValueError):
                pass

        speaker_value = self.speaker_dropdown.value or None
        self.controller.tts.set_output_device(speaker_value)

        self.audio_settings_status.value = "Aplicado ✓"
        self.page.update()

    def _on_mic_state(self, **kwargs) -> None:
        """Actualiza el indicador flotante segun encienda/apague el microfono."""
        active = bool(kwargs.get("active"))

        def update():
            if active:
                self.mic_icon.name = ft.Icons.MIC
                self.mic_icon.color = "#00f5a0"
                self.mic_label.value = "Escuchando..."
            else:
                self.mic_icon.name = ft.Icons.MIC_OFF
                self.mic_icon.color = "#ff6b6b"
                self.mic_label.value = "Silenciado"
            self.page.update()

        self._run_on_ui(update)
    
    def _create_compact_option(self, number, title, icon, color):
        """Crear opción compacta horizontal con diseño moderno"""
        return ft.Container(
            content=ft.Row([
                ft.Container(
                    content=ft.Text(
                        number,
                        size=18,
                        color=color,
                        weight=ft.FontWeight.BOLD,
                    ),
                    bgcolor="#F28D35",
                    border_radius=30,
                    width=36,
                    height=36,
                    alignment=ft.Alignment.CENTER,
                    border=ft.Border.all(1, "#0D1F2340"),
                    shadow=ft.BoxShadow(
                        spread_radius=0,
                        blur_radius=8,
                        color="#0D1F2325",
                        offset=ft.Offset(0, 2),
                    ),
                ),
                # `color` es el del numero dentro del circulo naranja; sobre el
                # fondo oscuro del header hace falta el tono claro de acento.
                ft.Icon(icon, color="#F28D35", size=22),
                ft.Text(
                    title.replace("\n", " "),
                    size=11,
                    color="#ffffff",
                    weight=ft.FontWeight.W_500,
                    no_wrap=False,
                ),
            ], spacing=6, alignment=ft.MainAxisAlignment.START, vertical_alignment=ft.CrossAxisAlignment.CENTER),
            bgcolor="#1a1f3a80",
            border_radius=15,
            padding=8,
            border=ft.Border.all(1, "#0D1F2320"),
            shadow=ft.BoxShadow(
                spread_radius=0,
                blur_radius=10,
                color="#0D1F2315",
                offset=ft.Offset(0, 3),
            ),
        )
    
    def _create_option_card(self, number, title, icon, color):
        """Crear tarjeta de opción con diseño moderno y hover effect"""
        card_content = ft.Column([
            ft.Container(
                content=ft.Text(
                    number,
                    size=28,
                    color=color,
                    weight=ft.FontWeight.BOLD,
                ),
                bgcolor="#1a1f3a",
                border_radius=50,
                width=50,
                height=50,
                alignment=ft.Alignment.CENTER,
                border=ft.Border.all(2, color + "40"),
                shadow=ft.BoxShadow(
                    spread_radius=0,
                    blur_radius=15,
                    color=color + "30",
                    offset=ft.Offset(0, 5),
                ),
                animate_scale=ft.Animation(300, ft.AnimationCurve.EASE_OUT),
            ),
            ft.Icon(icon, color=color, size=30),
            ft.Text(
                title,
                size=11,
                color="#ffffff",
                weight=ft.FontWeight.W_500,
                text_align=ft.TextAlign.CENTER,
            ),
        ], spacing=8, horizontal_alignment=ft.CrossAxisAlignment.CENTER)
        
        return ft.Container(
            content=card_content,
            bgcolor="#1a1f3a80",
            border_radius=20,
            padding=15,
            width=110,
            border=ft.Border.all(1, color + "30"),
            shadow=ft.BoxShadow(
                spread_radius=0,
                blur_radius=20,
                color=color + "20",
                offset=ft.Offset(0, 8),
            ),
            animate_scale=ft.Animation(200, ft.AnimationCurve.EASE_OUT),
        )


# El punto de entrada real de la aplicacion es app/main_flet.py, que construye
# el Controller con sus dependencias (TTS/STT) y lo enlaza con esta vista.













