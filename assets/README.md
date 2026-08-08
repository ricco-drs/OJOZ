# 📁 Carpeta de Assets - OJOZ

Esta carpeta contiene todos los recursos multimedia personalizados para la interfaz de OJOZ.

## 📂 Estructura

```
assets/
├── images/     # Imágenes (PNG, JPG, JPEG, SVG)
├── videos/     # Videos (MP4, WEBM, AVI)
├── gifs/       # GIFs animados
└── README.md   # Este archivo
```

## 🎨 Cómo usar tus propios recursos

### **Imágenes**
Coloca tus imágenes en la carpeta `images/`. Formatos soportados:
- PNG
- JPG/JPEG
- SVG
- WebP

**Ejemplo de uso en el código:**
```python
ft.Image(
    src="assets/images/mi_logo.png",
    width=200,
    height=200
)
```

### **Videos**
Coloca tus videos en la carpeta `videos/`. Formatos soportados:
- MP4
- WEBM
- AVI

**Ejemplo de uso en el código:**
```python
ft.Video(
    src="assets/videos/mi_video.mp4",
    autoplay=True,
    loop=True
)
```

### **GIFs Animados**
Coloca tus GIFs en la carpeta `gifs/`.

**Ejemplo de uso en el código:**
```python
ft.Image(
    src="assets/gifs/mi_animacion.gif",
    width=300,
    height=300
)
```

## 💡 Sugerencias de recursos

### Para el logo de OJOZ:
- `images/logo.png` - Logo principal
- `images/logo_animated.gif` - Logo animado

### Para el fondo:
- `images/background.jpg` - Imagen de fondo
- `videos/background.mp4` - Video de fondo animado

### Para efectos:
- `gifs/loading.gif` - Animación de carga
- `gifs/speaking.gif` - Animación cuando OJOZ habla
- `gifs/listening.gif` - Animación cuando escucha

### Para las funciones:
- `images/ocr_icon.png` - Icono para lectura de documentos
- `images/money_icon.png` - Icono para identificación de dinero
- `images/calendar_icon.png` - Icono para verificación de fechas

## 🔧 Integración en el código

Para usar estos recursos en `ui/flet_app.py`, simplemente referencia la ruta relativa:

```python
# Imagen estática
logo = ft.Image(src="assets/images/logo.png")

# GIF animado
animation = ft.Image(src="assets/gifs/loading.gif")

# Video de fondo
background = ft.Video(src="assets/videos/background.mp4")
```

## 📝 Notas

- Los archivos deben tener nombres sin espacios (usa guiones bajos o guiones)
- Mantén los tamaños de archivo razonables para mejor rendimiento
- Las imágenes SVG son ideales para logos e iconos (escalables sin pérdida)
- Los videos pueden afectar el rendimiento, úsalos con moderación

---

**¡Agrega tus recursos personalizados aquí para hacer OJOZ único! 🎨✨**
