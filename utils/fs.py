# utils/fs.py
from __future__ import annotations

import hashlib
import re
import unicodedata
from pathlib import Path
from typing import Iterator, Tuple

from app.config.settings import vision

_DISPLAY_NAME_FILE = ".display_name"


def _slugify(name: str) -> str:
    """Convierte un nombre arbitrario en uno seguro para el sistema de archivos."""
    normalized = (
        unicodedata.normalize("NFKD", name)
        .encode("ascii", "ignore")
        .decode("ascii", errors="ignore")
    )
    normalized = normalized.lower()
    normalized = re.sub(r"[^a-z0-9]+", "_", normalized).strip("_")
    if not normalized:
        normalized = "usuario"
    digest = hashlib.sha1(name.encode("utf-8")).hexdigest()[:6]
    return f"{normalized}_{digest}"


def user_folder(name: str) -> Path:
    """Devuelve la carpeta (slug) donde se guardan las fotos del usuario."""
    slug_path = vision.fotos_dir / _slugify(name)
    legacy_path = vision.fotos_dir / name
    if slug_path.exists():
        return slug_path
    if legacy_path.exists():
        return legacy_path
    return slug_path


def ensure_user_folder(name: str) -> Path:
    folder = vision.fotos_dir / _slugify(name)
    legacy = vision.fotos_dir / name
    if legacy.exists() and not folder.exists():
        try:
            legacy.rename(folder)
        except OSError:
            folder = legacy
    folder.mkdir(parents=True, exist_ok=True)
    _write_display_name(folder, name)
    return folder


def _write_display_name(folder: Path, name: str) -> None:
    try:
        (folder / _DISPLAY_NAME_FILE).write_text(name, encoding="utf-8")
    except Exception:
        pass


def write_display_name(folder: Path, name: str) -> None:
    """Guarda el nombre visible de una carpeta que no fue creada por nombre
    (p. ej. una carpeta identificada por ID de usuario)."""
    _write_display_name(folder, name)


def get_display_name(folder: Path) -> str:
    meta = folder / _DISPLAY_NAME_FILE
    if meta.exists():
        try:
            return meta.read_text(encoding="utf-8").strip() or folder.name
        except Exception:
            return folder.name
    return folder.name


def iter_user_folders() -> Iterator[Tuple[Path, str]]:
    ensure_base_dirs()
    entries = [p for p in vision.fotos_dir.iterdir() if p.is_dir()]
    entries.sort(key=lambda x: x.name.lower())
    for entry in entries:
        yield entry, get_display_name(entry)


def list_users() -> list[str]:
    return [display for _, display in iter_user_folders()]


def model_file() -> Path:
    return vision.model_file


def ensure_base_dirs() -> None:
    vision.fotos_dir.mkdir(parents=True, exist_ok=True)
    vision.modelos_dir.mkdir(parents=True, exist_ok=True)
