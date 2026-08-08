# app/db/engine.py
from __future__ import annotations

import sqlite3
from contextlib import contextmanager
from typing import Iterator

from app.config.settings import vision  # usamos base_dir para ubicar la BD

DB_DIR = vision.base_dir / "data" / "db"
DB_DIR.mkdir(parents=True, exist_ok=True)
DB_PATH = DB_DIR / "app.sqlite3"

SCHEMA_PATH = vision.base_dir / "db" / "schema.sql"


def connect() -> sqlite3.Connection:
    """
    Abre una conexion nueva. El llamador es responsable de cerrarla;
    normalmente conviene usar get_conn() como context manager.
    """
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    # PRAGMA foreign_keys es por conexion: hay que activarlo siempre, no solo
    # al crear el esquema, o las claves foraneas no se aplican en runtime.
    conn.execute("PRAGMA foreign_keys = ON")
    return conn


@contextmanager
def get_conn() -> Iterator[sqlite3.Connection]:
    """
    Context manager que hace commit al salir bien, rollback si hay excepcion
    y **cierra** la conexion en ambos casos.

    El context manager nativo de sqlite3 solo confirma/revierte la transaccion,
    nunca cierra la conexion: usarlo directamente filtraba una conexion por
    cada llamada del DAO.
    """
    conn = connect()
    try:
        with conn:
            yield conn
    finally:
        conn.close()


def init_db() -> None:
    """Crea el esquema si no existe. Es idempotente (el schema usa IF NOT EXISTS)."""
    with get_conn() as conn:
        conn.executescript(SCHEMA_PATH.read_text(encoding="utf-8"))
