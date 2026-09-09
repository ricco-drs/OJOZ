# app/db/dao.py
from typing import Optional, Iterable
from app.db.engine import get_conn

# ---------- Users ----------
def get_or_create_user(name: str) -> int:
    with get_conn() as conn:
        cur = conn.execute("SELECT id FROM users WHERE name = ?", (name,))
        row = cur.fetchone()
        if row:
            return row["id"]
        cur = conn.execute("INSERT INTO users(name) VALUES (?)", (name,))
        return cur.lastrowid

def create_user(name: str) -> int:
    """
    Crea SIEMPRE un usuario nuevo, sin reutilizar el id de otro con el mismo
    nombre. La identidad la decide el rostro capturado antes de llamar a esto,
    no el nombre, asi que dos personas distintas pueden llamarse igual.
    """
    with get_conn() as conn:
        cur = conn.execute("INSERT INTO users(name) VALUES (?)", (name,))
        return cur.lastrowid

def get_user_by_name(name: str) -> Optional[dict]:
    """Retorna los datos del usuario si existe, None si no."""
    with get_conn() as conn:
        cur = conn.execute("SELECT * FROM users WHERE name = ?", (name,))
        row = cur.fetchone()
        return dict(row) if row else None

def get_all_users() -> list[dict]:
    """Retorna todos los usuarios registrados."""
    with get_conn() as conn:
        cur = conn.execute("SELECT * FROM users ORDER BY created_at DESC")
        return [dict(row) for row in cur.fetchall()]

# ---------- Photos ----------
def insert_face_photos(
    user_id: int,
    file_paths: Iterable[str],
    w: Optional[int] = None,
    h: Optional[int] = None,
) -> int:
    """Inserta las fotos y devuelve cuantas filas se insertaron."""
    rows = [(user_id, p, w, h) for p in file_paths]
    with get_conn() as conn:
        cur = conn.executemany(
            "INSERT INTO face_photos(user_id, file_path, width, height) VALUES (?, ?, ?, ?)",
            rows
        )
        # cur.rowcount cuenta solo este executemany; conn.total_changes es el
        # acumulado de toda la conexion y daria un numero enganoso.
        return cur.rowcount if cur.rowcount != -1 else len(rows)

def replace_face_photos(
    user_id: int,
    file_paths: Iterable[str],
    w: Optional[int] = None,
    h: Optional[int] = None,
) -> int:
    """Reemplaza en una transaccion las capturas vigentes de un usuario."""
    rows = [(user_id, path, w, h) for path in file_paths]
    with get_conn() as conn:
        conn.execute("DELETE FROM face_photos WHERE user_id = ?", (user_id,))
        if not rows:
            return 0
        cur = conn.executemany(
            "INSERT INTO face_photos(user_id, file_path, width, height) VALUES (?, ?, ?, ?)",
            rows,
        )
        return cur.rowcount if cur.rowcount != -1 else len(rows)

def insert_enrollment(user_id: int, photos_count: int, notes: Optional[str] = None) -> int:
    with get_conn() as conn:
        cur = conn.execute(
            "INSERT INTO enrollments(user_id, photos_count, notes) VALUES (?, ?, ?)",
            (user_id, photos_count, notes)
        )
        return cur.lastrowid

# ---------- Models ----------
def upsert_global_model(
    file_path: str,
    threshold: float,
    version: str = "1.0",
    model_type: str = "ArcFace",
) -> int:
    with get_conn() as conn:
        # modelo global: user_id NULL
        cur = conn.execute(
            "INSERT INTO models(user_id, model_type, version, file_path, threshold) VALUES (NULL, ?, ?, ?, ?)",
            (model_type, version, file_path, threshold)
        )
        return cur.lastrowid

# ---------- Sessions ----------
def start_session(kind: str, user_id: Optional[int] = None) -> int:
    with get_conn() as conn:
        cur = conn.execute(
            "INSERT INTO sessions(kind, user_id) VALUES (?, ?)", (kind, user_id)
        )
        return cur.lastrowid

def finish_session(session_id: int, ok: Optional[bool], details: Optional[str] = None) -> None:
    with get_conn() as conn:
        conn.execute(
            "UPDATE sessions SET finished_at = CURRENT_TIMESTAMP, ok = ?, details = ? WHERE id = ?",
            (int(ok) if ok is not None else None, details, session_id)
        )

# ---------- OCR / Currency / Expiry ----------
def insert_ocr_result(session_id: int, text: str, language: Optional[str], confidence: Optional[float]) -> int:
    with get_conn() as conn:
        cur = conn.execute(
            "INSERT INTO ocr_results(session_id, text, language, confidence) VALUES (?, ?, ?, ?)",
            (session_id, text, language, confidence)
        )
        return cur.lastrowid

def insert_currency_detection(session_id: int, currency: str, value: float, confidence: Optional[float]) -> int:
    with get_conn() as conn:
        cur = conn.execute(
            "INSERT INTO currency_detections(session_id, currency, value, confidence) VALUES (?, ?, ?, ?)",
            (session_id, currency, value, confidence)
        )
        return cur.lastrowid

def insert_expiry_check(session_id: int, product_name: Optional[str], expiry_date: Optional[str],
                        is_expired: Optional[bool], confidence: Optional[float], raw_text: Optional[str]) -> int:
    with get_conn() as conn:
        cur = conn.execute(
            "INSERT INTO expiry_checks(session_id, product_name, expiry_date, is_expired, confidence, raw_text) VALUES (?,?,?,?,?,?)",
            (session_id, product_name, expiry_date, int(is_expired) if is_expired is not None else None, confidence, raw_text)
        )
        return cur.lastrowid
