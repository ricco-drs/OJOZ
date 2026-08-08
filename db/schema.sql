PRAGMA foreign_keys = ON;

-- =========================
-- Actores / Identidad
-- =========================
CREATE TABLE IF NOT EXISTS users (
  id            INTEGER PRIMARY KEY AUTOINCREMENT,
  name          TEXT NOT NULL UNIQUE,      -- "Ricco"
  alias         TEXT,                      -- opcional
  created_at    DATETIME DEFAULT CURRENT_TIMESTAMP,
  updated_at    DATETIME
);

-- Fotos capturadas para enrolamiento (cada recorte guardado)
CREATE TABLE IF NOT EXISTS face_photos (
  id            INTEGER PRIMARY KEY AUTOINCREMENT,
  user_id       INTEGER NOT NULL REFERENCES users(id) ON DELETE CASCADE,
  file_path     TEXT NOT NULL,             -- app/data/fotos/ricco/rostro_0001.jpg
  width         INTEGER,                   -- p.ej., 120
  height        INTEGER,                   -- p.ej., 120
  taken_at      DATETIME DEFAULT CURRENT_TIMESTAMP
);

-- Enrolamientos (una “sesión” de registro)
CREATE TABLE IF NOT EXISTS enrollments (
  id            INTEGER PRIMARY KEY AUTOINCREMENT,
  user_id       INTEGER NOT NULL REFERENCES users(id) ON DELETE CASCADE,
  photos_count  INTEGER NOT NULL,          -- cuántas fotos se tomaron
  notes         TEXT,                      -- info adicional si quieres
  created_at    DATETIME DEFAULT CURRENT_TIMESTAMP
);

-- Modelos de reconocimiento
-- Usa una fila para el modelo GLOBAL (user_id NULL) y opcionalmente por-usuario (user_id no nulo)
CREATE TABLE IF NOT EXISTS models (
  id            INTEGER PRIMARY KEY AUTOINCREMENT,
  user_id       INTEGER REFERENCES users(id) ON DELETE CASCADE, -- NULL = modelo global
  model_type    TEXT NOT NULL,             -- "LBPH"
  version       TEXT,                      -- "1.0", hash o semver si quieres
  file_path     TEXT NOT NULL,             -- app/data/modelos/modeloLBPHFace.xml
  threshold     REAL,                      -- p.ej., 70
  checksum_sha256 TEXT,                    -- opcional (para detectar cambios)
  trained_at    DATETIME DEFAULT CURRENT_TIMESTAMP
);

-- =========================
-- Sesiones y resultados
-- =========================
-- Sesiones de interacción (enrolamiento, auth, o tareas de visión)
CREATE TABLE IF NOT EXISTS sessions (
  id            INTEGER PRIMARY KEY AUTOINCREMENT,
  kind          TEXT NOT NULL,             -- 'enroll' | 'auth' | 'ocr' | 'currency' | 'expiry'
  user_id       INTEGER REFERENCES users(id) ON DELETE SET NULL,
  started_at    DATETIME DEFAULT CURRENT_TIMESTAMP,
  finished_at   DATETIME,
  ok            INTEGER,                   -- 1/0 si aplica
  details       TEXT                       -- JSON/nota libre
);

-- Resultado de OCR (leer hoja)
CREATE TABLE IF NOT EXISTS ocr_results (
  id            INTEGER PRIMARY KEY AUTOINCREMENT,
  session_id    INTEGER NOT NULL REFERENCES sessions(id) ON DELETE CASCADE,
  text          TEXT NOT NULL,             -- texto reconocido
  language      TEXT,                      -- "es-PE" etc.
  confidence    REAL,                      -- si tu OCR lo provee
  created_at    DATETIME DEFAULT CURRENT_TIMESTAMP
);

-- Detección de moneda (valor nominal)
CREATE TABLE IF NOT EXISTS currency_detections (
  id            INTEGER PRIMARY KEY AUTOINCREMENT,
  session_id    INTEGER NOT NULL REFERENCES sessions(id) ON DELETE CASCADE,
  currency      TEXT,                      -- "PEN", "USD"
  value         REAL,                      -- 0.10, 1.00, etc.
  confidence    REAL,
  created_at    DATETIME DEFAULT CURRENT_TIMESTAMP
);

-- Verificación de fecha de vencimiento
CREATE TABLE IF NOT EXISTS expiry_checks (
  id            INTEGER PRIMARY KEY AUTOINCREMENT,
  session_id    INTEGER NOT NULL REFERENCES sessions(id) ON DELETE CASCADE,
  product_name  TEXT,                      -- si lo reconoces
  expiry_date   DATE,                      -- fecha detectada
  is_expired    INTEGER,                   -- 1/0
  confidence    REAL,
  raw_text      TEXT,                      -- texto de donde salió la fecha
  created_at    DATETIME DEFAULT CURRENT_TIMESTAMP
);

-- =========================
-- Índices útiles
-- =========================
CREATE INDEX IF NOT EXISTS idx_face_photos_user ON face_photos(user_id);
CREATE INDEX IF NOT EXISTS idx_models_user ON models(user_id);
CREATE INDEX IF NOT EXISTS idx_sessions_kind ON sessions(kind);
CREATE INDEX IF NOT EXISTS idx_sessions_user ON sessions(user_id);
