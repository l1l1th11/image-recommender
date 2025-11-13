CREATE TABLE IF NOT EXISTS images (
    image_id INTEGER PRIMARY KEY,
    path TEXT NOT NULL UNIQUE,
    width INTEGER,
    height INTEGER,
    ext TEXT,
    bytes INTEGER,
    added_at TEXT
    );
