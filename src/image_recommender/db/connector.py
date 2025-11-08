import sqlite3
from contextlib import contextmanager

DB_PATH = "data/metadata.db"  # default path to the DB


@contextmanager  # manage DB connections
def get_conn(db_path=None):
    """
    Provides a SQLite connection as a context manager.
    Commits changes and closes the connection automatically.
    """
    db_path = db_path or DB_PATH
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row  # rows behave like dictionaries

    try:
        yield conn  # stage for DB operations
        conn.commit()
    finally:
        conn.close()  # automatic commit and close


def init_db(db_path=None):
    """
    Initializes the database schema from the schema.sql file.
    """
    with get_conn(db_path=db_path) as conn:
        with open("src/image_recommender/db/schema.sql", encoding="utf-8") as f:
            conn.executescript(f.read())  # execute SQL script


def upsert_image(
    path=str, width=None, height=None, ext=None, bytes_=None, added_at=None, db_path=None
):
    """
    Updates or inserts an image into the database.
    """
    with get_conn(db_path=db_path) as conn:
        conn.execute(
            """
            INSERT INTO images (path, width, height, ext, bytes, added_at)  # sql command
            VALUES (?, ?, ?, ?, ?, ?)

            ON CONFLICT(path) DO UPDATE SET  # update path in case of conflict
                width = excluded.width,
                height = excluded.height,
                ext = excluded.ext,
                bytes = excluded.bytes,
                added_at = excluded.added_at
            """,
            (path, width, height, ext, bytes_, added_at),
        )
        cur = conn.execute("SELECT image_id FROM images WHERE path = ?", (path,))  # fetch image_id
        return cur.fetchone()["image_id"]  # return image_id
