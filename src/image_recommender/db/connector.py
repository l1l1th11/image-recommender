import sqlite3
from collections.abc import Iterator
from contextlib import contextmanager

DB_PATH = "data/metadata.db"  # default path to the DB


@contextmanager  # manage DB connections
def get_conn(db_path: str | None = None) -> Iterator[sqlite3.Connection]:
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


def init_db(db_path: str | None = None) -> None:
    """
    Initializes the database schema from the schema.sql file.
    """
    with get_conn(db_path=db_path) as conn:
        with open("src/image_recommender/db/schema.sql", encoding="utf-8") as f:
            conn.executescript(f.read())  # execute SQL script


#  CRUD OPERATIONS (Create, Read, Update, Delete):

# ---------- CREATE / UPDATE ----------


def upsert_image(
    path: str,
    width: int | None = None,
    height: int | None = None,
    ext: str | None = None,
    bytes_: bytes | None = None,
    added_at: str | None = None,
    db_path: str | None = None,
):
    """
    Updates or inserts an image into the database.
    """
    with get_conn(db_path=db_path) as conn:
        conn.execute(
            """
            INSERT INTO images (path, width, height, ext, bytes, added_at)
            VALUES (?, ?, ?, ?, ?, ?)

            ON CONFLICT(path) DO UPDATE SET
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


# ---------- READ ----------


def get_by_id(image_id: int, db_path: str | None = None) -> sqlite3.Row | None:
    """
    Retrieves an image by its ID.
    """
    with get_conn(db_path=db_path) as conn:
        cur = conn.execute("SELECT * FROM images WHERE image_id = ?", (image_id,))
        return cur.fetchone()  # return the image row (or None)


def get_by_path(path: str, db_path: str | None = None) -> sqlite3.Row | None:
    """
    Retrieves an image by its path.
    """
    with get_conn(db_path=db_path) as conn:
        cur = conn.execute("SELECT * FROM images WHERE path = ?", (path,))
        return cur.fetchone()  # return the image row (or None)


def get_path_by_id(image_id: int, db_path: str | None = None) -> str | None:
    """Retrieves only the file path for a given image_id."""
    with get_conn(db_path=db_path) as conn:
        cur = conn.execute("SELECT path FROM images WHERE image_id = ?", (image_id,))
        row = cur.fetchone()
        return row["path"] if row else None


# ---------- DELETE ----------


def delete_by_id(image_id: int, db_path: str | None = None) -> None:
    """Deletes an image by its ID."""
    with get_conn(db_path=db_path) as conn:
        conn.execute("DELETE FROM images WHERE image_id = ?", (image_id,))


def delete_by_path(path: str, db_path: str | None = None) -> None:
    """Deletes an image by its path."""
    with get_conn(db_path=db_path) as conn:
        conn.execute("DELETE FROM images WHERE path = ?", (path,))


# ---------- PERFORMANCE SANITY (Nice-to-have) ----------


def count(db_path: str | None = None) -> int:
    """Return the total number of images in the database."""
    with get_conn(db_path=db_path) as conn:
        cur = conn.execute("SELECT COUNT(*) AS n FROM images")
        return cur.fetchone()["n"]


def iter_all_ids(db_path: str | None = None) -> Iterator[int]:
    """Iterate over all image IDs in the database."""
    with get_conn(db_path=db_path) as conn:
        cur = conn.execute("SELECT image_id FROM images")
        for row in cur:
            yield row[0]  # yield image_id
