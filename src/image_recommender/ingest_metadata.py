import argparse
import datetime
from pathlib import Path

from PIL import Image, UnidentifiedImageError
from tqdm import tqdm

from image_recommender.db import connector


def get_existing_paths(db_path: str | None = None) -> set[str]:
    """Loads all existing paths from DB."""
    existing_paths: set[str] = set(connector.iter_all_paths(db_path=db_path))
    return existing_paths


def scan_and_ingest_metadata(base_path: str, db_path: str | None = None) -> None:
    """Scan images and ingest metadata."""
    base_path_obj: Path = Path(base_path)  # ensure Path object
    connector.init_db(db_path=db_path)  # initialize DB

    existing_paths: set[str] = get_existing_paths(db_path=db_path)

    files: list[Path] = list(base_path_obj.rglob("*.*"))
    total_files: int = len(files)

    print(f"Found {total_files} files to process in {base_path_obj}")  # assumed amount of files

    for path in tqdm(
        files, total=total_files, desc="Scanning images", unit="img", dynamic_ncols=True
    ):

        # ---------- INGEST / UPSERT METADATA LOGIC ----------

        abs_path: str = str(path.resolve())
        if abs_path in existing_paths:
            continue

        try:
            with Image.open(path) as img:
                width, height = img.size
                bytes_: int = path.stat().st_size
                ext: str = path.suffix.lower()
                added_at: str = datetime.datetime.now().isoformat()

                # Pass db_path, not conn
                connector.upsert_image(
                    abs_path, width, height, ext, bytes_, added_at, db_path=db_path
                )

        except (UnidentifiedImageError, OSError):
            continue  # ...skip it.

    total_ingested: int = connector.count(db_path=db_path)
    print(f"Total images ingested: {total_ingested}")  # final count of ingested images after scan


def main() -> None:
    parser: argparse.ArgumentParser = argparse.ArgumentParser(
        description="Scan images and ingest metadata"
    )
    parser.add_argument("--root", required=True, help="Root folder with images")
    parser.add_argument("--db", default=None, help="SQLite DB path")
    args: argparse.Namespace = parser.parse_args()

    scan_and_ingest_metadata(base_path=args.root, db_path=args.db)


if __name__ == "__main__":
    main()
