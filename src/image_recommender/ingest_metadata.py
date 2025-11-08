import argparse
import datetime
from pathlib import Path

from PIL import Image, UnidentifiedImageError
from tqdm import tqdm

from image_recommender.db import connector


def get_existing_paths(db_path=None):
    """Loads all existing paths from DB."""
    return set(connector.iter_all_paths(db_path=db_path))  # set of paths


def scan_and_ingest_metadata(base_path: str, db_path=None):
    """Scan images and ingest metadata."""
    base_path = Path(base_path)  # ensure Path object
    connector.init_db(db_path=db_path)  # initialize DB

    existing_paths = get_existing_paths(db_path=db_path)

    files = list(base_path.rglob("*.*"))
    total_files = len(files)

    print(f"Found {total_files} files to process in {base_path}")  # assumed amount of files

    with connector.get_conn(db_path=db_path) as conn:  # single DB connection
        for path in tqdm(
            files, total=total_files, desc="Scanning images", unit="img", dynamic_ncols=True
        ):

            # ---------- INGEST / UPSERT METADATA LOGIC ----------

            abs_path = str(path.resolve())
            if abs_path in existing_paths:
                continue

            try:
                with Image.open(path) as img:
                    width, height = img.size
                    bytes_ = path.stat().st_size
                    ext = path.suffix.lower()
                    added_at = datetime.datetime.now().isoformat()

                    connector.upsert_image(
                        abs_path, width, height, ext, bytes_, added_at, conn=conn
                    )

            except (
                UnidentifiedImageError,
                OSError,
            ):  # If image cannot be identified or other OS error...
                continue  # ...skip it.

    print(
        f"Total images ingested: {connector.count(db_path=db_path)}"
    )  # final count of ingested images after scan


def main():
    parser = argparse.ArgumentParser(description="Scan images and ingest metadata")
    parser.add_argument("--root", required=True, help="Root folder with images")
    parser.add_argument("--db", default=None, help="SQLite DB path")
    args = parser.parse_args()

    scan_and_ingest_metadata(args.root, db_path=args.db)


if __name__ == "__main__":
    main()
