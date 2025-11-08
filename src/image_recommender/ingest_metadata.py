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

    total_files = sum(1 for _ in base_path.rglob("*.*"))  # rglob: all files containing "." in path

    print(f"Found {total_files} files to process in {base_path}")  # assumed amount of files

    for path in tqdm(  # using tqdm for progress bar
        base_path.rglob("*.*"),
        total=total_files,
        desc="Scanning images",
        unit="img",
        dynamic_ncols=True,
        mininterval=0.1,  # update interval in seconds
    ):

        # ---------- INGEST / UPSERT METADATA LOGIC ----------

        abs_path = str(path.resolve())  # absolute path as string

        if abs_path in existing_paths:  # If path already exists in DB...
            continue  # ...skip and go to next file.

        try:
            with Image.open(path) as img:  # read image information
                width, height = img.size
                bytes_ = path.stat().st_size
                ext = path.suffix.lower()
                added_at = datetime.datetime.now().isoformat()

                connector.upsert_image(  # using upsert_image from connector.py
                    abs_path, width, height, ext, bytes_, added_at, db_path=db_path
                )

        except (
            UnidentifiedImageError,
            OSError,
        ):  # If image cannot be identified or other OS error...
            continue  # ...skip it.

    total_ingested = connector.count(db_path=db_path)
    print(f"Total images ingested: {total_ingested}")  # final count of ingested images after scan


def main():
    parser = argparse.ArgumentParser(description="Scan images and ingest metadata")
    parser.add_argument("--root", required=True, help="Root folder with images")
    parser.add_argument("--db", default=None, help="SQLite DB path")
    args = parser.parse_args()

    scan_and_ingest_metadata(args.root, db_path=args.db)


if __name__ == "__main__":
    main()
