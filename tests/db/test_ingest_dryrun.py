import os
import shutil
from pathlib import Path

from PIL import Image, UnidentifiedImageError

from image_recommender.db import connector
from image_recommender.ingest_metadata import scan_and_ingest_metadata


def test_ingest_dryrun(tmp_path):
    """
    Tests ingesting metadata from a directory containing:
    - A valid image file
    - An image file with the same path already in the DB
    """
    samples_dir = (
        Path(__file__).resolve().parent.parent.parent / "data" / "samples"
    )  # directory with sample images
    assert samples_dir.exists(), f"{samples_dir} does not exist."

    db_path = os.path.join(tmp_path, "test_samples.db")  # temporary DB
    test_dir = os.path.join(tmp_path, "copied_samples")
    os.makedirs(test_dir, exist_ok=True)

    connector.init_db(db_path=str(db_path))

    # ---------- COPY SAMPLE IMAGES ----------

    copied_files = []
    for f in samples_dir.iterdir():
        if f.suffix.lower() in {".jpg", ".jpeg", ".png"}:  # pick valid image files
            dst = os.path.join(test_dir, f.name)
            shutil.copy(f, dst)  # use shutil to copy files
            copied_files.append(dst)

    assert len(copied_files) > 0, "There must be at least one sample image"

    # ---------- CREATE INVALID IMAGE ----------

    bad_file = os.path.join(test_dir, "invalid_image_file.png")
    with open(bad_file, "wb") as f:
        f.write(b"\x00\x00NOT_A_REAL_IMAGE")

    # ---------- INGEST CHECK ----------

    scan_and_ingest_metadata(base_path=str(test_dir), db_path=str(db_path))  # ingestion on copy

    # ---------- VERIFY VALID FILES ----------

    valid_files = []  # list of valid image files
    for f in copied_files:
        try:
            with Image.open(f):
                valid_files.append(f)
        except (UnidentifiedImageError, OSError):
            continue

    assert connector.count(db_path=str(db_path)) == len(
        valid_files
    )  # Are there as many images in the DB as valid files?

    for f in valid_files:
        row = connector.get_by_path(str(Path(f).resolve()), db_path=db_path)
        assert row is not None
        assert Path(row["path"]).is_absolute()
        assert row["ext"] == os.path.splitext(f)[1].lower()  # Are the extensions the same?

    # ---------- IDMPOTENCE CHECK ----------

    image_ids_before = {
        f: connector.get_by_path(str(Path(f).resolve()), db_path=db_path)["image_id"]
        for f in valid_files
    }

    scan_and_ingest_metadata(base_path=str(test_dir), db_path=str(db_path))

    for f in valid_files:
        row = connector.get_by_path(str(Path(f).resolve()), db_path=db_path)
        assert (
            row["image_id"] == image_ids_before[f]
        )  # Are the IDs the same as before (idempotence)?

    # ---------- ADD AN EXTRA BAD FILE ----------

    extra_bad_file = os.path.join(test_dir, "corrupt_image.jpg")
    with open(extra_bad_file, "wb") as f:
        f.write(b"\x00\x00INVALID")

    scan_and_ingest_metadata(base_path=str(test_dir), db_path=str(db_path))

    assert (
        connector.get_by_path(str(Path(extra_bad_file).resolve()), db_path=db_path) is None
    )  # Is the bad file ignored?
