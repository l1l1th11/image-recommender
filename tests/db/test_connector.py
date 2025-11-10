import os
import tempfile
from collections.abc import Generator
from datetime import datetime
from typing import Any

import pytest

from image_recommender.db import connector


@pytest.fixture
def tmp_db() -> Generator[str, None, None]:
    """Creates a temporary database to run tests."""
    fd, path = tempfile.mkstemp(suffix=".db")  # create temp file as .db
    os.close(fd)  # only path not file descriptor needed

    try:
        connector.init_db(db_path=path)  # initialize schema using init_db from connector.py
        yield path
    finally:
        os.remove(path)


@pytest.fixture
def example_image() -> dict[str, Any]:
    """Provides an example image."""
    return {
        "path": "images/example.jpg",
        "width": 128,
        "height": 256,
        "ext": ".jpg",
        "bytes_": 123,
        "added_at": datetime.now().isoformat(),
    }


# ---------- CRUD TESTS ----------


def test_crud_roundtrip(tmp_db: str, example_image: dict[str, Any]) -> None:
    # CREATE
    image_id = connector.upsert_image(
        **example_image, db_path=tmp_db
    )  # ** unpacks dictionaries into function arguments
    assert (
        image_id is not None
    )  # Was the image_id inserted correctly? If yes, it should not be None.

    # READ by id
    row = connector.get_by_id(image_id, db_path=tmp_db)
    assert row["path"] == example_image["path"]  # Was the correct image returned?

    # READ by path
    row2 = connector.get_by_path(example_image["path"], db_path=tmp_db)
    assert row2["image_id"] == image_id  # Does get_by_path return the same image_id?

    # UPDATE via upsert
    new_width = 512
    example_image["width"] = new_width
    updated_id = connector.upsert_image(**example_image, db_path=tmp_db)
    assert updated_id == image_id
    updated_row = connector.get_by_id(image_id, db_path=tmp_db)
    assert updated_row["width"] == new_width  # Was the width updated from 128 to 512?

    # DELETE by id
    connector.delete_by_id(image_id, db_path=tmp_db)
    assert connector.get_by_id(image_id, db_path=tmp_db) is None


# ---------- UNIQUE PATH TESTS ----------


def test_unique_path(tmp_db: str, example_image: dict[str, Any]) -> None:
    id1 = connector.upsert_image(**example_image, db_path=tmp_db)
    id2 = connector.upsert_image(**example_image, db_path=tmp_db)
    assert id1 == id2  # Is the same image_id returned for the same path?
    rows = list(connector.iter_all_ids(db_path=tmp_db))
    assert len(rows) == 1  # Is there only one entry in the DB?


# ---------- PERFORMANCE SANITY TESTS ----------


def test_count_and_iter_all(tmp_db: str, example_image: dict[str, Any]) -> None:
    assert connector.count(db_path=tmp_db) == 0  # Is the count zero initially?

    for i in range(5):
        img = example_image.copy()
        img["path"] = f"images/sample_{i}.jpg"  # different path for each image (five images)
        connector.upsert_image(**img, db_path=tmp_db)

    assert connector.count(db_path=tmp_db) == 5  # Is the count five after inserting five images?

    ids = list(connector.iter_all_ids(db_path=tmp_db))

    assert len(ids) == 5  # Are there five images in the DB (in the list)?
