import os
import tempfile
from datetime import datetime

import pytest

from image_recommender.db import connector


@pytest.fixture
def tmp_db():
    """Creates a temporary database to run tests."""
    fd, path = tempfile.mkstemp(suffix=".db")  # create temp file as .db
    os.close(fd)  # only path not file descriptor needed

    try:
        connector.init_db(db_path=path)  # initialize schema using init_db from connector.py
        yield path
    finally:
        os.remove(path)


@pytest.fixture
def example_image():
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


def test_crud_roundtrip(tmp_db, example_image):
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
