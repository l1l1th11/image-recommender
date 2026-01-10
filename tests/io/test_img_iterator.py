import logging
import os
import tempfile
from collections.abc import Generator

import numpy as np
import pytest
from PIL import Image

import image_recommender.io.img_iterator as mod
from image_recommender.constants import SAMPLES_DIR
from image_recommender.db.connector import init_db, upsert_image
from image_recommender.util.errors import ImageLoadError
from image_recommender.util.sampler import list_samples


@pytest.fixture
def tmp_db() -> Generator[str, None, None]:
    """Creates a temporary database to run tests."""
    fd, path = tempfile.mkstemp(suffix=".db")
    os.close(fd)

    try:
        init_db(db_path=path)
        yield path
    finally:
        os.remove(path)


def test_happy_path(tmp_db: str) -> None:
    # get list of 3 sample images as path objects
    sample_path_objects = list_samples(root=SAMPLES_DIR, limit=3)

    # for each sample
    for sample in sample_path_objects:
        # compute absolute path string
        abs_path = str(sample.resolve())

        # set added_at
        added_at = "test_img_iterator"

        # get width, height, ext and bytes_
        with Image.open(sample) as img:
            width, height = img.size
            ext: str = sample.suffix.lower()
            bytes_: int = sample.stat().st_size

            # add entry to temporary db
            upsert_image(abs_path, width, height, ext, bytes_, added_at, db_path=tmp_db)

    # get ids and images from db
    happy_pairs = list(mod.iter_id_images_from_db(db_path=tmp_db))

    # check number of images in db
    assert len(happy_pairs) == 3

    # check ids are sorted ascending
    happy_ids = [image_id for image_id, _ in happy_pairs]
    assert happy_ids == sorted(happy_ids)

    for _, img_array in happy_pairs:
        # check shape of images
        assert img_array.ndim == 3
        assert img_array.shape[2] == 3  # RGB channels

        # check type of images
        assert img_array.dtype == np.uint8


def test_skip_and_log(tmp_db: str, caplog) -> None:
    # get list of 2 sample images as path objects
    sample_path_objects = list_samples(root=SAMPLES_DIR, limit=2)

    # for each sample
    for sample in sample_path_objects:
        # compute absolute path string
        abs_path = str(sample.resolve())
        # set added_at
        added_at = "test_img_iterator"

        # get width, height, ext and bytes_
        with Image.open(sample) as img:
            width, height = img.size
            ext: str = sample.suffix.lower()
            bytes_: int = sample.stat().st_size

            # add entry to temporary db
            upsert_image(abs_path, width, height, ext, bytes_, added_at, db_path=tmp_db)

    # add invalid entry
    upsert_image(
        "bad/invalid_image.jpg", 123, 456, ".jpg", 789, "test_img_iterator", db_path=tmp_db
    )

    # set logger level
    caplog.set_level(logging.ERROR)

    # get ids and images from db, skip and log
    pairs = list(mod.iter_id_images_from_db(db_path=tmp_db, policy="skip_and_log"))
    # check number of images in db
    assert len(pairs) == 2

    # retrieve logs
    records = [r for r in caplog.records if r.levelname == "ERROR"]
    # check only one log was captured
    assert len(records) == 1
    # check module is included
    rec = records[0]
    assert str(mod.__name__) == rec.name
    # check path is included
    msg = rec.getMessage()  # retrieve log
    assert "invalid_image.jpg" in msg
    # check reason is included
    assert "file not found" in msg


def test_raise(tmp_db: str) -> None:
    # get list of 2 sample images as path objects
    sample_path_objects = list_samples(root=SAMPLES_DIR, limit=2)

    # for each sample
    for sample in sample_path_objects:
        # compute absolute path string
        abs_path = str(sample.resolve())
        # set added_at
        added_at = "test_img_iterator"

        # get width, height, ext and bytes_
        with Image.open(sample) as img:
            width, height = img.size
            ext: str = sample.suffix.lower()
            bytes_: int = sample.stat().st_size

            # add entry to temporary db
            upsert_image(abs_path, width, height, ext, bytes_, added_at, db_path=tmp_db)

    # add invalid entry and capture id
    bad_id = upsert_image(
        "/non_existent_image.jpg", 123, 456, ".jpg", 789, "test_img_iterator", db_path=tmp_db
    )

    # re-raise error from img_loader with img_iterator
    with pytest.raises(ImageLoadError) as excinfo:
        # convert to list and return (needed to raise)
        list(mod.iter_id_images_from_db(db_path=tmp_db, policy="raise"))

    msg = str(excinfo.value)
    # check id is included
    assert str(bad_id) in msg
    # check path is included
    assert "non_existent_image.jpg" in msg
    # check reason is included
    assert "file not found" in msg
