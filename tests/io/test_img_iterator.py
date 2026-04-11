import csv
import logging
import os
import tempfile
from collections.abc import Generator
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

import image_recommender.io.img_iterator as mod
from image_recommender.config import SAMPLES_DIR
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


@pytest.fixture
# create temporary pilot dir
def pilot_dir(tmp_path: Path) -> Path:
    return tmp_path


# ---------------------------- Test DB Run ---------------------------------


def test_db_happy_path(tmp_db: str) -> None:
    # get list of 3 sample images as path objects
    sample_path_objects = list_samples(root=SAMPLES_DIR, extset={"jpg", "jpeg", "png"}, limit=3)

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

    # check number of entries
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


def test_db_skip_and_log(tmp_db: str, caplog) -> None:
    # get list of 2 sample images as path objects
    sample_path_objects = list_samples(root=SAMPLES_DIR, extset={"jpg", "jpeg", "png"}, limit=2)

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
    # check number of entries
    assert len(pairs) == 2

    # retrieve logs
    records = [r for r in caplog.records if (r.levelname == "ERROR") and (r.name == mod.__name__)]
    # check only one log was captured
    assert len(records) == 1
    # retrieve message
    rec = records[0]
    msg = rec.getMessage()
    # check path is included
    assert "invalid_image.jpg" in msg
    # check reason is included
    assert "file not found" in msg


def test_db_raise(tmp_db: str) -> None:
    # get list of 2 sample images as path objects
    sample_path_objects = list_samples(root=SAMPLES_DIR, extset={"jpg", "jpeg", "png"}, limit=2)

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


# ---------------------------- Test Pilot Run ---------------------------------


def test_iter_ids_pilot(pilot_dir):
    # add csv file to tmp dir
    pilot_path = pilot_dir / "ids.csv"
    with open(pilot_path, "w", newline="") as csvfile:
        writer = csv.writer(csvfile)
        # add 3 valid entries
        writer.writerow([42])
        writer.writerow([67])
        writer.writerow(["11"])

        # add 1 empty line
        csvfile.write("\n")

    # convert to list
    pilot_ids = list(mod.iter_ids_pilot(start=1, pilot_path=pilot_path))

    # check number of ids
    assert len(pilot_ids) == 2
    # check id order & data type
    assert pilot_ids == [67, 11]


def test_pilot_happy_path(pilot_dir, tmp_db: str) -> None:
    # get list of 3 sample images as path objects
    sample_path_objects = list_samples(root=SAMPLES_DIR, extset={"jpg", "jpeg", "png"}, limit=3)

    # capture ids from db insertion
    db_ids = []

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
            db_ids.append(
                upsert_image(abs_path, width, height, ext, bytes_, added_at, db_path=tmp_db)
            )

    # shuffle id order for pilot
    pilot_ids = list(reversed(db_ids))

    # add csv file to tmp dir
    pilot_path = pilot_dir / "ids.csv"
    with open(pilot_path, "w", newline="") as csvfile:
        writer = csv.writer(csvfile)
        # add 3 valid entries
        for image_id in pilot_ids:
            writer.writerow([image_id])

        # add 1 empty line
        csvfile.write("\n")

    # get ids from pilot and images from db
    happy_pairs = list(mod.iter_id_images_from_pilot(pilot_path=pilot_path, db_path=tmp_db))

    # check number of entries
    assert len(happy_pairs) == 3

    # check pilots id order is preserved
    returned_ids = [image_id for image_id, _ in happy_pairs]
    assert returned_ids == pilot_ids

    for _, image_array in happy_pairs:
        # check shape of images
        assert image_array.ndim == 3
        assert image_array.shape[2] == 3  # RGB channels

        # check type of images
        assert image_array.dtype == np.uint8


def test_pilot_skip_and_log_missing_id(pilot_dir, tmp_db: str, caplog) -> None:
    # get list of 2 sample images as path objects
    sample_path_objects = list_samples(root=SAMPLES_DIR, extset={"jpg", "jpeg", "png"}, limit=2)

    # capture ids from db insertion
    db_ids = []

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
            db_ids.append(
                upsert_image(abs_path, width, height, ext, bytes_, added_at, db_path=tmp_db)
            )

    # generate missing id
    missing_id = max(db_ids) + 9999

    # add csv file to tmp dir
    pilot_path = pilot_dir / "ids.csv"
    with open(pilot_path, "w", newline="") as csvfile:
        writer = csv.writer(csvfile)
        # add 2 valid entries
        for image_id in db_ids:
            writer.writerow([image_id])

        # add invalid id
        writer.writerow([missing_id])

        # add 1 empty line
        csvfile.write("\n")

    # set logger level
    caplog.set_level(logging.ERROR)

    # get ids from pilot and images from db, skip and log
    pairs = list(
        mod.iter_id_images_from_pilot(pilot_path=pilot_path, db_path=tmp_db, policy="skip_and_log")
    )

    # check number of entries
    assert len(pairs) == 2

    # retrieve logs
    records = [r for r in caplog.records if (r.levelname == "ERROR") and (r.name == mod.__name__)]
    # check only one log was captured
    assert len(records) == 1
    # retrieve message
    rec = records[0]
    msg = rec.getMessage()
    # check id is included
    assert str(missing_id) in msg
    # check reason is included
    assert "not present in database" in msg


def test_pilot_raise_load_error(pilot_dir, tmp_db: str) -> None:
    # get list of 2 sample images as path objects
    sample_path_objects = list_samples(root=SAMPLES_DIR, extset={"jpg", "jpeg", "png"}, limit=2)

    # capture ids from db insertion
    db_ids = []

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
            db_ids.append(
                upsert_image(abs_path, width, height, ext, bytes_, added_at, db_path=tmp_db)
            )

    # add invalid entry and capture id
    bad_id = upsert_image(
        "/non_existent_image.jpg", 123, 456, ".jpg", 789, "test_img_iterator", db_path=tmp_db
    )

    # add csv file to tmp dir
    pilot_path = pilot_dir / "ids.csv"
    with open(pilot_path, "w", newline="") as csvfile:
        writer = csv.writer(csvfile)
        # add 2 valid entries
        for image_id in db_ids:
            writer.writerow([image_id])

        # add bad images id
        writer.writerow([bad_id])

        # add 1 empty line
        csvfile.write("\n")

    # re-raise error from img_loader with img_iterator
    with pytest.raises(ImageLoadError) as excinfo:
        # convert to list and return (needed to raise)
        list(mod.iter_id_images_from_pilot(pilot_path=pilot_path, db_path=tmp_db, policy="raise"))

    msg = str(excinfo.value)
    # check id is included
    assert str(bad_id) in msg
    # check path is included
    assert "non_existent_image.jpg" in msg
    # check reason is included
    assert "file not found" in msg
