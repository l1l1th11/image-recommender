from collections.abc import Iterator
from pathlib import Path

import numpy as np

from image_recommender.config import PILOT_IDS_CSV, SAMPLES_DIR
from image_recommender.db.connector import get_path_by_id, iter_id_paths
from image_recommender.db.pilot import load_ids_pilot
from image_recommender.io.img_loader import load_rgb
from image_recommender.util.errors import ImageLoadError
from image_recommender.util.logs import get_logger

# module level logger
log = get_logger(__name__)  # pass modules name


# ---------------------------- DB Run ---------------------------------


def iter_id_images_from_db(
    start: int = 0,
    stop: int | None = None,
    db_path: str | None = None,
    policy: str = "skip_and_log",
) -> Iterator[tuple[int, np.ndarray]]:
    """
    Yields pairs of image_id and img_array.
    Ordered by image_id.
    0 based index slicing, start inclusive, stop exclusive.
    Iterates from start to end if stop is None.
    ImageLoadError message includes image_id | path | reason.
    """
    # catch wrong input
    if policy not in {"skip_and_log", "raise"}:
        raise ValueError("policy must be either skip_and_log or raise")

    # get id, path from db
    for image_id, path in iter_id_paths(start=start, stop=stop, db_path=db_path):

        try:
            # load image
            img_array = load_rgb(path)

        except ImageLoadError as e:

            # log on error
            if policy == "skip_and_log":
                loader_msg = str(e)
                msg = f"{image_id} | {loader_msg}"
                log.error(msg)
                continue

            # raise on error
            elif policy == "raise":
                loader_msg = str(e)
                msg = f"{image_id} | {loader_msg}"
                raise ImageLoadError(msg) from e

        yield image_id, img_array


# ---------------------------- Pilot Run ---------------------------------


def iter_ids_pilot(
    start: int = 0, stop: int | None = None, pilot_path: str | Path = PILOT_IDS_CSV
) -> Iterator[int]:
    """
    Yields image_ids.
    0 based index slicing, start inclusive, stop exclusive.
    Iterates from start to end if stop is None.
    """
    # convert to path object
    pilot_path = Path(pilot_path)

    # validate slice bounds
    if start < 0:
        raise ValueError("start can not be negative")
    if (stop is not None) and (stop < start):
        raise ValueError("stop can not be smaller than start")

    # load ids list
    image_ids = load_ids_pilot(pilot_path=pilot_path)

    # apply slicing
    image_ids = image_ids[start:stop]

    yield from image_ids


def iter_id_images_from_pilot(
    start: int = 0,
    stop: int | None = None,
    pilot_path: str | Path = PILOT_IDS_CSV,
    db_path: str | None = None,
    policy: str = "skip_and_log",
) -> Iterator[tuple[int, np.ndarray]]:
    """
    Yields pairs of image_id (from pilot csv) and img_array (from db lookup).
    0 based index slicing, start inclusive, stop exclusive.
    Iterates from start to end if stop is None.
    ImageLoadError message includes image_id | path | reason.
    """
    # catch wrong input
    if policy not in {"skip_and_log", "raise"}:
        raise ValueError("policy must be either skip_and_log or raise")

    # iterate ids from pilot
    for image_id in iter_ids_pilot(start=start, stop=stop, pilot_path=pilot_path):

        # lookup matching path in db
        path = get_path_by_id(image_id=image_id, db_path=db_path)

        if path is None:

            # log on error
            if policy == "skip_and_log":
                msg = f"{image_id} | <no path> | not present in database"
                log.error(msg)
                continue

            # raise on error
            elif policy == "raise":
                msg = f"{image_id} | <no path> | not present in database"
                raise ValueError(msg)

        else:
            # load image
            try:
                img_array = load_rgb(path)

            except ImageLoadError as e:

                # log on error
                if policy == "skip_and_log":
                    loader_msg = str(e)
                    msg = f"{image_id} | {loader_msg}"
                    log.error(msg)
                    continue

                # raise on error
                elif policy == "raise":
                    loader_msg = str(e)
                    msg = f"{image_id} | {loader_msg}"
                    raise ImageLoadError(msg) from e

            yield image_id, img_array


# ---------------------------- Samples Run ---------------------------------


def iter_id_images_from_samples(
    samples_path: str | Path = SAMPLES_DIR,
    policy: str = "skip_and_log",
) -> Iterator[tuple[int, np.ndarray]]:
    """
    Yields pairs of image_id and img_array (from samples directory).
    ImageLoadError message includes image_id | path | reason.
    """
    # catch wrong input
    if policy not in {"skip_and_log", "raise"}:
        raise ValueError("policy must be either skip_and_log or raise")

    # collect paths from samples/
    samples_path = Path(samples_path)
    paths = []
    for path in samples_path.iterdir():
        if path.is_file() and path.suffix.lower() in {".jpg", ".jpeg", ".png"}:
            paths.append(path)

    # log warning if samples dir is empty
    if len(paths) == 0:
        log.warning("Samples directory is empty")
        return

    # sort paths by filename
    sorted_paths = sorted(paths, key=lambda path: path.name)

    # produce id path pairs
    for image_id, path in enumerate(sorted_paths):

        # load image
        try:
            img_array = load_rgb(path)

        except ImageLoadError as e:

            # log on error
            if policy == "skip_and_log":
                loader_msg = str(e)
                msg = f"{image_id} | {loader_msg}"
                log.error(msg)
                continue

            # raise on error
            elif policy == "raise":
                loader_msg = str(e)
                msg = f"{image_id} | {loader_msg}"
                raise ImageLoadError(msg) from e

        yield image_id, img_array
