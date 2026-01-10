from collections.abc import Iterator

import numpy as np

from image_recommender.db.connector import iter_id_paths
from image_recommender.io.img_loader import load_rgb
from image_recommender.util.errors import ImageLoadError
from image_recommender.util.logs import get_logger

# module level logger
log = get_logger(__name__)  # pass modules name


# for full runs
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
