from pathlib import Path

from image_recommender.config import DEFAULT_FULL_SHARD_SIZE, DEFAULT_PILOT_SHARD_SIZE
from image_recommender.constants import SUPPORTED_FEATURES
from image_recommender.io.img_iterator import (
    iter_id_images_from_db,
    iter_id_images_from_pilot,
)
from image_recommender.util.logs import get_logger

# module level logger
log = get_logger(__name__)  # pass modules name


def run_extraction(
    feature_type: str,
    input_mode: str,
    run_dir: Path,
    shard_start: int | None,
    shard_stop: int | None,
    pilot_path: Path,
    db_path: Path,
    shard_size: int | None,
    policy: str,
) -> None:
    # validate inputs
    if feature_type not in SUPPORTED_FEATURES:
        raise ValueError(f"Unsupported feature type. Supported: {', '.join(SUPPORTED_FEATURES)}")
    elif input_mode not in {"pilot", "db"}:
        raise ValueError("Unsupported input mode. Supported: pilot, db")

    # ensure run dir exists
    run_dir.mkdir(parents=True, exist_ok=True)

    # resolve shard size
    if shard_size is None:
        if input_mode == "pilot":
            shard_size = DEFAULT_PILOT_SHARD_SIZE
        else:
            shard_size = DEFAULT_FULL_SHARD_SIZE

    # select pilot iterator
    if input_mode == "pilot":
        iterator = iter_id_images_from_pilot(pilot_path=pilot_path, policy=policy)

    # select db iterator
    else:
        iterator = iter_id_images_from_db(db_path=db_path, policy=policy)

    _ = iterator

    # log summary
    log.info("Feature: %s", feature_type)
    log.info("Mode: %s", input_mode)
    log.info("Shard size: %d", shard_size)
    log.info("Shard range: %s - %s", shard_start, shard_stop)

    return
