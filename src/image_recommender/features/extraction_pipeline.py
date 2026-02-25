from math import ceil
from pathlib import Path

from image_recommender.config import DEFAULT_FULL_SHARD_SIZE, DEFAULT_PILOT_SHARD_SIZE
from image_recommender.constants import SUPPORTED_FEATURES
from image_recommender.db.connector import count_images
from image_recommender.db.pilot import load_ids_pilot
from image_recommender.features.storage import success_marker_path
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

    # select iterator
    if input_mode == "pilot":
        iterator = iter_id_images_from_pilot(pilot_path=pilot_path, policy=policy)
    else:
        iterator = iter_id_images_from_db(db_path=db_path, policy=policy)

    _ = iterator

    # get number of total items
    if input_mode == "pilot":
        total_imgs = len(load_ids_pilot(pilot_path=pilot_path))
    else:
        total_imgs = count_images(db_path=db_path)

    # no op if 0 total items
    if total_imgs == 0:
        log.info("There are no items to process. Returning early.")
        return

    # calculate total shards
    total_shards = ceil(total_imgs / shard_size)

    # calculate default shard range
    if shard_start is None:
        shard_start = 0
    if shard_stop is None:
        shard_stop = total_shards

    # clamp shard range
    if shard_start < 0:
        shard_start = 0
        log.info("Shard start clamped to %d", shard_start)
    elif shard_start > total_shards:
        shard_start = total_shards
        log.info("Shard start clamped to %d", shard_start)
    if shard_stop < 0:
        shard_stop = 0
        log.info("Shard stop clamped to %d", shard_stop)
    elif shard_stop > total_shards:
        shard_stop = total_shards
        log.info("Shard stop clamped to %d", shard_stop)

    # error if no shard to process
    if shard_start >= shard_stop:
        raise ValueError("No shard to process. Shard start must be smaller shard stop")

    # extraction loop skeleton
    for shard_idx in range(shard_start, shard_stop):

        # calculate image range
        img_start = shard_idx * shard_size
        img_stop = min(img_start + shard_size, total_imgs)

        # skip shards with success marker
        marker_path = success_marker_path(
            run_dir=run_dir, feature_type=feature_type, shard_id=shard_idx
        )
        if marker_path.exists():
            log.info("shard_%04d already completed, skipping", shard_idx)
            continue

        # calculate actual shard size
        act_shard_size = img_stop - img_start

        # log summary
        log.info(
            "Extracting shard_%04d of %d total shards, image range: [%d - %d), image count: %d",
            shard_idx,
            total_shards,
            img_start,
            img_stop,
            act_shard_size,
        )

    # log summary
    log.info("Feature: %s", feature_type)
    log.info("Mode: %s", input_mode)
    log.info("Shard size: %d", shard_size)
    log.info("Shard range: %s - %s", shard_start, shard_stop)

    return
