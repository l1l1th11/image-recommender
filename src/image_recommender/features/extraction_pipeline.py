import datetime
from collections.abc import Iterator
from math import ceil
from pathlib import Path

import numpy as np

from image_recommender.config import (
    DEFAULT_FULL_SHARD_SIZE,
    DEFAULT_PILOT_SHARD_SIZE,
    SAMPLES_DIR,
)
from image_recommender.constants import SUPPORTED_FEATURES
from image_recommender.db.connector import count_images
from image_recommender.db.pilot import load_ids_pilot
from image_recommender.features.embedding import extract_embeddings_batch
from image_recommender.features.hsv import hsv_features
from image_recommender.features.phash import extract_phash
from image_recommender.features.storage import (
    VERSION,
    mark_success,
    success_marker_path,
    write_validate_shard_atomic,
)
from image_recommender.io.img_iterator import (
    iter_id_images_from_db,
    iter_id_images_from_pilot,
    iter_id_images_from_samples,
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
    elif input_mode not in {"samples", "pilot", "db"}:
        raise ValueError("Unsupported input mode. Supported: samples, pilot, db")

    # ensure run dir exists
    run_dir.mkdir(parents=True, exist_ok=True)

    # resolve shard size
    if shard_size is None:
        if input_mode == "pilot":
            shard_size = DEFAULT_PILOT_SHARD_SIZE
        else:
            shard_size = DEFAULT_FULL_SHARD_SIZE

    # get number of total items
    if input_mode == "samples":
        img_paths = []
        for path in SAMPLES_DIR.iterdir():
            if path.is_file():
                img_paths.append(path)
        total_imgs = len(img_paths)
    elif input_mode == "pilot":
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

    # extraction loop
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

        # calculate expected shard size
        exp_shard_size = img_stop - img_start

        # log summary
        log.info(
            "Extracting shard_%04d (%d of %d total shards), image range: [%d - %d), image count: %d",
            shard_idx,
            (shard_idx + 1),
            total_shards,
            img_start,
            img_stop,
            exp_shard_size,
        )

        # store features & ids
        features = []
        ids = []

        batch_imgs = []
        batch_ids = []

        # call iterator
        for image_id, img_array in iterator_wrapper(
            input_mode=input_mode,
            start=img_start,
            stop=img_stop,
            pilot_path=pilot_path,
            db_path=db_path,
            policy=policy,
        ):

            if img_array is None:
                log.warning("Image %d missing, skipping", image_id)
                continue

            if feature_type == "hsv":
                feature = hsv_features(img_rgb=img_array)
                features.append(feature)
                ids.append(image_id)

            elif feature_type == "phash":
                feature = extract_phash(img_rgb=img_array)
                features.append(feature)
                ids.append(image_id)

            elif feature_type == "embedding":
                batch_imgs.append(img_array)
                batch_ids.append(image_id)

        if feature_type == "embedding" and batch_imgs:
            emb = extract_embeddings_batch(batch_imgs)
            features.extend(emb)
            ids.extend(batch_ids)

        # get count of successfully extracted images
        actual_count = len(ids)

        # skip writing shard if no features were extracted
        if actual_count == 0:
            log.warning("No features were extracted for shard_%04d, skipping writing", shard_idx)
            continue

        # convert to array
        features = np.asarray(features)

        # create timestamp
        timestamp = datetime.datetime.now(datetime.UTC).isoformat()

        # create meta dict
        meta = {
            "feature_type": feature_type,
            "feature_dim": features.shape[1],
            "feature_dtype": str(features.dtype),
            "shard_size": actual_count,
            "created_at": timestamp,
            "version": VERSION,
        }

        # write shard
        write_validate_shard_atomic(
            run_dir=run_dir,
            shard_id=shard_idx,
            feature_type=feature_type,
            features=features,
            ids=ids,
            meta=meta,
        )

        # create success marker
        mark_success(run_dir=run_dir, feature_type=feature_type, shard_id=shard_idx)

    # log summary
    log.info("Feature: %s", feature_type)
    log.info("Mode: %s", input_mode)
    log.info("Shard size: %d", shard_size)
    log.info("Shard range: [%s - %s)", shard_start, shard_stop)


def iterator_wrapper(
    input_mode: str, start: int, stop: int, pilot_path: Path, db_path: Path, policy: str
) -> Iterator[tuple[int, np.ndarray]]:
    # select iterator
    if input_mode == "samples":
        return iter_id_images_from_samples(policy=policy)
    elif input_mode == "pilot":
        return iter_id_images_from_pilot(
            start=start, stop=stop, pilot_path=pilot_path, db_path=db_path, policy=policy
        )
    else:
        return iter_id_images_from_db(start=start, stop=stop, db_path=db_path, policy=policy)
