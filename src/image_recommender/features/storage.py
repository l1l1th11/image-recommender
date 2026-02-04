import json
from collections.abc import Sequence
from pathlib import Path
from typing import BinaryIO

import numpy as np

from image_recommender.util.fsatomic import write_tmp_then_rename
from image_recommender.util.shard_validation import validate_shard

VERSION = 1

REQUIRED_META_KEYS = {
    "feature_type",
    "feature_dim",
    "feature_dtype",
    "shard_size",
    "created_at",
    "version",
}


def shard_dir(run_dir: Path | str, feature_type: str, shard_id: int) -> Path:
    # convert to path
    run_dir = Path(run_dir)
    # build shard directory
    shard_dir = run_dir / feature_type / f"shard_{shard_id:04d}"  # 4 digit zero padding

    return shard_dir


def shard_paths(run_dir: Path | str, feature_type: str, shard_id: int) -> tuple[Path, Path, Path]:
    """
    Return artifact paths for one shard in a run directory.

    Example:
        >>> shard_paths("runs/run_1", "hsv", 1)
        (Path("runs/run_1/hsv/shard_0001/features.npy"),
         Path("runs/run_1/hsv/shard_0001/ids.npy"),
         Path("runs/run_1/hsv/shard_0001/meta.json"))

    Notes:
        See required metadata keys in REQUIRED_META_KEYS.
    """
    # build shard directory
    shard_path = shard_dir(run_dir=run_dir, feature_type=feature_type, shard_id=shard_id)
    # construct artifact paths
    features_path = shard_path / "features.npy"
    ids_path = shard_path / "ids.npy"
    meta_path = shard_path / "meta.json"

    return features_path, ids_path, meta_path


def _write_meta_json_atomic(meta_path: Path, meta: dict[str, object]) -> None:
    # build json string from meta dict
    json_meta = json.dumps(meta, indent=2, sort_keys=True)
    # encode to bytes
    json_meta = json_meta.encode("utf-8")

    # internal helper: write to file in binary
    def write_meta(f: BinaryIO) -> None:
        f.write(json_meta)

    # atomically write meta
    write_tmp_then_rename(final=meta_path, write_fn=write_meta)


def _write_ids_npy_atomic(ids_path: Path, ids: Sequence[int]) -> None:
    # convert to array
    ids_arr = np.asarray(ids, dtype=np.int64)
    # check its 1D
    if ids_arr.ndim != 1:
        raise ValueError("Ids must be a 1D array.")

    # internal helper: write to file in binary
    def write_ids(f: BinaryIO) -> None:
        # save array to file handle
        np.save(f, ids_arr)

    # atomically write ids
    write_tmp_then_rename(final=ids_path, write_fn=write_ids)


def _write_features_npy_atomic(features_path: Path, features: np.ndarray) -> None:
    # check array is 2D
    if features.ndim != 2:
        raise ValueError("Features must be a 2D array.")
    # ensure array is C-contiguous
    if not features.flags["C_CONTIGUOUS"]:
        features = np.ascontiguousarray(features)

    # internal helper: write file in binary
    def write_features(f: BinaryIO) -> None:
        # save array to file handle
        np.save(f, features)

    # atomically write features
    write_tmp_then_rename(final=features_path, write_fn=write_features)


def write_validate_shard_atomic(
    run_dir: Path,
    shard_id: int,
    feature_type: str,
    features: np.ndarray,
    ids: Sequence[int],
    meta: dict[str, object],
) -> None:
    # validate
    validate_shard(
        feature_type=feature_type,
        features=features,
        ids=ids,
        meta=meta,
        required_keys=REQUIRED_META_KEYS,
        expected_version=VERSION,
    )
    # create artifact paths
    features_path, ids_path, meta_path = shard_paths(
        run_dir=run_dir, feature_type=feature_type, shard_id=shard_id
    )
    # ensure shard dir exists
    features_path.parent.mkdir(parents=True, exist_ok=True)
    # write features
    _write_features_npy_atomic(features_path=features_path, features=features)
    # write ids
    _write_ids_npy_atomic(ids_path=ids_path, ids=ids)
    # write meta
    _write_meta_json_atomic(meta_path=meta_path, meta=meta)


def success_marker_path(run_dir: Path | str, feature_type: str, shard_id: int) -> Path:
    # build shard directory
    shard_path = shard_dir(run_dir=run_dir, feature_type=feature_type, shard_id=shard_id)
    # build marker path
    marker_path = shard_path / "_SUCCESS"

    return marker_path


def mark_success(run_dir: Path | str, feature_type: str, shard_id: int) -> None:
    # build marker path
    marker_path = success_marker_path(run_dir=run_dir, feature_type=feature_type, shard_id=shard_id)
    # check shard dir exists
    shard_path = marker_path.parent
    if not shard_path.is_dir():
        raise FileNotFoundError(f"{shard_path} is missing, or not a directory.")
    # idempotent create
    if not marker_path.exists():
        marker_path.touch()


def list_pending(run_dir: Path | str, feature_type: str, n_shards: int) -> list[int]:
    # validate inputs
    if n_shards < 0:
        raise ValueError("Number of shards must be >= 0.")
    run_dir = Path(run_dir)
    if not run_dir.is_dir():
        raise ValueError(f"{run_dir} is missing, or not a directory.")
    # pending shards
    pending_shard_ids = []
    # compute marker paths
    for idx in range(n_shards):
        marker_path = success_marker_path(run_dir=run_dir, feature_type=feature_type, shard_id=idx)
        # add idx with missing marker
        if not marker_path.exists():
            pending_shard_ids.append(idx)

    return pending_shard_ids


def read_validate_shard(
    run_dir: Path | str, feature_type: str, shard_id: int, mmap: bool = False
) -> tuple[np.ndarray | np.memmap, list[int]]:
    # check shard_id is not negative
    if shard_id < 0:
        raise ValueError("Shard id must be >= 0.")
    # check success marker exists
    marker_path = success_marker_path(run_dir=run_dir, feature_type=feature_type, shard_id=shard_id)
    if not marker_path.exists():
        raise ValueError(f"Shard not completed, missing success marker at: {marker_path}")
    # check artifact file paths exist
    features_path, ids_path, meta_path = shard_paths(
        run_dir=run_dir, feature_type=feature_type, shard_id=shard_id
    )
    if not features_path.exists():
        raise ValueError(f"Features at {features_path} for this shard are missing.")
    if not ids_path.exists():
        raise ValueError(f"Ids at {ids_path} for this shard are missing.")
    if not meta_path.exists():
        raise ValueError(f"Metadata at {meta_path} for this shard is missing.")

    # load features
    if mmap:
        try:
            # load as memory mapped array
            features_array = np.load(features_path, mmap_mode="r")
        except Exception as e:
            raise ValueError(f"Failed to load features in mmap mode: {e}") from e
        # check type
        if not isinstance(features_array, np.memmap):
            raise ValueError("Features is not mmap backed.")
    else:
        try:
            # load as normal array
            features_array = np.load(features_path)
        except Exception as e:
            raise ValueError(f"Failed to load features in non-mmap mode: {e}") from e

    # load ids
    try:
        ids_array = np.load(ids_path)
        ids_list = [int(x) for x in ids_array.tolist()]
    except Exception as e:
        raise ValueError(f"Failed to load ids: {e}") from e

    # load metadata
    try:
        text = meta_path.read_text(encoding="utf-8")
        meta_dict = json.loads(text)
    except Exception as e:
        raise ValueError(f"Failed to load metadata: {e}") from e
    # check meta is dict
    if not isinstance(meta_dict, dict):
        raise ValueError("Metadata is not a dictionary.")

    # validate
    validate_shard(
        feature_type=feature_type,
        features=features_array,
        ids=ids_list,
        meta=meta_dict,
        required_keys=REQUIRED_META_KEYS,
        expected_version=VERSION,
    )

    return features_array, ids_list
