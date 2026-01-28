import json
from collections.abc import Sequence
from pathlib import Path
from typing import BinaryIO

import numpy as np

from image_recommender.util.fsatomic import write_tmp_then_rename

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


def _validate_shard_inputs(
    feature_type: str, features: np.ndarray, ids: Sequence[int], meta: dict[str, object]
) -> None:
    # check features dimensions
    if features.ndim != 2:
        raise ValueError("Feature array has wrong dimensions.")

    # check length matches
    if len(ids) != features.shape[0]:
        raise ValueError("The number of features and ids is mismatched.")

    # check meta keys
    if set(meta.keys()) != REQUIRED_META_KEYS:
        raise ValueError("The meta keys are incorrect.")

    # check meta version
    if meta["version"] != VERSION:
        raise ValueError("Meta version mismatch.")

    # check consistency with data
    if meta["feature_dim"] != features.shape[1]:
        raise ValueError("Metadata and data feature dimensions are mismatched.")
    if meta["feature_dtype"] != str(features.dtype):
        raise ValueError("Metadata and data type are mismatched.")
    if meta["shard_size"] != features.shape[0]:
        raise ValueError("Metadata and data size are mismatched.")
    if meta["feature_type"] != feature_type:
        raise ValueError("Metadata and data feature type are mismatched.")


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


def write_shard_atomic(
    run_dir: Path,
    shard_id: int,
    feature_type: str,
    features: np.ndarray,
    ids: Sequence[int],
    meta: dict[str, object],
) -> None:
    # validate
    _validate_shard_inputs(feature_type=feature_type, features=features, ids=ids, meta=meta)
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
