import json
from pathlib import Path

import numpy as np
import pytest

from image_recommender.features.storage import (
    REQUIRED_META_KEYS,
    VERSION,
    _validate_shard_inputs,
    _write_features_npy_atomic,
    _write_ids_npy_atomic,
    _write_meta_json_atomic,
    shard_paths,
    write_shard_atomic,
)


def test_shard_paths() -> None:
    # create test paths
    test_paths_1 = shard_paths(run_dir="runs/run_1", feature_type="hsv", shard_id=1)
    test_paths_2 = shard_paths(run_dir="runs/run_1", feature_type="embedding", shard_id=67)
    # check padding and returned paths
    assert test_paths_1 == (
        Path("runs/run_1/hsv/shard_0001/features.npy"),
        Path("runs/run_1/hsv/shard_0001/ids.npy"),
        Path("runs/run_1/hsv/shard_0001/meta.json"),
    )
    # check padding and returned paths
    assert test_paths_2 == (
        Path("runs/run_1/embedding/shard_0067/features.npy"),
        Path("runs/run_1/embedding/shard_0067/ids.npy"),
        Path("runs/run_1/embedding/shard_0067/meta.json"),
    )


def test_constants() -> None:
    # check current version
    assert VERSION == 1
    # check meta keys
    assert REQUIRED_META_KEYS == {
        "feature_type",
        "feature_dim",
        "feature_dtype",
        "shard_size",
        "created_at",
        "version",
    }


def test_validate_shard_inputs_raises_on_id_count_mismatch() -> None:
    # create small 2D test array
    test_array = np.array([[6, 7], [7, 6]])
    # create longer ids list
    mismatched_ids = [15, 2, 23, 42, 5]
    # create metadata
    metadata_placeholder = {
        "feature_type": "hsv",
        "feature_dim": 2,
        "feature_dtype": str(test_array.dtype),
        "shard_size": 2,
        "created_at": "2026-01-12T00:00:00Z",
        "version": VERSION,
    }
    # check mismatch is caught
    with pytest.raises(ValueError) as exc_info:
        _validate_shard_inputs(
            feature_type="hsv", features=test_array, ids=mismatched_ids, meta=metadata_placeholder
        )
    assert str(exc_info.value) == "The number of features and ids is mismatched."


def test_write_meta_json_atomic(tmp_path) -> None:
    # get meta path using helper
    tmp_meta_path = shard_paths(run_dir=tmp_path, feature_type="hsv", shard_id=9)[2]
    # create parent dir
    tmp_meta_path.parent.mkdir(parents=True, exist_ok=True)
    # create metadata
    test_metadata = {
        "feature_type": "hsv",
        "feature_dim": 2,
        "feature_dtype": "float32",
        "shard_size": 2,
        "created_at": "2026-01-12T00:00:00Z",
        "version": VERSION,
    }
    # write metadata
    _write_meta_json_atomic(meta_path=tmp_meta_path, meta=test_metadata)
    # check path exists
    assert tmp_meta_path.exists()
    # check data matches meta
    text = tmp_meta_path.read_text(encoding="utf-8")
    data = json.loads(text)
    assert data == test_metadata


def test_write_ids_npy_atomic(tmp_path) -> None:
    # get ids path using helper
    tmp_ids_path = shard_paths(run_dir=tmp_path, feature_type="embedding", shard_id=5)[1]
    # create parent dir
    tmp_ids_path.parent.mkdir(parents=True, exist_ok=True)
    # create ids
    test_ids = [4, 90, 3]
    # write ids
    _write_ids_npy_atomic(ids_path=tmp_ids_path, ids=test_ids)
    # check path exists
    assert tmp_ids_path.exists()
    # check data matches meta
    data = np.load(tmp_ids_path)
    assert data.tolist() == test_ids


def test_write_features_npy_atomic(tmp_path) -> None:
    # get features path using helper
    tmp_features_path = shard_paths(run_dir=tmp_path, feature_type="hsv", shard_id=41)[0]
    # create parent dir
    tmp_features_path.parent.mkdir(parents=True, exist_ok=True)
    # create base array
    base_array = np.array([[42, 27, 36, 95], [13, 56, 23, 68]])
    # create non-contiguous view
    test_features = base_array[:, ::2]  # columns 0 and 2
    assert test_features.flags["C_CONTIGUOUS"] is False
    # write features
    _write_features_npy_atomic(features_path=tmp_features_path, features=test_features)
    # check path exists
    assert tmp_features_path.exists()
    # check data matches meta
    data = np.load(tmp_features_path)
    assert data.tolist() == test_features.tolist()


def test_write_shard_happy_path(tmp_path) -> None:
    # get artifact paths
    tmp_features_path, tmp_ids_path, tmp_meta_path = shard_paths(
        run_dir=tmp_path, feature_type="embedding", shard_id=50
    )
    # create features
    test_features = np.array([[1, 2], [3, 4]], dtype=np.float32)
    # create ids
    test_ids = [1, 3]
    # create meta
    test_meta = {
        "feature_type": "embedding",
        "feature_dim": 2,
        "feature_dtype": "float32",
        "shard_size": 2,
        "created_at": "2026-01-12T00:00:00Z",
        "version": VERSION,
    }
    # write shard
    write_shard_atomic(
        run_dir=tmp_path,
        shard_id=50,
        feature_type="embedding",
        features=test_features,
        ids=test_ids,
        meta=test_meta,
    )
    # check artifact paths
    assert tmp_features_path.exists()
    assert tmp_ids_path.exists()
    assert tmp_meta_path.exists()
    # verify contents
    assert np.load(tmp_ids_path).tolist() == test_ids
    assert np.load(tmp_features_path).tolist() == test_features.tolist()
    assert json.loads(tmp_meta_path.read_text("utf-8")) == test_meta
