import json
from collections.abc import Sequence
from pathlib import Path

import numpy as np
import pytest

from image_recommender.features.storage import (
    REQUIRED_META_KEYS,
    VERSION,
    _write_features_npy_atomic,
    _write_ids_npy_atomic,
    _write_meta_json_atomic,
    list_pending,
    mark_success,
    read_validate_shard,
    shard_dir,
    shard_paths,
    success_marker_path,
    write_validate_shard_atomic,
)
from image_recommender.util.shard_validation import validate_shard


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
        validate_shard(
            feature_type="hsv",
            features=test_array,
            ids=mismatched_ids,
            meta=metadata_placeholder,
            required_keys=REQUIRED_META_KEYS,
            expected_version=VERSION,
        )
    assert str(exc_info.value) == "The number of features and ids is mismatched."


def test_write_meta_json_atomic(tmp_path: Path) -> None:
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


def test_write_ids_npy_atomic(tmp_path: Path) -> None:
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


def test_write_features_npy_atomic(tmp_path: Path) -> None:
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


def test_write_shard_correct_artifacts(tmp_path: Path) -> None:
    # get artifact paths
    tmp_features_path, tmp_ids_path, tmp_meta_path = shard_paths(
        run_dir=tmp_path, feature_type="embedding", shard_id=50
    )
    # create shard
    test_features, test_ids, test_meta = _create_shard_success(tmp_path)
    # check artifact paths
    assert tmp_features_path.exists()
    assert tmp_ids_path.exists()
    assert tmp_meta_path.exists()
    # verify contents
    assert np.load(tmp_ids_path).tolist() == test_ids
    assert np.load(tmp_features_path).tolist() == test_features.tolist()
    assert json.loads(tmp_meta_path.read_text("utf-8")) == test_meta


def test_idempotent_marker_create(tmp_path: Path) -> None:
    # build shard path
    test_shard_path = shard_dir(run_dir=tmp_path, feature_type="hsv", shard_id=7)
    # create shard dir
    test_shard_path.mkdir(parents=True, exist_ok=True)
    # call write success marker twice
    mark_success(run_dir=tmp_path, feature_type="hsv", shard_id=7)
    mark_success(run_dir=tmp_path, feature_type="hsv", shard_id=7)
    # check success marker exists
    marker_path = success_marker_path(run_dir=tmp_path, feature_type="hsv", shard_id=7)
    assert marker_path.exists()


def test_list_pending(tmp_path: Path) -> None:
    # write 3 shards with success markers
    for shard_id in range(3):
        # build shard path
        test_shard_path = shard_dir(run_dir=tmp_path, feature_type="embedding", shard_id=shard_id)
        # create shard dir
        test_shard_path.mkdir(parents=True, exist_ok=True)
        # mark success
        mark_success(run_dir=tmp_path, feature_type="embedding", shard_id=shard_id)
    # collect pending shard ids
    pending_shard_ids = list_pending(run_dir=tmp_path, feature_type="embedding", n_shards=5)
    # check expected ids are included
    assert pending_shard_ids == [3, 4]


def test_list_pending_bad_inputs(tmp_path: Path) -> None:
    # n_shards < 0
    with pytest.raises(ValueError, match=r">= 0"):
        list_pending(run_dir=tmp_path, feature_type="embedding", n_shards=-5)
    # run_dir missing
    missing = tmp_path / "does_not_exist"
    with pytest.raises(ValueError, match=r"missing"):
        list_pending(run_dir=missing, feature_type="hsv", n_shards=8)


def _create_shard_success(run_dir: Path) -> tuple[np.ndarray, Sequence[int], dict[str, object]]:
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
    write_validate_shard_atomic(
        run_dir=run_dir,
        shard_id=50,
        feature_type="embedding",
        features=test_features,
        ids=test_ids,
        meta=test_meta,
    )
    # write success marker
    mark_success(run_dir=run_dir, feature_type="embedding", shard_id=50)

    return test_features, test_ids, test_meta


def test_read_incomplete_shard(tmp_path) -> None:
    # try to load non existent shard
    with pytest.raises(ValueError, match=r"missing success marker"):
        read_validate_shard(run_dir=tmp_path, feature_type="hsv", shard_id=7)


data_missing_artifacts = [
    ("features", r"Features at .*features\.npy"),
    ("ids", r"Ids at .*ids\.npy"),
    ("meta", r"Metadata at .*meta\.json"),
]


@pytest.mark.parametrize(
    "artifact_key, error_message", data_missing_artifacts, ids=["features", "ids", "meta"]
)
def test_read_shards_missing_artifacts(tmp_path, artifact_key, error_message) -> None:
    # create shard
    _create_shard_success(tmp_path)
    # get artifact paths
    features_path, ids_path, meta_path = shard_paths(
        run_dir=tmp_path, feature_type="embedding", shard_id=50
    )
    # assign correct artifact path
    if artifact_key == "features":
        artifact_path = features_path
    elif artifact_key == "ids":
        artifact_path = ids_path
    else:
        artifact_path = meta_path
    # delete artifact
    artifact_path.unlink()
    # attempt to load shard and check error message
    with pytest.raises(ValueError, match=error_message):
        read_validate_shard(run_dir=tmp_path, feature_type="embedding", shard_id=50)


data_corrupt_artifacts = [("features", "load features in non-mmap"), ("ids", "load ids")]


@pytest.mark.parametrize(
    "artifact_key, error_message", data_corrupt_artifacts, ids=["features", "ids"]
)
def test_read_shards_corrupt_features_ids(tmp_path, artifact_key, error_message) -> None:
    # create shard
    _create_shard_success(run_dir=tmp_path)
    # get artifact paths
    features_path, ids_path, _ = shard_paths(
        run_dir=tmp_path, feature_type="embedding", shard_id=50
    )
    # assign correct artifact path
    if artifact_key == "features":
        artifact_path = features_path
    else:
        artifact_path = ids_path
    # corrupt artifact
    data = artifact_path.read_bytes()
    artifact_path.write_bytes(data[:10])
    # attempt to load shard and check error message
    with pytest.raises(ValueError, match=error_message):
        read_validate_shard(run_dir=tmp_path, feature_type="embedding", shard_id=50)


def test_read_shards_corrupt_meta(tmp_path) -> None:
    # create shard
    _create_shard_success(run_dir=tmp_path)
    # get meta path
    _, _, meta_path = shard_paths(run_dir=tmp_path, feature_type="embedding", shard_id=50)
    # corrupt meta
    meta_path.write_text("{not json", encoding="utf-8")
    # attempt to load shard and check error message
    with pytest.raises(ValueError, match="load metadata"):
        read_validate_shard(run_dir=tmp_path, feature_type="embedding", shard_id=50)


def test_read_shard_mmap(tmp_path) -> None:
    # create shard
    _create_shard_success(run_dir=tmp_path)
    # load shard in mmap mode
    features_array, _ = read_validate_shard(
        run_dir=tmp_path, feature_type="embedding", shard_id=50, mmap=True
    )
    # double check type
    assert isinstance(features_array, np.memmap)


def test_meta_keys_mismatch(tmp_path) -> None:
    # create shard
    _create_shard_success(run_dir=tmp_path)
    # get meta path
    _, _, meta_path = shard_paths(run_dir=tmp_path, feature_type="embedding", shard_id=50)
    # load metadata
    meta_dict = json.loads(meta_path.read_text(encoding="utf-8"))
    # remove one meta key
    meta_dict.pop("version")
    # persist changes
    meta_path.write_text(json.dumps(meta_dict), encoding="utf-8")
    # attempt to load shard and check error message
    with pytest.raises(ValueError, match="meta keys"):
        read_validate_shard(run_dir=tmp_path, feature_type="embedding", shard_id=50)


def test_validation_mismatch(tmp_path) -> None:
    # create shard
    _create_shard_success(run_dir=tmp_path)
    # get meta path
    _, _, meta_path = shard_paths(run_dir=tmp_path, feature_type="embedding", shard_id=50)
    # load metadata
    meta_dict = json.loads(meta_path.read_text(encoding="utf-8"))
    # change meta dimension value
    meta_dict["feature_dim"] = 1
    # persist changes
    meta_path.write_text((json.dumps(meta_dict)), encoding="utf-8")
    # attempt to load shard and check error message
    with pytest.raises(ValueError, match="feature dimensions"):
        read_validate_shard(run_dir=tmp_path, feature_type="embedding", shard_id=50)


def test_read_valid_shard(tmp_path) -> None:
    # create shard
    test_features, test_ids, _ = _create_shard_success(run_dir=tmp_path)
    # load shard
    features_array, ids_list = read_validate_shard(
        run_dir=tmp_path, feature_type="embedding", shard_id=50
    )
    # compare created and loaded shard
    assert test_features.shape == features_array.shape
    assert test_features.dtype == features_array.dtype
    np.testing.assert_array_equal(features_array, test_features)
    assert test_ids == ids_list
