from pathlib import Path

from image_recommender.features.storage import REQUIRED_META_KEYS, VERSION, shard_paths


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
