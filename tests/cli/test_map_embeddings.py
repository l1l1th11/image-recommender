from pathlib import Path

import numpy as np

from image_recommender.cli.main import main


def create_dummy_shards(run_dir: Path, feature_type: str, n_shards=2, n_points=5, dim=4):
    """Creates dummy shards for testing."""
    from image_recommender.features.storage import (
        mark_success,
        write_validate_shard_atomic,
    )

    rng = np.random.default_rng(0)
    for shard_id in range(n_shards):
        features = rng.random((n_points, dim), dtype=np.float32)
        ids = list(range(shard_id * n_points, (shard_id + 1) * n_points))
        meta = {
            "feature_type": feature_type,
            "feature_dim": dim,
            "feature_dtype": str(features.dtype),
            "shard_size": n_points,
            "created_at": "2026-03-13T00:00:00",
            "version": 1,
        }
        write_validate_shard_atomic(
            run_dir=run_dir,
            shard_id=shard_id,
            feature_type=feature_type,
            features=features,
            ids=ids,
            meta=meta,
        )
        mark_success(run_dir=run_dir, feature_type=feature_type, shard_id=shard_id)


def test_map_embeddings_loads_shards(tmp_path):
    """Tests that shards are loaded and concatenated correctly."""
    run_dir = tmp_path / "run"
    feature_type = "embedding"
    create_dummy_shards(run_dir, feature_type, n_shards=2, n_points=5, dim=4)

    argv = [
        "map-embeddings",
        "--run-dir",
        str(run_dir),
        "--feature-type",
        feature_type,
    ]

    ret_code = main(argv)
    assert ret_code == 0, "Shards should be loaded successfully!"


def test_map_embeddings_no_shards(tmp_path):
    """Tests that the function fails when no shards are present."""
    run_dir = tmp_path / "empty_run"
    feature_type = "embedding"

    argv = [
        "map-embeddings",
        "--run-dir",
        str(run_dir),
        "--feature-type",
        feature_type,
    ]

    ret_code = main(argv)
    assert ret_code == 1, "Missing shards should trigger an error!"
