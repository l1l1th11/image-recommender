import json
from pathlib import Path

import numpy as np
import pytest

from image_recommender.search.linear import LinearSearchBackend


def euclidean_distance(query, candidates):
    diff = candidates - query
    return np.sqrt(np.sum(diff**2, axis=1))


# temporary dummy shards:


@pytest.fixture
def fake_shards(tmp_path: Path):
    run_dir = tmp_path
    feature_type = "embedding"
    feature_dir = run_dir / feature_type
    feature_dir.mkdir()

    for shard_id in range(2):
        shard_path = feature_dir / f"shard_{shard_id:04d}"
        shard_path.mkdir()

        features = np.eye(3, dtype=np.float32) * (shard_id + 1)
        ids = np.arange(shard_id * 3, shard_id * 3 + 3, dtype=np.int64)

        np.save(shard_path / "features.npy", features)
        np.save(shard_path / "ids.npy", ids)

        meta = {
            "feature_type": feature_type,
            "feature_dim": 3,
            "feature_dtype": "float32",
            "shard_size": 3,
            "created_at": "now",
            "version": 1,
        }
        (shard_path / "meta.json").write_text(json.dumps(meta))
        (shard_path / "_SUCCESS").touch()

    return run_dir, feature_type


def test_shard_discovery(fake_shards):
    """Tests that shards are discovered correctly."""
    run_dir, feature_type = fake_shards
    backend = LinearSearchBackend(
        run_dir=run_dir,
        feature_type=feature_type,
        distance_fn=euclidean_distance,
        k=1,
    )

    assert len(backend.shard_ids) == 2  # Are there 2 shards?
    # check order of the shards:
    assert backend.shard_ids[0] == 0
    assert backend.shard_ids[1] == 1
