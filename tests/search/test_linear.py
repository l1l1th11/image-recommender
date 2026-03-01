import json
from pathlib import Path

import numpy as np
import pytest

from image_recommender.search.linear import LinearSearchBackend


def euclidean_distance(query: np.ndarray, candidates: np.ndarray) -> np.ndarray:
    diff = candidates - query
    return np.sqrt(np.sum(diff**2, axis=1))


# temporary dummy shards:


@pytest.fixture
def fake_shards(tmp_path: Path) -> tuple[Path, str]:
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


def test_topk_sorted_and_exact_length(fake_shards):
    """Tests that the top-k results are sorted and have the correct length."""
    run_dir, feature_type = fake_shards
    backend = LinearSearchBackend(
        run_dir=run_dir,
        feature_type=feature_type,
        distance_fn=euclidean_distance,
        k=4,
    )
    query = np.array([1.0, 0.0, 0.0], dtype=np.float32)
    ids, dists = backend.search(query)

    assert len(ids) == 4
    assert dists.dtype == np.float32
    assert np.all(np.diff(dists) >= 0)


def test_self_match_distance_zero(fake_shards):
    """Tests that the distance to self is zero."""
    run_dir, feature_type = fake_shards
    backend = LinearSearchBackend(
        run_dir=run_dir,
        feature_type=feature_type,
        distance_fn=euclidean_distance,
        k=1,
    )
    query = np.array([1.0, 0.0, 0.0], dtype=np.float32)
    _, dists = backend.search(query)
    assert pytest.approx(dists[0], abs=1e-10) == 0.0


def test_zero_query_raises(fake_shards):
    """Tests that a zero query raises an error."""
    run_dir, feature_type = fake_shards
    backend = LinearSearchBackend(
        run_dir=run_dir,
        feature_type=feature_type,
        distance_fn=euclidean_distance,
        k=2,
    )
    with pytest.raises(ValueError):
        backend.search(np.zeros(3, dtype=np.float32))


def test_dimensionality_mismatch(fake_shards):
    """Tests that a dimensionality mismatch raises an error."""
    run_dir, feature_type = fake_shards
    backend = LinearSearchBackend(
        run_dir=run_dir,
        feature_type=feature_type,
        distance_fn=euclidean_distance,
        k=2,
    )
    with pytest.raises(ValueError):
        backend.search(np.array([1.0, 2.0]))


def test_anomaly_logging(fake_shards, caplog):
    """Tests that anomaly logging works."""
    run_dir, feature_type = fake_shards

    def distance_with_inf(query: np.ndarray, candidates: np.ndarray) -> np.ndarray:
        d = euclidean_distance(query, candidates)
        d[0] = np.inf
        d[1] = np.nan
        return d

    backend = LinearSearchBackend(
        run_dir=run_dir,
        feature_type=feature_type,
        distance_fn=distance_with_inf,
        k=2,
    )
    query = np.array([1.0, 0.0, 0.0], dtype=np.float32)
    with caplog.at_level("WARNING"):
        backend.search(query)
    assert any("invalid candidate distances" in r.message for r in caplog.records)


def test_mmap_support(fake_shards):
    run_dir, feature_type = fake_shards
    backend = LinearSearchBackend(
        run_dir=run_dir,
        feature_type=feature_type,
        distance_fn=euclidean_distance,
        k=2,
        mmap=True,
    )
    query = np.array([1.0, 0.0, 0.0], dtype=np.float32)
    ids, _ = backend.search(query)
    assert len(ids) == 2
