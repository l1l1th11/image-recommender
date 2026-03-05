import json
import time
from pathlib import Path

import numpy as np
import pytest

from image_recommender.features.storage import read_validate_shard
from image_recommender.search.annoy import AnnoySearchBackend


@pytest.fixture(scope="module")
def pilot_data():
    run_dir = Path("data/features/pilot")
    features, ids = read_validate_shard(run_dir=run_dir, feature_type="hsv", shard_id=0)
    return run_dir, features.astype(np.float32), np.array(ids, dtype=np.int32)


@pytest.fixture(scope="module")
def annoy_backend(pilot_data):
    """Returns an AnnoySearchBackend instance for the pilot dataset."""
    run_dir, _, _ = pilot_data
    backend = AnnoySearchBackend(run_dir=run_dir, feature_type="hsv", k=5)
    return backend


def test_index_files_created(annoy_backend):
    """Tests that index files are created."""
    assert annoy_backend.index_path.exists()
    assert annoy_backend.mapping_path.exists()
    assert annoy_backend.meta_path.exists()


def test_reload_uses_persisted_index(pilot_data):
    """Tests that Annoy index is loaded from persisted files."""
    run_dir, _, _ = pilot_data
    backend1 = AnnoySearchBackend(run_dir, "hsv", k=5)
    backend2 = AnnoySearchBackend(run_dir, "hsv", k=5)
    assert backend2._index is not None
    assert backend2._dim == backend1._dim


def test_query_returns_self_top1(annoy_backend, pilot_data):
    """Tests that querying a vector returns its own ID as the top-1 result."""
    _, features, ids = pilot_data
    query = features[0]
    result_ids, _ = annoy_backend.search(query)
    assert result_ids[0] == ids[0]


def test_result_length_equals_k(annoy_backend, pilot_data):
    """Tests that query returns k results."""
    _, features, _ = pilot_data
    ids, _ = annoy_backend.search(features[0])
    assert len(ids) == annoy_backend.k


def test_distances_sorted(annoy_backend, pilot_data):
    """Tests that distances are sorted."""
    _, features, _ = pilot_data
    _, dists = annoy_backend.search(features[0])
    assert np.all(dists[:-1] <= dists[1:])


def test_invalid_k():
    """Tests that an invalid k raises an error."""
    with pytest.raises(ValueError):
        AnnoySearchBackend(Path("data/features/pilot"), "hsv", k=0)


def test_dimensionality_mismatch(annoy_backend):
    """Tests that a dimensionality mismatch raises an error."""
    wrong_vec = np.random.rand(999).astype(np.float32)
    with pytest.raises(ValueError):
        annoy_backend.search(wrong_vec)


@pytest.mark.integration
def test_full_pipeline(pilot_data):
    """Tests that the full pipeline works."""
    run_dir, features, _ = pilot_data
    backend = AnnoySearchBackend(run_dir, "hsv", k=5)
    ids, dists = backend.search(features[0])
    assert len(ids) == backend.k
    assert len(dists) == backend.k


@pytest.mark.integration
def test_refresh_adds_new_shard(tmp_path):
    """Tests that the refresh function adds new shards to the index."""

    # Setup pilot directory for HSV features:

    run_dir = tmp_path / "pilot"
    features_dir = run_dir / "hsv"
    features_dir.mkdir(parents=True, exist_ok=True)

    # Create first shard (shard_0000) with some dummy data:

    shard0_dir = features_dir / "shard_0000"
    shard0_dir.mkdir(parents=True, exist_ok=True)
    features0 = np.random.rand(5, 10).astype(np.float32)
    ids0 = np.arange(5, dtype=np.int32)  # IDs from 0 to 4

    # Save features and IDs for shard 0:

    np.save(shard0_dir / "features.npy", features0)
    np.save(shard0_dir / "ids.npy", ids0)

    # Save shard metadata:

    meta0 = {
        "created_at": time.time(),
        "feature_dim": features0.shape[1],
        "feature_dtype": str(features0.dtype),
        "feature_type": "hsv",
        "shard_size": features0.shape[0],
        "version": 1,
    }
    with open(shard0_dir / "meta.json", "w", encoding="utf-8") as f:
        json.dump(meta0, f)
    (shard0_dir / "_SUCCESS").touch()  # successful write

    # Initialize backend and verify the first vector is retrievable:

    backend = AnnoySearchBackend(run_dir, "hsv", k=3)
    ids_before, _ = backend.search(features0[0])
    assert ids_before[0] == ids0[0]  # Is the top-1 result the original vector?

    # Create a second shard (shard_0001) with new vectors:

    shard1_dir = features_dir / "shard_0001"
    shard1_dir.mkdir(parents=True, exist_ok=True)
    features1 = np.random.rand(3, 10).astype(np.float32)
    ids1 = np.arange(5, 8, dtype=np.int32)  # IDs from 5 to 7

    # Save features and IDs for shard 1:

    np.save(shard1_dir / "features.npy", features1)
    np.save(shard1_dir / "ids.npy", ids1)

    # Save metadata for shard 1:

    meta1 = {
        "created_at": time.time(),
        "feature_dim": features1.shape[1],
        "feature_dtype": str(features1.dtype),
        "feature_type": "hsv",
        "shard_size": features1.shape[0],
        "version": 1,
    }
    with open(shard1_dir / "meta.json", "w", encoding="utf-8") as f:
        json.dump(meta1, f)
    (shard1_dir / "_SUCCESS").touch()

    backend._build_index()  # Rebuild index

    # Verify that a vector from the new shard is retrievable:

    ids_after, _ = backend.search(features1[0])
    assert ids_after[0] == ids1[0]
