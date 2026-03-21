import json
import time

import numpy as np
import pytest

from image_recommender.search.annoy import AnnoySearchBackend


@pytest.fixture(scope="module")
def dummy_data(tmp_path_factory):
    """Creates dummy embeddings, IDs, _SUCCESS marker and meta.json in a temporary directory as fake shards."""
    tmp_dir = tmp_path_factory.mktemp("pilot_dummy")
    feature_type = "embedding"
    feature_dir = tmp_dir / feature_type
    feature_dir.mkdir(parents=True, exist_ok=True)

    # Create initial dummy shard
    shard_dir = feature_dir / "shard_0000"
    shard_dir.mkdir()
    vectors = np.random.rand(9, 16).astype(np.float32)
    zero_vector = np.zeros((1, 16), dtype=np.float32)
    vectors = np.vstack([vectors, zero_vector])
    ids = np.arange(1000, 1010).astype(np.int32)
    np.save(shard_dir / "features.npy", vectors)
    np.save(shard_dir / "ids.npy", ids)
    (shard_dir / "_SUCCESS").touch()

    meta = {
        "created_at": time.time(),
        "feature_dim": vectors.shape[1],
        "feature_dtype": str(vectors.dtype),
        "feature_type": feature_type,
        "shard_size": len(vectors),
        "version": 1,
    }
    with open(shard_dir / "meta.json", "w") as f:
        json.dump(meta, f)
        f.flush()

    return tmp_dir, feature_type, vectors, ids


@pytest.fixture(scope="module")
def annoy_backend(dummy_data):
    """Returns an AnnoySearchBackend instance for the pilot dataset."""
    tmp_dir, feature_type, _, _ = dummy_data
    return AnnoySearchBackend(run_dir=tmp_dir, feature_type=feature_type, k=5)


@pytest.mark.integration
def test_index_files_created(annoy_backend):
    """Tests that index files are created."""
    assert annoy_backend.index_path.exists()
    assert annoy_backend.mapping_path.exists()
    assert annoy_backend.meta_path.exists()


@pytest.mark.integration
def test_reload_uses_persisted_index(dummy_data):
    """Tests that Annoy index is loaded from persisted files."""
    tmp_dir, feature_type, _, _ = dummy_data
    backend1 = AnnoySearchBackend(tmp_dir, feature_type, k=5)
    backend2 = AnnoySearchBackend(tmp_dir, feature_type, k=5)

    assert backend2._index is not None
    assert backend2._dim == backend1._dim
    assert len(backend2._id_mapping) == len(backend1._id_mapping)


@pytest.mark.integration
def test_query_returns_self_top1(annoy_backend, dummy_data):
    """Tests that querying a vector returns its own ID as the top-1 result."""
    _, _, vectors, ids = dummy_data
    query = vectors[0]
    result_ids, _ = annoy_backend.search(query)
    assert int(result_ids[0]) == int(ids[0])


@pytest.mark.integration
def test_result_length_equals_k(annoy_backend, dummy_data):
    """Tests that query returns k results."""
    _, _, vectors, _ = dummy_data
    ids, _ = annoy_backend.search(vectors[0])
    assert len(ids) == annoy_backend.k


@pytest.mark.integration
def test_distances_sorted(annoy_backend, dummy_data):
    """Tests that distances are sorted."""
    _, _, vectors, _ = dummy_data
    _, dists = annoy_backend.search(vectors[0])
    assert np.all(dists[:-1] <= dists[1:])


@pytest.mark.integration
def test_invalid_k(dummy_data):
    """Tests that an invalid k raises an error."""
    tmp_dir, feature_type, _, _ = dummy_data
    with pytest.raises(ValueError):
        AnnoySearchBackend(tmp_dir, feature_type, k=0)


@pytest.mark.integration
def test_dimensionality_mismatch(annoy_backend):
    """Tests that a dimensionality mismatch raises an error."""
    wrong_vec = np.random.rand(999).astype(np.float32)
    with pytest.raises(ValueError):
        annoy_backend.search(wrong_vec)


@pytest.mark.integration
def test_full_pipeline(annoy_backend, dummy_data):
    """Tests that the full pipeline works."""
    _, _, vectors, _ = dummy_data
    ids, dists = annoy_backend.search(vectors[0])
    assert len(ids) == annoy_backend.k
    assert len(dists) == annoy_backend.k
    assert dists.dtype == np.float32


@pytest.mark.integration
def test_invalid_vectors_are_skipped(tmp_path_factory, caplog):
    """Tests that zero vectors are skipped and logged."""
    tmp_dir = tmp_path_factory.mktemp("invalid_vector_test")
    feature_type = "embedding"
    shard_dir = tmp_dir / feature_type / "shard_0000"
    shard_dir.mkdir(parents=True, exist_ok=True)

    vectors = np.array([[1.0] * 16, np.zeros(16)], dtype=np.float32)
    ids = np.array([0, 1], dtype=np.int32)
    np.save(shard_dir / "features.npy", vectors)
    np.save(shard_dir / "ids.npy", ids)
    (shard_dir / "_SUCCESS").touch()

    meta = {
        "created_at": time.time(),
        "feature_dim": 16,
        "feature_dtype": str(vectors.dtype),
        "feature_type": feature_type,
        "shard_size": len(vectors),
        "version": 1,
    }
    with open(shard_dir / "meta.json", "w") as f:
        json.dump(meta, f)

    with caplog.at_level("WARNING"):
        backend = AnnoySearchBackend(tmp_dir, feature_type, k=1)
        assert "invalid vectors skipped" in caplog.text.lower()
        assert len(backend._id_mapping) == 1
