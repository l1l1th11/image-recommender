from pathlib import Path

import numpy as np
import pytest

from image_recommender.search.annoy import AnnoySearchBackend


@pytest.fixture(scope="module")
def annoy_backend():
    """Returns a AnnoySearchBackend instance for the pilot dataset."""
    run_dir = Path("data/features/pilot")
    return AnnoySearchBackend(run_dir=run_dir, feature_type="hsv", k=5)


@pytest.fixture(scope="module")
def sample_query():
    """Returns a dummy feature vector and ID for initial tests."""
    dummy_vec = np.array([1.0, 2.0, 3.0])
    dummy_id = 0
    return dummy_vec, dummy_id


def test_discover_shards(annoy_backend):
    shards = annoy_backend._discover_shards()
    assert isinstance(shards, list)  # Is the result a list?
    assert all(isinstance(s, int) for s in shards)  # Are all elements integers?


def test_build_index_accepts_vectors(annoy_backend):
    dummy_vectors = [np.array([1.0, 0.0, 0.0]), np.array([0.0, 1.0, 0.0])]
    annoy_backend._build_index(dummy_vectors)
    assert annoy_backend._index is not None  # Is the index built?
    assert annoy_backend._dim == 3  # Is the dimensionality correct?
