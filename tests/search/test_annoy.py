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
    dummy_vec = np.array([0.0, 0.0, 0.0])
    dummy_id = 0
    return dummy_vec, dummy_id
