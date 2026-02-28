import numpy as np
import pytest

from image_recommender.metrics.cosine import (
    cosine_distance,
    cosine_distance_to_many,
)


def test_self_distance_is_zero():
    """Tests if the distance between a vector and itself is zero (within numerical tolerance)."""
    a = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    d = cosine_distance(a, a)
    assert abs(d) < 1e-5


def test_symmetry():
    """Tests if cosine_distance(a, b) == cosine_distance(b, a)"""
    a = np.array([1.0, 0.0, 1.0], dtype=np.float32)
    b = np.array([0.0, 1.0, 1.0], dtype=np.float32)
    assert abs(cosine_distance(a, b) - cosine_distance(b, a)) < 1e-5


def test_shape_mismatch_raises():
    """Tests if ValueError is raised when input vectors have different shapes."""
    a = np.array([1.0, 2.0], dtype=np.float32)
    b = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    with pytest.raises(ValueError):
        cosine_distance(a, b)


def test_zero_query_raises():
    """Tests if ValueError is raised when either input vector is a zero vector."""
    q = np.array([0.0, 0.0], dtype=np.float32)
    c = np.array([1.0, 2.0], dtype=np.float32)
    with pytest.raises(ValueError):
        cosine_distance(q, c)


def test_zero_candidate_returns_inf():
    """Tests if cosine_distance returns +inf when candidate vector is zero."""
    q = np.array([1.0, 1.0], dtype=np.float32)
    candidates = np.array([[0.0, 0.0], [1.0, 1.0]], dtype=np.float32)
    dists = cosine_distance_to_many(q, candidates)
    assert np.isinf(dists[0])
    assert not np.isinf(dists[1])


def test_vectorized_matches_scalar():
    """Tests if the vectorized cosine_distance_to_many matches the scalar cosine_distance for each candidate."""
    q = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    candidates = np.array([[1.0, 2.0, 3.0], [3.0, 2.0, 1.0]], dtype=np.float32)
    vec = cosine_distance_to_many(q, candidates)
    expected = np.array(
        [cosine_distance(q, candidates[0]), cosine_distance(q, candidates[1])], dtype=np.float32
    )
    np.testing.assert_allclose(vec, expected, rtol=1e-5)


def test_vectorized_shape():
    """Tests if the output shape of cosine_distance_to_many is correct."""
    q = np.array([1.0, 0.0], dtype=np.float32)
    X = np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32)
    dists = cosine_distance_to_many(q, X)
    assert dists.shape == (2,)
