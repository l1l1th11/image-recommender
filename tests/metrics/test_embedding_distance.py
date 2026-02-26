import numpy as np
import pytest

from image_recommender.metrics.embedding_distance import cosine_distance


def test_self_distance_is_zero():
    """Tests if the distance between a vector and itself is zero (within numerical tolerance)."""
    a = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    d = cosine_distance(a, a)
    assert abs(d) < 1e-10


def test_symmetry():
    """Tests if cosine_distance(a,b) = cosine_distance(b,a)."""
    a = np.array([1.0, 0.0, 1.0], dtype=np.float32)
    b = np.array([0.0, 1.0, 1.0], dtype=np.float32)

    d1 = cosine_distance(a, b)
    d2 = cosine_distance(b, a)

    assert abs(d1 - d2) < 1e-10


def test_shape_mismatch_raises():
    """Tests if ValueError is raised when input vectors have different shapes."""
    a = np.array([1.0, 2.0])
    b = np.array([1.0, 2.0, 3.0])

    with pytest.raises(ValueError):
        cosine_distance(a, b)


def test_zero_vector_raises():
    """
    Tests if ValueError is raised when either input vector is a zero vector.
    """
    a = np.array([0.0, 0.0, 0.0])
    b = np.array([1.0, 2.0, 3.0])

    with pytest.raises(ValueError):
        cosine_distance(a, b)
