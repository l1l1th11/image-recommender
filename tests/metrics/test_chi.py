import numpy as np

from image_recommender.metrics.chi import (
    chi_distance,
    chi_distance_to_many,
)


def test_chi_distance_self_distance():
    """Tests if self-distance is zero."""
    vec = np.random.rand(432).astype(np.float32)  # 432 = 12 * 6 * 6 bins
    assert np.isclose(
        chi_distance(vec, vec), 0.0, atol=1e-5
    )  # Is the self-distance approximately zero?


def test_chi_distance_symmetry():
    """Tests symmetry between the vectors a and b."""
    a = np.random.rand(432).astype(np.float32)
    b = np.random.rand(432).astype(np.float32)
    np.testing.assert_allclose(
        chi_distance(a, b), chi_distance(b, a), rtol=1e-5
    )  # Is χ²(a, b) = χ²(b, a)?


def test_chi_distance_pair():  # might seem redundant, but matches issue
    """Tests if self-distance is zero and smaller than distance to a different vector.
    Expectation: chi_distance(a, a) = 0 and chi_distance(a, b) > 0 for a != b.
    Similar to test_chi_distance_self_distance, but this test focuses on a pair of different vectors.
    """
    a = np.zeros(432, dtype=np.float32)
    b = np.ones(432, dtype=np.float32)
    assert chi_distance(a, a) < chi_distance(
        a, b
    )  # Is the self-distance smaller than the distance to a different vector?


def test_chi_vectorized_matches_scalar():
    """Tests that the vectorized chi-squared distance matches the scalar version
    for multiple candidate histograms.
    Expectation:
    chi_distance_to_many(query, candidates)[i] == chi_distance(query, candidates[i]) for all i.
    """
    query = np.random.rand(432).astype(np.float32)
    candidates = np.random.rand(5, 432).astype(np.float32)

    vec_result = chi_distance_to_many(query, candidates)
    scalar_result = np.array(
        [chi_distance(query, candidates[i]) for i in range(candidates.shape[0])],
        dtype=np.float32,
    )

    np.testing.assert_allclose(vec_result, scalar_result, rtol=1e-5)


def test_vectorized_returns_float32():
    """Tests if the output dtype of chi_distance_to_many is float32."""
    q = np.random.rand(432).astype(np.float32)
    X = np.random.rand(3, 432).astype(np.float32)
    dists = chi_distance_to_many(q, X)
    assert dists.dtype == np.float32
