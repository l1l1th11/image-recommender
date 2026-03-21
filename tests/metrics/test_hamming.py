import numpy as np
import pytest

from image_recommender.metrics.hamming import hamming_distance, hamming_distance_to_many


def test_self_distance():
    vector = np.array([1, 0, 1, 0, 1, 0, 1, 0], dtype=np.uint8)
    result = hamming_distance(query=vector, candidate=vector)

    assert result == 0


def test_symmetry():
    vector_1 = np.array([1, 0, 1, 0, 1, 0, 1, 0], dtype=np.uint8)
    vector_2 = np.array([1, 1, 1, 1, 1, 1, 1, 1], dtype=np.uint8)

    result_1 = hamming_distance(query=vector_1, candidate=vector_2)
    result_2 = hamming_distance(query=vector_2, candidate=vector_1)

    assert result_1 == result_2


def test_vectorized_matches_scalar():
    vector_1 = np.array([1, 0, 1, 0, 1, 0, 1, 0], dtype=np.uint8)
    vector_2 = np.array([1, 1, 1, 1, 1, 1, 1, 1], dtype=np.uint8)
    vector_3 = np.array([0, 0, 0, 0, 0, 0, 0, 0], dtype=np.uint8)
    vector_4 = np.array([0, 1, 0, 1, 0, 1, 0, 1], dtype=np.uint8)

    vectors = np.asarray([vector_2, vector_3, vector_4])

    # calculate hamming distance pair by pair
    s_result = []
    for v in vectors:
        s_result.append(hamming_distance(query=vector_1, candidate=v))
    s_result = np.asarray(s_result)

    # calculate hamming distance for all pairs at once
    v_result = hamming_distance_to_many(query=vector_1, candidates=vectors)

    assert np.array_equal(s_result, v_result)


def test_vectorized_output_shape():
    vector_1 = np.array([1, 0, 1, 0, 1, 0, 1, 0], dtype=np.uint8)
    vector_2 = np.array([1, 1, 1, 1, 1, 1, 1, 1], dtype=np.uint8)
    vector_3 = np.array([0, 0, 0, 0, 0, 0, 0, 0], dtype=np.uint8)
    vector_4 = np.array([0, 1, 0, 1, 0, 1, 0, 1], dtype=np.uint8)

    vectors = np.asarray([vector_2, vector_3, vector_4])

    result = hamming_distance_to_many(query=vector_1, candidates=vectors)

    assert result.shape == (3,)


def test_dimensionality_mismatch():
    vector_1 = np.array([1, 0, 1, 0, 1, 0, 1, 0, 1], dtype=np.uint8)
    vector_2 = np.array([1, 1, 1, 1, 1, 1, 1, 1], dtype=np.uint8)

    vector_3 = np.array([0, 0, 0, 0, 0, 0, 0], dtype=np.uint8)
    vector_4 = np.array([0, 1, 0, 1, 0, 1, 0], dtype=np.uint8)
    vectors = np.asarray([vector_3, vector_4])

    with pytest.raises(ValueError):
        hamming_distance(query=vector_1, candidate=vector_2)

    with pytest.raises(ValueError):
        hamming_distance_to_many(query=vector_1, candidates=vectors)
