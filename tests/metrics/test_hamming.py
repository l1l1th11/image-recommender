import numpy as np

from image_recommender.metrics.hamming import hamming_distance


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
