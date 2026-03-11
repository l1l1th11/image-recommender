import numpy as np


def hamming_distance(query: np.ndarray, candidate: np.ndarray) -> int:
    # ensure input shapes match
    if query.shape != candidate.shape:
        raise ValueError(
            f"Shapes of query: {query.shape} and candidate: {candidate.shape} don't match"
        )

    # compare vectors element wise
    bit_difference = query != candidate

    # compute hamming distance
    distance = int(np.count_nonzero(bit_difference))

    return distance
