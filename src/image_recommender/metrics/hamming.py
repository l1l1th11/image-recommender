import numpy as np


def hamming_distance(query: np.ndarray, candidate: np.ndarray) -> int:
    """
    Computes hamming distance between two vectors.
    Hamming distance is the number of positions at which the corresponding elements of the vectors differ.

    Input:
        query: 1D binary vector of shape (D,)
        candidate: 1D binary vector of shape (D,)

    Output:
        Integer representing the number of differing positions
    """
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


def hamming_distance_to_many(query: np.ndarray, candidates: np.ndarray) -> np.ndarray:
    """
    Computes hamming distances between one query vector and multiple candidates.
    Compares the query vector to each row in the candidate matrix and returns the number of differing positions for each candidate.

    Input:
        query: 1D binary vector of shape (D,)
        candidates: 2D matrix of binary vectors of shape (N, D)

    Output:
        1D array of shape (N,) containing the Hamming distance to each candidate
    """
    # ensure input dimensions are correct
    if query.ndim != 1 or candidates.ndim != 2:
        raise ValueError(
            f"Dimension of query: {query.ndim} or candidate: {candidates.ndim} are wrong. Expected query: 1, candidates: 2 "
        )

    # ensure input lengths match
    if query.shape[0] != candidates.shape[1]:
        raise ValueError(
            f"Length of query: {query.shape} and candidate vectors: {candidates.shape} doesn't match"
        )

    # compare query vector to each candidate row
    bit_difference = query != candidates

    # compute hamming distances
    distances = np.count_nonzero(bit_difference, axis=1)

    return distances
