import numpy as np


def compute_umap(
    embeddings: np.ndarray,
    *,  # n_components must be specified as keyword
    n_components: int = 2,
) -> np.ndarray:
    """
    Computes a deterministic UMAP projection of embedding vectors.
    Input:
    - embeddings (embedding matrix of shape (N, D))
    - n_components (2 or 3)
    """

    if not isinstance(embeddings, np.ndarray):
        raise ValueError("Embeddings must be a numpy array!")

    if embeddings.ndim != 2:
        raise ValueError("Embeddings must have shape (N, D)!")

    n, d = embeddings.shape

    if n < 2:
        raise ValueError("Embeddings must contain at least two samples!")

    if d < 1:
        raise ValueError("Embedding dimension must be >= 1")

    if n_components not in {2, 3}:
        raise ValueError("n_components must be 2 or 3")
