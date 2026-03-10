import numpy as np
import umap

from image_recommender.config import DR_SEED


def compute_umap(
    embeddings: np.ndarray,
    *,  # n_components must be specified as keyword
    n_components: int = 2,
    **umap_kwargs,  # umap parameters to be passed as keyword
) -> np.ndarray:
    """
    Computes a deterministic UMAP projection of embedding vectors.
    Input:
    - embeddings (embedding matrix of shape (N, D))
    - n_components (2 or 3)
    - umap_kwargs
    Output: coordinates of shape (N, n_components)
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

    embeddings = np.asarray(embeddings, dtype=np.float32)

    reducer = umap.UMAP(
        n_components=n_components,
        metric="cosine",
        random_state=DR_SEED,
        **umap_kwargs,
    )

    coords = reducer.fit_transform(embeddings)

    coords = coords.astype(np.float32)

    if coords.shape != (n, n_components):
        raise RuntimeError("Unexpected UMAP output shape!")

    return coords
