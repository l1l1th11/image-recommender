import numpy as np
from sklearn.cluster import KMeans

from image_recommender.config import DR_SEED


def compute_kmeans(coords: np.ndarray, n_clusters: int, n_init: int = 10) -> np.ndarray:
    """
    Computes k-means clustering on coordinates.
    Input:
    - coords (array of shape (N, D), where D = 2 or 3)
    - n_clusters (number of clusters)
    - n_init (number of initializations)
    Output: cluster labels (N,)
    """
    kmeans = KMeans(
        n_clusters=n_clusters,
        random_state=DR_SEED,
        n_init=n_init,
    )
    return kmeans.fit_predict(coords)
