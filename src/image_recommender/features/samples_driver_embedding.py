from pathlib import Path

import numpy as np
from PIL import Image

from ..metrics.cosine import cosine_distance_to_many


def load_sample_images(samples_dir: Path) -> tuple[list[str], list[np.ndarray]]:
    """
    Loads all images from data/samples/.
    Input: directory path
    Output: (ids, images)
    """
    image_paths = sorted(
        p for p in samples_dir.iterdir() if p.suffix.lower() in {".jpg", ".jpeg", ".png"}
    )

    if not image_paths:
        raise ValueError(f"No images found in {samples_dir}")

    images: list[np.ndarray] = []
    for path in image_paths:
        img = np.asarray(Image.open(path).convert("RGB"))
        images.append(img)

    ids = [str(p) for p in image_paths]  # Convert: Path --> String

    return ids, images


def compute_topk(
    ids: list[str],
    embeddings: np.ndarray,
    k: int,
) -> list[list[tuple[str, float]]]:
    """
    Computes top-k nearest neighbors per embedding using cosine distance.
    Input: (ids, embeddings, k)
    Output: List of length N, each element is a list of k tuples (neighbor_id, distance)
    """
    n = embeddings.shape[0]
    results: list[list[tuple[str, float]]] = []

    if k <= 0:
        raise ValueError("k must be a positive integer.")

    k = min(k, n)

    for i in range(n):
        query = embeddings[i]
        distances = cosine_distance_to_many(query, embeddings)
        order = np.argsort(distances)
        top_indices = order[:k]
        topk = [(ids[j], float(distances[j])) for j in top_indices]
        results.append(topk)

    return results


def print_results(
    ids: list[str],
    results: list[list[tuple[str, float]]],
) -> None:
    """
    Prints compact results: query id + (neighbor id, distance)
    """
    for query_id, neighbors in zip(ids, results, strict=True):
        print(f"\nQuery: {query_id}")
        for neighbor_id, dist in neighbors:
            print(f"  {neighbor_id}  |  {dist:.6f}")
