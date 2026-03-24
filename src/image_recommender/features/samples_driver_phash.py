from pathlib import Path

import numpy as np
from PIL import Image

from image_recommender.metrics.hamming import hamming_distance_to_many


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
    phashes: np.ndarray,
    k: int,
) -> list[list[tuple[str, int]]]:
    """
    Computes top-k nearest neighbors per perceptual hash using hamming distance.
    Input: (ids, phashes, k)
    Output: List of length N, each element is a list of k tuples (neighbor_id, distance)
    """
    n = phashes.shape[0]  # number of hashes
    results: list[list[tuple[str, int]]] = []

    # validate k
    if k <= 0:
        raise ValueError("k must be a positive integer.")

    # clamp k to n
    k = min(k, n)

    # compute distance between each neighbor and return top k
    for i in range(n):
        query = phashes[i]
        distances = hamming_distance_to_many(query, phashes)
        order = np.argsort(distances)
        top_indices = order[:k]
        topk = [(ids[j], distances[j]) for j in top_indices]
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
            print(f"  {neighbor_id}  |  {dist}")
