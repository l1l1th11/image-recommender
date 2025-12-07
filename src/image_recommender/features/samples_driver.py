from pathlib import Path

import numpy as np

from image_recommender.features.hsv import hsv_features
from image_recommender.io.img_loader import load_rgb
from image_recommender.metrics.hsv_distance import hsv_distance
from image_recommender.util.sampler import list_samples


def topk_on_samples(
    paths: list[Path] | None = None, k: int = 3
) -> dict[Path, list[tuple[Path, float]]]:
    """
    Computes top-k nearest neighbors for each image in paths based on HSV features.
    Skips any images that cannot be loaded.
    """
    if paths is None:
        from image_recommender.constants import IMAGE_EXTS, SAMPLES_DIR

        paths = list_samples(SAMPLES_DIR, extset=IMAGE_EXTS)

    features: dict[Path, np.ndarray] = {}  # Mapping: Path --> HSV vector

    for path in paths:
        try:
            img = load_rgb(path)

            feat = hsv_features(img)
            features[path] = feat
        except Exception:  # If image loading fails...
            print(f"Skipping {path.name}")  # ... print message and skip.

    result: dict[Path, list[tuple[Path, float]]] = {}  # Mapping: Path --> Neighbor Path, Distance

    for p1, f1 in features.items():
        distances: list[tuple[Path, float]] = [
            (p2, hsv_distance(f1, f2))
            for p2, f2 in features.items()
            if p1 != p2  # calculate distances
        ]
        topk = sorted(distances, key=lambda x: x[1])[:k]  # sort and get top-k
        result[p1] = topk

    return result
