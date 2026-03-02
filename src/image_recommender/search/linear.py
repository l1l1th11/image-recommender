import heapq
import logging
import re
from collections.abc import Callable
from pathlib import Path

import numpy as np

from image_recommender.features.storage import read_validate_shard

logger = logging.getLogger("project.search")


class LinearSearchBackend:
    """
    Exact linear search backend that retrieves the k nearest neighbors
    across all feature shards using a provided vectorized distance function.
    """

    def __init__(
        self,
        run_dir: Path,
        feature_type: str,
        distance_fn: Callable[[np.ndarray, np.ndarray], np.ndarray],
        k: int,
        mmap: bool = False,
    ) -> None:
        if not isinstance(k, int) or k <= 0:
            raise ValueError(f"Invalid k={k}. Must be a positive integer.")

        self.run_dir = Path(run_dir)
        self.feature_type = feature_type
        self.distance_fn = distance_fn
        self.k = k
        self.mmap = mmap

        self.shard_ids = self._discover_shards()

    def _discover_shards(self) -> list[int]:
        """Discovers all shards in the run directory."""
        feature_dir = self.run_dir / self.feature_type
        if not feature_dir.exists():
            raise ValueError(f"Shard directory {feature_dir} does not exist.")

        shard_ids: list[int] = []
        for path in feature_dir.iterdir():
            if path.is_dir():
                match = re.fullmatch(r"shard_(\d{4})", path.name)  # e.g., shard_0001
                if match:
                    shard_ids.append(int(match.group(1)))

        if not shard_ids:
            raise ValueError(f"No shards found in {feature_dir}.")

        return sorted(shard_ids)

    def _load_shard(self, shard_id: int) -> tuple[np.ndarray, list[int]]:
        """Loads a shard from the run directory."""
        return read_validate_shard(
            run_dir=self.run_dir,
            feature_type=self.feature_type,
            shard_id=shard_id,
            mmap=self.mmap,
        )

    def search(self, query: np.ndarray) -> tuple[list[int], np.ndarray]:
        """
        Performs a linear search over all shards and returns top-k nearest neighbors.
        Input: query
        Output: ids, distances (sorted list)
        """
        if not isinstance(query, np.ndarray):
            raise ValueError("Query must be a numpy array.")
        if query.ndim != 1 or query.size == 0:
            raise ValueError("Query must be a non-empty 1D array.")
        if not np.isfinite(query).all():
            raise ValueError("Query contains invalid (nan or inf) values.")
        if np.linalg.norm(query) == 0:
            raise ValueError("Query vector must not be zero.")

        heap: list[tuple[float, int]] = []
        total_valid_candidates = 0
        expected_dim: int | None = None

        for shard_id in self.shard_ids:
            features, ids = self._load_shard(shard_id)

            if expected_dim is None:
                expected_dim = features.shape[1]

            if query.shape[0] != expected_dim:
                raise ValueError("Dimensionality mismatch between query and features.")

            distances = self.distance_fn(query, features)

            if not isinstance(distances, np.ndarray) or distances.ndim != 1:
                raise ValueError("Distance function must return a 1D numpy array.")

            if distances.shape[0] != features.shape[0]:
                raise ValueError("Distance output size mismatch.")

            invalid_mask = ~np.isfinite(distances)
            anomaly_count = int(np.sum(invalid_mask))

            if anomaly_count > 0:
                logger.warning(
                    "%d invalid candidate distances (+inf) in shard %04d",  # e.g. 0001 instead of 1
                    anomaly_count,
                    shard_id,
                )
                distances = distances.copy()
                distances[invalid_mask] = np.inf

            valid_mask = np.isfinite(distances)
            total_valid_candidates += int(np.sum(valid_mask))

            for dist, img_id in zip(distances, ids, strict=True):
                if not np.isfinite(dist):
                    continue

                if len(heap) < self.k:
                    heapq.heappush(heap, (-float(dist), img_id))
                else:
                    if dist < -heap[0][0]:  # If the new candidate is closer than the farthest...
                        heapq.heapreplace(
                            heap, (-float(dist), img_id)
                        )  # ...replace the farthest with the new closer one.

        if len(heap) < self.k:
            raise ValueError(
                f"Not enough valid candidates ({total_valid_candidates}) "
                f"to retrieve k={self.k} neighbors."
            )

        sorted_topk = sorted(  # sort ascending by distance
            [(-dist, img_id) for dist, img_id in heap],
            key=lambda x: x[0],
        )

        distances_sorted, ids_sorted = zip(*sorted_topk, strict=True)

        return list(ids_sorted), np.asarray(distances_sorted, dtype=np.float32)
