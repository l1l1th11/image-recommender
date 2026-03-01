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
    ):
        if not isinstance(k, int) or k <= 0:
            raise ValueError(f"Invalid k={k}. Must be a positive integer.")

        self.run_dir = Path(run_dir)
        self.feature_type = feature_type
        self.distance_fn = distance_fn
        self.k = k
        self.mmap = mmap

        self.shard_ids = self._discover_shards()

    def _discover_shards(self) -> list[int]:
        """
        Discovers all shards in the run directory.
        """
        feature_dir = self.run_dir / self.feature_type
        if not feature_dir.exists():
            raise ValueError(f"Shard directory {feature_dir} does not exist.")

        shard_ids: list[int] = []
        for path in feature_dir.iterdir():
            if path.is_dir():
                match = re.fullmatch(r"shard_(\d{4})", path.name)
                if match:
                    shard_ids.append(int(match.group(1)))

        if not shard_ids:
            raise ValueError(f"No shards found in {feature_dir}.")

        return sorted(shard_ids)

    def _load_shard(self, shard_id: int) -> tuple[np.ndarray, list[int]]:
        """
        Loads a shard from the run directory.
        """
        return read_validate_shard(
            run_dir=self.run_dir,
            feature_type=self.feature_type,
            shard_id=shard_id,
            mmap=self.mmap,
        )

    def _compute_shard_distances(
        self, query: np.ndarray, candidates: np.ndarray
    ) -> tuple[np.ndarray, int]:
        """
        Computes distances for a shard and handles anomalies (NaN/Inf).
        Input: query, candidates
        Output: distances, anomalies
        """
        if query.ndim != 1 or query.size == 0:
            raise ValueError("Query must be a non-empty 1D array.")
        if candidates.ndim != 2:
            raise ValueError("Candidates must be a 2D array.")
        if query.shape[0] != candidates.shape[1]:
            raise ValueError("Dimensionality mismatch between query and candidates.")

        dists = self.distance_fn(query, candidates)
        if not isinstance(dists, np.ndarray) or dists.ndim != 1:
            raise ValueError("Distance function must return a 1D numpy array.")
        if dists.shape[0] != candidates.shape[0]:
            raise ValueError("Distance output size mismatch.")

        invalid_mask = ~np.isfinite(dists)
        anomalies = int(np.sum(invalid_mask))
        if anomalies > 0:
            dists = dists.copy()
            dists[invalid_mask] = np.inf

        return dists.astype(np.float32, copy=False), anomalies

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

        heap: list[tuple[np.float32, int]] = []
        total_candidates = 0

        for shard_id in self.shard_ids:
            candidates, candidate_ids = self._load_shard(shard_id)
            total_candidates += candidates.shape[0]

            dists, anomalies = self._compute_shard_distances(query, candidates)

            if anomalies > 0:
                logger.warning(
                    "%d invalid candidate distances (+inf) in shard %04d",
                    anomalies,
                    shard_id,
                )

            for dist, idx in zip(dists, candidate_ids, strict=True):
                if not np.isfinite(dist):
                    continue
                if len(heap) < self.k:
                    heapq.heappush(heap, (-dist, idx))
                else:
                    if dist < -heap[0][0]:  # If the new candidate is closer than the farthest...
                        heapq.heapreplace(
                            heap, (-dist, idx)
                        )  # ...replace the farthest with the new closer one.

        if len(heap) < self.k:
            raise ValueError(
                f"Not enough valid candidates ({total_candidates}) to retrieve k={self.k} neighbors."
            )

        sorted_topk = sorted([(-dist, idx) for dist, idx in heap], key=lambda x: x[0])
        distances, ids = zip(*sorted_topk, strict=True)
        return list(ids), np.asarray(distances, dtype=np.float32)
