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
