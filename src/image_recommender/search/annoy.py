from __future__ import annotations

from pathlib import Path

import numpy as np
from annoy import AnnoyIndex

from image_recommender.config import (
    ANNOY_DEFAULT_METRIC,
    ANNOY_DEFAULT_N_TREES,
    ANNOY_DEFAULT_SEARCH_K,
)
from image_recommender.util.logs import get_logger

logger = get_logger(__name__)


class AnnoySearchBackend:
    def __init__(
        self,
        run_dir: Path,
        feature_type: str,
        k: int,
        *,
        n_trees: int | None = None,
        metric: str | None = None,
        search_k: int | None = None,
    ) -> None:
        if not isinstance(k, int) or k <= 0:
            raise ValueError(f"Invalid k={k}. Must be positive!")

        self.run_dir = Path(run_dir)
        self.feature_type = feature_type
        self.k = k
        self.n_trees = n_trees if n_trees else ANNOY_DEFAULT_N_TREES
        self.metric = metric if metric else ANNOY_DEFAULT_METRIC
        self.search_k = search_k if search_k else ANNOY_DEFAULT_SEARCH_K

        self.index_dir = self.run_dir / self.feature_type / "annoy"
        self.index_dir.mkdir(parents=True, exist_ok=True)

        self.index_path = self.index_dir / "index.ann"
        self.mapping_path = self.index_dir / "id_mapping.npy"
        self.meta_path = self.index_dir / "metadata.json"

        self._index: AnnoyIndex | None = None
        self._id_mapping: list[int] = []
        self._dim: int | None = None

    def _discover_shards(self) -> list[int]:
        feature_dir = self.run_dir / self.feature_type
        if not feature_dir.exists():
            raise ValueError(f"Shard directory {feature_dir} does not exist!")

        shard_ids = []
        for path in feature_dir.iterdir():
            if path.is_dir() and path.name.startswith("shard_"):
                try:
                    shard_ids.append(int(path.name.split("_")[1]))
                except ValueError:
                    continue

        if not shard_ids:
            raise ValueError(f"No shards found in {feature_dir}.")

        return sorted(shard_ids)

    def _build_index(self, dummy_vectors: list[np.ndarray]):
        if not dummy_vectors:
            raise ValueError("No vectors provided to build index!")

        dim = dummy_vectors[0].shape[0]
        index = AnnoyIndex(dim, self.metric)
        for i, vec in enumerate(dummy_vectors):
            index.add_item(i, vec.tolist())

        index.build(self.n_trees)
        self._index = index
        self._dim = dim
