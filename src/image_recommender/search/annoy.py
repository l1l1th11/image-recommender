from __future__ import annotations

from pathlib import Path

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
