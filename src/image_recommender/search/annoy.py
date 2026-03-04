from __future__ import annotations

import json
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

        if self.index_path.exists() and self.mapping_path.exists():  # If index exists...
            self._load_index()  # ...load it.

    def _discover_shards(self) -> list[int]:
        """Discovers shard directories."""
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

    def _load_shard(self, shard_dir: Path) -> tuple[list[np.ndarray], list[int]]:
        """Loads vectors and IDs from a shard directory."""
        vectors_path = shard_dir / "vectors.npy"
        ids_path = shard_dir / "ids.npy"

        if not vectors_path.exists() or not ids_path.exists():
            raise ValueError(f"Shard files missing in {shard_dir}")

        vectors = np.load(vectors_path).astype(np.float32)
        ids = np.load(ids_path).astype(np.int32)

        mask = np.any(vectors != 0, axis=1)  # remove zero vectors
        vectors = vectors[mask]
        ids = ids[mask].tolist()
        return [vec for vec in vectors], ids

    def _load_all_shards(self) -> tuple[list[np.ndarray], list[int]]:
        """Loads all shards under feature_type."""
        vectors: list[np.ndarray] = []
        ids: list[int] = []
        for shard_id in self._discover_shards():
            shard_dir = self.run_dir / self.feature_type / f"shard_{shard_id}"
            shard_vectors, shard_ids = self._load_shard(shard_dir)
            vectors.extend(shard_vectors)
            ids.extend(shard_ids)
        return vectors, ids

    def _build_index(self, vectors: list[np.ndarray], ids: list[int] | None = None):
        """Builds Annoy index from given vectors."""
        if not vectors:
            raise ValueError("No vectors provided to build index!")

        dim = vectors[0].shape[0]
        index = AnnoyIndex(dim, self.metric)
        self._id_mapping = ids if ids is not None else list(range(len(vectors)))

        for i, vec in enumerate(vectors):
            if vec is None or not np.any(vec):
                logger.warning("Skipping zero or None vector at position %d", i)
                continue
            index.add_item(i, vec.tolist())

        index.build(self.n_trees)
        self._index = index
        self._dim = dim

        self._persist_index()  # Save index, mapping, metadata

    def _persist_index(self):
        """Persists Annoy index, ID mapping, and metadata."""
        if self._index is None or self._dim is None:
            raise ValueError("Cannot persist index before building!")

        self.index_dir.mkdir(parents=True, exist_ok=True)

        self._index.save(str(self.index_path))
        logger.info("Annoy index saved to %s", self.index_path)

        np.save(self.mapping_path, np.array(self._id_mapping, dtype=np.int32))
        logger.info("ID mapping saved to %s", self.mapping_path)

        meta = {
            "dim": self._dim,
            "n_trees": self.n_trees,
            "metric": self.metric,
            "k": self.k,
            "num_vectors": len(self._id_mapping),
        }
        with open(self.meta_path, "w", encoding="utf-8") as f:
            json.dump(meta, f)
        logger.info("Metadata saved to %s", self.meta_path)

    def _load_index(self):
        """Loads persisted index and ID mapping."""
        with open(self.meta_path, encoding="utf-8") as f:
            meta = json.load(f)
        self._dim = meta["dim"]
        self._index = AnnoyIndex(self._dim, meta["metric"])
        self._index.load(str(self.index_path))
        self._id_mapping = np.load(self.mapping_path).astype(np.int32).tolist()
        logger.info("Loaded existing Annoy index from %s", self.index_path)

    def query(self, vector: np.ndarray) -> tuple[list[int], list[float]]:
        """Returns k nearest neighbors and distances for the given vector."""
        if self._index is None or self._dim is None:
            raise ValueError("Index not built yet!")
        if vector.shape[0] != self._dim:
            raise ValueError(
                f"Dimensionality mismatch! Query dimension: {vector.shape[0]}, index dimension: {self._dim}"
            )

        indices, distances = self._index.get_nns_by_vector(
            vector.tolist(), self.k, search_k=self.search_k, include_distances=True
        )
        mapped_ids = [self._id_mapping[i] for i in indices]
        return mapped_ids, distances
