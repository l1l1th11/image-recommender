from __future__ import annotations

import json
import re
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

import numpy as np
from annoy import AnnoyIndex

from image_recommender.config import (
    ANNOY_DEFAULT_METRIC,
    ANNOY_DEFAULT_N_TREES,
    ANNOY_DEFAULT_SEARCH_K,
)
from image_recommender.features.storage import read_validate_shard
from image_recommender.util.logs import get_logger

logger = get_logger(__name__)

try:
    ANNOY_VERSION = version("annoy")
except PackageNotFoundError:
    ANNOY_VERSION = "unknown"


class AnnoySearchBackend:
    """
    Approximate nearest neighbor backend using Annoy.
    Builds or loads a persistent index over all feature shards.
    """

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
        if not self.run_dir.exists():
            raise ValueError(f"run_dir does not exist: {self.run_dir}!")

        self.feature_type = feature_type
        self.k = k
        self.n_trees = n_trees or ANNOY_DEFAULT_N_TREES
        self.metric = metric or ANNOY_DEFAULT_METRIC
        self.search_k = search_k or ANNOY_DEFAULT_SEARCH_K

        valid_metrics = {"angular", "euclidean", "manhattan", "hamming", "dot"}
        if self.metric not in valid_metrics:
            raise ValueError(f"Invalid metric={self.metric}")

        self.index_dir = self.run_dir / self.feature_type / "annoy"
        self.index_dir.mkdir(parents=True, exist_ok=True)

        safe_metric = re.sub(
            r"[^a-zA-Z0-9_-]", "_", self.metric
        )  # sanitize metric name for filename
        config_key = f"{safe_metric}_{self.n_trees}_{self.search_k}"

        self.index_path = self.index_dir / f"index_{config_key}.ann"
        self.mapping_path = self.index_dir / f"id_mapping_{config_key}.npy"
        self.meta_path = self.index_dir / f"metadata_{config_key}.json"

        self._index: AnnoyIndex | None = None
        self._id_mapping: np.ndarray = np.array([], dtype=np.int32)
        self._dim: int | None = None
        self._shards: list[int] = []

        if self._can_load_existing() and not self._shards_changed():
            logger.info("Loading persisted Annoy index.")
            self._load_index()
        else:
            logger.info("Building new Annoy index.")
            self._build_index()

    def _discover_shards(self) -> list[int]:
        """Discovers shard directories."""
        feature_dir = self.run_dir / self.feature_type
        if not feature_dir.exists():
            raise ValueError(f"Shard directory {feature_dir} does not exist!")

        shard_ids = [
            int(m.group(1))
            for p in feature_dir.iterdir()
            if p.is_dir() and (m := re.fullmatch(r"shard_(\d{4})", p.name))
        ]

        if not shard_ids:
            raise ValueError(f"No shards found in {feature_dir}.")

        return sorted(shard_ids)

    def _load_all_shards(self) -> tuple[np.ndarray, np.ndarray]:
        """Loads all valid feature vectors and IDs from all shard directories."""
        self._shards = self._discover_shards()
        vectors_list, ids_list = [], []

        for shard_id in self._shards:
            features, shard_ids = read_validate_shard(
                run_dir=self.run_dir,
                feature_type=self.feature_type,
                shard_id=shard_id,
            )
            shard_ids = np.array(shard_ids, dtype=np.int32)

            valid_mask = np.linalg.norm(features, axis=1) != 0
            invalid_count = np.sum(~valid_mask)

            if invalid_count > 0:
                logger.warning("%d invalid vectors skipped in shard %04d", invalid_count, shard_id)

            if valid_mask.sum() == 0:
                continue

            vectors_list.append(features[valid_mask])
            ids_list.append(shard_ids[valid_mask])

        if not vectors_list:
            raise ValueError("No valid vectors found across shards!")

        all_vectors = np.vstack(vectors_list)
        all_ids = np.concatenate(ids_list)
        return all_vectors, all_ids

    def _build_index(self) -> None:
        """Builds Annoy index from all valid vectors across all shards."""
        vectors, ids = self._load_all_shards()
        self._dim = vectors.shape[1]

        index = AnnoyIndex(self._dim, self.metric)
        for i, vec in enumerate(vectors):
            index.add_item(i, vec)
        index.build(self.n_trees)

        self._index = index
        self._id_mapping = ids

        if self.k > len(self._id_mapping):
            raise ValueError("k cannot exceed number of indexed vectors!")

        self._persist_index()

    def _persist_index(self) -> None:
        """Persists Annoy index, ID mapping, metadata, and shard list."""
        if self._index is None or self._dim is None:
            raise RuntimeError("Cannot persist uninitialized index!")

        self._index.save(str(self.index_path))
        np.save(self.mapping_path, self._id_mapping)

        meta = {
            "dim": self._dim,
            "metric": self.metric,
            "n_trees": self.n_trees,
            "search_k": self.search_k,
            "num_vectors": len(self._id_mapping),
            "annoy_version": ANNOY_VERSION,
            "shards": self._shards,
        }

        with open(self.meta_path, "w", encoding="utf-8") as f:
            json.dump(meta, f)

    def _can_load_existing(self) -> bool:
        """Checks if persisted index, mapping, and metadata exist and match current config."""
        if not (
            self.index_path.exists() and self.mapping_path.exists() and self.meta_path.exists()
        ):
            return False

        try:
            with open(self.meta_path, encoding="utf-8") as f:
                meta = json.load(f)
        except Exception:
            return False

        self._shards = meta.get("shards", [])
        return (
            meta.get("metric") == self.metric
            and meta.get("n_trees") == self.n_trees
            and meta.get("search_k") == self.search_k
        )

    def _shards_changed(self) -> bool:
        """Returns True if new shards have been added since last persisted index."""
        return self._discover_shards() != self._shards

    def _load_index(self) -> None:
        """Loads persisted Annoy index, ID mapping, and metadata."""
        with open(self.meta_path, encoding="utf-8") as f:
            meta = json.load(f)

        self._dim = meta["dim"]
        self._shards = meta.get("shards", [])

        index = AnnoyIndex(self._dim, self.metric)
        index.load(str(self.index_path))
        self._index = index
        self._id_mapping = np.load(self.mapping_path).astype(np.int32)

        if index.get_n_items() != len(self._id_mapping):
            raise RuntimeError("Index and ID mapping size mismatch!")

    def refresh(self) -> None:
        """Refreshes the index if shard files have changed."""
        if self._shards_changed():
            logger.info("New shards detected, rebuilding Annoy index.")
            self._build_index()
        else:
            logger.info("No new shards detected, index is up-to-date.")

    def search(self, query: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Searches Annoy index for k nearest neighbors to the given query vector."""
        if self._index is None or self._dim is None:
            raise RuntimeError("Index not initialized.")

        if not isinstance(query, np.ndarray):
            raise ValueError("Query must be a numpy array.")

        if query.ndim != 1:
            raise ValueError("Query must be 1D vector.")

        if query.shape[0] != self._dim:
            raise ValueError("Query dimensionality mismatch.")

        if not np.isfinite(query).all():
            raise ValueError("Query contains invalid values.")

        if np.linalg.norm(query) == 0:
            raise ValueError("Query vector must not be zero.")

        query = query.astype(np.float32)

        indices, distances = self._index.get_nns_by_vector(
            query, self.k, search_k=self.search_k, include_distances=True
        )

        if len(indices) != self.k:
            raise RuntimeError("Annoy returned fewer results than k!")

        ids = np.array([self._id_mapping[i] for i in indices], dtype=np.int32)
        dists = np.array(distances, dtype=np.float32)

        return ids, dists
