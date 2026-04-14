from __future__ import annotations

import time
from pathlib import Path

import numpy as np

from image_recommender.features.storage import read_validate_shard


def build_id_to_vec_map(
    run_dir: Path,
    feature_type: str,
) -> tuple[dict[int, np.ndarray], int]:
    """
    Build an in memory mapping from image ID to feature vector.
    This replaces repeated shard scanning by enabling O(1) lookup of feature vectors for arbitrary candidate subsets.

    Input:
        run_dir: Root directory containing feature shards (run_dir/<feature_type>/shard_xxxx)

        feature_type: Feature type to load ("embedding", "hsv", "phash").

    Output:
        id_to_vec: Dictionary mapping image_id -> feature vector (np.ndarray).
        total_vectors: Total number of vectors loaded (sanity check).

    Notes:
        - Loads the full dataset into memory.
        - Requires sufficient RAM (ca. 1 GB for embeddings).
    """
    feature_dir = run_dir / feature_type
    shard_dirs = sorted(p for p in feature_dir.glob("shard_*") if p.is_dir())

    if not shard_dirs:
        raise ValueError(f"No shards found for feature '{feature_type}'")

    id_to_vec: dict[int, np.ndarray] = {}
    total_vectors = 0

    for shard_dir in shard_dirs:
        shard_id = int(shard_dir.name.split("_")[1])

        features, ids = read_validate_shard(
            run_dir=run_dir,
            feature_type=feature_type,
            shard_id=shard_id,
        )

        if len(features) != len(ids):
            raise ValueError(f"Mismatch in shard {shard_id}")

        for id_, vec in zip(ids, features, strict=True):
            id_to_vec[id_] = vec

        total_vectors += len(ids)

    if not id_to_vec:
        raise ValueError("Mapping is empty")

    return id_to_vec, total_vectors


def estimate_memory_bytes(id_to_vec: dict[int, np.ndarray]) -> int:
    """
    Estimate memory usage of the mapping (vectors only).

    Input: id_to_vec: Mapping from id to vector

    Returns: Approximate memory usage
    """
    sample_vec = next(iter(id_to_vec.values()))
    return sample_vec.nbytes * len(id_to_vec)


# optional standalone benchmark for manual inspection

if __name__ == "__main__":
    RUN_DIR = Path("data/features/db")
    FEATURE_TYPE = "embedding"

    print("\n=== ID → VECTOR MAPPING BENCHMARK ===\n")
    print(f"run_dir      : {RUN_DIR}")
    print(f"feature_type : {FEATURE_TYPE}\n")

    t0 = time.perf_counter()
    id_to_vec, total_vectors = build_id_to_vec_map(RUN_DIR, FEATURE_TYPE)
    t1 = time.perf_counter()

    build_time = t1 - t0
    mem_bytes = estimate_memory_bytes(id_to_vec)

    print("=== RESULTS ===")
    print(f"total_vectors : {total_vectors}")
    print(f"build_time    : {build_time:.2f} s")
    print(f"vector_shape  : {next(iter(id_to_vec.values())).shape}")
    print(f"dtype         : {next(iter(id_to_vec.values())).dtype}")
    print(f"est_memory    : {mem_bytes / (1024**2):.2f} MB")

    print("\nDone.\n")
