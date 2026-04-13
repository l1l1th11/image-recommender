"""
Generate deterministic set of query ids for evaluation.

Loads all candidate ids from feature shards (embedding) and samples a fixed subset using a seeded random dumber generator.
Resulting ids are used as the basis for evaluation steps (feature loading, benchmarking, inspection).
"""

from pathlib import Path

import numpy as np


def load_all_ids(run_dir: Path, feature_type: str) -> np.ndarray:
    feature_dir = run_dir / feature_type

    if not feature_dir.exists():
        raise FileNotFoundError(f"Feature directory not found: {feature_dir}")

    shard_dirs = sorted(
        p for p in feature_dir.iterdir() if p.is_dir() and p.name.startswith("shard_")
    )

    if not shard_dirs:
        raise ValueError(f"No shard directories found in: {feature_dir}")

    all_ids = []

    for shard_dir in shard_dirs:
        ids_path = shard_dir / "ids.npy"
        if not ids_path.exists():
            raise FileNotFoundError(f"Missing ids file: {ids_path}")

        ids = np.load(ids_path)
        all_ids.append(ids)

    return np.concatenate(all_ids)


def main():
    run_dir = Path("data/features/db")
    feature_type = "embedding"
    n_queries = 10
    seed = 42

    ids = load_all_ids(run_dir, feature_type)

    rng = np.random.default_rng(seed)
    query_ids = rng.choice(ids, size=n_queries, replace=False)

    out_path = Path("data/eval/query_ids.npy")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(out_path, query_ids)

    print(f"Saved {len(query_ids)} query ids to {out_path}")
    print("Query IDs:", query_ids)


if __name__ == "__main__":
    main()
