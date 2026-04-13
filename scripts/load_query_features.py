"""
Load aligned feature vectors for a fixed set of query ids.

Retrieves the corresponding feature vectors for each feature type from sharded storage and ensures consistent id alignment across all features.
Outputs numpy arrays used for benchmarking and analysis.
"""

from pathlib import Path

import numpy as np


def load_query_ids(path: Path) -> np.ndarray:
    if not path.exists():
        raise FileNotFoundError(f"Query id file not found: {path}")
    return np.load(path)


def load_query_features(
    run_dir: Path,
    feature_type: str,
    query_ids: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    feature_dir = run_dir / feature_type

    shard_dirs = sorted(
        p for p in feature_dir.iterdir() if p.is_dir() and p.name.startswith("shard_")
    )

    id_to_vec = {}

    for shard_dir in shard_dirs:
        ids = np.load(shard_dir / "ids.npy")
        features = np.load(shard_dir / "features.npy")

        for i, img_id in enumerate(ids):
            if int(img_id) in query_ids:
                id_to_vec[int(img_id)] = features[i]

    vectors = []
    found_ids = []

    for qid in query_ids:
        qid_int = int(qid)

        if qid_int not in id_to_vec:
            raise ValueError(f"Query id not found in {feature_type}: {qid_int}")

        vectors.append(id_to_vec[qid_int])
        found_ids.append(qid_int)

    return np.vstack(vectors), np.array(found_ids)


def main():
    run_dir = Path("data/features/db")
    query_ids_path = Path("data/eval/query_ids.npy")

    query_ids = load_query_ids(query_ids_path)

    # ---- load all feature types ----
    emb_vectors, ids_emb = load_query_features(run_dir, "embedding", query_ids)
    hsv_vectors, ids_hsv = load_query_features(run_dir, "hsv", query_ids)
    phash_vectors, ids_phash = load_query_features(run_dir, "phash", query_ids)

    # ---- sanity checks ----
    if not (np.array_equal(ids_emb, ids_hsv) and np.array_equal(ids_emb, ids_phash)):
        raise ValueError("ID mismatch across feature types")

    # ---- save ----
    out_dir = Path("data/eval")
    out_dir.mkdir(parents=True, exist_ok=True)

    np.save(out_dir / "query_vectors_embedding.npy", emb_vectors)
    np.save(out_dir / "query_vectors_hsv.npy", hsv_vectors)
    np.save(out_dir / "query_vectors_phash.npy", phash_vectors)

    print("Saved query feature vectors:")
    print(f"Embedding: {emb_vectors.shape}")
    print(f"HSV:       {hsv_vectors.shape}")
    print(f"pHash:     {phash_vectors.shape}")


if __name__ == "__main__":
    main()
