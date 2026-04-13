"""
Visually compare results between linear and annoy backends.

Displays query images alongside top k results from both backends for selected queries and k_candidates settings.
Used for qualitative assessment of result relevance.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

from image_recommender.db.connector import get_path_by_id
from image_recommender.recommender.single_image_query import (
    _compute_full_scores_from_features,
)
from image_recommender.search.annoy import AnnoySearchBackend


def load_image(path: str):
    try:
        return Image.open(path).convert("RGB")
    except Exception:
        return None


def topk(ids: list[int], scores: np.ndarray, k: int) -> list[int]:
    idx = np.argsort(scores)[:k]
    return [ids[i] for i in idx]


def show_comparison(query_path, linear_paths, annoy_paths, k_candidates):
    fig, axes = plt.subplots(3, 5, figsize=(15, 9))
    fig.suptitle(f"k_candidates = {k_candidates}", fontsize=16)

    # ---- query ----
    axes[0, 0].imshow(load_image(query_path))
    axes[0, 0].set_title("Query")
    axes[0, 0].axis("off")

    for j in range(1, 5):
        axes[0, j].axis("off")

    # ---- linear ----
    for i, path in enumerate(linear_paths):
        img = load_image(path)
        if img:
            axes[1, i].imshow(img)
        axes[1, i].set_title(f"L{i+1}")
        axes[1, i].axis("off")

    # ---- annoy ----
    for i, path in enumerate(annoy_paths):
        img = load_image(path)
        if img:
            axes[2, i].imshow(img)
        axes[2, i].set_title(f"A{i+1}")
        axes[2, i].axis("off")

    plt.tight_layout()
    plt.show()


def main():
    run_dir = Path("data/features/db")
    k = 5
    k_candidates_list = [50, 200, 1000, 10000]

    emb = np.load("data/eval/query_vectors_embedding.npy")
    hsv = np.load("data/eval/query_vectors_hsv.npy")
    phash = np.load("data/eval/query_vectors_phash.npy")
    ids = np.load("data/eval/query_ids.npy")

    inspect_indices = [0, 1, 2]

    for k_candidates in k_candidates_list:

        print(f"\n===== k_candidates={k_candidates} =====")

        annoy = AnnoySearchBackend(
            run_dir=run_dir,
            feature_type="embedding",
            k=k_candidates,
        )

        for i in inspect_indices:

            queries_by_feature = {
                "embedding": emb[i],
                "hsv": hsv[i],
                "phash": phash[i],
            }

            # ---- linear ----
            scores_lin, ids_lin, _ = _compute_full_scores_from_features(
                queries_by_feature=queries_by_feature,
                run_dir=run_dir,
                backend="linear",
            )
            top_lin = topk(ids_lin, scores_lin, k)

            # ---- annoy ----
            scores_ann, ids_ann, _ = _compute_full_scores_from_features(
                queries_by_feature=queries_by_feature,
                run_dir=run_dir,
                backend="annoy",
                k_candidates=k_candidates,
                annoy_backend=annoy,
            )
            top_ann = topk(ids_ann, scores_ann, k)

            # ---- paths ----
            query_path = get_path_by_id(int(ids[i]))
            linear_paths = [get_path_by_id(cid) for cid in top_lin]
            annoy_paths = [get_path_by_id(cid) for cid in top_ann]

            show_comparison(query_path, linear_paths, annoy_paths, k_candidates)


if __name__ == "__main__":
    main()
