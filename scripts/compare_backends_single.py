"""
Run quantitative comparison between linear and annoy backends.

Executes single image queries across multiple k_candidates settings and records runtime, overlap at k, and self match rank per query.
Results are written to a csv for analysis.
"""

import csv
import time
from pathlib import Path

import numpy as np

from image_recommender.recommender.single_image_query import (
    _compute_full_scores_from_features,
)
from image_recommender.search.annoy import AnnoySearchBackend


def topk(ids: list[int], scores: np.ndarray, k: int) -> list[int]:
    idx = np.argsort(scores)[:k]
    return [ids[i] for i in idx]


def overlap_at_k(a: list[int], b: list[int]) -> float:
    return len(set(a) & set(b)) / len(a)


def rank_of_id(ids: list[int], scores: np.ndarray, target_id: int) -> int | None:
    ranked_idx = np.argsort(scores)
    ranked_ids = [ids[i] for i in ranked_idx]

    try:
        return ranked_ids.index(target_id) + 1
    except ValueError:
        return None


def main():
    run_dir = Path("data/features/db")
    out_path = Path("data/eval/backend_comparison.csv")

    k = 5
    k_candidates_list = [50, 200, 1000, 10000]

    emb = np.load("data/eval/query_vectors_embedding.npy")
    hsv = np.load("data/eval/query_vectors_hsv.npy")
    phash = np.load("data/eval/query_vectors_phash.npy")
    query_ids = np.load("data/eval/query_ids.npy")

    out_path.parent.mkdir(parents=True, exist_ok=True)

    with out_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "query_id",
                "k_candidates",
                "linear_time_ms",
                "annoy_time_ms",
                "overlap_at_5",
                "self_match_rank_linear",
                "self_match_rank_annoy",
            ],
        )
        writer.writeheader()

        for k_candidates in k_candidates_list:
            print(f"\n===== k_candidates={k_candidates} =====")

            annoy = AnnoySearchBackend(
                run_dir=run_dir,
                feature_type="embedding",
                k=k_candidates,
            )

            for i in range(len(query_ids)):
                query_id = int(query_ids[i])

                queries_by_feature = {
                    "embedding": emb[i],
                    "hsv": hsv[i],
                    "phash": phash[i],
                }

                t0 = time.perf_counter()
                scores_lin, ids_lin, _ = _compute_full_scores_from_features(
                    queries_by_feature=queries_by_feature,
                    run_dir=run_dir,
                    backend="linear",
                )
                t1 = time.perf_counter()

                t2 = time.perf_counter()
                scores_ann, ids_ann, _ = _compute_full_scores_from_features(
                    queries_by_feature=queries_by_feature,
                    run_dir=run_dir,
                    backend="annoy",
                    k_candidates=k_candidates,
                    annoy_backend=annoy,
                )
                t3 = time.perf_counter()

                top_lin = topk(ids_lin, scores_lin, k)
                top_ann = topk(ids_ann, scores_ann, k)

                overlap = overlap_at_k(top_lin, top_ann)
                self_rank_lin = rank_of_id(ids_lin, scores_lin, query_id)
                self_rank_ann = rank_of_id(ids_ann, scores_ann, query_id)

                row = {
                    "query_id": query_id,
                    "k_candidates": k_candidates,
                    "linear_time_ms": round((t1 - t0) * 1000, 2),
                    "annoy_time_ms": round((t3 - t2) * 1000, 2),
                    "overlap_at_5": round(overlap, 2),
                    "self_match_rank_linear": self_rank_lin,
                    "self_match_rank_annoy": self_rank_ann,
                }
                writer.writerow(row)

                print(
                    f"Query {query_id} | "
                    f"overlap@5={row['overlap_at_5']:.2f} | "
                    f"linear={row['linear_time_ms']:.2f} ms | "
                    f"annoy={row['annoy_time_ms']:.2f} ms | "
                    f"self_lin={self_rank_lin} | "
                    f"self_ann={self_rank_ann}"
                )

    print(f"\nSaved results to {out_path}")


if __name__ == "__main__":
    main()
