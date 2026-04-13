from pathlib import Path

import numpy as np

from image_recommender.recommender.single_image_query import _compute_full_scores


def run_queries(query_ids, run_dir, backend):
    results = []

    for qid in query_ids:
        # resolve path via db
        from image_recommender.db.connector import get_path_by_id

        query_path = get_path_by_id(int(qid))

        _, _, _, timings = _compute_full_scores(
            query_path=query_path,
            run_dir=run_dir,
            backend=backend,
        )

        results.append(timings)

    return results


def aggregate(results):
    keys = results[0].keys()

    agg = {}
    for k in keys:
        agg[k] = np.mean([r[k] for r in results])

    return agg


def print_report(name, agg):
    print(f"\n=== {name} ===")

    total = agg["total"]

    for k, v in sorted(agg.items(), key=lambda x: -x[1]):
        pct = (v / total) * 100 if total > 0 else 0
        print(f"{k:25s}: {v*1000:8.2f} ms  ({pct:5.1f}%)")


def main():
    run_dir = Path("data/features/db")

    # reuse eval queries
    query_ids = np.load("data/eval/query_ids.npy")
    query_ids = query_ids[:5]

    # warm up to stabilize io & caches
    print("Running warm-up...")
    _ = run_queries(query_ids[:2], run_dir, backend="linear")
    _ = run_queries(query_ids[:2], run_dir, backend="annoy")

    # ---- linear ----
    lin_results = run_queries(query_ids, run_dir, backend="linear")
    lin_agg = aggregate(lin_results)
    print_report("LINEAR", lin_agg)

    # ---- annoy ----
    ann_results = run_queries(query_ids, run_dir, backend="annoy")
    ann_agg = aggregate(ann_results)
    print_report("ANNOY", ann_agg)


if __name__ == "__main__":
    main()
