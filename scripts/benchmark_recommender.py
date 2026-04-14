import time
from pathlib import Path

import numpy as np

from image_recommender.constants import SUPPORTED_FEATURES
from image_recommender.db.connector import get_path_by_id
from image_recommender.recommender.build_benchmark_id_mapping import build_id_to_vec_map
from image_recommender.recommender.single_image_query import _compute_full_scores


def build_maps(run_dir: Path) -> dict[str, dict[int, np.ndarray]]:
    print("Building id -> vector maps (once)...")

    maps = {}
    for feature in SUPPORTED_FEATURES:
        id_to_vec, _ = build_id_to_vec_map(run_dir, feature)
        maps[feature] = id_to_vec

    print("Done.\n")
    return maps


def run_queries(
    query_ids,
    run_dir,
    backend,
    k_candidates=None,
    id_to_vec_maps=None,
):
    times = []

    for qid in query_ids:
        query_path = get_path_by_id(int(qid))

        t0 = time.perf_counter()

        _compute_full_scores(
            query_path=query_path,
            run_dir=run_dir,
            backend=backend,
            k_candidates=k_candidates,
            id_to_vec_maps=id_to_vec_maps,
        )

        t1 = time.perf_counter()
        times.append(t1 - t0)

    return np.array(times)


def benchmark(run_dir, query_ids, backend, k_candidates=None, id_to_vec_maps=None, repeats=3):
    all_runs = []

    for r in range(repeats):
        print(f"{backend} run {r+1}/{repeats}")
        times = run_queries(
            query_ids,
            run_dir,
            backend,
            k_candidates=k_candidates,
            id_to_vec_maps=id_to_vec_maps,
        )
        all_runs.append(times)

    all_runs = np.vstack(all_runs)
    per_query_median = np.median(all_runs, axis=0)

    return {
        "mean": float(np.mean(per_query_median)),
        "median": float(np.median(per_query_median)),
        "min": float(np.min(per_query_median)),
        "max": float(np.max(per_query_median)),
    }


def print_result(name, stats):
    print(f"\n=== {name} ===")
    print(f"mean   : {stats['mean']*1000:.2f} ms")
    print(f"median : {stats['median']*1000:.2f} ms")
    print(f"min    : {stats['min']*1000:.2f} ms")
    print(f"max    : {stats['max']*1000:.2f} ms")


def main():
    run_dir = Path("data/features/db")
    query_ids = np.load("data/eval/query_ids.npy")[:5]

    # build mapping once
    id_to_vec_maps = build_maps(run_dir)

    # warm-up
    print("Warm-up...")
    _ = run_queries(query_ids[:2], run_dir, backend="linear", id_to_vec_maps=id_to_vec_maps)
    _ = run_queries(
        query_ids[:2],
        run_dir,
        backend="annoy",
        k_candidates=10000,
        id_to_vec_maps=id_to_vec_maps,
    )

    # benchmark
    lin_stats = benchmark(
        run_dir,
        query_ids,
        backend="linear",
        id_to_vec_maps=id_to_vec_maps,
    )
    print_result("LINEAR", lin_stats)

    ann_stats = benchmark(
        run_dir,
        query_ids,
        backend="annoy",
        k_candidates=10000,
        id_to_vec_maps=id_to_vec_maps,
    )
    print_result("ANNOY", ann_stats)


if __name__ == "__main__":
    main()
