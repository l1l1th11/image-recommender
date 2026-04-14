from pathlib import Path

import numpy as np

from image_recommender.constants import SUPPORTED_FEATURES
from image_recommender.recommender.build_benchmark_id_mapping import build_id_to_vec_map
from image_recommender.recommender.single_image_query import _compute_full_scores


def build_maps(run_dir: Path) -> dict[str, dict[int, np.ndarray]]:
    print("Building id -> vector maps (once)...")

    maps: dict[str, dict[int, np.ndarray]] = {}
    for feature in SUPPORTED_FEATURES:
        id_to_vec, _ = build_id_to_vec_map(run_dir, feature)
        maps[feature] = id_to_vec

    print("Done.\n")
    return maps


def run_queries(
    query_ids: np.ndarray,
    run_dir: Path,
    backend: str,
    id_to_vec_maps: dict[str, dict[int, np.ndarray]] | None,
) -> list[dict[str, float]]:
    # run full pipeline (includes timing instrumentation)
    results = []

    from image_recommender.db.connector import get_path_by_id

    for qid in query_ids:
        query_path = get_path_by_id(int(qid))

        _, _, _, timings = _compute_full_scores(
            query_path=query_path,
            run_dir=run_dir,
            backend=backend,
            id_to_vec_maps=id_to_vec_maps,
        )

        results.append(timings)

    return results


def aggregate(results: list[dict[str, float]]) -> dict[str, float]:
    keys = results[0].keys()

    agg = {}
    for key in keys:
        agg[key] = float(np.mean([result[key] for result in results]))

    return agg


def print_report(name: str, agg: dict[str, float]) -> None:
    print(f"\n=== {name} ===")

    total = agg["total"]

    for key, value in sorted(agg.items(), key=lambda item: -item[1]):
        pct = (value / total) * 100 if total > 0 else 0
        print(f"{key:25s}: {value*1000:8.2f} ms  ({pct:5.1f}%)")


def main() -> None:
    run_dir = Path("data/features/db")
    query_ids = np.load("data/eval/query_ids.npy")[:5]

    # build once for annoy profiling
    id_to_vec_maps = build_maps(run_dir)

    # warm up
    print("Running warm-up...")
    _ = run_queries(query_ids[:2], run_dir, backend="linear", id_to_vec_maps=None)
    _ = run_queries(query_ids[:2], run_dir, backend="annoy", id_to_vec_maps=id_to_vec_maps)

    # linear
    lin_results = run_queries(query_ids, run_dir, backend="linear", id_to_vec_maps=None)
    lin_agg = aggregate(lin_results)
    print_report("LINEAR", lin_agg)

    # annoy
    ann_results = run_queries(
        query_ids,
        run_dir,
        backend="annoy",
        id_to_vec_maps=id_to_vec_maps,
    )
    ann_agg = aggregate(ann_results)
    print_report("ANNOY", ann_agg)


if __name__ == "__main__":
    main()
