import time
from pathlib import Path

import numpy as np

from image_recommender.db.connector import get_path_by_id
from image_recommender.recommender.load_persistent_mapping import (
    load_persistent_mapping,
)
from image_recommender.recommender.multi_image_query import multi_image_query
from image_recommender.recommender.single_image_query import single_image_query
from image_recommender.search.annoy import AnnoySearchBackend


def run_query(query_path, run_dir, annoy_backend, id_to_vec_maps):
    t0 = time.perf_counter()

    _ = single_image_query(
        query_path=query_path,
        run_dir=run_dir,
        k=5,
        backend="annoy",
        k_candidates=None,
        annoy_backend=annoy_backend,
        id_to_vec_maps=id_to_vec_maps,
    )

    return time.perf_counter() - t0


def main():
    run_dir = Path("data/features/db")
    query_ids = np.load("data/eval/query_ids.npy")[:3]

    # enforce Path objects
    query_paths = [Path(get_path_by_id(int(q))) for q in query_ids]

    print("Loading mappings once...")
    id_to_vec_maps = load_persistent_mapping(run_dir)

    print("Initializing Annoy once...")
    annoy_backend = AnnoySearchBackend(
        run_dir=run_dir,
        feature_type="embedding",
        k=10000,
    )

    print("\nWarming up...")

    _ = single_image_query(
        query_path=query_paths[0],
        run_dir=run_dir,
        k=5,
        backend="annoy",
        k_candidates=None,
        annoy_backend=annoy_backend,
        id_to_vec_maps=id_to_vec_maps,
    )

    _ = multi_image_query(
        query_paths=query_paths,
        run_dir=run_dir,
        k=5,
        backend="annoy",
        k_candidates=None,
        annoy_backend=annoy_backend,
        id_to_vec_maps=id_to_vec_maps,
    )

    print("\n=== SINGLE QUERY VALIDATION ===\n")

    times = []

    for i, path in enumerate(query_paths):
        print(f"Query {i+1}")

        t = run_query(path, run_dir, annoy_backend, id_to_vec_maps)

        print(f"Time: {t*1000:.2f} ms\n")
        times.append(t)

    print("\n=== MULTI QUERY VALIDATION ===\n")

    t0 = time.perf_counter()

    _ = multi_image_query(
        query_paths=query_paths,
        run_dir=run_dir,
        k=5,
        backend="annoy",
        k_candidates=None,
        annoy_backend=annoy_backend,
        id_to_vec_maps=id_to_vec_maps,
    )

    t1 = time.perf_counter()

    print(f"Multi-query time: {(t1 - t0)*1000:.2f} ms")

    print("\n=== SUMMARY ===")
    print(f"Query 1 : {times[0]*1000:.2f} ms")
    print(f"Query 2 : {times[1]*1000:.2f} ms")
    print(f"Query 3 : {times[2]*1000:.2f} ms")


if __name__ == "__main__":
    main()
