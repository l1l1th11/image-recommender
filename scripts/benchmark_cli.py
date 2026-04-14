import time

import numpy as np

from image_recommender.cli.commands import handle_query
from image_recommender.db.connector import get_path_by_id


class Args:
    def __init__(self, image_paths, run_dir):
        self.image_path = image_paths if len(image_paths) > 1 else image_paths[0]
        self.run_dir = run_dir
        self.backend = "annoy"
        self.k_candidates = None
        self.k = 5
        self.feature_types = None
        self.display = False


def run_once(args):
    t0 = time.perf_counter()
    handle_query(args)
    return time.perf_counter() - t0


def benchmark(args_builder, label, repeats=3):
    times = []

    for i in range(repeats):
        print(f"{label} run {i+1}/{repeats}")
        args = args_builder()
        t = run_once(args)
        times.append(t)

    times = np.array(times)

    return {
        "mean": float(np.mean(times)),
        "median": float(np.median(times)),
        "min": float(np.min(times)),
        "max": float(np.max(times)),
    }


def print_result(name, stats):
    print(f"\n=== {name} ===")
    print(f"mean   : {stats['mean']*1000:.2f} ms")
    print(f"median : {stats['median']*1000:.2f} ms")
    print(f"min    : {stats['min']*1000:.2f} ms")
    print(f"max    : {stats['max']*1000:.2f} ms")


def main():
    run_dir = "data/features/db"
    query_ids = np.load("data/eval/query_ids.npy")

    single_path = str(get_path_by_id(int(query_ids[0])))
    multi_paths = [str(get_path_by_id(int(q))) for q in query_ids[:3]]

    print("Running in-process CLI benchmarks...\n")

    single_stats = benchmark(lambda: Args([single_path], run_dir), "single-annoy")
    print_result("SINGLE (ANNOY)", single_stats)

    multi_stats = benchmark(lambda: Args(multi_paths, run_dir), "multi-annoy")
    print_result("MULTI (ANNOY)", multi_stats)


if __name__ == "__main__":
    main()
