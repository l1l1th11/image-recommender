"""
Aggregate and summarize backend comparison results.

Reads csv output from benchmarking and computes per k statistics such as average overlap, minimum overlap, runtime, and relative speedup.
Used to for parameter selection and trade off analysis.
"""

import csv
from collections import defaultdict
from pathlib import Path


def main():
    path = Path("data/eval/backend_comparison.csv")

    data = defaultdict(list)

    with path.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)

        for row in reader:
            k = int(row["k_candidates"])

            data[k].append(
                {
                    "overlap": float(row["overlap_at_5"]),
                    "annoy_time": float(row["annoy_time_ms"]),
                    "linear_time": float(row["linear_time_ms"]),
                }
            )

    print("\n=== Aggregated Results ===\n")

    for k in sorted(data.keys()):
        rows = data[k]

        avg_overlap = sum(r["overlap"] for r in rows) / len(rows)
        min_overlap = min(r["overlap"] for r in rows)

        avg_annoy = sum(r["annoy_time"] for r in rows) / len(rows)
        avg_linear = sum(r["linear_time"] for r in rows) / len(rows)

        speedup = avg_linear / avg_annoy if avg_annoy > 0 else float("inf")

        print(f"k_candidates = {k}")
        print(f"  avg_overlap     = {avg_overlap:.2f}")
        print(f"  min_overlap     = {min_overlap:.2f}")
        print(f"  avg_annoy_time  = {avg_annoy:.2f} ms")
        print(f"  avg_linear_time = {avg_linear:.2f} ms")
        print(f"  speedup         = {speedup:.2f}x\n")


if __name__ == "__main__":
    main()
