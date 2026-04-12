from pathlib import Path

from image_recommender.recommender.single_image_query import single_image_query


def run_single_image_query_benchmark(query_path: Path, run_dir: Path) -> None:
    """
    Runs a benchmark for single image query.
    """
    single_image_query(
        query_path=query_path,
        run_dir=run_dir,
        k=5,
        feature_types=None,
        weights=None,
    )
