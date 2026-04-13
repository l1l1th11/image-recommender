from pathlib import Path

from image_recommender.recommender.multi_image_query import multi_image_query
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


def run_multi_image_query_benchmark(query_paths: list[Path], run_dir: Path) -> None:
    """
    Runs a benchmark for multi image query.
    """

    multi_image_query(
        query_paths=query_paths,
        run_dir=run_dir,
        k=5,
        feature_types=None,
        weights=None,
    )
