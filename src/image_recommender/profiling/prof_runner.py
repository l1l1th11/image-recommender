import pstats
from pathlib import Path

from image_recommender.profiling.prof import load_stats, run_profile
from image_recommender.profiling.prof_viz import (
    plot_top_bottlenecks,
    pstats_to_dataframe,
)


def run_query_profiling(func, *args, output_dir: Path, top_n: int = 10, **kwargs):
    """
    Profiles a function and writes stats and plot.
    """

    output_dir = Path(output_dir)

    stats_path = output_dir / "profile.stats"
    png_path = output_dir / "bottlenecks.png"

    run_profile(
        stats_path,
        func,
        *args,
        **kwargs,
    )

    stats = load_stats(stats_path, top_n=top_n)
    df = pstats_to_dataframe(stats)

    plot_top_bottlenecks(df, png_path)

    return df, stats, stats_path, png_path


def print_profile_insights(stats_path: Path, project_filter: str | None = "image_recommender"):
    """
    Prints insights from pstats to console.
    """

    stats = pstats.Stats(str(stats_path))
    stats.strip_dirs()
    stats.sort_stats("cumulative")

    print("\n=== TOP PROJECT HOTSPOTS ===")

    if project_filter:
        stats.print_stats(project_filter, 30)
    else:
        stats.print_stats(30)

    print("\n=== WHO CALLS torch.serialization ===")
    stats.print_callers("torch.serialization", 30)  # PyTorch serialization hotspot

    print("\n=== WHO CALLS persistent_load ===")
    stats.print_callers("persistent_load", 30)  # Model loading / deserialization hotspot
