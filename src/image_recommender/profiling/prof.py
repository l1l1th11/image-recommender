import cProfile
import pstats
from pathlib import Path


def run_profile(output_path: Path, func, *args, **kwargs):
    """
    Profiles a function and writes stats to file.
    """
    profiler = cProfile.Profile()
    profiler.enable()

    result = func(*args, **kwargs)

    profiler.disable()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    profiler.dump_stats(str(output_path))

    return output_path, result


def load_stats(stats_path: Path, top_n: int = 10):
    """
    Loads pstats for further processing.
    """
    stats = pstats.Stats(str(stats_path))
    stats.sort_stats("cumulative")
    stats.print_stats(top_n)
    return stats
