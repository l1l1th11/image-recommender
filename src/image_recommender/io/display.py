from pathlib import Path

from PIL import Image, UnidentifiedImageError


def display_results(top_k_resolved: list[tuple[Path, float]]) -> None:
    """
    Prints (path, score) pairs and opens corresponding images one by one.

    Input:
        top_k_resolved: List of (image_path, score) pairs, ordered by score (best match first)

    Notes:
        Skips images that can not be loaded or displayed and prints an error message including file path and exception
    """
    for filepath, score in top_k_resolved:

        # print score & path
        print(f"{filepath} {score}")

        try:
            # open image
            with Image.open(filepath) as img:

                # display image
                img.show()

        except (FileNotFoundError, PermissionError, UnidentifiedImageError, OSError) as e:
            print(f"Failed to open/display image {filepath}: {e}")
