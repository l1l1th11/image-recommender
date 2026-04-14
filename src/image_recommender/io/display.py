from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, UnidentifiedImageError


def display_results(
    top_k_resolved: list[tuple[Path, float]],
    query_path: Path | None = None,
    query_paths: list[Path] | None = None,
) -> None:
    """
    Displays top k results in a grid with rank and score.

    Input:
        top_k_resolved: List of (image_path, score) pairs, ordered by score (best match first)
        query_paths: Optional list of query images to display alongside results

    Notes:
        Saves the composed preview image and opens it via PIL.
        This avoids reliance on an interactive matplotlib backend.
    """
    items = top_k_resolved

    # normalize input (backward compatibility)
    if query_paths is None:
        query_paths = [query_path] if query_path is not None else []

    # prepend query images if provided
    if query_paths:
        query_items = [(p, -1.0) for p in query_paths]
        items = query_items + items

    n = len(items)

    if n == 0:
        print("No results to display.")
        return

    if n <= 10:
        cols = n
        rows = 1
    else:
        cols = 5
        rows = (n + cols - 1) // cols

    fig, axes = plt.subplots(rows, cols, figsize=(3 * cols, 3 * rows))
    axes = np.array(axes).reshape(-1)

    num_queries = len(query_paths) if query_paths else 0

    for i, (filepath, score) in enumerate(items):
        ax = axes[i]

        try:
            with Image.open(filepath) as img:
                ax.imshow(img.convert("RGB"))

            if i < num_queries:
                ax.set_title(f"QUERY {i+1}", fontsize=10)
            else:
                rank = i - num_queries + 1
                ax.set_title(f"#{rank}\n{score:.3f}", fontsize=10)

        except (FileNotFoundError, PermissionError, UnidentifiedImageError, OSError):
            ax.text(0.5, 0.5, "Error", ha="center", va="center")

            if i < num_queries:
                ax.set_title(f"QUERY {i+1}", fontsize=10)
            else:
                rank = i - num_queries + 1
                ax.set_title(f"#{rank}\n{score:.3f}", fontsize=10)

        ax.axis("off")

    for j in range(n, len(axes)):
        axes[j].axis("off")

    fig.tight_layout()

    output_path = Path("results_preview.png")
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)

    print(f"Saved results to {output_path}")

    try:
        with Image.open(output_path) as img:
            img.show()
    except Exception as e:
        print(f"Failed to open result image: {e}")
