import json
from pathlib import Path

from image_recommender.db.connector import get_path_by_id


def resolve_id_to_path(
    top_k: list[tuple[int, float]],
    run_dir: Path | str,
) -> list[tuple[Path, float]]:
    """
    Resolves (id, score) pairs to (path, score).

    Uses:
        - Samples mapping if run_dir == data/samples
        - DB resolver otherwise
    """
    run_dir = Path(run_dir)

    # samples mode
    if run_dir == Path("data/samples"):
        samples_dir = run_dir
        mapping_path = samples_dir / "id_to_filename.json"

        with open(mapping_path) as f:
            mapping = json.load(f)

        mapping = {int(k): v for k, v in mapping.items()}

        top_k_resolved = []

        for image_id, score in top_k:
            try:
                filename = mapping[image_id]
            except KeyError as e:
                raise ValueError(f"Id {image_id} not found in samples mapping") from e

            filepath = samples_dir / filename
            top_k_resolved.append((filepath, score))

        return top_k_resolved

    # full dataset mode
    top_k_resolved = []

    for image_id, score in top_k:
        try:
            path_str = get_path_by_id(image_id)
        except Exception as e:
            raise ValueError(f"Id {image_id} could not be resolved via DB") from e

        filepath = Path(path_str)

        if not filepath.exists():
            raise ValueError(f"Resolved path does not exist: {filepath}")

        top_k_resolved.append((filepath, score))

    return top_k_resolved
