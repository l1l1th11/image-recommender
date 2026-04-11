import json
from pathlib import Path


def resolve_id_to_path(top_k: list[tuple[int, float]]) -> list[tuple[Path, float]]:
    """
    Resolves query results from (id, score) pairs to (path, score) pairs using samples id to filename mapping.

    Input:
        top_k: List of (image_id, score) pairs returned by query pipeline

    Output:
        List of (image_path, score) pairs in same order as input

    Raises:
        ValueError: If an image_id is not found in the mapping
    """
    # build mapping path
    samples_dir = Path("data/samples")
    mapping_path = samples_dir / "id_to_filename.json"

    top_k_resolved = []

    # load mapping
    with open(mapping_path) as f:
        mapping = json.load(f)

    # normalize (keys -> int)
    mapping = {int(k): v for k, v in mapping.items()}

    for image_id, score in top_k:

        # map filename to id
        try:
            filename = mapping[image_id]

        except KeyError as e:
            raise ValueError(f"Id {image_id} not found in mapping") from e

        # append path prefix
        filepath = samples_dir / filename

        # build tuples
        top_k_resolved.append((filepath, score))

    return top_k_resolved
