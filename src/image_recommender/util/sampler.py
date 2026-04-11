from pathlib import Path


def list_samples(
    root: Path,
    extset: set[str] | None = None,  # allowed image file extensions
    limit: int | None = None,  # num of items returned
) -> list[Path]:  # returns list of path objects
    """
    List top level files in root with stable sort and optional filtering
    """
    # collect files
    items = []
    for p in root.iterdir():
        if not p.is_file():
            continue

        # filter if extset is given
        if extset is not None:
            suffix = p.suffix.lower().lstrip(".")
            if suffix not in extset:
                continue

        items.append(p)

    # deterministic order
    items = sorted(items, key=lambda p: p.name)

    # apply limit
    if limit is not None:
        items = items[:limit]

    return items
