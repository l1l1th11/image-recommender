from pathlib import Path


def list_samples(
    root: Path,  # sample dir
    extset: set[str] | None = None,  # allowed image file extensions
    limit: int | None = None,  # num of items returned
) -> list[Path]:  # returns list of path objects
    """
    List top level files in root with stable sort and optional filtering
    """
    # iterate over files in root
    items = [p for p in root.iterdir() if p.is_file()]

    if extset is not None:
        # return normalized file extensions
        def suf(p: Path) -> str:
            return p.suffix[1:].lower() if p.suffix else ""

        # append if image file
        items = [p for p in items if suf(p) in extset]

    # sort by path (deterministic)
    items.sort(key=lambda p: p.name)

    if limit and limit > 0:
        # shorten list to limit
        items = items[:limit]

    return items
