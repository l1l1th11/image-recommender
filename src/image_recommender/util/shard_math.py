def compute_range(items: int, parts: int, idx: int) -> tuple[int, int]:
    """
    Returns range of a part index (idx).
    Output is [start, stop) within [0, items). Empty ranges allowed when items < parts.
    Contiguous coverage, part lengths differ by at most 1.
    """
    # validate inputs
    if items < 0:
        raise ValueError("Items must be >= 0.")
    if parts <= 0:
        raise ValueError("Parts must be >= 1.")
    if idx < 0 or idx >= parts:
        raise ValueError("Index must be in [0, parts).")
    # base length of parts
    base = items // parts
    # number of parts larger by 1
    remainder = items % parts
    # calculate length of part based on index
    length = base + 1 if idx < remainder else base
    # calculate range
    start = idx * base + min(idx, remainder)
    stop = start + length

    return start, stop
