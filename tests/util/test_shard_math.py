from image_recommender.util.shard_math import compute_range


def test_shard_math_balanced_partition() -> None:
    # define inputs
    items = 10
    parts = 4
    # build list of ranges
    ranges = [compute_range(items=items, parts=parts, idx=i) for i in range(parts)]
    # split start and stop values
    starts = [start for start, _ in ranges]
    stops = [stop for _, stop in ranges]
    # check valid bounds
    for i in range(len(ranges)):
        assert 0 <= starts[i] <= stops[i] <= items
    # check contiguity
    for i in range(len(ranges) - 1):
        assert stops[i] == starts[i + 1]
    # check part sizes
    sizes = [(stop - start) for start, stop in ranges]
    assert max(sizes) - min(sizes) <= 1
    # check full coverage
    assert starts[0] == 0
    assert stops[-1] == items
    assert sum(sizes) == items
