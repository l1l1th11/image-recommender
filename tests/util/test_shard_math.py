import pytest

from image_recommender.util.shard_math import compute_range


def _assert_partition_invariants(items: int, parts: int) -> None:
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


def test_shard_math_happy() -> None:
    # ranges are contiguous, cover all, sizes differ by <= 1
    _assert_partition_invariants(items=1000, parts=4)


def test_shard_math_edge_cases() -> None:
    # items=0
    _assert_partition_invariants(items=0, parts=3)

    # parts=1
    _assert_partition_invariants(items=6, parts=1)

    # items < parts
    _assert_partition_invariants(items=2, parts=5)

    # items not divisible by parts
    _assert_partition_invariants(items=10, parts=4)


def test_shard_math_bad_inputs() -> None:
    # items < 0
    with pytest.raises(ValueError, match=r">= 0"):
        compute_range(items=-1, parts=3, idx=2)

    # parts = 0
    with pytest.raises(ValueError, match=r">= 1"):
        compute_range(items=10, parts=0, idx=2)

    # idx < 0
    with pytest.raises(ValueError, match=r"\[0, parts\)"):
        compute_range(items=10, parts=2, idx=-2)

    # idx = parts
    with pytest.raises(ValueError, match=r"\[0, parts\)"):
        compute_range(items=10, parts=2, idx=2)
