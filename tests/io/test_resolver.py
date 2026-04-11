from pathlib import Path

import pytest

from image_recommender.io.resolver import resolve_id_to_path


def test_happy_path():
    top_k = [(0, 0.1), (1, 0.2)]
    top_k_resolved = resolve_id_to_path(top_k=top_k)

    # check output matches input length
    assert len(top_k) == len(top_k_resolved)

    # check (path, score) match expected filepaths
    filepath_0 = Path("data/samples/image_0007.jpeg")
    filepath_1 = Path("data/samples/image_0022.png")

    assert top_k_resolved == [(filepath_0, 0.1), (filepath_1, 0.2)]

    # check order is preserved
    top_k_descending = [(1, 0.2), (0, 0.1)]
    top_k_resolved_descending = resolve_id_to_path(top_k=top_k_descending)

    assert top_k_resolved_descending == [(filepath_1, 0.2), (filepath_0, 0.1)]


def test_missing_id():
    top_k = [(21, 0.1), (1, 0.2)]

    # ensure missing id raises
    with pytest.raises(ValueError):
        resolve_id_to_path(top_k=top_k)
