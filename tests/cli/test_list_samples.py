from pathlib import Path

import pytest

from image_recommender.util.sampler import list_samples


@pytest.fixture
# create three files in a temporary dir
def samples_dir(tmp_path: Path) -> Path:
    for name in ["b.jpg", "a.png", "c.gif"]:
        (tmp_path / name).touch()
    return tmp_path


def test_order(samples_dir):
    ordered_p = list_samples(root=samples_dir, extset=None, limit=None)
    assert [p.name for p in ordered_p] == ["a.png", "b.jpg", "c.gif"]


def test_filter(samples_dir):
    filtered_p = list_samples(root=samples_dir, extset={"png"}, limit=None)
    assert [p.name for p in filtered_p] == ["a.png"]


def test_limit(samples_dir):
    shortened_p = list_samples(root=samples_dir, extset=None, limit=2)
    assert [p.name for p in shortened_p] == ["a.png", "b.jpg"]
