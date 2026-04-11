from argparse import Namespace
from pathlib import Path

from image_recommender.cli.commands import handle_query


def test_happy_path(capsys, monkeypatch):
    # construct fake functions
    def fake_single_image_query(**kwargs):
        return [(1, 0.0)], {"hsv"}

    def fake_resolve_id_to_path(**kwargs):
        return [(Path("data/samples/image_0022.png"), 0.0)]

    # patch fake functions into handler
    monkeypatch.setattr(
        "image_recommender.cli.commands.single_image_query",
        fake_single_image_query,
    )
    monkeypatch.setattr(
        "image_recommender.cli.commands.resolve_id_to_path",
        fake_resolve_id_to_path,
    )

    args = Namespace(
        image_path="data/samples/image_0022.png",
        run_dir="data/samples",
        k=1,
        feature_types=["hsv"],
    )

    # run handler
    result = handle_query(args)

    # check cli ran without errors
    assert result == 0

    # check output matches expected result
    captured = capsys.readouterr()
    expected = f"{Path('data/samples/image_0022.png')} 0.0"

    assert expected in captured.out


def test_missing_requested_feature(capsys, monkeypatch):
    # construct fake functions
    def fake_single_image_query(**kwargs):
        return [(1, 0.0)], {"hsv"}

    # patch fake function into handler
    monkeypatch.setattr(
        "image_recommender.cli.commands.single_image_query",
        fake_single_image_query,
    )

    args = Namespace(
        image_path="data/samples/image_0022.png",
        run_dir="data/samples",
        k=1,
        feature_types=["hsv", "embedding"],
    )

    # run handler
    result = handle_query(args)

    # check cli returns error code
    assert result == 1

    # check error message matches expected one
    captured = capsys.readouterr()
    expected = "Requested features"

    assert expected in captured.out
