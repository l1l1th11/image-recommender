from pathlib import Path

import pytest

from image_recommender.cli.commands import handle_make_pilot
from image_recommender.cli.main import main
from image_recommender.db import connector


@pytest.fixture
def small_db(tmp_path: Path) -> tuple[Path, list[str]]:
    """Creates a tiny in-memory DB with predictable image_ids."""
    db_path = tmp_path / "metadata.db"  # temporary DB
    connector.init_db(db_path=db_path)

    # Insert 5 images
    image_paths: list[str] = [f"/img{i}.jpg" for i in range(5)]
    for i, path in enumerate(image_paths):  # upserting dummy metadata
        connector.upsert_image(
            path=path,
            width=100 + i,
            height=200 + i,
            ext=".jpg",
            bytes_=1234 + i,
            added_at="2025-11-09T00:00:00",
            db_path=db_path,
        )
    return db_path, image_paths


# ---------- TEST: DETERMINISTIC PILOT ----------


def test_make_pilot_deterministic(tmp_path: Path, small_db: tuple[Path, list[str]]) -> None:
    """Tests that the same seed produces the same pilot set."""
    db_path, _ = small_db
    out1 = tmp_path / "pilot1.csv"
    out2 = tmp_path / "pilot2.csv"

    args = type("Args", (), {})()  # empty class instance

    # parameters for make_pilot:

    args.db = str(db_path)
    args.n = 3  # sample
    args.seed = 42  # seed
    args.out = str(out1)  # output file path

    handle_make_pilot(args)  # <-- make pilot giving parameters

    with open(out1) as f:
        ids1 = [line.strip() for line in f]

    args.out = str(out2)
    handle_make_pilot(args)
    with open(out2) as f:
        ids2 = [line.strip() for line in f]

    assert ids1 == ids2, "Same seed should produce identical pilot IDs"

    # Using different seed: 123 instead of 42

    args.seed = 123
    args.out = str(out2)
    handle_make_pilot(args)
    with open(out2) as f:
        ids3 = [line.strip() for line in f]

    assert ids1 != ids3, "Different seed should produce different pilot IDs"


# ---------- CLI SMOKE ----------


def test_make_pilot_cli(tmp_path: Path, small_db: tuple[Path, list[str]], monkeypatch) -> None:
    """Tests that the CLI produces the same pilot set."""
    db_path, _ = small_db
    out = tmp_path / "pilot_cli.csv"

    # simulating command line

    monkeypatch.setattr(
        "sys.argv",
        [  # monkeypatch overrides sys.argv
            "image-recommender",
            "make-pilot",
            "--db",
            str(db_path),
            "--seed",
            "42",
            "--n",
            "2",
            "--out",
            str(out),
        ],
    )

    rc = main()  # run cli
    assert rc == 0  # no error
    assert out.exists()  # output file exists

    with open(out) as f:
        lines = [line.strip() for line in f]
    assert len(lines) == 2  # two Ids
