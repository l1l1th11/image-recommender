from image_recommender.db import connector
from image_recommender.ingest_metadata import scan_and_ingest_metadata


def test_ingest_dryrun(tmp_path):
    """
    Tests ingesting metadata from a directory containing:
    - A valid image file
    - An image file with the same path already in the DB
    """
    example_dir = tmp_path / "samples"
    example_dir.mkdir()

    img1 = example_dir / "img1.jpg"
    img1.write_bytes(b"\xff\xd8\xff\xe0" + b"not really .jpeg")  # dummy JPEG content

    db_path = tmp_path / "test.db"  # path to temporary DB

    connector.init_db(db_path=str(db_path))  # initialize DB

    connector.upsert_image(  # using upsert_image method from connector.py
        path=str(img1.resolve()),
        width=128,
        height=256,
        ext=".jpg",
        bytes_=len(b"\xff\xd8\xff\xe0" + b"not really .jpeg"),
        added_at="2020-01-01T00:00:00",
        db_path=str(db_path),
    )

    bad_file = example_dir / "bad.png"  # invalid image file
    bad_file.write_bytes(b"\x00\x00\x00INVALIDIMAGE")  # dummy invalid content

    scan_and_ingest_metadata(
        base_path=str(example_dir), db_path=str(db_path)  # what to scan  # where to store metadata
    )

    row = connector.get_by_path(str(img1.resolve()), db_path=str(db_path))

    assert row is not None  # Is the row existing? It should be.
    assert row["width"] == 128  # Was the width unchanged?

    bad_row = connector.get_by_path(str(bad_file.resolve()), db_path=str(db_path))
    assert bad_row is None  # Is the bad file ingested? It should not be.

    assert (
        connector.count(db_path=str(db_path)) == 1
    )  # Is there only one row in DB? There should be.
