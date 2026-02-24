import random
from pathlib import Path

from image_recommender.config import PILOT_IDS_CSV
from image_recommender.db import connector


def create_pilot_set(db_path: Path, n: int, seed: int, out_path: Path) -> int:
    """
    Creates a deterministic pilot set of image_ids and writes them to CSV.

    Sampling rules:
      1. Select all image_ids from the images table.
      2. Order them ascending by image_id.
      3. Sample n unique IDs via random.Random(seed).sample(ids, n).
      4. Write one ID per line into the output CSV.

    Determinism: With the *same DB* and *same seed*, the resulting CSV file
    will contain the exact same IDs in the same order (same output bytes).
    """
    if not db_path.exists():  # check if .db exists
        print(f"ERROR: DB not found: {db_path}")
        return 1  # Error code for missing DB

    try:
        # Use connector to get DB connection (automatically commits/closes)
        with connector.get_conn(str(db_path)) as conn:
            cur = conn.execute(
                "SELECT image_id FROM images ORDER BY image_id ASC"
            )  # image_ids sorted ascending
            ids: list[int] = [row["image_id"] for row in cur]  # list of rows (with image_ids)
    except Exception as e:
        print(f"ERROR: Cannot read DB: {e}")
        return 1

    if len(ids) < n:  # If there are less than n ids in the db...
        print(f"ERROR: DB has only {len(ids)} images; cannot sample {n}")
        return 1  # ...the amounts are not corresponding, return error code.

    rng = random.Random(seed)
    selected: list[int] = rng.sample(ids, n)

    try:
        out_path.parent.mkdir(parents=True, exist_ok=True)  # /data/pilot/pilot_1k_ids.csv
    except Exception as e:
        print(f"ERROR: Cannot create output directory: {e}")
        return 1

    try:
        with open(out_path, "w", encoding="utf-8") as f:  # open .csv
            for image_id in selected:
                f.write(f"{image_id}\n")  # write image_ids to .csv

    except Exception as e:
        print(f"ERROR: Cannot write output file: {e}")
        return 1  # Error

    print(f"Wrote {len(selected)} IDs to {out_path}")
    return 0  # Success


def load_ids_pilot(pilot_path: str | Path = PILOT_IDS_CSV) -> list[int]:
    # read ids from csv
    with pilot_path.open("r", encoding="utf-8") as f:

        image_ids = []

        for line in f:
            # remove trailing spaces
            stripped_line = line.strip()
            # remove empty lines
            if not stripped_line:
                continue
            # convert to int
            image_id = int(stripped_line)

            # add ids to list
            image_ids.append(image_id)

    return image_ids
