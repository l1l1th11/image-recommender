import random
import sqlite3
from pathlib import Path


def make_pilot(args):
    """Creates a deterministic pilot set of image_ids."""

    db_path = Path(args.db)  # String --> Path
    out_path = Path(args.out)

    if not db_path.exists():  # check if .db exists
        print(f"ERROR: DB not found: {db_path}")
        return 1  # Error code for missing DB

    conn = sqlite3.connect(db_path)  # connect to the database
    cur = conn.cursor()
    cur.execute("SELECT image_id FROM images ORDER BY image_id ASC")  # image_ids sorted ascending
    ids = [row[0] for row in cur.fetchall()]  # list of rows (with image_ids)
    conn.close()  # close connection

    if len(ids) < args.n:  # If there are less than n ids in the db...
        print(f"ERROR: DB has only {len(ids)} images; cannot sample {args.n}")
        return 1  # ...the amounts are not corresponding, return error code.

    rng = random.Random(args.seed)
    selected = rng.sample(ids, args.n)

    out_path.parent.mkdir(parents=True, exist_ok=True)  # /data/pilot/pilot_1k_ids.csv

    try:
        with open(out_path, "w", encoding="utf-8") as f:  # open .csv
            for image_id in selected:
                f.write(f"{image_id}\n")  # write image_ids to .csv

    except Exception as e:
        print(f"ERROR: Cannot write output file: {e}")
        return 1  # Error

    print(f"Wrote {len(selected)} IDs to {out_path}")
    return 0  # Success
