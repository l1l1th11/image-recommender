import sqlite3

from image_recommender.config import DB_PATH, PILOT_IDS_CSV


def load_pilot_ids(path):
    """
    Loads pilot IDs from a CSV file. Expects one ID per line.

    Input: Path to CSV file

    Output: List of integers
    """
    with open(path) as f:
        return [int(line.strip()) for line in f if line.strip()]


def main():
    """
    Verifies that all pilot IDs from the CSV file exist in the database.
    """
    pilot_ids = load_pilot_ids(PILOT_IDS_CSV)
    print(f"Pilot size: {len(pilot_ids)}")

    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()

    missing = []
    for pid in pilot_ids:
        cur.execute("SELECT 1 FROM images WHERE image_id = ?", (pid,))
        if cur.fetchone() is None:
            missing.append(pid)

    conn.close()

    print("\n=== RESULT ===")
    print(f"Missing IDs: {len(missing)}")

    if missing:
        print("Sample missing IDs:")
        print(missing[:10])
    else:
        print("All pilot IDs are valid.")


if __name__ == "__main__":
    main()
