from pathlib import Path
from typing import Any

from image_recommender.db.pilot import create_pilot_set


def make_pilot(args: Any) -> int:
    """
    CLI adapter for pilot creation.
    Receives argparse Namespace with: db, n, seed, out
    """
    db_path = Path(args.db)
    out_path = Path(args.out)
    n = args.n
    seed = args.seed
    return create_pilot_set(db_path=db_path, n=n, seed=seed, out_path=out_path)
