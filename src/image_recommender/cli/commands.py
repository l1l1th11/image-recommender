import logging
from pathlib import Path

from image_recommender.constants import IMAGE_EXTS, SAMPLES_DIR
from image_recommender.db.pilot import create_pilot_set
from image_recommender.features.samples_driver import topk_on_samples
from image_recommender.util.sampler import list_samples


def handle_list_samples(args) -> int:
    # normalize extensions provided by cli, or provide default set
    extset = {e.lower().lstrip(".") for e in args.ext} if args.ext else IMAGE_EXTS

    # collect paths (deterministic order)
    paths = list_samples(root=SAMPLES_DIR, extset=extset, limit=args.limit)

    # print filenames (one per line)
    for p in paths:
        print(p.name)

    return 0


def handle_make_pilot(args) -> int:
    """
    Handles the "make-pilot" CLI command.
    """
    db_path = Path(args.db)
    out_path = Path(args.out)
    n = args.n
    seed = args.seed

    return create_pilot_set(  # uses create_pilot_set from db.pilot
        db_path=db_path,
        n=n,
        seed=seed,
        out_path=out_path,
    )


def handle_hsv_on_samples(args) -> int:
    """
    Handles the "hsv-on-samples" CLI command.
    """
    if getattr(args, "quiet", False):  # If --quiet flag is set...
        logging.getLogger("image_recommender").setLevel(
            logging.CRITICAL
        )  # ... suppress error information.

    result = topk_on_samples(k=args.k)  # all neighbors

    if args.id:  # If a specific image ID is provided...
        target_path = next(
            (p for p in result.keys() if p.stem == args.id), None
        )  # ... find its Path.
        if target_path is None:  # If path not found...
            print(f"ID {args.id} not found")  # ... print error message.
            return 1  # error

        print(f"Top-{args.k} neighbors for {args.id}:")  # show header
        for neighbor, dist in result[target_path]:
            print(f"{neighbor.name}\t{dist:.5f}")
    else:  # If no specific ID is provided...
        for p, neighbors in result.items():
            formatted = ", ".join(f"{n.name} ({dist:.5f})" for n, dist in neighbors)
            print(f"{p.name} --> {formatted}")

    return 0  # success
