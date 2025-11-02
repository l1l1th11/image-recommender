from image_recommender.constants import IMAGE_EXTS, SAMPLES_DIR
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
