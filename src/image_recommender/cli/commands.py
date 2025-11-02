from image_recommender.constants import IMAGE_EXTS, SAMPLES_DIR
from image_recommender.util.sampler import list_samples


def handle_list_samples(args) -> int:
    # no op call
    _ = list_samples(root=SAMPLES_DIR, extset=IMAGE_EXTS, limit=None)
    return 0
