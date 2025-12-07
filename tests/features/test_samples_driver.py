from image_recommender.constants import SAMPLES_DIR
from image_recommender.features.samples_driver import topk_on_samples
from image_recommender.io.img_loader import load_rgb


def test_topk_smoke():
    paths = []
    for p in SAMPLES_DIR.iterdir():
        if p.suffix.lower() in (".jpg", ".jpeg", ".png"):
            try:
                load_rgb(p)
                paths.append(p)
            except Exception:
                continue
    paths = paths[:3]  # use only first 3 valid images for test

    result = topk_on_samples(paths, k=1)  # get top-1 neighbor

    for p, neighbors in result.items():
        assert p not in [n for n, _ in neighbors]  # Is the image its own neighbor?
