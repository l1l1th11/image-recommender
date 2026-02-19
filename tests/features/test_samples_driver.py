from image_recommender.config import SAMPLES_DIR
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

    assert len(paths) >= 2, "Need at least two valid sample images for the test."

    result = topk_on_samples(paths, k=1)  # get top-1 neighbor

    for p, neighbors in result.items():
        top_neighbor, dist = neighbors[0]

        assert top_neighbor == p  # Is itself the closest?
        assert dist < 1e-6  # Is the distance nearly zero?
