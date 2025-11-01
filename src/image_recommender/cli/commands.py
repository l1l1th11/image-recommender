from image_recommender.util.sampler import list_samples


def handle_list_samples(args) -> int:
    return int(list_samples(args))
