import logging
from pathlib import Path

import numpy as np

import image_recommender.features.samples_driver_embedding as embedding_driver
import image_recommender.features.samples_driver_phash as phash_driver
from image_recommender.config import SAMPLES_DIR
from image_recommender.constants import IMAGE_EXTS
from image_recommender.db.pilot import create_pilot_set
from image_recommender.features.embedding import extract_embeddings_batch
from image_recommender.features.extraction_pipeline import run_extraction
from image_recommender.features.phash import extract_phashes
from image_recommender.features.samples_driver_hsv import topk_on_samples
from image_recommender.util.sampler import list_samples
from image_recommender.viz.explorer import run_embedding_explorer
from image_recommender.viz.map_embeddings import run_map_embeddings


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


def handle_extract_features(args) -> int:
    """
    Handles the "extract" CLI command.
    """
    feature_type = args.feature_type
    input_mode = args.input_mode
    run_dir = Path(args.run_dir)
    shard_start = args.shard_start
    shard_stop = args.shard_stop
    pilot_path = Path(args.pilot_path)
    db_path = Path(args.db_path)
    shard_size = args.shard_size
    policy = args.policy

    # call pipeline
    try:
        run_extraction(
            feature_type=feature_type,
            input_mode=input_mode,
            run_dir=run_dir,
            shard_start=shard_start,
            shard_stop=shard_stop,
            pilot_path=pilot_path,
            db_path=db_path,
            shard_size=shard_size,
            policy=policy,
        )

        return 0  # success

    # log exception
    except Exception:
        logging.error("Extraction failed", exc_info=True)

        return 1  # failure


def handle_embedding_on_samples(args) -> int:
    """
    Handles the "embedding-on-samples" CLI command.
    """
    samples_dir = Path("data/samples")
    ids, images = embedding_driver.load_sample_images(samples_dir)

    embeddings = extract_embeddings_batch(
        images,
        model_name=args.model,
        device=args.device,
        pretrained=bool(args.pretrained),
    )

    results = embedding_driver.compute_topk(ids, embeddings, k=args.k)

    for i, neighbors in enumerate(results):
        top_id, top_dist = neighbors[0]
        if top_id != ids[i] or not np.isclose(top_dist, 0.0, atol=1e-6):
            print(f"Self-match failed for {ids[i]}")
            return 1

    embedding_driver.print_results(ids, results)
    return 0


def handle_phash_on_samples(args) -> int:
    """
    Handles the "phash-on-samples" CLI command.
    """
    samples_dir = SAMPLES_DIR
    ids, images = phash_driver.load_sample_images(samples_dir=samples_dir)

    phashes = extract_phashes(imgs_rgb=images)

    results = phash_driver.compute_topk(ids=ids, phashes=phashes, k=args.k)

    for i, neighbors in enumerate(results):
        top_id, top_dist = neighbors[0]
        if top_id != ids[i]:
            print(f"Self match failed for {ids[i]}")
            return 1

    phash_driver.print_results(ids=ids, results=results)
    return 0


def handle_map_embeddings(args) -> int:
    """
    Handles the "map-embeddings" CLI command.
    """
    try:
        run_map_embeddings(
            run_dir=Path(args.run_dir),
            feature_type=args.feature_type,
            dims=args.dims,
            sample_size=args.sample_size,
        )
        return 0
    except Exception as e:
        logging.error(f"Embedding mapping failed: {e}", exc_info=True)
        return 1


def handle_explore_map(args) -> int:
    """
    Handles the "explore-map" CLI command.
    """
    try:
        run_embedding_explorer(
            coords_path=Path(args.coords),
            ids_path=Path(args.ids),
            db_path=Path(args.db_path),
            k=args.k,
        )
        return 0
    except Exception:
        logging.error("Explorer failed", exc_info=True)
        return 1
