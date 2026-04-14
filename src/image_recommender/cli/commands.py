import logging
import time
from pathlib import Path

import numpy as np

import image_recommender.features.samples_driver_embedding as embedding_driver
import image_recommender.features.samples_driver_phash as phash_driver
from image_recommender.config import DEFAULT_K_CANDIDATES, SAMPLES_DIR
from image_recommender.constants import IMAGE_EXTS
from image_recommender.db.pilot import create_pilot_set
from image_recommender.features.embedding import extract_embeddings_batch
from image_recommender.features.extraction_pipeline import run_extraction
from image_recommender.features.phash import extract_phashes
from image_recommender.features.samples_driver_hsv import topk_on_samples
from image_recommender.io.display import display_results
from image_recommender.io.resolver import resolve_id_to_path
from image_recommender.profiling.prof_benchmark import (
    run_multi_image_query_benchmark,
    run_single_image_query_benchmark,
)
from image_recommender.profiling.prof_runner import (
    print_profile_insights,
    run_query_profiling,
)
from image_recommender.recommender.load_persistent_mapping import (
    load_persistent_mapping,
)
from image_recommender.recommender.multi_image_query import multi_image_query
from image_recommender.recommender.single_image_query import single_image_query
from image_recommender.search.annoy import AnnoySearchBackend
from image_recommender.util.sampler import list_samples
from image_recommender.viz.explorer import run_embedding_explorer
from image_recommender.viz.map_embeddings import run_map_embeddings

# global cache for mappings
_GLOBAL_MAPPING_CACHE = None


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
            embeddings_path=Path(args.embeddings),
            db_path=Path(args.db_path),
            k=args.k,
        )
        return 0

    except FileNotFoundError as e:
        logging.error(f"File not found: {e}")
        return 1

    except ValueError as e:
        logging.error(f"Invalid data: {e}")
        return 1

    except Exception:
        logging.error("Explorer failed", exc_info=True)
        return 1


def handle_query(args):
    """
    Handles the "query" CLI command.
    """
    try:
        # normalize image_path to list
        if isinstance(args.image_path, str):
            query_paths = [Path(args.image_path)]
        else:
            query_paths = [Path(p) for p in args.image_path]

        backend = getattr(args, "backend", "linear")
        k_candidates = getattr(args, "k_candidates", None)

        annoy_backend = None
        id_to_vec_maps = None  # added

        if backend == "annoy":
            # apply default
            effective_k = k_candidates if k_candidates is not None else DEFAULT_K_CANDIDATES

            annoy_backend = AnnoySearchBackend(
                run_dir=Path(args.run_dir),
                feature_type="embedding",
                k=effective_k,
            )

            # load mapping once per process
            global _GLOBAL_MAPPING_CACHE
            if _GLOBAL_MAPPING_CACHE is None:
                print("Loading id to vector mappings (once)...")
                from image_recommender.recommender.load_persistent_mapping import (
                    load_persistent_mapping,
                )

                _GLOBAL_MAPPING_CACHE = load_persistent_mapping(Path(args.run_dir))

            id_to_vec_maps = _GLOBAL_MAPPING_CACHE

        # for one query
        if len(query_paths) == 1:
            top_k, used_features = single_image_query(
                query_path=query_paths[0],
                run_dir=Path(args.run_dir),
                k=args.k,
                feature_types=args.feature_types,
                backend=backend,
                k_candidates=k_candidates,
                annoy_backend=annoy_backend,
                id_to_vec_maps=id_to_vec_maps,
            )

        # for multiple queries
        else:
            top_k, used_features = multi_image_query(
                query_paths=query_paths,
                run_dir=Path(args.run_dir),
                k=args.k,
                feature_types=args.feature_types,
                backend=backend,
                k_candidates=k_candidates,
                annoy_backend=annoy_backend,
                id_to_vec_maps=id_to_vec_maps,
            )

        # ensure requested features were used for query
        if args.feature_types is not None:
            if used_features != set(args.feature_types):
                raise ValueError(
                    f"Requested features {args.feature_types} but used {used_features}"
                )

        # resolve (id, score) -> (filepath, score)
        top_k_resolved = resolve_id_to_path(top_k=top_k, run_dir=args.run_dir)

        if args.display:
            # call display function (backward compatible)
            try:
                display_results(
                    top_k_resolved=top_k_resolved,
                    query_paths=query_paths,
                )
            except TypeError:
                # fallback for older signature (tests / mocks)
                display_results(top_k_resolved)

        else:
            # print results
            for filepath, score in top_k_resolved:
                print(f"{filepath} {score}")

        return 0

    # output error message
    except ValueError as e:
        print(f"Query failed: {e}")
        return 1


def handle_query_loop(args):
    """
    Runs an interactive query loop for the recommender system.

    This command allows repeated execution of single and multi-image queries
    within the same process. For the annoy backend, an explicit initialization
    step loads the persistent index and id to vector mappings once, enabling
    faster subsequent queries.

    Interaction flow:
        - User enters "init" to initialize (required for annoy backend)
        - User enters one or multiple image paths (semicolon-separated)
        - System executes query and prints timing information
        - Optional result visualization is triggered if enabled
        - Loop continues until user enters "exit"

    Input:
        args: Parsed CLI arguments containing:
            - run_dir: Directory containing feature shards
            - backend: "linear" or "annoy"
            - k: Number of results to return
            - feature_types: Optional subset of features
            - k_candidates: Candidate subset size for annoy
            - display: Whether to visualize results

    Output:
        Returns 0 on normal exit, 1 on fatal error

    Notes:
        - Annoy backend requires explicit initialization before querying
        - Initialization is performed once and reused across queries
        - Multi-image queries reuse candidate subsets for performance
        - Timing output separates query, resolve, and display stages
    """
    try:
        run_dir = Path(args.run_dir)
        backend = getattr(args, "backend", "linear")
        k_candidates = getattr(args, "k_candidates", None)

        # initialization state for annoy backend
        id_to_vec_maps = None
        annoy_backend = None
        initialized = False

        print("Interactive query loop started.")
        print("Type 'init' to initialize system.")
        print("Type 'exit' to quit.\n")

        while True:
            user_input = input("Enter image path (or 'init' / 'exit'): ").strip()

            # exit command
            if user_input.lower() == "exit":
                print("Exiting.")
                return 0

            # initialization command
            if user_input.lower() == "init":
                if backend != "annoy":
                    print("Initialization only required for annoy backend.\n")
                    initialized = True
                    continue

                print("Initializing system...")

                t0 = time.perf_counter()

                # load persistent id -> vector mappings
                id_to_vec_maps = load_persistent_mapping(run_dir)

                # determine effective candidate size
                effective_k = k_candidates if k_candidates is not None else DEFAULT_K_CANDIDATES

                # load annoy index
                annoy_backend = AnnoySearchBackend(
                    run_dir=run_dir,
                    feature_type="embedding",
                    k=effective_k,
                )

                initialized = True

                print(
                    f"Initialization complete in {time.perf_counter() - t0:.3f}s. "
                    "You can now run queries.\n"
                )
                continue

            # ensure initialization for annoy backend
            if backend == "annoy" and not initialized:
                print("System not initialized. Type 'init' first.\n")
                continue

            # parse user input (support multiple paths via ';')
            paths = [p.strip() for p in user_input.split(";") if p.strip()]

            if not paths:
                print("No valid input provided.\n")
                continue

            query_paths = [Path(p) for p in paths]

            try:
                t_query_start = time.perf_counter()

                # dispatch to single or multi image query
                if len(query_paths) == 1:
                    top_k, used_features = single_image_query(
                        query_path=query_paths[0],
                        run_dir=run_dir,
                        k=args.k,
                        feature_types=args.feature_types,
                        backend=backend,
                        k_candidates=k_candidates,
                        annoy_backend=annoy_backend,
                        id_to_vec_maps=id_to_vec_maps,
                    )
                else:
                    top_k, used_features = multi_image_query(
                        query_paths=query_paths,
                        run_dir=run_dir,
                        k=args.k,
                        feature_types=args.feature_types,
                        backend=backend,
                        k_candidates=k_candidates,
                        annoy_backend=annoy_backend,
                        id_to_vec_maps=id_to_vec_maps,
                    )

                query_time = time.perf_counter() - t_query_start

                # resolve image ids to file paths
                t_resolve_start = time.perf_counter()
                top_k_resolved = resolve_id_to_path(top_k=top_k, run_dir=run_dir)
                resolve_time = time.perf_counter() - t_resolve_start

                print(
                    f"[timing] query={query_time:.3f}s | resolve={resolve_time:.3f}s | "
                    f"features={sorted(used_features)}"
                )

                # optional visualization
                if args.display:
                    t_display_start = time.perf_counter()

                    display_results(
                        top_k_resolved=top_k_resolved,
                        query_paths=query_paths,
                    )

                    display_time = time.perf_counter() - t_display_start
                    print(f"[timing] display={display_time:.3f}s")
                else:
                    for filepath, score in top_k_resolved:
                        print(f"{filepath} {score}")

                total_time = time.perf_counter() - t_query_start
                print(f"[timing] total={total_time:.3f}s\n")

            except ValueError as e:
                print(f"Query failed: {e}\n")

    except Exception as e:
        print(f"Fatal error: {e}")
        return 1


def handle_profile_query(args) -> int:
    """
    Handles the "profile-query" CLI command.
    """

    mode = args.mode
    output_dir = Path(args.output_dir)

    if mode == "single":
        if not args.image_path:
            raise ValueError("single mode requires --image-path")

        _, __, stats_path, png_path = run_query_profiling(
            func=run_single_image_query_benchmark,
            query_path=Path(args.image_path),
            run_dir=Path(args.run_dir),
            output_dir=output_dir,
        )

    elif mode == "multi":
        if not args.image_paths:
            raise ValueError("multi mode requires --image-paths")

        _, __, stats_path, png_path = run_query_profiling(
            func=run_multi_image_query_benchmark,
            query_paths=[Path(p) for p in args.image_paths],
            run_dir=Path(args.run_dir),
            output_dir=output_dir,
        )

    else:
        raise ValueError(f"Unknown profiling mode: {mode}")

    if args.verbose:
        print_profile_insights(stats_path)

    print(f"Saved profiling plot to: {png_path}")
    return 0
