import sys
from argparse import ArgumentParser
from collections.abc import Sequence

from image_recommender.config import DB_PATH, DR_SAMPLE_SIZE, PILOT_IDS_CSV
from image_recommender.constants import SUPPORTED_FEATURES
from image_recommender.db.pilot import create_pilot_set
from image_recommender.util.logs import setup_basic_logging

from .commands import (
    handle_embedding_on_samples,
    handle_explore_map,
    handle_extract_features,
    handle_hsv_on_samples,
    handle_list_samples,
    handle_make_pilot,
    handle_map_embeddings,
    handle_phash_on_samples,
    handle_profile_query,
    handle_query,
)


def build_parser() -> ArgumentParser:
    """
    Builds the CLI parser with subcommands and their respective arguments.
    Each subcommand is associated with a handler function that implements its logic.
    """
    # build root parser (top level cli)
    parser = ArgumentParser(prog="image-recommender", description="Image Recommender CLI")
    # build subparser (register commands)
    subparsers = parser.add_subparsers(dest="cmd", required=True)

    # list samples command
    cmd_ls = subparsers.add_parser(
        "list-samples",
        help="List files in data/samples/ (stable order)",  # shown in parent --help
        description="List files from data/samples/ (stable order)",  # shown in list-samples --help
    )
    # call function (via handler)
    cmd_ls.set_defaults(run=handle_list_samples)
    # filter by extension
    cmd_ls.add_argument(
        "--ext",
        nargs="+",
        metavar="EXT",
        help="Filter by extensions (space separated). Example: --ext jpg png",
    )
    # limit number of files
    cmd_ls.add_argument(
        "--limit", type=int, default=None, metavar="LIM", help="Max number of files to list"
    )

    # make pilot command
    cmd_pilot = subparsers.add_parser(
        "make-pilot",
        help="Creates a deterministic pilot set of image_ids",
        description=create_pilot_set.__doc__,
    )
    cmd_pilot.set_defaults(run=handle_make_pilot)
    cmd_pilot.add_argument(
        "--db", default=DB_PATH, help="Path to database (default: data/metadata.db)"
    )
    cmd_pilot.add_argument("--seed", type=int, required=True, help="Random seed")
    cmd_pilot.add_argument("--n", type=int, required=True, help="Number of image_ids to sample")
    cmd_pilot.add_argument(
        "--out",
        default=PILOT_IDS_CSV,
        help="Output CSV path (default: data/pilot/pilot_1k_ids.csv)",
    )

    # hsv on samples command
    cmd_hsv = subparsers.add_parser(
        "hsv-on-samples",
        help="Computes top-k HSV neighbors over samples",
    )
    cmd_hsv.set_defaults(run=handle_hsv_on_samples)
    cmd_hsv.add_argument("--k", type=int, default=3, help="Top-k neighbors")
    cmd_hsv.add_argument(
        "--id", type=str, default=None, help="Print top-k for specific image id (stem)"
    )
    cmd_hsv.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress error information, only print top-k neighbors and skipped images",
    )

    # embedding on samples command
    cmd_embed = subparsers.add_parser(
        "embedding-on-samples",
        help="Computes top-k embedding neighbors over samples",
    )
    cmd_embed.set_defaults(run=handle_embedding_on_samples)
    cmd_embed.add_argument("--k", type=int, default=5)
    cmd_embed.add_argument("--model", type=str, default="resnet18")
    cmd_embed.add_argument("--device", type=str, default="cpu")
    cmd_embed.add_argument("--pretrained", type=int, default=0)

    # phash on samples command
    cmd_phash = subparsers.add_parser(
        "phash-on-samples", help="Compute top-k perceptual hash neighbors over samples"
    )
    cmd_phash.set_defaults(run=handle_phash_on_samples)
    cmd_phash.add_argument("--k", type=int, default=5)

    # extraction pipeline command
    cmd_extract = subparsers.add_parser(
        "extract",
        help="Run feature extraction pipeline",
        description="Extract specified features for pilot or DB and persist as shards",
    )
    cmd_extract.set_defaults(run=handle_extract_features)
    cmd_extract.add_argument(
        "--feature-type",
        type=str,
        required=True,
        choices=SUPPORTED_FEATURES,
        help=f"Available types: {', '.join(SUPPORTED_FEATURES)}",
    )
    cmd_extract.add_argument(
        "--input-mode",
        type=str,
        required=True,
        choices=("samples", "pilot", "db"),
        help="Available modes: samples, pilot, db",
    )
    cmd_extract.add_argument(
        "--run-dir",
        type=str,
        required=True,
        help="Directory under which extracted features will be saved",
    )
    cmd_extract.add_argument(
        "--shard-start",
        type=int,
        help="Inclusive start for shard extraction range. Ex.: 3 would start from shard 0003",
    )
    cmd_extract.add_argument(
        "--shard-stop",
        type=int,
        help="Exclusive stop for shard extraction range. Ex.: 3 would stop at shard 0002",
    )
    cmd_extract.add_argument(
        "--pilot-path",
        type=str,
        default=PILOT_IDS_CSV,
        help="Path to CSV pilot (default: data/pilot/pilot_1k_ids.csv)",
    )
    cmd_extract.add_argument(
        "--db-path", type=str, default=DB_PATH, help="Path to database (default: data/metadata.db)"
    )
    cmd_extract.add_argument("--shard-size", type=int, default=None)
    cmd_extract.add_argument(
        "--policy",
        type=str,
        default="skip_and_log",
        choices=("skip_and_log", "raise"),
        help="Error policy which will be passed to iterator. Available modes: skip_and_log, raise",
    )

    # map embeddings command
    cmd_map = subparsers.add_parser(
        "map-embeddings",
        help="Run UMAP on embedding shards and export coordinates, metadata and preview plots",
    )
    cmd_map.set_defaults(run=handle_map_embeddings)
    cmd_map.add_argument(
        "--run-dir",
        required=True,
        help="Directory under which embedding shards are stored",
    )
    cmd_map.add_argument(
        "--feature-type", required=True, choices=SUPPORTED_FEATURES, help="Feature type to project"
    )
    cmd_map.add_argument(
        "--dims", type=int, default=2, choices=[2, 3], help="Number of UMAP dimensions (2 or 3)"
    )
    cmd_map.add_argument(
        "--sample-size",
        type=int,
        default=DR_SAMPLE_SIZE,
        help="Optional number of embeddings to sample for UMAP projection",
    )

    # explore map command
    cmd_explore = subparsers.add_parser(
        "explore-map",
        help="Launch interactive embedding explorer",
        description="Interactive visualization for exploring embedding layouts",
    )
    cmd_explore.set_defaults(run=handle_explore_map)
    cmd_explore.add_argument(
        "--coords",
        type=str,
        required=True,
        help="Path to coordinate npy file (N,2)",
    )
    cmd_explore.add_argument(
        "--ids",
        type=str,
        required=True,
        help="Path to numpy array containing image IDs",
    )
    cmd_explore.add_argument(
        "--embeddings",
        type=str,
        required=True,
        help="Path to embedding shards directory",
    )
    cmd_explore.add_argument(
        "--db-path",
        type=str,
        default=DB_PATH,
        help="Path to database",
    )
    cmd_explore.add_argument(
        "--k",
        type=int,
        default=5,
        help="Number of nearest neighbors",
    )

    # run query command
    cmd_query = subparsers.add_parser(
        "query",
        help=f"Recommend k most similar images for provided query image and selected feature types (auto: {SUPPORTED_FEATURES})",
        description="Run search, resolve outputs to k (path, score) pairs (better matches first)",
    )
    cmd_query.set_defaults(run=handle_query)
    cmd_query.add_argument(
        "--image-path",
        type=str,
        nargs="+",
        required=True,
        help="Path to one or more query images",
    )
    cmd_query.add_argument(
        "--backend",
        type=str,
        choices=["linear", "annoy"],
        default="linear",
        help="Search backend to use (default: linear)",
    )
    cmd_query.add_argument(
        "--k-candidates",
        type=int,
        default=None,
        help="Number of candidates for Annoy backend (required if backend=annoy)",
    )
    cmd_query.add_argument(
        "--run-dir",
        type=str,
        required=True,
        help="Directory containing feature folders",
    )
    cmd_query.add_argument(
        "--k",
        type=int,
        default=5,
        help="Number of nearest neighbors to be returned",
    )
    cmd_query.add_argument(
        "--feature-types",
        type=str,
        nargs="+",
        default=None,
        choices=SUPPORTED_FEATURES,
        help=f"Feature types to use (default: auto). Choices: {', '.join(SUPPORTED_FEATURES)}",
    )
    cmd_query.add_argument(
        "--display",
        action="store_true",
        default=False,
        help="Prints (path, score) pairs and opens corresponding images one by one.",
    )

    # profile command
    cmd_profile = subparsers.add_parser(
        "profile-query",
        help="Profile full single-image query pipeline",
        description="Runs single-image query with cProfile and outputs stats and bottleneck plot",
    )
    cmd_profile.set_defaults(run=handle_profile_query)
    cmd_profile.add_argument(
        "--mode",
        type=str,
        required=True,
        help="Profiling mode to run. Options: single | multi",
    )
    cmd_profile.add_argument(
        "--image-path",
        required=False,
        help="Single image path (required for single mode)",
    )
    cmd_profile.add_argument(
        "--image-paths",
        nargs="+",
        required=False,
        help="Multiple image paths (required for multi mode)",
    )
    cmd_profile.add_argument(
        "--run-dir",
        required=True,
        help="Directory containing feature folders",
    )
    cmd_profile.add_argument(
        "--output-dir",
        type=str,
        default="data/profiling",
        help="Directory where profiling outputs are saved",
    )
    cmd_profile.add_argument(
        "--verbose",
        action="store_true",
        help="Print detailed profiling insights",
    )

    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """
    Parse arguments and dispatch to selected subcommand
    - If argv is None: read args from sys.argv[1:]
    - If argv is a sequence: use provided args (for tests)
    """
    setup_basic_logging()  # configure logging once

    parser = build_parser()
    args = parser.parse_args(argv)
    # return exit code (0: success, non-zero: error)
    return int(args.run(args))


if __name__ == "__main__":  # execute only if run as a script (otherwise importable)
    sys.exit(
        main(sys.argv[1:])
    )  # call main() with command-line arguments and exit with its return code
