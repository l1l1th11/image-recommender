import sys
from argparse import ArgumentParser
from collections.abc import Sequence

from image_recommender.db.pilot import create_pilot_set
from image_recommender.util.logs import setup_basic_logging

from .commands import handle_hsv_on_samples, handle_list_samples, handle_make_pilot


def build_parser() -> ArgumentParser:
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
    cmd_pilot.add_argument("--db", required=True, help="Path to metadata.db")
    cmd_pilot.add_argument("--seed", type=int, required=True, help="Random seed")
    cmd_pilot.add_argument("--n", type=int, required=True, help="Number of image_ids to sample")
    cmd_pilot.add_argument(
        "--out",
        default="data/pilot/pilot_1k_ids.csv",
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
