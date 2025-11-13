from argparse import ArgumentParser
from collections.abc import Sequence

from image_recommender.util.logs import setup_basic_logging

from .commands import handle_list_samples


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
