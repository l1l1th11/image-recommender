import logging
import os

LEVEL_ENV = "IMREC_LOG_LEVEL"  # global level var for root logger
FORMAT = "%(levelname)s | %(name)s | %(message)s"  # format for log output


def setup_basic_logging() -> None:
    """Configure global logging once with level from IMREC_LOG_LEVEL (default INFO)"""
    root = logging.getLogger()

    # check if handlers are already configured
    if root.handlers:
        return  # avoid duplicates

    # read level from env (fallback to INFO)
    level_name = os.getenv(LEVEL_ENV, "INFO").upper()
    # get associated numeric value for logging (e.g., 20 for INFO)
    level = getattr(logging, level_name, logging.INFO)
    # configure with level and format
    logging.basicConfig(level=level, format=FORMAT)


def get_logger(name: str) -> logging.Logger:
    """Return named logger (assumes setup_basic_logging() already ran)"""
    return logging.getLogger(name)  # typically called with __name__
