import logging

from _pytest.logging import LogCaptureHandler  # capture handler (only for tests)

from image_recommender.util.logs import get_logger, setup_basic_logging


def test_env_level(monkeypatch, caplog):
    # set global log level via env (DEBUG)
    monkeypatch.setenv("IMREC_LOG_LEVEL", "DEBUG")

    # clear root handlers so setup can re-run, keep pytests capture handler
    root = logging.getLogger()  # root logger
    for h in list(root.handlers):
        if isinstance(h, LogCaptureHandler):  # keep caplog working
            continue
        root.removeHandler(h)  # remove non-pytest handlers

    # configure root with DEBUG and single line format
    setup_basic_logging()

    # capture DEBUG and above
    caplog.set_level(logging.DEBUG)

    # child logger (inherits root config)
    log = get_logger(__name__)

    # emit DEBUG line
    log.debug("test")

    # DEBUG log exists
    assert "DEBUG" in caplog.text


def test_idempotent_logs(caplog):
    # second setup should not add duplicate handlers
    setup_basic_logging()

    # capture ERROR and above
    caplog.set_level(logging.ERROR)

    # child logger (inherits root config)
    log = get_logger(__name__)

    # emit ERROR line
    log.error("test 2")

    # collect ERROR records
    records = [r for r in caplog.records if r.levelname == "ERROR"]

    # only one ERROR record
    assert len(records) == 1
