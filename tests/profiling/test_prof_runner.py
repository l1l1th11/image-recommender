from types import SimpleNamespace

import pytest

from image_recommender.cli.commands import handle_profile_query
from image_recommender.config import SAMPLES_DIR


@pytest.fixture(scope="session")
def samples():
    return [
        p
        for p in SAMPLES_DIR.iterdir()
        if p.is_file() and p.suffix.lower() in {".jpg", ".jpeg", ".png"}
    ]


def _base_args(tmp_path, run_dir, mode, verbose=False):
    return SimpleNamespace(
        mode=mode,
        image_path=None,
        image_paths=None,
        run_dir=str(run_dir),
        output_dir=str(tmp_path / "profiling"),
        verbose=verbose,
    )


def _single_args(tmp_path, run_dir, img, verbose=False):
    args = _base_args(tmp_path, run_dir, "single", verbose)
    args.image_path = str(img)
    return args


def _multi_args(tmp_path, run_dir, imgs):
    args = _base_args(tmp_path, run_dir, "multi")
    args.image_paths = [str(p) for p in imgs]
    return args


def _assert_outputs(tmp_path):
    out = tmp_path / "profiling"
    assert (out / "profile.stats").exists()
    assert (out / "bottlenecks.png").exists()


def test_cli_single_smoke(tmp_path, samples):
    """
    Tests that the single image profiling CLI command runs without errors and produces outputs.
    """
    args = _single_args(tmp_path, SAMPLES_DIR, samples[0])
    handle_profile_query(args)
    _assert_outputs(tmp_path)


def test_cli_multi_smoke(tmp_path, samples):
    """
    Tests that the multi image profiling CLI command runs without errors and produces outputs.
    """
    args = _multi_args(tmp_path, SAMPLES_DIR, samples[:3])
    handle_profile_query(args)
    _assert_outputs(tmp_path)


def test_verbose_mode_no_crash(tmp_path, samples, capsys):
    """
    Tests that verbose mode prints insights without crashing.
    """
    args = _single_args(tmp_path, SAMPLES_DIR, samples[0], verbose=True)
    handle_profile_query(args)

    out = capsys.readouterr().out.lower()

    assert "profiling" in out or "bottleneck" in out or "top" in out


def test_invalid_inputs_raise():
    """
    Tests that invalid profiling modes raise ValueError.
    """
    args = SimpleNamespace(
        mode="invalid",
        image_path=None,
        image_paths=None,
        run_dir="dummy",
        output_dir="dummy",
        verbose=False,
    )

    with pytest.raises(ValueError):
        handle_profile_query(args)
