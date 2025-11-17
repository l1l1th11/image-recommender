import logging
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

import image_recommender.io.img_loader as mod
from image_recommender.util.errors import ImageLoadError


@pytest.fixture
# create temporary directory
def samples_dir(tmp_path: Path) -> Path:
    return tmp_path


def test_happy_path(samples_dir):
    # create path
    happy_path = samples_dir / "rgb.jpeg"
    # create image
    happy_img = Image.new(mode="RGB", size=(10, 15), color=(10, 20, 30))  # size: width, height
    # save image to path
    happy_img.save(happy_path, format="JPEG")
    # load image
    happy_array = mod.read_rgb(happy_path)
    # should return rgb array of type uint8
    assert happy_array.shape == (15, 10, 3)  # shape: height, width, channels
    assert happy_array.dtype == np.uint8


def test_gray_to_rgb(samples_dir):
    gray_path = samples_dir / "gray.png"
    gray_img = Image.new(mode="L", size=(20, 10), color=42)  # grayscale
    gray_img.save(gray_path, format="PNG")
    rgb_array = mod.read_rgb(gray_path)
    # should now be rgb array of type uint8
    assert rgb_array.shape == (10, 20, 3)
    assert rgb_array.dtype == np.uint8


def test_non_img(samples_dir, caplog):
    caplog.set_level(logging.ERROR)
    non_img_path = samples_dir / "non_img.txt"
    non_img_path.write_text("test")
    # raise custom error (wraps PIL errors)
    with pytest.raises(ImageLoadError):
        mod.read_rgb(non_img_path)
    # retrieve logs
    records = [r for r in caplog.records if r.levelname == "ERROR"]
    # only one log captured
    assert len(records) == 1
    # module included
    rec = records[0]
    assert str(mod.__name__) == rec.name
    # path included
    msg = rec.getMessage()  # retrieve log
    assert str(non_img_path) in msg
    # reason included
    assert "unidentified/unsupported image" in msg


def test_file_not_found(samples_dir):
    missing_path = samples_dir / "missing.png"
    with pytest.raises(ImageLoadError) as e:
        mod.read_rgb(missing_path)
    assert str(missing_path) in str(e.value)


def test_dtype_conversion(samples_dir):
    # create float32 image and save
    float_img_path = samples_dir / "float_img.png"
    arr = (np.random.rand(5, 5, 3) * 255).astype(np.float32)
    img = Image.fromarray(arr.astype(np.uint8))  # PIL only saves uint8, but simulate dtype branch
    img.save(float_img_path, format="PNG")
    loaded = mod.read_rgb(float_img_path)
    assert loaded.dtype == np.uint8
    assert loaded.shape == (5, 5, 3)
