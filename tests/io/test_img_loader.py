from pathlib import Path

import numpy as np
import pytest
from PIL import Image, UnidentifiedImageError

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
    happy_array = mod.load_rgb(happy_path)
    # should return rgb array of type uint8
    assert happy_array.shape == (15, 10, 3)  # shape: height, width, channels
    assert happy_array.dtype == np.uint8


def test_gray_to_rgb(samples_dir):
    gray_path = samples_dir / "gray.png"
    gray_img = Image.new(mode="L", size=(20, 10), color=42)  # grayscale
    gray_img.save(gray_path, format="PNG")
    rgb_array = mod.load_rgb(gray_path)
    # should now be rgb array of type uint8
    assert rgb_array.shape == (10, 20, 3)
    assert rgb_array.dtype == np.uint8


def test_non_img(samples_dir):
    non_img_path = samples_dir / "non_img.txt"
    non_img_path.write_text("test")
    # raise custom error (wraps PIL errors)
    with pytest.raises(ImageLoadError) as excinfo:
        mod.load_rgb(non_img_path)
    # path included
    msg = str(excinfo.value)
    assert str(non_img_path) in msg
    # reason included
    assert "unidentified/unsupported image" in msg
    # original cause included
    assert isinstance(excinfo.value.__cause__, UnidentifiedImageError)
