from pathlib import Path

import numpy as np
from PIL import Image, UnidentifiedImageError

from image_recommender.util.errors import ImageLoadError


def load_rgb(path: str | Path) -> np.ndarray:
    """
    Loads an image from a given path using PIL (lazy)
    """
    # convert to path object
    path = Path(path)

    try:
        # open with context manager
        with Image.open(path) as img:  # may raise FileNotFoundError, UnidentifiedImageError

            # convert to RGB
            img = img.convert("RGB")  # decoding may raise OSError (corrupt/truncated)

            # convert to array (shape: H, W, 3)
            img_array = np.asarray(img)  # decoding/materialize may raise OSError

    except FileNotFoundError as e:
        msg = f"{path} | file not found"
        raise ImageLoadError(msg) from e

    except UnidentifiedImageError as e:
        msg = f"{path} | unidentified/unsupported image"
        raise ImageLoadError(msg) from e

    except OSError as e:
        msg = f"{path} | corrupt or unreadable image"
        raise ImageLoadError(msg) from e

    except Exception as e:
        msg = f"{path} | unexpected error: {e.__class__.__name__}"
        raise ImageLoadError(msg) from e

    # validate final shape
    if img_array.ndim != 3 or img_array.shape[2] != 3:
        msg = f"{path} | invalid shape {tuple(img_array.shape)} (expected [H,W,3])"
        raise ImageLoadError(msg)

    return img_array
