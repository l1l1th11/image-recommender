import cv2
import numpy as np


def hsv_features(img_rgb: np.ndarray) -> np.ndarray:
    """
    Extracts a histogram of the HSV color space of an image.

    Input: RGB image.
    Output: 1D vector of length 432, dtype float32, normalized to sum 1

    Hue: color without brightness (h_bins: 12, range 0-180)
    Saturation: intensity of the color (s_bins: 6, range 0-256)
    Value: brightness (v_bins: 6, range 0-256)

    Unlike RGB, where lighting changes strongly affect the values,
    HSV provides a more stable representation of the actual color.
    """
    if img_rgb.ndim != 3 or img_rgb.shape[2] != 3:  # If not RGB...
        raise ValueError("Input must be RGB image with shape (H, W, 3)")  # ...raise an error.

    h_bins = 12
    s_bins = 6
    v_bins = 6

    hsv = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2HSV)

    hist = cv2.calcHist([hsv], [0, 1, 2], None, [h_bins, s_bins, v_bins], [0, 180, 0, 256, 0, 256])

    hist = hist.flatten()
    hist_sum = hist.sum()
    if hist_sum > 0:
        hist = hist / hist_sum  # normalize to sum = 1
    return hist.astype(np.float32)
