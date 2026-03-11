import cv2
import numpy as np

from image_recommender.config import PHASH_SIZE


def extract_phash(img_rgb: np.ndarray) -> np.ndarray:
    """
    Extracts a perceptual hash from an RGB image.
    A perceptual hash keeps low frequency structural information (large shapes and brightness patterns),
    while discarding high frequency details such as noise, texture, and small edges.

    Converts the image to grayscale, resizes to a fixed resolution, computes the DCT, and encodes the lowest-frequency
    coefficients as binary vector.

    Returns 1D vector of length 64 with dtype uint8 and values in {0,1}.
    """
    # validate input
    if img_rgb.ndim != 3 or img_rgb.shape[2] != 3:
        raise ValueError("Input must be RGB image with shape (H, W, 3)")

    # convert to grayscale
    img_gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)

    # resize to fixed square (32 x 32)
    img_resized = cv2.resize(img_gray, (PHASH_SIZE, PHASH_SIZE))

    # convert to float matrix
    img_float = img_resized.astype(np.float32)

    # perform discrete cosine transform
    img_coef = cv2.dct(img_float)

    # slice low frequencies (top left 8 x 8)
    low_freq = img_coef[:8, :8]

    # flatten to 1D vector
    low_freq_vector = low_freq.flatten()

    # compute median
    median = np.median(low_freq_vector)

    # compare to median to produce binary hash
    binary_vector = low_freq_vector > median

    # convert to uint8
    binary_vector = binary_vector.astype(np.uint8)

    return binary_vector
