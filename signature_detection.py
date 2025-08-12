import cv2 as cv
import numpy as np
from typing import Tuple

# Normalization size matches main.py so that percentages align
IMG_WIDTH = 1024
IMG_HEIGHT = 1448


def preprocess(img_path: str) -> np.ndarray:
    """Load an image and return a binarized version with dark ink as white.

    Parameters
    ----------
    img_path: str
        Path to the input page image.

    Returns
    -------
    np.ndarray
        Binarized image resized to the standard dimensions where the
        signature detection operates. Ink strokes are white (255) and
        background is black (0).
    """
    img = cv.imread(img_path)
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    gray = cv.resize(gray, (IMG_WIDTH, IMG_HEIGHT))
    gray = cv.GaussianBlur(gray, (3, 3), 0)
    # THRESH_BINARY_INV makes dark ink become white (255)
    bin_img = cv.adaptiveThreshold(
        gray, 255, cv.ADAPTIVE_THRESH_MEAN_C, cv.THRESH_BINARY_INV, 15, 10
    )
    return bin_img


def extract_region(
    image: np.ndarray, rect_pct: Tuple[float, float, float, float]
) -> np.ndarray:
    """Extract a rectangular region using percentage-based coordinates.

    Parameters
    ----------
    image: np.ndarray
        Preprocessed page image.
    rect_pct: Tuple[float, float, float, float]
        (x_percent, y_percent, width_percent, height_percent) where each value
        is between 0.0 and 1.0 representing the fraction of the document's
        width or height.

    Returns
    -------
    np.ndarray
        Sub-image defined by the percentage rectangle.
    """
    x_pct, y_pct, w_pct, h_pct = rect_pct
    h, w = image.shape[:2]
    x = int(x_pct * w)
    y = int(y_pct * h)
    w_px = int(w_pct * w)
    h_px = int(h_pct * h)
    return image[y : y + h_px, x : x + w_px]


def detect_signature(
    img_path: str,
    rect_pct: Tuple[float, float, float, float],
    ink_threshold: float = 0.01,
) -> Tuple[bool, float]:
    """Determine whether a signature is present in the specified region.

    Parameters
    ----------
    img_path: str
        Path to the page image to inspect.
    rect_pct: Tuple[float, float, float, float]
        Percentage-based rectangle describing where the signature should
        appear on the page.
    ink_threshold: float, default 0.01
        Minimum ratio of ink (non-background) pixels required to consider
        that a signature is present. The value should be between 0.0 and 1.0.

    Returns
    -------
    Tuple[bool, float]
        A tuple containing:
        * bool: True if a signature is detected.
        * float: The computed ink ratio for the region.
    """
    bin_img = preprocess(img_path)
    region = extract_region(bin_img, rect_pct)

    # Optionally clean small noise
    region = cv.medianBlur(region, 3)

    # Since ink is white (255) after preprocessing, count non-zero pixels
    ink_pixels = cv.countNonZero(region)
    total_pixels = region.shape[0] * region.shape[1]
    ink_ratio = ink_pixels / float(total_pixels) if total_pixels else 0.0

    return ink_ratio > ink_threshold, ink_ratio


if __name__ == "__main__":
    signature, ratio = detect_signature("./1344-sig-3-2023-2/")

    import argparse

    parser = argparse.ArgumentParser(description="Detect signature presence in a page")
    parser.add_argument("image", help="Path to the image file")
    parser.add_argument(
        "rect",
        nargs=4,
        type=float,
        metavar=("X", "Y", "W", "H"),
        help="Signature rectangle as percentages (0-1) of width/height",
    )
    parser.add_argument(
        "--threshold", "-t", type=float, default=0.01, help="Ink ratio threshold"
    )
    args = parser.parse_args()

    signature, ratio = detect_signature(args.image, tuple(args.rect), args.threshold)
    print(f"Signature detected: {signature} (ink ratio={ratio:.4f})")
