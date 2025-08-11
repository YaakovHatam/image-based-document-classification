import cv2 as cv
import numpy as np
from typing import Tuple
import os

IMG_WIDTH = 1024
IMG_HEIGHT = 1448


def preprocess(img_path: str) -> np.ndarray:
    img = cv.imread(img_path)
    if img is None:
        raise FileNotFoundError(f"Could not read image: {img_path}")
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    gray = cv.resize(gray, (IMG_WIDTH, IMG_HEIGHT))
    gray = cv.GaussianBlur(gray, (3, 3), 0)
    bin_img = cv.adaptiveThreshold(
        gray, 255, cv.ADAPTIVE_THRESH_MEAN_C, cv.THRESH_BINARY_INV, 15, 10
    )
    return bin_img


def extract_region(image: np.ndarray, rect_pct: Tuple[float, float, float, float]) -> np.ndarray:
    x_pct, y_pct, w_pct, h_pct = rect_pct
    h, w = image.shape[:2]
    x = int(x_pct * w)
    y = int(y_pct * h)
    w_px = int(w_pct * w)
    h_px = int(h_pct * h)
    return image[y:y + h_px, x:x + w_px]


def save_signature_region(img_path: str, rect_pct: Tuple[float, float, float, float], output_path: str):
    # Ensure the output directory exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    img = cv.imread(img_path)
    if img is None:
        raise FileNotFoundError(f"Could not read image: {img_path}")

    img_resized = cv.resize(img, (IMG_WIDTH, IMG_HEIGHT))
    h, w = img_resized.shape[:2]
    x = int(rect_pct[0] * w)
    y = int(rect_pct[1] * h)
    w_px = int(rect_pct[2] * w)
    h_px = int(rect_pct[3] * h)

    signature_crop = img_resized[y:y + h_px, x:x + w_px]

    success = cv.imwrite(output_path, signature_crop)
    if not success:
        raise IOError(f"Failed to save image to {output_path}")
    print(f"Signature region saved to {output_path}")


if __name__ == "__main__":
    # Example usage:
    # Suppose signature is in bottom right area
    # rect = (0.70, 0.80, 0.25, 0.10)  # x%, y%, width%, height%
    rect = (0.048, 0.71, 0.148, 0.082)  # x%, y%, width%, height%
    save_signature_region("./signature_cutting_template/_page4.png",
                          rect, "./out/signature/signature_only.png")
