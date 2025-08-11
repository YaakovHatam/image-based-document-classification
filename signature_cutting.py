import cv2 as cv
import numpy as np
from typing import Tuple
import os

try:
    import tomllib  # Python 3.11+
except ModuleNotFoundError:
    import tomli as tomllib  # pip install tomli for older Python

IMG_WIDTH = 1024
IMG_HEIGHT = 1448


def load_rect_from_config(config_path: str, doc_type: str) -> Tuple[float, float, float, float]:
    with open(config_path, "rb") as f:
        cfg = tomllib.load(f)
    rects = cfg.get("rects", {})
    if doc_type in rects:
        return tuple(rects[doc_type])
    elif "default" in rects:
        return tuple(rects["default"])
    else:
        raise KeyError(
            f"No rect found for type '{doc_type}' and no default in config")


def save_signature_region(img_path: str, rect_pct: Tuple[float, float, float, float], output_path: str):
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

    if not cv.imwrite(output_path, signature_crop):
        raise IOError(f"Failed to save image to {output_path}")

    print(f"Signature region saved to {output_path}")


if __name__ == "__main__":
    config_path = "./config.toml"
    doc_type = "1385"  # the type you want
    rect = load_rect_from_config(config_path, doc_type)

    save_signature_region(
        "./signature_cutting_template/_page4.png",
        rect,
        "./out/signature/signature_only_1385.png"
    )

    doc_type = "1301"  # the type you want
    rect = load_rect_from_config(config_path, doc_type)

    save_signature_region(
        "./signature_cutting_template/_page1.png",
        rect,
        "./out/signature/signature_only_1301.png"
    )
