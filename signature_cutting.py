import cv2 as cv
import numpy as np
from typing import Tuple
import os
from pathlib import Path

try:
    import tomllib  # Python 3.11+
except ModuleNotFoundError:
    import tomli as tomllib  # pip install tomli for older Python

IMG_WIDTH = 1024
IMG_HEIGHT = 1448


def load_rect_from_config(
    config_path: str, doc_type: str
) -> Tuple[float, float, float, float]:
    with open(config_path, "rb") as f:
        cfg = tomllib.load(f)
    rects = cfg.get("rects", {})
    if doc_type in rects:
        return tuple(rects[doc_type])
    elif "default" in rects:
        return tuple(rects["default"])
    else:
        raise KeyError(f"No rect found for type '{doc_type}' and no default in config")


def save_signature_region(
    img_path: str, rect_pct: Tuple[float, float, float, float], output_path: str
):
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

    signature_crop = img_resized[y : y + h_px, x : x + w_px]

    if not cv.imwrite(output_path, signature_crop):
        raise IOError(f"Failed to save image to {output_path}")

    print(f"[OK] {Path(img_path).name} -> {output_path}")


def process_folder_by_type_prefix(
    input_dir: str,
    output_dir: str,
    config_path: str,
    valid_exts=(".png", ".jpg", ".jpeg", ".tif", ".tiff"),
) -> None:
    """
    For each file in input_dir:
      - Take everything before the first underscore as the doc_type
      - Use doc_type as key in config.toml to load the rect
    """
    inp = Path(input_dir)
    for customer_dir in inp.iterdir():
        if not customer_dir.is_dir():
            continue

        out = Path(output_dir) / customer_dir.name
        out.mkdir(parents=True, exist_ok=True)

        files = [
            p for p in customer_dir.iterdir() if p.is_file() and p.suffix.lower() in valid_exts
        ]
        if not files:
            print(f"[WARN] No image files in {input_dir} matching {valid_exts}")
            return

        for p in files:
            stem = p.stem  # e.g., "1301-2022_customer1_page1"
            if "_" not in stem:
                print(f"[SKIP] {p.name}: no underscore to split type prefix")
                continue

            doc_type = stem.split("_", 1)[0]  # e.g., "1301-2022" or "1385"

            try:
                rect = load_rect_from_config(config_path, doc_type)
            except KeyError as e:
                print(f"[SKIP] {p.name}: {e}")
                continue

            out_name = f"{stem}_signature{p.suffix.lower()}"
            out_path = str(out / out_name)

            try:
                save_signature_region(str(p), rect, out_path)
            except Exception as e:
                print(f"[ERR]  {p.name}: {e}")


# -------- Example direct run --------
if __name__ == "__main__":
    config_path = "./config.toml"
    process_folder_by_type_prefix(
        input_dir="./output_assets",
        output_dir="./out/signature",
        config_path=config_path,
    )
